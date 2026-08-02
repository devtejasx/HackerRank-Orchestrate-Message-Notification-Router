"""The routing rules, as small independent functions.

Each rule inspects the :class:`~src.routing.models.DecisionContext` and argues
for an action by returning :class:`~src.routing.models.RuleOutcome` objects.
The engine sums their weights and takes the strongest, so this is a set of
independent arguments rather than one long if/elif chain: rules can be added,
reweighted or removed without disturbing each other, and every one carries the
sentence used if it turns out to be decisive.

Two layers:

* **Type priors** give every message a starting position from its Phase 2
  category. The category is by far the strongest single predictor of the
  action, and the priors below mirror the distribution actually observed in
  the labelled examples rather than being invented.
* **Adjustment rules** then move that starting position using the Phase 3
  signals, which is where personalisation enters: the same promotion can end
  up ``digest`` for one user and ``mute`` for another.

A small number of rules may **override**. Those are reserved for safety, where
no amount of enthusiasm elsewhere should be able to let a message through.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from src.classifier.enums import KeywordCategory, MessageType
from src.routing.models import DecisionContext, RoutingAction, RuleOutcome

__all__ = ["Thresholds", "DEFAULT_THRESHOLDS", "TYPE_PRIORS", "DEFAULT_RULES"]

N = RoutingAction.NOTIFY
D = RoutingAction.DIGEST
M = RoutingAction.MUTE


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Every tunable magnitude used by the rules.

    Signal scores run ``0..1`` with ``0.5`` neutral, so the thresholds below
    are read against that scale. Grouped by the rule that consumes them.
    """

    # -- what counts as a strong or weak signal ------------------------- #
    strong_signal: float = 0.65
    weak_signal: float = 0.35
    high_risk: float = 0.70
    high_fatigue: float = 0.55
    strong_trust: float = 0.70

    # -- safety --------------------------------------------------------- #
    scam_override_confidence: float = 0.55
    risk_suppression: float = 1.60
    stranger_money_weight: float = 2.40

    # -- counterparty standing ------------------------------------------ #
    trusted_urgency: float = 1.60
    verified_transaction: float = 1.40
    strong_counterparty: float = 1.10
    weak_counterparty: float = 0.90

    # -- promotions ------------------------------------------------------ #
    opted_out_promotion: float = 1.80
    dismissed_promotion: float = 1.20
    trusted_promotion: float = 1.00

    # -- conversation state ---------------------------------------------- #
    muted_group: float = 1.50
    heavy_forwarding: float = 1.40
    direct_mention: float = 1.20

    # -- attention economics ---------------------------------------------- #
    fatigue_dampener: float = 0.90
    quiet_hours_dampener: float = 0.70
    disengaged_user: float = 0.80
    historical_importance: float = 1.00

    # -- forwarding ------------------------------------------------------- #
    heavy_forward_count: int = 8
    moderate_forward_count: int = 4


DEFAULT_THRESHOLDS: Final[Thresholds] = Thresholds()


#: Starting weights per action for each message category.
#:
#: Derived from the labelled examples in ``sample_messages.csv``: every scam
#: there is muted, every urgent message notified, promotions split evenly
#: between digest and mute, and personal messages lean digest. Categories with
#: no labelled example (``payment``) are positioned from the challenge brief,
#: which describes a payment reminder as legitimate from a trusted admin and
#: risky from a new sender - so it starts near the notify/digest boundary and
#: lets the personalisation rules decide.
TYPE_PRIORS: Final[Mapping[MessageType, Mapping[RoutingAction, float]]] = {
    MessageType.SCAM: {M: 3.00},
    MessageType.SPAM: {M: 2.20, D: 0.40},
    MessageType.FORWARD: {M: 1.60, D: 0.90},
    MessageType.URGENT: {N: 2.40, D: 0.50},
    MessageType.EVENT: {N: 1.50, D: 1.10},
    MessageType.PAYMENT: {N: 1.40, D: 1.10},
    MessageType.BUSINESS_UPDATE: {N: 0.70, D: 1.70},
    MessageType.PROMOTION: {D: 1.30, M: 1.10},
    MessageType.GREETING: {D: 1.20, M: 0.90},
    MessageType.PERSONAL: {N: 0.90, D: 1.50},
    MessageType.UNKNOWN: {D: 1.40, M: 0.50},
}


def type_prior(context: DecisionContext, _: Thresholds) -> Iterable[RuleOutcome]:
    """Open with the position implied by the message's category."""
    priors = TYPE_PRIORS.get(context.classification.message_type, {D: 1.0})
    label = context.message_type.replace("_", " ")
    for action, weight in priors.items():
        yield RuleOutcome(
            rule="type_prior",
            action=action,
            weight=weight,
            reason=f"The message is classified as {label}.",
        )


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #


def scam_override(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Mute a confirmed scam outright.

    The one place an override is unambiguously right: a scam that reaches the
    user can cost them money or their account, and no amount of engagement
    history should be able to buy it an interruption.
    """
    if context.classification.message_type is not MessageType.SCAM:
        return
    if context.classification.confidence < thresholds.scam_override_confidence:
        return
    yield RuleOutcome(
        rule="scam_override",
        action=M,
        weight=context.classification.confidence * 3.0,
        reason="The message shows clear scam characteristics and is unsafe to deliver.",
        override=True,
    )


def risk_suppression(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Suppress anything the risk signal rates highly, scam or not."""
    risk = context.signals.risk_modifier
    if risk.score < thresholds.high_risk:
        return
    yield RuleOutcome(
        rule="risk_suppression",
        action=M,
        weight=thresholds.risk_suppression * risk.score * risk.confidence,
        reason="Multiple risk signals point to unwanted or unsafe content.",
    )


def stranger_requesting_money(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Mute money requests from senders the user has never dealt with.

    The combination is what matters. A payment request from a known business
    is routine; the same words from a stranger with no history are the classic
    advance-fee setup, and the brief calls this out explicitly.
    """
    features = context.features
    wants_money = (
        features.keywords.has(KeywordCategory.PAYMENT)
        or features.text.contains_payment_symbol
        or features.text.contains_currency
    )
    if not wants_money:
        return

    history = context.features.history
    is_stranger = (
        history.sender_message_count == 0 and not history.has_business_relationship
    )
    if not is_stranger:
        return

    yield RuleOutcome(
        rule="stranger_requesting_money",
        action=M,
        weight=thresholds.stranger_money_weight,
        reason="An unfamiliar sender is asking for money or payment details.",
    )


# --------------------------------------------------------------------------- #
# Counterparty standing
# --------------------------------------------------------------------------- #


def urgent_trusted_sender(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Let genuinely urgent messages through when the sender has standing."""
    if context.signals.urgency_modifier.score <= thresholds.strong_signal:
        return
    trust = context.signals.trust_modifier
    sender = context.signals.sender_priority
    group = context.signals.group_priority
    standing = max(trust.score, sender.score, group.score)
    if standing < thresholds.strong_signal:
        return
    yield RuleOutcome(
        rule="urgent_trusted_sender",
        action=N,
        weight=thresholds.trusted_urgency * standing,
        reason="A trusted sender raised something time-sensitive.",
    )


def verified_business_transaction(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Notify on transactional messages from a business the user deals with.

    An order update or payment confirmation from a verified brand the user
    actually buys from is something they are waiting for.
    """
    if context.classification.message_type not in _TRANSACTIONAL_TYPES:
        return
    if not context.features.context.is_trusted_business:
        return
    if not context.features.history.has_business_relationship:
        return
    yield RuleOutcome(
        rule="verified_business_transaction",
        action=N,
        weight=thresholds.verified_transaction,
        reason="A verified business the user deals with sent a transactional update.",
    )


def counterparty_standing(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Move on how much this particular counterparty matters to this user.

    The single clearest expression of personalisation: an identical message
    from a close contact and from someone the user ignores should not land the
    same way.
    """
    signals = context.signals
    applicable = [
        signal
        for signal in (
            signals.sender_priority,
            signals.business_priority,
            signals.group_priority,
        )
        if signal.confidence > 0.0
    ]
    if not applicable:
        return

    best = max(applicable, key=lambda signal: signal.score)
    if best.score >= thresholds.strong_signal:
        yield RuleOutcome(
            rule="counterparty_standing",
            action=N,
            weight=thresholds.strong_counterparty * best.score * best.confidence,
            reason="This sender consistently matters to the user.",
        )
    elif best.score <= thresholds.weak_signal:
        yield RuleOutcome(
            rule="counterparty_standing",
            action=D,
            weight=thresholds.weak_counterparty * (1.0 - best.score) * best.confidence,
            reason="The user rarely engages with this sender.",
        )


def historical_importance(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Move on how the user treated comparable messages before."""
    signal = context.signals.historical_importance
    if signal.confidence == 0.0:
        return
    if signal.score >= thresholds.strong_signal:
        yield RuleOutcome(
            rule="historical_importance",
            action=N,
            weight=thresholds.historical_importance * signal.score * signal.confidence,
            reason="Comparable messages from this source have mattered before.",
        )
    elif signal.score <= thresholds.weak_signal:
        yield RuleOutcome(
            rule="historical_importance",
            action=D,
            weight=thresholds.historical_importance
            * (1.0 - signal.score)
            * signal.confidence,
            reason="Comparable messages have previously been ignored or dismissed.",
        )


def direct_mention(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """A message that names the recipient is addressed to them, not the room."""
    text = context.features.text
    if text.is_empty:
        return
    if f"@{context.user_id}".casefold() not in text.normalized_text:
        return
    yield RuleOutcome(
        rule="direct_mention",
        action=N,
        weight=thresholds.direct_mention,
        reason="The message addresses the user directly.",
    )


# --------------------------------------------------------------------------- #
# Promotions
# --------------------------------------------------------------------------- #


def promotion_opted_out(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Mute marketing the user explicitly opted out of.

    An opt-out is an instruction, not a preference to be weighed.
    """
    if context.classification.message_type is not MessageType.PROMOTION:
        return
    if not context.features.history.opted_out_of_promotions:
        return
    yield RuleOutcome(
        rule="promotion_opted_out",
        action=M,
        weight=thresholds.opted_out_promotion,
        reason="The user opted out of promotions from this business.",
    )


def promotion_previously_dismissed(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Mute marketing the user has shown they do not read."""
    if context.classification.message_type is not MessageType.PROMOTION:
        return
    importance = context.signals.historical_importance
    if importance.confidence == 0.0 or importance.score > thresholds.weak_signal:
        return
    yield RuleOutcome(
        rule="promotion_previously_dismissed",
        action=M,
        weight=thresholds.dismissed_promotion * importance.confidence,
        reason="Previous promotions from this source were dismissed.",
    )


def promotion_from_trusted_business(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Keep marketing from a trusted, engaged-with brand in the digest.

    Worth showing later, never worth interrupting for.
    """
    if context.classification.message_type is not MessageType.PROMOTION:
        return
    if context.features.history.opted_out_of_promotions:
        return
    if context.signals.trust_modifier.score < thresholds.strong_trust:
        return
    yield RuleOutcome(
        rule="promotion_from_trusted_business",
        action=D,
        weight=thresholds.trusted_promotion,
        reason="Promotional content from a business the user engages with.",
    )


# --------------------------------------------------------------------------- #
# Conversation state
# --------------------------------------------------------------------------- #


def muted_group(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Respect a muted group, but not at the cost of a genuine emergency.

    Muting means "stop interrupting me about this", not "hide this from me".
    A directly-addressed or genuinely urgent message still gets through, which
    is the behaviour the brief describes.
    """
    membership = (
        context.repo.get_group_member(context.features.group_id, context.user_id)
        if context.features.group_id
        else None
    )
    if membership is None or not membership.group_muted_by_user:
        return

    urgent = context.signals.urgency_modifier.score >= thresholds.strong_signal
    if urgent:
        yield RuleOutcome(
            rule="muted_group",
            action=N,
            weight=thresholds.muted_group * 0.5,
            reason="The group is muted, but this message looks genuinely urgent.",
        )
        return

    yield RuleOutcome(
        rule="muted_group",
        action=D,
        weight=thresholds.muted_group,
        reason="The user has muted this group.",
    )


def heavy_forwarding(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Lower the priority of chain messages.

    A high forwarding count means many people have passed this along
    unchanged, which is the signature of a chain message rather than of
    something written for this recipient.
    """
    count = context.features.forwarded_count
    if count >= thresholds.heavy_forward_count:
        yield RuleOutcome(
            rule="heavy_forwarding",
            action=M,
            weight=thresholds.heavy_forwarding,
            reason=f"The message has been forwarded {count} times.",
        )
    elif count >= thresholds.moderate_forward_count:
        yield RuleOutcome(
            rule="heavy_forwarding",
            action=D,
            weight=thresholds.heavy_forwarding * 0.6,
            reason=f"The message has been forwarded {count} times.",
        )


# --------------------------------------------------------------------------- #
# Attention economics
# --------------------------------------------------------------------------- #


def notification_fatigue(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Raise the bar for interrupting an already-overloaded user.

    Argues for digest rather than against notify: the message is still worth
    showing, just not worth breaking into the user's day for.
    """
    fatigue = context.signals.fatigue_modifier
    if fatigue.score < thresholds.high_fatigue:
        return
    yield RuleOutcome(
        rule="notification_fatigue",
        action=D,
        weight=thresholds.fatigue_dampener * fatigue.score * fatigue.confidence,
        reason="The user is already receiving a heavy volume of notifications.",
    )


def quiet_hours(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Hold non-urgent messages that arrive inside do-not-disturb."""
    if not context.features.context.in_quiet_hours:
        return
    if context.signals.urgency_modifier.score >= thresholds.strong_signal:
        return
    yield RuleOutcome(
        rule="quiet_hours",
        action=D,
        weight=thresholds.quiet_hours_dampener,
        reason="The message arrived during the user's quiet hours.",
    )


def disengaged_user(
    context: DecisionContext, thresholds: Thresholds
) -> Iterable[RuleOutcome]:
    """Interrupt a broadly disengaged user less often."""
    engagement = context.signals.engagement_modifier
    if engagement.score > thresholds.weak_signal or engagement.confidence == 0.0:
        return
    yield RuleOutcome(
        rule="disengaged_user",
        action=D,
        weight=thresholds.disengaged_user * engagement.confidence,
        reason="The user rarely acts on notifications.",
    )


#: Categories that describe a real transaction rather than marketing.
_TRANSACTIONAL_TYPES: Final[frozenset[MessageType]] = frozenset(
    {MessageType.BUSINESS_UPDATE, MessageType.PAYMENT, MessageType.EVENT}
)

#: Signature every rule implements.
RoutingRule = Callable[[DecisionContext, Thresholds], Iterable[RuleOutcome]]

#: Every rule the engine runs, in the order their reasons are collected.
#: Order affects explanation ordering only, never the outcome, because the
#: engine sums weights.
DEFAULT_RULES: Final[tuple[RoutingRule, ...]] = (
    type_prior,
    scam_override,
    risk_suppression,
    stranger_requesting_money,
    urgent_trusted_sender,
    verified_business_transaction,
    counterparty_standing,
    historical_importance,
    direct_mention,
    promotion_opted_out,
    promotion_previously_dismissed,
    promotion_from_trusted_business,
    muted_group,
    heavy_forwarding,
    notification_fatigue,
    quiet_hours,
    disengaged_user,
)

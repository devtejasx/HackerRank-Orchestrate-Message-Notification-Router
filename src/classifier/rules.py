"""Scoring rules that turn features into weighted evidence.

Each rule is a small pure function of ``(features, weights)`` yielding
:class:`Signal` objects. The classifier sums signal weights per
:class:`~src.classifier.enums.MessageType` and takes the strongest.

Why weighted signals rather than an if/elif ladder:

* several categories legitimately co-fire (a payment scam is both), and a
  ladder forces an arbitrary early exit;
* every contribution carries its own explanation, so the classification reason
  is derived from the evidence rather than written separately and drifting;
* adding a rule cannot break an existing one, which matters for Phase 3+.

Every magnitude lives in :class:`Weights`, so behaviour is tunable without
editing logic and no bare number appears in a comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

from src.classifier.enums import KeywordCategory, MessageType
from src.features.feature_models import MessageFeatures

__all__ = ["Signal", "Weights", "DEFAULT_WEIGHTS", "RULES", "collect_signals"]


@dataclass(frozen=True, slots=True)
class Signal:
    """One piece of weighted evidence for a category.

    Attributes:
        message_type: The category this evidence supports.
        weight: Strength of the evidence. Always positive; a rule that argues
            *against* a category simply does not emit a signal for it.
        reason: Short human-readable justification, reused verbatim in the
            classification reason.
    """

    message_type: MessageType
    weight: float
    reason: str


@dataclass(frozen=True, slots=True)
class Weights:
    """Every tunable magnitude used by the rules.

    Grouped by the rule that consumes them. Defaults were calibrated against
    the labelled examples in ``sample_messages.csv``.
    """

    # -- per-keyword contributions ------------------------------------- #
    scam_keyword: float = 1.40
    spam_keyword: float = 1.10
    urgent_keyword: float = 1.50
    payment_keyword: float = 1.00
    promotion_keyword: float = 1.15
    event_keyword: float = 1.35
    greeting_keyword: float = 1.60
    forward_keyword: float = 1.00
    transactional_keyword: float = 1.10

    #: Keyword matches counted per category before the contribution plateaus.
    #: Prevents one keyword-dense message from swamping every other signal.
    keyword_cap: int = 3

    # -- scam escalation ------------------------------------------------ #
    credential_request: float = 2.60
    domain_mismatch: float = 2.20
    scam_link: float = 1.60
    unverified_business_payment: float = 1.30
    stranger_payment_request: float = 1.60

    # -- spam ----------------------------------------------------------- #
    untrusted_bulk_sender: float = 1.40
    shouty_promotion: float = 0.90
    silent_business_media: float = 1.50

    # -- promotion / business --------------------------------------------#
    business_promotion: float = 1.00
    peer_selling: float = 1.40
    business_baseline: float = 0.70
    trusted_business_update: float = 1.10
    business_relationship: float = 0.50
    #: Appointments, bookings and collection slots are events even when a
    #: business is the one announcing them. Weighted to outrank the stacked
    #: business-update baseline, because "your appointment is at 4pm" is an
    #: event first and a business message second.
    business_event: float = 2.20
    #: A photo shared in a group with no scheduling or greeting language is
    #: usually an item being offered. Kept modest so any real keyword wins.
    peer_listing: float = 1.50

    # -- urgency / events ------------------------------------------------#
    admin_urgency: float = 1.20
    direct_mention: float = 1.30
    group_event: float = 1.50

    # -- forwarding ------------------------------------------------------#
    heavily_forwarded: float = 2.50
    moderately_forwarded: float = 0.90

    # -- conversational --------------------------------------------------#
    personal_baseline: float = 1.30
    group_chatter: float = 1.20
    silent_media_personal: float = 1.40
    #: Baseline for a one-to-one message from someone with no prior contact
    #: and no distinguishing keywords. Deliberately below
    #: :attr:`minimum_commit_score`, so such a message resolves to ``unknown``
    #: rather than being asserted as ordinary personal chat.
    personal_stranger: float = 0.80

    # -- thresholds ------------------------------------------------------#
    #: Forwarding count at which forwarding becomes the dominant story. A
    #: labelled greeting carries a count of 6 while the labelled forward
    #: carries 11, so the strong threshold sits above the former.
    forward_strong_threshold: int = 10
    forward_moderate_threshold: int = 5

    #: Business report count above which a sender is treated as bulk/untrusted.
    #: Set above the median because a modest report count is normal for any
    #: high-volume brand and is not by itself evidence of spam.
    business_report_threshold: int = 8

    #: Business account age below which a sender is treated as new.
    new_business_age_days: int = 120

    #: Total score below which the classifier declines to commit and returns
    #: ``unknown`` instead of guessing.
    minimum_commit_score: float = 1.10


DEFAULT_WEIGHTS: Final[Weights] = Weights()

#: Keyword category to the category it most directly supports, with the
#: per-match weight attribute name. Rules that need more than a direct mapping
#: are written explicitly below.
_DIRECT_KEYWORD_SUPPORT: Final[
    tuple[tuple[KeywordCategory, MessageType, str], ...]
] = (
    (KeywordCategory.SCAM, MessageType.SCAM, "scam_keyword"),
    (KeywordCategory.SPAM, MessageType.SPAM, "spam_keyword"),
    (KeywordCategory.URGENT, MessageType.URGENT, "urgent_keyword"),
    (KeywordCategory.PAYMENT, MessageType.PAYMENT, "payment_keyword"),
    (KeywordCategory.PROMOTION, MessageType.PROMOTION, "promotion_keyword"),
    (KeywordCategory.EVENT, MessageType.EVENT, "event_keyword"),
    (KeywordCategory.GREETING, MessageType.GREETING, "greeting_keyword"),
    (KeywordCategory.FORWARD, MessageType.FORWARD, "forward_keyword"),
    (
        KeywordCategory.TRANSACTIONAL,
        MessageType.BUSINESS_UPDATE,
        "transactional_keyword",
    ),
)

#: Keyword families that indicate a credential or one-time-code request.
_CREDENTIAL_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "otp", "login code", "verification code", "password", "cvv", "pin",
        "6 digit", "six digit", "reply with the",
    }
)

#: Keyword families that indicate account-loss pressure.
_ACCOUNT_PRESSURE_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "verify now", "verify account", "verify identity", "will be blocked",
        "temporarily blocked", "suspended", "reactivate", "kyc",
        "security alert", "support alert", "unusual activity",
    }
)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


def keyword_support(features: MessageFeatures, weights: Weights) -> Iterable[Signal]:
    """Convert direct keyword matches into weighted support.

    Contribution is capped at :attr:`Weights.keyword_cap` matches per category
    so a single keyword-dense message cannot drown out context.
    """
    for category, message_type, weight_field in _DIRECT_KEYWORD_SUPPORT:
        count = features.keywords.count(category)
        if count == 0:
            continue
        per_match = getattr(weights, weight_field)
        effective = min(count, weights.keyword_cap)
        matched = ", ".join(features.keywords.words(category)[: weights.keyword_cap])
        yield Signal(
            message_type,
            per_match * effective,
            f"matched {category.value} keywords ({matched})",
        )


def credential_harvesting(
    features: MessageFeatures, weights: Weights
) -> Iterable[Signal]:
    """Escalate to scam when the message asks for a secret under pressure.

    Asking for a one-time code, PIN or password is the single most reliable
    scam tell in this dataset, and it is stronger still when paired with
    account-loss pressure such as "verify now or your profile is blocked".
    """
    matched = set(features.keywords.words(KeywordCategory.SCAM))
    asks_for_secret = bool(matched & _CREDENTIAL_KEYWORDS)
    applies_pressure = bool(matched & _ACCOUNT_PRESSURE_KEYWORDS)

    if asks_for_secret and applies_pressure:
        yield Signal(
            MessageType.SCAM,
            weights.credential_request,
            "requests a credential while threatening account loss",
        )
    elif asks_for_secret:
        yield Signal(
            MessageType.SCAM,
            weights.credential_request * 0.6,
            "requests a one-time code or password",
        )


def impersonation(features: MessageFeatures, weights: Weights) -> Iterable[Signal]:
    """Escalate to scam when a business sends from a domain that is not its own.

    A mismatch alone is not proof of impersonation. Large brands routinely send
    from a marketing subdomain or an email-service domain, so a *verified*
    business that the recipient already deals with, sending nothing that asks
    for anything, is not treated as a scam. The mismatch becomes strong
    evidence again as soon as the sender is unverified, unknown to the user, or
    using scam language.
    """
    if features.context.has_domain_mismatch and not _is_known_verified_sender(features):
        yield Signal(
            MessageType.SCAM,
            weights.domain_mismatch,
            "business is sending from a domain that is not its official one",
        )

    if features.text.contains_url and features.keywords.has(KeywordCategory.SCAM):
        yield Signal(
            MessageType.SCAM,
            weights.scam_link,
            "links to an external site alongside scam language",
        )


def _is_known_verified_sender(features: MessageFeatures) -> bool:
    """Whether the sender is a verified brand the recipient already deals with.

    Requires the absence of scam language, so a compromised or spoofed
    verified account still escalates.
    """
    return (
        features.context.business_verified
        and features.history.has_business_relationship
        and not features.keywords.has(KeywordCategory.SCAM)
    )


def risky_payment_request(
    features: MessageFeatures, weights: Weights
) -> Iterable[Signal]:
    """Escalate payment language to scam when the sender has no standing.

    A payment request is ordinary from a business the user already deals with
    and suspicious from an unverified business or an unknown individual.
    """
    wants_money = features.keywords.has(KeywordCategory.PAYMENT) or (
        features.text.contains_payment_symbol
    )
    if not wants_money:
        return

    if features.context.business_exists and not features.context.business_verified:
        yield Signal(
            MessageType.SCAM,
            weights.unverified_business_payment,
            "unverified business is asking for a payment",
        )

    is_stranger = (
        features.context.is_personal
        and features.history.sender_message_count == 0
    )
    if is_stranger:
        yield Signal(
            MessageType.SCAM,
            weights.stranger_payment_request,
            "payment language from a sender with no prior contact",
        )


def bulk_sender(features: MessageFeatures, weights: Weights) -> Iterable[Signal]:
    """Treat heavily reported or brand-new business senders as spam-leaning.

    Suppressed when the user already deals with this business: a brand the
    recipient orders from is not spamming them merely because strangers have
    reported it.
    """
    context = features.context
    if not context.business_exists or features.history.has_business_relationship:
        return

    reports = context.business_reports_30d or 0
    is_new = (
        context.business_age_days is not None
        and context.business_age_days < weights.new_business_age_days
    )
    if reports > weights.business_report_threshold or is_new:
        yield Signal(
            MessageType.SPAM,
            weights.untrusted_bulk_sender,
            f"unfamiliar sender with {reports} recent report(s)"
            + (" and a new account" if is_new else ""),
        )

    if features.text.is_shouty and features.keywords.has(KeywordCategory.PROMOTION):
        yield Signal(
            MessageType.SPAM,
            weights.shouty_promotion,
            "shouting promotional copy",
        )


def silent_media(features: MessageFeatures, weights: Weights) -> Iterable[Signal]:
    """Handle messages that carry media and no text at all.

    Without OCR or speech recognition the body is unreadable, so the sender
    context is the only evidence available: an unsolicited business voice note
    reads as spam, while the same thing between people is ordinary chatter.
    """
    if not (features.has_media and features.text.is_empty):
        return

    if features.context.is_business:
        yield Signal(
            MessageType.SPAM,
            weights.silent_business_media,
            "business sent media with no accompanying text",
        )
    else:
        yield Signal(
            MessageType.PERSONAL,
            weights.silent_media_personal,
            f"{features.media_type} note with no text, from a person",
        )


def business_intent(features: MessageFeatures, weights: Weights) -> Iterable[Signal]:
    """Separate transactional business updates from business marketing."""
    context = features.context
    if not context.is_business:
        return

    is_marketing = features.keywords.has(KeywordCategory.PROMOTION) or (
        features.keywords.has(KeywordCategory.SPAM)
    )
    if is_marketing:
        yield Signal(
            MessageType.PROMOTION,
            weights.business_promotion,
            "business account sending promotional copy",
        )
        return

    yield Signal(
        MessageType.BUSINESS_UPDATE,
        weights.business_baseline,
        "message from a business account",
    )
    if context.is_trusted_business:
        yield Signal(
            MessageType.BUSINESS_UPDATE,
            weights.trusted_business_update,
            "verified business using its official domain",
        )
    if features.history.has_business_relationship:
        yield Signal(
            MessageType.BUSINESS_UPDATE,
            weights.business_relationship,
            "user already has a relationship with this business",
        )
    if features.keywords.has(KeywordCategory.EVENT):
        yield Signal(
            MessageType.EVENT,
            weights.business_event,
            "business announcing an appointment or scheduled slot",
        )


#: Keyword families that give a group photo a purpose other than selling.
_LISTING_EXCLUSIONS: Final[tuple[KeywordCategory, ...]] = (
    KeywordCategory.EVENT,
    KeywordCategory.GREETING,
    KeywordCategory.URGENT,
    KeywordCategory.SCAM,
    KeywordCategory.SPAM,
    KeywordCategory.TRANSACTIONAL,
)


def peer_selling(features: MessageFeatures, weights: Weights) -> Iterable[Signal]:
    """Recognise person-to-person selling inside a group as a promotion.

    Two routes. Explicit selling language is the strong one. The weaker one is
    a photo posted to a group with no scheduling, greeting or risk language:
    that is far more often an item being offered than anything else, and it is
    the only signal available when the listing carries no sales vocabulary at
    all ("Photos for the kurta set are attached").
    """
    if features.context.is_business or not features.context.is_group:
        return

    if features.keywords.has(KeywordCategory.PROMOTION):
        yield Signal(
            MessageType.PROMOTION,
            weights.peer_selling,
            "group member advertising an item",
        )
        return

    is_bare_photo = (
        features.media_type == "image"
        and not features.text.is_empty
        and not any(features.keywords.has(family) for family in _LISTING_EXCLUSIONS)
    )
    if is_bare_photo:
        yield Signal(
            MessageType.PROMOTION,
            weights.peer_listing,
            "photo shared in a group with no scheduling or greeting context",
        )


def urgency_context(features: MessageFeatures, weights: Weights) -> Iterable[Signal]:
    """Amplify urgency that comes from an authority or names the recipient."""
    if not features.keywords.has(KeywordCategory.URGENT):
        return

    if features.context.sender_is_admin:
        yield Signal(
            MessageType.URGENT,
            weights.admin_urgency,
            "group admin raising a time-sensitive issue",
        )
    if _mentions_recipient(features):
        yield Signal(
            MessageType.URGENT,
            weights.direct_mention,
            "message names the recipient directly",
        )


def event_context(features: MessageFeatures, weights: Weights) -> Iterable[Signal]:
    """Group logistics are usually events rather than personal chatter."""
    if features.context.is_group and features.keywords.has(KeywordCategory.EVENT):
        yield Signal(
            MessageType.EVENT,
            weights.group_event,
            "group coordinating a scheduled activity",
        )


def forwarding(features: MessageFeatures, weights: Weights) -> Iterable[Signal]:
    """Weigh the platform forwarding counter.

    Two thresholds rather than one: a moderately forwarded good-wishes message
    is still a greeting, so only a high count makes forwarding the story.
    """
    count = features.forwarded_count
    if count >= weights.forward_strong_threshold:
        yield Signal(
            MessageType.FORWARD,
            weights.heavily_forwarded,
            f"forwarded {count} times",
        )
    elif count >= weights.forward_moderate_threshold:
        yield Signal(
            MessageType.FORWARD,
            weights.moderately_forwarded,
            f"forwarded {count} times",
        )


def conversational(features: MessageFeatures, weights: Weights) -> Iterable[Signal]:
    """Baseline support for ordinary human conversation.

    A one-to-one message from someone the user has never heard from, carrying
    no distinguishing keywords, gets a deliberately weak baseline. There is
    genuinely not enough evidence to call it ordinary personal chat, and
    ``unknown`` is the more honest verdict.
    """
    if features.context.is_personal:
        is_stranger = (
            features.history.sender_message_count == 0
            and features.keywords.total_matches == 0
        )
        yield Signal(
            MessageType.PERSONAL,
            weights.personal_stranger if is_stranger else weights.personal_baseline,
            "message from an unfamiliar sender with no distinctive content"
            if is_stranger
            else "one-to-one conversation",
        )
    elif features.context.is_group and not features.text.is_empty:
        yield Signal(
            MessageType.PERSONAL,
            weights.group_chatter,
            "conversational message in a group",
        )
    if _mentions_recipient(features) and not features.keywords.has(
        KeywordCategory.URGENT
    ):
        yield Signal(
            MessageType.PERSONAL,
            weights.direct_mention,
            "message names the recipient directly",
        )


#: Every rule applied, in declaration order. Order affects only the order of
#: reasons, never the outcome, because scores are summed.
RULES: Final[tuple[Callable[[MessageFeatures, Weights], Iterable[Signal]], ...]] = (
    keyword_support,
    credential_harvesting,
    impersonation,
    risky_payment_request,
    bulk_sender,
    silent_media,
    business_intent,
    peer_selling,
    urgency_context,
    event_context,
    forwarding,
    conversational,
)


def collect_signals(
    features: MessageFeatures, weights: Weights = DEFAULT_WEIGHTS
) -> tuple[Signal, ...]:
    """Run every rule and return the evidence they produce.

    Args:
        features: Extracted features for one message.
        weights: Tuning to apply.

    Returns:
        Every signal emitted, unaggregated, in rule declaration order.
    """
    return tuple(
        signal for rule in RULES for signal in rule(features, weights)
    )


def _mentions_recipient(features: MessageFeatures) -> bool:
    """Whether the body names the recipient with an ``@user_id`` mention."""
    if features.text.is_empty:
        return False
    return f"@{features.user_id}".casefold() in features.text.normalized_text

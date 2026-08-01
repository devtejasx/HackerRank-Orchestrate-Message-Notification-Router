"""How much this business matters to this recipient.

Reads ``business_accounts.csv`` for what the brand is, and
``user_business_history.csv`` for what the user actually does with it. The
second matters more: a verified household-name brand the user never opens is
worth less than a small vendor whose delivery updates they read every time.

``why_user_knows_account`` is free text, but it is not unstructured. Across the
dataset it decomposes into a *commitment* term (``booking``, ``payment``,
``order``, ``subscription``) or an *intent* term (``search``, ``watchlist``,
``interest``), optionally qualified by recency (``active_``, ``recent_``,
``upcoming_`` against ``old_``, ``abandoned_``). Parsing those tokens gives a
transaction likelihood grounded in the data rather than an invented constant.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

from src.data.models import BusinessAccount, UserBusinessHistory
from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.normalization import (
    NEUTRAL,
    Contribution,
    days_between,
    decay,
    evidence_confidence,
    saturating,
)
from src.personalization.signal_models import SignalPolarity
from src.utils.text_utils import tokenize

__all__ = ["BusinessPriorityCalculator", "transaction_likelihood"]

#: Tokens meaning the user committed to something: money moved, a slot was
#: held, or goods are in transit.
_COMMITMENT_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "booking", "booked", "order", "orders", "purchase", "payment", "paid",
        "bill", "account", "appointment", "refill", "claim", "renewal",
        "reservation", "receipt", "registration", "registered", "subscription",
        "membership", "delivery", "wallet", "challan", "insurance", "loan",
        "card", "fee", "signup", "ticket", "prescription", "medicine", "service",
    }
)

#: Tokens meaning the user only looked. Browsing is far weaker evidence of
#: interest than committing.
_INTENT_TOKENS: Final[frozenset[str]] = frozenset(
    {"search", "watchlist", "interest", "listing", "coupon", "promotions", "adaptation"}
)

#: Qualifiers marking the relationship as live.
_ACTIVE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "active", "recent", "frequent", "upcoming", "confirmed", "today",
        "weekly", "monthly", "daily", "daytime", "quick", "new", "expected",
    }
)

#: Qualifiers marking it as lapsed.
_STALE_TOKENS: Final[frozenset[str]] = frozenset(
    {"old", "abandoned", "saved", "ignored", "former", "past"}
)

#: Likelihood assigned before recency qualifiers are applied.
_COMMITMENT_BASE: Final[float] = 0.75
_INTENT_BASE: Final[float] = 0.30

#: How far an active or stale qualifier moves the base likelihood.
_QUALIFIER_SHIFT: Final[float] = 0.20

#: Recorded interactions at which a business relationship is half-established.
ACTIVITY_HALF_POINT: Final[float] = 4.0

#: Days after which a business interaction feels half as current. Longer than
#: the sender half-life: a monthly bill is still a live relationship.
RECENCY_HALF_LIFE_DAYS: Final[float] = 30.0

#: Reports at which a brand's reputation is half-eroded.
REPORTS_HALF_POINT: Final[float] = 10.0

#: Account age at which a brand counts as established.
ESTABLISHED_HALF_POINT: Final[float] = 365.0

#: Messages needed before business statistics are half-trusted.
CONFIDENCE_HALF_POINT: Final[float] = 3.0


def transaction_likelihood(why_user_knows_account: str) -> float:
    """Score how likely this relationship involves a real transaction.

    Args:
        why_user_knows_account: The free-text reason from
            ``user_business_history.csv``, e.g. ``recent_flight_booking``.

    Returns:
        A value in ``[0, 1]``: high for live commitments such as
        ``confirmed_travel_booking``, low for mere browsing such as
        ``abandoned_travel_search``, and neutral when nothing is recognised.
    """
    tokens = set(tokenize(why_user_knows_account.replace("_", " ")))
    if not tokens:
        return NEUTRAL

    if tokens & _COMMITMENT_TOKENS:
        base = _COMMITMENT_BASE
    elif tokens & _INTENT_TOKENS:
        base = _INTENT_BASE
    else:
        base = NEUTRAL

    if tokens & _ACTIVE_TOKENS:
        base += _QUALIFIER_SHIFT
    elif tokens & _STALE_TOKENS:
        base -= _QUALIFIER_SHIFT
    return min(1.0, max(0.0, base))


class BusinessPriorityCalculator(SignalCalculator):
    """Scores the standing of a business sender with this recipient."""

    name: ClassVar[str] = "business_priority"
    polarity: ClassVar[SignalPolarity] = SignalPolarity.BOOST

    def contributions(self, context: SignalContext) -> Sequence[Contribution]:
        """Return the evidence about this business.

        Empty when no business sent the message, landing the signal on neutral.
        """
        business = (
            context.repo.get_business(context.business_id)
            if context.business_id
            else None
        )
        if business is None:
            return ()

        relationship = context.repo.get_user_business(
            context.user_id, business.business_id
        )
        return (
            *self._brand_contributions(business),
            *self._relationship_contributions(relationship, context),
            self._engagement_contribution(context),
        )

    def _brand_contributions(
        self, business: BusinessAccount
    ) -> tuple[Contribution, ...]:
        """Evidence about the brand itself, independent of this user."""
        return (
            Contribution(
                name="verified",
                value=1.0 if business.verified else 0.0,
                weight=0.15,
                high_reason=f"Verified business ({business.display_name}).",
                low_reason=f"Business is not verified ({business.display_name}).",
            ),
            Contribution(
                name="low_reports",
                value=1.0 - saturating(business.user_reports_30d, REPORTS_HALF_POINT),
                weight=0.12,
                low_reason=(
                    f"Business has {business.user_reports_30d} recent user reports."
                ),
            ),
            Contribution(
                name="established",
                value=saturating(business.account_age_days, ESTABLISHED_HALF_POINT),
                weight=0.06,
                low_reason="Business account is new.",
            ),
        )

    def _relationship_contributions(
        self, relationship: UserBusinessHistory | None, context: SignalContext
    ) -> tuple[Contribution, ...]:
        """Evidence about what this user has actually done with the brand."""
        if relationship is None:
            return (
                Contribution(
                    name="permission",
                    value=NEUTRAL,
                    weight=0.15,
                ),
                Contribution(
                    name="transaction_likelihood",
                    value=0.0,
                    weight=0.18,
                    low_reason="User has no recorded relationship with this business.",
                ),
                Contribution(
                    name="recency",
                    value=0.0,
                    weight=0.14,
                ),
            )

        elapsed = days_between(relationship.last_activity_at, context.now)
        likelihood = transaction_likelihood(relationship.why_user_knows_account)
        activity = saturating(relationship.activity_count_180d, ACTIVITY_HALF_POINT)

        return (
            Contribution(
                name="permission",
                value=self._permission_value(relationship),
                weight=0.15,
                high_reason="User opted into messages from this business.",
                low_reason=(
                    "User opted out of promotions from this business."
                    if relationship.has_opted_out
                    else "User has not opted into promotions from this business."
                ),
            ),
            Contribution(
                name="transaction_likelihood",
                value=(likelihood + activity) / 2.0,
                weight=0.18,
                high_reason=(
                    f"Active commercial relationship "
                    f"({relationship.why_user_knows_account.replace('_', ' ')})."
                ),
                low_reason=(
                    f"Relationship is browsing-only or lapsed "
                    f"({relationship.why_user_knows_account.replace('_', ' ')})."
                ),
            ),
            Contribution(
                name="recency",
                value=decay(elapsed, RECENCY_HALF_LIFE_DAYS),
                weight=0.14,
                high_reason="Recent activity with this business.",
                low_reason="No recent activity with this business.",
            ),
        )

    @staticmethod
    def _permission_value(relationship: UserBusinessHistory) -> float:
        """Map opt-in state onto a score.

        An explicit opt-out is a direct instruction and scores zero; an opt-in
        scores one; silence is neutral rather than either.
        """
        if relationship.has_opted_out:
            return 0.0
        return 1.0 if relationship.allows_promotions else NEUTRAL

    @staticmethod
    def _engagement_contribution(context: SignalContext) -> Contribution:
        """How this user has treated this brand's past messages."""
        stats = context.business_stats
        return Contribution(
            name="engagement",
            value=stats.engagement,
            weight=0.20,
            high_reason="User regularly engages with this business's messages.",
            low_reason=(
                "User usually ignores this business's messages."
                if stats.has_history
                else "No prior messages from this business."
            ),
        )

    def confidence(self, context: SignalContext) -> float:
        """Confidence comes from prior message volume plus a known relationship.

        A recorded relationship is itself evidence even before any message has
        been exchanged, so it counts toward the sample.
        """
        if context.business_id is None:
            return 0.0
        has_relationship = (
            context.repo.get_user_business(context.user_id, context.business_id)
            is not None
        )
        sample = context.business_stats.total + (1 if has_relationship else 0)
        return evidence_confidence(sample, CONFIDENCE_HALF_POINT)

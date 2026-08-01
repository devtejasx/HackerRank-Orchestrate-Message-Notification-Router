"""How trustworthy the sender is, from whatever standing is on record.

Trust is deliberately separate from priority. A verified bank the user has
never opened is highly trustworthy and low priority; a chatty friend is high
priority and carries no institutional trust at all. Phase 4 needs both, and
collapsing them would lose the distinction.

Only the components that apply contribute. Because the score is a weighted
mean, omitting an inapplicable component re-normalises the rest automatically
rather than dragging the result toward zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.normalization import (
    NEUTRAL,
    Contribution,
    evidence_confidence,
    saturating,
)
from src.personalization.signal_models import SignalPolarity

__all__ = ["TrustCalculator"]

#: Reports at which a brand's standing is half-eroded.
REPORTS_HALF_POINT: Final[float] = 10.0

#: Account age at which a brand counts as half-established.
ESTABLISHED_HALF_POINT: Final[float] = 365.0

#: Prior messages at which a sender is half-vouched-for by familiarity alone.
FAMILIARITY_HALF_POINT: Final[float] = 5.0

#: Evidence items needed before the trust assessment is half-trusted.
CONFIDENCE_HALF_POINT: Final[float] = 3.0


class TrustCalculator(SignalCalculator):
    """Scores how much standing the sender has with this recipient."""

    name: ClassVar[str] = "trust_modifier"
    polarity: ClassVar[SignalPolarity] = SignalPolarity.BOOST

    def contributions(self, context: SignalContext) -> Sequence[Contribution]:
        """Return whichever trust components apply to this message."""
        return (
            *self._sender_trust(context),
            *self._business_trust(context),
            *self._group_trust(context),
            *self._relationship_trust(context),
        )

    @staticmethod
    def _sender_trust(context: SignalContext) -> tuple[Contribution, ...]:
        """Standing earned by an individual through prior contact."""
        if context.sender_user_id is None:
            return ()
        stats = context.sender_stats
        return (
            Contribution(
                name="sender_familiarity",
                value=saturating(stats.total, FAMILIARITY_HALF_POINT),
                weight=0.20,
                high_reason="Sender is a known contact.",
                low_reason="Sender is unknown to this user.",
            ),
            Contribution(
                name="sender_unreported",
                value=1.0 - stats.report_rate,
                weight=0.15,
                low_reason="User has reported this sender before.",
            ),
        )

    @staticmethod
    def _business_trust(context: SignalContext) -> tuple[Contribution, ...]:
        """Standing carried by a business account's own credentials."""
        business = (
            context.repo.get_business(context.business_id)
            if context.business_id
            else None
        )
        if business is None:
            return ()

        domain_matches = business.sender_domain_matches_official
        return (
            Contribution(
                name="business_verified",
                value=1.0 if business.verified else 0.0,
                weight=0.22,
                high_reason="Verified business account.",
                low_reason="Business account is unverified.",
            ),
            Contribution(
                name="business_domain",
                # Unknown is neutral, not suspicious: absence of data is not
                # evidence of wrongdoing.
                value=NEUTRAL if domain_matches is None else float(domain_matches),
                weight=0.20,
                high_reason="Sending domain matches the brand's official domain.",
                low_reason="Sending domain does not match the brand's official domain.",
            ),
            Contribution(
                name="business_standing",
                value=1.0 - saturating(business.user_reports_30d, REPORTS_HALF_POINT),
                weight=0.14,
                low_reason=f"Business has {business.user_reports_30d} recent reports.",
            ),
            Contribution(
                name="business_established",
                value=saturating(business.account_age_days, ESTABLISHED_HALF_POINT),
                weight=0.08,
                low_reason="Business account was created recently.",
            ),
        )

    @staticmethod
    def _group_trust(context: SignalContext) -> tuple[Contribution, ...]:
        """Standing conferred by the group context and the sender's role in it."""
        if context.group_id is None:
            return ()

        membership = context.repo.get_group_member(context.group_id, context.user_id)
        return (
            Contribution(
                name="group_membership",
                value=1.0 if membership is not None else 0.0,
                weight=0.12,
                low_reason="User is not a recorded member of this group.",
            ),
            Contribution(
                name="group_authority",
                value=1.0 if context.features.context.sender_is_admin else NEUTRAL,
                weight=0.10,
                high_reason="Sender is an admin of this group.",
            ),
        )

    @staticmethod
    def _relationship_trust(context: SignalContext) -> tuple[Contribution, ...]:
        """Standing earned by an existing commercial relationship."""
        if context.business_id is None:
            return ()

        relationship = context.repo.get_user_business(
            context.user_id, context.business_id
        )
        return (
            Contribution(
                name="relationship_trust",
                value=1.0 if relationship is not None else 0.0,
                weight=0.16,
                high_reason="User has an established relationship with this business.",
                low_reason="No established relationship with this business.",
            ),
        )

    def confidence(self, context: SignalContext) -> float:
        """Confidence grows with how many independent trust components applied."""
        return evidence_confidence(
            len(self.contributions(context)), CONFIDENCE_HALF_POINT
        )

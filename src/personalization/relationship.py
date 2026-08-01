"""Strength of the tie between the recipient and whoever sent this message.

Distinct from the three priority signals, which each ask "how much does this
*channel* matter". This one asks the channel-agnostic question: how strong is
the bond with the counterparty, whoever they are. It resolves to the sender,
the business or the group depending on how the message arrived, so Phase 4 can
read one number without first working out which kind of message it is holding.

Three things make a tie strong, and they are deliberately different from the
priority signals' inputs so this is not a restatement of them:

* **volume** - how much has passed between the two parties;
* **reciprocity** - whether the recipient answers, which is what separates a
  relationship from a subscription;
* **duration** - how long the tie has existed, which no single interaction
  reveals.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.interaction_stats import InteractionStats
from src.personalization.normalization import (
    Contribution,
    days_between,
    decay,
    evidence_confidence,
    saturating,
)
from src.personalization.signal_models import SignalPolarity

__all__ = ["RelationshipStrengthCalculator"]

#: Exchanged messages at which a tie is half-formed.
VOLUME_HALF_POINT: Final[float] = 6.0

#: Days of shared history at which a tie counts as half-established. Roughly a
#: season: long enough to outlast a single burst of activity.
DURATION_HALF_POINT: Final[float] = 90.0

#: Days after which a dormant tie feels half as live.
DORMANCY_HALF_LIFE_DAYS: Final[float] = 21.0

#: Interactions needed before the tie assessment is half-trusted.
CONFIDENCE_HALF_POINT: Final[float] = 5.0


class RelationshipStrengthCalculator(SignalCalculator):
    """Scores how strong the recipient's tie is to this counterparty."""

    name: ClassVar[str] = "relationship_strength"
    polarity: ClassVar[SignalPolarity] = SignalPolarity.BOOST

    def contributions(self, context: SignalContext) -> Sequence[Contribution]:
        """Return the evidence about the tie, whichever counterparty applies."""
        stats, label = self._counterparty(context)
        if stats is None:
            return ()

        elapsed = days_between(stats.last_seen, context.now)
        return (
            Contribution(
                name="volume",
                value=saturating(stats.total, VOLUME_HALF_POINT),
                weight=0.35,
                high_reason=f"Substantial message history with this {label}.",
                low_reason=f"Little or no history with this {label}.",
            ),
            Contribution(
                name="reciprocity",
                value=stats.reply_rate,
                weight=0.40,
                high_reason=f"User actively converses with this {label}.",
                low_reason=f"Communication with this {label} is one-way.",
            ),
            Contribution(
                name="duration",
                value=saturating(stats.span_days, DURATION_HALF_POINT),
                weight=0.15,
                high_reason=f"Long-standing relationship with this {label}.",
            ),
            Contribution(
                name="liveness",
                value=decay(elapsed, DORMANCY_HALF_LIFE_DAYS),
                weight=0.10,
                low_reason=f"Relationship with this {label} has gone quiet.",
            ),
        )

    @staticmethod
    def _counterparty(
        context: SignalContext,
    ) -> tuple[InteractionStats | None, str]:
        """Resolve which counterparty this message's tie is with.

        Returns:
            The applicable statistics and a noun for use in explanations, or
            ``(None, "")`` when the message has no identifiable counterparty.
        """
        if context.sender_user_id is not None:
            return context.sender_stats, "sender"
        if context.business_id is not None:
            return context.business_stats, "business"
        if context.group_id is not None:
            return context.group_stats, "group"
        return None, ""

    def confidence(self, context: SignalContext) -> float:
        """Confidence grows with the number of interactions behind the tie."""
        stats, _ = self._counterparty(context)
        if stats is None:
            return 0.0
        return evidence_confidence(stats.total, CONFIDENCE_HALF_POINT)

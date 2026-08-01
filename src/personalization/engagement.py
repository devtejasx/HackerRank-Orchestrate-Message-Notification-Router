"""How engaged this recipient is with notifications in general.

A user-level baseline rather than a message-level judgement. Someone who opens
and answers almost everything can absorb an interruption cheaply; someone who
dismisses most of what arrives cannot, and the same message should reach them
differently.

The base rates come straight from Phase 2's ``features.history`` - they are
already computed and there is no reason to derive them twice. The trend is the
one thing Phase 2 does not provide, so it comes from the scoped statistics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.normalization import (
    Contribution,
    evidence_confidence,
    saturating,
)
from src.personalization.signal_models import SignalPolarity

__all__ = ["EngagementCalculator"]

#: Recorded interactions at which a user's habits are half-known.
CONFIDENCE_HALF_POINT: Final[float] = 8.0

#: Messages opened in 30 days at which a user counts as broadly active. Used
#: only as a weak corroborating signal, since it is not personalised to the
#: sender in any way.
RECENT_ACTIVITY_HALF_POINT: Final[float] = 40.0


class EngagementCalculator(SignalCalculator):
    """Scores this recipient's overall receptiveness to notifications."""

    name: ClassVar[str] = "engagement_modifier"
    polarity: ClassVar[SignalPolarity] = SignalPolarity.BOOST

    def contributions(self, context: SignalContext) -> Sequence[Contribution]:
        """Return the evidence about this user's notification habits."""
        history = context.features.history
        stats = context.user_stats

        return (
            Contribution(
                name="open_rate",
                value=history.open_rate,
                weight=0.26,
                high_reason=f"User opens most notifications ({history.open_rate:.0%}).",
                low_reason=f"User opens few notifications ({history.open_rate:.0%}).",
            ),
            Contribution(
                name="reply_rate",
                value=history.reply_rate,
                weight=0.28,
                high_reason=f"User replies frequently ({history.reply_rate:.0%}).",
                low_reason="User rarely replies to messages.",
            ),
            Contribution(
                name="not_dismissed",
                value=1.0 - history.dismiss_rate,
                weight=0.18,
                low_reason=(
                    f"User dismisses notifications often ({history.dismiss_rate:.0%})."
                ),
            ),
            Contribution(
                name="not_reported",
                value=1.0 - history.report_rate,
                weight=0.12,
                low_reason=(
                    f"User reports messages at an elevated rate "
                    f"({history.report_rate:.0%})."
                ),
            ),
            Contribution(
                name="engagement_trend",
                value=stats.engagement_trend,
                weight=0.10,
                high_reason="User's engagement is rising.",
                low_reason="User's engagement is declining.",
            ),
            Contribution(
                name="recent_activity",
                value=saturating(
                    context.features.context.recent_activity, RECENT_ACTIVITY_HALF_POINT
                ),
                weight=0.06,
                low_reason="User has been largely inactive recently.",
            ),
        )

    def confidence(self, context: SignalContext) -> float:
        """Confidence grows with the user's total recorded interactions."""
        return evidence_confidence(
            context.features.history.total_interactions, CONFIDENCE_HALF_POINT
        )

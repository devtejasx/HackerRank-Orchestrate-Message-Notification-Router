"""How overloaded the recipient already is.

A **suppressing** signal: a rising score argues for holding the message back.
The same message is worth interrupting for on a quiet day and not worth it on
the user's tenth notification of the morning.

Note a real limitation of the shipped dataset, established in Phase 1:
``daily_notification_summary`` covers 2026-07-04 to 07-17 while every incoming
message falls in 2026-07-18 to 07-31, so a same-day lookup misses for all 110
messages. The rolling window therefore ends at the last *recorded* day rather
than at the message date. That is the most recent evidence available, and
treating an absent day as zero load would be a much worse error - it would
read every user as completely unfatigued.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

from src.data.models import NotificationSummary
from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.normalization import (
    NEUTRAL,
    Contribution,
    evidence_confidence,
    saturating,
    trend_score,
)
from src.personalization.signal_models import SignalPolarity
from src.utils.helpers import ratio

__all__ = ["FatigueCalculator"]

#: Notifications per day at which a user counts as half-overloaded. The
#: dataset's per-user averages run from 3.6 to 11.1, so this sits mid-range
#: and separates the light from the heavy without saturating either.
OVERLOAD_HALF_POINT: Final[float] = 8.0

#: Days in the rolling window. A week smooths weekday and weekend rhythms
#: without reaching so far back that it stops describing current load.
ROLLING_WINDOW_DAYS: Final[int] = 7

#: Recorded days needed before the fatigue estimate is half-trusted.
CONFIDENCE_HALF_POINT: Final[float] = 5.0


class FatigueCalculator(SignalCalculator):
    """Scores how much notification load this recipient is already carrying."""

    name: ClassVar[str] = "fatigue_modifier"
    polarity: ClassVar[SignalPolarity] = SignalPolarity.SUPPRESS

    def contributions(self, context: SignalContext) -> Sequence[Contribution]:
        """Return the evidence about this user's notification load."""
        rows = self._recorded_days(context)
        quiet_hours = self._quiet_hours_contribution(context)

        if not rows:
            return (quiet_hours,)

        rolling = self._mean_sent(rows[-ROLLING_WINDOW_DAYS:])
        overall = self._mean_sent(rows)
        dismiss_rate = context.features.context.notification_dismiss_rate

        return (
            Contribution(
                name="rolling_volume",
                value=saturating(rolling, OVERLOAD_HALF_POINT),
                weight=0.34,
                high_reason=(
                    f"High notification load ({rolling:.1f} per day over the last "
                    f"{min(len(rows), ROLLING_WINDOW_DAYS)} recorded days)."
                ),
                low_reason=f"Light notification load ({rolling:.1f} per day).",
            ),
            Contribution(
                name="dismissal_pressure",
                value=dismiss_rate,
                weight=0.28,
                high_reason=(
                    f"User dismisses {dismiss_rate:.0%} of the notifications they get."
                ),
                low_reason="User rarely dismisses notifications.",
            ),
            Contribution(
                name="load_trend",
                value=trend_score(
                    saturating(rolling, OVERLOAD_HALF_POINT),
                    saturating(overall, OVERLOAD_HALF_POINT),
                ),
                weight=0.14,
                high_reason="Notification load is rising.",
                low_reason="Notification load is falling.",
            ),
            quiet_hours,
        )

    @staticmethod
    def _quiet_hours_contribution(context: SignalContext) -> Contribution:
        """Landing inside do-not-disturb makes any interruption costlier."""
        in_quiet_hours = context.features.context.in_quiet_hours
        return Contribution(
            name="quiet_hours",
            value=1.0 if in_quiet_hours else 0.0,
            weight=0.24,
            high_reason="Message arrived inside the user's quiet hours.",
            low_reason=None,
        )

    @staticmethod
    def _recorded_days(context: SignalContext) -> tuple[NotificationSummary, ...]:
        """Return this user's notification days, oldest first."""
        rows = context.repo.get_notification_summary(context.user_id)
        return tuple(sorted(rows, key=lambda row: row.date))

    @staticmethod
    def _mean_sent(rows: Sequence[NotificationSummary]) -> float:
        """Mean notifications sent per day across ``rows``."""
        if not rows:
            return 0.0
        return ratio(sum(row.notifications_sent for row in rows), len(rows))

    def confidence(self, context: SignalContext) -> float:
        """Confidence grows with the number of recorded notification days.

        Quiet hours alone is a real but thin basis, so a user with no recorded
        days still gets non-zero confidence when the message lands in their
        do-not-disturb window.
        """
        recorded = len(self._recorded_days(context))
        base = evidence_confidence(recorded, CONFIDENCE_HALF_POINT)
        if recorded == 0 and context.features.context.in_quiet_hours:
            return QUIET_HOURS_ONLY_CONFIDENCE
        return base


#: Confidence when quiet hours is the only fatigue evidence available. Low, but
#: not zero: the do-not-disturb window is a setting the user chose themselves.
QUIET_HOURS_ONLY_CONFIDENCE: Final[float] = 0.30

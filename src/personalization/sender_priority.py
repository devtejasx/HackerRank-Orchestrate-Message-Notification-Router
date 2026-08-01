"""How much this individual sender matters to this recipient.

Applies when a person sent the message - a personal chat or a named sender in
a group. Business messages have no individual sender, so the signal reports
neutral with zero confidence rather than inventing a value.

The ordering of the weights encodes one claim: **replying is the strongest
evidence that a sender matters**. Opening a message shows it got through;
answering it shows the recipient chose to spend time on that person.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.normalization import (
    Contribution,
    days_between,
    decay,
    evidence_confidence,
    saturating,
)
from src.personalization.signal_models import SignalPolarity

__all__ = ["SenderPriorityCalculator"]

#: Prior messages at which familiarity with a sender is half-formed. Set low
#: because in one-to-one messaging a handful of exchanges already marks
#: someone as a known contact.
FAMILIARITY_HALF_POINT: Final[float] = 5.0

#: Days after which a contact feels half as current. Two weeks of silence from
#: a regular correspondent is noticeable but not yet a lapsed relationship.
RECENCY_HALF_LIFE_DAYS: Final[float] = 14.0

#: Messages needed before sender statistics are half-trusted.
CONFIDENCE_HALF_POINT: Final[float] = 5.0


class SenderPriorityCalculator(SignalCalculator):
    """Scores the standing of an individual human sender with this recipient."""

    name: ClassVar[str] = "sender_priority"
    polarity: ClassVar[SignalPolarity] = SignalPolarity.BOOST

    def contributions(self, context: SignalContext) -> Sequence[Contribution]:
        """Return the evidence about this sender.

        Returns an empty sequence when no individual sent the message, which
        lands the signal on neutral rather than on a fabricated score.
        """
        if context.sender_user_id is None:
            return ()

        stats = context.sender_stats
        elapsed = days_between(stats.last_seen, context.now)
        known = stats.has_history

        return (
            Contribution(
                name="familiarity",
                value=saturating(stats.total, FAMILIARITY_HALF_POINT),
                weight=0.20,
                high_reason=f"Frequent contact with this sender ({stats.total} prior messages).",
                low_reason=(
                    "Sender has not messaged this user before."
                    if not known
                    else "Little prior contact with this sender."
                ),
            ),
            Contribution(
                name="responsiveness",
                value=stats.reply_rate,
                weight=0.28,
                high_reason=f"User replies often to this sender ({stats.reply_rate:.0%} reply rate).",
                low_reason="User rarely replies to this sender.",
            ),
            Contribution(
                name="attention",
                value=stats.open_rate,
                weight=0.18,
                high_reason=f"User usually opens this sender's messages ({stats.open_rate:.0%}).",
                low_reason="User usually leaves this sender's messages unopened.",
            ),
            Contribution(
                name="recency",
                value=decay(elapsed, RECENCY_HALF_LIFE_DAYS),
                weight=0.14,
                high_reason="Recent conversation with this sender.",
                low_reason=(
                    "No recent contact with this sender."
                    if known
                    else "No prior conversation with this sender."
                ),
            ),
            Contribution(
                name="not_dismissed",
                value=1.0 - stats.dismiss_rate,
                weight=0.12,
                low_reason=(
                    f"User dismisses this sender's notifications "
                    f"({stats.dismiss_rate:.0%} dismissed)."
                ),
            ),
            Contribution(
                name="not_reported",
                value=1.0 - stats.report_rate,
                weight=0.08,
                low_reason="User has reported messages from this sender.",
            ),
        )

    def confidence(self, context: SignalContext) -> float:
        """Confidence grows with the number of prior messages from this sender."""
        if context.sender_user_id is None:
            return 0.0
        return evidence_confidence(context.sender_stats.total, CONFIDENCE_HALF_POINT)

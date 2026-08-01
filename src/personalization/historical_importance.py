"""How the recipient treated comparable messages in the past.

"Comparable" means same counterparty **and** same conversation type. That
pairing is what makes the signal a genuine predictor rather than a restatement
of the counterparty priority signals: it asks not "does this sender matter"
but "when this sender sent this kind of message before, did the user care".

Two trends are read separately, because they can disagree and the disagreement
is informative:

* **interaction trend** - is the user engaging with these messages more or less
  than they used to;
* **relationship trend** - is the counterparty's volume rising or falling.

A relationship where volume climbs while engagement falls is exactly the shape
of a sender becoming a nuisance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

from src.data.models import MessageHistory
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

__all__ = ["HistoricalImportanceCalculator"]

#: Comparable prior messages at which the pattern is half-established.
SIMILAR_VOLUME_HALF_POINT: Final[float] = 5.0

#: Minimum comparable messages before a trend is meaningful. Below this, one
#: message flipping would swing the whole trend.
MIN_RECORDS_FOR_TREND: Final[int] = 4

#: Comparable messages needed before the assessment is half-trusted.
CONFIDENCE_HALF_POINT: Final[float] = 4.0


class HistoricalImportanceCalculator(SignalCalculator):
    """Scores how much comparable messages have mattered to this user before."""

    name: ClassVar[str] = "historical_importance"
    polarity: ClassVar[SignalPolarity] = SignalPolarity.BOOST

    def contributions(self, context: SignalContext) -> Sequence[Contribution]:
        """Return the evidence from comparable past messages."""
        similar = self._similar_messages(context)
        if not similar:
            return (
                Contribution(
                    name="long_term_relevance",
                    value=0.0,
                    weight=1.0,
                    low_reason="No comparable messages in this user's history.",
                ),
            )

        opened, replied, dismissed = self._reaction_rates(context, similar)
        return (
            Contribution(
                name="similar_volume",
                value=saturating(len(similar), SIMILAR_VOLUME_HALF_POINT),
                weight=0.18,
                high_reason=(
                    f"{len(similar)} comparable messages already received from "
                    f"this source."
                ),
            ),
            Contribution(
                name="similar_attention",
                value=opened,
                weight=0.24,
                high_reason=f"Comparable messages are usually opened ({opened:.0%}).",
                low_reason=f"Comparable messages are usually ignored ({opened:.0%} opened).",
            ),
            Contribution(
                name="similar_response",
                value=replied,
                weight=0.24,
                high_reason=f"Comparable messages usually get a reply ({replied:.0%}).",
                low_reason="Comparable messages rarely get a reply.",
            ),
            Contribution(
                name="not_dismissed",
                value=1.0 - dismissed,
                weight=0.10,
                low_reason=(
                    f"Historical dismissals reduce importance "
                    f"({dismissed:.0%} of comparable messages dismissed)."
                ),
            ),
            Contribution(
                name="interaction_trend",
                value=self._interaction_trend(context, similar),
                weight=0.14,
                high_reason="User is engaging with these messages more than before.",
                low_reason="User is engaging with these messages less than before.",
            ),
            Contribution(
                name="relationship_trend",
                value=self._relationship_trend(similar),
                weight=0.10,
                high_reason="This source is becoming more active.",
                low_reason="This source is becoming less active.",
            ),
        )

    @staticmethod
    def _similar_messages(context: SignalContext) -> tuple[MessageHistory, ...]:
        """Return past messages from the same counterparty and conversation type.

        Records arrive oldest-first from the Phase 1 indexes, which the trend
        calculations rely on.
        """
        if context.sender_user_id is not None:
            history = context.repo.get_sender_history(context.sender_user_id)
        elif context.business_id is not None:
            history = context.repo.get_business_history(context.business_id)
        elif context.group_id is not None:
            history = context.repo.get_group_history(context.group_id)
        else:
            return ()

        conversation_type = context.features.conversation_type
        return tuple(
            record
            for record in history
            if record.user_id == context.user_id
            and record.conversation_type == conversation_type
        )

    @staticmethod
    def _reaction_rates(
        context: SignalContext, similar: Sequence[MessageHistory]
    ) -> tuple[float, float, float]:
        """Return open, reply and dismiss rates across ``similar``."""
        events = [
            event
            for record in similar
            if (event := context.repo.get_message_event(record.message_id)) is not None
        ]
        if not events:
            return 0.0, 0.0, 0.0
        total = len(events)
        return (
            ratio(sum(e.message_opened for e in events), total),
            ratio(sum(e.message_replied for e in events), total),
            ratio(sum(e.notification_dismissed for e in events), total),
        )

    def _interaction_trend(
        self, context: SignalContext, similar: Sequence[MessageHistory]
    ) -> float:
        """Compare engagement on recent comparable messages against earlier ones."""
        if len(similar) < MIN_RECORDS_FOR_TREND:
            return NEUTRAL
        midpoint = len(similar) // 2
        earlier = self._engagement(context, similar[:midpoint])
        recent = self._engagement(context, similar[midpoint:])
        return trend_score(recent, earlier)

    @staticmethod
    def _engagement(
        context: SignalContext, records: Sequence[MessageHistory]
    ) -> float:
        """Mean of open and reply rate across a slice of comparable messages."""
        events = [
            event
            for record in records
            if (event := context.repo.get_message_event(record.message_id)) is not None
        ]
        if not events:
            return 0.0
        opened = ratio(sum(e.message_opened for e in events), len(events))
        replied = ratio(sum(e.message_replied for e in events), len(events))
        return (opened + replied) / 2.0

    @staticmethod
    def _relationship_trend(similar: Sequence[MessageHistory]) -> float:
        """Compare message volume in the recent half of the span against the earlier.

        Measured over elapsed time rather than record count, so a genuine
        change in cadence is visible instead of being split evenly by
        construction.
        """
        if len(similar) < MIN_RECORDS_FOR_TREND:
            return NEUTRAL

        first, last = similar[0].created_at, similar[-1].created_at
        span = (last - first).total_seconds()
        if span <= 0:
            return NEUTRAL

        midpoint = first + (last - first) / 2
        recent = sum(1 for record in similar if record.created_at >= midpoint)
        earlier = len(similar) - recent
        return trend_score(
            ratio(recent, len(similar)), ratio(earlier, len(similar))
        )

    def confidence(self, context: SignalContext) -> float:
        """Confidence grows with the number of comparable prior messages."""
        return evidence_confidence(
            len(self._similar_messages(context)), CONFIDENCE_HALF_POINT
        )

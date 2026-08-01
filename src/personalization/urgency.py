"""How time-critical the message appears.

Reuses Phase 2 rather than re-reading the text: the classifier already decided
whether this is an ``urgent`` message and the feature record already holds the
urgency keyword matches. What this calculator adds is the *contextual*
amplification Phase 2 deliberately kept out of classification - who is
speaking and whether the message names the recipient.

The distinction matters. "The lift is out until 6pm" from a building admin who
names your flat is more time-critical than the same words from a stranger, but
it is the same category of message, so the difference belongs here rather than
in the classifier.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

from src.classifier.enums import KeywordCategory, MessageType
from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.normalization import (
    NEUTRAL,
    Contribution,
    evidence_confidence,
    saturating,
)
from src.personalization.signal_models import SignalPolarity

__all__ = ["UrgencyCalculator"]

#: Urgency keywords at which the lexical case is half-made.
KEYWORD_HALF_POINT: Final[float] = 2.0

#: Evidence items needed before the urgency assessment is half-trusted.
CONFIDENCE_HALF_POINT: Final[float] = 2.0

#: Categories that are inherently time-bound even when no urgency word appears:
#: an appointment or a payment deadline carries its own clock.
_TIME_BOUND_TYPES: Final[frozenset[MessageType]] = frozenset(
    {MessageType.EVENT, MessageType.PAYMENT}
)


class UrgencyCalculator(SignalCalculator):
    """Scores how time-critical the message appears for this recipient."""

    name: ClassVar[str] = "urgency_modifier"
    polarity: ClassVar[SignalPolarity] = SignalPolarity.BOOST

    #: Most messages are not urgent, and that is unremarkable rather than an
    #: argument for suppressing them, so absence lands on neutral.
    one_sided: ClassVar[bool] = True

    def contributions(self, context: SignalContext) -> Sequence[Contribution]:
        """Return the urgency evidence, reusing the Phase 2 verdict and keywords."""
        features = context.features
        verdict = context.classification.message_type
        urgent_keywords = features.keywords.count(KeywordCategory.URGENT)

        return (
            Contribution(
                name="classified_urgent",
                value=self._verdict_value(verdict),
                weight=0.40,
                high_reason="Classified as time-sensitive.",
                low_reason=None,
            ),
            Contribution(
                name="urgency_language",
                value=saturating(urgent_keywords, KEYWORD_HALF_POINT),
                weight=0.24,
                high_reason=(
                    "Message uses time-pressure language "
                    f"({', '.join(features.keywords.words(KeywordCategory.URGENT))})."
                ),
                low_reason=None,
            ),
            Contribution(
                name="named_recipient",
                value=1.0 if self._mentions_recipient(context) else 0.0,
                weight=0.20,
                high_reason="Message names the recipient directly.",
                low_reason=None,
            ),
            Contribution(
                name="authority",
                value=1.0 if features.context.sender_is_admin else 0.0,
                weight=0.16,
                high_reason="Raised by a group admin.",
                low_reason=None,
            ),
        )

    @staticmethod
    def _verdict_value(verdict: MessageType) -> float:
        """Map the Phase 2 category onto an urgency level.

        ``urgent`` is unambiguous; events and payments carry an implicit clock;
        everything else is simply not time-bound, which is zero rather than
        neutral because the classifier actively considered and rejected it.
        """
        if verdict is MessageType.URGENT:
            return 1.0
        if verdict in _TIME_BOUND_TYPES:
            return NEUTRAL
        return 0.0

    @staticmethod
    def _mentions_recipient(context: SignalContext) -> bool:
        """Whether the body names the recipient with an ``@user_id`` mention."""
        if context.features.text.is_empty:
            return False
        mention = f"@{context.user_id}".casefold()
        return mention in context.features.text.normalized_text

    def confidence(self, context: SignalContext) -> float:
        """Confidence grows with how many independent urgency cues were found.

        With no cues at all the signal sits at neutral and says nothing, so the
        confidence attached to it cannot change any outcome. It is reported as
        the classifier's own confidence, which is the honest description of how
        firmly "not urgent" was established.
        """
        cues = sum(
            (
                context.classification.message_type is MessageType.URGENT,
                context.features.keywords.has(KeywordCategory.URGENT),
                self._mentions_recipient(context),
                context.features.context.sender_is_admin,
            )
        )
        if cues == 0:
            return context.classification.confidence
        return evidence_confidence(cues, CONFIDENCE_HALF_POINT)

"""How unsafe the message looks, built entirely from Phase 2's verdict.

A **suppressing** signal: a rising score argues for holding the message back.

Nothing here re-classifies. The Phase 2 classifier already weighed keywords,
sender standing and context to reach a verdict, and re-deriving any of that
would risk the two disagreeing. This calculator reads only
:class:`~src.classifier.message_classifier.MessageClassification` plus two
plain facts from the feature record.

The classifier's per-category scores are turned into *shares* of the total
evidence, which is what makes the reuse principled: a scam score of 5.8 means
nothing on its own, but "scam accounts for 78% of everything the classifier
found" is directly interpretable and already normalised.

Confidence is taken straight from the classifier rather than recomputed - it
is a judgement about the same verdict, so inventing a second one would be
both duplicated and liable to drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

from src.classifier.enums import MessageType
from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.normalization import Contribution, explain, saturating
from src.personalization.signal_models import RoutingSignal, SignalPolarity

__all__ = ["RiskCalculator"]

#: Forward count at which forwarding is half-suspicious. Chain messages in the
#: dataset reach counts of eleven and above, while an ordinary re-share sits
#: in the low single digits.
FORWARD_HALF_POINT: Final[float] = 8.0

#: Share of classifier evidence above which a category is called dominant.
DOMINANT_SHARE: Final[float] = 0.5


class RiskCalculator(SignalCalculator):
    """Scores how unsafe the message looks, reusing the Phase 2 verdict."""

    name: ClassVar[str] = "risk_modifier"
    polarity: ClassVar[SignalPolarity] = SignalPolarity.SUPPRESS

    #: A clean message is not an argument for interrupting someone, so the
    #: absence of risk lands on neutral rather than pushing priority upward.
    one_sided: ClassVar[bool] = True

    def contributions(self, context: SignalContext) -> Sequence[Contribution]:
        """Return the risk evidence, all of it derived from Phase 2 outputs."""
        shares = self._evidence_shares(context)
        verdict = context.classification.message_type

        return (
            # The verdict itself is stated once, by calculate(); these read on
            # the underlying characteristics so the two never restate each other.
            Contribution(
                name="scam",
                value=shares[MessageType.SCAM],
                weight=0.34,
                high_reason="Message carries scam characteristics.",
            ),
            Contribution(
                name="spam",
                value=shares[MessageType.SPAM],
                weight=0.18,
                high_reason="Message carries spam characteristics.",
            ),
            Contribution(
                name="promotion",
                value=self._promotion_value(context, shares[MessageType.PROMOTION]),
                weight=0.14,
                high_reason=(
                    "Unwanted promotional content: the user opted out of "
                    "promotions from this business."
                    if context.features.history.opted_out_of_promotions
                    else "Promotional content."
                ),
            ),
            Contribution(
                name="forwarding",
                value=saturating(
                    context.features.forwarded_count, FORWARD_HALF_POINT
                ),
                weight=0.20,
                high_reason=(
                    f"Widely forwarded chain message "
                    f"({context.features.forwarded_count} forwards)."
                ),
            ),
            Contribution(
                name="impersonation",
                value=1.0 if context.features.context.has_domain_mismatch else 0.0,
                weight=0.14,
                high_reason="Sender's domain does not match the brand it claims.",
            ),
        )

    @staticmethod
    def _evidence_shares(context: SignalContext) -> dict[MessageType, float]:
        """Return each category's share of the classifier's total evidence.

        Normalising by the total is what lets raw rule weights be reused
        without inventing a scale for them. Every share is in ``[0, 1]``.
        """
        scores = context.classification.scores
        total = sum(scores.values())
        if total <= 0:
            return dict.fromkeys(MessageType, 0.0)
        return {
            category: scores.get(category, 0.0) / total for category in MessageType
        }

    @staticmethod
    def _promotion_value(context: SignalContext, promotion_share: float) -> float:
        """Score promotional content, escalated when the user opted out.

        A promotion is only mildly unwanted in general, but one sent after an
        explicit opt-out is a direct breach of the user's instruction.
        """
        if context.features.history.opted_out_of_promotions:
            return max(promotion_share, DOMINANT_SHARE)
        return promotion_share

    def confidence(self, context: SignalContext) -> float:
        """Reuse the Phase 2 classifier's own confidence in its verdict."""
        return context.classification.confidence

    def calculate(self, context: SignalContext) -> RoutingSignal:
        """Produce the risk signal.

        Overrides the base assembly for one reason: when Phase 2 has already
        committed to ``scam`` or ``spam``, the verdict itself is stronger
        evidence than any blend of its parts, so the score is floored at the
        classifier's own confidence in that verdict.
        """
        contributions = list(self.contributions(context))
        score = self.score(contributions)
        reasons = list(explain(contributions, limit=self.max_reasons))

        verdict = context.classification.message_type
        if verdict.is_risk:
            score = max(score, context.classification.confidence)
            reasons.insert(0, f"Classified as {verdict.value}.")

        return RoutingSignal(
            name=self.name,
            score=score,
            confidence=self.confidence(context),
            polarity=self.polarity,
            reasons=tuple(reasons[: self.max_reasons]),
        )

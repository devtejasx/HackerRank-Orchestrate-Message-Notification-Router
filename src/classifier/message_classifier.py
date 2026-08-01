"""The Phase 2 classification entry point.

:class:`MessageClassifier` aggregates rule signals into a per-category score,
selects one winner and explains the choice. It is deterministic and pure: the
same features always produce the same classification.

No LLMs, machine learning, embeddings, OCR or speech recognition are involved,
by design.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from src.classifier.confidence import (
    DEFAULT_CONFIDENCE,
    ConfidenceModel,
    score_confidence,
)
from src.classifier.enums import MessageType
from src.classifier.rules import DEFAULT_WEIGHTS, Signal, Weights, collect_signals
from src.features.feature_models import MessageFeatures

__all__ = ["MessageClassification", "MessageClassifier"]

#: Tie-break order when two categories score identically. Safety first: a
#: message that scores equally as promotion and scam is treated as a scam.
#: Also the deterministic ordering that makes results reproducible.
_PRIORITY: Final[tuple[MessageType, ...]] = (
    MessageType.SCAM,
    MessageType.SPAM,
    MessageType.URGENT,
    MessageType.PAYMENT,
    MessageType.EVENT,
    MessageType.PROMOTION,
    MessageType.BUSINESS_UPDATE,
    MessageType.FORWARD,
    MessageType.GREETING,
    MessageType.PERSONAL,
    MessageType.UNKNOWN,
)

_PRIORITY_RANK: Final[Mapping[MessageType, int]] = {
    message_type: rank for rank, message_type in enumerate(_PRIORITY)
}

#: How many contributing reasons are woven into the explanation.
_MAX_REASONS: Final[int] = 3


@dataclass(frozen=True, slots=True)
class MessageClassification:
    """The classification verdict for one message.

    Attributes:
        message_id: The message this verdict describes.
        message_type: Exactly one category from :class:`MessageType`.
        confidence: How sure the classifier is, in ``[0, 1]``.
        matched_keywords: Every keyword that matched, across all categories.
        classification_reason: Human-readable explanation, assembled from the
            evidence that actually drove the decision.
        scores: Total weight per category. Retained so Phase 4 routing can see
            the runner-up and the margin rather than only the winner.
        signals: The raw evidence, for debugging and for Phase 5 explanation.
    """

    message_id: str
    message_type: MessageType
    confidence: float
    matched_keywords: tuple[str, ...]
    classification_reason: str
    scores: Mapping[MessageType, float] = field(default_factory=dict)
    signals: tuple[Signal, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the score mapping so the record is genuinely immutable."""
        object.__setattr__(self, "scores", dict(self.scores))

    @property
    def is_risk(self) -> bool:
        """Whether the verdict is ``scam`` or ``spam``."""
        return self.message_type.is_risk

    @property
    def runner_up(self) -> MessageType | None:
        """The next-best category, or ``None`` when nothing else scored."""
        rivals = {
            category: score
            for category, score in self.scores.items()
            if category is not self.message_type and score > 0.0
        }
        if not rivals:
            return None
        return max(rivals, key=lambda category: (rivals[category], -_PRIORITY_RANK[category]))

    @property
    def margin(self) -> float:
        """Gap between the winning score and the runner-up's."""
        winning = self.scores.get(self.message_type, 0.0)
        runner_up = self.runner_up
        if runner_up is None:
            return winning
        return winning - self.scores[runner_up]

    def to_dict(self) -> dict[str, object]:
        """Return a flat, JSON-friendly view for logging and debugging."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "confidence": self.confidence,
            "matched_keywords": list(self.matched_keywords),
            "classification_reason": self.classification_reason,
            "scores": {
                category.value: round(score, 3)
                for category, score in sorted(
                    self.scores.items(), key=lambda item: -item[1]
                )
            },
        }


class MessageClassifier:
    """Assigns exactly one :class:`MessageType` to a message.

    Args:
        weights: Rule weighting. Override to retune without editing rules.
        confidence_model: Confidence tuning.

    Example:
        >>> classifier = MessageClassifier()                    # doctest: +SKIP
        >>> classifier.classify(features).message_type          # doctest: +SKIP
        <MessageType.SCAM: 'scam'>
    """

    def __init__(
        self,
        weights: Weights = DEFAULT_WEIGHTS,
        confidence_model: ConfidenceModel = DEFAULT_CONFIDENCE,
    ) -> None:
        self._weights = weights
        self._confidence_model = confidence_model

    @property
    def weights(self) -> Weights:
        """The rule weighting in use."""
        return self._weights

    def classify(self, features: MessageFeatures) -> MessageClassification:
        """Classify one message.

        Args:
            features: Extracted features, from
                :class:`~src.features.extractor.FeatureExtractor`.

        Returns:
            A verdict naming exactly one category, with its confidence, the
            keywords that matched and an explanation derived from the evidence.
        """
        signals = collect_signals(features, self._weights)
        scores = _aggregate(signals)
        winner = self._select(scores)
        confidence = score_confidence(
            features, scores, winner, self._confidence_model
        )

        return MessageClassification(
            message_id=features.message_id,
            message_type=winner,
            confidence=confidence,
            matched_keywords=features.matched_keywords,
            classification_reason=_explain(winner, signals, features),
            scores=scores,
            signals=signals,
        )

    def classify_many(
        self, features: tuple[MessageFeatures, ...]
    ) -> tuple[MessageClassification, ...]:
        """Classify many messages, preserving input order."""
        return tuple(self.classify(item) for item in features)

    def _select(self, scores: Mapping[MessageType, float]) -> MessageType:
        """Pick the winning category.

        Falls back to ``unknown`` when the best score is too weak to commit,
        which is a real answer rather than a guess. Ties break by
        :data:`_PRIORITY`, so results are deterministic and risk-leaning.
        """
        if not scores:
            return MessageType.UNKNOWN

        best = max(
            scores.items(),
            key=lambda item: (item[1], -_PRIORITY_RANK[item[0]]),
        )
        category, score = best
        if score < self._weights.minimum_commit_score:
            return MessageType.UNKNOWN
        return category


def _aggregate(signals: tuple[Signal, ...]) -> dict[MessageType, float]:
    """Sum signal weights per category."""
    scores: dict[MessageType, float] = {}
    for signal in signals:
        scores[signal.message_type] = scores.get(signal.message_type, 0.0) + signal.weight
    return scores


def _explain(
    winner: MessageType, signals: tuple[Signal, ...], features: MessageFeatures
) -> str:
    """Build the explanation from the evidence that produced ``winner``.

    Only signals supporting the winning category are used, strongest first, so
    the reason always matches the decision instead of being written separately
    and drifting out of sync.
    """
    supporting = sorted(
        (signal for signal in signals if signal.message_type is winner),
        key=lambda signal: -signal.weight,
    )
    if not supporting:
        return _explain_unknown(features)

    reasons = [signal.reason for signal in supporting[:_MAX_REASONS]]
    joined = "; ".join(reasons)
    return f"Classified as {winner.value} because it {joined}."


def _explain_unknown(features: MessageFeatures) -> str:
    """Explain why no category could be committed to."""
    if features.text.is_empty and features.has_media:
        return (
            f"Classified as unknown: the message is a {features.media_type} note with "
            "no text, and its sender context is not distinctive enough to categorise."
        )
    if features.keywords.total_matches == 0:
        return (
            "Classified as unknown: no category keywords matched and the sender "
            "context provides no distinctive signal."
        )
    return (
        "Classified as unknown: the available signals were too weak or too evenly "
        "split to commit to a category."
    )

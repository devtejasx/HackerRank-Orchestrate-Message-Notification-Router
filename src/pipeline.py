"""The Phase 2 pipeline: message in, features and classification out.

This is the seam later phases attach to. Phase 3 (personalisation), Phase 4
(routing) and Phase 5 (evidence retrieval) all consume
:class:`MessageAnalysis` and none of them needs to know how features were
extracted or how the verdict was reached.

    Incoming Message -> Repository lookup -> Feature extraction ->
    Keyword detection -> Context -> History -> Classification ->
    Confidence -> MessageAnalysis

Nothing here decides ``notify``, ``digest`` or ``mute``, and nothing writes
``output.csv``. Those are later phases by design.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from src import config
from src.classifier.confidence import ConfidenceModel, DEFAULT_CONFIDENCE
from src.classifier.keyword_rules import KeywordMatcher
from src.classifier.message_classifier import MessageClassification, MessageClassifier
from src.classifier.rules import DEFAULT_WEIGHTS, Weights
from src.data.models import Message
from src.data.repository import DataRepository
from src.features.extractor import FeatureExtractor
from src.features.feature_models import MessageFeatures

__all__ = ["MessageAnalysis", "MessagePipeline"]

_LOGGER = config.get_logger("pipeline")


@dataclass(frozen=True, slots=True)
class MessageAnalysis:
    """The complete Phase 2 result for one message.

    Attributes:
        features: Everything extracted about the message.
        classification: The category verdict, with confidence and reasoning.
    """

    features: MessageFeatures
    classification: MessageClassification

    @property
    def message_id(self) -> str:
        """Identifier of the analysed message."""
        return self.features.message_id

    @property
    def user_id(self) -> str:
        """Recipient of the analysed message."""
        return self.features.user_id

    def to_dict(self) -> dict[str, object]:
        """Return a flat, JSON-friendly view of both halves."""
        return {
            "features": self.features.to_dict(),
            "classification": self.classification.to_dict(),
        }


class MessagePipeline:
    """Runs feature extraction and classification over messages.

    Owns the extractor and classifier so their caches and compiled patterns
    are built once. Construct one pipeline and reuse it.

    Args:
        repo: A loaded repository.
        weights: Rule weighting override.
        confidence_model: Confidence tuning override.
        matcher: Keyword matcher override, for custom dictionaries.

    Example:
        >>> pipeline = MessagePipeline.load()               # doctest: +SKIP
        >>> analysis = pipeline.analyse_all()[0]            # doctest: +SKIP
        >>> analysis.classification.message_type            # doctest: +SKIP
        <MessageType.PROMOTION: 'promotion'>
    """

    def __init__(
        self,
        repo: DataRepository,
        weights: Weights = DEFAULT_WEIGHTS,
        confidence_model: ConfidenceModel = DEFAULT_CONFIDENCE,
        matcher: KeywordMatcher | None = None,
    ) -> None:
        self._repo = repo
        self._extractor = FeatureExtractor(repo, matcher=matcher)
        self._classifier = MessageClassifier(weights, confidence_model)

    @classmethod
    def load(cls, dataset_dir: Path | None = None, **kwargs: object) -> Self:
        """Load the dataset and build a ready-to-use pipeline.

        Args:
            dataset_dir: Dataset directory. Defaults to
                :data:`src.config.DATASET_DIR`.
            **kwargs: Forwarded to :class:`MessagePipeline`.

        Returns:
            A pipeline over a loaded, validated and indexed repository.
        """
        return cls(DataRepository.load(dataset_dir), **kwargs)  # type: ignore[arg-type]

    @property
    def repository(self) -> DataRepository:
        """The underlying repository, for callers that need raw records."""
        return self._repo

    @property
    def extractor(self) -> FeatureExtractor:
        """The feature extractor in use."""
        return self._extractor

    @property
    def classifier(self) -> MessageClassifier:
        """The classifier in use."""
        return self._classifier

    def analyse(self, message: Message) -> MessageAnalysis:
        """Extract features and classify one message.

        Args:
            message: The incoming message.

        Returns:
            Both halves of the Phase 2 result.
        """
        features = self._extractor.extract(message)
        return MessageAnalysis(features, self._classifier.classify(features))

    def analyse_many(self, messages: Iterable[Message]) -> tuple[MessageAnalysis, ...]:
        """Analyse many messages, preserving input order."""
        results = tuple(self.analyse(message) for message in messages)
        _LOGGER.info("Analysed %d message(s)", len(results))
        return results

    def analyse_all(self) -> tuple[MessageAnalysis, ...]:
        """Analyse every incoming message in the dataset."""
        return self.analyse_many(self._repo.get_messages())

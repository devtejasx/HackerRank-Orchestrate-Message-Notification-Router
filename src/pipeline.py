"""The analysis pipeline: message in, features, classification and signals out.

This is the seam later phases attach to. Phase 4 (routing) and Phase 5
(evidence retrieval) consume :class:`MessageAnalysis` and neither needs to
know how features were extracted, how the verdict was reached, or how the
routing signals were scored.

    Incoming Message -> Repository lookup -> Feature extraction ->
    Keyword detection -> Context -> History -> Classification ->
    Confidence -> Routing signals -> MessageAnalysis

Personalisation can be switched off with ``personalize=False``, in which case
:attr:`MessageAnalysis.routing` is ``None`` and the pipeline behaves exactly as
it did before Phase 3.

Nothing here decides ``notify``, ``digest`` or ``mute``, and nothing writes
``output.csv``. Those are later phases by design.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from src import config
from src.classifier.confidence import DEFAULT_CONFIDENCE, ConfidenceModel
from src.classifier.keyword_rules import KeywordMatcher
from src.classifier.message_classifier import MessageClassification, MessageClassifier
from src.classifier.rules import DEFAULT_WEIGHTS, Weights
from src.data.models import Message
from src.data.repository import DataRepository
from src.features.extractor import FeatureExtractor
from src.features.feature_models import MessageFeatures
from src.media.understanding import MediaUnderstanding
from src.personalization.engine import PersonalizationEngine
from src.personalization.signal_models import RoutingSignals

__all__ = ["MessageAnalysis", "MessagePipeline"]

_LOGGER = config.get_logger("pipeline")


@dataclass(frozen=True, slots=True)
class MessageAnalysis:
    """The complete analysis of one message, across every phase run so far.

    Attributes:
        features: Everything extracted about the message (Phase 2).
        classification: The category verdict, with confidence and reasoning
            (Phase 2).
        routing: The personalised routing signals (Phase 3). ``None`` when
            personalisation was switched off, which keeps every Phase 2 caller
            working unchanged.
    """

    features: MessageFeatures
    classification: MessageClassification
    routing: RoutingSignals | None = None

    @property
    def message_id(self) -> str:
        """Identifier of the analysed message."""
        return self.features.message_id

    @property
    def user_id(self) -> str:
        """Recipient of the analysed message."""
        return self.features.user_id

    def to_dict(self) -> dict[str, object]:
        """Return a flat, JSON-friendly view of every part."""
        payload: dict[str, object] = {
            "features": self.features.to_dict(),
            "classification": self.classification.to_dict(),
        }
        if self.routing is not None:
            payload["routing"] = self.routing.to_dict()
        return payload


class MessagePipeline:
    """Runs feature extraction and classification over messages.

    Owns the extractor and classifier so their caches and compiled patterns
    are built once. Construct one pipeline and reuse it.

    Args:
        repo: A loaded repository.
        weights: Rule weighting override.
        confidence_model: Confidence tuning override.
        matcher: Keyword matcher override, for custom dictionaries.
        understanding: OCR / speech-to-text provider. Defaults to the null
            provider, which recovers nothing; see :mod:`src.media`.

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
        personalize: bool = True,
        engine: PersonalizationEngine | None = None,
        understanding: MediaUnderstanding | None = None,
    ) -> None:
        self._repo = repo
        self._extractor = FeatureExtractor(repo, matcher=matcher, understanding=understanding)
        self._classifier = MessageClassifier(weights, confidence_model)
        self._engine: PersonalizationEngine | None = None
        if personalize:
            self._engine = engine if engine is not None else PersonalizationEngine(repo)

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

    @property
    def engine(self) -> PersonalizationEngine | None:
        """The personalisation engine, or ``None`` when it is switched off."""
        return self._engine

    def analyse(self, message: Message) -> MessageAnalysis:
        """Run every enabled phase over one message.

        Args:
            message: The incoming message.

        Returns:
            Features and classification, plus routing signals when
            personalisation is enabled.
        """
        features = self._extractor.extract(message)
        classification = self._classifier.classify(features)
        routing = (
            self._engine.compute(features, classification)
            if self._engine is not None
            else None
        )
        return MessageAnalysis(features, classification, routing)

    def analyse_many(self, messages: Iterable[Message]) -> tuple[MessageAnalysis, ...]:
        """Analyse many messages, preserving input order."""
        results = tuple(self.analyse(message) for message in messages)
        _LOGGER.info("Analysed %d message(s)", len(results))
        return results

    def analyse_all(self) -> tuple[MessageAnalysis, ...]:
        """Analyse every incoming message in the dataset."""
        return self.analyse_many(self._repo.get_messages())

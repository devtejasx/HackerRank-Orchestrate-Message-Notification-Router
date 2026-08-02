"""The Phase 2 feature extraction entry point.

:class:`FeatureExtractor` composes the four feature blocks into a single
:class:`~src.features.feature_models.MessageFeatures` record. It owns the
expensive-to-build collaborators - the compiled keyword matcher and the
memoising historical extractor - so construct it once and reuse it across the
whole dataset.
"""

from __future__ import annotations

from collections.abc import Iterable

from src import config
from src.classifier.keyword_rules import KeywordMatcher
from src.data.models import MessageRecord
from src.data.repository import DataRepository
from src.features.context_features import extract_context_features
from src.features.feature_models import MessageFeatures
from src.features.historical_features import (
    EngagementWeights,
    HistoricalFeatureExtractor,
)
from src.features.keyword_features import extract_keyword_features
from src.features.text_features import extract_text_features

__all__ = ["FeatureExtractor"]

_LOGGER = config.get_logger("features")


class FeatureExtractor:
    """Turns messages into feature records.

    Args:
        repo: Loaded repository. Every context and history lookup goes through
            it; no CSV is read here.
        matcher: Keyword matcher to reuse. A default one is built when
            omitted. Pass a custom matcher to swap the dictionaries.
        engagement_weights: Weighting for engagement scores.

    Example:
        >>> extractor = FeatureExtractor(repo)              # doctest: +SKIP
        >>> features = extractor.extract(message)           # doctest: +SKIP
        >>> features.keywords.categories                    # doctest: +SKIP
        (<KeywordCategory.SCAM: 'scam'>,)
    """

    def __init__(
        self,
        repo: DataRepository,
        matcher: KeywordMatcher | None = None,
        engagement_weights: EngagementWeights | None = None,
    ) -> None:
        self._repo = repo
        self._matcher = matcher if matcher is not None else KeywordMatcher()
        self._historical = HistoricalFeatureExtractor(
            repo,
            engagement_weights if engagement_weights is not None else EngagementWeights(),
        )

    @property
    def matcher(self) -> KeywordMatcher:
        """The keyword matcher in use, exposed for inspection and testing."""
        return self._matcher

    def extract(self, message: MessageRecord) -> MessageFeatures:
        """Build the complete feature record for one message.

        Accepts any record carrying the shared message envelope, so historical
        messages can be put through the same extraction as incoming ones. Phase
        4 relies on that to classify history when searching for evidence.

        Args:
            message: The message to analyse, incoming or historical.

        Returns:
            An immutable, self-contained feature record. No further repository
            access is needed to read any field.
        """
        return MessageFeatures(
            message_id=message.message_id,
            user_id=message.user_id,
            sender_user_id=message.sender_user_id,
            group_id=message.group_id,
            business_id=message.business_id,
            created_at=message.created_at,
            text=extract_text_features(message.message_text),
            context=extract_context_features(message, self._repo),
            history=self._historical.extract(message),
            keywords=extract_keyword_features(message.message_text, self._matcher),
        )

    def extract_many(
        self, messages: Iterable[MessageRecord]
    ) -> tuple[MessageFeatures, ...]:
        """Build feature records for many messages, reusing all caches.

        Args:
            messages: Messages to analyse, in any order.

        Returns:
            Feature records in the same order as the input.
        """
        features = tuple(self.extract(message) for message in messages)
        _LOGGER.info("Extracted features for %d message(s)", len(features))
        return features

    def extract_all(self) -> tuple[MessageFeatures, ...]:
        """Build feature records for every incoming message in the dataset."""
        return self.extract_many(self._repo.get_messages())

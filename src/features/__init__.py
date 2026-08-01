"""Phase 2 feature extraction.

Turns a :class:`~src.data.models.Message` plus repository context into a
:class:`~src.features.feature_models.MessageFeatures` record.

    from src.features import FeatureExtractor

    features = FeatureExtractor(repo).extract(message)
"""

from src.features.context_features import extract_context_features
from src.features.extractor import FeatureExtractor
from src.features.feature_models import (
    ContextFeatures,
    HistoricalFeatures,
    KeywordFeatures,
    MessageFeatures,
    TextFeatures,
)
from src.features.historical_features import (
    EngagementWeights,
    HistoricalFeatureExtractor,
)
from src.features.keyword_features import extract_keyword_features
from src.features.text_features import extract_text_features

__all__ = [
    "ContextFeatures",
    "EngagementWeights",
    "FeatureExtractor",
    "HistoricalFeatureExtractor",
    "HistoricalFeatures",
    "KeywordFeatures",
    "MessageFeatures",
    "TextFeatures",
    "extract_context_features",
    "extract_keyword_features",
    "extract_text_features",
]

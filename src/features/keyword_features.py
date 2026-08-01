"""Keyword feature extraction.

A thin adapter between :class:`~src.classifier.keyword_rules.KeywordMatcher`
and the feature record, so the extractor pipeline stays uniform across blocks.
"""

from __future__ import annotations

from src.classifier.keyword_rules import KeywordMatcher
from src.features.feature_models import KeywordFeatures

__all__ = ["extract_keyword_features"]


def extract_keyword_features(
    message_text: object, matcher: KeywordMatcher
) -> KeywordFeatures:
    """Match keyword dictionaries against a message body.

    Args:
        message_text: Raw text, possibly ``None`` or empty.
        matcher: A pre-compiled matcher. Reuse one instance across messages;
            constructing a matcher recompiles every pattern.

    Returns:
        The matched keywords grouped by category. Empty when the message has
        no text, which is the normal case for voice notes.
    """
    return KeywordFeatures(matches=matcher.match_by_category(message_text))

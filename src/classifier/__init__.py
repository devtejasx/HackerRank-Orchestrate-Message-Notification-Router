"""Phase 2 classification: keyword matching, rule scoring and confidence.

Rule-based and deterministic by design - no LLMs, machine learning,
embeddings, OCR or speech recognition.

    from src.classifier import MessageClassifier

    classification = MessageClassifier().classify(features)
"""

from src.classifier.confidence import (
    DEFAULT_CONFIDENCE,
    ConfidenceModel,
    score_confidence,
)
from src.classifier.enums import KeywordCategory, MessageType
from src.classifier.keyword_rules import (
    DEFAULT_KEYWORDS,
    KeywordMatch,
    KeywordMatcher,
    default_matcher,
)
from src.classifier.message_classifier import MessageClassification, MessageClassifier
from src.classifier.rules import DEFAULT_WEIGHTS, Signal, Weights, collect_signals

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_KEYWORDS",
    "DEFAULT_WEIGHTS",
    "ConfidenceModel",
    "KeywordCategory",
    "KeywordMatch",
    "KeywordMatcher",
    "MessageClassification",
    "MessageClassifier",
    "MessageType",
    "Signal",
    "Weights",
    "collect_signals",
    "default_matcher",
    "score_confidence",
]

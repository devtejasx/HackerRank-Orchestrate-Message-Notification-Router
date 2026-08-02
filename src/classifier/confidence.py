"""Confidence scoring for a classification decision.

Confidence answers "how sure are we that the winning category is right", which
is a different question from "how strong is the evidence". A message with
overwhelming evidence for two categories at once is *not* a confident call.
The model therefore reads three things:

* **margin** - how far the winner sits above the runner-up. This dominates,
  because a clear winner is what confidence actually means.
* **evidence** - the winner's absolute score, so a decision resting on one
  weak signal is never highly confident.
* **corroboration** - independent context that agrees with the verdict
  (multiple keywords, a verified sender, real history, an unambiguous scam
  pattern).

Ambiguity penalties apply when there is little to go on: no text, no keywords,
or no history.

Both margin and evidence are squashed with ``tanh`` so the result saturates
smoothly and stays inside the configured band without clipping artefacts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from src.classifier.enums import KeywordCategory, MessageType
from src.utils.helpers import clamp

if TYPE_CHECKING:
    # Annotation-only. Importing this at runtime closes a cycle - features
    # imports classifier.enums, which initialises the classifier package, which
    # imports this module - and makes `import src.media` fail depending on
    # which package the process happens to touch first. Annotations are strings
    # here (`from __future__ import annotations`), so nothing needs it at
    # runtime.
    from src.features.feature_models import MessageFeatures

__all__ = ["ConfidenceModel", "DEFAULT_CONFIDENCE", "score_confidence"]


@dataclass(frozen=True, slots=True)
class ConfidenceModel:
    """Tunable parameters for confidence scoring.

    Attributes:
        base: Starting point before any evidence is considered.
        margin_weight: Maximum contribution from a decisive win.
        evidence_weight: Maximum contribution from absolute evidence strength.
        corroboration_weight: Maximum contribution from agreeing context.
        margin_scale: Margin, in score units, at which the margin component is
            roughly three-quarters saturated.
        evidence_scale: Winning score at which the evidence component is
            roughly three-quarters saturated.
        no_text_penalty: Applied when the message has no readable text.
        no_keyword_penalty: Applied when no keyword matched at all.
        no_history_penalty: Applied when the user has no interaction history.
        floor: Lowest confidence ever returned.
        ceiling: Highest confidence ever returned. Below 1.0 because a
            rule-based classifier should never claim certainty.
    """

    base: float = 0.40
    margin_weight: float = 0.30
    evidence_weight: float = 0.16
    corroboration_weight: float = 0.14

    margin_scale: float = 2.20
    evidence_scale: float = 4.00

    no_text_penalty: float = 0.10
    no_keyword_penalty: float = 0.08
    no_history_penalty: float = 0.04

    floor: float = 0.25
    ceiling: float = 0.95


DEFAULT_CONFIDENCE: Final[ConfidenceModel] = ConfidenceModel()

#: Corroboration is expressed as a fraction of :attr:`corroboration_weight`.
#: The fractions below sum to 1.0 so a fully corroborated call earns the whole
#: weight and no more.
_MULTI_KEYWORD_SHARE: Final[float] = 0.30
_TRUSTED_SENDER_SHARE: Final[float] = 0.25
_HISTORY_SHARE: Final[float] = 0.20
_UNAMBIGUOUS_RISK_SHARE: Final[float] = 0.25

#: Distinct keyword matches for the winning category that count as "multiple".
_MULTI_KEYWORD_THRESHOLD: Final[int] = 2

#: Historical interactions above which history counts as substantive.
_SUBSTANTIVE_HISTORY: Final[int] = 5


def score_confidence(
    features: MessageFeatures,
    scores: Mapping[MessageType, float],
    winner: MessageType,
    model: ConfidenceModel = DEFAULT_CONFIDENCE,
) -> float:
    """Score how confident the classifier is in ``winner``.

    Args:
        features: The features the decision was made from.
        scores: Total weight accumulated per category.
        winner: The category selected.
        model: Tuning to apply.

    Returns:
        A value in ``[model.floor, model.ceiling]``.
    """
    if winner is MessageType.UNKNOWN:
        return _unknown_confidence(features, model)

    winning_score = scores.get(winner, 0.0)
    runner_up = _runner_up_score(scores, winner)
    margin = max(0.0, winning_score - runner_up)

    confidence = model.base
    confidence += model.margin_weight * math.tanh(margin / model.margin_scale)
    confidence += model.evidence_weight * math.tanh(winning_score / model.evidence_scale)
    confidence += model.corroboration_weight * _corroboration(features, winner)
    confidence -= _ambiguity_penalty(features, model)

    return round(clamp(confidence, model.floor, model.ceiling), 2)


def _runner_up_score(
    scores: Mapping[MessageType, float], winner: MessageType
) -> float:
    """Return the highest score among the categories that did not win."""
    rivals = [score for category, score in scores.items() if category is not winner]
    return max(rivals, default=0.0)


def _corroboration(features: MessageFeatures, winner: MessageType) -> float:
    """Return agreeing-context strength for ``winner``, in ``[0, 1]``."""
    share = 0.0

    if _winning_keyword_count(features, winner) >= _MULTI_KEYWORD_THRESHOLD:
        share += _MULTI_KEYWORD_SHARE

    if features.context.is_trusted_business and winner in (
        MessageType.BUSINESS_UPDATE,
        MessageType.PROMOTION,
    ):
        share += _TRUSTED_SENDER_SHARE

    if features.history.total_interactions >= _SUBSTANTIVE_HISTORY:
        share += _HISTORY_SHARE

    if winner is MessageType.SCAM and (
        features.context.has_domain_mismatch
        or features.keywords.count(KeywordCategory.SCAM) >= _MULTI_KEYWORD_THRESHOLD
    ):
        share += _UNAMBIGUOUS_RISK_SHARE

    return clamp(share)


def _winning_keyword_count(features: MessageFeatures, winner: MessageType) -> int:
    """Return how many keywords of the family backing ``winner`` matched.

    Returns ``0`` for categories with no dedicated keyword family, such as
    ``personal``, whose support comes from context instead.
    """
    family = _KEYWORD_FAMILY_FOR_TYPE.get(winner)
    if family is None:
        return 0
    return features.keywords.count(family)


def _ambiguity_penalty(features: MessageFeatures, model: ConfidenceModel) -> float:
    """Return the total penalty for having little to go on."""
    penalty = 0.0
    if features.text.is_empty:
        penalty += model.no_text_penalty
    if features.keywords.total_matches == 0:
        penalty += model.no_keyword_penalty
    if not features.history.has_history:
        penalty += model.no_history_penalty
    return penalty


def _unknown_confidence(
    features: MessageFeatures, model: ConfidenceModel
) -> float:
    """Score an explicit ``unknown`` verdict.

    Declining to guess is itself a decision, and it is more defensible the
    less evidence there was. A message with no text and no keywords is
    confidently unclassifiable; one with conflicting signals is not.
    """
    confidence = model.base
    if features.keywords.total_matches == 0:
        confidence += model.margin_weight * 0.5
    if features.text.is_empty:
        confidence += model.evidence_weight * 0.5
    return round(clamp(confidence, model.floor, model.ceiling), 2)


#: Message types whose evidence comes from a dedicated keyword family.
_KEYWORD_FAMILY_FOR_TYPE: Final[Mapping[MessageType, KeywordCategory]] = {
    MessageType.SCAM: KeywordCategory.SCAM,
    MessageType.SPAM: KeywordCategory.SPAM,
    MessageType.URGENT: KeywordCategory.URGENT,
    MessageType.PAYMENT: KeywordCategory.PAYMENT,
    MessageType.PROMOTION: KeywordCategory.PROMOTION,
    MessageType.EVENT: KeywordCategory.EVENT,
    MessageType.GREETING: KeywordCategory.GREETING,
    MessageType.FORWARD: KeywordCategory.FORWARD,
    MessageType.BUSINESS_UPDATE: KeywordCategory.TRANSACTIONAL,
}

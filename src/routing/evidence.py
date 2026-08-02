"""Finds the historical messages that justify a routing decision.

Evidence has to *match the decision*, not merely relate to the message. If the
system muted something, the useful evidence is the comparable messages this
user dismissed or reported; if it notified, the ones they opened and replied
to. Returning "here are some earlier messages from this sender" regardless of
outcome would look like evidence while explaining nothing.

Candidates are drawn from ``message_history.csv`` scoped to this recipient and
counterparty, then scored on four axes:

* **same counterparty** - the sender, business or group that sent this;
* **same category** - historical messages carry no ``message_type``, so the
  Phase 2 classifier is reused to label them, cached across the run;
* **reaction matching the decision** - the axis that makes evidence
  explanatory rather than decorative;
* **recency** - recent behaviour describes the user better than old behaviour.

Returns at most :data:`MAX_EVIDENCE` ids, or an empty result that the output
layer renders as the required ``none`` sentinel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from src import config
from src.classifier.enums import MessageType
from src.classifier.message_classifier import MessageClassifier
from src.data.models import MessageEvent, MessageHistory
from src.data.repository import DataRepository
from src.features.extractor import FeatureExtractor
from src.personalization.normalization import decay
from src.routing.models import DecisionContext, RoutingAction, RoutingEvidence

__all__ = ["EvidenceEngine", "MAX_EVIDENCE"]

_LOGGER = config.get_logger("routing.evidence")

#: Most ids ever returned. The output format accepts a list, but a short one
#: is more useful to a human than an exhaustive one.
MAX_EVIDENCE: Final[int] = 3

#: Score below which a candidate is not worth citing. Prevents padding the
#: list with history that merely exists.
MIN_RELEVANCE: Final[float] = 0.35

#: Days at which a historical message's recency contribution halves.
RECENCY_HALF_LIFE_DAYS: Final[float] = 45.0

#: Relative importance of each matching axis. A weighted mean, so the
#: relevance score stays in [0, 1] whatever these are set to.
_WEIGHT_COUNTERPARTY: Final[float] = 0.30
_WEIGHT_CATEGORY: Final[float] = 0.25
_WEIGHT_REACTION: Final[float] = 0.30
_WEIGHT_RECENCY: Final[float] = 0.15


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One historical message being considered as evidence."""

    record: MessageHistory
    relevance: float
    matched_reaction: bool
    matched_category: bool


class EvidenceEngine:
    """Selects historical messages supporting a routing decision.

    Args:
        repo: A loaded repository.
        extractor: Feature extractor used to classify history. One is built
            when omitted; pass a shared instance to reuse its caches.
        classifier: Classifier used to label history.

    Example:
        >>> engine = EvidenceEngine(repo)                       # doctest: +SKIP
        >>> engine.find(context, RoutingAction.MUTE).message_ids
        ('message_0107', 'message_0231')
    """

    def __init__(
        self,
        repo: DataRepository,
        extractor: FeatureExtractor | None = None,
        classifier: MessageClassifier | None = None,
    ) -> None:
        self._repo = repo
        self._extractor = extractor if extractor is not None else FeatureExtractor(repo)
        self._classifier = classifier if classifier is not None else MessageClassifier()
        self._category_cache: dict[str, MessageType] = {}

    def find(
        self, context: DecisionContext, action: RoutingAction
    ) -> RoutingEvidence:
        """Return the historical messages that best justify ``action``.

        Args:
            context: The routed message's full context.
            action: The action that was decided.

        Returns:
            Up to :data:`MAX_EVIDENCE` ids ordered by relevance, or an empty
            result when nothing in the user's history is genuinely relevant.
        """
        candidates = self._score_candidates(context, action)
        chosen = [c for c in candidates if c.relevance >= MIN_RELEVANCE][:MAX_EVIDENCE]
        if not chosen:
            return RoutingEvidence(rationale="No comparable history for this user.")

        return RoutingEvidence(
            message_ids=tuple(c.record.message_id for c in chosen),
            rationale=self._describe(chosen, action),
        )

    # ------------------------------------------------------------------ #
    # Candidate scoring
    # ------------------------------------------------------------------ #

    def _score_candidates(
        self, context: DecisionContext, action: RoutingAction
    ) -> list[_Candidate]:
        """Score and rank every plausible piece of supporting history."""
        target_category = context.classification.message_type
        scored: list[_Candidate] = []

        for record in self._candidate_pool(context):
            event = self._repo.get_message_event(record.message_id)
            same_counterparty = _shares_counterparty(record, context)
            category = self._category_of(record)
            same_category = category is target_category
            reaction = _reaction_match(event, action)

            relevance = (
                _WEIGHT_COUNTERPARTY * float(same_counterparty)
                + _WEIGHT_CATEGORY * float(same_category)
                + _WEIGHT_REACTION * reaction
                + _WEIGHT_RECENCY
                * decay(
                    max(
                        0.0,
                        (context.features.created_at - record.created_at).days,
                    ),
                    RECENCY_HALF_LIFE_DAYS,
                )
            )
            scored.append(
                _Candidate(record, relevance, reaction > 0.0, same_category)
            )

        scored.sort(key=lambda c: (-c.relevance, c.record.message_id))
        return scored

    def _candidate_pool(self, context: DecisionContext) -> Sequence[MessageHistory]:
        """Return this recipient's history, newest first.

        Scoped to the recipient throughout: another user's reaction says
        nothing about how *this* user treats a message.
        """
        return self._repo.get_user_history(context.user_id, newest_first=True)

    def _category_of(self, record: MessageHistory) -> MessageType:
        """Classify a historical message, caching the result.

        ``message_history.csv`` carries no ``message_type``, so the Phase 2
        classifier supplies one. Reusing it rather than approximating with
        keyword overlap keeps the notion of "same category" identical to the
        one the routing decision was made with.
        """
        cached = self._category_cache.get(record.message_id)
        if cached is None:
            features = self._extractor.extract(record)
            cached = self._classifier.classify(features).message_type
            self._category_cache[record.message_id] = cached
        return cached

    @staticmethod
    def _describe(candidates: Sequence[_Candidate], action: RoutingAction) -> str:
        """Explain why these particular messages were cited."""
        parts = [f"{len(candidates)} comparable message(s)"]
        if any(c.matched_category for c in candidates):
            parts.append("of the same type")
        if any(c.matched_reaction for c in candidates):
            parts.append(f"the user reacted to consistently with {action.value}")
        return " ".join(parts) + "."


def _shares_counterparty(record: MessageHistory, context: DecisionContext) -> bool:
    """Whether a historical message came from the same sender, business or group."""
    features = context.features
    if features.sender_user_id and record.sender_user_id == features.sender_user_id:
        return True
    if features.business_id and record.business_id == features.business_id:
        return True
    return bool(features.group_id) and record.group_id == features.group_id


def _reaction_match(event: MessageEvent | None, action: RoutingAction) -> float:
    """Score how well a past reaction supports the decided action.

    This is what keeps evidence honest. Muting is justified by messages the
    user dismissed, muted or reported; notifying by messages they opened and
    replied to; digesting by the quieter middle, opened without urgency.

    Args:
        event: The recorded reaction, if any.
        action: The action being justified.

    Returns:
        ``0.0`` when the reaction contradicts the action or is unknown, rising
        to ``1.0`` for an unambiguous match.
    """
    if event is None:
        return 0.0

    if action is RoutingAction.MUTE:
        negatives = (
            event.notification_dismissed,
            event.muted_after_message,
            event.message_reported,
        )
        return min(1.0, sum(negatives) / 2.0)

    if action is RoutingAction.NOTIFY:
        if event.message_replied:
            return 1.0
        return 0.6 if event.message_opened else 0.0

    # digest: opened but not acted on urgently is the shape of "useful later".
    if event.message_opened and not event.message_replied:
        return 0.8
    return 0.3 if event.message_opened else 0.0

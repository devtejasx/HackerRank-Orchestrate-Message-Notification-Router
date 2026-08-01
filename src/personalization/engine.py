"""The Phase 3 entry point: features and a verdict in, routing signals out.

:class:`PersonalizationEngine` owns the ten calculators and the shared
statistics cache, then runs each calculator over the same read-only context.

It contains no scoring logic of its own, deliberately. Every judgement lives in
a calculator that can be constructed, exercised and argued with on its own, and
the engine's only job is to run them and collect the results. Swapping,
reweighting or adding a calculator therefore never touches this file's logic.

The engine produces **no routing decision**. ``notify``, ``digest`` and
``mute`` belong to Phase 4, which will combine these signals.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import fields

from src import config
from src.classifier.message_classifier import MessageClassification
from src.data.repository import DataRepository
from src.features.feature_models import MessageFeatures
from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.business_priority import BusinessPriorityCalculator
from src.personalization.engagement import EngagementCalculator
from src.personalization.fatigue import FatigueCalculator
from src.personalization.group_priority import GroupPriorityCalculator
from src.personalization.historical_importance import HistoricalImportanceCalculator
from src.personalization.interaction_stats import InteractionStatsProvider
from src.personalization.relationship import RelationshipStrengthCalculator
from src.personalization.risk import RiskCalculator
from src.personalization.sender_priority import SenderPriorityCalculator
from src.personalization.signal_models import RoutingSignal, RoutingSignals
from src.personalization.trust import TrustCalculator
from src.personalization.urgency import UrgencyCalculator

__all__ = ["PersonalizationEngine", "DEFAULT_CALCULATORS"]

_LOGGER = config.get_logger("personalization")

#: The calculators run for every message, in the order their signals appear on
#: :class:`~src.personalization.signal_models.RoutingSignals`.
DEFAULT_CALCULATORS: tuple[type[SignalCalculator], ...] = (
    SenderPriorityCalculator,
    BusinessPriorityCalculator,
    GroupPriorityCalculator,
    RelationshipStrengthCalculator,
    HistoricalImportanceCalculator,
    EngagementCalculator,
    FatigueCalculator,
    RiskCalculator,
    TrustCalculator,
    UrgencyCalculator,
)


class PersonalizationEngine:
    """Computes routing signals for a message, personalised to its recipient.

    Args:
        repo: A loaded repository.
        calculators: Calculator instances to run. Defaults to one instance of
            each entry in :data:`DEFAULT_CALCULATORS`. Pass your own to
            retune or extend without editing this class.

    Example:
        >>> engine = PersonalizationEngine(repo)                # doctest: +SKIP
        >>> signals = engine.compute(features, classification)  # doctest: +SKIP
        >>> signals.fatigue_modifier.score                      # doctest: +SKIP
        0.71
    """

    def __init__(
        self,
        repo: DataRepository,
        calculators: Sequence[SignalCalculator] | None = None,
    ) -> None:
        self._repo = repo
        self._stats = InteractionStatsProvider(repo)
        self._calculators: tuple[SignalCalculator, ...] = (
            tuple(calculators)
            if calculators is not None
            else tuple(factory() for factory in DEFAULT_CALCULATORS)
        )
        self._validate_calculators()

    @property
    def calculators(self) -> tuple[SignalCalculator, ...]:
        """The calculators this engine runs."""
        return self._calculators

    def compute(
        self, features: MessageFeatures, classification: MessageClassification
    ) -> RoutingSignals:
        """Compute every routing signal for one message.

        Args:
            features: Phase 2 feature record.
            classification: Phase 2 verdict for the same message.

        Returns:
            The complete signal set for this message and recipient.

        Raises:
            ValueError: If the feature record and verdict describe different
                messages, which would silently personalise the wrong thing.
        """
        if features.message_id != classification.message_id:
            raise ValueError(
                "features and classification describe different messages: "
                f"{features.message_id!r} vs {classification.message_id!r}"
            )

        context = SignalContext(
            repo=self._repo,
            features=features,
            classification=classification,
            stats=self._stats,
        )
        signals = {
            calculator.name: calculator.calculate(context)
            for calculator in self._calculators
        }
        return self._assemble(features, signals)

    def compute_many(
        self,
        pairs: Iterable[tuple[MessageFeatures, MessageClassification]],
    ) -> tuple[RoutingSignals, ...]:
        """Compute signals for many messages, reusing the statistics cache."""
        results = tuple(
            self.compute(features, classification) for features, classification in pairs
        )
        _LOGGER.info("Computed routing signals for %d message(s)", len(results))
        return results

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _validate_calculators(self) -> None:
        """Fail fast when the configured calculators cannot fill the output.

        Raises:
            ValueError: If a required signal is missing or produced twice.
        """
        produced = [calculator.name for calculator in self._calculators]
        duplicates = {name for name in produced if produced.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate calculator names: {sorted(duplicates)}")

        missing = set(_SIGNAL_FIELDS) - set(produced)
        if missing:
            raise ValueError(
                f"No calculator produces required signal(s): {sorted(missing)}"
            )

    @staticmethod
    def _assemble(
        features: MessageFeatures, signals: dict[str, RoutingSignal]
    ) -> RoutingSignals:
        """Build the output record from the named signals."""
        return RoutingSignals(
            message_id=features.message_id,
            user_id=features.user_id,
            **{field: signals[field] for field in _SIGNAL_FIELDS},
        )


#: Identifier fields on :class:`RoutingSignals` that are not signals.
_IDENTIFIER_FIELDS: frozenset[str] = frozenset({"message_id", "user_id"})

#: Names of the signal fields on :class:`RoutingSignals`, which double as the
#: required calculator names. Derived from the dataclass so the two cannot
#: drift apart.
_SIGNAL_FIELDS: tuple[str, ...] = tuple(
    spec.name for spec in fields(RoutingSignals) if spec.name not in _IDENTIFIER_FIELDS
)

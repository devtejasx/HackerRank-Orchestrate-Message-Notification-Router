"""The contract every routing-signal calculator implements.

Each calculator is independent: it receives the same read-only
:class:`SignalContext` and returns exactly one :class:`RoutingSignal`. None of
them may see another's output, which is what keeps them separately testable and
lets Phase 4 reweigh any of them without touching the rest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from src.classifier.message_classifier import MessageClassification
from src.data.repository import DataRepository
from src.features.feature_models import MessageFeatures
from src.personalization.interaction_stats import (
    InteractionStats,
    InteractionStatsProvider,
)
from src.personalization.normalization import Contribution, blend, explain
from src.personalization.signal_models import RoutingSignal, SignalPolarity

__all__ = ["SignalContext", "SignalCalculator"]


@dataclass(frozen=True, slots=True)
class SignalContext:
    """Everything a calculator may read, and nothing it may change.

    Bundles the Phase 1 repository with the Phase 2 outputs so calculators
    never re-derive what an earlier phase already established.

    Attributes:
        repo: The loaded repository.
        features: Phase 2 feature record for this message.
        classification: Phase 2 verdict for this message.
        stats: Provider of scoped interaction statistics.
    """

    repo: DataRepository
    features: MessageFeatures
    classification: MessageClassification
    stats: InteractionStatsProvider

    # -- Identity shortcuts ------------------------------------------- #

    @property
    def message_id(self) -> str:
        """Identifier of the message being scored."""
        return self.features.message_id

    @property
    def user_id(self) -> str:
        """The recipient these signals are personalised for."""
        return self.features.user_id

    @property
    def sender_user_id(self) -> str | None:
        """The sending user, when a person sent this."""
        return self.features.sender_user_id

    @property
    def group_id(self) -> str | None:
        """The originating group, when applicable."""
        return self.features.group_id

    @property
    def business_id(self) -> str | None:
        """The sending business, when applicable."""
        return self.features.business_id

    @property
    def now(self) -> datetime:
        """The message's arrival time, used as "now" for every recency decay.

        Recency is measured against the message rather than the wall clock, so
        results are reproducible and independent of when the code runs.
        """
        return self.features.created_at

    # -- Scoped statistics -------------------------------------------- #

    @property
    def user_stats(self) -> InteractionStats:
        """How this user reacts to notifications overall."""
        return self.stats.for_user(self.user_id)

    @property
    def sender_stats(self) -> InteractionStats:
        """How this user reacts to this individual sender."""
        return self.stats.for_sender(self.user_id, self.sender_user_id)

    @property
    def business_stats(self) -> InteractionStats:
        """How this user reacts to this business."""
        return self.stats.for_business(self.user_id, self.business_id)

    @property
    def group_stats(self) -> InteractionStats:
        """How this user reacts to messages from this group."""
        return self.stats.for_group(self.user_id, self.group_id)


class SignalCalculator(ABC):
    """Base class for the ten routing-signal calculators.

    Subclasses declare :attr:`name` and :attr:`polarity`, then implement
    :meth:`contributions`. The base assembles the signal, so scoring,
    explanation and range-checking happen the same way everywhere and a
    subclass cannot accidentally return a reason that disagrees with its score.

    A subclass that needs bespoke assembly may override :meth:`calculate`
    instead; :class:`~src.personalization.risk.RiskCalculator` does, because it
    takes its confidence from the Phase 2 classifier rather than from sample
    size.
    """

    #: Stable identifier for the produced signal.
    name: ClassVar[str] = ""

    #: Which way a rising score moves routing priority.
    polarity: ClassVar[SignalPolarity] = SignalPolarity.BOOST

    #: Maximum explanations carried on the signal, so reasons stay readable.
    max_reasons: ClassVar[int] = 3

    @abstractmethod
    def contributions(self, context: SignalContext) -> Sequence[Contribution]:
        """Return the normalised inputs to this signal.

        Args:
            context: Read-only view of the repository and Phase 2 outputs.

        Returns:
            The contributions to blend. An empty sequence means "no evidence",
            and the signal lands on neutral with zero confidence.
        """

    @abstractmethod
    def confidence(self, context: SignalContext) -> float:
        """Return how much evidence backed the score, in ``[0, 1]``.

        Args:
            context: Read-only view of the repository and Phase 2 outputs.
        """

    def calculate(self, context: SignalContext) -> RoutingSignal:
        """Produce this calculator's signal.

        Args:
            context: Read-only view of the repository and Phase 2 outputs.

        Returns:
            A signal whose reasons are generated from the same contributions
            that produced its score.
        """
        contributions = list(self.contributions(context))
        return RoutingSignal(
            name=self.name,
            score=blend(contributions),
            confidence=self.confidence(context),
            polarity=self.polarity,
            reasons=explain(contributions, limit=self.max_reasons),
        )

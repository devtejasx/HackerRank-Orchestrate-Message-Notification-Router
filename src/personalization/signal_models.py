"""The Phase 3 output: independent routing signals.

Phase 3 deliberately does **not** decide ``notify``, ``digest`` or ``mute``.
It produces the evidence that decision will rest on, as a set of independent
signals rather than a single blended number, so Phase 4 can weigh them for
itself and explain what it did.

Every signal carries three things:

* **score** - the measurement, always in ``[0, 1]``, with ``0.5`` meaning
  "no information either way";
* **confidence** - how much evidence stood behind the score, which is a
  different question from the score itself;
* **reasons** - the explanations, generated from the same contributions that
  produced the score, so the two can never disagree.

Signals also declare a :class:`SignalPolarity`. Most raise priority as they
rise, but fatigue and risk lower it. Stating the direction on the signal means
Phase 4 never has to remember which is which.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any, Final

from src.personalization.normalization import NEUTRAL

__all__ = ["SignalPolarity", "RoutingSignal", "RoutingSignals"]


class SignalPolarity(StrEnum):
    """Which way a rising score moves routing priority."""

    #: A higher score argues for interrupting the user sooner.
    BOOST = "boost"
    #: A higher score argues for holding the message back.
    SUPPRESS = "suppress"


@dataclass(frozen=True, slots=True)
class RoutingSignal:
    """One independent measurement feeding the future routing decision.

    Attributes:
        name: Stable identifier, e.g. ``"sender_priority"``.
        score: The measurement in ``[0, 1]``, where ``0.5`` is neutral.
        confidence: How much evidence backed the score, in ``[0, 1]``.
        polarity: Whether a rising score raises or lowers priority.
        reasons: Human-readable explanations, strongest first.
    """

    name: str
    score: float
    confidence: float
    polarity: SignalPolarity = SignalPolarity.BOOST
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Guard the range contract every consumer relies on."""
        for field_name in ("score", "confidence"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{self.name}.{field_name} = {value} outside [0, 1]")

    @property
    def reason(self) -> str:
        """The explanations as one sentence, or a neutral statement."""
        if not self.reasons:
            return "No distinguishing evidence."
        return "; ".join(self.reasons) + "."

    @property
    def is_neutral(self) -> bool:
        """Whether the score is close enough to neutral to carry no argument."""
        return abs(self.score - NEUTRAL) < NEUTRAL_TOLERANCE

    @property
    def signed_strength(self) -> float:
        """How far this signal pushes priority, as a value in ``[-1, 1]``.

        Positive raises priority and negative lowers it, with the sign already
        resolved from :attr:`polarity` and the magnitude scaled by
        :attr:`confidence`. Phase 4 can sum these directly; it is offered as a
        convenience, not as a decision.
        """
        offset = (self.score - NEUTRAL) * 2.0
        direction = 1.0 if self.polarity is SignalPolarity.BOOST else -1.0
        return offset * direction * self.confidence

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-friendly view."""
        return {
            "name": self.name,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "polarity": self.polarity.value,
            "reasons": list(self.reasons),
        }


#: How far from neutral a score must sit before it counts as saying anything.
NEUTRAL_TOLERANCE: Final[float] = 0.05


@dataclass(frozen=True, slots=True)
class RoutingSignals:
    """The complete set of routing signals for one message.

    Everything Phase 4 needs to make a personalised routing decision, and
    nothing that presumes one.

    Attributes:
        message_id: The message these signals describe.
        user_id: The recipient they were personalised for.
        sender_priority: How much this individual sender matters to this user.
        business_priority: How much this business matters to this user.
        group_priority: How much this group matters to this user.
        relationship_strength: Strength of the tie to whoever sent this.
        historical_importance: How the user treated comparable messages before.
        engagement_modifier: How engaged this user is with notifications overall.
        fatigue_modifier: How overloaded the user already is. Suppressing.
        risk_modifier: How unsafe the message looks. Suppressing.
        trust_modifier: How trustworthy the sender is.
        urgency_modifier: How time-critical the content appears.
    """

    message_id: str
    user_id: str
    sender_priority: RoutingSignal
    business_priority: RoutingSignal
    group_priority: RoutingSignal
    relationship_strength: RoutingSignal
    historical_importance: RoutingSignal
    engagement_modifier: RoutingSignal
    fatigue_modifier: RoutingSignal
    risk_modifier: RoutingSignal
    trust_modifier: RoutingSignal
    urgency_modifier: RoutingSignal

    @property
    def all_signals(self) -> tuple[RoutingSignal, ...]:
        """Every signal, in declaration order."""
        return tuple(
            getattr(self, spec.name)
            for spec in fields(self)
            if spec.name not in _NON_SIGNAL_FIELDS
        )

    def by_name(self, name: str) -> RoutingSignal | None:
        """Return the signal called ``name``, or ``None``."""
        return next((s for s in self.all_signals if s.name == name), None)

    @property
    def reasons(self) -> tuple[str, ...]:
        """Every explanation across every signal, strongest signal first.

        These become the routing reasons in a later phase.
        """
        ordered = sorted(
            self.all_signals, key=lambda signal: -abs(signal.signed_strength)
        )
        collected: list[str] = []
        for signal in ordered:
            for reason in signal.reasons:
                if reason not in collected:
                    collected.append(reason)
        return tuple(collected)

    @property
    def boosting(self) -> tuple[RoutingSignal, ...]:
        """Signals currently arguing for a sooner interruption."""
        return tuple(s for s in self.all_signals if s.signed_strength > 0)

    @property
    def suppressing(self) -> tuple[RoutingSignal, ...]:
        """Signals currently arguing for holding the message back."""
        return tuple(s for s in self.all_signals if s.signed_strength < 0)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-friendly view."""
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "signals": {signal.name: signal.to_dict() for signal in self.all_signals},
            "reasons": list(self.reasons),
        }


#: Fields of :class:`RoutingSignals` that are identifiers rather than signals.
_NON_SIGNAL_FIELDS: Final[frozenset[str]] = frozenset({"message_id", "user_id"})

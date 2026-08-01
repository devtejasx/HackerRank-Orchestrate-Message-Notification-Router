"""Normalisation primitives shared by every routing-signal calculator.

The brief says not to invent arbitrary weights. Two ideas keep that honest:

* **Every raw quantity is mapped into ``[0, 1]`` by a named curve** whose one
  parameter has a stated meaning - the point at which the curve reaches its
  half-way value. "Five prior messages is where familiarity is half-formed" is
  a claim that can be argued with; "multiply by 0.17" is not.
* **Scores are weighted means of those normalised values.** A weighted mean of
  numbers in ``[0, 1]`` is provably in ``[0, 1]``, so no clamping is needed and
  weights express nothing more than relative importance. Absent inputs are
  simply omitted and the remaining weights re-normalise themselves.

The same :class:`Contribution` objects that produce the score also produce the
explanation, so a signal's reasons can never drift out of sync with its value.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from src.utils.helpers import clamp

__all__ = [
    "Contribution",
    "NEUTRAL",
    "HIGH_THRESHOLD",
    "LOW_THRESHOLD",
    "blend",
    "one_sided",
    "explain",
    "saturating",
    "decay",
    "evidence_confidence",
    "days_between",
    "trend_score",
]

#: The "no information either way" point. Every score is expressed on a scale
#: where this value means the signal neither raises nor lowers priority.
NEUTRAL: Final[float] = 0.5

#: A contribution at or above this counts as notably high, and at or below
#: :data:`LOW_THRESHOLD` as notably low. Only notable contributions produce an
#: explanation, so reasons stay short and meaningful.
HIGH_THRESHOLD: Final[float] = 0.65
LOW_THRESHOLD: Final[float] = 0.35


@dataclass(frozen=True, slots=True)
class Contribution:
    """One normalised input to a signal score.

    Attributes:
        name: Short identifier, used in diagnostics.
        value: The normalised quantity, in ``[0, 1]``.
        weight: Relative importance against the other contributions. Only the
            ratio between weights matters; they need not sum to anything.
        high_reason: Explanation emitted when the value is notably high.
        low_reason: Explanation emitted when the value is notably low.
    """

    name: str
    value: float
    weight: float
    high_reason: str | None = None
    low_reason: str | None = None

    def __post_init__(self) -> None:
        """Guard the invariants the blend relies on."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"{self.name}: value {self.value} outside [0, 1]")
        if self.weight <= 0.0:
            raise ValueError(f"{self.name}: weight {self.weight} must be positive")

    @property
    def reason(self) -> str | None:
        """The explanation for this contribution, if it is notable."""
        if self.value >= HIGH_THRESHOLD:
            return self.high_reason
        if self.value <= LOW_THRESHOLD:
            return self.low_reason
        return None


def blend(contributions: Sequence[Contribution], default: float = NEUTRAL) -> float:
    """Combine contributions into a single score in ``[0, 1]``.

    A weighted mean, so the result is bounded by construction rather than by
    clamping, and dropping an unavailable input re-normalises the rest
    automatically.

    Args:
        contributions: The inputs. May be empty.
        default: Returned when there is nothing to blend, so a signal with no
            evidence lands on neutral rather than zero. Zero would be a strong
            claim; neutral is the honest one.

    Returns:
        The weighted mean, or ``default`` when no contributions were given.
    """
    if not contributions:
        return default
    total_weight = sum(item.weight for item in contributions)
    weighted = sum(item.value * item.weight for item in contributions)
    return clamp(weighted / total_weight)


def one_sided(value: float) -> float:
    """Rescale ``[0, 1]`` onto ``[0.5, 1]`` so absence of a thing stays neutral.

    Most signals are genuinely two-sided: a sender the user never engages with
    is real evidence for holding a message back, not merely an absence of
    evidence for sending it. A few are not. "This message shows no sign of
    being a scam" is not an argument for interrupting someone, and "this
    message is not urgent" - true of most messages - is not an argument for
    suppressing it. Without this rescaling both would push as hard as their
    positive counterparts, purely as an artefact of sitting below neutral.

    Args:
        value: Strength of the phenomenon, in ``[0, 1]``, where ``0`` means
            "no sign of it".

    Returns:
        ``0.5`` when the phenomenon is absent, rising to ``1.0`` when it is
        unmistakable.
    """
    return NEUTRAL + clamp(value) * (1.0 - NEUTRAL)


def explain(contributions: Iterable[Contribution], limit: int | None = None) -> tuple[str, ...]:
    """Collect the explanations of the notable contributions.

    Ordered by how far each contribution sits from neutral, so the strongest
    reason comes first.

    Args:
        contributions: The same inputs given to :func:`blend`.
        limit: Maximum number of reasons to return.

    Returns:
        Distinct explanations, strongest first.
    """
    scored = [
        (abs(item.value - NEUTRAL), item.reason)
        for item in contributions
        if item.reason
    ]
    scored.sort(key=lambda pair: -pair[0])

    seen: list[str] = []
    for _, reason in scored:
        if reason not in seen:
            seen.append(reason)
    return tuple(seen[:limit]) if limit is not None else tuple(seen)


def saturating(value: float, half_point: float) -> float:
    """Map a non-negative count onto ``[0, 1)`` with diminishing returns.

    ``value / (value + half_point)``. The second message from a sender says far
    more than the twentieth, which is exactly this shape.

    Args:
        value: A non-negative quantity, typically a count.
        half_point: The value at which the result is ``0.5``. This is the only
            parameter, and it is a statement about the domain: "five prior
            messages is where familiarity is half-formed".

    Returns:
        ``0.0`` for zero input, approaching ``1.0`` as the value grows.
    """
    if value <= 0.0:
        return 0.0
    if half_point <= 0.0:
        raise ValueError("half_point must be positive")
    return value / (value + half_point)


def decay(elapsed_days: float | None, half_life_days: float) -> float:
    """Map an age in days onto ``[0, 1]`` by exponential decay.

    Args:
        elapsed_days: Age of the event. ``None`` means "never happened".
        half_life_days: Age at which the result is ``0.5``.

    Returns:
        ``1.0`` for something that just happened, halving every half-life, and
        ``0.0`` when the event never happened at all.
    """
    if elapsed_days is None:
        return 0.0
    if half_life_days <= 0.0:
        raise ValueError("half_life_days must be positive")
    return clamp(0.5 ** (max(0.0, elapsed_days) / half_life_days))


def evidence_confidence(sample_size: int, half_point: float) -> float:
    """Score how much a conclusion can be trusted, given how much backed it.

    Confidence answers a different question from the score itself: a 100% reply
    rate over one message and over forty are the same score and very different
    certainties.

    Args:
        sample_size: Number of observations behind the score.
        half_point: Sample size at which confidence reaches ``0.5``.

    Returns:
        A value in ``[0, 1)``, zero when there is no evidence at all.
    """
    return saturating(max(0, sample_size), half_point)


def days_between(earlier: datetime | None, later: datetime) -> float | None:
    """Return whole days from ``earlier`` to ``later``.

    Args:
        earlier: The past moment, or ``None`` if it never happened.
        later: The reference moment, normally the incoming message's timestamp.

    Returns:
        Days elapsed, never negative, or ``None`` when ``earlier`` is ``None``.
        A future timestamp yields ``0.0`` rather than a negative age.
    """
    if earlier is None:
        return None
    return max(0.0, (later - earlier).total_seconds() / _SECONDS_PER_DAY)


def trend_score(recent: float, earlier: float) -> float:
    """Express a change between two rates as a value around neutral.

    Args:
        recent: The more recent rate, in ``[0, 1]``.
        earlier: The older rate, in ``[0, 1]``.

    Returns:
        ``0.5`` when flat, above when improving, below when declining. The
        difference is halved so a total collapse maps to ``0.0`` and a total
        surge to ``1.0``.
    """
    return clamp(NEUTRAL + (recent - earlier) / 2.0)


_SECONDS_PER_DAY: Final[float] = 86_400.0

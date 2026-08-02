"""Records produced by the routing engine.

:class:`RoutingResult` is the final artefact of the whole system. It carries
exactly the six fields the submission format needs, so Phase 5 has only to
write them to CSV - no further derivation, lookup or formatting.

Everything else here exists to make that result explainable:
:class:`RoutingDecision` records which rules fired and how the actions scored,
:class:`RoutingEvidence` records which historical messages justify it, and
:class:`RoutingReason` records the sentence shown to a human.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from src.classifier.message_classifier import MessageClassification
from src.data.repository import DataRepository
from src.features.feature_models import MessageFeatures
from src.personalization.signal_models import RoutingSignals

__all__ = [
    "RoutingAction",
    "RuleOutcome",
    "RoutingDecision",
    "RoutingEvidence",
    "RoutingReason",
    "DecisionContext",
    "RoutingResult",
    "NO_EVIDENCE",
]

#: The sentinel the submission format requires when no evidence applies.
NO_EVIDENCE: Final[str] = "none"

#: Separator for the evidence id list in the output format.
EVIDENCE_SEPARATOR: Final[str] = ";"


class RoutingAction(StrEnum):
    """What the system decides to do with a message.

    The three values the challenge allows, and the only ones this system may
    emit.
    """

    #: Interrupt the user now.
    NOTIFY = "notify"
    #: Useful, but it can wait for a batched summary.
    DIGEST = "digest"
    #: Low-value, unwanted, repetitive or unsafe. Suppress it.
    MUTE = "mute"

    @property
    def interrupts(self) -> bool:
        """Whether this action breaks into the user's attention now."""
        return self is RoutingAction.NOTIFY


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    """One rule's argument for an action.

    Attributes:
        rule: Name of the rule that produced this.
        action: The action being argued for.
        weight: Strength of the argument. Always positive; a rule that opposes
            an action argues for a different one rather than voting negatively.
        reason: Human-readable justification, reused verbatim when this rule
            turns out to be decisive.
        override: When true this rule forces its action regardless of the
            accumulated scores. Reserved for safety: a confirmed scam is muted
            even if every other signal is enthusiastic.
    """

    rule: str
    action: RoutingAction
    weight: float
    reason: str
    override: bool = False

    def __post_init__(self) -> None:
        """Guard the invariant the aggregation relies on."""
        if self.weight <= 0.0:
            raise ValueError(f"{self.rule}: weight {self.weight} must be positive")


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Which action won, and the full record of how.

    Attributes:
        action: The winning action.
        scores: Total weight accumulated per action.
        outcomes: Every rule outcome, in the order the rules ran.
        decisive: The outcomes that supported the winning action, strongest
            first. These drive the generated reason.
        overridden: Whether a safety override forced the action.
    """

    action: RoutingAction
    scores: Mapping[RoutingAction, float]
    outcomes: tuple[RuleOutcome, ...]
    decisive: tuple[RuleOutcome, ...]
    overridden: bool = False

    def __post_init__(self) -> None:
        """Freeze the score mapping so the record is genuinely immutable."""
        object.__setattr__(self, "scores", dict(self.scores))

    @property
    def runner_up(self) -> RoutingAction | None:
        """The next-best action, or ``None`` when nothing else scored."""
        rivals = {
            action: score
            for action, score in self.scores.items()
            if action is not self.action and score > 0.0
        }
        if not rivals:
            return None
        return max(rivals, key=lambda action: rivals[action])

    @property
    def margin(self) -> float:
        """Gap between the winning score and the runner-up's.

        Large when the rules agree, near zero when the call was close. The
        confidence calibrator reads this as its primary input.
        """
        winning = self.scores.get(self.action, 0.0)
        runner_up = self.runner_up
        if runner_up is None:
            return winning
        return winning - self.scores[runner_up]

    @property
    def rules_fired(self) -> tuple[str, ...]:
        """Names of every rule that produced an outcome."""
        return tuple(dict.fromkeys(outcome.rule for outcome in self.outcomes))

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-friendly view."""
        return {
            "action": self.action.value,
            "scores": {a.value: round(s, 3) for a, s in self.scores.items()},
            "margin": round(self.margin, 3),
            "overridden": self.overridden,
            "rules_fired": list(self.rules_fired),
            "decisive": [o.rule for o in self.decisive],
        }


@dataclass(frozen=True, slots=True)
class RoutingEvidence:
    """Historical messages that justify the decision.

    Attributes:
        message_ids: Historical ids, strongest first.
        rationale: Why these were selected, for debugging and explanation.
    """

    message_ids: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def has_evidence(self) -> bool:
        """Whether any supporting history was found."""
        return bool(self.message_ids)

    def formatted(self) -> str:
        """Render for the submission format.

        Returns:
            Semicolon-separated ids, or the literal ``none`` sentinel the
            output contract requires when nothing useful was found.
        """
        if not self.message_ids:
            return NO_EVIDENCE
        return EVIDENCE_SEPARATOR.join(self.message_ids)


@dataclass(frozen=True, slots=True)
class RoutingReason:
    """The human-readable explanation for a decision.

    Attributes:
        text: One sentence, derived from the rules that actually decided.
        supporting: The individual rule reasons it was built from.
    """

    text: str
    supporting: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Everything a routing rule may read.

    Bundles the outputs of all three earlier phases so no rule ever needs to
    re-derive what has already been established.

    Attributes:
        features: Phase 2 feature record.
        classification: Phase 2 verdict.
        signals: Phase 3 routing signals.
        repo: Phase 1 repository, for the few rules needing raw records.
    """

    features: MessageFeatures
    classification: MessageClassification
    signals: RoutingSignals
    repo: DataRepository

    @property
    def message_id(self) -> str:
        """Identifier of the message being routed."""
        return self.features.message_id

    @property
    def user_id(self) -> str:
        """The recipient this decision is personalised for."""
        return self.features.user_id

    @property
    def message_type(self) -> str:
        """The Phase 2 category, as its string value."""
        return self.classification.message_type.value


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """The final routing outcome for one message.

    Carries exactly what the submission format needs. Phase 5 writes
    :meth:`to_output_row` straight to CSV.

    Attributes:
        message_id: The routed message.
        action: ``notify``, ``digest`` or ``mute``.
        message_type: The Phase 2 category.
        reason: One-sentence explanation.
        confidence: Calibrated certainty in ``[0, 1]``.
        evidence: Supporting historical messages.
        decision: The full decision breakdown, for inspection and debugging.
    """

    message_id: str
    action: RoutingAction
    message_type: str
    reason: RoutingReason
    confidence: float
    evidence: RoutingEvidence = field(default_factory=RoutingEvidence)
    decision: RoutingDecision | None = None

    def __post_init__(self) -> None:
        """Guard the range the output contract requires."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")

    @property
    def evidence_message_ids(self) -> str:
        """Evidence rendered for the output format, or ``none``."""
        return self.evidence.formatted()

    def to_output_row(self) -> dict[str, Any]:
        """Return the row Phase 5 will write, in the required column order.

        Returns:
            A mapping whose keys are exactly the six output columns.
        """
        return {
            "message_id": self.message_id,
            "action": self.action.value,
            "message_type": self.message_type,
            "reason": self.reason.text,
            "confidence": round(self.confidence, 2),
            "evidence_message_ids": self.evidence_message_ids,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a full, JSON-friendly view including the breakdown."""
        payload: dict[str, Any] = dict(self.to_output_row())
        payload["evidence_rationale"] = self.evidence.rationale
        payload["supporting_reasons"] = list(self.reason.supporting)
        if self.decision is not None:
            payload["decision_breakdown"] = self.decision.to_dict()
        return payload


#: Column order required by the submission format.
OUTPUT_COLUMNS: Final[Sequence[str]] = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)

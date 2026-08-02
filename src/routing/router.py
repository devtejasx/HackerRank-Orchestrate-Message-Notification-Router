"""Composes the routing stages into one result.

    decide -> find evidence -> generate reason -> calibrate confidence

Order matters and is not arbitrary. Evidence is retrieved *after* the decision
because it must justify the action actually taken; the reason is written after
the evidence so it can mention it; confidence is calibrated last because it
reads all three.

The router owns no routing logic. It sequences the four engines, each of which
can be replaced independently.
"""

from __future__ import annotations

from src import config
from src.routing.confidence import ConfidenceCalibrator
from src.routing.decision_engine import DecisionEngine
from src.routing.evidence import EvidenceEngine
from src.routing.models import DecisionContext, RoutingResult
from src.routing.reason_generator import ReasonGenerator

__all__ = ["Router"]

_LOGGER = config.get_logger("routing.router")


class Router:
    """Produces a complete :class:`RoutingResult` for one message.

    Args:
        decision_engine: Chooses the action.
        evidence_engine: Finds supporting history. Required, because evidence
            needs repository access the other stages do not.
        reason_generator: Writes the explanation.
        calibrator: Scores confidence.

    Example:
        >>> router = Router(evidence_engine=EvidenceEngine(repo))  # doctest: +SKIP
        >>> router.route(context).action                           # doctest: +SKIP
        <RoutingAction.MUTE: 'mute'>
    """

    def __init__(
        self,
        evidence_engine: EvidenceEngine,
        decision_engine: DecisionEngine | None = None,
        reason_generator: ReasonGenerator | None = None,
        calibrator: ConfidenceCalibrator | None = None,
    ) -> None:
        self._decision_engine = decision_engine or DecisionEngine()
        self._evidence_engine = evidence_engine
        self._reason_generator = reason_generator or ReasonGenerator()
        self._calibrator = calibrator or ConfidenceCalibrator()

    @property
    def decision_engine(self) -> DecisionEngine:
        """The decision engine in use."""
        return self._decision_engine

    @property
    def evidence_engine(self) -> EvidenceEngine:
        """The evidence engine in use."""
        return self._evidence_engine

    def route(self, context: DecisionContext) -> RoutingResult:
        """Route one message end to end.

        Args:
            context: The outputs of Phases 1-3 for this message.

        Returns:
            The action, its explanation, its confidence, the supporting
            evidence and the full decision breakdown.
        """
        decision = self._decision_engine.decide(context)
        evidence = self._evidence_engine.find(context, decision.action)
        reason = self._reason_generator.generate(decision, evidence, context)
        confidence = self._calibrator.calibrate(context, decision, evidence)

        _LOGGER.debug(
            "%s -> %s (confidence %.2f, %d evidence)",
            context.message_id,
            decision.action.value,
            confidence,
            len(evidence.message_ids),
        )
        return RoutingResult(
            message_id=context.message_id,
            action=decision.action,
            message_type=context.message_type,
            reason=reason,
            confidence=confidence,
            evidence=evidence,
            decision=decision,
        )

"""Phase 4 routing: the final notify / digest / mute decision.

Consumes Phase 1 (repository), Phase 2 (features and classification) and
Phase 3 (routing signals). Rebuilds none of them.

    from src.routing import RoutingPipeline

    result = RoutingPipeline.load().route_all()[0]
    result.action                    # <RoutingAction.MUTE: 'mute'>
    result.to_output_row()           # exactly the six submission columns
"""

from src.routing.confidence import (
    DEFAULT_CALIBRATION,
    CalibrationModel,
    ConfidenceCalibrator,
)
from src.routing.decision_engine import DecisionEngine
from src.routing.evidence import EvidenceEngine
from src.routing.models import (
    NO_EVIDENCE,
    OUTPUT_COLUMNS,
    DecisionContext,
    RoutingAction,
    RoutingDecision,
    RoutingEvidence,
    RoutingReason,
    RoutingResult,
    RuleOutcome,
)
from src.routing.pipeline import RoutingPipeline
from src.routing.reason_generator import ReasonGenerator
from src.routing.router import Router
from src.routing.rules import DEFAULT_RULES, DEFAULT_THRESHOLDS, TYPE_PRIORS, Thresholds

__all__ = [
    "DEFAULT_CALIBRATION",
    "DEFAULT_RULES",
    "DEFAULT_THRESHOLDS",
    "NO_EVIDENCE",
    "OUTPUT_COLUMNS",
    "TYPE_PRIORS",
    "CalibrationModel",
    "ConfidenceCalibrator",
    "DecisionContext",
    "DecisionEngine",
    "EvidenceEngine",
    "ReasonGenerator",
    "Router",
    "RoutingAction",
    "RoutingDecision",
    "RoutingEvidence",
    "RoutingPipeline",
    "RoutingReason",
    "RoutingResult",
    "RuleOutcome",
    "Thresholds",
]

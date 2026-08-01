"""Phase 3 personalisation: routing signals, not routing decisions.

Turns Phase 2 outputs plus repository context into ten independent, normalised,
explained signals that Phase 4 will combine into a ``notify`` / ``digest`` /
``mute`` decision. Nothing here decides anything.

    from src.personalization import PersonalizationEngine

    signals = PersonalizationEngine(repo).compute(features, classification)
    signals.fatigue_modifier.score      # 0.71
    signals.reasons                     # ordered explanations
"""

from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.business_priority import BusinessPriorityCalculator
from src.personalization.engagement import EngagementCalculator
from src.personalization.engine import DEFAULT_CALCULATORS, PersonalizationEngine
from src.personalization.fatigue import FatigueCalculator
from src.personalization.group_priority import GroupPriorityCalculator
from src.personalization.historical_importance import HistoricalImportanceCalculator
from src.personalization.interaction_stats import (
    InteractionScope,
    InteractionStats,
    InteractionStatsProvider,
)
from src.personalization.normalization import (
    Contribution,
    blend,
    decay,
    evidence_confidence,
    explain,
    one_sided,
    saturating,
)
from src.personalization.relationship import RelationshipStrengthCalculator
from src.personalization.risk import RiskCalculator
from src.personalization.sender_priority import SenderPriorityCalculator
from src.personalization.signal_models import (
    RoutingSignal,
    RoutingSignals,
    SignalPolarity,
)
from src.personalization.trust import TrustCalculator
from src.personalization.urgency import UrgencyCalculator

__all__ = [
    "DEFAULT_CALCULATORS",
    "BusinessPriorityCalculator",
    "Contribution",
    "EngagementCalculator",
    "FatigueCalculator",
    "GroupPriorityCalculator",
    "HistoricalImportanceCalculator",
    "InteractionScope",
    "InteractionStats",
    "InteractionStatsProvider",
    "PersonalizationEngine",
    "RelationshipStrengthCalculator",
    "RiskCalculator",
    "RoutingSignal",
    "RoutingSignals",
    "SenderPriorityCalculator",
    "SignalCalculator",
    "SignalContext",
    "SignalPolarity",
    "TrustCalculator",
    "UrgencyCalculator",
    "blend",
    "decay",
    "evidence_confidence",
    "explain",
    "one_sided",
    "saturating",
]

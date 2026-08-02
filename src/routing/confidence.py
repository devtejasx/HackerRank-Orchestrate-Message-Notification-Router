"""Calibrates confidence in a routing decision.

Confidence answers "how sure are we this action is right", which is a
different question from "how strong was the winning score". A message with
overwhelming arguments for both ``notify`` and ``mute`` is not a confident
call, however large those arguments are.

Four inputs:

* **margin** - how far the winning action outscored the runner-up. Dominant,
  because a clear winner is what confidence means here.
* **agreement** - what share of the total rule weight backed the winner. A
  decision carried by one rule against four is shakier than the margin alone
  suggests.
* **upstream certainty** - the Phase 2 classifier's own confidence, since the
  category drives the type prior. A decision built on a shaky classification
  cannot be firmer than the classification.
* **corroboration** - independent support: historical evidence, agreeing
  Phase 3 signals, a verified sender, an unambiguous safety override.

Margin and agreement are squashed with ``tanh`` so the result saturates
smoothly, and the whole thing is bounded to a band that never claims certainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from src.personalization.signal_models import RoutingSignals
from src.routing.models import (
    DecisionContext,
    RoutingAction,
    RoutingDecision,
    RoutingEvidence,
)
from src.utils.helpers import clamp

__all__ = ["ConfidenceCalibrator", "DEFAULT_CALIBRATION"]


@dataclass(frozen=True, slots=True)
class CalibrationModel:
    """Tunable parameters for routing confidence.

    Attributes:
        base: Starting point before any evidence is considered.
        margin_weight: Maximum contribution from a decisive win.
        agreement_weight: Maximum contribution from rules pulling together.
        upstream_weight: Maximum contribution from classifier certainty.
        corroboration_weight: Maximum contribution from independent support.
        margin_scale: Margin, in score units, at which the margin component is
            roughly three-quarters saturated.
        conflict_penalty: Applied when a substantial share of the rule weight
            argued for a different action.
        conflict_threshold: Share of weight opposing the winner that counts as
            genuine conflict.
        floor: Lowest confidence ever returned.
        ceiling: Highest confidence ever returned. Below 1.0 because a
            rule-based router should never claim certainty.
    """

    base: float = 0.42
    margin_weight: float = 0.26
    agreement_weight: float = 0.14
    upstream_weight: float = 0.10
    corroboration_weight: float = 0.14

    margin_scale: float = 1.80
    conflict_penalty: float = 0.08
    conflict_threshold: float = 0.40

    floor: float = 0.30
    ceiling: float = 0.95


DEFAULT_CALIBRATION: Final[CalibrationModel] = CalibrationModel()

#: Corroboration is expressed as shares of ``corroboration_weight``. They sum
#: to 1.0, so a fully corroborated decision earns the whole weight and no more.
_EVIDENCE_SHARE: Final[float] = 0.30
_SIGNAL_AGREEMENT_SHARE: Final[float] = 0.30
_TRUSTED_SENDER_SHARE: Final[float] = 0.20
_SAFETY_OVERRIDE_SHARE: Final[float] = 0.20

#: Signals whose direction is compared against the decision.
_DECISIVE_SIGNAL_NAMES: Final[tuple[str, ...]] = (
    "sender_priority",
    "historical_importance",
    "risk_modifier",
    "trust_modifier",
    "engagement_modifier",
)

#: Share of the named signals that must agree before agreement is credited.
_SIGNAL_AGREEMENT_RATIO: Final[float] = 0.6


class ConfidenceCalibrator:
    """Scores how much to trust a routing decision.

    Args:
        model: Tuning to apply.
    """

    def __init__(self, model: CalibrationModel = DEFAULT_CALIBRATION) -> None:
        self._model = model

    @property
    def model(self) -> CalibrationModel:
        """The calibration in use."""
        return self._model

    def calibrate(
        self,
        context: DecisionContext,
        decision: RoutingDecision,
        evidence: RoutingEvidence,
    ) -> float:
        """Return confidence in ``decision``, in ``[floor, ceiling]``.

        Args:
            context: The routed message's full context.
            decision: The decision being scored.
            evidence: Supporting history found for it.

        Returns:
            A rounded value inside the configured band.
        """
        model = self._model
        total = sum(decision.scores.values())
        winning = decision.scores.get(decision.action, 0.0)
        agreement = winning / total if total > 0 else 0.0

        confidence = model.base
        confidence += model.margin_weight * math.tanh(
            decision.margin / model.margin_scale
        )
        confidence += model.agreement_weight * agreement
        confidence += model.upstream_weight * context.classification.confidence
        confidence += model.corroboration_weight * self._corroboration(
            context, decision, evidence
        )
        confidence -= self._conflict_penalty(agreement)

        return round(clamp(confidence, model.floor, model.ceiling), 2)

    def _corroboration(
        self,
        context: DecisionContext,
        decision: RoutingDecision,
        evidence: RoutingEvidence,
    ) -> float:
        """Return independent support for the decision, in ``[0, 1]``."""
        share = 0.0
        if evidence.has_evidence:
            share += _EVIDENCE_SHARE
        if _signals_agree(context.signals, decision.action):
            share += _SIGNAL_AGREEMENT_SHARE
        if context.features.context.is_trusted_business:
            share += _TRUSTED_SENDER_SHARE
        if decision.overridden:
            share += _SAFETY_OVERRIDE_SHARE
        return clamp(share)

    def _conflict_penalty(self, agreement: float) -> float:
        """Penalise decisions a large share of the rules argued against."""
        if agreement >= (1.0 - self._model.conflict_threshold):
            return 0.0
        return self._model.conflict_penalty


def _signals_agree(signals: RoutingSignals, action: RoutingAction) -> bool:
    """Whether the Phase 3 signals point the same way as the decision.

    Only signals with real confidence are counted, so a message where most
    signals do not apply is not credited with agreement it never had.

    Args:
        signals: The Phase 3 signal set.
        action: The decided action.

    Returns:
        ``True`` when a clear majority of applicable signals push the same way.
    """
    applicable = [
        signal
        for name in _DECISIVE_SIGNAL_NAMES
        if (signal := signals.by_name(name)) is not None and signal.confidence > 0.0
    ]
    if not applicable:
        return False

    if action is RoutingAction.NOTIFY:
        agreeing = sum(1 for signal in applicable if signal.signed_strength > 0)
    elif action is RoutingAction.MUTE:
        agreeing = sum(1 for signal in applicable if signal.signed_strength < 0)
    else:
        # digest is the middle ground: agreement means nothing pulls strongly.
        agreeing = sum(1 for signal in applicable if abs(signal.signed_strength) < 0.5)

    return agreeing / len(applicable) >= _SIGNAL_AGREEMENT_RATIO

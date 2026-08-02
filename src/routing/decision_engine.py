"""Aggregates rule outcomes into one routing decision.

The engine holds no routing knowledge of its own. It runs the rules, sums what
they argue for, applies safety overrides and picks a winner - so changing how
the system routes means changing a rule, never this file.

Resolution order:

1. **Safety overrides.** If any rule declared an override, the strongest one
   wins outright. Reserved for cases where no accumulation of positive signals
   should be able to let a message through.
2. **Weighted argmax.** Otherwise the action with the most accumulated weight
   wins.
3. **Deterministic tie-break.** Exact ties resolve toward the more
   conservative action, so an unresolvable call never interrupts the user.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from src import config
from src.routing.models import (
    DecisionContext,
    RoutingAction,
    RoutingDecision,
    RuleOutcome,
)
from src.routing.rules import DEFAULT_RULES, DEFAULT_THRESHOLDS, RoutingRule, Thresholds

__all__ = ["DecisionEngine"]

_LOGGER = config.get_logger("routing.decision")

#: Tie-break order, most conservative first. When two actions score exactly
#: the same the system declines to interrupt: a wrong ``mute`` costs the user
#: one missed message, a wrong ``notify`` costs them their attention and
#: erodes trust in every future notification.
_TIE_BREAK_ORDER: Final[tuple[RoutingAction, ...]] = (
    RoutingAction.MUTE,
    RoutingAction.DIGEST,
    RoutingAction.NOTIFY,
)

_TIE_BREAK_RANK: Final[Mapping[RoutingAction, int]] = {
    action: rank for rank, action in enumerate(_TIE_BREAK_ORDER)
}

#: How many supporting outcomes are kept as decisive, for the reason generator.
_MAX_DECISIVE: Final[int] = 3


class DecisionEngine:
    """Turns Phase 1-3 output into a routing decision.

    Args:
        rules: The rules to run. Defaults to
            :data:`~src.routing.rules.DEFAULT_RULES`. Pass your own to retune
            or extend without editing this class.
        thresholds: Tuning applied to every rule.

    Example:
        >>> engine = DecisionEngine()                     # doctest: +SKIP
        >>> engine.decide(context).action                 # doctest: +SKIP
        <RoutingAction.MUTE: 'mute'>
    """

    def __init__(
        self,
        rules: Sequence[RoutingRule] | None = None,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
    ) -> None:
        self._rules: tuple[RoutingRule, ...] = (
            tuple(rules) if rules is not None else DEFAULT_RULES
        )
        self._thresholds = thresholds

    @property
    def rules(self) -> tuple[RoutingRule, ...]:
        """The rules this engine runs."""
        return self._rules

    @property
    def thresholds(self) -> Thresholds:
        """The tuning in use."""
        return self._thresholds

    def decide(self, context: DecisionContext) -> RoutingDecision:
        """Choose an action for one message.

        Args:
            context: The outputs of Phases 1-3 for this message.

        Returns:
            The winning action, the accumulated scores, every rule outcome and
            the ones that decided it.
        """
        outcomes = self._collect(context)
        scores = _accumulate(outcomes)

        override = _strongest_override(outcomes)
        if override is not None:
            action, overridden = override.action, True
        else:
            action, overridden = _argmax(scores), False

        decision = RoutingDecision(
            action=action,
            scores=scores,
            outcomes=outcomes,
            decisive=_decisive_for(outcomes, action, override),
            overridden=overridden,
        )
        _LOGGER.debug(
            "%s -> %s (margin %.2f, %d rule(s))",
            context.message_id,
            action.value,
            decision.margin,
            len(decision.rules_fired),
        )
        return decision

    def _collect(self, context: DecisionContext) -> tuple[RuleOutcome, ...]:
        """Run every rule and gather what they argue for."""
        return tuple(
            outcome
            for rule in self._rules
            for outcome in rule(context, self._thresholds)
        )


def _accumulate(outcomes: Sequence[RuleOutcome]) -> dict[RoutingAction, float]:
    """Sum rule weights per action, including actions nothing argued for."""
    scores = dict.fromkeys(RoutingAction, 0.0)
    for outcome in outcomes:
        scores[outcome.action] += outcome.weight
    return scores


def _strongest_override(outcomes: Sequence[RuleOutcome]) -> RuleOutcome | None:
    """Return the heaviest overriding outcome, or ``None`` if none fired."""
    overrides = [outcome for outcome in outcomes if outcome.override]
    if not overrides:
        return None
    return max(overrides, key=lambda outcome: outcome.weight)


def _argmax(scores: Mapping[RoutingAction, float]) -> RoutingAction:
    """Return the highest-scoring action, breaking ties conservatively."""
    return max(
        scores,
        key=lambda action: (scores[action], -_TIE_BREAK_RANK[action]),
    )


def _decisive_for(
    outcomes: Sequence[RuleOutcome],
    action: RoutingAction,
    override: RuleOutcome | None,
) -> tuple[RuleOutcome, ...]:
    """Return the outcomes that carried the winning action, strongest first.

    An override is always listed first: when a safety rule forced the outcome,
    that is the honest explanation regardless of what else was accumulating.
    """
    supporting = sorted(
        (outcome for outcome in outcomes if outcome.action is action),
        key=lambda outcome: -outcome.weight,
    )
    if override is not None:
        supporting = [override] + [o for o in supporting if o is not override]
    return tuple(supporting[:_MAX_DECISIVE])

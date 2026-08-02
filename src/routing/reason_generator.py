"""Writes the one-sentence explanation shown with a routing decision.

The reason is assembled from the rules that actually decided, never composed
separately. That is the same discipline used in Phases 2 and 3, and for the
same reason: an explanation written independently of the mechanism drifts out
of sync with it and eventually starts lying.

Concretely, the generator only ever reads
:attr:`~src.routing.models.RoutingDecision.decisive` - the outcomes that
carried the winning action. It cannot mention a factor that did not contribute,
and it cannot omit an override that forced the outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from src.routing.models import (
    RoutingAction,
    RoutingDecision,
    RoutingEvidence,
    RoutingReason,
    RuleOutcome,
)

__all__ = ["ReasonGenerator"]

#: Most rule reasons woven into one sentence. Beyond this it stops reading as
#: an explanation and starts reading as a log.
MAX_CLAUSES: Final[int] = 2

#: Hard cap on the generated sentence, so it stays readable in a CSV cell.
MAX_LENGTH: Final[int] = 240

#: Clause appended when supporting history was found, phrased so it reads as
#: part of the sentence rather than as a footnote.
_EVIDENCE_CLAUSE: Final[str] = "consistent with how the user treated similar messages"

#: Used when no rule produced a usable sentence, which should not happen while
#: the type prior always fires but is handled rather than assumed.
_FALLBACK: Final[dict[RoutingAction, str]] = {
    RoutingAction.NOTIFY: "The message appears important enough to interrupt the user.",
    RoutingAction.DIGEST: "The message is useful but does not need immediate attention.",
    RoutingAction.MUTE: "The message is low value for this user.",
}


class ReasonGenerator:
    """Builds the explanation for a decision from the rules that made it."""

    def generate(
        self,
        decision: RoutingDecision,
        evidence: RoutingEvidence | None = None,
    ) -> RoutingReason:
        """Return the explanation for ``decision``.

        Args:
            decision: The decision to explain.
            evidence: Supporting history, mentioned only when it exists.

        Returns:
            A single sentence plus the individual rule reasons behind it.
        """
        clauses = _clauses(decision.decisive)
        if not clauses:
            return RoutingReason(text=_FALLBACK[decision.action])

        sentence = _join(clauses)
        if evidence is not None and evidence.has_evidence:
            sentence = f"{sentence} This is {_EVIDENCE_CLAUSE}."

        return RoutingReason(
            text=_truncate(sentence),
            supporting=tuple(outcome.reason for outcome in decision.decisive),
        )


def _clauses(decisive: Sequence[RuleOutcome]) -> list[str]:
    """Return the distinct rule sentences, strongest first, capped."""
    seen: list[str] = []
    for outcome in decisive:
        reason = outcome.reason.strip()
        if reason and reason not in seen:
            seen.append(reason)
    return seen[:MAX_CLAUSES]


def _join(clauses: Sequence[str]) -> str:
    """Join rule sentences into one flowing explanation.

    Each rule reason is already a complete sentence, so the second is folded
    in as a subordinate clause rather than simply concatenated.
    """
    if len(clauses) == 1:
        return clauses[0]

    lead, *rest = clauses
    follow_on = " ".join(_lower_first(clause.rstrip(".")) + "." for clause in rest)
    return f"{lead.rstrip('.')}, and {follow_on}"


def _lower_first(text: str) -> str:
    """Lowercase the first character unless it begins a proper noun.

    Only the first word is inspected: rule sentences start with an article or
    a determiner far more often than with a name, and getting this wrong is
    merely cosmetic.
    """
    if not text:
        return text
    first, rest = text[0], text[1:]
    if rest[:1].isupper():
        return text
    return first.lower() + rest


def _truncate(sentence: str) -> str:
    """Keep the sentence inside :data:`MAX_LENGTH`, cutting at a word boundary."""
    if len(sentence) <= MAX_LENGTH:
        return sentence
    clipped = sentence[: MAX_LENGTH - 1].rsplit(" ", 1)[0].rstrip(",;.")
    return f"{clipped}."

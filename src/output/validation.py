"""Validates predictions before they are written.

A malformed submission scores zero however good the reasoning behind it was,
so every structural guarantee the output contract makes is checked here rather
than trusted. The checks run against the finished
:class:`~src.routing.models.RoutingResult` list and the input messages, so
they catch both format faults and coverage gaps - a prediction that is
perfectly formatted but missing for one message is still a failed submission.

Mirrors the severity split used by the Phase 1 dataset validator: structural
faults raise, quality concerns warn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from src import config
from src.classifier.enums import MessageType
from src.data.models import Message
from src.routing.models import NO_EVIDENCE, RoutingAction, RoutingResult

__all__ = [
    "OutputSeverity",
    "OutputIssue",
    "OutputValidationReport",
    "OutputValidationError",
    "validate_results",
]

_LOGGER = config.get_logger("output.validation")

#: Separator the evidence column uses.
_EVIDENCE_SEPARATOR: Final[str] = ";"

#: Longest reason accepted. Beyond this the cell stops being readable, which
#: matters because a human grades it.
_MAX_REASON_LENGTH: Final[int] = 300

#: Shortest reason that could plausibly explain anything.
_MIN_REASON_LENGTH: Final[int] = 10

#: Share of identical reasons above which the explanations look templated.
_MAX_DUPLICATE_REASON_SHARE: Final[float] = 0.5

#: Every value the ``action`` column may take.
_VALID_ACTIONS: Final[frozenset[str]] = frozenset(a.value for a in RoutingAction)

#: Every value the ``message_type`` column may take.
_VALID_MESSAGE_TYPES: Final[frozenset[str]] = frozenset(t.value for t in MessageType)


class OutputSeverity(StrEnum):
    """How serious an output problem is."""

    #: Makes the submission invalid. Blocks writing.
    ERROR = "ERROR"
    #: Worth knowing about; the submission is still well-formed.
    WARNING = "WARNING"


@dataclass(frozen=True, slots=True)
class OutputIssue:
    """One problem found in the predictions.

    Attributes:
        severity: Whether this blocks the submission.
        check: Short machine-readable name, e.g. ``"missing_prediction"``.
        message: Human-readable description.
        count: How many rows are affected.
        examples: Up to :data:`src.config.MAX_ISSUE_EXAMPLES` offending ids.
    """

    severity: OutputSeverity
    check: str
    message: str
    count: int = 0
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        head = f"[{self.severity.value}] {self.check}: {self.message}"
        if self.examples:
            return f"{head} (examples: {', '.join(self.examples)})"
        return head


@dataclass(slots=True)
class OutputValidationReport:
    """Findings from one validation run, plus the summary counts."""

    issues: list[OutputIssue] = field(default_factory=list)
    total: int = 0
    action_counts: Mapping[str, int] = field(default_factory=dict)

    def add(self, issue: OutputIssue) -> None:
        """Record a finding."""
        self.issues.append(issue)

    @property
    def errors(self) -> tuple[OutputIssue, ...]:
        """Findings that block writing."""
        return tuple(i for i in self.issues if i.severity is OutputSeverity.ERROR)

    @property
    def warnings(self) -> tuple[OutputIssue, ...]:
        """Findings that do not block writing."""
        return tuple(i for i in self.issues if i.severity is OutputSeverity.WARNING)

    @property
    def is_valid(self) -> bool:
        """Whether the predictions are safe to submit."""
        return not self.errors

    def log(self) -> None:
        """Emit every finding at its matching level."""
        for issue in self.issues:
            emit = (
                _LOGGER.error
                if issue.severity is OutputSeverity.ERROR
                else _LOGGER.warning
            )
            emit("%s", issue)

    def raise_for_errors(self) -> None:
        """Raise if the predictions are not submittable.

        Raises:
            OutputValidationError: When any blocking error was recorded.
        """
        if self.errors:
            detail = "\n  ".join(str(issue) for issue in self.errors)
            raise OutputValidationError(
                f"Predictions failed validation with {len(self.errors)} error(s):"
                f"\n  {detail}",
                report=self,
            )

    def summary_lines(self) -> tuple[str, ...]:
        """Return the human-readable summary the CLI prints."""
        lines = [f"messages processed : {self.total}"]
        for action in RoutingAction:
            count = self.action_counts.get(action.value, 0)
            share = count / self.total if self.total else 0.0
            lines.append(f"{action.value + ' count':<19}: {count:>4}  ({share:.0%})")
        lines.append(
            f"validation         : {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s)"
        )
        return tuple(lines)


class OutputValidationError(RuntimeError):
    """Raised when predictions are not fit to submit.

    Attributes:
        report: The full report, so callers can inspect warnings too.
    """

    def __init__(self, message: str, report: OutputValidationReport) -> None:
        super().__init__(message)
        self.report = report


def validate_results(
    results: Sequence[RoutingResult], messages: Sequence[Message]
) -> OutputValidationReport:
    """Check predictions against the output contract.

    Args:
        results: The predictions about to be written.
        messages: The input rows they must cover, in dataset order.

    Returns:
        The report. Nothing is raised here; call
        :meth:`OutputValidationReport.raise_for_errors` to enforce it.
    """
    report = OutputValidationReport(
        total=len(results), action_counts=_count_actions(results)
    )

    _check_coverage(results, messages, report)
    _check_uniqueness(results, report)
    for check in (_check_actions, _check_message_types, _check_confidence,
                  _check_reasons, _check_evidence):
        check(results, report)
    _check_reason_variety(results, report)
    return report


def _count_actions(results: Sequence[RoutingResult]) -> dict[str, int]:
    """Count predictions per action."""
    counts = dict.fromkeys((a.value for a in RoutingAction), 0)
    for result in results:
        counts[result.action.value] += 1
    return counts


def _check_coverage(
    results: Sequence[RoutingResult],
    messages: Sequence[Message],
    report: OutputValidationReport,
) -> None:
    """Every input row must get exactly one prediction, and no extras."""
    predicted = [result.message_id for result in results]
    expected = [message.message_id for message in messages]

    missing = tuple(sorted(set(expected) - set(predicted)))
    if missing:
        report.add(
            OutputIssue(
                OutputSeverity.ERROR,
                "missing_prediction",
                f"{len(missing)} input message(s) have no prediction",
                count=len(missing),
                examples=missing[: config.MAX_ISSUE_EXAMPLES],
            )
        )

    unexpected = tuple(sorted(set(predicted) - set(expected)))
    if unexpected:
        report.add(
            OutputIssue(
                OutputSeverity.ERROR,
                "unexpected_prediction",
                f"{len(unexpected)} prediction(s) do not correspond to any input row",
                count=len(unexpected),
                examples=unexpected[: config.MAX_ISSUE_EXAMPLES],
            )
        )

    if not missing and not unexpected and predicted != expected:
        report.add(
            OutputIssue(
                OutputSeverity.WARNING,
                "row_order",
                "Predictions are not in the same order as the input rows",
            )
        )


def _check_uniqueness(
    results: Sequence[RoutingResult], report: OutputValidationReport
) -> None:
    """No message may be predicted twice."""
    seen: dict[str, int] = {}
    for result in results:
        seen[result.message_id] = seen.get(result.message_id, 0) + 1
    duplicates = tuple(sorted(mid for mid, count in seen.items() if count > 1))
    if duplicates:
        report.add(
            OutputIssue(
                OutputSeverity.ERROR,
                "duplicate_message_id",
                f"{len(duplicates)} message id(s) appear more than once",
                count=len(duplicates),
                examples=duplicates[: config.MAX_ISSUE_EXAMPLES],
            )
        )


def _check_actions(
    results: Sequence[RoutingResult], report: OutputValidationReport
) -> None:
    """Only the three allowed actions may appear."""
    offenders = tuple(
        r.message_id for r in results if r.action.value not in _VALID_ACTIONS
    )
    if offenders:
        report.add(
            OutputIssue(
                OutputSeverity.ERROR,
                "invalid_action",
                f"{len(offenders)} row(s) carry an action outside {sorted(_VALID_ACTIONS)}",
                count=len(offenders),
                examples=offenders[: config.MAX_ISSUE_EXAMPLES],
            )
        )


def _check_message_types(
    results: Sequence[RoutingResult], report: OutputValidationReport
) -> None:
    """Only the eleven allowed categories may appear."""
    offenders = tuple(
        r.message_id for r in results if r.message_type not in _VALID_MESSAGE_TYPES
    )
    if offenders:
        report.add(
            OutputIssue(
                OutputSeverity.ERROR,
                "invalid_message_type",
                f"{len(offenders)} row(s) carry an undeclared message_type",
                count=len(offenders),
                examples=offenders[: config.MAX_ISSUE_EXAMPLES],
            )
        )


def _check_confidence(
    results: Sequence[RoutingResult], report: OutputValidationReport
) -> None:
    """Confidence must be a real number inside ``[0, 1]``."""
    offenders = tuple(
        r.message_id
        for r in results
        if not isinstance(r.confidence, (int, float))
        or not 0.0 <= r.confidence <= 1.0
        or r.confidence != r.confidence  # NaN
    )
    if offenders:
        report.add(
            OutputIssue(
                OutputSeverity.ERROR,
                "confidence_out_of_range",
                f"{len(offenders)} row(s) have a confidence outside [0, 1]",
                count=len(offenders),
                examples=offenders[: config.MAX_ISSUE_EXAMPLES],
            )
        )


def _check_reasons(
    results: Sequence[RoutingResult], report: OutputValidationReport
) -> None:
    """Reasons must be present, substantive and readable."""
    empty = tuple(r.message_id for r in results if not r.reason.text.strip())
    if empty:
        report.add(
            OutputIssue(
                OutputSeverity.ERROR,
                "empty_reason",
                f"{len(empty)} row(s) have no reason",
                count=len(empty),
                examples=empty[: config.MAX_ISSUE_EXAMPLES],
            )
        )

    too_short = tuple(
        r.message_id
        for r in results
        if 0 < len(r.reason.text.strip()) < _MIN_REASON_LENGTH
    )
    if too_short:
        report.add(
            OutputIssue(
                OutputSeverity.WARNING,
                "reason_too_short",
                f"{len(too_short)} reason(s) are under {_MIN_REASON_LENGTH} characters",
                count=len(too_short),
                examples=too_short[: config.MAX_ISSUE_EXAMPLES],
            )
        )

    too_long = tuple(
        r.message_id for r in results if len(r.reason.text) > _MAX_REASON_LENGTH
    )
    if too_long:
        report.add(
            OutputIssue(
                OutputSeverity.WARNING,
                "reason_too_long",
                f"{len(too_long)} reason(s) exceed {_MAX_REASON_LENGTH} characters",
                count=len(too_long),
                examples=too_long[: config.MAX_ISSUE_EXAMPLES],
            )
        )


def _check_evidence(
    results: Sequence[RoutingResult], report: OutputValidationReport
) -> None:
    """Evidence must be the sentinel or a clean semicolon-separated id list."""
    malformed: list[str] = []
    for result in results:
        rendered = result.evidence_message_ids
        if rendered == NO_EVIDENCE:
            continue
        parts = rendered.split(_EVIDENCE_SEPARATOR)
        is_blank_or_padded = not all(parts) or any(
            part != part.strip() for part in parts
        )
        if is_blank_or_padded or len(set(parts)) != len(parts):
            malformed.append(result.message_id)

    if malformed:
        report.add(
            OutputIssue(
                OutputSeverity.ERROR,
                "malformed_evidence",
                f"{len(malformed)} row(s) have a blank, padded or duplicated evidence id",
                count=len(malformed),
                examples=tuple(malformed)[: config.MAX_ISSUE_EXAMPLES],
            )
        )


def _check_reason_variety(
    results: Sequence[RoutingResult], report: OutputValidationReport
) -> None:
    """Warn when explanations look templated rather than derived.

    Not a format fault, but a scored one: identical reasoning across most rows
    means the explanations are not describing the individual decisions.
    """
    if not results:
        return
    distinct = len({r.reason.text for r in results})
    duplicate_share = 1.0 - (distinct / len(results))
    if duplicate_share > _MAX_DUPLICATE_REASON_SHARE:
        report.add(
            OutputIssue(
                OutputSeverity.WARNING,
                "repetitive_reasons",
                f"Only {distinct} distinct reason(s) across {len(results)} rows",
                count=len(results) - distinct,
            )
        )

"""Measures the system against the labelled examples.

``sample_messages.csv`` is the only labelled data in the dataset: thirty rows
carrying a ground-truth ``action`` and ``message_type``. This module replays
them through the real pipeline and reports how well the system agrees.

Five things are measured, because a submission is graded on more than whether
the action matched:

* **action agreement** - the headline number;
* **message_type agreement** - a wrong category usually drags the action with it;
* **confidence behaviour** - calibration means being *less* sure when wrong,
  so the gap between confidence on correct and incorrect decisions is the
  measure, not the average;
* **reason quality** - length, variety and whether each traces to a rule;
* **evidence quality** - whether cited ids exist, belong to the right
  recipient, and carry a reaction consistent with the action taken.

The result is an optimistic estimate, not held-out performance: the system was
refined against these same rows. :meth:`EvaluationReport.summary_lines` says so
in its own output rather than leaving the reader to assume otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from src import config
from src.data.models import Message, SampleMessage
from src.data.repository import DataRepository
from src.routing.models import RoutingAction, RoutingResult
from src.routing.pipeline import RoutingPipeline

__all__ = ["EvaluationReport", "SampleOutcome", "evaluate_samples", "as_message"]

_LOGGER = config.get_logger("evaluation")

#: Fields shared by :class:`SampleMessage` and :class:`Message`. The labelled
#: rows carry the same envelope plus their labels, so replaying one as an
#: incoming message is a projection rather than a conversion.
_MESSAGE_FIELDS: Final[tuple[str, ...]] = (
    "message_id",
    "user_id",
    "conversation_type",
    "group_id",
    "business_id",
    "sender_user_id",
    "created_at",
    "message_text",
    "media_type",
    "media_id",
    "forwarded_count",
)


def as_message(sample: SampleMessage) -> Message:
    """Reinterpret a labelled example as an incoming message.

    Args:
        sample: A row of ``sample_messages.csv``.

    Returns:
        The same message without its labels, so the pipeline cannot see them.
    """
    return Message(**{field: getattr(sample, field) for field in _MESSAGE_FIELDS})


@dataclass(frozen=True, slots=True)
class SampleOutcome:
    """What the system predicted for one labelled example."""

    message_id: str
    expected_action: str
    predicted_action: str
    expected_type: str
    predicted_type: str
    confidence: float
    evidence_count: int
    reason: str

    @property
    def action_correct(self) -> bool:
        """Whether the routing action matched the label."""
        return self.expected_action == self.predicted_action

    @property
    def type_correct(self) -> bool:
        """Whether the category matched the label."""
        return self.expected_type == self.predicted_type


@dataclass(slots=True)
class EvaluationReport:
    """Aggregated results of one evaluation run."""

    outcomes: list[SampleOutcome] = field(default_factory=list)
    evidence_total: int = 0
    evidence_resolvable: int = 0
    evidence_right_recipient: int = 0
    evidence_consistent_reaction: int = 0

    # -- headline agreement -------------------------------------------- #

    @property
    def total(self) -> int:
        """Number of labelled examples evaluated."""
        return len(self.outcomes)

    @property
    def action_accuracy(self) -> float:
        """Share of examples whose action matched."""
        return self._share(o.action_correct for o in self.outcomes)

    @property
    def type_accuracy(self) -> float:
        """Share of examples whose category matched."""
        return self._share(o.type_correct for o in self.outcomes)

    @property
    def both_accuracy(self) -> float:
        """Share of examples where action and category both matched."""
        return self._share(
            o.action_correct and o.type_correct for o in self.outcomes
        )

    # -- confidence ----------------------------------------------------- #

    @property
    def confidence_when_correct(self) -> float:
        """Mean confidence on decisions that matched the label."""
        return self._mean(o.confidence for o in self.outcomes if o.action_correct)

    @property
    def confidence_when_wrong(self) -> float:
        """Mean confidence on decisions that did not match."""
        return self._mean(o.confidence for o in self.outcomes if not o.action_correct)

    @property
    def calibration_gap(self) -> float:
        """Confidence on correct decisions minus confidence on wrong ones.

        Positive means the system is less sure when it is wrong, which is what
        calibration is for. Zero or negative would mean confidence carries no
        information.
        """
        if not any(not o.action_correct for o in self.outcomes):
            return self.confidence_when_correct
        return self.confidence_when_correct - self.confidence_when_wrong

    # -- reasons and evidence -------------------------------------------- #

    @property
    def distinct_reasons(self) -> int:
        """Number of distinct explanations produced."""
        return len({o.reason for o in self.outcomes})

    @property
    def mean_reason_length(self) -> float:
        """Mean explanation length in characters."""
        return self._mean(float(len(o.reason)) for o in self.outcomes)

    @property
    def results_with_evidence(self) -> int:
        """How many examples cited at least one historical message."""
        return sum(1 for o in self.outcomes if o.evidence_count > 0)

    @property
    def evidence_precision(self) -> float:
        """Share of cited ids whose recorded reaction fits the action taken."""
        if self.evidence_total == 0:
            return 0.0
        return self.evidence_consistent_reaction / self.evidence_total

    # -- misses ---------------------------------------------------------- #

    @property
    def action_misses(self) -> tuple[SampleOutcome, ...]:
        """Examples whose action did not match, for inspection."""
        return tuple(o for o in self.outcomes if not o.action_correct)

    def action_confusion(self) -> dict[tuple[str, str], int]:
        """Return ``(expected, predicted) -> count`` for mismatches."""
        confusion: dict[tuple[str, str], int] = {}
        for outcome in self.action_misses:
            key = (outcome.expected_action, outcome.predicted_action)
            confusion[key] = confusion.get(key, 0) + 1
        return confusion

    def summary_lines(self) -> tuple[str, ...]:
        """Return the human-readable metrics block."""
        if not self.outcomes:
            return ("No labelled examples available.",)
        return (
            f"action agreement       : {self._hits(lambda o: o.action_correct)}"
            f"/{self.total} = {self.action_accuracy:.1%}",
            f"message_type agreement : {self._hits(lambda o: o.type_correct)}"
            f"/{self.total} = {self.type_accuracy:.1%}",
            f"both correct           : "
            f"{self._hits(lambda o: o.action_correct and o.type_correct)}"
            f"/{self.total} = {self.both_accuracy:.1%}",
            "",
            f"confidence when correct: {self.confidence_when_correct:.2f}",
            f"confidence when wrong  : {self.confidence_when_wrong:.2f}",
            f"calibration gap        : {self.calibration_gap:+.2f}"
            f"  (positive is correct behaviour)",
            "",
            f"distinct reasons       : {self.distinct_reasons}/{self.total}",
            f"mean reason length     : {self.mean_reason_length:.0f} characters",
            "",
            f"rows citing evidence   : {self.results_with_evidence}/{self.total}",
            f"evidence ids resolvable: {self.evidence_resolvable}/{self.evidence_total}",
            f"correct recipient      : {self.evidence_right_recipient}"
            f"/{self.evidence_total}",
            f"reaction fits action   : {self.evidence_consistent_reaction}"
            f"/{self.evidence_total} = {self.evidence_precision:.0%}",
            "",
            "Note: the system was refined against these same rows, so this is an",
            "optimistic estimate rather than held-out performance.",
        )

    # -- helpers ---------------------------------------------------------- #

    def _hits(self, predicate) -> int:
        """Count outcomes satisfying ``predicate``."""
        return sum(1 for outcome in self.outcomes if predicate(outcome))

    def _share(self, flags) -> float:
        """Return the share of true values, or ``0.0`` when empty."""
        values = list(flags)
        return sum(values) / len(values) if values else 0.0

    def _mean(self, values) -> float:
        """Return the mean, or ``0.0`` when empty."""
        collected = list(values)
        return sum(collected) / len(collected) if collected else 0.0


def evaluate_samples(pipeline: RoutingPipeline) -> EvaluationReport:
    """Replay every labelled example through the pipeline and score it.

    Args:
        pipeline: A ready routing pipeline.

    Returns:
        The full report. Empty when the optional ``sample_messages.csv`` is
        absent, which is not an error.
    """
    repo = pipeline.repository
    if not repo.loader.is_available("sample_messages"):
        _LOGGER.warning("sample_messages.csv is not present; skipping evaluation")
        return EvaluationReport()

    report = EvaluationReport()
    for sample in repo.loader.records("sample_messages"):
        result = pipeline.route(as_message(sample))
        report.outcomes.append(_outcome_for(sample, result))
        _score_evidence(sample, result, repo, report)

    _LOGGER.info(
        "Evaluated %d labelled example(s): action agreement %.1f%%",
        report.total,
        report.action_accuracy * 100,
    )
    return report


def _outcome_for(sample: SampleMessage, result: RoutingResult) -> SampleOutcome:
    """Build the per-example record."""
    return SampleOutcome(
        message_id=sample.message_id,
        expected_action=sample.action,
        predicted_action=result.action.value,
        expected_type=sample.message_type,
        predicted_type=result.message_type,
        confidence=result.confidence,
        evidence_count=len(result.evidence.message_ids),
        reason=result.reason.text,
    )


def _score_evidence(
    sample: SampleMessage,
    result: RoutingResult,
    repo: DataRepository,
    report: EvaluationReport,
) -> None:
    """Accumulate evidence-quality counts for one example.

    Quality here means three separate things: the id resolves to a real
    historical message, that message was received by *this* recipient, and the
    reaction recorded against it is consistent with the action taken.
    """
    for message_id in result.evidence.message_ids:
        report.evidence_total += 1
        record = repo.get_history_message(message_id)
        if record is None:
            continue
        report.evidence_resolvable += 1
        if record.user_id == sample.user_id:
            report.evidence_right_recipient += 1
        if _reaction_fits(repo, message_id, result.action):
            report.evidence_consistent_reaction += 1


def _reaction_fits(
    repo: DataRepository, message_id: str, action: RoutingAction
) -> bool:
    """Whether a cited message's recorded reaction supports ``action``."""
    event = repo.get_message_event(message_id)
    if event is None:
        return False
    if action is RoutingAction.MUTE:
        return event.is_negative_signal
    if action is RoutingAction.NOTIFY:
        return event.message_opened or event.message_replied
    return event.message_opened or not event.is_negative_signal


def format_misses(report: EvaluationReport, limit: int = 10) -> tuple[str, ...]:
    """Return one line per action mismatch, for diagnosis.

    Args:
        report: A completed evaluation.
        limit: Maximum lines returned.
    """
    return tuple(
        f"{outcome.message_id:<16} expected {outcome.expected_action:<7} "
        f"got {outcome.predicted_action:<7} "
        f"(type expected {outcome.expected_type}, got {outcome.predicted_type}, "
        f"confidence {outcome.confidence:.2f})"
        for outcome in report.action_misses[:limit]
    )

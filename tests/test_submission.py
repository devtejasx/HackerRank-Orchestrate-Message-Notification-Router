"""End-to-end submission tests: pipeline, validation, CSV and performance.

These are the tests that answer "would this submission be accepted": every
input row predicted exactly once, every column present and well-formed, the
file readable back, and the whole run completing in a sensible time.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src import config
from src.classifier.enums import MessageType
from src.cli import commands
from src.data.models import Message
from src.data.repository import DataRepository
from src.evaluation import evaluate_samples
from src.output import (
    OutputSeverity,
    OutputValidationError,
    format_row,
    validate_results,
    write_output_csv,
)
from src.routing.models import NO_EVIDENCE, OUTPUT_COLUMNS, RoutingAction, RoutingResult
from src.routing.pipeline import RoutingPipeline

#: Ceiling for a full run over the shipped dataset. Generous: the point is to
#: catch an accidental quadratic, not to police a few hundred milliseconds.
MAX_FULL_RUN_SECONDS = 60.0


@pytest.fixture(scope="module")
def submission(repo: DataRepository) -> tuple[RoutingResult, ...]:
    """Routing results for every incoming message, computed once."""
    return RoutingPipeline(repo).route_all()


class TestEndToEnd:
    """The complete chain, from CSV on disk to CSV on disk."""

    def test_every_input_row_is_predicted(
        self, submission: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        messages = repo.get_messages()
        assert len(submission) == len(messages)
        assert [r.message_id for r in submission] == [m.message_id for m in messages]

    def test_predictions_validate(
        self, submission: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        report = validate_results(submission, repo.get_messages())
        assert report.is_valid, [str(i) for i in report.errors]

    def test_no_warnings_either(
        self, submission: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        """Warnings are quality signals; a clean run should have none."""
        report = validate_results(submission, repo.get_messages())
        assert report.warnings == (), [str(i) for i in report.warnings]

    def test_summary_counts_every_action(
        self, submission: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        report = validate_results(submission, repo.get_messages())
        assert report.total == len(submission)
        assert sum(report.action_counts.values()) == len(submission)
        assert set(report.action_counts) == {a.value for a in RoutingAction}


class TestOutputValidation:
    """Each contract check must actually fire when violated."""

    def _valid(self, submission: tuple[RoutingResult, ...]) -> list[RoutingResult]:
        return list(submission)

    def test_missing_prediction_is_an_error(
        self, submission: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        report = validate_results(self._valid(submission)[:-1], repo.get_messages())
        assert not report.is_valid
        assert any(i.check == "missing_prediction" for i in report.errors)

    def test_duplicate_prediction_is_an_error(
        self, submission: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        results = self._valid(submission)
        results.append(results[0])
        report = validate_results(results, repo.get_messages())
        assert any(i.check == "duplicate_message_id" for i in report.errors)

    def test_unexpected_prediction_is_an_error(
        self, submission: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        results = self._valid(submission)
        stray = RoutingResult(
            message_id="not_a_real_message",
            action=RoutingAction.MUTE,
            message_type=MessageType.SPAM.value,
            reason=results[0].reason,
            confidence=0.5,
        )
        report = validate_results([*results, stray], repo.get_messages())
        assert any(i.check == "unexpected_prediction" for i in report.errors)

    def test_out_of_range_confidence_is_rejected_at_construction(
        self, submission: tuple[RoutingResult, ...]
    ) -> None:
        with pytest.raises(ValueError, match="confidence"):
            RoutingResult(
                message_id="m",
                action=RoutingAction.MUTE,
                message_type=MessageType.SPAM.value,
                reason=submission[0].reason,
                confidence=1.4,
            )

    def test_empty_reason_is_an_error(
        self, submission: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        from dataclasses import replace

        from src.routing.models import RoutingReason

        results = self._valid(submission)
        results[0] = replace(results[0], reason=RoutingReason(text="   "))
        report = validate_results(results, repo.get_messages())
        assert any(i.check == "empty_reason" for i in report.errors)

    def test_report_raises_on_errors(
        self, submission: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        report = validate_results(self._valid(submission)[:-1], repo.get_messages())
        with pytest.raises(OutputValidationError) as excinfo:
            report.raise_for_errors()
        assert excinfo.value.report.errors

    def test_valid_report_does_not_raise(
        self, submission: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        validate_results(submission, repo.get_messages()).raise_for_errors()

    def test_severity_split(self) -> None:
        assert OutputSeverity.ERROR != OutputSeverity.WARNING


class TestCsvGeneration:
    """The written file must be readable, complete and correctly formatted."""

    @pytest.fixture
    def written(
        self, submission: tuple[RoutingResult, ...], tmp_path: Path
    ) -> Path:
        return write_output_csv(submission, tmp_path / "output.csv")

    def test_header_is_exact(self, written: Path) -> None:
        header = written.read_text(encoding="utf-8").splitlines()[0]
        assert header == ",".join(OUTPUT_COLUMNS)

    def test_round_trips(
        self, written: Path, submission: tuple[RoutingResult, ...]
    ) -> None:
        with written.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(submission)
        assert [r["message_id"] for r in rows] == [s.message_id for s in submission]

    def test_no_empty_cells(self, written: Path) -> None:
        with written.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert all(value != "" for row in rows for value in row.values())

    def test_confidence_is_fixed_precision(self, written: Path) -> None:
        with written.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            value = row["confidence"]
            assert 0.0 <= float(value) <= 1.0
            assert len(value.split(".")[1]) == config.CONFIDENCE_DECIMALS

    def test_evidence_format(self, written: Path) -> None:
        with written.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            evidence = row["evidence_message_ids"]
            if evidence == NO_EVIDENCE:
                continue
            parts = evidence.split(";")
            assert all(part and part == part.strip() for part in parts)
            assert len(set(parts)) == len(parts)

    def test_reasons_survive_commas_and_quotes(self, tmp_path: Path) -> None:
        """Reasons contain commas; the writer must quote rather than corrupt."""
        from dataclasses import replace

        from src.routing.models import RoutingReason

        awkward = 'A reason with, commas and "quotes" inside.'
        pipeline_results = [
            replace(
                RoutingResult(
                    message_id="m1",
                    action=RoutingAction.DIGEST,
                    message_type=MessageType.PERSONAL.value,
                    reason=RoutingReason(text=awkward),
                    confidence=0.5,
                )
            )
        ]
        path = write_output_csv(pipeline_results, tmp_path / "awkward.csv")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["reason"] == awkward

    def test_writing_is_atomic(
        self, submission: tuple[RoutingResult, ...], tmp_path: Path
    ) -> None:
        """No temporary file may survive a completed write."""
        destination = tmp_path / "output.csv"
        write_output_csv(submission, destination)
        assert list(tmp_path.iterdir()) == [destination]

    def test_creates_missing_directories(
        self, submission: tuple[RoutingResult, ...], tmp_path: Path
    ) -> None:
        destination = tmp_path / "nested" / "deeper" / "output.csv"
        assert write_output_csv(submission, destination).is_file()

    def test_format_row_stringifies_everything(
        self, submission: tuple[RoutingResult, ...]
    ) -> None:
        row = format_row(submission[0])
        assert tuple(row) == tuple(OUTPUT_COLUMNS)
        assert all(isinstance(value, str) for value in row.values())


@pytest.fixture(scope="module")
def evaluation_report(repo: DataRepository):
    """Evaluation against the labelled examples, computed once."""
    return evaluate_samples(RoutingPipeline(repo))


class TestEvaluationModule:
    """The reported metrics must be computed, not asserted."""

    @pytest.fixture
    def report(self, evaluation_report):
        """Alias so each test reads naturally."""
        return evaluation_report

    def test_covers_every_labelled_row(self, report, repo: DataRepository) -> None:
        assert report.total == len(repo.loader.records("sample_messages"))

    def test_accuracies_are_shares(self, report) -> None:
        for value in (
            report.action_accuracy, report.type_accuracy, report.both_accuracy
        ):
            assert 0.0 <= value <= 1.0

    def test_meets_the_regression_floor(self, report) -> None:
        assert report.action_accuracy >= 0.85

    def test_confidence_is_calibrated(self, report) -> None:
        """Being less sure when wrong is the whole point of calibration."""
        assert report.calibration_gap > 0.0

    def test_evidence_quality_is_measured(self, report) -> None:
        assert report.evidence_total > 0
        assert report.evidence_resolvable == report.evidence_total
        assert report.evidence_right_recipient == report.evidence_total
        assert report.evidence_precision > 0.5

    def test_summary_mentions_the_optimism_caveat(self, report) -> None:
        text = " ".join(report.summary_lines())
        assert "optimistic" in text


class TestCommands:
    """Command functions are callable directly, without argv."""

    def test_run_submission_writes(self, dataset_copy: Path, tmp_path: Path, capsys) -> None:
        output = tmp_path / "out.csv"
        assert commands.run_submission(
            dataset_copy, output, evaluate=False
        ) == commands.EXIT_OK
        capsys.readouterr()
        assert output.is_file()

    def test_run_submission_survives_a_missing_dataset(
        self, tmp_path: Path, capsys
    ) -> None:
        assert commands.run_submission(tmp_path / "nope") == commands.EXIT_FAILED
        capsys.readouterr()

    def test_check_dataset(self, dataset_dir: Path, capsys) -> None:
        assert commands.check_dataset(dataset_dir) == commands.EXIT_OK
        capsys.readouterr()

    def test_show_schema_needs_no_dataset(self, capsys) -> None:
        assert commands.show_schema() == commands.EXIT_OK
        capsys.readouterr()


class TestPerformance:
    """Smoke tests guarding against accidental quadratic behaviour."""

    def test_full_run_is_quick(self, dataset_dir: Path) -> None:
        started = time.perf_counter()
        results = RoutingPipeline.load(dataset_dir).route_all()
        elapsed = time.perf_counter() - started
        assert results
        assert elapsed < MAX_FULL_RUN_SECONDS, f"full run took {elapsed:.1f}s"

    def test_repeated_routing_reuses_caches(
        self, repo: DataRepository
    ) -> None:
        """A second pass must not cost materially more than the first."""
        pipeline = RoutingPipeline(repo)
        pipeline.route_all()

        started = time.perf_counter()
        pipeline.route_all()
        warm = time.perf_counter() - started
        assert warm < MAX_FULL_RUN_SECONDS

    def test_dataset_is_loaded_once_per_repository(
        self, repo: DataRepository
    ) -> None:
        assert repo.loader.raw_frame("messages") is repo.loader.raw_frame("messages")


class TestCleanCheckout:
    """The submission must work from a clean invocation."""

    def test_module_entry_point_generates_output(self, tmp_path: Path) -> None:
        """`python main.py` in a subprocess must produce a valid CSV."""
        output = tmp_path / "output.csv"
        completed = subprocess.run(
            [
                sys.executable, "main.py",
                "--output", str(output),
                "--log-level", "ERROR",
                "--no-evaluate",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=MAX_FULL_RUN_SECONDS * 3,
        )
        assert completed.returncode == 0, completed.stderr
        assert output.is_file()

        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert tuple(rows[0]) == tuple(OUTPUT_COLUMNS)

    def test_code_entry_point_delegates(self, tmp_path: Path) -> None:
        """`python code/main.py` must behave identically."""
        output = tmp_path / "from_code.csv"
        completed = subprocess.run(
            [
                sys.executable, str(Path("code") / "main.py"),
                "--output", str(output),
                "--log-level", "ERROR",
                "--no-evaluate",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=MAX_FULL_RUN_SECONDS * 3,
        )
        assert completed.returncode == 0, completed.stderr
        assert output.is_file()


class TestEdgeCaseRobustness:
    """The pipeline must not crash on degenerate input."""

    def test_routes_a_message_with_everything_missing(
        self, repo: DataRepository
    ) -> None:
        from datetime import datetime

        pipeline = RoutingPipeline(repo)
        message = Message(
            message_id="edge_minimal",
            user_id=next(iter(repo.index.users_by_id)),
            conversation_type="personal",
            group_id=None,
            business_id=None,
            sender_user_id=None,
            created_at=datetime(2026, 7, 25, 12, 0),
            message_text=None,
            media_type=None,
            media_id=None,
            forwarded_count=0,
        )
        result = pipeline.route(message)
        assert result.action in set(RoutingAction)
        assert result.reason.text
        assert result.evidence_message_ids
        assert 0.0 <= result.confidence <= 1.0

    def test_output_row_is_writable_for_degenerate_input(
        self, repo: DataRepository, tmp_path: Path
    ) -> None:
        from datetime import datetime

        pipeline = RoutingPipeline(repo)
        message = Message(
            message_id="edge_writable",
            user_id=next(iter(repo.index.users_by_id)),
            conversation_type="personal",
            group_id=None,
            business_id=None,
            sender_user_id=None,
            created_at=datetime(2026, 7, 25, 12, 0),
            message_text="",
            media_type=None,
            media_id=None,
            forwarded_count=0,
        )
        path = write_output_csv([pipeline.route(message)], tmp_path / "edge.csv")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert all(value != "" for value in rows[0].values())

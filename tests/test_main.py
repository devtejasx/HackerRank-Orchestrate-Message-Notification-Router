"""Tests for the :mod:`main` entry point and its command dispatch."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import main


class TestDefaultRun:
    """With no arguments, main.py must produce the submission."""

    def test_writes_output_csv(
        self, dataset_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "output.csv"
        exit_code = main.main(
            ["--dataset", str(dataset_copy), "--output", str(output),
             "--log-level", "ERROR", "--no-evaluate"]
        )
        capsys.readouterr()
        assert exit_code == 0
        assert output.is_file()
        assert output.read_text(encoding="utf-8").splitlines()[0] == (
            "message_id,action,message_type,reason,confidence,evidence_message_ids"
        )

    def test_prints_the_required_summary(
        self, dataset_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main.main(
            ["--dataset", str(dataset_copy), "--output", str(tmp_path / "o.csv"),
             "--log-level", "ERROR", "--no-evaluate"]
        )
        out = capsys.readouterr().out
        assert "messages processed" in out
        for action in ("notify count", "digest count", "mute count"):
            assert action in out

    def test_no_write_leaves_disk_untouched(
        self, dataset_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "absent.csv"
        exit_code = main.main(
            ["--dataset", str(dataset_copy), "--output", str(output),
             "--log-level", "ERROR", "--no-write", "--no-evaluate"]
        )
        capsys.readouterr()
        assert exit_code == 0
        assert not output.exists()

    def test_includes_evaluation_by_default(
        self, dataset_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main.main(
            ["--dataset", str(dataset_copy), "--output", str(tmp_path / "o.csv"),
             "--log-level", "ERROR"]
        )
        out = capsys.readouterr().out
        assert "EVALUATION AGAINST LABELLED EXAMPLES" in out
        assert "action agreement" in out

    def test_missing_dataset_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(tmp_path / "absent"), "--log-level", "ERROR"]
        ) == 1
        capsys.readouterr()

    def test_broken_dataset_exits_nonzero(
        self, dataset_copy: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (dataset_copy / "users.csv").unlink()
        assert main.main(["--dataset", str(dataset_copy), "--log-level", "ERROR"]) == 1
        capsys.readouterr()


class TestModes:
    """Each mode flag dispatches to its own command."""

    def test_schema_only(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main.main(["--schema-only"]) == 0
        out = capsys.readouterr().out
        assert "DATASET SCHEMA" in out
        assert "PHASE 1 - DATA LAYER" not in out

    def test_data_only(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR", "--data-only"]
        ) == 0
        out = capsys.readouterr().out
        assert "PHASE 1 - DATA LAYER" in out
        assert "PHASE 2" not in out

    def test_evaluate(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR", "--evaluate"]
        ) == 0
        out = capsys.readouterr().out
        assert "action agreement" in out
        assert "calibration gap" in out

    def test_inspect_writes_nothing(
        self, dataset_copy: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output_csv = dataset_copy / "output.csv"
        before = output_csv.read_bytes()
        assert main.main(
            ["--dataset", str(dataset_copy), "--log-level", "ERROR", "--inspect"]
        ) == 0
        capsys.readouterr()
        assert output_csv.read_bytes() == before

    def test_message_selection_implies_inspect(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR",
             "--message", "msg_091"]
        ) == 0
        out = capsys.readouterr().out
        assert "Message msg_091" in out
        assert "Routing decision (Phase 4)" in out

    def test_unknown_message_id_exits(self, dataset_dir: Path) -> None:
        with pytest.raises(SystemExit):
            main.main(
                ["--dataset", str(dataset_dir), "--log-level", "ERROR",
                 "--message", "does_not_exist"]
            )

    def test_limit_selects_a_prefix(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR", "--limit", "4"]
        ) == 0
        assert "4 message(s) inspected" in capsys.readouterr().out

    def test_all_prints_summaries(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR", "--all"]
        ) == 0
        out = capsys.readouterr().out
        assert "PHASE 2 - CLASSIFICATION SUMMARY" in out
        assert "PHASE 4 - ROUTING SUMMARY" in out

    def test_no_personalize_stops_after_phase_two(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR",
             "--inspect", "--no-personalize"]
        ) == 0
        out = capsys.readouterr().out
        assert "PHASE 2 - FEATURES AND CLASSIFICATION" in out
        assert "Routing signals" not in out

    def test_no_route_stops_after_phase_three(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR",
             "--inspect", "--no-route"]
        ) == 0
        out = capsys.readouterr().out
        assert "Routing signals" in out
        assert "Routing decision (Phase 4)" not in out


class TestArguments:
    """Argument parsing contract."""

    def test_defaults(self) -> None:
        args = main.parse_args([])
        assert args.dataset is None
        assert args.output is None
        assert args.strict is False
        assert args.inspect is False
        assert args.no_write is False
        assert args.message is None

    def test_selection_flags_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            main.parse_args(["--all", "--limit", "3"])

    def test_rejects_unknown_log_level(self) -> None:
        with pytest.raises(SystemExit):
            main.parse_args(["--log-level", "LOUD"])

    def test_short_message_flag(self) -> None:
        assert main.parse_args(["-m", "msg_001"]).message == "msg_001"


class TestInspectOutput:
    """The diagnostic view must show real data, not placeholders."""

    def test_reports_media_bytes_on_disk(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main.main(["--dataset", str(dataset_dir), "--log-level", "ERROR", "--inspect"])
        out = capsys.readouterr().out
        assert "MISSING" not in out
        assert "bytes" in out

    def test_shows_every_routing_signal(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR",
             "--message", "msg_091"]
        )
        out = capsys.readouterr().out
        for signal_name in (
            "sender_priority", "business_priority", "group_priority",
            "relationship_strength", "historical_importance",
            "engagement_modifier", "fatigue_modifier", "risk_modifier",
            "trust_modifier", "urgency_modifier",
        ):
            assert signal_name in out

    def test_shows_the_submission_row(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR",
             "--message", "msg_091"]
        )
        out = capsys.readouterr().out
        assert "Submission row" in out
        for column in (
            "message_id", "action", "message_type",
            "reason", "confidence", "evidence_message_ids",
        ):
            assert column in out

    def test_reports_an_action(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR",
             "--message", "msg_091"]
        )
        out = capsys.readouterr().out
        assert "ACTION" in out
        assert re.search(r"\b(notify|digest|mute)\b", out, re.IGNORECASE)

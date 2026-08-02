"""Tests for the :mod:`main` entry point."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import main


class TestCli:
    """Argument parsing and exit codes."""

    def test_schema_only_skips_loading(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main.main(["--schema-only"]) == 0
        out = capsys.readouterr().out
        assert "DATASET SCHEMA" in out
        assert "PHASE 1 - DATA LAYER" not in out

    def test_data_only_stops_before_phase_two(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR", "--data-only"]
        ) == 0
        out = capsys.readouterr().out
        assert "PHASE 1 - DATA LAYER" in out
        assert "PHASE 2" not in out

    def test_full_run_succeeds(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(["--dataset", str(dataset_dir), "--log-level", "ERROR"]) == 0
        out = capsys.readouterr().out
        for section in (
            "PHASE 1 - DATA LAYER",
            "PHASE 1 - REPOSITORY LOOKUPS",
            "FEATURES, CLASSIFICATION, SIGNALS AND ROUTING",
            "Routing decision (Phase 4)",
            "RESULT",
        ):
            assert section in out
        assert "no exceptions raised" in out

    def test_no_route_stops_after_phase_three(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR", "--no-route"]
        ) == 0
        out = capsys.readouterr().out
        assert "Routing signals" in out
        assert "Routing decision (Phase 4)" not in out

    def test_no_personalize_stops_after_phase_two(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR", "--no-personalize"]
        ) == 0
        out = capsys.readouterr().out
        assert "PHASE 2 - FEATURES AND CLASSIFICATION" in out
        assert "Routing signals" not in out

    def test_single_message_report(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR",
             "--message", "msg_005"]
        ) == 0
        out = capsys.readouterr().out
        assert "Message msg_005" in out
        assert "message_type" in out
        assert "confidence" in out

    def test_unknown_message_id_exits(self, dataset_dir: Path) -> None:
        with pytest.raises(SystemExit):
            main.main(
                ["--dataset", str(dataset_dir), "--log-level", "ERROR",
                 "--message", "does_not_exist"]
            )

    def test_all_prints_summary(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR", "--all"]
        ) == 0
        out = capsys.readouterr().out
        assert "PHASE 2 - CLASSIFICATION SUMMARY" in out
        assert "110 message(s) analysed" in out

    def test_limit_selects_a_prefix(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR", "--limit", "4"]
        ) == 0
        assert "4 message(s) analysed" in capsys.readouterr().out

    def test_missing_dataset_exits_nonzero(self, tmp_path: Path) -> None:
        assert main.main(
            ["--dataset", str(tmp_path / "absent"), "--log-level", "ERROR"]
        ) == 1

    def test_broken_dataset_exits_nonzero(self, dataset_copy: Path) -> None:
        (dataset_copy / "users.csv").unlink()
        assert main.main(
            ["--dataset", str(dataset_copy), "--log-level", "ERROR"]
        ) == 1

    def test_defaults(self) -> None:
        args = main.parse_args([])
        assert args.dataset is None
        assert args.strict is False
        assert args.no_validate is False
        assert args.schema_only is False
        assert args.all is False
        assert args.message is None

    def test_selection_flags_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            main.parse_args(["--all", "--limit", "3"])

    def test_rejects_unknown_log_level(self) -> None:
        with pytest.raises(SystemExit):
            main.parse_args(["--log-level", "LOUD"])


class TestOutputContent:
    """The demo must report real data, not placeholders."""

    def test_reports_media_bytes_on_disk(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main.main(["--dataset", str(dataset_dir), "--log-level", "ERROR"])
        out = capsys.readouterr().out
        assert "MISSING" not in out
        assert "bytes" in out

    def test_reports_features_and_classification(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR",
             "--message", "msg_091"]
        )
        out = capsys.readouterr().out
        for label in ("Features", "Keywords", "Classification", "reason", "scores"):
            assert label in out

    def test_does_not_write_output_csv(
        self, dataset_copy: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Output generation belongs to a later phase.

        Asserted against the file itself rather than the printed text, which
        legitimately contains words like "interactions".
        """
        output_csv = dataset_copy / "output.csv"
        before = output_csv.read_bytes()

        main.main(["--dataset", str(dataset_copy), "--log-level", "ERROR", "--all"])

        assert output_csv.read_bytes() == before
        capsys.readouterr()

    def test_reports_both_a_category_and_an_action(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Phase 4 assigns an action; the category from Phase 2 is kept too."""
        main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR",
             "--message", "msg_091"]
        )
        out = capsys.readouterr().out
        assert "message_type" in out
        assert "ACTION" in out

        action_word = re.compile(r"\b(notify|digest|mute)\b", re.IGNORECASE)
        assert action_word.search(out)

    def test_prints_the_submission_row(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every output column must be visible in the demo."""
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

    def test_still_does_not_write_output_csv(
        self, dataset_copy: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Routing decides; exporting is Phase 5 and must not happen yet."""
        output_csv = dataset_copy / "output.csv"
        before = output_csv.read_bytes()
        main.main(["--dataset", str(dataset_copy), "--log-level", "ERROR", "--all"])
        assert output_csv.read_bytes() == before
        capsys.readouterr()

    def test_shows_routing_signals(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Phase 3 output must show every signal with score and confidence."""
        main.main(
            ["--dataset", str(dataset_dir), "--log-level", "ERROR",
             "--message", "msg_091"]
        )
        out = capsys.readouterr().out
        assert "Routing signals" in out
        for signal_name in (
            "sender_priority", "business_priority", "group_priority",
            "relationship_strength", "historical_importance",
            "engagement_modifier", "fatigue_modifier", "risk_modifier",
            "trust_modifier", "urgency_modifier",
        ):
            assert signal_name in out

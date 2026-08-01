"""Tests for the :mod:`main` smoke-test entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

import main


class TestCli:
    """Argument parsing and exit codes."""

    def test_schema_only_skips_loading(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main.main(["--schema-only"]) == 0
        out = capsys.readouterr().out
        assert "DATASET SCHEMA" in out
        assert "LOAD SUMMARY" not in out

    def test_full_run_succeeds(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main.main(["--dataset", str(dataset_dir), "--log-level", "ERROR"]) == 0
        out = capsys.readouterr().out
        for section in ("DATASET SCHEMA", "LOAD SUMMARY", "SMOKE LOOKUPS", "RESULT"):
            assert section in out
        assert "Hour 1 data layer is ready" in out

    def test_missing_dataset_exits_nonzero(self, tmp_path: Path) -> None:
        assert main.main(["--dataset", str(tmp_path / "absent"), "--log-level", "ERROR"]) == 1

    def test_broken_dataset_exits_nonzero(self, dataset_copy: Path) -> None:
        (dataset_copy / "users.csv").unlink()
        assert main.main(["--dataset", str(dataset_copy), "--log-level", "ERROR"]) == 1

    def test_defaults(self) -> None:
        args = main.parse_args([])
        assert args.dataset is None
        assert args.strict is False
        assert args.no_validate is False
        assert args.schema_only is False

    def test_rejects_unknown_log_level(self) -> None:
        with pytest.raises(SystemExit):
            main.parse_args(["--log-level", "LOUD"])


class TestSmokeOutput:
    """The smoke lookups must exercise real records, not placeholders."""

    def test_reports_media_bytes_on_disk(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main.main(["--dataset", str(dataset_dir), "--log-level", "ERROR"])
        out = capsys.readouterr().out
        assert "MISSING" not in out
        assert "bytes" in out

    def test_unknown_ids_degrade_gracefully(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main.main(["--dataset", str(dataset_dir), "--log-level", "ERROR"])
        out = capsys.readouterr().out
        assert "Unknown ids" in out

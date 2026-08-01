"""Tests for :mod:`src.utils.helpers`."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
import pytest

from src.utils import helpers


class TestIsMissing:
    """Missing-value detection across Python and pandas sentinels."""

    @pytest.mark.parametrize(
        "value",
        [None, float("nan"), pd.NA, pd.NaT, "", "   ", "nan", "NaN", "<NA>", "NaT"],
    )
    def test_detects_missing(self, value: object) -> None:
        assert helpers.is_missing(value) is True

    @pytest.mark.parametrize(
        "value",
        [0, 0.0, False, "0", "x", "none", "null", "na", datetime(2026, 1, 1), date(2026, 1, 1)],
    )
    def test_detects_present(self, value: object) -> None:
        assert helpers.is_missing(value) is False

    def test_none_is_not_a_null_token(self) -> None:
        """The literal 'none' is a real sentinel in evidence_message_ids."""
        assert helpers.safe_text("none") == "none"

    def test_handle_missing_substitutes_default(self) -> None:
        assert helpers.handle_missing("", "fallback") == "fallback"
        assert helpers.handle_missing("value", "fallback") == "value"


class TestScalarCoercion:
    """safe_* coercion never raises and honours its default."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("12", 12), (" 12 ", 12), (12.0, 12), ("12.0", 12), (True, 1)],
    )
    def test_safe_int_parses(self, value: object, expected: int) -> None:
        assert helpers.safe_int(value) == expected

    @pytest.mark.parametrize("value", ["", None, pd.NA, "many", "12abc"])
    def test_safe_int_falls_back(self, value: object) -> None:
        assert helpers.safe_int(value, -1) == -1

    def test_safe_float_parses(self) -> None:
        assert helpers.safe_float("0.89") == pytest.approx(0.89)
        assert helpers.safe_float("bad", 0.0) == 0.0

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("1", True), ("0", False), (1.0, True), ("yes", True), ("NO", False), (True, True)],
    )
    def test_safe_bool_parses(self, value: object, expected: bool) -> None:
        assert helpers.safe_bool(value) is expected

    def test_safe_bool_falls_back(self) -> None:
        assert helpers.safe_bool("maybe", False) is False
        assert helpers.safe_bool(None) is None

    def test_safe_text_strips(self) -> None:
        assert helpers.safe_text("  hi  ") == "hi"
        assert helpers.safe_text("") is None
        assert helpers.safe_text("", "-") == "-"

    def test_normalize_string_collapses_whitespace(self) -> None:
        assert helpers.normalize_string("  Amazon   India ") == "amazon india"
        assert helpers.normalize_string(None) == ""

    def test_truncate_respects_limit(self) -> None:
        assert helpers.truncate("a" * 50, 20) == "a" * 17 + "..."
        assert helpers.truncate("short", 20) == "short"
        assert helpers.truncate("a\nb", 20) == "a b"


class TestTemporalParsing:
    """Both dataset timestamp layouts parse; junk returns None."""

    def test_parse_timestamp_minute_precision(self) -> None:
        assert helpers.parse_timestamp("2026-07-30 22:19") == datetime(2026, 7, 30, 22, 19)

    def test_parse_timestamp_accepts_date_only(self) -> None:
        assert helpers.parse_timestamp("2026-07-30") == datetime(2026, 7, 30)

    @pytest.mark.parametrize("value", ["", "not-a-date", "30-07-2026", None, pd.NaT])
    def test_parse_timestamp_rejects_junk(self, value: object) -> None:
        assert helpers.parse_timestamp(value) is None

    def test_parse_date(self) -> None:
        assert helpers.parse_date("2023-06-13") == date(2023, 6, 13)
        assert helpers.parse_date("2026-05-23 17:46") == date(2026, 5, 23)
        assert helpers.parse_date("nope") is None

    def test_parse_dnd_window(self) -> None:
        assert helpers.parse_dnd_window("22:00-07:00") == (time(22, 0), time(7, 0))

    @pytest.mark.parametrize("value", ["", "22:00", "bad-window", "22:00-07:00-08:00", None])
    def test_parse_dnd_window_rejects_junk(self, value: object) -> None:
        assert helpers.parse_dnd_window(value) is None


class TestFilesystem:
    """Dataset-relative path resolution."""

    def test_resolve_relative_against_base(self, tmp_path: Path) -> None:
        assert helpers.resolve_dataset_path("media/a.jpg", tmp_path) == tmp_path / "media/a.jpg"

    def test_absolute_path_passes_through(self, tmp_path: Path) -> None:
        absolute = tmp_path / "a.jpg"
        assert helpers.resolve_dataset_path(absolute, tmp_path) == absolute

    def test_file_exists(self, tmp_path: Path) -> None:
        (tmp_path / "present.txt").write_text("x", encoding="utf-8")
        assert helpers.file_exists("present.txt", tmp_path) is True
        assert helpers.file_exists("absent.txt", tmp_path) is False


class TestCollectionShaping:
    """index_by / group_by behaviour, including the nullable-key contract."""

    def test_index_by_last_wins(self) -> None:
        pairs = [("a", 1), ("b", 2), ("a", 3)]
        assert helpers.index_by(pairs, lambda p: p[0]) == {"a": ("a", 3), "b": ("b", 2)}

    def test_group_by_collects(self) -> None:
        pairs = [("a", 1), ("b", 2), ("a", 3)]
        assert helpers.group_by(pairs, lambda p: p[0]) == {
            "a": [("a", 1), ("a", 3)],
            "b": [("b", 2)],
        }

    def test_group_by_skips_none_keys(self) -> None:
        """Nullable foreign keys must not create a None bucket."""
        pairs = [("a", 1), (None, 2)]
        assert helpers.group_by(pairs, lambda p: p[0]) == {"a": [("a", 1)]}

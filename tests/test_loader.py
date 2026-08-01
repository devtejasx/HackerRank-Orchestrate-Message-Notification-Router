"""Tests for :mod:`src.data.loader` and :mod:`src.data.schema`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import schema
from src.data.loader import DataLoader, MissingDatasetFileError
from src.data.models import MODEL_BY_TABLE

#: Row counts of the shipped dataset, established by profiling the CSVs.
EXPECTED_COUNTS = {
    "users": 54,
    "groups": 23,
    "group_members": 401,
    "business_accounts": 110,
    "user_business_history": 106,
    "messages": 110,
    "message_history": 412,
    "message_events": 412,
    "images": 20,
    "voice_notes": 13,
    "daily_notification_summary": 756,
    "sample_messages": 30,
}


class TestSchemaRegistry:
    """The registry must stay internally consistent."""

    def test_every_table_has_a_model(self) -> None:
        assert set(schema.TABLES) == set(MODEL_BY_TABLE)

    def test_primary_keys_are_declared_columns(self) -> None:
        for spec in schema.TABLES.values():
            assert set(spec.primary_key) <= set(spec.column_names), spec.name

    def test_foreign_keys_reference_known_tables_and_columns(self) -> None:
        for spec in schema.TABLES.values():
            for foreign_key in spec.foreign_keys:
                assert foreign_key.column in spec.column_names
                target = schema.TABLES[foreign_key.target_table]
                assert foreign_key.target_column in target.column_names

    def test_model_fields_match_declared_columns(self) -> None:
        """A drifting dataclass would silently drop a column."""
        from dataclasses import fields

        for name, spec in schema.TABLES.items():
            model_fields = {f.name for f in fields(MODEL_BY_TABLE[name])}
            assert model_fields == set(spec.column_names), name

    def test_get_spec_rejects_unknown_table(self) -> None:
        with pytest.raises(KeyError):
            schema.get_spec("no_such_table")

    def test_display_label_falls_back_to_name(self) -> None:
        assert schema.get_spec("users").display_label == "Users"
        assert schema.get_spec("message_history").display_label == "History"


class TestLoading:
    """Every CSV loads with the expected shape."""

    def test_load_all_returns_expected_counts(self, loader: DataLoader) -> None:
        assert loader.summary() == EXPECTED_COUNTS

    @pytest.mark.parametrize("table", sorted(EXPECTED_COUNTS))
    def test_raw_frame_has_declared_columns(self, loader: DataLoader, table: str) -> None:
        frame = loader.raw_frame(table)
        assert list(frame.columns) == list(schema.get_spec(table).column_names)

    def test_no_required_tables_missing(self, loader: DataLoader) -> None:
        assert loader.missing_required_tables() == ()

    def test_available_tables_covers_registry(self, loader: DataLoader) -> None:
        assert set(loader.available_tables()) == set(schema.TABLES)


class TestCaching:
    """Files are read once; every view is memoised."""

    def test_frames_and_records_are_cached(self, loader: DataLoader) -> None:
        assert loader.raw_frame("users") is loader.raw_frame("users")
        assert loader.frame("users") is loader.frame("users")
        assert loader.records("users") is loader.records("users")

    def test_file_is_read_only_once(self, dataset_copy: Path, monkeypatch) -> None:
        instance = DataLoader(dataset_copy)
        calls: list[Path] = []
        original = pd.read_csv

        def counting_read_csv(path, *args, **kwargs):
            calls.append(Path(path))
            return original(path, *args, **kwargs)

        monkeypatch.setattr(pd, "read_csv", counting_read_csv)
        for _ in range(3):
            instance.raw_frame("users")
            instance.frame("users")
            instance.records("users")

        assert calls.count(dataset_copy / "users.csv") == 1


class TestTypedFrames:
    """Logical types survive the round trip into pandas dtypes."""

    def test_integers_are_nullable(self, loader: DataLoader) -> None:
        assert loader.messages["forwarded_count"].dtype == "Int64"

    def test_booleans_are_nullable(self, loader: DataLoader) -> None:
        assert loader.group_members["group_muted_by_user"].dtype == "boolean"

    def test_timestamps_parse_without_loss(self, loader: DataLoader) -> None:
        raw = loader.raw_frame("messages")["created_at"]
        typed = loader.messages["created_at"]
        assert str(typed.dtype).startswith("datetime64")
        assert int(typed.isna().sum()) == int((raw == "").sum())

    def test_date_only_columns_parse(self, loader: DataLoader) -> None:
        assert int(loader.groups["created_at"].isna().sum()) == 0

    def test_blank_cells_become_na(self, loader: DataLoader) -> None:
        raw = loader.raw_frame("business_accounts")["official_domain"]
        typed = loader.business_accounts["official_domain"]
        assert int(typed.isna().sum()) == int((raw == "").sum()) > 0

    def test_raw_frame_keeps_blanks_as_empty_strings(self, loader: DataLoader) -> None:
        raw = loader.raw_frame("messages")
        assert not raw.isna().to_numpy().any()


class TestFailureModes:
    """Missing files fail loudly and specifically."""

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MissingDatasetFileError, match="Dataset directory not found"):
            DataLoader(tmp_path / "absent").load_all()

    def test_missing_required_file_raises(self, dataset_copy: Path) -> None:
        (dataset_copy / "users.csv").unlink()
        with pytest.raises(MissingDatasetFileError, match="users.csv"):
            DataLoader(dataset_copy).load_all()

    def test_missing_optional_file_is_tolerated(self, dataset_copy: Path) -> None:
        (dataset_copy / "sample_messages.csv").unlink()
        instance = DataLoader(dataset_copy)
        counts = instance.load_all()
        assert "sample_messages" not in counts
        assert instance.missing_required_tables() == ()

    def test_requesting_absent_table_raises(self, dataset_copy: Path) -> None:
        (dataset_copy / "sample_messages.csv").unlink()
        with pytest.raises(MissingDatasetFileError):
            DataLoader(dataset_copy).raw_frame("sample_messages")

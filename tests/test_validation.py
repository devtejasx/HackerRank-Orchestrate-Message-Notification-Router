"""Tests for :mod:`src.data.validation`.

Each check gets a purpose-built corruption, because a validator that never
fires is indistinguishable from one that does not work.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.data.loader import DataLoader
from src.data.validation import (
    DatasetValidationError,
    Severity,
    validate_dataset,
)

Mutator = Callable[[Path, Callable[[list[dict[str, str]], list[str]], tuple]], None]


def _validate(dataset: Path):
    """Validate ``dataset`` without raising, returning the report."""
    return validate_dataset(DataLoader(dataset), raise_on_error=False)


def _checks(report) -> set[str]:
    """Return the set of check names that fired."""
    return {issue.check for issue in report.issues}


class TestCleanDataset:
    """The shipped dataset is expected to be pristine."""

    def test_no_issues(self, dataset_dir: Path) -> None:
        report = _validate(dataset_dir)
        assert report.issues == [], [str(i) for i in report.issues]
        assert report.is_valid is True

    def test_repository_load_validates_by_default(self, repo) -> None:
        assert repo.validation_report is not None
        assert repo.validation_report.is_valid is True


class TestStructuralErrors:
    """Structural damage is an ERROR and blocks the load."""

    def test_duplicate_primary_key_warns_and_keeps_the_first_row(
        self, dataset_copy: Path, mutate_csv: Mutator
    ) -> None:
        # Reported, but not blocking: failing the load would turn a two-row
        # defect into a zero-row submission. The loader keeps the first
        # occurrence so the indexes stay one-to-one.
        mutate_csv(
            dataset_copy / "users.csv",
            lambda rows, fields: ([*rows, dict(rows[0])], fields),
        )
        report = _validate(dataset_copy)
        assert "duplicate_primary_key" in _checks(report)
        assert report.is_valid is True

        loader = DataLoader(dataset_copy)
        users = loader.records("users")
        assert len({user.user_id for user in users}) == len(users)
        assert loader.dropped_rows["users"] == 1

    def test_blank_primary_key(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def blank(rows, fields):
            rows[0]["user_id"] = ""
            return rows, fields

        mutate_csv(dataset_copy / "users.csv", blank)
        assert "blank_primary_key" in _checks(_validate(dataset_copy))

    def test_missing_column(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def drop_role(rows, fields):
            for row in rows:
                row.pop("role", None)
            return rows, [f for f in fields if f != "role"]

        mutate_csv(dataset_copy / "group_members.csv", drop_role)
        report = _validate(dataset_copy)
        assert "missing_columns" in _checks(report)

    def test_missing_column_skips_that_tables_content_checks(
        self, dataset_copy: Path, mutate_csv: Mutator
    ) -> None:
        """One bad column must not cascade into unrelated noise."""

        def drop_role(rows, fields):
            for row in rows:
                row.pop("role", None)
            return rows, [f for f in fields if f != "role"]

        mutate_csv(dataset_copy / "group_members.csv", drop_role)
        report = _validate(dataset_copy)
        member_checks = {i.check for i in report.issues if i.table == "group_members"}
        assert member_checks == {"missing_columns"}

    def test_empty_required_table(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        mutate_csv(dataset_copy / "users.csv", lambda _rows, fields: ([], fields))
        report = _validate(dataset_copy)
        assert "empty_table" in _checks(report)
        assert report.is_valid is False

    def test_empty_optional_table_is_only_a_warning(
        self, dataset_copy: Path, mutate_csv: Mutator
    ) -> None:
        # The media registries only map an attachment to a file on disk, so
        # losing one degrades the run rather than ending it.
        mutate_csv(dataset_copy / "images.csv", lambda _rows, fields: ([], fields))
        report = _validate(dataset_copy)
        assert "empty_table" in _checks(report)
        assert report.is_valid is True

    def test_missing_required_file(self, dataset_copy: Path) -> None:
        (dataset_copy / "users.csv").unlink()
        report = _validate(dataset_copy)
        assert "missing_file" in _checks(report)
        assert report.is_valid is False

    def test_missing_optional_file_is_only_a_warning(self, dataset_copy: Path) -> None:
        (dataset_copy / "sample_messages.csv").unlink()
        report = _validate(dataset_copy)
        assert report.is_valid is True
        assert any(i.check == "missing_file" and i.severity is Severity.WARNING
                   for i in report.issues)


class TestContentWarnings:
    """Content defects warn but do not block."""

    def test_unexpected_null(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def blank_count(rows, fields):
            rows[0]["messages_opened_30d"] = ""
            return rows, fields

        mutate_csv(dataset_copy / "users.csv", blank_count)
        report = _validate(dataset_copy)
        assert "unexpected_null" in _checks(report)
        assert report.is_valid is True

    def test_malformed_timestamp(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def break_time(rows, fields):
            rows[0]["created_at"] = "31/07/2026"
            return rows, fields

        mutate_csv(dataset_copy / "messages.csv", break_time)
        assert "malformed_timestamp" in _checks(_validate(dataset_copy))

    def test_malformed_date(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def break_date(rows, fields):
            rows[0]["created_at"] = "not-a-date"
            return rows, fields

        mutate_csv(dataset_copy / "groups.csv", break_date)
        assert "malformed_date" in _checks(_validate(dataset_copy))

    def test_malformed_int(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def break_int(rows, fields):
            rows[0]["messages_opened_30d"] = "many"
            return rows, fields

        mutate_csv(dataset_copy / "users.csv", break_int)
        assert "malformed_int" in _checks(_validate(dataset_copy))

    def test_unexpected_enum_value(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def break_enum(rows, fields):
            rows[0]["conversation_type"] = "carrier_pigeon"
            return rows, fields

        mutate_csv(dataset_copy / "messages.csv", break_enum)
        assert "unexpected_value" in _checks(_validate(dataset_copy))

    def test_unexpected_column(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def add_column(rows, fields):
            for row in rows:
                row["surprise"] = "x"
            return rows, [*fields, "surprise"]

        mutate_csv(dataset_copy / "message_events.csv", add_column)
        assert "unexpected_columns" in _checks(_validate(dataset_copy))

    def test_broken_reference(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def break_fk(rows, fields):
            target = next(r for r in rows if r["conversation_type"] == "group")
            target["group_id"] = "group_999"
            return rows, fields

        mutate_csv(dataset_copy / "messages.csv", break_fk)
        assert "broken_reference" in _checks(_validate(dataset_copy))


class TestDatasetInvariants:
    """conversation_type and media pairing rules."""

    def test_required_reference_missing(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def blank_group(rows, fields):
            target = next(r for r in rows if r["conversation_type"] == "group")
            target["group_id"] = ""
            return rows, fields

        mutate_csv(dataset_copy / "messages.csv", blank_group)
        assert "conversation_reference_missing" in _checks(_validate(dataset_copy))

    def test_forbidden_reference_present(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def add_group_to_business(rows, fields):
            target = next(r for r in rows if r["conversation_type"] == "business")
            target["group_id"] = "group_001"
            return rows, fields

        mutate_csv(dataset_copy / "messages.csv", add_group_to_business)
        assert "conversation_reference_unexpected" in _checks(_validate(dataset_copy))

    def test_half_declared_media(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def half_media(rows, fields):
            rows[0]["media_type"] = "image"
            rows[0]["media_id"] = ""
            return rows, fields

        mutate_csv(dataset_copy / "messages.csv", half_media)
        assert "media_pair_mismatch" in _checks(_validate(dataset_copy))

    @pytest.mark.parametrize(
        ("media_type", "media_id"), [("image", "img_999"), ("voice", "vn_999")]
    )
    def test_unknown_media_id(
        self, dataset_copy: Path, mutate_csv: Mutator, media_type: str, media_id: str
    ) -> None:
        def unknown_media(rows, fields):
            rows[0]["media_type"] = media_type
            rows[0]["media_id"] = media_id
            return rows, fields

        mutate_csv(dataset_copy / "messages.csv", unknown_media)
        assert "unknown_media_id" in _checks(_validate(dataset_copy))

    def test_missing_media_file(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def point_nowhere(rows, fields):
            rows[0]["file_path"] = "media/audio/absent.mp3"
            return rows, fields

        mutate_csv(dataset_copy / "voice_notes.csv", point_nowhere)
        assert "missing_media_file" in _checks(_validate(dataset_copy))

    def test_every_shipped_media_file_is_on_disk(self, dataset_dir: Path) -> None:
        report = _validate(dataset_dir)
        assert not [i for i in report.issues if i.check == "missing_media_file"]


class TestReportBehaviour:
    """Raising, strict mode and report shape."""

    def test_raises_on_structural_error(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def blank_key(rows, fields):
            rows[0]["user_id"] = ""
            return rows, fields

        mutate_csv(dataset_copy / "users.csv", blank_key)
        with pytest.raises(DatasetValidationError) as excinfo:
            validate_dataset(DataLoader(dataset_copy))
        assert excinfo.value.report.errors

    def test_does_not_raise_on_warnings_alone(
        self, dataset_copy: Path, mutate_csv: Mutator
    ) -> None:
        def blank_count(rows, fields):
            rows[0]["messages_opened_30d"] = ""
            return rows, fields

        mutate_csv(dataset_copy / "users.csv", blank_count)
        report = validate_dataset(DataLoader(dataset_copy))
        assert report.warnings and report.is_valid

    def test_strict_promotes_warnings_to_errors(
        self, dataset_copy: Path, mutate_csv: Mutator
    ) -> None:
        def blank_count(rows, fields):
            rows[0]["messages_opened_30d"] = ""
            return rows, fields

        mutate_csv(dataset_copy / "users.csv", blank_count)
        report = validate_dataset(DataLoader(dataset_copy), strict=True, raise_on_error=False)
        assert report.warnings == ()
        assert report.errors
        assert report.is_valid is False

    def test_issue_reports_count_and_examples(
        self, dataset_copy: Path, mutate_csv: Mutator
    ) -> None:
        def break_enum(rows, fields):
            rows[0]["conversation_type"] = "carrier_pigeon"
            return rows, fields

        mutate_csv(dataset_copy / "messages.csv", break_enum)
        issue = next(
            i for i in _validate(dataset_copy).issues if i.check == "unexpected_value"
        )
        assert issue.count == 1
        assert issue.examples == ("carrier_pigeon",)
        assert "carrier_pigeon" in str(issue)

    def test_tables_with_errors(self, dataset_copy: Path, mutate_csv: Mutator) -> None:
        def blank_key(rows, fields):
            rows[0]["user_id"] = ""
            return rows, fields

        mutate_csv(dataset_copy / "users.csv", blank_key)
        assert _validate(dataset_copy).tables_with_errors() == frozenset({"users"})

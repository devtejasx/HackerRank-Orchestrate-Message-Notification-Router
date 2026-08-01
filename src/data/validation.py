"""Dataset validation driven entirely by :mod:`src.data.schema`.

Two severities, and the distinction matters:

* :attr:`Severity.ERROR` - structural damage that makes the data layer
  unusable (absent file, absent column, duplicate primary key, empty required
  table). These raise :class:`DatasetValidationError`.
* :attr:`Severity.WARNING` - content defects worth knowing about that the rest
  of the pipeline can still work around (a broken reference, an unexpected
  enum value, an unparseable cell). These are logged, not raised, unless
  :data:`src.config.STRICT_VALIDATION` is enabled.

Every check reads the *raw* string frames, because coercion would otherwise
repair exactly the defects being looked for.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from src import config
from src.data import schema
from src.data.loader import DataLoader, DatasetError
from src.data.schema import ColumnType, TableSpec
from src.utils.helpers import file_exists, truncate

__all__ = [
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "DatasetValidationError",
    "validate_dataset",
]

_LOGGER = config.get_logger("validation")

#: Longest example string included in an issue, to keep logs readable.
_EXAMPLE_WIDTH = 60


class Severity(StrEnum):
    """How serious a validation finding is."""

    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single validation finding.

    Attributes:
        severity: Whether this blocks use of the dataset.
        table: Logical table the finding relates to.
        check: Short machine-readable check name, e.g. ``"duplicate_primary_key"``.
        message: Human-readable description.
        count: How many rows or values are affected.
        examples: Up to :data:`src.config.MAX_ISSUE_EXAMPLES` offending values.
    """

    severity: Severity
    table: str
    check: str
    message: str
    count: int = 0
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        head = f"[{self.severity.value}] {self.table}.{self.check}: {self.message}"
        if self.examples:
            return f"{head} (examples: {', '.join(self.examples)})"
        return head


@dataclass(slots=True)
class ValidationReport:
    """Aggregated findings for one validation run."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        """Record a finding."""
        self.issues.append(issue)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Findings that block use of the dataset."""
        return tuple(i for i in self.issues if i.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Findings that are safe to continue past."""
        return tuple(i for i in self.issues if i.severity is Severity.WARNING)

    @property
    def is_valid(self) -> bool:
        """Whether the dataset is free of blocking errors."""
        return not self.errors

    def tables_with_errors(self) -> frozenset[str]:
        """Logical names of tables carrying at least one error."""
        return frozenset(issue.table for issue in self.errors)

    def log(self) -> None:
        """Emit every finding at its matching log level."""
        for issue in self.issues:
            level = _LOGGER.error if issue.severity is Severity.ERROR else _LOGGER.warning
            level("%s", issue)
        if not self.issues:
            _LOGGER.info("Validation passed with no issues")
        else:
            _LOGGER.info(
                "Validation finished: %d error(s), %d warning(s)",
                len(self.errors),
                len(self.warnings),
            )

    def raise_for_errors(self) -> None:
        """Raise if any blocking error was recorded.

        Raises:
            DatasetValidationError: When :attr:`errors` is non-empty.
        """
        if self.errors:
            detail = "\n  ".join(str(issue) for issue in self.errors)
            raise DatasetValidationError(
                f"Dataset failed validation with {len(self.errors)} error(s):\n  {detail}",
                report=self,
            )


class DatasetValidationError(DatasetError):
    """Raised when the dataset has blocking structural problems.

    Attributes:
        report: The full report, so callers can inspect warnings too.
    """

    def __init__(self, message: str, report: ValidationReport) -> None:
        super().__init__(message)
        self.report = report


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def validate_dataset(
    loader: DataLoader,
    *,
    strict: bool | None = None,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Run every check against the dataset behind ``loader``.

    Checks run per table. If a table fails a structural check, its remaining
    checks are skipped so one missing column cannot produce a cascade of
    unrelated noise.

    Args:
        loader: Loader pointing at the dataset to validate.
        strict: Promote warnings to errors. Defaults to
            :data:`src.config.STRICT_VALIDATION`.
        raise_on_error: Whether to raise once blocking errors are found.

    Returns:
        The complete report, including warnings.

    Raises:
        DatasetValidationError: If blocking errors were found and
            ``raise_on_error`` is true.
    """
    is_strict = config.STRICT_VALIDATION if strict is None else strict
    report = ValidationReport()

    _check_required_files(loader, report)
    healthy = _check_table_structure(loader, report)

    for table in healthy:
        spec = schema.get_spec(table)
        frame = loader.raw_frame(table)
        _check_primary_key(spec, frame, report)
        _check_required_values(spec, frame, report)
        _check_value_parsing(loader, spec, frame, report)
        _check_allowed_values(spec, frame, report)

    _check_foreign_keys(loader, healthy, report)
    _check_conversation_invariants(loader, healthy, report)
    _check_media_references(loader, healthy, report)

    if is_strict:
        _promote_warnings(report)

    report.log()
    if raise_on_error:
        report.raise_for_errors()
    return report


def _promote_warnings(report: ValidationReport) -> None:
    """Rewrite every warning as an error, for strict mode."""
    report.issues = [
        i if i.severity is Severity.ERROR else _as_error(i) for i in report.issues
    ]


def _as_error(issue: ValidationIssue) -> ValidationIssue:
    """Return ``issue`` with ERROR severity."""
    return ValidationIssue(
        severity=Severity.ERROR,
        table=issue.table,
        check=issue.check,
        message=issue.message,
        count=issue.count,
        examples=issue.examples,
    )


# --------------------------------------------------------------------------- #
# Structural checks (ERROR)
# --------------------------------------------------------------------------- #


def _check_required_files(loader: DataLoader, report: ValidationReport) -> None:
    """Confirm every required CSV is present; warn about absent optional ones."""
    if not loader.dataset_dir.is_dir():
        report.add(
            ValidationIssue(
                Severity.ERROR,
                "dataset",
                "missing_directory",
                f"Dataset directory not found: {loader.dataset_dir}",
            )
        )
        return

    for name, spec in schema.TABLES.items():
        if loader.is_available(name):
            continue
        severity = Severity.ERROR if spec.required else Severity.WARNING
        report.add(
            ValidationIssue(
                severity,
                name,
                "missing_file",
                f"{'Required' if spec.required else 'Optional'} file not found: {spec.filename}",
            )
        )


def _check_table_structure(loader: DataLoader, report: ValidationReport) -> tuple[str, ...]:
    """Check columns and emptiness, returning tables safe for further checks.

    Args:
        loader: Loader to read from.
        report: Report to append findings to.

    Returns:
        Logical names of tables that are present, non-empty and fully columned.
    """
    healthy: list[str] = []
    for name, spec in schema.TABLES.items():
        if not loader.is_available(name):
            continue

        frame = loader.raw_frame(name)
        declared = set(spec.column_names)
        present = set(frame.columns)

        missing = tuple(sorted(declared - present))
        if missing:
            report.add(
                ValidationIssue(
                    Severity.ERROR,
                    name,
                    "missing_columns",
                    f"Declared columns absent from {spec.filename}: {list(missing)}",
                    count=len(missing),
                    examples=missing[: config.MAX_ISSUE_EXAMPLES],
                )
            )

        unexpected = tuple(sorted(present - declared))
        if unexpected:
            report.add(
                ValidationIssue(
                    Severity.WARNING,
                    name,
                    "unexpected_columns",
                    f"Columns in {spec.filename} not declared in the schema: {list(unexpected)}",
                    count=len(unexpected),
                    examples=unexpected[: config.MAX_ISSUE_EXAMPLES],
                )
            )

        if frame.empty:
            report.add(
                ValidationIssue(
                    Severity.ERROR if spec.required else Severity.WARNING,
                    name,
                    "empty_table",
                    f"{spec.filename} contains a header but no rows",
                )
            )
            continue

        if not missing:
            healthy.append(name)

    return tuple(healthy)


def _check_primary_key(
    spec: TableSpec, frame: pd.DataFrame, report: ValidationReport
) -> None:
    """Confirm the primary key is present and unique on every row."""
    key_columns = list(spec.primary_key)
    subset = frame[key_columns]

    blank_rows = (subset == "").any(axis=1)
    if bool(blank_rows.any()):
        report.add(
            ValidationIssue(
                Severity.ERROR,
                spec.name,
                "blank_primary_key",
                f"Primary key {tuple(key_columns)} is blank on {int(blank_rows.sum())} row(s)",
                count=int(blank_rows.sum()),
                examples=_row_examples(frame.index[blank_rows]),
            )
        )

    populated = subset[~blank_rows]
    duplicated = populated.duplicated(keep=False)
    if bool(duplicated.any()):
        offenders = populated[duplicated].drop_duplicates()
        report.add(
            ValidationIssue(
                Severity.ERROR,
                spec.name,
                "duplicate_primary_key",
                f"Primary key {tuple(key_columns)} repeats on {int(duplicated.sum())} row(s)",
                count=int(duplicated.sum()),
                examples=_key_examples(offenders, key_columns),
            )
        )


# --------------------------------------------------------------------------- #
# Content checks (WARNING)
# --------------------------------------------------------------------------- #


def _check_required_values(
    spec: TableSpec, frame: pd.DataFrame, report: ValidationReport
) -> None:
    """Warn when a non-nullable column is blank."""
    for column in spec.columns:
        if column.nullable or column.name not in frame.columns:
            continue
        blanks = frame[column.name] == ""
        if bool(blanks.any()):
            report.add(
                ValidationIssue(
                    Severity.WARNING,
                    spec.name,
                    "unexpected_null",
                    f"Column {column.name!r} is declared non-nullable but is blank "
                    f"on {int(blanks.sum())} row(s)",
                    count=int(blanks.sum()),
                    examples=_row_examples(frame.index[blanks]),
                )
            )


def _check_value_parsing(
    loader: DataLoader, spec: TableSpec, frame: pd.DataFrame, report: ValidationReport
) -> None:
    """Warn about cells that are populated but do not parse to their declared type.

    Covers malformed timestamps, non-numeric integers and unrecognised
    booleans in one pass, by diffing the raw frame against the typed one.
    """
    typed = loader.frame(spec.name)
    for column in spec.columns:
        if column.type is ColumnType.TEXT or column.name not in frame.columns:
            continue
        populated = frame[column.name] != ""
        unparsed = populated & typed[column.name].isna()
        if bool(unparsed.any()):
            report.add(
                ValidationIssue(
                    Severity.WARNING,
                    spec.name,
                    f"malformed_{column.type.value}",
                    f"Column {column.name!r} has {int(unparsed.sum())} value(s) that are not "
                    f"valid {column.type.value}",
                    count=int(unparsed.sum()),
                    examples=_value_examples(frame.loc[unparsed, column.name]),
                )
            )


def _check_allowed_values(
    spec: TableSpec, frame: pd.DataFrame, report: ValidationReport
) -> None:
    """Warn when a closed-vocabulary column holds an undeclared value."""
    for column in spec.columns:
        if not column.allowed or column.name not in frame.columns:
            continue
        values = frame[column.name]
        offending = values[(values != "") & ~values.isin(column.allowed)]
        if not offending.empty:
            report.add(
                ValidationIssue(
                    Severity.WARNING,
                    spec.name,
                    "unexpected_value",
                    f"Column {column.name!r} holds {len(offending)} value(s) outside "
                    f"{list(column.allowed)}",
                    count=len(offending),
                    examples=_value_examples(offending),
                )
            )


def _check_foreign_keys(
    loader: DataLoader, healthy: Sequence[str], report: ValidationReport
) -> None:
    """Warn about references that do not resolve in their target table."""
    healthy_set = set(healthy)
    for table in healthy:
        spec = schema.get_spec(table)
        frame = loader.raw_frame(table)
        for foreign_key in spec.foreign_keys:
            if foreign_key.target_table not in healthy_set:
                continue
            if foreign_key.column not in frame.columns:
                continue

            target = loader.raw_frame(foreign_key.target_table)
            if foreign_key.target_column not in target.columns:
                continue

            known = set(target[foreign_key.target_column])
            values = frame[foreign_key.column]
            dangling = values[(values != "") & ~values.isin(known)]
            if not dangling.empty:
                report.add(
                    ValidationIssue(
                        Severity.WARNING,
                        table,
                        "broken_reference",
                        f"{foreign_key.column!r} has {dangling.nunique()} value(s) absent from "
                        f"{foreign_key.target_table}.{foreign_key.target_column}",
                        count=len(dangling),
                        examples=_value_examples(dangling.drop_duplicates()),
                    )
                )


def _check_conversation_invariants(
    loader: DataLoader, healthy: Sequence[str], report: ValidationReport
) -> None:
    """Warn when ``conversation_type`` disagrees with the populated reference columns.

    A ``business`` message must name a business and must not name a group; a
    ``group`` message must name a group, and so on. This invariant holds across
    all 552 message rows shipped with the dataset.
    """
    for table in schema.MESSAGE_SHAPED_TABLES:
        if table not in healthy:
            continue
        frame = loader.raw_frame(table)
        if "conversation_type" not in frame.columns:
            continue

        for conversation_type, required in schema.CONVERSATION_TYPE_REQUIRED_REFERENCE.items():
            rows = frame[frame["conversation_type"] == conversation_type]
            if rows.empty or required not in rows.columns:
                continue
            missing = rows[rows[required] == ""]
            if not missing.empty:
                report.add(
                    ValidationIssue(
                        Severity.WARNING,
                        table,
                        "conversation_reference_missing",
                        f"{len(missing)} {conversation_type!r} row(s) have no {required!r}",
                        count=len(missing),
                        examples=_id_examples(missing),
                    )
                )

        for conversation_type, forbidden in schema.CONVERSATION_TYPE_FORBIDDEN_REFERENCES.items():
            rows = frame[frame["conversation_type"] == conversation_type]
            if rows.empty:
                continue
            for column in forbidden:
                if column not in rows.columns:
                    continue
                populated = rows[rows[column] != ""]
                if not populated.empty:
                    report.add(
                        ValidationIssue(
                            Severity.WARNING,
                            table,
                            "conversation_reference_unexpected",
                            f"{len(populated)} {conversation_type!r} row(s) unexpectedly "
                            f"populate {column!r}",
                            count=len(populated),
                            examples=_id_examples(populated),
                        )
                    )


def _check_media_references(
    loader: DataLoader, healthy: Sequence[str], report: ValidationReport
) -> None:
    """Warn about media that is half-declared, unknown, or absent from disk."""
    for table in schema.MESSAGE_SHAPED_TABLES:
        if table not in healthy:
            continue
        frame = loader.raw_frame(table)
        if not {"media_type", "media_id"} <= set(frame.columns):
            continue

        mismatched = frame[(frame["media_type"] == "") != (frame["media_id"] == "")]
        if not mismatched.empty:
            report.add(
                ValidationIssue(
                    Severity.WARNING,
                    table,
                    "media_pair_mismatch",
                    f"{len(mismatched)} row(s) set only one of 'media_type'/'media_id'",
                    count=len(mismatched),
                    examples=_id_examples(mismatched),
                )
            )

        for media_type, (media_table, id_column) in schema.MEDIA_TYPE_TO_TABLE.items():
            if media_table not in healthy:
                continue
            known = set(loader.raw_frame(media_table)[id_column])
            referenced = frame.loc[frame["media_type"] == media_type, "media_id"]
            dangling = referenced[(referenced != "") & ~referenced.isin(known)]
            if not dangling.empty:
                report.add(
                    ValidationIssue(
                        Severity.WARNING,
                        table,
                        "unknown_media_id",
                        f"{dangling.nunique()} {media_type!r} id(s) absent from {media_table}",
                        count=len(dangling),
                        examples=_value_examples(dangling.drop_duplicates()),
                    )
                )

    for media_table, id_column in schema.MEDIA_TYPE_TO_TABLE.values():
        if media_table not in healthy:
            continue
        frame = loader.raw_frame(media_table)
        if "file_path" not in frame.columns:
            continue
        absent = frame[
            ~frame["file_path"].map(lambda p: file_exists(p, loader.dataset_dir))
        ]
        if not absent.empty:
            report.add(
                ValidationIssue(
                    Severity.WARNING,
                    media_table,
                    "missing_media_file",
                    f"{len(absent)} referenced media file(s) are not on disk",
                    count=len(absent),
                    examples=_value_examples(absent[id_column]),
                )
            )


# --------------------------------------------------------------------------- #
# Example formatting
# --------------------------------------------------------------------------- #


def _value_examples(values: Iterable[object]) -> tuple[str, ...]:
    """Return up to ``MAX_ISSUE_EXAMPLES`` truncated string examples."""
    return tuple(
        truncate(value, _EXAMPLE_WIDTH)
        for value in list(values)[: config.MAX_ISSUE_EXAMPLES]
    )


def _row_examples(index: Iterable[object]) -> tuple[str, ...]:
    """Return offending row numbers, 1-based and header-adjusted for CSV lines."""
    return tuple(
        f"line {int(position) + 2}"
        for position in list(index)[: config.MAX_ISSUE_EXAMPLES]
    )


def _key_examples(frame: pd.DataFrame, key_columns: Sequence[str]) -> tuple[str, ...]:
    """Render composite key tuples as ``a|b`` strings."""
    rows = frame.head(config.MAX_ISSUE_EXAMPLES).to_dict(orient="records")
    return tuple("|".join(str(row[column]) for column in key_columns) for row in rows)


def _id_examples(frame: pd.DataFrame) -> tuple[str, ...]:
    """Prefer message ids for examples, falling back to row numbers."""
    if "message_id" in frame.columns:
        return _value_examples(frame["message_id"])
    return _row_examples(frame.index)

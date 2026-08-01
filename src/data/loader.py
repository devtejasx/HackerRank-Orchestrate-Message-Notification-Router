"""CSV loading. The only module in the project that touches the filesystem for data.

:class:`DataLoader` reads each file at most once and memoises three views of it:

* :meth:`DataLoader.raw_frame` - every cell as a stripped string, exactly as on
  disk. This is what :mod:`src.data.validation` inspects, because coercion must
  not hide the defects the validator is looking for.
* :meth:`DataLoader.frame` - the same data with real dtypes (``Int64``,
  ``boolean``, ``datetime64``), for pandas-style analysis.
* :meth:`DataLoader.records` - immutable :mod:`src.data.models` dataclasses.

Loading is lazy per table; :meth:`DataLoader.load_all` eagerly warms every
table and logs the record counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from src import config
from src.data import schema
from src.data.models import MODEL_BY_TABLE, Record
from src.data.schema import ColumnType
from src.utils.helpers import safe_bool

__all__ = ["DataLoader", "DatasetError", "MissingDatasetFileError"]

_LOGGER = config.get_logger("loader")

#: Pandas extension dtypes per logical column type. Nullable throughout so a
#: missing integer stays missing instead of silently becoming a float.
_PANDAS_DTYPE: Final[dict[ColumnType, str]] = {
    ColumnType.INT: "Int64",
    ColumnType.FLOAT: "Float64",
    ColumnType.BOOL: "boolean",
    ColumnType.TEXT: "string",
}


class DatasetError(RuntimeError):
    """Base class for unrecoverable dataset problems."""


class MissingDatasetFileError(DatasetError):
    """Raised when a required CSV is absent from the dataset directory."""


class DataLoader:
    """Reads the dataset once and hands out cached views of it.

    Args:
        dataset_dir: Directory holding the CSVs. Defaults to
            :data:`src.config.DATASET_DIR`.

    Example:
        >>> loader = DataLoader()          # doctest: +SKIP
        >>> loader.load_all()              # doctest: +SKIP
        >>> loader.messages.shape          # doctest: +SKIP
        (110, 11)
    """

    def __init__(self, dataset_dir: Path | None = None) -> None:
        self._dataset_dir: Path = Path(dataset_dir) if dataset_dir else config.DATASET_DIR
        self._raw_frames: dict[str, pd.DataFrame] = {}
        self._typed_frames: dict[str, pd.DataFrame] = {}
        self._records: dict[str, tuple[Record, ...]] = {}

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def dataset_dir(self) -> Path:
        """Directory the loader reads from."""
        return self._dataset_dir

    def path_for(self, table: str) -> Path:
        """Return the on-disk path of ``table``."""
        return self._dataset_dir / schema.get_spec(table).filename

    def is_available(self, table: str) -> bool:
        """Whether ``table``'s CSV exists on disk."""
        return self.path_for(table).is_file()

    def available_tables(self) -> tuple[str, ...]:
        """Logical names of every declared table whose file is present."""
        return tuple(name for name in schema.TABLES if self.is_available(name))

    def missing_required_tables(self) -> tuple[str, ...]:
        """Required tables whose CSV is absent."""
        return tuple(
            name for name in schema.REQUIRED_TABLE_NAMES if not self.is_available(name)
        )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load_all(self) -> dict[str, int]:
        """Load every available table and log its record count.

        Optional tables that are absent are skipped with a warning; missing
        required tables raise.

        Returns:
            Mapping of logical table name to number of rows loaded.

        Raises:
            MissingDatasetFileError: If the dataset directory or any required
                CSV is missing.
        """
        if not self._dataset_dir.is_dir():
            raise MissingDatasetFileError(
                f"Dataset directory not found: {self._dataset_dir}"
            )

        counts: dict[str, int] = {}
        for name, spec in schema.TABLES.items():
            if not self.is_available(name):
                if spec.required:
                    raise MissingDatasetFileError(
                        f"Required file missing: {self.path_for(name)}"
                    )
                _LOGGER.warning(
                    "Skipped %s - optional file not found (%s)",
                    spec.display_label,
                    spec.filename,
                )
                continue
            frame = self.raw_frame(name)
            counts[name] = len(frame)
            _LOGGER.info("Loaded %-22s %6d records", spec.display_label, len(frame))
        return counts

    def raw_frame(self, table: str) -> pd.DataFrame:
        """Return ``table`` as an all-string DataFrame, cached.

        Empty cells stay as empty strings and every value is whitespace
        stripped, so validation sees the file as it really is.

        Args:
            table: Logical table name.

        Raises:
            MissingDatasetFileError: If the file does not exist.
        """
        cached = self._raw_frames.get(table)
        if cached is not None:
            return cached

        spec = schema.get_spec(table)
        path = self.path_for(table)
        if not path.is_file():
            raise MissingDatasetFileError(f"File missing for {table!r}: {path}")

        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_values=[],
            encoding="utf-8",
        )
        for column in frame.columns:
            frame[column] = frame[column].str.strip()

        self._raw_frames[table] = frame
        _LOGGER.debug("Read %s (%d rows, %d cols)", spec.filename, *frame.shape)
        return frame

    def frame(self, table: str) -> pd.DataFrame:
        """Return ``table`` with real dtypes applied, cached.

        Columns absent from the file are left absent rather than fabricated;
        :mod:`src.data.validation` reports them.

        Args:
            table: Logical table name.
        """
        cached = self._typed_frames.get(table)
        if cached is not None:
            return cached

        spec = schema.get_spec(table)
        typed = self.raw_frame(table).copy()
        for column in spec.columns:
            if column.name in typed.columns:
                typed[column.name] = _coerce_series(typed[column.name], column.type)

        self._typed_frames[table] = typed
        return typed

    def records(self, table: str) -> tuple[Record, ...]:
        """Return ``table`` as immutable dataclass records, cached.

        Built from the raw string frame so coercion follows exactly the same
        rules as :mod:`src.utils.helpers` everywhere else.

        Args:
            table: Logical table name.

        Raises:
            RecordCoercionError: If a row lacks a non-nullable field. Run
                :func:`src.data.validation.validate_dataset` first for a full
                report rather than a single failure.
        """
        cached = self._records.get(table)
        if cached is not None:
            return cached

        model = MODEL_BY_TABLE[table]
        rows = self.raw_frame(table).to_dict(orient="records")
        built = tuple(model.from_row(row) for row in rows)

        self._records[table] = built
        return built

    def summary(self) -> dict[str, int]:
        """Return row counts for every table already loaded."""
        return {name: len(frame) for name, frame in self._raw_frames.items()}

    # ------------------------------------------------------------------ #
    # Convenience accessors
    #
    # Typed DataFrames, so `loader.messages` behaves like a normal frame.
    # ------------------------------------------------------------------ #

    @property
    def users(self) -> pd.DataFrame:
        """``users.csv`` as a typed DataFrame."""
        return self.frame("users")

    @property
    def groups(self) -> pd.DataFrame:
        """``groups.csv`` as a typed DataFrame."""
        return self.frame("groups")

    @property
    def group_members(self) -> pd.DataFrame:
        """``group_members.csv`` as a typed DataFrame."""
        return self.frame("group_members")

    @property
    def business_accounts(self) -> pd.DataFrame:
        """``business_accounts.csv`` as a typed DataFrame."""
        return self.frame("business_accounts")

    @property
    def user_business_history(self) -> pd.DataFrame:
        """``user_business_history.csv`` as a typed DataFrame."""
        return self.frame("user_business_history")

    @property
    def messages(self) -> pd.DataFrame:
        """``messages.csv`` as a typed DataFrame."""
        return self.frame("messages")

    @property
    def message_history(self) -> pd.DataFrame:
        """``message_history.csv`` as a typed DataFrame."""
        return self.frame("message_history")

    @property
    def message_events(self) -> pd.DataFrame:
        """``message_events.csv`` as a typed DataFrame."""
        return self.frame("message_events")

    @property
    def images(self) -> pd.DataFrame:
        """``images.csv`` as a typed DataFrame."""
        return self.frame("images")

    @property
    def voice_notes(self) -> pd.DataFrame:
        """``voice_notes.csv`` as a typed DataFrame."""
        return self.frame("voice_notes")

    @property
    def daily_notification_summary(self) -> pd.DataFrame:
        """``daily_notification_summary.csv`` as a typed DataFrame."""
        return self.frame("daily_notification_summary")

    @property
    def sample_messages(self) -> pd.DataFrame:
        """``sample_messages.csv`` as a typed DataFrame (optional table)."""
        return self.frame("sample_messages")


# --------------------------------------------------------------------------- #
# Column coercion
# --------------------------------------------------------------------------- #


def _coerce_series(series: pd.Series, column_type: ColumnType) -> pd.Series:
    """Convert one all-string column to its logical dtype.

    Unparseable values become ``NA``/``NaT`` rather than raising, so a single
    bad cell cannot abort a load. Validation reports them separately.

    Args:
        series: All-string column straight from :meth:`DataLoader.raw_frame`.
        column_type: Target logical type.

    Returns:
        A new, converted Series.
    """
    blanks = series == ""

    if column_type is ColumnType.TEXT:
        return series.mask(blanks, pd.NA).astype(_PANDAS_DTYPE[ColumnType.TEXT])

    if column_type is ColumnType.BOOL:
        return series.map(safe_bool).astype(_PANDAS_DTYPE[ColumnType.BOOL])

    if column_type in (ColumnType.INT, ColumnType.FLOAT):
        numeric = pd.to_numeric(series.mask(blanks, pd.NA), errors="coerce")
        return numeric.astype(_PANDAS_DTYPE[column_type])

    formats = (
        config.DATE_FORMATS if column_type is ColumnType.DATE else config.TIMESTAMP_FORMATS
    )
    return _to_datetime(series.mask(blanks, pd.NA), formats)


def _to_datetime(series: pd.Series, formats: tuple[str, ...]) -> pd.Series:
    """Parse a column against each accepted layout, filling gaps in order.

    Args:
        series: Column of timestamp strings with blanks already set to ``NA``.
        formats: Layouts to try, most likely first.

    Returns:
        A ``datetime64`` Series; values matching no layout become ``NaT``.
    """
    parsed = pd.to_datetime(series, format=formats[0], errors="coerce")
    for fmt in formats[1:]:
        remaining = parsed.isna() & series.notna()
        if not remaining.any():
            break
        parsed = parsed.fillna(
            pd.to_datetime(series.where(remaining), format=fmt, errors="coerce")
        )
    return parsed

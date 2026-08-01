"""Small, reusable coercion and parsing helpers.

Every function here is total: it returns a sensible value or ``None`` instead
of raising on malformed input. Callers that need strictness check for ``None``
themselves. This keeps a single messy row from aborting a whole load.

The module deliberately avoids importing pandas so it stays usable from
tests, scripts and any future non-pandas code path.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Hashable, Iterable
from datetime import date, datetime, time
from pathlib import Path
from typing import TypeVar

from src import config

__all__ = [
    "is_missing",
    "handle_missing",
    "safe_text",
    "safe_int",
    "safe_float",
    "safe_bool",
    "normalize_string",
    "parse_timestamp",
    "parse_date",
    "parse_dnd_window",
    "file_exists",
    "resolve_dataset_path",
    "index_by",
    "group_by",
    "truncate",
]

_T = TypeVar("_T")
_K = TypeVar("_K", bound=Hashable)

_WHITESPACE_RUN = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Missing-value handling
# --------------------------------------------------------------------------- #


def is_missing(value: object) -> bool:
    """Return ``True`` when ``value`` carries no information.

    Recognises ``None``, float ``NaN``, pandas ``NA``/``NaT`` and the string
    sentinels listed in :data:`src.config.NULL_TOKENS`.

    Args:
        value: Any cell value read from a CSV or DataFrame.

    Returns:
        ``True`` if the value should be treated as absent.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, (datetime, date, time)):
        # pandas NaT is a datetime subclass instance, and like NaN it is the
        # only temporal value that compares unequal to itself.
        return value != value
    if isinstance(value, (int, bool)):
        return False
    return str(value).strip().lower() in config.NULL_TOKENS


def handle_missing(value: object, default: _T) -> object | _T:
    """Return ``value``, or ``default`` when the value is missing."""
    return default if is_missing(value) else value


# --------------------------------------------------------------------------- #
# Scalar coercion
# --------------------------------------------------------------------------- #


def safe_text(value: object, default: str | None = None) -> str | None:
    """Coerce ``value`` to a stripped string.

    Args:
        value: Raw cell value.
        default: Returned when the value is missing.

    Returns:
        The stripped string, or ``default``.
    """
    if is_missing(value):
        return default
    return str(value).strip()


def safe_int(value: object, default: int | None = None) -> int | None:
    """Coerce ``value`` to an ``int``, tolerating float-formatted integers.

    ``"12"``, ``12.0`` and ``" 12 "`` all yield ``12``. Genuinely fractional
    input such as ``"12.5"`` is truncated toward zero, matching ``int()``.

    Args:
        value: Raw cell value.
        default: Returned when the value is missing or unparseable.
    """
    if is_missing(value):
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def safe_float(value: object, default: float | None = None) -> float | None:
    """Coerce ``value`` to a ``float``, or return ``default``."""
    if is_missing(value):
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def safe_bool(value: object, default: bool | None = None) -> bool | None:
    """Coerce ``value`` to a ``bool``.

    The dataset encodes booleans as ``0``/``1`` integers; the textual forms in
    :data:`src.config.TRUE_TOKENS` and :data:`src.config.FALSE_TOKENS` are
    accepted defensively.

    Args:
        value: Raw cell value.
        default: Returned when the value is missing or unrecognised.
    """
    if is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in config.TRUE_TOKENS:
        return True
    if token in config.FALSE_TOKENS:
        return False
    # Tolerate "1.0"/"0.0" produced by float round-tripping.
    number = safe_float(token)
    if number is not None:
        return number != 0.0
    return default


def normalize_string(value: object, default: str = "") -> str:
    """Lowercase, collapse internal whitespace and strip.

    Intended for comparison keys (domains, categories, free text lookups), not
    for values that will be shown to a user.

    Args:
        value: Raw cell value.
        default: Returned when the value is missing.
    """
    text = safe_text(value)
    if text is None:
        return default
    return _WHITESPACE_RUN.sub(" ", text).strip().lower()


def truncate(value: object, limit: int, suffix: str = "...") -> str:
    """Shorten a value's string form to ``limit`` characters for logging."""
    text = safe_text(value, default="") or ""
    single_line = _WHITESPACE_RUN.sub(" ", text)
    if len(single_line) <= limit:
        return single_line
    return single_line[: max(0, limit - len(suffix))] + suffix


# --------------------------------------------------------------------------- #
# Temporal parsing
# --------------------------------------------------------------------------- #


def _parse_with_formats(value: object, formats: Iterable[str]) -> datetime | None:
    """Try each ``strptime`` format in order and return the first success."""
    if is_missing(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip()
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_timestamp(value: object) -> datetime | None:
    """Parse a dataset timestamp into a ``datetime``.

    Accepts the layouts in :data:`src.config.TIMESTAMP_FORMATS` and falls back
    to the date-only layouts so a date column never fails purely on precision.

    Returns:
        The parsed ``datetime``, or ``None`` when missing or malformed.
    """
    return _parse_with_formats(
        value, (*config.TIMESTAMP_FORMATS, *config.DATE_FORMATS)
    )


def parse_date(value: object) -> date | None:
    """Parse a dataset date into a ``date``.

    Returns:
        The parsed ``date``, or ``None`` when missing or malformed.
    """
    parsed = _parse_with_formats(
        value, (*config.DATE_FORMATS, *config.TIMESTAMP_FORMATS)
    )
    return parsed.date() if parsed is not None else None


def parse_dnd_window(value: object) -> tuple[time, time] | None:
    """Parse a ``do_not_disturb_window`` such as ``"22:00-07:00"``.

    The window may wrap past midnight; that is the caller's concern. This
    function only splits and parses the two endpoints.

    Returns:
        ``(start, end)`` times, or ``None`` when missing or malformed.
    """
    text = safe_text(value)
    if text is None:
        return None
    parts = text.split(config.DND_WINDOW_SEPARATOR)
    if len(parts) != 2:
        return None
    try:
        start = time.fromisoformat(parts[0].strip())
        end = time.fromisoformat(parts[1].strip())
    except ValueError:
        return None
    return start, end


# --------------------------------------------------------------------------- #
# Filesystem
# --------------------------------------------------------------------------- #


def file_exists(path: str | Path, base_dir: Path | None = None) -> bool:
    """Return whether ``path`` points at an existing file.

    Args:
        path: Absolute path, or a path relative to ``base_dir``.
        base_dir: Directory used to resolve relative paths.
    """
    return resolve_dataset_path(path, base_dir).is_file()


def resolve_dataset_path(path: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a dataset-relative path (e.g. ``media/images/img_001.jpg``).

    Args:
        path: Path as stored in the CSV.
        base_dir: Directory the path is relative to. Defaults to
            :data:`src.config.DATASET_DIR`.

    Returns:
        An absolute path. Absolute inputs are returned unchanged.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir or config.DATASET_DIR) / candidate


# --------------------------------------------------------------------------- #
# Collection shaping
# --------------------------------------------------------------------------- #


def index_by(
    items: Iterable[_T], key: Callable[[_T], _K]
) -> dict[_K, _T]:
    """Build a one-to-one lookup from ``items``.

    On duplicate keys the last item wins; callers that care about duplicates
    detect them during validation, not here.

    Args:
        items: Records to index.
        key: Extracts the lookup key from a record.
    """
    return {key(item): item for item in items}


def group_by(
    items: Iterable[_T], key: Callable[[_T], _K | None]
) -> dict[_K, list[_T]]:
    """Build a one-to-many lookup from ``items``.

    Records whose key is ``None`` are skipped, which is what makes this safe
    for nullable foreign keys such as ``group_id``.

    Args:
        items: Records to group.
        key: Extracts the grouping key, or ``None`` to skip the record.
    """
    grouped: dict[_K, list[_T]] = {}
    for item in items:
        group_key = key(item)
        if group_key is None:
            continue
        grouped.setdefault(group_key, []).append(item)
    return grouped

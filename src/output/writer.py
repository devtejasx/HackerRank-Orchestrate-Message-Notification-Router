"""Writes predictions to the submission CSV.

Deliberately small. The rows are already correct by the time they arrive -
:meth:`~src.routing.models.RoutingResult.to_output_row` emits exactly the six
required columns in order - so this module only handles encoding, quoting and
atomicity.

Writing goes through a temporary file that is renamed into place, so an
interrupted run leaves the previous ``output.csv`` intact rather than a
half-written one. That matters when the file being replaced is the submission.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from pathlib import Path

from src import config
from src.routing.models import OUTPUT_COLUMNS, RoutingResult

__all__ = ["write_output_csv", "format_row"]

_LOGGER = config.get_logger("output.writer")


def format_row(result: RoutingResult) -> dict[str, str]:
    """Render one result as CSV-ready strings.

    Confidence is formatted to a fixed number of decimals so the column reads
    consistently rather than mixing ``0.9`` with ``0.85``.

    Args:
        result: The prediction to render.

    Returns:
        A mapping keyed by the output columns, with every value a string.
    """
    row = result.to_output_row()
    row["confidence"] = f"{result.confidence:.{config.CONFIDENCE_DECIMALS}f}"
    return {column: str(row[column]) for column in OUTPUT_COLUMNS}


def write_output_csv(
    results: Sequence[RoutingResult], path: Path | None = None
) -> Path:
    """Write predictions to ``path``.

    Args:
        results: Predictions, in the order they should appear.
        path: Destination. Defaults to :data:`src.config.OUTPUT_CSV`.

    Returns:
        The path written to.

    Raises:
        OSError: If the destination cannot be written.
    """
    destination = Path(path) if path is not None else config.OUTPUT_CSV
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_name(f"{destination.name}.tmp")
    try:
        _write_rows(results, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    _LOGGER.info("Wrote %d prediction(s) to %s", len(results), destination)
    return destination


def _write_rows(results: Sequence[RoutingResult], path: Path) -> None:
    """Write the header and every row to ``path``.

    Quoting is minimal, which keeps the file readable, but reasons contain
    commas and occasionally quotes and the writer escapes those itself.
    """
    with path.open(
        "w",
        newline="",
        encoding=config.OUTPUT_ENCODING,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(OUTPUT_COLUMNS),
            lineterminator=config.OUTPUT_LINE_TERMINATOR,
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(format_row(result) for result in results)

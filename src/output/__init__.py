"""Submission output: validation and CSV generation.

    from src.output import validate_results, write_submission

    report = validate_results(results, messages, repo)
    report.raise_for_errors()
    write_submission(results)

Use :func:`write_output_csv` when exactly one file is wanted;
:func:`write_submission` also writes the mirror copies.
"""

from src.output.validation import (
    OutputIssue,
    OutputSeverity,
    OutputValidationError,
    OutputValidationReport,
    validate_results,
)
from src.output.writer import format_row, write_output_csv, write_submission

__all__ = [
    "OutputIssue",
    "OutputSeverity",
    "OutputValidationError",
    "OutputValidationReport",
    "format_row",
    "validate_results",
    "write_output_csv",
    "write_submission",
]

"""Submission output: validation and CSV generation.

    from src.output import validate_results, write_output_csv

    report = validate_results(results, messages)
    report.raise_for_errors()
    write_output_csv(results)
"""

from src.output.validation import (
    OutputIssue,
    OutputSeverity,
    OutputValidationError,
    OutputValidationReport,
    validate_results,
)
from src.output.writer import format_row, write_output_csv

__all__ = [
    "OutputIssue",
    "OutputSeverity",
    "OutputValidationError",
    "OutputValidationReport",
    "format_row",
    "validate_results",
    "write_output_csv",
]

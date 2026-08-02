"""Command-line interface: argument handling, commands and console rendering.

Split three ways so each part can change without disturbing the others:
:mod:`src.cli.render` formats, :mod:`src.cli.commands` drives the pipeline, and
``main.py`` only parses arguments and dispatches.
"""

from src.cli import render
from src.cli.commands import (
    EXIT_FAILED,
    EXIT_OK,
    check_dataset,
    inspect_messages,
    run_evaluation,
    run_submission,
    show_schema,
)

__all__ = [
    "EXIT_FAILED",
    "EXIT_OK",
    "check_dataset",
    "inspect_messages",
    "render",
    "run_evaluation",
    "run_submission",
    "show_schema",
]

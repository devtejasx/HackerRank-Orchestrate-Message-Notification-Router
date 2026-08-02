"""Message Notification Router - entry point.

Run with no arguments to produce the submission:

    python main.py

That loads ``dataset/messages.csv``, runs every phase, validates the
predictions and writes ``dataset/output.csv``.

    Repository -> Features -> Classification -> Routing signals
      -> Decision -> Evidence -> Reason -> Confidence -> output.csv

Voice notes are transcribed with Whisper before they enter that pipeline, and
transcripts are cached in ``transcripts.json`` so repeat runs cost nothing.
Use ``--no-transcribe`` to route them on context alone.

Other modes:

    python main.py --inspect                 walk through a few messages
    python main.py --inspect --message ID    walk through one message
    python main.py --evaluate                metrics against the labelled rows
    python main.py --schema-only             print the dataset schema
    python main.py --data-only               Phase 1 checks only
    python main.py --no-write                run and validate, write nothing

This module only parses arguments and dispatches; the work lives in
:mod:`src.cli.commands` and the formatting in :mod:`src.cli.render`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import config
from src.cli import commands

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "WhatsApp Message Notification Router. With no arguments, runs the "
            "full pipeline and writes output.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py                        generate output.csv\n"
            "  python main.py --inspect -m msg_091   explain one decision\n"
            "  python main.py --evaluate             score against labelled rows\n"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        metavar="DIR",
        help=f"Dataset directory (default: {config.DATASET_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Where to write predictions (default: {config.OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--log-level",
        default=config.DEFAULT_LOG_LEVEL,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log level.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat dataset validation warnings as blocking errors.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip dataset validation. Only for inspecting broken data.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run and validate everything, but do not write output.csv.",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip the evaluation against the labelled examples.",
    )

    voice = parser.add_argument_group("voice transcription")
    voice.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Do not transcribe voice notes. They route on sender context alone.",
    )
    voice.add_argument(
        "--whisper-model",
        metavar="SIZE",
        default=None,
        help="Whisper weights to use, e.g. tiny, base, small (default: base).",
    )
    voice.add_argument(
        "--refresh-transcripts",
        action="store_true",
        help="Discard cached transcripts and re-run speech-to-text from the audio.",
    )

    mode = parser.add_argument_group("modes")
    mode.add_argument(
        "--inspect",
        action="store_true",
        help="Walk through the pipeline for selected messages instead of writing.",
    )
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Print agreement metrics against the labelled examples and stop.",
    )
    mode.add_argument(
        "--schema-only",
        action="store_true",
        help="Print the dataset schema and stop.",
    )
    mode.add_argument(
        "--data-only",
        action="store_true",
        help="Run the Phase 1 data-layer checks and stop.",
    )

    inspect_group = parser.add_argument_group("inspection")
    inspect_group.add_argument(
        "--no-personalize",
        action="store_true",
        help="Skip Phase 3 routing signals when inspecting.",
    )
    inspect_group.add_argument(
        "--no-route",
        action="store_true",
        help="Skip the Phase 4 decision when inspecting.",
    )

    selection = inspect_group.add_mutually_exclusive_group()
    selection.add_argument(
        "-m", "--message", metavar="ID", help="Inspect one message by id."
    )
    selection.add_argument(
        "--limit", type=int, metavar="N", help="Inspect the first N messages."
    )
    selection.add_argument(
        "--all", action="store_true", help="Inspect every message."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the requested command.

    Args:
        argv: Argument list, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    args = parse_args(argv)
    config.configure_logging(args.log_level)

    if args.schema_only:
        return commands.show_schema()

    if args.data_only:
        return commands.check_dataset(args.dataset, strict=args.strict)

    understanding = _build_understanding(args)

    if args.evaluate:
        return commands.run_evaluation(
            args.dataset, strict=args.strict, understanding=understanding
        )

    if args.inspect or args.message or args.limit or args.all:
        return commands.inspect_messages(
            args.dataset,
            message_id=args.message,
            limit=args.limit,
            all_messages=args.all,
            personalize=not args.no_personalize,
            route=not args.no_route,
            validate_dataset=not args.no_validate,
            strict=args.strict,
            understanding=understanding,
        )

    return commands.run_submission(
        args.dataset,
        args.output,
        validate_dataset=not args.no_validate,
        strict=args.strict,
        write=not args.no_write,
        evaluate=not args.no_evaluate,
        understanding=understanding,
    )


def _build_understanding(args: argparse.Namespace) -> object:
    """Build the media provider the run should use, from the voice flags.

    Args:
        args: The parsed namespace.

    Returns:
        A provider, always. ``--no-transcribe`` yields one that recovers
        nothing rather than ``None``, so the choice is explicit rather than
        left to a downstream default.
    """
    from src.media.cache import TranscriptCache
    from src.media.understanding import default_understanding

    if args.refresh_transcripts:
        # Emptied before the provider reads it, so this run re-transcribes and
        # writes the results back.
        cache = TranscriptCache()
        cache.clear()
        cache.save()

    return default_understanding(
        transcribe=not args.no_transcribe, model_size=args.whisper_model
    )


if __name__ == "__main__":
    sys.exit(main())

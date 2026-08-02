"""What the command-line interface actually does.

Each function here is one command: it drives the pipeline, prints through
:mod:`src.cli.render` and returns a process exit code. Splitting them out of
``main.py`` keeps argument parsing separate from behaviour, so a command can be
called directly from a test or a notebook without going through argv.

Every command is safe to run against a broken dataset: loading failures are
caught and reported as a non-zero exit rather than a traceback.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from src import config
from src.cli import render
from src.data.loader import DatasetError
from src.data.models import MessageRecord
from src.data.repository import DataRepository
from src.evaluation import evaluate_samples
from src.media.understanding import MediaUnderstanding
from src.output import validate_results, write_submission
from src.pipeline import MessagePipeline
from src.routing.models import RoutingResult
from src.routing.pipeline import RoutingPipeline

__all__ = [
    "EXIT_OK",
    "EXIT_FAILED",
    "run_submission",
    "inspect_messages",
    "show_schema",
    "check_dataset",
    "run_evaluation",
]

_LOGGER = config.get_logger("cli")

#: Process exit codes.
EXIT_OK = 0
EXIT_FAILED = 1

#: How many messages the inspect command shows in full when given no selection.
DEFAULT_SAMPLE_SIZE = 3


def load_repository(
    dataset_dir: Path | None, *, validate: bool = True, strict: bool = False
) -> DataRepository:
    """Load, validate and index the dataset.

    Args:
        dataset_dir: Dataset directory, or ``None`` for the configured default.
        validate: Whether to run Phase 1 validation.
        strict: Whether to treat validation warnings as errors.

    Returns:
        A ready repository.

    Raises:
        DatasetError: If the dataset cannot be loaded or fails validation.
    """
    return DataRepository.load(
        dataset_dir, validate=validate, strict=strict or None
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def run_submission(
    dataset_dir: Path | None = None,
    output_path: Path | None = None,
    *,
    validate_dataset: bool = True,
    strict: bool = False,
    write: bool = True,
    evaluate: bool = True,
    understanding: MediaUnderstanding | None = None,
) -> int:
    """Run every phase over the whole dataset and write ``output.csv``.

    This is the submission path: load, extract, classify, personalise, route,
    validate and write. Predictions are validated *before* the file is
    touched, so an invalid run leaves any previous submission intact.

    Args:
        dataset_dir: Dataset directory, or ``None`` for the default.
        output_path: Destination CSV, or ``None`` for the configured default.
        validate_dataset: Whether to run Phase 1 validation on load.
        strict: Treat dataset validation warnings as errors.
        write: Whether to write the file. ``False`` runs everything and reports
            without touching disk.
        evaluate: Whether to also measure against the labelled examples.
        understanding: Speech-to-text / OCR provider. ``None`` uses
            :func:`~src.media.understanding.default_understanding`.

    Returns:
        ``EXIT_OK`` on success, ``EXIT_FAILED`` if the dataset could not be
        loaded or the predictions failed validation.
    """
    started = time.perf_counter()
    try:
        repo = load_repository(dataset_dir, validate=validate_dataset, strict=strict)
    except DatasetError as exc:
        _LOGGER.error("Could not load the dataset: %s", exc)
        return EXIT_FAILED

    pipeline = RoutingPipeline(repo, understanding=understanding)
    messages = repo.get_messages()

    render.heading("MESSAGE NOTIFICATION ROUTER")
    print(f"  dataset  : {repo.loader.dataset_dir}")
    print(f"  messages : {len(messages)}")
    print(f"  media    : {pipeline.analysis.extractor.media.understanding.name}")

    results = pipeline.route_many(messages)
    elapsed = time.perf_counter() - started

    report = validate_results(results, messages, repo)
    report.log()
    render.print_output_validation(report)

    if not report.is_valid:
        _LOGGER.error("Refusing to write predictions that failed validation")
        return EXIT_FAILED

    render.print_routing_summary(results)

    if evaluate:
        render.print_evaluation(evaluate_samples(pipeline))

    render.heading("RESULT")
    if write:
        written = write_submission(results, output_path)
        print(f"  Wrote {len(results)} prediction(s) to {written[0]}")
        for mirror in written[1:]:
            print(f"  Mirrored to {mirror}")
    else:
        print(f"  Validated {len(results)} prediction(s); nothing written.")
    print(f"  Completed in {elapsed:.1f}s.")
    return EXIT_OK


def inspect_messages(
    dataset_dir: Path | None = None,
    *,
    message_id: str | None = None,
    limit: int | None = None,
    all_messages: bool = False,
    personalize: bool = True,
    route: bool = True,
    validate_dataset: bool = True,
    strict: bool = False,
    understanding: MediaUnderstanding | None = None,
) -> int:
    """Walk through the pipeline for one or more messages, in detail.

    The diagnostic view: every feature, signal, rule and decision printed for
    inspection. Writes nothing.

    Args:
        dataset_dir: Dataset directory, or ``None`` for the default.
        message_id: Inspect exactly this message.
        limit: Inspect the first N messages.
        all_messages: Inspect every message, summarised.
        personalize: Whether to compute Phase 3 signals.
        route: Whether to compute the Phase 4 decision.
        validate_dataset: Whether to run Phase 1 validation on load.
        strict: Treat dataset validation warnings as errors.

    Returns:
        ``EXIT_OK`` on success, ``EXIT_FAILED`` if the dataset failed to load.

    Raises:
        SystemExit: If ``message_id`` names a message that does not exist.
    """
    try:
        repo = load_repository(dataset_dir, validate=validate_dataset, strict=strict)
    except DatasetError as exc:
        _LOGGER.error("Could not load the dataset: %s", exc)
        return EXIT_FAILED

    render.print_dataset_summary(repo)
    render.print_repository_lookups(repo)

    analysis_pipeline = MessagePipeline(
        repo, personalize=personalize, understanding=understanding
    )
    routing = (
        RoutingPipeline(repo, analysis_pipeline=analysis_pipeline)
        if route and personalize
        else None
    )

    selected = _select_messages(repo, message_id, limit, all_messages)
    analyses = analysis_pipeline.analyse_many(selected)

    render.heading(_phase_heading(personalize, routing is not None))
    detailed = analyses if len(analyses) <= DEFAULT_SAMPLE_SIZE else analyses[:1]
    results: list[RoutingResult] = []

    for analysis in detailed:
        render.print_analysis(analysis, repo)
        if routing is not None:
            result = routing.route_analysis(analysis)
            results.append(result)
            render.print_routing_decision(result)

    if routing is not None and len(analyses) > len(detailed):
        results = list(routing.route_many(selected))

    if len(analyses) > 1:
        render.print_analysis_summary(analyses)
        signals = [a.routing for a in analyses if a.routing is not None]
        if signals:
            render.print_signal_summary(signals)
        if results:
            render.print_routing_summary(results)

    render.heading("RESULT")
    print(f"  {len(analyses)} message(s) inspected, no exceptions raised.")
    print("  Nothing was written; use `python main.py` to produce output.csv.")
    return EXIT_OK


def show_schema() -> int:
    """Print the declared dataset schema and stop.

    Needs no dataset on disk.
    """
    render.print_schema()
    return EXIT_OK


def check_dataset(
    dataset_dir: Path | None = None, *, strict: bool = False
) -> int:
    """Run the Phase 1 data-layer checks and report.

    Args:
        dataset_dir: Dataset directory, or ``None`` for the default.
        strict: Treat validation warnings as errors.

    Returns:
        ``EXIT_OK`` when the dataset is usable, ``EXIT_FAILED`` otherwise.
    """
    try:
        repo = load_repository(dataset_dir, strict=strict)
    except DatasetError as exc:
        _LOGGER.error("Dataset check failed: %s", exc)
        return EXIT_FAILED

    render.print_dataset_summary(repo)
    render.print_repository_lookups(repo)
    render.heading("RESULT")
    print("  Data layer is ready. No exceptions raised.")
    return EXIT_OK


def run_evaluation(
    dataset_dir: Path | None = None,
    *,
    strict: bool = False,
    understanding: MediaUnderstanding | None = None,
) -> int:
    """Measure the system against the labelled examples and print metrics.

    Args:
        dataset_dir: Dataset directory, or ``None`` for the default.
        strict: Treat dataset validation warnings as errors.

    Returns:
        ``EXIT_OK`` on success, ``EXIT_FAILED`` if the dataset failed to load.
    """
    try:
        repo = load_repository(dataset_dir, strict=strict)
    except DatasetError as exc:
        _LOGGER.error("Could not load the dataset: %s", exc)
        return EXIT_FAILED

    render.print_evaluation(
        evaluate_samples(RoutingPipeline(repo, understanding=understanding))
    )
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _select_messages(
    repo: DataRepository,
    message_id: str | None,
    limit: int | None,
    all_messages: bool,
) -> tuple[MessageRecord, ...]:
    """Choose which messages to inspect.

    Raises:
        SystemExit: If ``message_id`` does not name a real message.
    """
    messages = repo.get_messages()
    if message_id:
        chosen = repo.get_message(message_id)
        if chosen is None:
            raise SystemExit(f"No such message: {message_id}")
        return (chosen,)
    if all_messages:
        return messages
    if limit:
        return messages[:limit]
    return _representative_sample(messages)


def _representative_sample(
    messages: Sequence[MessageRecord],
) -> tuple[MessageRecord, ...]:
    """Pick one message per conversation type, so the demo shows real variety.

    Chosen from the data rather than hardcoded, so it keeps working if the
    dataset is swapped.
    """
    chosen: list[MessageRecord] = []
    for conversation_type in config.CONVERSATION_TYPES:
        match = next(
            (m for m in messages if m.conversation_type == conversation_type), None
        )
        if match is not None:
            chosen.append(match)
    return tuple(chosen[:DEFAULT_SAMPLE_SIZE]) or tuple(messages[:DEFAULT_SAMPLE_SIZE])


def _phase_heading(personalize: bool, route: bool) -> str:
    """Return the heading matching the phases actually being run."""
    if not personalize:
        return "PHASE 2 - FEATURES AND CLASSIFICATION"
    if not route:
        return "PHASE 2 + 3 - FEATURES, CLASSIFICATION AND ROUTING SIGNALS"
    return "PHASES 2-4 - FEATURES, CLASSIFICATION, SIGNALS AND ROUTING"

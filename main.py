"""Entry point: demonstrates the data layer (Phase 1) and analysis (Phase 2).

Phase 1 loads, validates and indexes the dataset. Phase 2 turns each incoming
message into features and a classification.

No routing decision is made and no ``output.csv`` is written - both belong to
later phases.

Usage:
    python main.py                      # phase 1 checks, then analyse a sample
    python main.py --message msg_005    # analyse one specific message
    python main.py --limit 10           # analyse the first 10 messages
    python main.py --all                # analyse every message, with a summary
    python main.py --schema-only        # print the dataset schema and stop
    python main.py --data-only          # run the Phase 1 smoke test only
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from src import config
from src.data import schema
from src.data.loader import DatasetError
from src.data.models import MessageRecord
from src.data.repository import DataRepository
from src.personalization.signal_models import RoutingSignal, RoutingSignals
from src.pipeline import MessageAnalysis, MessagePipeline
from src.routing.models import RoutingResult
from src.routing.pipeline import RoutingPipeline
from src.utils.helpers import truncate

_LOGGER = config.get_logger("main")

#: Width of the printed section rules.
_RULE_WIDTH = 78

#: How much message text to show in the demo lookups.
_TEXT_PREVIEW = 70

#: How much message body to show in a full analysis report.
_BODY_PREVIEW = 300

#: How many messages to analyse when no selection is given.
_DEFAULT_SAMPLE_SIZE = 3

#: How many routing reasons to list under a message.
_MAX_ROUTING_REASONS = 6

#: Signed strength below which a signal is shown as saying nothing.
_NEGLIGIBLE_PUSH = 0.05

#: Bars used to render the strongest possible push.
_MAX_PUSH_BARS = 3


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #


def _heading(title: str) -> None:
    """Print a titled section rule."""
    print(f"\n{'=' * _RULE_WIDTH}\n{title}\n{'=' * _RULE_WIDTH}")


def _subheading(title: str) -> None:
    """Print a minor section rule."""
    print(f"\n{title}\n{'-' * len(title)}")


def _field(label: str, value: object) -> None:
    """Print one aligned ``label: value`` line."""
    print(f"  {label:<26} {value}")


# --------------------------------------------------------------------------- #
# Phase 1: dataset
# --------------------------------------------------------------------------- #


def print_schema_summary() -> None:
    """Print the declared schema: columns, keys and relationships.

    Reads :mod:`src.data.schema`, so it needs no dataset on disk and stays
    correct by construction as the registry evolves.
    """
    _heading("DATASET SCHEMA")
    for spec in schema.TABLES.values():
        flag = "" if spec.required else "  (optional)"
        print(f"\n{spec.filename}{flag}")
        print(f"  {spec.description}")
        _field("primary key", " + ".join(spec.primary_key))

        columns = ", ".join(
            f"{c.name}:{c.type.value}{'?' if c.nullable else ''}" for c in spec.columns
        )
        _field("columns", columns)

        if spec.foreign_keys:
            refs = ", ".join(
                f"{fk.column} -> {fk.target_table}.{fk.target_column}"
                for fk in spec.foreign_keys
            )
            _field("references", refs)

    print(
        "\n  Legend: '?' marks a nullable column. Indexes are built on every "
        "primary\n  key and every foreign key listed above."
    )


def print_dataset_summary(repo: DataRepository) -> None:
    """Print record counts, validation outcome and index sizes."""
    _heading("PHASE 1 - DATA LAYER")
    print(repo.describe())

    report = repo.validation_report
    if report is not None and report.warnings:
        print("\nValidation warnings:")
        for issue in report.warnings:
            print(f"  - {issue}")


def run_data_lookups(repo: DataRepository) -> None:
    """Exercise the repository helpers against ids discovered from the data."""
    _heading("PHASE 1 - REPOSITORY LOOKUPS")

    user_id = max(
        repo.index.history_by_user,
        key=lambda uid: len(repo.index.history_by_user[uid]),
    )
    user = repo.get_user(user_id)
    assert user is not None, "index keys must resolve in users_by_id"

    _subheading("User")
    _field("user_id", user.user_id)
    _field("quiet hours", user.quiet_hours)
    _field("groups", len(repo.get_user_groups(user.user_id)))
    _field("history rows", len(repo.get_user_history(user.user_id)))
    for record in repo.get_user_history(user.user_id, limit=2, newest_first=True):
        print(
            f"    {record.message_id}  {record.created_at:%Y-%m-%d %H:%M}  "
            f"{truncate(record.message_text, _TEXT_PREVIEW)}"
        )

    group_id = max(
        repo.index.group_members_by_group,
        key=lambda gid: len(repo.index.group_members_by_group[gid]),
    )
    group = repo.get_group(group_id)
    assert group is not None, "index keys must resolve in groups_by_id"

    _subheading("Group")
    _field("group_id", f"{group.group_id} ({group.group_type})")
    _field("membership rows", len(repo.get_group_members(group_id)))
    _field("admins", [m.user_id for m in repo.get_group_admins(group_id)])

    _subheading("Media")
    for media_type in ("image", "voice"):
        message = next(
            (m for m in repo.get_messages() if m.media_type == media_type), None
        )
        if message is None:
            print(f"  no {media_type} message present")
            continue
        path = repo.get_media_path(message)
        size = path.stat().st_size if path is not None and path.is_file() else None
        _field(
            f"{media_type} ({message.media_id})",
            f"{size} bytes" if size is not None else "MISSING",
        )


# --------------------------------------------------------------------------- #
# Phase 2: analysis
# --------------------------------------------------------------------------- #


def print_analysis(analysis: MessageAnalysis, repo: DataRepository) -> None:
    """Print one message's features and classification in full."""
    features = analysis.features
    result = analysis.classification
    message = repo.get_message(features.message_id)

    _subheading(f"Message {features.message_id}")

    body = message.message_text if message else None
    _field("recipient", features.user_id)
    _field("conversation", features.conversation_type)
    _field(
        "sender",
        features.sender_user_id or features.business_id or features.group_id or "-",
    )
    _field("received", f"{features.created_at:%Y-%m-%d %H:%M}")
    if body:
        print(f"  {'body':<26} {truncate(body, _BODY_PREVIEW)}")
    else:
        _field("body", f"(no text - {features.media_type or 'none'} attachment)")

    print("\n  Features")
    text = features.text
    _field(
        "  text",
        f"{text.length} chars, {text.word_count} words, "
        f"{text.sentence_count} sentence(s), {text.unique_token_count} unique tokens",
    )
    _field(
        "  signals in text",
        f"digits={text.digit_count} upper={text.uppercase_ratio:.0%} "
        f"punct={text.punctuation_count} emoji={text.emoji_count}",
    )
    _field(
        "  contains",
        _flags(
            url=text.contains_url,
            email=text.contains_email,
            phone=text.contains_phone_number,
            currency=text.contains_currency,
            payment=text.contains_payment_symbol,
            media=features.has_media,
        ),
    )
    if text.urls:
        _field("  links", ", ".join(text.urls))

    context = features.context
    _field(
        "  context",
        f"forwarded={context.forwarded_count} quiet_hours={context.in_quiet_hours} "
        f"group_muted={context.group_muted} sender_admin={context.sender_is_admin}",
    )
    if context.business_exists:
        _field(
            "  business",
            f"verified={context.business_verified} "
            f"domain_matches={context.business_domain_matches} "
            f"age={context.business_age_days}d reports={context.business_reports_30d}",
        )
    if context.group_exists:
        _field(
            "  group",
            f"size={context.group_size} volume_30d={context.group_message_volume_30d}",
        )

    history = features.history
    _field(
        "  history",
        f"interactions={history.total_interactions} "
        f"from_sender={history.sender_message_count} "
        f"from_group={history.group_message_count} "
        f"from_business={history.business_message_count}",
    )
    _field(
        "  reception",
        f"open={history.open_rate:.0%} reply={history.reply_rate:.0%} "
        f"dismiss={history.dismiss_rate:.0%} report={history.report_rate:.0%}",
    )
    _field(
        "  engagement",
        f"user={history.user_engagement:.2f} group={history.group_engagement:.2f} "
        f"business={history.business_engagement:.2f}",
    )
    _field(
        "  notification load",
        f"{context.avg_daily_notifications:.1f}/day, "
        f"{context.notification_dismiss_rate:.0%} dismissed",
    )

    print("\n  Keywords")
    if features.keywords.total_matches == 0:
        _field("  matched", "(none)")
    else:
        for category in features.keywords.categories:
            _field(f"  {category.value}", ", ".join(features.keywords.words(category)))

    print("\n  Classification")
    _field("  message_type", result.message_type.value.upper())
    _field("  confidence", f"{result.confidence:.2f}")
    _field("  runner-up", f"{result.runner_up.value if result.runner_up else '-'} "
                          f"(margin {result.margin:.2f})")
    _field("  reason", result.classification_reason)
    _field(
        "  scores",
        ", ".join(
            f"{category.value}={score:.2f}"
            for category, score in sorted(result.scores.items(), key=lambda i: -i[1])
        )
        or "(none)",
    )

    if analysis.routing is not None:
        print_routing_signals(analysis.routing)


def print_routing_decision(result: RoutingResult) -> None:
    """Print the Phase 4 decision, its evidence, reason and confidence."""
    decision = result.decision
    print("\n  Routing decision (Phase 4)")

    if decision is not None:
        print(f"\n    {'rule':<34}{'argues for':>12}{'weight':>9}")
        print(f"    {'-' * 55}")
        for outcome in sorted(decision.outcomes, key=lambda o: -o.weight):
            flag = "  <-- override" if outcome.override else ""
            print(
                f"    {outcome.rule:<34}{outcome.action.value:>12}"
                f"{outcome.weight:>9.2f}{flag}"
            )
        totals = ", ".join(
            f"{action.value}={score:.2f}"
            for action, score in sorted(decision.scores.items(), key=lambda i: -i[1])
        )
        _field("  totals", totals)
        _field(
            "  runner-up",
            f"{decision.runner_up.value if decision.runner_up else '-'} "
            f"(margin {decision.margin:.2f})",
        )

    print()
    _field("  ACTION", result.action.value.upper())
    _field("  confidence", f"{result.confidence:.2f}")
    _field("  evidence", result.evidence_message_ids)
    if result.evidence.rationale:
        _field("  evidence basis", result.evidence.rationale)
    _field("  reason", result.reason.text)

    print("\n  Submission row")
    for column, value in result.to_output_row().items():
        _field(f"  {column}", value)


def print_routing_signals(signals: RoutingSignals) -> None:
    """Print every routing signal with its score, confidence and direction.

    Phase 3 output only. No routing decision is made or implied; the arrows
    show which way each signal argues, not what will be done about it.
    """
    print("\n  Routing signals (Phase 3 - inputs to a Phase 4 decision)")
    print(
        f"    {'signal':<24}{'score':>7}{'conf':>7}{'push':>8}   direction / evidence"
    )
    print(f"    {'-' * 68}")

    for signal in signals.all_signals:
        push = signal.signed_strength
        print(
            f"    {signal.name:<24}{signal.score:>7.2f}{signal.confidence:>7.2f}"
            f"{push:>+8.2f}   {_push_marker(push)} {_push_note(signal, push)}"
        )

    boosting = signals.boosting
    suppressing = signals.suppressing
    print(
        f"\n    {len(boosting)} signal(s) argue for sooner, "
        f"{len(suppressing)} for later; "
        f"{len(signals.all_signals) - len(boosting) - len(suppressing)} neutral"
    )

    if signals.reasons:
        print("\n  Why")
        for reason in signals.reasons[:_MAX_ROUTING_REASONS]:
            print(f"    - {reason}")


def _push_marker(push: float) -> str:
    """Return a small bar showing how hard a signal pushes, and which way."""
    if abs(push) < _NEGLIGIBLE_PUSH:
        return " . "
    magnitude = min(_MAX_PUSH_BARS, max(1, round(abs(push) * _MAX_PUSH_BARS)))
    return ("^" if push > 0 else "v") * magnitude


def _push_note(signal: RoutingSignal, push: float) -> str:
    """Describe what this signal is currently arguing for.

    Reports the effect rather than the polarity: a boosting signal with a low
    score is arguing for *later*, and saying "raises priority" there would be
    actively misleading.
    """
    if signal.confidence == 0.0:
        return "not applicable to this message"
    if abs(push) < _NEGLIGIBLE_PUSH:
        return "says nothing either way"
    return "argues for sooner" if push > 0 else "argues for later"


def _flags(**named: bool) -> str:
    """Render only the flags that are set, or ``(none)``."""
    present = [name for name, value in named.items() if value]
    return ", ".join(present) if present else "(none)"


def print_analysis_summary(analyses: tuple[MessageAnalysis, ...]) -> None:
    """Print aggregate classification statistics across many messages."""
    _heading("PHASE 2 - CLASSIFICATION SUMMARY")

    types = Counter(a.classification.message_type.value for a in analyses)
    print(f"  {len(analyses)} message(s) analysed\n")
    width = max(len(name) for name in types) if types else 0
    for name, count in types.most_common():
        bar = "#" * count
        print(f"  {name:<{width}}  {count:>3}  {bar}")

    confidences = [a.classification.confidence for a in analyses]
    print(
        f"\n  confidence  min={min(confidences):.2f}  "
        f"mean={sum(confidences) / len(confidences):.2f}  max={max(confidences):.2f}"
    )

    keyword_hits = Counter(
        category.value
        for a in analyses
        for category in a.features.keywords.categories
    )
    print(f"  keyword families hit: {dict(keyword_hits.most_common())}")

    silent = sum(1 for a in analyses if a.features.is_empty_text)
    risky = sum(1 for a in analyses if a.classification.is_risk)
    print(f"  media-only messages: {silent}   risk-flagged (scam/spam): {risky}")

    routed = [a.routing for a in analyses if a.routing is not None]
    if routed:
        print_signal_summary(routed)


def print_signal_summary(routed: Sequence[RoutingSignals]) -> None:
    """Print the distribution of each routing signal across many messages."""
    _heading("PHASE 3 - ROUTING SIGNAL SUMMARY")
    print(
        f"  {'signal':<24}{'mean':>7}{'min':>7}{'max':>7}{'conf':>7}"
        f"{'boost':>7}{'suppr':>7}"
    )
    print(f"  {'-' * 66}")

    for name in (signal.name for signal in routed[0].all_signals):
        values = [s.by_name(name) for s in routed]
        scores = [signal.score for signal in values if signal is not None]
        confidences = [signal.confidence for signal in values if signal is not None]
        pushes = [signal.signed_strength for signal in values if signal is not None]
        print(
            f"  {name:<24}{sum(scores) / len(scores):>7.2f}{min(scores):>7.2f}"
            f"{max(scores):>7.2f}{sum(confidences) / len(confidences):>7.2f}"
            f"{sum(1 for p in pushes if p > 0):>7}{sum(1 for p in pushes if p < 0):>7}"
        )

    print(
        "\n  These are inputs, not decisions. Phase 4 combines them into "
        "notify / digest / mute."
    )


def _select_messages(
    repo: DataRepository, args: argparse.Namespace
) -> tuple[MessageRecord, ...]:
    """Choose which messages the demo analyses, based on the CLI arguments."""
    messages = repo.get_messages()
    if args.message:
        chosen = repo.get_message(args.message)
        if chosen is None:
            raise SystemExit(f"No such message: {args.message}")
        return (chosen,)
    if args.all:
        return messages
    if args.limit:
        return messages[: args.limit]
    return _representative_sample(messages)


def _representative_sample(
    messages: tuple[MessageRecord, ...],
) -> tuple[MessageRecord, ...]:
    """Pick one message per conversation type, so the demo shows real variety.

    Chosen from the data rather than hardcoded, so the demo keeps working if
    the dataset is swapped.
    """
    chosen: list[MessageRecord] = []
    for conversation_type in config.CONVERSATION_TYPES:
        match = next(
            (m for m in messages if m.conversation_type == conversation_type), None
        )
        if match is not None:
            chosen.append(match)
    return tuple(chosen[:_DEFAULT_SAMPLE_SIZE]) or messages[:_DEFAULT_SAMPLE_SIZE]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Message Notification Router - Phase 1 data layer and Phase 2 "
            "feature extraction and classification."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=f"Dataset directory (default: {config.DATASET_DIR}).",
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
        help="Treat validation warnings as blocking errors.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation. Only useful when deliberately inspecting broken data.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Print the schema summary and exit without loading the dataset.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Run the Phase 1 data-layer checks and stop.",
    )
    parser.add_argument(
        "--no-personalize",
        action="store_true",
        help="Skip Phase 3 routing signals and show Phase 2 output only.",
    )
    parser.add_argument(
        "--no-route",
        action="store_true",
        help="Skip the Phase 4 routing decision and stop after routing signals.",
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--message", metavar="ID", help="Analyse one message by id, e.g. msg_005."
    )
    selection.add_argument(
        "--limit", type=int, metavar="N", help="Analyse the first N messages."
    )
    selection.add_argument(
        "--all", action="store_true", help="Analyse every incoming message."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the demo.

    Returns:
        ``0`` on success, ``1`` if the dataset could not be loaded or validated.
    """
    args = parse_args(argv)
    config.configure_logging(args.log_level)

    if args.schema_only:
        print_schema_summary()
        return 0

    try:
        repo = DataRepository.load(
            args.dataset,
            validate=not args.no_validate,
            strict=args.strict or None,
        )
    except DatasetError as exc:
        _LOGGER.error("Could not load the dataset: %s", exc)
        return 1

    print_dataset_summary(repo)
    run_data_lookups(repo)

    if args.data_only:
        _heading("RESULT")
        print("  Phase 1 data layer is ready. No exceptions raised.")
        return 0

    route = not (args.no_personalize or args.no_route)
    pipeline = MessagePipeline(repo, personalize=not args.no_personalize)
    router = RoutingPipeline(repo, analysis_pipeline=pipeline) if route else None

    selected = _select_messages(repo, args)
    analyses = pipeline.analyse_many(selected)

    _heading(_analysis_heading(args))
    detailed = analyses if len(analyses) <= _DEFAULT_SAMPLE_SIZE else analyses[:1]
    results: list[RoutingResult] = []

    for analysis in detailed:
        print_analysis(analysis, repo)
        if router is not None:
            result = router.route_analysis(analysis)
            results.append(result)
            print_routing_decision(result)

    if router is not None and len(analyses) > len(detailed):
        results = list(router.route_many(selected))

    if len(analyses) > 1:
        print_analysis_summary(analyses)
        if results:
            print_routing_summary(results)

    _heading("RESULT")
    phases = "Phases 1 + 2 + 3 + 4" if route else _partial_phase_label(args)
    print(f"  {phases} complete. {len(analyses)} message(s) processed, no exceptions raised.")
    if route:
        print("  Every message has an action, a reason, a confidence and its evidence.")
        print("  output.csv is not written here: exporting is Phase 5.")
    return 0


def _analysis_heading(args: argparse.Namespace) -> str:
    """Return the heading matching the phases actually being run."""
    if args.no_personalize:
        return "PHASE 2 - FEATURES AND CLASSIFICATION"
    if args.no_route:
        return "PHASE 2 + 3 - FEATURES, CLASSIFICATION AND ROUTING SIGNALS"
    return "PHASES 2-4 - FEATURES, CLASSIFICATION, SIGNALS AND ROUTING"


def _partial_phase_label(args: argparse.Namespace) -> str:
    """Return the phase label when routing is switched off."""
    return "Phases 1 + 2" if args.no_personalize else "Phases 1 + 2 + 3"


def print_routing_summary(results: Sequence[RoutingResult]) -> None:
    """Print the action distribution and decision quality across many messages."""
    _heading("PHASE 4 - ROUTING SUMMARY")

    actions = Counter(result.action.value for result in results)
    width = max((len(name) for name in actions), default=0)
    print(f"  {len(results)} message(s) routed\n")
    for name, count in actions.most_common():
        print(f"  {name:<{width}}  {count:>3}  {'#' * count}")

    confidences = [result.confidence for result in results]
    with_evidence = sum(1 for result in results if result.evidence.has_evidence)
    overridden = sum(
        1 for result in results if result.decision and result.decision.overridden
    )
    print(
        f"\n  confidence  min={min(confidences):.2f}  "
        f"mean={sum(confidences) / len(confidences):.2f}  max={max(confidences):.2f}"
    )
    print(f"  with supporting evidence: {with_evidence}/{len(results)}")
    print(f"  safety overrides applied: {overridden}")

    by_type = Counter(
        (result.message_type, result.action.value) for result in results
    )
    print("\n  message_type -> action")
    for (message_type, action), count in sorted(by_type.items()):
        print(f"    {message_type:<18}{action:<9}{count:>4}")


if __name__ == "__main__":
    sys.exit(main())

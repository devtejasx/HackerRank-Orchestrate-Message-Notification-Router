"""Console rendering for the command-line interface.

Presentation only. Nothing here loads data, decides anything or changes state;
every function takes already-computed records and prints them. Keeping this
separate from :mod:`src.cli.commands` means the pipeline can be driven from a
notebook or a test without dragging formatting along, and the formatting can
change without risk to the pipeline.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from src import config
from src.data import schema
from src.data.repository import DataRepository
from src.evaluation import EvaluationReport, format_misses
from src.output import OutputValidationReport
from src.personalization.signal_models import RoutingSignal, RoutingSignals
from src.pipeline import MessageAnalysis
from src.routing.models import RoutingResult
from src.utils.helpers import truncate

__all__ = [
    "heading",
    "field",
    "print_schema",
    "print_dataset_summary",
    "print_repository_lookups",
    "print_analysis",
    "print_routing_signals",
    "print_routing_decision",
    "print_analysis_summary",
    "print_signal_summary",
    "print_routing_summary",
    "print_output_validation",
    "print_evaluation",
]

#: Width of the printed section rules.
RULE_WIDTH = 78

#: How much message text to show in a compact listing.
_TEXT_PREVIEW = 70

#: How much message body to show in a full report.
_BODY_PREVIEW = 300

#: How many routing reasons to list under a message.
_MAX_ROUTING_REASONS = 6

#: Signed strength below which a signal is shown as saying nothing.
_NEGLIGIBLE_PUSH = 0.05

#: Bars used to render the strongest possible push.
_MAX_PUSH_BARS = 3


def heading(title: str) -> None:
    """Print a titled section rule."""
    print(f"\n{'=' * RULE_WIDTH}\n{title}\n{'=' * RULE_WIDTH}")


def subheading(title: str) -> None:
    """Print a minor section rule."""
    print(f"\n{title}\n{'-' * len(title)}")


def field(label: str, value: object) -> None:
    """Print one aligned ``label: value`` line."""
    print(f"  {label:<26} {value}")


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #


def print_schema() -> None:
    """Print the declared dataset schema: columns, keys and relationships.

    Reads the schema registry, so it needs no dataset on disk and stays
    correct by construction as the registry evolves.
    """
    heading("DATASET SCHEMA")
    for spec in schema.TABLES.values():
        flag = "" if spec.required else "  (optional)"
        print(f"\n{spec.filename}{flag}")
        print(f"  {spec.description}")
        field("primary key", " + ".join(spec.primary_key))

        columns = ", ".join(
            f"{c.name}:{c.type.value}{'?' if c.nullable else ''}" for c in spec.columns
        )
        field("columns", columns)

        if spec.foreign_keys:
            refs = ", ".join(
                f"{fk.column} -> {fk.target_table}.{fk.target_column}"
                for fk in spec.foreign_keys
            )
            field("references", refs)

    print(
        "\n  Legend: '?' marks a nullable column. Indexes are built on every "
        "primary\n  key and every foreign key listed above."
    )


def print_dataset_summary(repo: DataRepository) -> None:
    """Print record counts, validation outcome and index sizes."""
    heading("PHASE 1 - DATA LAYER")
    print(repo.describe())

    report = repo.validation_report
    if report is not None and report.warnings:
        print("\nValidation warnings:")
        for issue in report.warnings:
            print(f"  - {issue}")


def print_repository_lookups(repo: DataRepository) -> None:
    """Exercise the repository helpers against ids discovered from the data."""
    heading("PHASE 1 - REPOSITORY LOOKUPS")

    user_id = max(
        repo.index.history_by_user,
        key=lambda uid: len(repo.index.history_by_user[uid]),
    )
    user = repo.get_user(user_id)
    if user is None:  # pragma: no cover - index keys always resolve
        return

    subheading("User")
    field("user_id", user.user_id)
    field("quiet hours", user.quiet_hours)
    field("groups", len(repo.get_user_groups(user.user_id)))
    field("history rows", len(repo.get_user_history(user.user_id)))
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
    if group is not None:
        subheading("Group")
        field("group_id", f"{group.group_id} ({group.group_type})")
        field("membership rows", len(repo.get_group_members(group_id)))
        field("admins", [m.user_id for m in repo.get_group_admins(group_id)])

    subheading("Media")
    for media_type in config.MEDIA_TYPES:
        message = next(
            (m for m in repo.get_messages() if m.media_type == media_type), None
        )
        if message is None:
            print(f"  no {media_type} message present")
            continue
        path = repo.get_media_path(message)
        size = path.stat().st_size if path is not None and path.is_file() else None
        field(
            f"{media_type} ({message.media_id})",
            f"{size} bytes" if size is not None else "MISSING",
        )


# --------------------------------------------------------------------------- #
# Phases 2 and 3
# --------------------------------------------------------------------------- #


def print_analysis(analysis: MessageAnalysis, repo: DataRepository) -> None:
    """Print one message's features and classification in full."""
    features = analysis.features
    result = analysis.classification
    message = repo.get_message(features.message_id)

    subheading(f"Message {features.message_id}")

    body = message.message_text if message else None
    field("recipient", features.user_id)
    field("conversation", features.conversation_type)
    field(
        "sender",
        features.sender_user_id or features.business_id or features.group_id or "-",
    )
    field("received", f"{features.created_at:%Y-%m-%d %H:%M}")
    if body:
        print(f"  {'body':<26} {truncate(body, _BODY_PREVIEW)}")
    else:
        field("body", f"(no text - {features.media_type or 'none'} attachment)")

    _print_feature_block(analysis)
    _print_media_block(analysis)
    _print_keyword_block(analysis)
    _print_classification_block(result)

    if analysis.routing is not None:
        print_routing_signals(analysis.routing)


def _print_media_block(analysis: MessageAnalysis) -> None:
    """Print what was attached, and what could be read out of it.

    Skipped entirely for text-only messages. For the rest it states plainly
    whether the content was recovered or the decision was made blind, which is
    the thing an inspector most needs to know about a voice note.
    """
    media = analysis.features.media
    if not media.has_attachment:
        return

    print("\n  Media")
    field("  attachment", f"{media.media_type or 'unknown type'} {media.media_id}")
    if not media.is_registered:
        field("  status", "not in the media registry")
    elif not media.file_exists:
        field("  status", "registered, but the file is missing from disk")
    else:
        field("  status", "located on disk")

    if media.has_derived_text:
        field(
            "  recovered",
            f"{truncate(media.derived_text, _BODY_PREVIEW)}",
        )
        field(
            "  provider",
            f"{media.derived_from} (confidence {media.derived_confidence:.2f})",
        )
    else:
        field(
            "  recovered",
            "nothing - no OCR or speech-to-text provider installed",
        )


def _print_feature_block(analysis: MessageAnalysis) -> None:
    """Print the extracted feature summary."""
    features = analysis.features
    text = features.text
    context = features.context
    history = features.history

    print("\n  Features")
    field(
        "  text",
        f"{text.length} chars, {text.word_count} words, "
        f"{text.sentence_count} sentence(s), {text.unique_token_count} unique tokens",
    )
    field(
        "  signals in text",
        f"digits={text.digit_count} upper={text.uppercase_ratio:.0%} "
        f"punct={text.punctuation_count} emoji={text.emoji_count}",
    )
    field(
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
        field("  links", ", ".join(text.urls))

    field(
        "  context",
        f"forwarded={context.forwarded_count} quiet_hours={context.in_quiet_hours} "
        f"group_muted={context.group_muted} sender_admin={context.sender_is_admin}",
    )
    if context.business_exists:
        field(
            "  business",
            f"verified={context.business_verified} "
            f"domain_matches={context.business_domain_matches} "
            f"age={context.business_age_days}d reports={context.business_reports_30d}",
        )
    if context.group_exists:
        field(
            "  group",
            f"size={context.group_size} volume_30d={context.group_message_volume_30d}",
        )
    field(
        "  history",
        f"interactions={history.total_interactions} "
        f"from_sender={history.sender_message_count} "
        f"from_group={history.group_message_count} "
        f"from_business={history.business_message_count}",
    )
    field(
        "  reception",
        f"open={history.open_rate:.0%} reply={history.reply_rate:.0%} "
        f"dismiss={history.dismiss_rate:.0%} report={history.report_rate:.0%}",
    )
    field(
        "  engagement",
        f"user={history.user_engagement:.2f} group={history.group_engagement:.2f} "
        f"business={history.business_engagement:.2f}",
    )
    field(
        "  notification load",
        f"{context.avg_daily_notifications:.1f}/day, "
        f"{context.notification_dismiss_rate:.0%} dismissed",
    )


def _print_keyword_block(analysis: MessageAnalysis) -> None:
    """Print matched keywords by category."""
    keywords = analysis.features.keywords
    print("\n  Keywords")
    if keywords.total_matches == 0:
        field("  matched", "(none)")
        return
    for category in keywords.categories:
        field(f"  {category.value}", ", ".join(keywords.words(category)))


def _print_classification_block(result) -> None:
    """Print the Phase 2 verdict."""
    print("\n  Classification")
    field("  message_type", result.message_type.value.upper())
    field("  confidence", f"{result.confidence:.2f}")
    field(
        "  runner-up",
        f"{result.runner_up.value if result.runner_up else '-'} "
        f"(margin {result.margin:.2f})",
    )
    field("  reason", result.classification_reason)
    field(
        "  scores",
        ", ".join(
            f"{category.value}={score:.2f}"
            for category, score in sorted(result.scores.items(), key=lambda i: -i[1])
        )
        or "(none)",
    )


def print_routing_signals(signals: RoutingSignals) -> None:
    """Print every routing signal with its score, confidence and direction."""
    print("\n  Routing signals (Phase 3)")
    print(f"    {'signal':<24}{'score':>7}{'conf':>7}{'push':>8}   what it argues")
    print(f"    {'-' * 68}")

    for signal in signals.all_signals:
        push = signal.signed_strength
        print(
            f"    {signal.name:<24}{signal.score:>7.2f}{signal.confidence:>7.2f}"
            f"{push:>+8.2f}   {_push_marker(push)} {_push_note(signal, push)}"
        )

    boosting = len(signals.boosting)
    suppressing = len(signals.suppressing)
    print(
        f"\n    {boosting} signal(s) argue for sooner, {suppressing} for later; "
        f"{len(signals.all_signals) - boosting - suppressing} neutral"
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
    score is arguing for *later*, and saying "raises priority" would mislead.
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


# --------------------------------------------------------------------------- #
# Phase 4
# --------------------------------------------------------------------------- #


def print_routing_decision(result: RoutingResult) -> None:
    """Print the routing decision, its evidence, reason and confidence."""
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
        field("  totals", totals)
        field(
            "  runner-up",
            f"{decision.runner_up.value if decision.runner_up else '-'} "
            f"(margin {decision.margin:.2f})",
        )

    print()
    field("  ACTION", result.action.value.upper())
    field("  confidence", f"{result.confidence:.2f}")
    field("  evidence", result.evidence_message_ids)
    if result.evidence.rationale:
        field("  evidence basis", result.evidence.rationale)
    field("  reason", result.reason.text)

    print("\n  Submission row")
    for column, value in result.to_output_row().items():
        field(f"  {column}", value)


# --------------------------------------------------------------------------- #
# Aggregate summaries
# --------------------------------------------------------------------------- #


def print_analysis_summary(analyses: Sequence[MessageAnalysis]) -> None:
    """Print aggregate classification statistics across many messages."""
    heading("PHASE 2 - CLASSIFICATION SUMMARY")

    types = Counter(a.classification.message_type.value for a in analyses)
    print(f"  {len(analyses)} message(s) analysed\n")
    width = max((len(name) for name in types), default=0)
    for name, count in types.most_common():
        print(f"  {name:<{width}}  {count:>3}  {'#' * count}")

    confidences = [a.classification.confidence for a in analyses]
    print(
        f"\n  confidence  min={min(confidences):.2f}  "
        f"mean={sum(confidences) / len(confidences):.2f}  max={max(confidences):.2f}"
    )

    silent = sum(1 for a in analyses if a.features.is_empty_text)
    risky = sum(1 for a in analyses if a.classification.is_risk)
    print(f"  media-only messages: {silent}   risk-flagged (scam/spam): {risky}")


def print_signal_summary(routed: Sequence[RoutingSignals]) -> None:
    """Print the distribution of each routing signal across many messages."""
    heading("PHASE 3 - ROUTING SIGNAL SUMMARY")
    print(
        f"  {'signal':<24}{'mean':>7}{'min':>7}{'max':>7}{'conf':>7}"
        f"{'boost':>7}{'suppr':>7}"
    )
    print(f"  {'-' * 66}")

    for name in (signal.name for signal in routed[0].all_signals):
        signals = [s.by_name(name) for s in routed]
        scores = [s.score for s in signals if s is not None]
        confidences = [s.confidence for s in signals if s is not None]
        pushes = [s.signed_strength for s in signals if s is not None]
        print(
            f"  {name:<24}{sum(scores) / len(scores):>7.2f}{min(scores):>7.2f}"
            f"{max(scores):>7.2f}{sum(confidences) / len(confidences):>7.2f}"
            f"{sum(1 for p in pushes if p > 0):>7}{sum(1 for p in pushes if p < 0):>7}"
        )


def print_routing_summary(results: Sequence[RoutingResult]) -> None:
    """Print the action distribution and decision quality across many messages."""
    heading("PHASE 4 - ROUTING SUMMARY")

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
    print(f"  distinct reasons: {len({r.reason.text for r in results})}/{len(results)}")

    by_type = Counter((r.message_type, r.action.value) for r in results)
    print("\n  message_type -> action")
    for (message_type, action), count in sorted(by_type.items()):
        print(f"    {message_type:<18}{action:<9}{count:>4}")


# --------------------------------------------------------------------------- #
# Submission
# --------------------------------------------------------------------------- #


def print_output_validation(report: OutputValidationReport) -> None:
    """Print the pre-write validation summary and any findings."""
    heading("OUTPUT VALIDATION")
    for line in report.summary_lines():
        print(f"  {line}")

    if report.issues:
        print()
        for issue in report.issues:
            print(f"  {issue}")
    else:
        print("\n  All output checks passed.")


def print_evaluation(report: EvaluationReport) -> None:
    """Print agreement metrics against the labelled examples."""
    heading("EVALUATION AGAINST LABELLED EXAMPLES")
    for line in report.summary_lines():
        print(f"  {line}" if line else "")

    misses = format_misses(report)
    if misses:
        print("\n  Disagreements:")
        for line in misses:
            print(f"    {line}")

    confusion = report.action_confusion()
    if confusion:
        print("\n  Action confusion (expected -> predicted):")
        for (expected, predicted), count in sorted(
            confusion.items(), key=lambda item: -item[1]
        ):
            print(f"    {expected:<8} -> {predicted:<8} x{count}")

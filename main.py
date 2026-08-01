"""Hour 1 entry point: prove the data layer loads, validates, indexes and reads.

This is a smoke test, not the solution. It performs no classification, no
routing and writes no output file - those belong to later phases.

Usage:
    python main.py                 # schema summary, load, validate, index, lookups
    python main.py --schema-only   # print the schema summary and stop
    python main.py --strict        # treat validation warnings as failures
    python main.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import config
from src.data import schema
from src.data.loader import DatasetError
from src.data.models import MessageRecord
from src.data.repository import DataRepository
from src.utils.helpers import truncate

_LOGGER = config.get_logger("main")

#: Width of the printed section rules.
_RULE_WIDTH = 78

#: How much message text to show in the demo lookups.
_TEXT_PREVIEW = 70


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #


def _heading(title: str) -> None:
    """Print a titled section rule."""
    print(f"\n{'=' * _RULE_WIDTH}\n{title}\n{'=' * _RULE_WIDTH}")


def _field(label: str, value: object) -> None:
    """Print one aligned ``label: value`` line."""
    print(f"  {label:<26} {value}")


# --------------------------------------------------------------------------- #
# Steps
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
    _heading("LOAD SUMMARY")
    print(repo.describe())

    report = repo.validation_report
    if report is not None and report.warnings:
        print("\nValidation warnings:")
        for issue in report.warnings:
            print(f"  - {issue}")


def run_smoke_lookups(repo: DataRepository) -> None:
    """Exercise every helper against ids discovered from the data itself.

    Nothing here is hardcoded to a particular id, so the smoke test keeps
    working if the dataset is swapped or extended.
    """
    _heading("SMOKE LOOKUPS")

    _lookup_user(repo)
    _lookup_group(repo)
    _lookup_business(repo)
    _lookup_message_and_media(repo)
    _lookup_negative_lookups(repo)


def _lookup_user(repo: DataRepository) -> None:
    """Find a well-connected user and read their context."""
    user_id = max(
        repo.index.history_by_user,
        key=lambda uid: len(repo.index.history_by_user[uid]),
    )
    user = repo.get_user(user_id)
    assert user is not None, "index keys must resolve in users_by_id"

    print("\n-- User --")
    _field("user_id", user.user_id)
    _field("quiet hours", user.quiet_hours)
    _field("opened / replied 30d", f"{user.messages_opened_30d} / {user.messages_replied_30d}")
    _field("groups", len(repo.get_user_groups(user.user_id)))
    _field("business relationships", len(repo.get_user_businesses(user.user_id)))
    _field("incoming messages", len(repo.get_user_messages(user.user_id)))

    history = repo.get_user_history(user.user_id, limit=3, newest_first=True)
    _field("history rows", len(repo.get_user_history(user.user_id)))
    print("  most recent history:")
    for record in history:
        event = repo.get_message_event(record.message_id)
        reaction = "no event" if event is None else f"opened={event.message_opened}"
        print(
            f"    {record.message_id}  {record.created_at:%Y-%m-%d %H:%M}  "
            f"{record.conversation_type:<9} {reaction:<14} "
            f"{truncate(record.message_text, _TEXT_PREVIEW)}"
        )

    summary = repo.get_notification_summary(user.user_id)
    if summary:
        busiest = max(summary, key=lambda row: row.notifications_sent)
        _field(
            "busiest day",
            f"{busiest.date} sent={busiest.notifications_sent} "
            f"dismissed={busiest.notifications_dismissed}",
        )


def _lookup_group(repo: DataRepository) -> None:
    """Find the largest group and read its membership."""
    group_id = max(
        repo.index.group_members_by_group,
        key=lambda gid: len(repo.index.group_members_by_group[gid]),
    )
    group = repo.get_group(group_id)
    assert group is not None, "index keys must resolve in groups_by_id"

    members = repo.get_group_members(group_id)
    admins = repo.get_group_admins(group_id)
    muted = sum(1 for member in members if member.group_muted_by_user)

    print("\n-- Group --")
    _field("group_id", group.group_id)
    _field("name / type", f"{group.group_name} ({group.group_type})")
    _field("declared members", group.member_count)
    _field("membership rows", len(members))
    _field("admins", f"{len(admins)} -> {[m.user_id for m in admins]}")
    _field("muted by", f"{muted} member(s)")
    _field("history rows", len(repo.get_group_history(group_id)))

    sample_member = members[0]
    _field(
        "example membership",
        f"{sample_member.user_id} role={sample_member.role} "
        f"joined={sample_member.joined_at} muted={sample_member.group_muted_by_user}",
    )


def _lookup_business(repo: DataRepository) -> None:
    """Find the busiest business sender and read its reputation signals."""
    business_id = max(
        repo.index.history_by_business,
        key=lambda bid: len(repo.index.history_by_business[bid]),
    )
    business = repo.get_business(business_id)
    assert business is not None, "index keys must resolve in business_by_id"

    print("\n-- Business --")
    _field("business_id", business.business_id)
    _field("display / brand", f"{business.display_name} / {business.brand_name}")
    _field("category", business.category)
    _field("verified", business.verified)
    _field("official domain", business.official_domain)
    _field("sender domain", business.domain_used_by_sender)
    _field("domain matches official", business.sender_domain_matches_official)
    _field("account age (days)", business.account_age_days)
    _field("user reports 30d", business.user_reports_30d)
    _field("users with a relationship", len(repo.get_business_users(business_id)))
    _field("history rows", len(repo.get_business_history(business_id)))


def _lookup_message_and_media(repo: DataRepository) -> None:
    """Read one image message and one voice message end to end."""
    print("\n-- Media --")
    for media_type in ("image", "voice"):
        message = _first_with_media(repo, media_type)
        if message is None:
            print(f"  no {media_type} message present")
            continue

        path = repo.get_media_path(message)
        metadata = (
            repo.get_image(message.media_id)
            if media_type == "image"
            else repo.get_voice(message.media_id)
        )
        size = path.stat().st_size if path is not None and path.is_file() else None

        _field(f"{media_type} message", message.message_id)
        _field("  media_id", message.media_id)
        _field("  metadata row", metadata)
        _field("  resolved path", path)
        _field("  on disk", f"{size} bytes" if size is not None else "MISSING")


def _first_with_media(repo: DataRepository, media_type: str) -> MessageRecord | None:
    """Return the first incoming message carrying ``media_type``, if any."""
    return next(
        (m for m in repo.get_messages() if m.media_type == media_type),
        None,
    )


def _lookup_negative_lookups(repo: DataRepository) -> None:
    """Confirm unknown ids degrade to ``None`` / empty rather than raising."""
    unknown = "does_not_exist"
    print("\n-- Unknown ids --")
    _field("get_user", repo.get_user(unknown))
    _field("get_group", repo.get_group(unknown))
    _field("get_business", repo.get_business(unknown))
    _field("get_message", repo.get_message(unknown))
    _field("get_image", repo.get_image(unknown))
    _field("get_voice", repo.get_voice(unknown))
    _field("get_group_members", repo.get_group_members(unknown))
    _field("get_user_history", repo.get_user_history(unknown))
    _field("get_message_events", repo.get_message_events(unknown))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Hour 1 data-layer smoke test for the Message Notification Router.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the smoke test.

    Returns:
        ``0`` on success, ``1`` if the dataset could not be loaded or validated.
    """
    args = parse_args(argv)
    config.configure_logging(args.log_level)

    print_schema_summary()
    if args.schema_only:
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
    run_smoke_lookups(repo)

    _heading("RESULT")
    print("  Hour 1 data layer is ready. No exceptions raised.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

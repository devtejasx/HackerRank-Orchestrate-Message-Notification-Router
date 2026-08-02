"""Declarative description of every CSV in the dataset.

This module is the single source of truth for the dataset's shape. The loader,
the validator and the index builder all read from :data:`TABLES` rather than
restating column names, so adding a column or a relationship is a one-line
change in exactly one place.

The specs below were derived by profiling the shipped CSVs, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from src import config


class ColumnType(StrEnum):
    """Logical type of a CSV column, independent of pandas dtypes."""

    TEXT = "text"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    TIMESTAMP = "timestamp"
    DATE = "date"


@dataclass(frozen=True, slots=True)
class Column:
    """One column of one table.

    Attributes:
        name: Column header exactly as it appears in the CSV.
        type: Logical type used for coercion and validation.
        nullable: Whether an empty value is legitimate for this column.
        allowed: Optional closed set of permitted values. Checked as a warning,
            so unseen-but-valid future values do not break the pipeline.
    """

    name: str
    type: ColumnType = ColumnType.TEXT
    nullable: bool = False
    allowed: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ForeignKey:
    """A reference from ``column`` to ``target_table.target_column``."""

    column: str
    target_table: str
    target_column: str


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Everything the data layer needs to know about a single CSV."""

    name: str
    filename: str
    primary_key: tuple[str, ...]
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()
    required: bool = True
    #: Whether the run genuinely cannot proceed unless this table has rows.
    #:
    #: Distinct from :attr:`required`, which is about the *file* existing. Only
    #: ``messages`` and ``users`` are essential: without them there is nothing
    #: to predict and nobody to predict for. Every other table enriches the
    #: decision, and an evaluation set that ships without history - a cold-start
    #: scenario is a perfectly reasonable thing to test - must still produce a
    #: full submission rather than none at all.
    requires_rows: bool = False
    description: str = ""
    label: str = ""

    @property
    def display_label(self) -> str:
        """Human-readable table name used in logs and summaries."""
        return self.label or self.name.replace("_", " ").title()

    @property
    def column_names(self) -> tuple[str, ...]:
        """Header names in declaration order."""
        return tuple(column.name for column in self.columns)

    @property
    def timestamp_columns(self) -> tuple[str, ...]:
        """Columns holding a date or datetime."""
        return tuple(
            column.name
            for column in self.columns
            if column.type in (ColumnType.TIMESTAMP, ColumnType.DATE)
        )

    def column(self, name: str) -> Column | None:
        """Return the named column spec, or ``None`` when it is not declared."""
        return next((c for c in self.columns if c.name == name), None)


# --------------------------------------------------------------------------- #
# Shared column groups
# --------------------------------------------------------------------------- #

_CONVERSATION_TYPE_VALUES: Final[tuple[str, ...]] = config.CONVERSATION_TYPES
_MEDIA_TYPE_VALUES: Final[tuple[str, ...]] = config.MEDIA_TYPES


def _conversation_columns() -> tuple[Column, ...]:
    """Columns shared verbatim by ``messages``, ``message_history`` and ``sample_messages``.

    All three files describe "a message delivered to a user", so they carry an
    identical envelope. Defining it once keeps the three specs in lockstep.
    """
    return (
        Column("message_id"),
        Column("user_id"),
        Column("conversation_type", allowed=_CONVERSATION_TYPE_VALUES),
        Column("group_id", nullable=True),
        Column("business_id", nullable=True),
        Column("sender_user_id", nullable=True),
        Column("created_at", ColumnType.TIMESTAMP),
        Column("message_text", nullable=True),
        Column("media_type", nullable=True, allowed=_MEDIA_TYPE_VALUES),
        Column("media_id", nullable=True),
        Column("forwarded_count", ColumnType.INT),
    )


def _conversation_foreign_keys() -> tuple[ForeignKey, ...]:
    """Foreign keys shared by every message-shaped table."""
    return (
        ForeignKey("user_id", "users", "user_id"),
        ForeignKey("group_id", "groups", "group_id"),
        ForeignKey("business_id", "business_accounts", "business_id"),
        ForeignKey("sender_user_id", "users", "user_id"),
    )


# --------------------------------------------------------------------------- #
# Table specs
# --------------------------------------------------------------------------- #

USERS = TableSpec(
    name="users",
    filename="users.csv",
    primary_key=("user_id",),
    requires_rows=True,
    columns=(
        Column("user_id"),
        Column("do_not_disturb_window"),
        Column("messages_opened_30d", ColumnType.INT),
        Column("messages_replied_30d", ColumnType.INT),
        Column("notifications_dismissed_30d", ColumnType.INT),
        Column("messages_reported_30d", ColumnType.INT),
    ),
    description="Per-user notification behaviour and quiet hours.",
)

GROUPS = TableSpec(
    name="groups",
    filename="groups.csv",
    primary_key=("group_id",),
    columns=(
        Column("group_id"),
        Column("group_name"),
        Column("group_type"),
        Column("member_count", ColumnType.INT),
        Column("admin_count", ColumnType.INT),
        Column("created_at", ColumnType.DATE),
        Column("messages_30d", ColumnType.INT),
    ),
    description="Group chat metadata: type, size and recent activity.",
)

GROUP_MEMBERS = TableSpec(
    name="group_members",
    filename="group_members.csv",
    primary_key=("group_id", "user_id"),
    columns=(
        Column("group_id"),
        Column("user_id"),
        Column("role", allowed=config.MEMBER_ROLES),
        Column("joined_at", ColumnType.DATE),
        Column("messages_sent_30d", ColumnType.INT),
        Column("messages_read_30d", ColumnType.INT),
        Column("replies_sent_30d", ColumnType.INT),
        Column("notifications_dismissed_30d", ColumnType.INT),
        Column("group_muted_by_user", ColumnType.BOOL),
    ),
    foreign_keys=(
        ForeignKey("group_id", "groups", "group_id"),
        ForeignKey("user_id", "users", "user_id"),
    ),
    description="How one user relates to one group: role, activity, mute state.",
)

BUSINESS_ACCOUNTS = TableSpec(
    name="business_accounts",
    filename="business_accounts.csv",
    primary_key=("business_id",),
    columns=(
        Column("business_id"),
        Column("display_name"),
        Column("brand_name"),
        Column("category"),
        Column("verified", ColumnType.BOOL),
        # 5 rows ship without an official domain; 1 row without a sender domain.
        Column("official_domain", nullable=True),
        Column("domain_used_by_sender", nullable=True),
        Column("account_age_days", ColumnType.INT),
        Column("messages_sent_30d", ColumnType.INT),
        Column("user_reports_30d", ColumnType.INT),
        Column("domain_used_by_sender_age_days", ColumnType.INT),
    ),
    description="Business sender identity, verification and domain reputation.",
    label="Businesses",
)

USER_BUSINESS_HISTORY = TableSpec(
    name="user_business_history",
    filename="user_business_history.csv",
    primary_key=("user_id", "business_id"),
    columns=(
        Column("user_id"),
        Column("business_id"),
        Column("why_user_knows_account"),
        Column("last_activity_at", ColumnType.TIMESTAMP),
        Column("allows_promotions", ColumnType.BOOL),
        Column("promotions_opted_out_at", ColumnType.TIMESTAMP, nullable=True),
        Column("activity_count_180d", ColumnType.INT),
        Column("messages_opened_30d", ColumnType.INT),
        Column("messages_dismissed_30d", ColumnType.INT),
        Column("messages_replied_30d", ColumnType.INT),
        Column("last_reply_at", ColumnType.TIMESTAMP, nullable=True),
    ),
    foreign_keys=(
        ForeignKey("user_id", "users", "user_id"),
        ForeignKey("business_id", "business_accounts", "business_id"),
    ),
    description="Whether a user has a real relationship with a business.",
    label="User-Business History",
)

MESSAGES = TableSpec(
    name="messages",
    filename="messages.csv",
    primary_key=("message_id",),
    requires_rows=True,
    columns=_conversation_columns(),
    foreign_keys=_conversation_foreign_keys(),
    description="Incoming messages awaiting a routing decision.",
)

MESSAGE_HISTORY = TableSpec(
    name="message_history",
    filename="message_history.csv",
    primary_key=("message_id",),
    columns=_conversation_columns(),
    foreign_keys=_conversation_foreign_keys(),
    description="Past messages received by users. Same envelope as messages.csv.",
    label="History",
)

MESSAGE_EVENTS = TableSpec(
    name="message_events",
    filename="message_events.csv",
    primary_key=("message_id",),
    columns=(
        Column("user_id"),
        Column("message_id"),
        Column("message_opened", ColumnType.BOOL),
        Column("message_replied", ColumnType.BOOL),
        Column("reaction_time_minutes", ColumnType.INT, nullable=True),
        Column("notification_dismissed", ColumnType.BOOL),
        Column("muted_after_message", ColumnType.BOOL),
        Column("message_reported", ColumnType.BOOL),
    ),
    foreign_keys=(
        ForeignKey("user_id", "users", "user_id"),
        ForeignKey("message_id", "message_history", "message_id"),
    ),
    description="How users reacted to historical messages. One row per historical message.",
    label="Events",
)

IMAGES = TableSpec(
    name="images",
    filename="images.csv",
    primary_key=("image_id",),
    columns=(Column("image_id"), Column("file_path")),
    description="Image media registry. Paths are relative to the dataset directory.",
    # Optional: the registry only resolves an attachment to a file on disk.
    # Without it, media messages still carry media_type and media_id and are
    # routed on those, so an absent registry degrades the run rather than
    # ending it.
    required=False,
)

VOICE_NOTES = TableSpec(
    name="voice_notes",
    filename="voice_notes.csv",
    primary_key=("voice_note_id",),
    columns=(Column("voice_note_id"), Column("file_path")),
    description="Voice-note media registry. Paths are relative to the dataset directory.",
    #: Optional for the same reason as :data:`IMAGES`.
    required=False,
)

DAILY_NOTIFICATION_SUMMARY = TableSpec(
    name="daily_notification_summary",
    filename="daily_notification_summary.csv",
    primary_key=("user_id", "date"),
    columns=(
        Column("user_id"),
        Column("date", ColumnType.DATE),
        Column("notifications_sent", ColumnType.INT),
        Column("notifications_dismissed", ColumnType.INT),
    ),
    foreign_keys=(ForeignKey("user_id", "users", "user_id"),),
    description="Daily notification load per user.",
    label="Notification Summary",
)

SAMPLE_MESSAGES = TableSpec(
    name="sample_messages",
    filename="sample_messages.csv",
    primary_key=("message_id",),
    columns=(
        *_conversation_columns(),
        Column("action", allowed=config.ROUTING_ACTIONS),
        Column("message_type", allowed=config.MESSAGE_TYPES),
        Column("reason"),
        Column("confidence", ColumnType.FLOAT),
        Column("evidence_message_ids"),
    ),
    foreign_keys=_conversation_foreign_keys(),
    required=False,
    description="Worked examples showing the expected output format. Reference only.",
)


#: Every table the data layer knows how to load, keyed by logical name.
TABLES: Final[dict[str, TableSpec]] = {
    spec.name: spec
    for spec in (
        USERS,
        GROUPS,
        GROUP_MEMBERS,
        BUSINESS_ACCOUNTS,
        USER_BUSINESS_HISTORY,
        MESSAGES,
        MESSAGE_HISTORY,
        MESSAGE_EVENTS,
        IMAGES,
        VOICE_NOTES,
        DAILY_NOTIFICATION_SUMMARY,
        SAMPLE_MESSAGES,
    )
}

#: Tables whose absence makes the data layer unusable.
REQUIRED_TABLE_NAMES: Final[tuple[str, ...]] = tuple(
    name for name, spec in TABLES.items() if spec.required
)

# --------------------------------------------------------------------------- #
# Dataset-specific invariants
# --------------------------------------------------------------------------- #

#: Tables carrying the shared message envelope, so cross-cutting rules
#: (media pairing, conversation_type consistency) apply uniformly.
MESSAGE_SHAPED_TABLES: Final[tuple[str, ...]] = (
    "messages",
    "message_history",
    "sample_messages",
)

#: Which reference column each ``conversation_type`` must populate.
#: Verified against all 552 message rows in the shipped dataset.
CONVERSATION_TYPE_REQUIRED_REFERENCE: Final[dict[str, str]] = {
    "personal": "sender_user_id",
    "group": "group_id",
    "business": "business_id",
}

#: Which reference column each ``conversation_type`` must leave empty.
CONVERSATION_TYPE_FORBIDDEN_REFERENCES: Final[dict[str, tuple[str, ...]]] = {
    "personal": ("group_id", "business_id"),
    "group": ("business_id",),
    "business": ("group_id", "sender_user_id"),
}

#: ``media_type`` value -> (table holding that media, its primary key column).
MEDIA_TYPE_TO_TABLE: Final[dict[str, tuple[str, str]]] = {
    "image": ("images", "image_id"),
    "voice": ("voice_notes", "voice_note_id"),
}


def get_spec(table_name: str) -> TableSpec:
    """Return the spec for ``table_name``.

    Args:
        table_name: Logical table name, e.g. ``"messages"``.

    Raises:
        KeyError: If the table is not declared in :data:`TABLES`.
    """
    try:
        return TABLES[table_name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown table {table_name!r}. Known tables: {sorted(TABLES)}"
        ) from exc

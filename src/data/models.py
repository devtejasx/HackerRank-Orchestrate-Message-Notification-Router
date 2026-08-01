"""Typed records - one frozen dataclass per CSV row.

Field names match CSV headers exactly, which lets a single generic
:meth:`Record.from_row` coerce any row using the declared annotations. That
removes eleven near-identical hand-written constructors.

Records are immutable and hashable so downstream phases can cache and share
them freely without defensive copying.
"""

from __future__ import annotations

import datetime as dt
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from functools import cache
from typing import Any, Self, Union, get_args, get_origin, get_type_hints

from src.utils.helpers import (
    parse_date,
    parse_dnd_window,
    parse_timestamp,
    safe_bool,
    safe_float,
    safe_int,
    safe_text,
)

__all__ = [
    "RecordCoercionError",
    "Record",
    "User",
    "Group",
    "GroupMember",
    "BusinessAccount",
    "UserBusinessHistory",
    "MessageRecord",
    "Message",
    "MessageHistory",
    "SampleMessage",
    "MessageEvent",
    "Image",
    "VoiceNote",
    "NotificationSummary",
    "MODEL_BY_TABLE",
]


class RecordCoercionError(ValueError):
    """Raised when a non-nullable field cannot be built from a row.

    In the normal pipeline :mod:`src.data.validation` runs first and reports
    such rows as issues, so this exception signals that records were built
    from unvalidated or structurally broken data.
    """


#: Coercion function per declared field type. Keys are matched exactly, so
#: ``dt.date`` and ``dt.datetime`` stay distinct despite the subclass relation.
_COERCERS: Mapping[Any, Callable[[Any], Any]] = {
    str: safe_text,
    int: safe_int,
    float: safe_float,
    bool: safe_bool,
    dt.datetime: parse_timestamp,
    dt.date: parse_date,
}


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Split ``T | None`` into ``(T, True)``; leave plain ``T`` as ``(T, False)``."""
    if get_origin(annotation) in (Union, types.UnionType):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return annotation, False


@cache
def _field_plan(cls: type) -> tuple[tuple[str, Callable[[Any], Any], bool], ...]:
    """Return ``(field_name, coercer, is_optional)`` for every field of ``cls``.

    Cached because the annotation resolution is identical for every row of a
    given table and would otherwise run hundreds of times per load.
    """
    hints = get_type_hints(cls)
    plan: list[tuple[str, Callable[[Any], Any], bool]] = []
    for spec in fields(cls):
        base_type, optional = _unwrap_optional(hints[spec.name])
        coercer = _COERCERS.get(base_type)
        if coercer is None:  # pragma: no cover - guards future edits
            raise TypeError(
                f"{cls.__name__}.{spec.name}: no coercer for {base_type!r}"
            )
        plan.append((spec.name, coercer, optional))
    return tuple(plan)


@dataclass(frozen=True, slots=True)
class Record:
    """Base for every dataset record.

    Subclasses declare fields whose names match their CSV headers; this base
    supplies construction from a raw row and conversion back to a plain dict.
    """

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Self:
        """Build a record from a raw CSV/DataFrame row.

        Args:
            row: Mapping of column name to raw cell value. Missing keys are
                treated the same as empty cells.

        Returns:
            A fully coerced, immutable record.

        Raises:
            RecordCoercionError: If a non-nullable field is absent or cannot be
                coerced to its declared type.
        """
        values: dict[str, Any] = {}
        for name, coerce, optional in _field_plan(cls):
            coerced = coerce(row.get(name))
            if coerced is None and not optional:
                raise RecordCoercionError(
                    f"{cls.__name__}.{name} is required but got {row.get(name)!r}"
                )
            values[name] = coerced
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return the record's fields as a plain dictionary."""
        return {spec.name: getattr(self, spec.name) for spec in fields(self)}


# --------------------------------------------------------------------------- #
# Reference entities
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class User(Record):
    """One row of ``users.csv`` - a notification recipient."""

    user_id: str
    do_not_disturb_window: str
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int

    @property
    def quiet_hours(self) -> tuple[dt.time, dt.time] | None:
        """Parsed ``(start, end)`` of the do-not-disturb window.

        The window may wrap past midnight (e.g. ``22:00-07:00``); interpreting
        that is left to the consumer.
        """
        return parse_dnd_window(self.do_not_disturb_window)


@dataclass(frozen=True, slots=True)
class Group(Record):
    """One row of ``groups.csv`` - a group chat."""

    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: dt.date
    messages_30d: int


@dataclass(frozen=True, slots=True)
class GroupMember(Record):
    """One row of ``group_members.csv`` - a user's membership in a group."""

    group_id: str
    user_id: str
    role: str
    joined_at: dt.date
    messages_sent_30d: int
    messages_read_30d: int
    replies_sent_30d: int
    notifications_dismissed_30d: int
    group_muted_by_user: bool

    @property
    def is_admin(self) -> bool:
        """Whether this member administers the group."""
        return self.role == "admin"


@dataclass(frozen=True, slots=True)
class BusinessAccount(Record):
    """One row of ``business_accounts.csv`` - a business sender."""

    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str | None
    domain_used_by_sender: str | None
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int

    @property
    def sender_domain_matches_official(self) -> bool | None:
        """Whether the sending domain equals the brand's official domain.

        Returns:
            ``None`` when either domain is absent, so callers can tell
            "unknown" apart from "mismatch".
        """
        if self.official_domain is None or self.domain_used_by_sender is None:
            return None
        return self.official_domain.casefold() == self.domain_used_by_sender.casefold()


@dataclass(frozen=True, slots=True)
class UserBusinessHistory(Record):
    """One row of ``user_business_history.csv`` - a user/business relationship."""

    user_id: str
    business_id: str
    why_user_knows_account: str
    last_activity_at: dt.datetime
    allows_promotions: bool
    promotions_opted_out_at: dt.datetime | None
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: dt.datetime | None

    @property
    def has_opted_out(self) -> bool:
        """Whether the user has explicitly opted out of promotions."""
        return self.promotions_opted_out_at is not None


# --------------------------------------------------------------------------- #
# Message envelope
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MessageRecord(Record):
    """Fields shared by ``messages.csv``, ``message_history.csv`` and ``sample_messages.csv``.

    Note the two distinct participants: ``user_id`` is the **recipient**,
    ``sender_user_id`` is the **sender**. They must never be conflated.
    """

    message_id: str
    user_id: str
    conversation_type: str
    group_id: str | None
    business_id: str | None
    sender_user_id: str | None
    created_at: dt.datetime
    message_text: str | None
    media_type: str | None
    media_id: str | None
    forwarded_count: int

    @property
    def is_personal(self) -> bool:
        """Whether this is a one-to-one conversation."""
        return self.conversation_type == "personal"

    @property
    def is_group(self) -> bool:
        """Whether this message arrived through a group chat."""
        return self.conversation_type == "group"

    @property
    def is_business(self) -> bool:
        """Whether this message came from a business account."""
        return self.conversation_type == "business"

    @property
    def has_media(self) -> bool:
        """Whether an image or voice note is attached."""
        return self.media_id is not None

    @property
    def has_text(self) -> bool:
        """Whether any text body is present (voice notes usually have none)."""
        return bool(self.message_text)

    @property
    def is_forwarded(self) -> bool:
        """Whether the message carries a non-zero forwarding count."""
        return self.forwarded_count > 0


@dataclass(frozen=True, slots=True)
class Message(MessageRecord):
    """One row of ``messages.csv`` - an incoming message awaiting routing."""


@dataclass(frozen=True, slots=True)
class MessageHistory(MessageRecord):
    """One row of ``message_history.csv`` - a message the user already received."""


@dataclass(frozen=True, slots=True)
class SampleMessage(MessageRecord):
    """One row of ``sample_messages.csv`` - a worked example with its label.

    Reference material for output formatting only. Hour 1 does not use these
    labels for anything, and later phases must not train or hardcode on them
    beyond understanding the expected shape.
    """

    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Split the semicolon-separated evidence list.

        Returns:
            Historical message IDs, or an empty tuple for the ``none`` sentinel.
        """
        raw = safe_text(self.evidence_message_ids, default="") or ""
        if raw.strip().casefold() == "none":
            return ()
        return tuple(part.strip() for part in raw.split(";") if part.strip())


# --------------------------------------------------------------------------- #
# Interactions and media
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MessageEvent(Record):
    """One row of ``message_events.csv`` - how a user reacted to a past message."""

    user_id: str
    message_id: str
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: int | None
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool

    @property
    def is_negative_signal(self) -> bool:
        """Whether the user dismissed, muted or reported this message."""
        return self.notification_dismissed or self.muted_after_message or self.message_reported


@dataclass(frozen=True, slots=True)
class Image(Record):
    """One row of ``images.csv`` - image media metadata."""

    image_id: str
    file_path: str


@dataclass(frozen=True, slots=True)
class VoiceNote(Record):
    """One row of ``voice_notes.csv`` - voice-note media metadata."""

    voice_note_id: str
    file_path: str


@dataclass(frozen=True, slots=True)
class NotificationSummary(Record):
    """One row of ``daily_notification_summary.csv`` - a user's daily load."""

    user_id: str
    date: dt.date
    notifications_sent: int
    notifications_dismissed: int


#: Logical table name -> record class. Keeps loader/index code table-agnostic.
MODEL_BY_TABLE: Mapping[str, type[Record]] = {
    "users": User,
    "groups": Group,
    "group_members": GroupMember,
    "business_accounts": BusinessAccount,
    "user_business_history": UserBusinessHistory,
    "messages": Message,
    "message_history": MessageHistory,
    "message_events": MessageEvent,
    "images": Image,
    "voice_notes": VoiceNote,
    "daily_notification_summary": NotificationSummary,
    "sample_messages": SampleMessage,
}

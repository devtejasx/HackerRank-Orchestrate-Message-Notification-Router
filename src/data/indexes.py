"""Precomputed lookup tables.

Every index is built once, up front, from the record tuples handed out by
:class:`~src.data.loader.DataLoader`. Downstream phases should never filter a
DataFrame in a loop; they should hit a dict here instead.

Two flavours of index:

* **unique** - ``Mapping[key, Record]``, backed by a primary key.
* **grouped** - ``Mapping[key, tuple[Record, ...]]``, for one-to-many
  relationships. Message collections are sorted oldest-first so downstream
  ordering is deterministic.

Mappings are exposed read-only so a consumer cannot corrupt shared state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TypeVar

from src import config
from src.data.loader import DataLoader
from src.data.models import (
    BusinessAccount,
    Group,
    GroupMember,
    Image,
    Message,
    MessageEvent,
    MessageHistory,
    MessageRecord,
    NotificationSummary,
    Record,
    SampleMessage,
    User,
    UserBusinessHistory,
    VoiceNote,
)
from src.utils.helpers import group_by, index_by

__all__ = ["DataIndex", "build_indexes"]

_LOGGER = config.get_logger("indexes")

_R = TypeVar("_R", bound=Record)
_M = TypeVar("_M", bound=MessageRecord)


def _unique(records: Iterable[_R], key: Callable[[_R], object]) -> Mapping[object, _R]:
    """Build a read-only one-to-one index."""
    return MappingProxyType(index_by(records, key))  # type: ignore[arg-type]


def _grouped(
    records: Iterable[_R], key: Callable[[_R], object]
) -> Mapping[object, tuple[_R, ...]]:
    """Build a read-only one-to-many index, preserving input order."""
    grouped = group_by(records, key)  # type: ignore[arg-type]
    return MappingProxyType({k: tuple(v) for k, v in grouped.items()})


def _grouped_by_time(
    records: Iterable[_M], key: Callable[[_M], str | None]
) -> Mapping[str, tuple[_M, ...]]:
    """Build a one-to-many index of message records sorted oldest-first."""
    grouped = group_by(records, key)
    return MappingProxyType(
        {
            k: tuple(sorted(v, key=lambda record: (record.created_at, record.message_id)))
            for k, v in grouped.items()
        }
    )


@dataclass(frozen=True, slots=True)
class DataIndex:
    """All lookup tables for one loaded dataset.

    Naming convention: ``<what>_by_<key>``. For message collections,
    ``by_user`` keys on the **recipient** (``user_id``) while ``by_sender``
    keys on the **sender** (``sender_user_id``); these are different people and
    must not be confused.
    """

    # --- unique, primary-key backed ---------------------------------- #
    users_by_id: Mapping[str, User]
    groups_by_id: Mapping[str, Group]
    business_by_id: Mapping[str, BusinessAccount]
    messages_by_id: Mapping[str, Message]
    history_by_id: Mapping[str, MessageHistory]
    images_by_id: Mapping[str, Image]
    voice_by_id: Mapping[str, VoiceNote]
    samples_by_id: Mapping[str, SampleMessage]

    # --- unique, composite-key backed -------------------------------- #
    group_member_by_key: Mapping[tuple[str, str], GroupMember]
    user_business_by_key: Mapping[tuple[str, str], UserBusinessHistory]
    #: ``message_events`` holds exactly one row per historical message.
    event_by_message: Mapping[str, MessageEvent]
    notification_summary_by_key: Mapping[tuple[str, object], NotificationSummary]

    # --- grouped: incoming messages ---------------------------------- #
    messages_by_user: Mapping[str, tuple[Message, ...]]
    messages_by_sender: Mapping[str, tuple[Message, ...]]
    messages_by_group: Mapping[str, tuple[Message, ...]]
    messages_by_business: Mapping[str, tuple[Message, ...]]

    # --- grouped: historical messages -------------------------------- #
    history_by_user: Mapping[str, tuple[MessageHistory, ...]]
    history_by_sender: Mapping[str, tuple[MessageHistory, ...]]
    history_by_group: Mapping[str, tuple[MessageHistory, ...]]
    history_by_business: Mapping[str, tuple[MessageHistory, ...]]

    # --- grouped: relationships and activity ------------------------- #
    events_by_user: Mapping[str, tuple[MessageEvent, ...]]
    group_members_by_group: Mapping[str, tuple[GroupMember, ...]]
    group_members_by_user: Mapping[str, tuple[GroupMember, ...]]
    user_business_by_user: Mapping[str, tuple[UserBusinessHistory, ...]]
    user_business_by_business: Mapping[str, tuple[UserBusinessHistory, ...]]
    notification_summary_by_user: Mapping[str, tuple[NotificationSummary, ...]]

    def sizes(self) -> dict[str, int]:
        """Return the entry count of every index, for logging and diagnostics."""
        return {spec.name: len(getattr(self, spec.name)) for spec in fields(self)}


def build_indexes(loader: DataLoader) -> DataIndex:
    """Build every lookup table from ``loader``.

    Records are materialised here, so run
    :func:`src.data.validation.validate_dataset` first: this is the point at
    which a structurally broken row would surface as a
    :class:`~src.data.models.RecordCoercionError`.

    Args:
        loader: A loader pointing at an already-validated dataset.

    Returns:
        A fully populated, immutable :class:`DataIndex`.
    """
    users: tuple[User, ...] = loader.records("users")  # type: ignore[assignment]
    groups: tuple[Group, ...] = loader.records("groups")  # type: ignore[assignment]
    members: tuple[GroupMember, ...] = loader.records("group_members")  # type: ignore[assignment]
    businesses: tuple[BusinessAccount, ...] = loader.records("business_accounts")  # type: ignore[assignment]
    relationships: tuple[UserBusinessHistory, ...] = loader.records("user_business_history")  # type: ignore[assignment]
    messages: tuple[Message, ...] = loader.records("messages")  # type: ignore[assignment]
    history: tuple[MessageHistory, ...] = loader.records("message_history")  # type: ignore[assignment]
    events: tuple[MessageEvent, ...] = loader.records("message_events")  # type: ignore[assignment]
    images: tuple[Image, ...] = loader.records("images")  # type: ignore[assignment]
    voice_notes: tuple[VoiceNote, ...] = loader.records("voice_notes")  # type: ignore[assignment]
    summaries: tuple[NotificationSummary, ...] = loader.records("daily_notification_summary")  # type: ignore[assignment]

    # sample_messages is reference-only and may legitimately be absent.
    samples: tuple[SampleMessage, ...] = (
        loader.records("sample_messages")  # type: ignore[assignment]
        if loader.is_available("sample_messages")
        else ()
    )

    index = DataIndex(
        users_by_id=_unique(users, lambda r: r.user_id),
        groups_by_id=_unique(groups, lambda r: r.group_id),
        business_by_id=_unique(businesses, lambda r: r.business_id),
        messages_by_id=_unique(messages, lambda r: r.message_id),
        history_by_id=_unique(history, lambda r: r.message_id),
        images_by_id=_unique(images, lambda r: r.image_id),
        voice_by_id=_unique(voice_notes, lambda r: r.voice_note_id),
        samples_by_id=_unique(samples, lambda r: r.message_id),
        group_member_by_key=_unique(members, lambda r: (r.group_id, r.user_id)),
        user_business_by_key=_unique(relationships, lambda r: (r.user_id, r.business_id)),
        event_by_message=_unique(events, lambda r: r.message_id),
        notification_summary_by_key=_unique(summaries, lambda r: (r.user_id, r.date)),
        messages_by_user=_grouped_by_time(messages, lambda r: r.user_id),
        messages_by_sender=_grouped_by_time(messages, lambda r: r.sender_user_id),
        messages_by_group=_grouped_by_time(messages, lambda r: r.group_id),
        messages_by_business=_grouped_by_time(messages, lambda r: r.business_id),
        history_by_user=_grouped_by_time(history, lambda r: r.user_id),
        history_by_sender=_grouped_by_time(history, lambda r: r.sender_user_id),
        history_by_group=_grouped_by_time(history, lambda r: r.group_id),
        history_by_business=_grouped_by_time(history, lambda r: r.business_id),
        events_by_user=_grouped(events, lambda r: r.user_id),
        group_members_by_group=_grouped(members, lambda r: r.group_id),
        group_members_by_user=_grouped(members, lambda r: r.user_id),
        user_business_by_user=_grouped(relationships, lambda r: r.user_id),
        user_business_by_business=_grouped(relationships, lambda r: r.business_id),
        notification_summary_by_user=_grouped(summaries, lambda r: r.user_id),
    )

    _LOGGER.info("Built %d indexes over %d records", len(index.sizes()), _total(loader))
    return index


def _total(loader: DataLoader) -> int:
    """Total number of rows currently loaded."""
    return sum(loader.summary().values())

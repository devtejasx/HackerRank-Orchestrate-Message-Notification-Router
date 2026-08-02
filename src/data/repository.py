"""The public face of the data layer.

Later phases should depend on :class:`DataRepository` and nothing else. It
wires together loading, validation and indexing, and exposes small read
helpers on top.

Return-value convention, applied consistently:

* single-entity getters (``get_user``, ``get_group``) return ``None`` when the
  id is unknown;
* collection getters (``get_group_members``, ``get_user_history``) return an
  empty tuple, never ``None``, so callers can iterate without a guard.

Example:
    >>> repo = DataRepository.load()            # doctest: +SKIP
    >>> repo.get_user("u_001").quiet_hours      # doctest: +SKIP
    (datetime.time(22, 0), datetime.time(7, 0))
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Self

from src import config
from src.data import schema
from src.data.indexes import DataIndex, build_indexes
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
    SampleMessage,
    User,
    UserBusinessHistory,
    VoiceNote,
)
from src.data.validation import ValidationReport, validate_dataset
from src.utils.helpers import resolve_dataset_path

__all__ = ["DataRepository"]

_LOGGER = config.get_logger("repository")


class DataRepository:
    """Read-only access to the whole dataset.

    Prefer :meth:`load` over constructing this directly; the constructor
    assumes the loader and index are already consistent with each other.

    Args:
        loader: Loader holding the parsed frames.
        index: Lookup tables built from the same loader.
        validation_report: Report from the validation pass, if one was run.
    """

    def __init__(
        self,
        loader: DataLoader,
        index: DataIndex,
        validation_report: ValidationReport | None = None,
    ) -> None:
        self._loader = loader
        self._index = index
        self._validation_report = validation_report

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def load(
        cls,
        dataset_dir: Path | None = None,
        *,
        validate: bool = True,
        strict: bool | None = None,
    ) -> Self:
        """Load, validate and index the dataset in one call.

        Args:
            dataset_dir: Directory holding the CSVs. Defaults to
                :data:`src.config.DATASET_DIR`.
            validate: Whether to run the validation pass before indexing.
                Leave enabled unless you are deliberately inspecting broken data.
            strict: Promote validation warnings to errors.

        Returns:
            A ready-to-use repository.

        Raises:
            MissingDatasetFileError: If a required CSV is absent.
            DatasetValidationError: If validation finds blocking errors.
        """
        loader = DataLoader(dataset_dir)
        loader.load_all()

        report = validate_dataset(loader, strict=strict) if validate else None
        index = build_indexes(loader)
        return cls(loader, index, report)

    # ------------------------------------------------------------------ #
    # Escape hatches
    # ------------------------------------------------------------------ #

    @property
    def loader(self) -> DataLoader:
        """Underlying loader, for code that genuinely needs DataFrames."""
        return self._loader

    @property
    def index(self) -> DataIndex:
        """Underlying lookup tables, for bulk iteration."""
        return self._index

    @property
    def validation_report(self) -> ValidationReport | None:
        """Report from the load-time validation pass, if one ran."""
        return self._validation_report

    # ------------------------------------------------------------------ #
    # Entities
    # ------------------------------------------------------------------ #

    def get_user(self, user_id: str) -> User | None:
        """Return the user with ``user_id``, or ``None``."""
        return self._index.users_by_id.get(user_id)

    def get_group(self, group_id: str) -> Group | None:
        """Return the group with ``group_id``, or ``None``."""
        return self._index.groups_by_id.get(group_id)

    def get_business(self, business_id: str) -> BusinessAccount | None:
        """Return the business account with ``business_id``, or ``None``."""
        return self._index.business_by_id.get(business_id)

    def get_message(self, message_id: str) -> Message | None:
        """Return the incoming message with ``message_id``, or ``None``."""
        return self._index.messages_by_id.get(message_id)

    def get_history_message(self, message_id: str) -> MessageHistory | None:
        """Return the historical message with ``message_id``, or ``None``.

        Incoming ids (``msg_*``) and historical ids (``message_*``) live in
        separate namespaces, so this never collides with :meth:`get_message`.
        """
        return self._index.history_by_id.get(message_id)

    def get_sample_message(self, message_id: str) -> SampleMessage | None:
        """Return the worked example with ``message_id``, or ``None``."""
        return self._index.samples_by_id.get(message_id)

    def get_image(self, media_id: str) -> Image | None:
        """Return image metadata for ``media_id``, or ``None``.

        ``media_id`` on a message corresponds to ``images.image_id``.
        """
        return self._index.images_by_id.get(media_id)

    def get_voice(self, media_id: str) -> VoiceNote | None:
        """Return voice-note metadata for ``media_id``, or ``None``.

        ``media_id`` on a message corresponds to ``voice_notes.voice_note_id``.
        """
        return self._index.voice_by_id.get(media_id)

    # ------------------------------------------------------------------ #
    # Groups and membership
    # ------------------------------------------------------------------ #

    def get_group_members(self, group_id: str) -> tuple[GroupMember, ...]:
        """Return every membership row for ``group_id``."""
        return self._index.group_members_by_group.get(group_id, ())

    def get_user_memberships(self, user_id: str) -> tuple[GroupMember, ...]:
        """Return every membership row belonging to ``user_id``."""
        return self._index.group_members_by_user.get(user_id, ())

    def get_group_member(self, group_id: str, user_id: str) -> GroupMember | None:
        """Return one user's membership of one group, or ``None``."""
        return self._index.group_member_by_key.get((group_id, user_id))

    def get_user_groups(self, user_id: str) -> tuple[Group, ...]:
        """Return the groups ``user_id`` belongs to.

        Memberships whose group is unknown are skipped rather than yielding
        ``None`` entries.
        """
        groups = (self.get_group(m.group_id) for m in self.get_user_memberships(user_id))
        return tuple(group for group in groups if group is not None)

    def get_group_admins(self, group_id: str) -> tuple[GroupMember, ...]:
        """Return the admin memberships of ``group_id``."""
        return tuple(m for m in self.get_group_members(group_id) if m.is_admin)

    # ------------------------------------------------------------------ #
    # Business relationships
    # ------------------------------------------------------------------ #

    def get_user_business(
        self, user_id: str, business_id: str
    ) -> UserBusinessHistory | None:
        """Return one user's relationship with one business, or ``None``."""
        return self._index.user_business_by_key.get((user_id, business_id))

    def get_user_businesses(self, user_id: str) -> tuple[UserBusinessHistory, ...]:
        """Return every business relationship held by ``user_id``."""
        return self._index.user_business_by_user.get(user_id, ())

    def get_business_users(self, business_id: str) -> tuple[UserBusinessHistory, ...]:
        """Return every user relationship held by ``business_id``."""
        return self._index.user_business_by_business.get(business_id, ())

    # ------------------------------------------------------------------ #
    # Incoming messages
    # ------------------------------------------------------------------ #

    def get_messages(self) -> tuple[Message, ...]:
        """Return every incoming message awaiting a routing decision."""
        return tuple(self._index.messages_by_id.values())

    def get_user_messages(self, user_id: str) -> tuple[Message, ...]:
        """Return incoming messages **received by** ``user_id``, oldest first."""
        return self._index.messages_by_user.get(user_id, ())

    def get_group_messages(self, group_id: str) -> tuple[Message, ...]:
        """Return incoming messages that arrived through ``group_id``."""
        return self._index.messages_by_group.get(group_id, ())

    def get_business_messages(self, business_id: str) -> tuple[Message, ...]:
        """Return incoming messages sent by ``business_id``."""
        return self._index.messages_by_business.get(business_id, ())

    # ------------------------------------------------------------------ #
    # Historical messages
    # ------------------------------------------------------------------ #

    def get_user_history(
        self, user_id: str, *, limit: int | None = None, newest_first: bool = False
    ) -> tuple[MessageHistory, ...]:
        """Return past messages **received by** ``user_id``.

        Args:
            user_id: The recipient.
            limit: Cap on the number of rows returned.
            newest_first: Reverse the default oldest-first ordering. Combine
                with ``limit`` to take the most recent N.
        """
        return _slice(self._index.history_by_user.get(user_id, ()), limit, newest_first)

    def get_sender_history(
        self, sender_id: str, *, limit: int | None = None, newest_first: bool = False
    ) -> tuple[MessageHistory, ...]:
        """Return past messages **sent by** ``sender_id``."""
        return _slice(self._index.history_by_sender.get(sender_id, ()), limit, newest_first)

    def get_group_history(
        self, group_id: str, *, limit: int | None = None, newest_first: bool = False
    ) -> tuple[MessageHistory, ...]:
        """Return past messages that arrived through ``group_id``."""
        return _slice(self._index.history_by_group.get(group_id, ()), limit, newest_first)

    def get_business_history(
        self, business_id: str, *, limit: int | None = None, newest_first: bool = False
    ) -> tuple[MessageHistory, ...]:
        """Return past messages sent by ``business_id``."""
        return _slice(
            self._index.history_by_business.get(business_id, ()), limit, newest_first
        )

    # ------------------------------------------------------------------ #
    # Interaction events
    # ------------------------------------------------------------------ #

    def get_message_event(self, message_id: str) -> MessageEvent | None:
        """Return the interaction event for a historical message, or ``None``.

        ``message_events`` is keyed one-to-one on ``message_id``.
        """
        return self._index.event_by_message.get(message_id)

    def get_message_events(self, message_id: str) -> tuple[MessageEvent, ...]:
        """Return the interaction events for ``message_id`` as a tuple.

        Collection-shaped form of :meth:`get_message_event`, for callers that
        would rather iterate than branch. Yields at most one element.
        """
        event = self.get_message_event(message_id)
        return (event,) if event is not None else ()

    def get_user_events(self, user_id: str) -> tuple[MessageEvent, ...]:
        """Return every interaction event recorded for ``user_id``."""
        return self._index.events_by_user.get(user_id, ())

    # ------------------------------------------------------------------ #
    # Notification load
    # ------------------------------------------------------------------ #

    def get_notification_summary(self, user_id: str) -> tuple[NotificationSummary, ...]:
        """Return ``user_id``'s daily notification rows."""
        return self._index.notification_summary_by_user.get(user_id, ())

    def get_notification_summary_for_date(
        self, user_id: str, day: date
    ) -> NotificationSummary | None:
        """Return one user's notification load on ``day``, or ``None``."""
        return self._index.notification_summary_by_key.get((user_id, day))

    # ------------------------------------------------------------------ #
    # Media
    # ------------------------------------------------------------------ #

    def get_media(self, media_type: str | None, media_id: str | None) -> Image | VoiceNote | None:
        """Return the media registry row for ``media_id``, or ``None``.

        Args:
            media_type: ``image`` or ``voice``. Anything else - including a
                modality a future dataset introduces - yields ``None`` rather
                than raising.
            media_id: The message's ``media_id`` cell.
        """
        if media_id is None or media_type is None:
            return None
        lookup = {"image": self.get_image, "voice": self.get_voice}.get(media_type)
        return lookup(media_id) if lookup is not None else None

    def get_media_path(self, message: MessageRecord) -> Path | None:
        """Return the absolute path of a message's attachment, or ``None``.

        Resolves ``media_type``/``media_id`` through the right media table.
        Returns ``None`` when the message has no media or the id is unknown.
        The path is not checked for existence; see
        :class:`~src.media.resolver.MediaResolver` for that.
        """
        media = self.get_media(message.media_type, message.media_id)
        if media is None:
            return None
        return resolve_dataset_path(media.file_path, self._loader.dataset_dir)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def record_counts(self) -> dict[str, int]:
        """Return row counts per table, keyed by display label."""
        return {
            schema.get_spec(name).display_label: count
            for name, count in self._loader.summary().items()
        }

    def describe(self) -> str:
        """Return a multi-line human-readable summary of what is loaded."""
        lines = [f"Dataset: {self._loader.dataset_dir}", "", "Records:"]
        lines += [f"  {label:<24} {count:>6}" for label, count in self.record_counts().items()]

        report = self._validation_report
        lines += ["", "Validation:"]
        if report is None:
            lines.append("  not run")
        else:
            lines.append(
                f"  {len(report.errors)} error(s), {len(report.warnings)} warning(s)"
            )

        lines += ["", "Indexes:"]
        lines += [f"  {name:<32} {size:>6}" for name, size in self._index.sizes().items()]
        return "\n".join(lines)


def _slice(
    records: Sequence[MessageHistory], limit: int | None, newest_first: bool
) -> tuple[MessageHistory, ...]:
    """Apply ordering then truncation to an already-sorted record sequence.

    Args:
        records: Oldest-first records straight from an index.
        limit: Maximum number of records to return.
        newest_first: Reverse before truncating, so ``limit`` takes the newest.
    """
    ordered = tuple(reversed(records)) if newest_first else tuple(records)
    return ordered[:limit] if limit is not None else ordered

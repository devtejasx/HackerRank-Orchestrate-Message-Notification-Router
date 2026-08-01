"""Tests for :mod:`src.data.indexes` and :mod:`src.data.repository`."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.data.loader import DataLoader
from src.data.repository import DataRepository

UNKNOWN = "does_not_exist"


class TestIndexIntegrity:
    """Indexes must partition the records without loss or duplication."""

    def test_unique_indexes_cover_every_record(self, repo: DataRepository) -> None:
        index = repo.index
        loader = repo.loader
        pairs = [
            (index.users_by_id, "users"),
            (index.groups_by_id, "groups"),
            (index.business_by_id, "business_accounts"),
            (index.messages_by_id, "messages"),
            (index.history_by_id, "message_history"),
            (index.images_by_id, "images"),
            (index.voice_by_id, "voice_notes"),
            (index.event_by_message, "message_events"),
            (index.group_member_by_key, "group_members"),
            (index.user_business_by_key, "user_business_history"),
            (index.notification_summary_by_key, "daily_notification_summary"),
        ]
        for mapping, table in pairs:
            assert len(mapping) == len(loader.raw_frame(table)), table

    @pytest.mark.parametrize(
        ("attribute", "table"),
        [
            ("messages_by_user", "messages"),
            ("history_by_user", "message_history"),
            ("group_members_by_group", "group_members"),
            ("group_members_by_user", "group_members"),
            ("events_by_user", "message_events"),
            ("user_business_by_user", "user_business_history"),
            ("notification_summary_by_user", "daily_notification_summary"),
        ],
    )
    def test_grouped_indexes_are_lossless(
        self, repo: DataRepository, attribute: str, table: str
    ) -> None:
        """Non-nullable grouping keys must account for every row exactly once."""
        grouped = getattr(repo.index, attribute)
        total = sum(len(bucket) for bucket in grouped.values())
        assert total == len(repo.loader.raw_frame(table))

    def test_nullable_grouping_keys_drop_only_null_rows(self, repo: DataRepository) -> None:
        raw = repo.loader.raw_frame("messages")
        expected = int((raw["group_id"] != "").sum())
        total = sum(len(bucket) for bucket in repo.index.messages_by_group.values())
        assert total == expected

    def test_indexes_are_read_only(self, repo: DataRepository) -> None:
        with pytest.raises(TypeError):
            repo.index.users_by_id["x"] = None  # type: ignore[index]

    def test_sender_and_recipient_are_not_conflated(self, repo: DataRepository) -> None:
        """user_id is the recipient; sender_user_id is the sender."""
        assert set(repo.index.history_by_user) != set(repo.index.history_by_sender)
        for sender_id, records in repo.index.history_by_sender.items():
            assert all(record.sender_user_id == sender_id for record in records)
        for user_id, records in repo.index.history_by_user.items():
            assert all(record.user_id == user_id for record in records)

    def test_message_collections_are_sorted_oldest_first(self, repo: DataRepository) -> None:
        for records in repo.index.history_by_user.values():
            timestamps = [record.created_at for record in records]
            assert timestamps == sorted(timestamps)


class TestEntityLookups:
    """Single-entity getters resolve known ids and return None otherwise."""

    def test_get_user(self, repo: DataRepository) -> None:
        user_id = next(iter(repo.index.users_by_id))
        assert repo.get_user(user_id).user_id == user_id

    def test_get_group(self, repo: DataRepository) -> None:
        group_id = next(iter(repo.index.groups_by_id))
        assert repo.get_group(group_id).group_id == group_id

    def test_get_business(self, repo: DataRepository) -> None:
        business_id = next(iter(repo.index.business_by_id))
        assert repo.get_business(business_id).business_id == business_id

    def test_get_message_and_history_namespaces_are_separate(
        self, repo: DataRepository
    ) -> None:
        message_id = next(iter(repo.index.messages_by_id))
        history_id = next(iter(repo.index.history_by_id))
        assert repo.get_message(message_id) is not None
        assert repo.get_history_message(message_id) is None
        assert repo.get_history_message(history_id) is not None
        assert repo.get_message(history_id) is None

    def test_get_image_and_voice(self, repo: DataRepository) -> None:
        image_id = next(iter(repo.index.images_by_id))
        voice_id = next(iter(repo.index.voice_by_id))
        assert repo.get_image(image_id).image_id == image_id
        assert repo.get_voice(voice_id).voice_note_id == voice_id

    @pytest.mark.parametrize(
        "method",
        [
            "get_user",
            "get_group",
            "get_business",
            "get_message",
            "get_history_message",
            "get_sample_message",
            "get_image",
            "get_voice",
            "get_message_event",
        ],
    )
    def test_unknown_id_returns_none(self, repo: DataRepository, method: str) -> None:
        assert getattr(repo, method)(UNKNOWN) is None

    @pytest.mark.parametrize(
        "method",
        [
            "get_group_members",
            "get_user_memberships",
            "get_user_groups",
            "get_group_admins",
            "get_user_businesses",
            "get_business_users",
            "get_user_messages",
            "get_group_messages",
            "get_business_messages",
            "get_user_history",
            "get_sender_history",
            "get_group_history",
            "get_business_history",
            "get_message_events",
            "get_user_events",
            "get_notification_summary",
        ],
    )
    def test_unknown_id_returns_empty_tuple(self, repo: DataRepository, method: str) -> None:
        assert getattr(repo, method)(UNKNOWN) == ()


class TestRelationshipHelpers:
    """Helpers agree with the underlying tables."""

    def test_group_members_and_user_groups_are_consistent(self, repo: DataRepository) -> None:
        group_id = max(
            repo.index.group_members_by_group,
            key=lambda gid: len(repo.index.group_members_by_group[gid]),
        )
        for member in repo.get_group_members(group_id):
            assert group_id in {g.group_id for g in repo.get_user_groups(member.user_id)}

    def test_get_group_member_matches_collection(self, repo: DataRepository) -> None:
        group_id = next(iter(repo.index.group_members_by_group))
        member = repo.get_group_members(group_id)[0]
        assert repo.get_group_member(group_id, member.user_id) == member

    def test_get_group_member_unknown_pair(self, repo: DataRepository) -> None:
        assert repo.get_group_member(UNKNOWN, UNKNOWN) is None

    def test_group_admins_are_a_subset_of_members(self, repo: DataRepository) -> None:
        group_id = next(iter(repo.index.group_members_by_group))
        members = set(repo.get_group_members(group_id))
        assert set(repo.get_group_admins(group_id)) <= members

    def test_user_groups_resolve_to_real_groups(self, repo: DataRepository) -> None:
        for user_id in list(repo.index.group_members_by_user)[:10]:
            memberships = repo.get_user_memberships(user_id)
            groups = repo.get_user_groups(user_id)
            assert len(groups) == len(memberships)
            assert all(group is not None for group in groups)

    def test_user_business_lookup(self, repo: DataRepository) -> None:
        user_id, business_id = next(iter(repo.index.user_business_by_key))
        relationship = repo.get_user_business(user_id, business_id)
        assert relationship is not None
        assert relationship in repo.get_user_businesses(user_id)
        assert relationship in repo.get_business_users(business_id)


class TestHistoryQueries:
    """Ordering and limiting on history helpers."""

    def test_default_is_oldest_first(self, repo: DataRepository, busy_user: str) -> None:
        history = repo.get_user_history(busy_user)
        assert [r.created_at for r in history] == sorted(r.created_at for r in history)

    def test_newest_first_reverses(self, repo: DataRepository, busy_user: str) -> None:
        oldest = repo.get_user_history(busy_user)
        newest = repo.get_user_history(busy_user, newest_first=True)
        assert newest == tuple(reversed(oldest))

    def test_limit_truncates(self, repo: DataRepository, busy_user: str) -> None:
        assert len(repo.get_user_history(busy_user, limit=3)) == 3

    def test_limit_with_newest_first_takes_the_newest(
        self, repo: DataRepository, busy_user: str
    ) -> None:
        full = repo.get_user_history(busy_user)
        assert repo.get_user_history(busy_user, limit=2, newest_first=True) == (
            full[-1],
            full[-2],
        )

    def test_limit_larger_than_available(self, repo: DataRepository, busy_user: str) -> None:
        full = repo.get_user_history(busy_user)
        assert repo.get_user_history(busy_user, limit=10_000) == full


class TestEvents:
    """message_events is one-to-one with message_history."""

    def test_every_history_message_has_exactly_one_event(self, repo: DataRepository) -> None:
        for message_id in repo.index.history_by_id:
            assert len(repo.get_message_events(message_id)) == 1

    def test_singular_and_plural_agree(self, repo: DataRepository) -> None:
        message_id = next(iter(repo.index.history_by_id))
        assert repo.get_message_events(message_id) == (repo.get_message_event(message_id),)

    def test_user_events_belong_to_that_user(self, repo: DataRepository) -> None:
        user_id = next(iter(repo.index.events_by_user))
        assert all(e.user_id == user_id for e in repo.get_user_events(user_id))


class TestMedia:
    """Media resolution reaches real bytes on disk."""

    @pytest.mark.parametrize("media_type", ["image", "voice"])
    def test_resolves_to_an_existing_file(
        self, repo: DataRepository, media_type: str
    ) -> None:
        message = next(m for m in repo.get_messages() if m.media_type == media_type)
        path = repo.get_media_path(message)
        assert path is not None and path.is_file() and path.stat().st_size > 0

    def test_returns_none_without_media(self, repo: DataRepository) -> None:
        message = next(m for m in repo.get_messages() if not m.has_media)
        assert repo.get_media_path(message) is None

    def test_every_referenced_media_resolves(self, repo: DataRepository) -> None:
        with_media = [m for m in repo.get_messages() if m.has_media]
        assert with_media
        assert all(repo.get_media_path(m) is not None for m in with_media)


class TestNotificationSummary:
    """Daily notification load, by user and by day."""

    def test_lookup_by_user_and_date(self, repo: DataRepository) -> None:
        user_id = next(iter(repo.index.notification_summary_by_user))
        rows = repo.get_notification_summary(user_id)
        assert rows
        row = rows[0]
        assert isinstance(row.date, date)
        assert repo.get_notification_summary_for_date(user_id, row.date) == row

    def test_unknown_date_returns_none(self, repo: DataRepository) -> None:
        user_id = next(iter(repo.index.notification_summary_by_user))
        assert repo.get_notification_summary_for_date(user_id, date(1999, 1, 1)) is None


class TestRepositoryConstruction:
    """Loading options and diagnostics."""

    def test_load_without_validation(self, dataset_dir: Path) -> None:
        repository = DataRepository.load(dataset_dir, validate=False)
        assert repository.validation_report is None
        assert repository.get_messages()

    def test_record_counts_use_display_labels(self, repo: DataRepository) -> None:
        counts = repo.record_counts()
        assert counts["Users"] == 54
        assert counts["History"] == 412

    def test_describe_mentions_key_sections(self, repo: DataRepository) -> None:
        described = repo.describe()
        assert "Records:" in described
        assert "Validation:" in described
        assert "Indexes:" in described

    def test_loader_escape_hatch_exposes_dataframes(self, repo: DataRepository) -> None:
        assert isinstance(repo.loader, DataLoader)
        assert repo.loader.messages.shape == (110, 11)

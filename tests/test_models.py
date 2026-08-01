"""Tests for :mod:`src.data.models`."""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, time

import pytest

from src.data.loader import DataLoader
from src.data.models import (
    MODEL_BY_TABLE,
    BusinessAccount,
    Message,
    MessageEvent,
    RecordCoercionError,
    SampleMessage,
    User,
)


def _message_row(**overrides: str) -> dict[str, str]:
    """Return a valid raw message row, with optional field overrides."""
    row = {
        "message_id": "msg_001",
        "user_id": "u_001",
        "conversation_type": "group",
        "group_id": "group_001",
        "business_id": "",
        "sender_user_id": "u_049",
        "created_at": "2026-07-30 22:19",
        "message_text": "hello",
        "media_type": "",
        "media_id": "",
        "forwarded_count": "3",
    }
    return row | overrides


class TestConstruction:
    """Every model builds from every real row in its CSV."""

    @pytest.mark.parametrize("table", sorted(MODEL_BY_TABLE))
    def test_builds_from_real_rows(self, loader: DataLoader, table: str) -> None:
        records = loader.records(table)
        assert len(records) == len(loader.raw_frame(table))
        assert isinstance(records[0], MODEL_BY_TABLE[table])

    def test_coerces_types(self) -> None:
        message = Message.from_row(_message_row())
        assert message.created_at == datetime(2026, 7, 30, 22, 19)
        assert message.forwarded_count == 3
        assert message.business_id is None

    def test_missing_key_is_treated_as_blank(self) -> None:
        row = _message_row()
        del row["media_type"]
        assert Message.from_row(row).media_type is None

    def test_records_are_immutable(self) -> None:
        message = Message.from_row(_message_row())
        with pytest.raises(dataclasses.FrozenInstanceError):
            message.message_id = "other"  # type: ignore[misc]

    def test_records_are_hashable(self) -> None:
        assert len({Message.from_row(_message_row())}) == 1

    def test_to_dict_round_trips_field_names(self) -> None:
        assert set(Message.from_row(_message_row()).to_dict()) == {
            f.name for f in dataclasses.fields(Message)
        }


class TestRequiredFields:
    """Non-nullable fields fail loudly rather than holding a silent None."""

    @pytest.mark.parametrize("column", ["message_id", "user_id", "created_at"])
    def test_blank_required_field_raises(self, column: str) -> None:
        with pytest.raises(RecordCoercionError, match=column):
            Message.from_row(_message_row(**{column: ""}))

    def test_unparseable_required_field_raises(self) -> None:
        with pytest.raises(RecordCoercionError, match="created_at"):
            Message.from_row(_message_row(created_at="not-a-date"))

    def test_nullable_field_accepts_blank(self) -> None:
        assert Message.from_row(_message_row(message_text="")).message_text is None


class TestDerivedProperties:
    """Computed properties describe the row; they make no routing decisions."""

    def test_conversation_flags_are_exclusive(self) -> None:
        message = Message.from_row(_message_row(conversation_type="group"))
        assert (message.is_group, message.is_personal, message.is_business) == (
            True,
            False,
            False,
        )

    def test_media_and_text_flags(self) -> None:
        voice = Message.from_row(
            _message_row(media_type="voice", media_id="vn_001", message_text="")
        )
        assert voice.has_media is True
        assert voice.has_text is False

    def test_forwarded_flag(self) -> None:
        assert Message.from_row(_message_row(forwarded_count="0")).is_forwarded is False
        assert Message.from_row(_message_row(forwarded_count="2")).is_forwarded is True

    def test_user_quiet_hours(self) -> None:
        user = User.from_row(
            {
                "user_id": "u_001",
                "do_not_disturb_window": "22:00-07:00",
                "messages_opened_30d": "45",
                "messages_replied_30d": "8",
                "notifications_dismissed_30d": "14",
                "messages_reported_30d": "2",
            }
        )
        assert user.quiet_hours == (time(22, 0), time(7, 0))

    def test_business_domain_comparison_is_tri_state(self, loader: DataLoader) -> None:
        accounts: tuple[BusinessAccount, ...] = loader.records("business_accounts")  # type: ignore[assignment]
        outcomes = {account.sender_domain_matches_official for account in accounts}
        assert outcomes == {True, False, None}, "expected match, mismatch and unknown"

    def test_business_domain_unknown_when_domain_absent(self) -> None:
        account = BusinessAccount.from_row(
            {
                "business_id": "b1",
                "display_name": "X",
                "brand_name": "X",
                "category": "bank",
                "verified": "1",
                "official_domain": "",
                "domain_used_by_sender": "x.com",
                "account_age_days": "10",
                "messages_sent_30d": "1",
                "user_reports_30d": "0",
                "domain_used_by_sender_age_days": "5",
            }
        )
        assert account.sender_domain_matches_official is None

    def test_event_negative_signal(self) -> None:
        base = {
            "user_id": "u_001",
            "message_id": "message_0001",
            "message_opened": "1",
            "message_replied": "0",
            "reaction_time_minutes": "",
            "notification_dismissed": "0",
            "muted_after_message": "0",
            "message_reported": "0",
        }
        assert MessageEvent.from_row(base).is_negative_signal is False
        assert MessageEvent.from_row(base | {"message_reported": "1"}).is_negative_signal is True

    def test_event_reaction_time_is_optional(self) -> None:
        event = MessageEvent.from_row(
            {
                "user_id": "u_001",
                "message_id": "message_0001",
                "message_opened": "0",
                "message_replied": "0",
                "reaction_time_minutes": "",
                "notification_dismissed": "1",
                "muted_after_message": "0",
                "message_reported": "0",
            }
        )
        assert event.reaction_time_minutes is None


class TestSampleMessage:
    """The `none` sentinel in evidence_message_ids must survive parsing."""

    def _sample(self, evidence: str) -> SampleMessage:
        return SampleMessage.from_row(
            _message_row(message_id="sample_msg_001")
            | {
                "action": "notify",
                "message_type": "urgent",
                "reason": "because",
                "confidence": "0.89",
                "evidence_message_ids": evidence,
            }
        )

    def test_none_sentinel_yields_empty_tuple(self) -> None:
        assert self._sample("none").evidence_ids == ()
        assert self._sample("none").evidence_message_ids == "none"

    def test_semicolon_list_is_split(self) -> None:
        assert self._sample("message_0013;message_0014").evidence_ids == (
            "message_0013",
            "message_0014",
        )

    def test_real_samples_reference_real_history(self, loader: DataLoader) -> None:
        samples: tuple[SampleMessage, ...] = loader.records("sample_messages")  # type: ignore[assignment]
        known = set(loader.raw_frame("message_history")["message_id"])
        referenced = {mid for sample in samples for mid in sample.evidence_ids}
        assert referenced and referenced <= known

    def test_confidence_is_float(self) -> None:
        assert self._sample("none").confidence == pytest.approx(0.89)


class TestDateHandling:
    """date and datetime columns must not be conflated."""

    def test_date_column_yields_date(self, loader: DataLoader) -> None:
        groups = loader.records("groups")
        assert isinstance(groups[0].created_at, date)
        assert not isinstance(groups[0].created_at, datetime)

    def test_timestamp_column_yields_datetime(self, loader: DataLoader) -> None:
        messages = loader.records("messages")
        assert isinstance(messages[0].created_at, datetime)

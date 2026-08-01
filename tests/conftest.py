"""Shared fixtures.

The real dataset is loaded once per session; tests that need to break
something work on a throwaway copy under ``tmp_path`` instead.
"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path

import pytest

from src import config
from src.data.loader import DataLoader
from src.data.models import Message
from src.data.repository import DataRepository
from src.pipeline import MessagePipeline

#: Row of a CSV as read by :class:`csv.DictReader`.
Row = dict[str, str]


@pytest.fixture(scope="session")
def dataset_dir() -> Path:
    """Directory holding the real, shipped dataset."""
    if not config.DATASET_DIR.is_dir():
        pytest.skip(f"dataset directory not found: {config.DATASET_DIR}")
    return config.DATASET_DIR


@pytest.fixture(scope="session")
def loader(dataset_dir: Path) -> DataLoader:
    """A loader over the real dataset, warmed once."""
    instance = DataLoader(dataset_dir)
    instance.load_all()
    return instance


@pytest.fixture(scope="session")
def repo(dataset_dir: Path) -> DataRepository:
    """A fully loaded, validated and indexed repository over the real dataset."""
    return DataRepository.load(dataset_dir)


@pytest.fixture(scope="session")
def busy_user(repo: DataRepository) -> str:
    """Id of the user with the most historical messages."""
    return max(
        repo.index.history_by_user,
        key=lambda user_id: len(repo.index.history_by_user[user_id]),
    )


@pytest.fixture(scope="session")
def pipeline(repo: DataRepository) -> MessagePipeline:
    """A Phase 2 pipeline over the real dataset."""
    return MessagePipeline(repo)


@pytest.fixture(scope="session")
def known_ids(repo: DataRepository) -> dict[str, str]:
    """Real ids satisfying the conditions the classifier tests need.

    Discovered from the data rather than hardcoded, so the suite keeps working
    if the dataset is swapped.
    """
    group_id, members = max(
        repo.index.group_members_by_group.items(), key=lambda item: len(item[1])
    )
    admin = next((m for m in members if m.is_admin), members[0])
    member = next((m for m in members if not m.is_admin), members[0])

    relationship = next(
        rel
        for rel in repo.loader.records("user_business_history")
        if (business := repo.get_business(rel.business_id)) is not None
        and business.verified
        and business.sender_domain_matches_official is True
    )

    # A business the same user has never dealt with, for the unsolicited case.
    known_business_ids = {
        rel.business_id for rel in repo.get_user_businesses(relationship.user_id)
    }
    unrelated_business_id = next(
        business_id
        for business_id in repo.index.business_by_id
        if business_id not in known_business_ids
    )

    # A one-to-one pair with real prior contact, so "stranger" rules do not
    # fire on an ordinary personal message.
    personal_history = next(
        record
        for record in repo.loader.records("message_history")
        if record.conversation_type == "personal" and record.sender_user_id
    )

    # A pair with genuinely no prior contact in either direction, for the
    # unfamiliar-sender case. Senders are drawn from the ids that actually
    # appear as senders, so the sender still resolves to a real user.
    recipient = personal_history.user_id
    contacted_by = {
        record.sender_user_id
        for record in repo.get_user_history(recipient)
        if record.sender_user_id
    }
    stranger_id = next(
        sender_id
        for sender_id in repo.index.history_by_sender
        if sender_id not in contacted_by and sender_id != recipient
    )

    return {
        "group_id": group_id,
        "admin_user_id": admin.user_id,
        "member_user_id": member.user_id,
        "user_id": relationship.user_id,
        "business_id": relationship.business_id,
        "unrelated_business_id": unrelated_business_id,
        "known_pair_user_id": personal_history.user_id,
        "known_pair_sender_id": personal_history.sender_user_id,
        "stranger_sender_id": stranger_id,
    }


@pytest.fixture(scope="session")
def make_message(known_ids: dict[str, str]) -> Callable[..., Message]:
    """Return a factory building valid :class:`Message` objects for tests.

    Defaults describe a plain group message; pass overrides for the case under
    test. Reference columns are filled to match ``conversation_type`` so the
    result obeys the dataset invariant.
    """

    def _make(
        text: str | None = "hello there",
        conversation_type: str = "group",
        *,
        message_id: str = "msg_test",
        user_id: str | None = None,
        sender_user_id: str | None = None,
        forwarded_count: int = 0,
        media_type: str | None = None,
        media_id: str | None = None,
        created_at: datetime | None = None,
        known_sender: bool = True,
        known_business: bool = True,
    ) -> Message:
        """Build a message.

        ``known_sender`` picks a one-to-one pair with prior contact;
        ``known_business`` picks a verified business the recipient deals with.
        Set either to ``False`` to exercise the unfamiliar-sender rules.
        """
        group_id = business_id = None
        if conversation_type == "group":
            group_id = known_ids["group_id"]
            user_id = user_id or known_ids["member_user_id"]
            sender_user_id = sender_user_id or known_ids["admin_user_id"]
        elif conversation_type == "business":
            business_id = (
                known_ids["business_id"]
                if known_business
                else known_ids["unrelated_business_id"]
            )
            user_id = user_id or known_ids["user_id"]
            sender_user_id = None
        else:
            user_id = user_id or known_ids["known_pair_user_id"]
            sender_user_id = sender_user_id or (
                known_ids["known_pair_sender_id"]
                if known_sender
                else known_ids["stranger_sender_id"]
            )

        return Message(
            message_id=message_id,
            user_id=user_id,
            conversation_type=conversation_type,
            group_id=group_id,
            business_id=business_id,
            sender_user_id=sender_user_id,
            created_at=created_at or datetime(2026, 7, 25, 14, 30),
            message_text=text,
            media_type=media_type,
            media_id=media_id,
            forwarded_count=forwarded_count,
        )

    return _make


@pytest.fixture
def dataset_copy(dataset_dir: Path, tmp_path: Path) -> Path:
    """Return a writable copy of the dataset that tests may corrupt."""
    destination = tmp_path / "dataset"
    shutil.copytree(dataset_dir, destination)
    return destination


@pytest.fixture
def read_csv() -> Callable[[Path], tuple[list[Row], list[str]]]:
    """Return a function reading a CSV into ``(rows, fieldnames)``."""

    def _read(path: Path) -> tuple[list[Row], list[str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            return list(reader), fieldnames

    return _read


@pytest.fixture
def write_csv() -> Callable[[Path, Sequence[str], Iterable[Row]], None]:
    """Return a function writing rows back to a CSV."""

    def _write(path: Path, fieldnames: Sequence[str], rows: Iterable[Row]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)

    return _write


@pytest.fixture
def mutate_csv(
    read_csv: Callable[[Path], tuple[list[Row], list[str]]],
    write_csv: Callable[[Path, Sequence[str], Iterable[Row]], None],
) -> Callable[[Path, Callable[[list[Row], list[str]], tuple[list[Row], list[str]]]], None]:
    """Return a function that rewrites a CSV through a transform.

    The transform receives ``(rows, fieldnames)`` and returns the new pair,
    which keeps corruption tests to a couple of readable lines.
    """

    def _mutate(
        path: Path,
        transform: Callable[[list[Row], list[str]], tuple[list[Row], list[str]]],
    ) -> None:
        rows, fieldnames = read_csv(path)
        new_rows, new_fieldnames = transform(rows, fieldnames)
        write_csv(path, new_fieldnames, new_rows)

    return _mutate

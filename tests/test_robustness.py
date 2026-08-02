"""What happens when the dataset is wrong.

The submission contract is unconditional: one prediction per row of
``messages.csv``, every time. A hidden evaluation set will not be as clean as
the shipped one, and the failure mode that costs the most is not a wrong
action - it is a traceback, because a traceback costs every row at once.

So each test here breaks the dataset in one specific way and asserts the run
still finishes with a complete, valid submission. The corruptions are the ones
real data actually produces: blank required cells, unparseable timestamps,
references to entities that were never exported, media that is registered but
absent, and values outside the declared vocabulary.

Two invariants recur and are worth naming, because together they *are* the
contract:

* **completeness** - ``len(results) == len(messages)``, always;
* **validity** - :func:`~src.output.validate_results` reports no errors.
"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from src.data.loader import DataLoader
from src.data.models import RecordCoercionError, User, coerce_row
from src.data.repository import DataRepository
from src.output import validate_results, write_output_csv
from src.routing.models import NO_EVIDENCE, RoutingResult
from src.routing.pipeline import RoutingPipeline

#: One row of ``messages.csv``.
Row = dict[str, str]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _append_messages(dataset: Path, extra: Sequence[dict[str, str]]) -> None:
    """Append rows to ``messages.csv``, filling unspecified columns as blank."""
    path = dataset / "messages.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or ())
        rows = list(reader)

    blank = dict.fromkeys(columns, "")
    rows.extend({**blank, **row} for row in extra)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _route(dataset: Path) -> tuple[RoutingResult, ...]:
    """Load a possibly-broken dataset and route all of it.

    Validation stays on: the point is that a dataset with *warnings* still
    produces a submission, not that the checks were skipped.
    """
    repo = DataRepository.load(dataset)
    results = RoutingPipeline(repo).route_all()
    report = validate_results(results, repo.get_messages(), repo)
    assert report.errors == (), f"invalid submission: {[str(e) for e in report.errors]}"
    assert len(results) == len(repo.get_messages())
    return results


def _result(results: Sequence[RoutingResult], message_id: str) -> RoutingResult:
    """Return the single prediction for ``message_id``."""
    matches = [r for r in results if r.message_id == message_id]
    assert len(matches) == 1, f"expected exactly one prediction for {message_id}"
    return matches[0]


@pytest.fixture
def broken(dataset_copy: Path) -> Callable[[Sequence[dict[str, str]]], Path]:
    """Return a factory that appends malformed messages to a throwaway copy."""

    def _add(rows: Sequence[dict[str, str]]) -> Path:
        _append_messages(dataset_copy, rows)
        return dataset_copy

    return _add


#: The minimum a well-formed row carries. Tests override one field at a time so
#: each failure is attributable to exactly one defect.
_BASE_ROW: dict[str, str] = {
    "message_id": "edge_000",
    "user_id": "u_001",
    "conversation_type": "personal",
    "sender_user_id": "u_002",
    "created_at": "2026-08-01 10:00",
    "message_text": "are you free this evening",
    "forwarded_count": "0",
}


# --------------------------------------------------------------------------- #
# Broken references
# --------------------------------------------------------------------------- #


class TestUnknownEntities:
    """Ids that point at nothing. The commonest defect in an exported subset."""

    @pytest.mark.parametrize(
        ("case", "overrides"),
        [
            ("unknown_sender", {"sender_user_id": "u_does_not_exist"}),
            ("unknown_recipient", {"user_id": "u_does_not_exist"}),
            (
                "missing_group",
                {"conversation_type": "group", "group_id": "group_zzz",
                 "sender_user_id": "u_002"},
            ),
            (
                "missing_business",
                {"conversation_type": "business", "business_id": "business_zzz",
                 "sender_user_id": ""},
            ),
            ("no_sender_at_all", {"sender_user_id": ""}),
            ("sender_is_the_recipient", {"sender_user_id": "u_001"}),
        ],
    )
    def test_still_produces_a_valid_prediction(
        self,
        broken: Callable[[Sequence[dict[str, str]]], Path],
        case: str,
        overrides: dict[str, str],
    ) -> None:
        message_id = f"edge_{case}"
        dataset = broken([{**_BASE_ROW, "message_id": message_id, **overrides}])
        result = _result(_route(dataset), message_id)
        assert result.action.value in ("notify", "digest", "mute")
        assert 0.0 <= result.confidence <= 1.0
        assert result.reason.text.strip()

    def test_unknown_entities_never_become_evidence(
        self, broken: Callable[[Sequence[dict[str, str]]], Path]
    ) -> None:
        # An unknown recipient has no history, so citing any would be a
        # fabrication - the output validator would reject it, but the engine
        # should not produce it in the first place.
        dataset = broken(
            [{**_BASE_ROW, "message_id": "edge_ghost", "user_id": "u_does_not_exist"}]
        )
        result = _result(_route(dataset), "edge_ghost")
        assert result.evidence_message_ids == NO_EVIDENCE


# --------------------------------------------------------------------------- #
# Malformed values
# --------------------------------------------------------------------------- #


class TestMalformedValues:
    """Cells that are present but unreadable."""

    @pytest.mark.parametrize(
        ("case", "overrides"),
        [
            ("unparseable_timestamp", {"created_at": "not-a-date"}),
            ("blank_timestamp", {"created_at": ""}),
            ("us_format_timestamp", {"created_at": "08/01/2026 10:00"}),
            ("non_numeric_forward_count", {"forwarded_count": "many"}),
            ("negative_forward_count", {"forwarded_count": "-4"}),
            ("blank_forward_count", {"forwarded_count": ""}),
            ("unknown_conversation_type", {"conversation_type": "broadcast"}),
            ("blank_conversation_type", {"conversation_type": ""}),
            ("empty_text", {"message_text": ""}),
            ("whitespace_only_text", {"message_text": "   \n  "}),
        ],
    )
    def test_still_produces_a_valid_prediction(
        self,
        broken: Callable[[Sequence[dict[str, str]]], Path],
        case: str,
        overrides: dict[str, str],
    ) -> None:
        message_id = f"edge_{case}"
        dataset = broken([{**_BASE_ROW, "message_id": message_id, **overrides}])
        result = _result(_route(dataset), message_id)
        assert 0.0 <= result.confidence <= 1.0

    def test_very_long_text_with_delimiters_survives_the_round_trip(
        self, broken: Callable[[Sequence[dict[str, str]]], Path], tmp_path: Path
    ) -> None:
        # Commas, quotes and newlines in a reason are the classic way to
        # produce a CSV that parses into the wrong number of columns.
        text = 'URGENT!!! "quoted", comma\nnewline ₹50,000 \U0001f600 ' + "a" * 4000
        dataset = broken(
            [{**_BASE_ROW, "message_id": "edge_huge", "message_text": text}]
        )
        results = _route(dataset)

        path = write_output_csv(results, tmp_path / "out.csv")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(results)
        assert all(row["message_id"] for row in rows)


class TestBrokenReferenceTables:
    """Defects in the tables messages point *at*, rather than in messages."""

    def test_blank_non_nullable_cells_are_repaired_not_fatal(
        self, dataset_copy: Path, mutate_csv: Callable[..., None]
    ) -> None:
        def blank_fields(rows: list[Row], fields: list[str]) -> tuple[list[Row], list[str]]:
            rows[0]["do_not_disturb_window"] = ""
            rows[1]["messages_opened_30d"] = ""
            rows[2]["messages_replied_30d"] = "abc"
            return rows, fields

        mutate_csv(dataset_copy / "users.csv", blank_fields)
        results = _route(dataset_copy)
        assert len(results) > 0

        loader = DataLoader(dataset_copy)
        loader.records("users")
        assert set(loader.repairs["users"]) == {
            "do_not_disturb_window",
            "messages_opened_30d",
            "messages_replied_30d",
        }

    def test_rows_with_no_primary_key_are_dropped_not_repaired(
        self, dataset_copy: Path, mutate_csv: Callable[..., None]
    ) -> None:
        # A record with no identity cannot be indexed or joined, so inventing
        # one would corrupt every lookup that touches it.
        def blank_key(rows: list[Row], fields: list[str]) -> tuple[list[Row], list[str]]:
            rows[0]["user_id"] = ""
            return rows, fields

        mutate_csv(dataset_copy / "users.csv", blank_key)
        loader = DataLoader(dataset_copy)
        users = loader.records("users")
        assert loader.dropped_rows["users"] == 1
        assert all(user.user_id for user in users)

    def test_unparseable_history_timestamps_do_not_break_evidence(
        self, dataset_copy: Path, mutate_csv: Callable[..., None]
    ) -> None:
        def break_times(rows: list[Row], fields: list[str]) -> tuple[list[Row], list[str]]:
            for row in rows[:20]:
                row["created_at"] = "nope"
            return rows, fields

        mutate_csv(dataset_copy / "message_history.csv", break_times)
        _route(dataset_copy)

    def test_a_group_whose_members_were_not_exported(
        self, dataset_copy: Path, mutate_csv: Callable[..., None]
    ) -> None:
        mutate_csv(
            dataset_copy / "group_members.csv", lambda _rows, fields: ([], fields)
        )
        _route(dataset_copy)

    def test_business_history_missing_entirely(
        self, dataset_copy: Path, mutate_csv: Callable[..., None]
    ) -> None:
        mutate_csv(
            dataset_copy / "user_business_history.csv",
            lambda _rows, fields: ([], fields),
        )
        _route(dataset_copy)


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #


class TestMediaFailures:
    @pytest.mark.parametrize(
        ("case", "overrides"),
        [
            ("unregistered_image", {"media_type": "image", "media_id": "img_zzz"}),
            ("unregistered_voice", {"media_type": "voice", "media_id": "vn_zzz"}),
            ("unknown_modality", {"media_type": "video", "media_id": "vid_001"}),
            ("media_type_without_id", {"media_type": "image", "media_id": ""}),
            ("media_id_without_type", {"media_type": "", "media_id": "img_001"}),
        ],
    )
    def test_broken_attachments_still_route(
        self,
        broken: Callable[[Sequence[dict[str, str]]], Path],
        case: str,
        overrides: dict[str, str],
    ) -> None:
        message_id = f"edge_{case}"
        dataset = broken(
            [{**_BASE_ROW, "message_id": message_id, "message_text": "", **overrides}]
        )
        result = _result(_route(dataset), message_id)
        assert 0.0 <= result.confidence <= 1.0

    def test_missing_media_registry_is_not_fatal(self, dataset_copy: Path) -> None:
        # The registry only maps an attachment to a file; routing does not
        # need the bytes.
        (dataset_copy / "images.csv").unlink()
        (dataset_copy / "voice_notes.csv").unlink()
        _route(dataset_copy)

    def test_missing_media_binaries_are_not_fatal(self, dataset_copy: Path) -> None:
        shutil.rmtree(dataset_copy / "media", ignore_errors=True)
        _route(dataset_copy)

    def test_unreadable_media_lowers_confidence(self, repo: DataRepository) -> None:
        # Deciding without being able to read the message is a real reason to
        # be less sure, and the confidence column should say so.
        voice = [
            m
            for m in repo.get_messages()
            if m.media_type == "voice" and not (m.message_text or "").strip()
        ]
        if not voice:
            pytest.skip("dataset has no text-free voice notes")

        pipeline = RoutingPipeline(repo)
        opaque = [pipeline.route(m).confidence for m in voice]
        readable = [
            pipeline.route(m).confidence
            for m in repo.get_messages()
            if m.media_id is None
        ]
        assert max(opaque) <= max(readable)


# --------------------------------------------------------------------------- #
# Degenerate datasets
# --------------------------------------------------------------------------- #


class TestDegenerateDatasets:
    def test_a_dataset_with_no_incoming_messages(
        self, dataset_copy: Path, mutate_csv: Callable[..., None], tmp_path: Path
    ) -> None:
        mutate_csv(dataset_copy / "messages.csv", lambda _rows, fields: ([], fields))
        repo = DataRepository.load(dataset_copy, validate=False)
        results = RoutingPipeline(repo).route_all()
        assert results == ()

        report = validate_results(results, repo.get_messages(), repo)
        assert report.is_valid is True

        path = write_output_csv(results, tmp_path / "empty.csv")
        assert path.read_text(encoding="utf-8").strip() == (
            "message_id,action,message_type,reason,confidence,evidence_message_ids"
        )

    def test_a_dataset_with_no_history_at_all(
        self, dataset_copy: Path, mutate_csv: Callable[..., None]
    ) -> None:
        # Every personalisation signal loses its input at once. Nothing may be
        # cited as evidence, and everything must still route.
        mutate_csv(
            dataset_copy / "message_events.csv", lambda _rows, fields: ([], fields)
        )
        mutate_csv(
            dataset_copy / "message_history.csv", lambda _rows, fields: ([], fields)
        )
        results = _route(dataset_copy)
        assert all(r.evidence_message_ids == NO_EVIDENCE for r in results)

    def test_a_duplicated_message_id_still_yields_one_row_per_input(
        self, broken: Callable[[Sequence[dict[str, str]]], Path]
    ) -> None:
        # Duplicate input rows are a dataset defect, but the contract is about
        # covering the input, so the duplicate must be predicted too.
        dataset = broken(
            [
                {**_BASE_ROW, "message_id": "edge_dupe"},
                {**_BASE_ROW, "message_id": "edge_dupe", "message_text": "again"},
            ]
        )
        repo = DataRepository.load(dataset)
        results = RoutingPipeline(repo).route_all()
        assert len(results) == len(repo.get_messages())


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


class TestDeterminism:
    """Same input, same bytes. Required by the brief and assumed by graders."""

    def test_two_runs_of_one_pipeline_agree(self, repo: DataRepository) -> None:
        pipeline = RoutingPipeline(repo)
        first = [r.to_output_row() for r in pipeline.route_all()]
        second = [r.to_output_row() for r in pipeline.route_all()]
        assert first == second

    def test_two_independently_loaded_pipelines_agree(self, dataset_dir: Path) -> None:
        # Catches order-dependence that a shared cache would hide.
        first = RoutingPipeline.load(dataset_dir).route_all()
        second = RoutingPipeline.load(dataset_dir).route_all()
        assert [r.to_output_row() for r in first] == [
            r.to_output_row() for r in second
        ]

    def test_written_files_are_byte_identical(
        self, repo: DataRepository, tmp_path: Path
    ) -> None:
        results = RoutingPipeline(repo).route_all()
        one = write_output_csv(results, tmp_path / "one.csv")
        two = write_output_csv(results, tmp_path / "two.csv")
        assert one.read_bytes() == two.read_bytes()

    def test_predictions_follow_input_order(self, repo: DataRepository) -> None:
        results = RoutingPipeline(repo).route_all()
        assert [r.message_id for r in results] == [
            m.message_id for m in repo.get_messages()
        ]


# --------------------------------------------------------------------------- #
# Record coercion, directly
# --------------------------------------------------------------------------- #


class TestRecordCoercion:
    def test_a_clean_row_reports_no_repairs(self) -> None:
        row = {
            "user_id": "u_001",
            "do_not_disturb_window": "22:00-07:00",
            "messages_opened_30d": "10",
            "messages_replied_30d": "4",
            "notifications_dismissed_30d": "2",
            "messages_reported_30d": "0",
        }
        record, repaired = coerce_row(User, row, required=("user_id",))
        assert repaired == ()
        assert record.messages_opened_30d == 10

    def test_repairs_are_neutral_values_not_invented_signals(self) -> None:
        record, repaired = coerce_row(
            User, {"user_id": "u_001"}, required=("user_id",)
        )
        assert set(repaired) == {
            "do_not_disturb_window",
            "messages_opened_30d",
            "messages_replied_30d",
            "notifications_dismissed_30d",
            "messages_reported_30d",
        }
        assert record.messages_opened_30d == 0
        assert record.quiet_hours is None

    def test_a_missing_identifier_still_raises(self) -> None:
        with pytest.raises(RecordCoercionError):
            coerce_row(User, {"messages_opened_30d": "3"}, required=("user_id",))

    def test_strict_construction_is_unchanged(self) -> None:
        # from_row is the deliberate, single-record path and stays strict.
        with pytest.raises(RecordCoercionError):
            User.from_row({"user_id": "u_001"})

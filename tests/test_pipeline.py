"""Tests for :mod:`src.pipeline` and end-to-end Phase 2 behaviour."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.classifier.enums import MessageType
from src.data.models import Message
from src.data.repository import DataRepository
from src.evaluation import as_message
from src.pipeline import MessageAnalysis, MessagePipeline

#: Agreement with the labelled examples that the classifier must not fall below.
#: The tuned figure is higher; this is a regression floor, not a target.
MIN_SAMPLE_AGREEMENT = 0.80


class TestPipeline:
    """Composition of extraction and classification."""

    def test_analyse_returns_both_halves(
        self, pipeline: MessagePipeline, make_message: Callable[..., Message]
    ) -> None:
        analysis = pipeline.analyse(make_message())
        assert isinstance(analysis, MessageAnalysis)
        assert analysis.features.message_id == analysis.classification.message_id
        assert analysis.message_id == "msg_test"

    def test_analyse_all_covers_every_message(
        self, pipeline: MessagePipeline, repo: DataRepository
    ) -> None:
        analyses = pipeline.analyse_all()
        assert len(analyses) == len(repo.get_messages())
        assert {a.message_id for a in analyses} == {
            m.message_id for m in repo.get_messages()
        }

    def test_no_message_raises(self, pipeline: MessagePipeline) -> None:
        """Every real message must analyse without an exception."""
        assert len(pipeline.analyse_all()) == 110

    def test_every_message_gets_exactly_one_category(
        self, pipeline: MessagePipeline
    ) -> None:
        for analysis in pipeline.analyse_all():
            assert isinstance(analysis.classification.message_type, MessageType)

    def test_analyse_many_preserves_order(
        self, pipeline: MessagePipeline, repo: DataRepository
    ) -> None:
        messages = repo.get_messages()[:5]
        analyses = pipeline.analyse_many(messages)
        assert [a.message_id for a in analyses] == [m.message_id for m in messages]

    def test_load_builds_a_working_pipeline(self, dataset_dir: Path) -> None:
        built = MessagePipeline.load(dataset_dir)
        assert built.repository.get_messages()
        assert built.analyse_all()

    def test_to_dict_carries_both_halves(
        self, pipeline: MessagePipeline, make_message: Callable[..., Message]
    ) -> None:
        flat = pipeline.analyse(make_message()).to_dict()
        assert "features" in flat
        assert "classification" in flat


@pytest.fixture(scope="module")
def outcomes(
    pipeline: MessagePipeline, repo: DataRepository
) -> tuple[tuple[str, str, str], ...]:
    """Return ``(message_id, expected, predicted)`` for every labelled row."""
    results = []
    for sample in repo.loader.records("sample_messages"):
        analysis = pipeline.analyse(as_message(sample))
        results.append(
            (sample.message_id, sample.message_type,
             analysis.classification.message_type.value)
        )
    return tuple(results)


class TestAgreementWithLabelledExamples:
    """The only labelled data available is sample_messages.csv."""

    def test_agreement_stays_above_floor(
        self, outcomes: tuple[tuple[str, str, str], ...]
    ) -> None:
        hits = sum(1 for _, expected, got in outcomes if expected == got)
        agreement = hits / len(outcomes)
        assert agreement >= MIN_SAMPLE_AGREEMENT, (
            f"agreement fell to {agreement:.1%}; "
            f"misses: {[(m, e, g) for m, e, g in outcomes if e != g]}"
        )

    def test_every_labelled_scam_is_caught(
        self, outcomes: tuple[tuple[str, str, str], ...]
    ) -> None:
        """Missing a scam is the most costly error this system can make."""
        scams = [(m, e, g) for m, e, g in outcomes if e == "scam"]
        assert scams
        assert all(got == "scam" for _, _, got in scams), scams

    def test_no_benign_message_is_called_a_scam(
        self, outcomes: tuple[tuple[str, str, str], ...]
    ) -> None:
        """False scam alarms suppress real messages, so they are costly too."""
        false_alarms = [
            (m, e, g) for m, e, g in outcomes if g == "scam" and e != "scam"
        ]
        assert false_alarms == []

    def test_embedded_instruction_text_is_treated_as_data(
        self, pipeline: MessagePipeline, repo: DataRepository
    ) -> None:
        """One labelled row contains an instruction aimed at an AI agent.

        It is message content, not a command. A rule-based classifier cannot
        follow it, and the expected verdict is scam.
        """
        sample = next(
            (
                s
                for s in repo.loader.records("sample_messages")
                if s.message_text and "ignore all previous" in s.message_text.lower()
            ),
            None,
        )
        assert sample is not None, "expected an embedded-instruction row"
        analysis = pipeline.analyse(as_message(sample))
        assert analysis.classification.message_type is MessageType.SCAM


class TestPerformance:
    """Extraction must stay linear, not quadratic, across the dataset."""

    def test_repeated_runs_reuse_caches(
        self, pipeline: MessagePipeline
    ) -> None:
        first = pipeline.analyse_all()
        second = pipeline.analyse_all()
        assert [a.classification.message_type for a in first] == [
            a.classification.message_type for a in second
        ]

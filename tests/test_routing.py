"""Tests for the Phase 4 routing engine.

Covers each action, the decision engine's resolution order, evidence
retrieval, reason generation, confidence calibration, the end-to-end pipeline,
and the edge cases the brief calls out.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest

from src.classifier.enums import MessageType
from src.data.models import Message
from src.data.repository import DataRepository
from src.routing.confidence import ConfidenceCalibrator
from src.routing.decision_engine import DecisionEngine
from src.routing.evidence import EvidenceEngine
from src.routing.models import (
    NO_EVIDENCE,
    OUTPUT_COLUMNS,
    DecisionContext,
    RoutingAction,
    RoutingDecision,
    RoutingEvidence,
    RoutingResult,
    RuleOutcome,
)
from src.routing.pipeline import RoutingPipeline
from src.routing.reason_generator import ReasonGenerator
from src.routing.rules import Thresholds


@pytest.fixture(scope="session")
def routing_pipeline(repo: DataRepository, pipeline) -> RoutingPipeline:
    """A routing pipeline sharing the session analysis pipeline."""
    return RoutingPipeline(repo, analysis_pipeline=pipeline)


@pytest.fixture(scope="session")
def all_results(routing_pipeline: RoutingPipeline) -> tuple[RoutingResult, ...]:
    """Routing results for every incoming message, computed once."""
    return routing_pipeline.route_all()


@pytest.fixture(scope="session")
def make_decision_context(
    repo: DataRepository, pipeline
) -> Callable[[Message], DecisionContext]:
    """Return a factory building a full decision context for one message."""

    def _make(message: Message) -> DecisionContext:
        analysis = pipeline.analyse(message)
        assert analysis.routing is not None
        return DecisionContext(
            features=analysis.features,
            classification=analysis.classification,
            signals=analysis.routing,
            repo=repo,
        )

    return _make


class TestRoutingActions:
    """Every allowed action must be reachable, and only those three."""

    def test_only_three_actions_exist(self) -> None:
        assert {a.value for a in RoutingAction} == {"notify", "digest", "mute"}

    def test_all_actions_are_produced(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        produced = {result.action for result in all_results}
        assert produced == set(RoutingAction)

    def test_scam_is_always_muted(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        """The single most important safety property of the system."""
        scams = [r for r in all_results if r.message_type == MessageType.SCAM.value]
        assert scams
        assert all(r.action is RoutingAction.MUTE for r in scams)

    def test_urgent_is_never_muted(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        """Suppressing a genuine emergency is the costliest possible error."""
        urgent = [r for r in all_results if r.message_type == MessageType.URGENT.value]
        assert urgent
        assert all(r.action is not RoutingAction.MUTE for r in urgent)

    def test_notify_reserved_for_a_minority(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        """Interrupting for everything is the same as interrupting for nothing."""
        notified = sum(1 for r in all_results if r.action.interrupts)
        assert notified < len(all_results) / 2


class TestDecisionEngine:
    """Resolution order: overrides, then weighted argmax, then tie-break."""

    def test_override_beats_a_higher_score(self, make_decision_context, repo) -> None:
        """A safety override must win even against overwhelming opposition."""

        def loud_notify(_context, _thresholds):
            yield RuleOutcome("loud", RoutingAction.NOTIFY, 99.0, "Very keen.")

        def veto(_context, _thresholds):
            yield RuleOutcome(
                "veto", RoutingAction.MUTE, 1.0, "Unsafe.", override=True
            )

        engine = DecisionEngine(rules=[loud_notify, veto])
        context = make_decision_context(repo.get_messages()[0])
        decision = engine.decide(context)
        assert decision.action is RoutingAction.MUTE
        assert decision.overridden is True

    def test_argmax_without_overrides(self, make_decision_context, repo) -> None:
        def votes(_context, _thresholds):
            yield RuleOutcome("a", RoutingAction.DIGEST, 2.0, "d")
            yield RuleOutcome("b", RoutingAction.NOTIFY, 1.0, "n")

        engine = DecisionEngine(rules=[votes])
        decision = engine.decide(make_decision_context(repo.get_messages()[0]))
        assert decision.action is RoutingAction.DIGEST
        assert decision.overridden is False

    def test_ties_break_conservatively(self, make_decision_context, repo) -> None:
        """An unresolvable call must not interrupt the user."""

        def tied(_context, _thresholds):
            yield RuleOutcome("a", RoutingAction.NOTIFY, 1.0, "n")
            yield RuleOutcome("b", RoutingAction.MUTE, 1.0, "m")

        engine = DecisionEngine(rules=[tied])
        decision = engine.decide(make_decision_context(repo.get_messages()[0]))
        assert decision.action is RoutingAction.MUTE

    def test_no_rules_yields_a_defined_decision(
        self, make_decision_context, repo
    ) -> None:
        engine = DecisionEngine(rules=[])
        decision = engine.decide(make_decision_context(repo.get_messages()[0]))
        assert decision.action in set(RoutingAction)
        assert decision.outcomes == ()

    def test_decisive_outcomes_support_the_winner(
        self, make_decision_context, repo
    ) -> None:
        engine = DecisionEngine()
        for message in repo.get_messages()[:20]:
            decision = engine.decide(make_decision_context(message))
            assert all(o.action is decision.action for o in decision.decisive)

    def test_margin_and_runner_up(self, make_decision_context, repo) -> None:
        engine = DecisionEngine()
        decision = engine.decide(make_decision_context(repo.get_messages()[0]))
        assert decision.margin >= 0.0
        assert decision.runner_up is not decision.action

    def test_thresholds_are_injectable(self, make_decision_context, repo) -> None:
        """Behaviour must be tunable without editing rule code."""
        context = make_decision_context(
            next(m for m in repo.get_messages() if m.forwarded_count >= 8)
        )
        default = DecisionEngine().decide(context)
        relaxed = DecisionEngine(
            thresholds=Thresholds(heavy_forward_count=999, moderate_forward_count=999)
        ).decide(context)
        assert "heavy_forwarding" in default.rules_fired
        assert "heavy_forwarding" not in relaxed.rules_fired

    def test_rejects_non_positive_weight(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RuleOutcome("bad", RoutingAction.MUTE, 0.0, "x")


class TestEvidence:
    """Evidence must justify the action actually taken."""

    def test_ids_are_real_history(
        self, all_results: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        for result in all_results:
            for message_id in result.evidence.message_ids:
                assert repo.get_history_message(message_id) is not None

    def test_evidence_belongs_to_the_recipient(
        self, all_results: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        """Another user's reaction says nothing about this user."""
        for result in all_results:
            message = repo.get_message(result.message_id)
            assert message is not None
            for message_id in result.evidence.message_ids:
                record = repo.get_history_message(message_id)
                assert record is not None
                assert record.user_id == message.user_id

    def test_mute_evidence_shows_negative_reactions(
        self, all_results: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        """Muting should cite messages the user actually rejected."""
        muted = [
            r for r in all_results
            if r.action is RoutingAction.MUTE and r.evidence.has_evidence
        ]
        assert muted
        negative = 0
        for result in muted:
            for message_id in result.evidence.message_ids:
                event = repo.get_message_event(message_id)
                if event is not None and event.is_negative_signal:
                    negative += 1
        total = sum(len(r.evidence.message_ids) for r in muted)
        assert negative / total > 0.5

    def test_notify_evidence_shows_engagement(
        self, all_results: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        notified = [
            r for r in all_results
            if r.action is RoutingAction.NOTIFY and r.evidence.has_evidence
        ]
        assert notified
        engaged = 0
        for result in notified:
            for message_id in result.evidence.message_ids:
                event = repo.get_message_event(message_id)
                if event is not None and event.message_opened:
                    engaged += 1
        total = sum(len(r.evidence.message_ids) for r in notified)
        assert engaged / total > 0.5

    def test_no_evidence_renders_the_sentinel(self) -> None:
        assert RoutingEvidence().formatted() == NO_EVIDENCE

    def test_evidence_is_capped(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        from src.routing.evidence import MAX_EVIDENCE

        assert all(len(r.evidence.message_ids) <= MAX_EVIDENCE for r in all_results)

    def test_ids_are_unique_within_a_result(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        for result in all_results:
            ids = result.evidence.message_ids
            assert len(set(ids)) == len(ids)

    def test_history_classification_is_cached(
        self, repo: DataRepository, pipeline
    ) -> None:
        engine = EvidenceEngine(
            repo, extractor=pipeline.extractor, classifier=pipeline.classifier
        )
        record = repo.loader.records("message_history")[0]
        first = engine._category_of(record)
        assert engine._category_of(record) is first


class TestReasonGeneration:
    """Reasons come from the rules that decided, never from a template."""

    def test_every_result_has_a_reason(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        assert all(result.reason.text.strip() for result in all_results)

    def test_reasons_are_sentences(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        for result in all_results:
            text = result.reason.text
            assert text[0].isupper()
            assert text.endswith(".")

    def test_reasons_are_not_all_identical(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        """Generic text would be worthless as an explanation."""
        assert len({r.reason.text for r in all_results}) > 10

    def test_reason_traces_to_the_decisive_rules(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        for result in all_results:
            assert result.decision is not None
            if result.decision.decisive:
                assert result.reason.supporting

    def test_falls_back_when_nothing_decided(self) -> None:
        empty = RoutingDecision(
            action=RoutingAction.DIGEST, scores={}, outcomes=(), decisive=()
        )
        reason = ReasonGenerator().generate(empty)
        assert reason.text
        assert reason.text.endswith(".")

    def test_evidence_is_mentioned_only_when_present(self) -> None:
        decision = RoutingDecision(
            action=RoutingAction.MUTE,
            scores={RoutingAction.MUTE: 1.0},
            outcomes=(),
            decisive=(RuleOutcome("r", RoutingAction.MUTE, 1.0, "It is unwanted."),),
        )
        generator = ReasonGenerator()
        without = generator.generate(decision, RoutingEvidence())
        with_evidence = generator.generate(
            decision, RoutingEvidence(message_ids=("message_0001",))
        )
        assert "similar messages" not in without.text
        assert "similar messages" in with_evidence.text

    def test_reason_length_is_bounded(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        from src.routing.reason_generator import MAX_LENGTH

        assert all(len(r.reason.text) <= MAX_LENGTH for r in all_results)


class TestConfidence:
    """Calibration must separate good decisions from shaky ones."""

    def test_bounded(self, all_results: tuple[RoutingResult, ...]) -> None:
        calibrator = ConfidenceCalibrator()
        for result in all_results:
            assert calibrator.model.floor <= result.confidence <= calibrator.model.ceiling

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            RoutingResult(
                message_id="m",
                action=RoutingAction.MUTE,
                message_type="scam",
                reason=ReasonGenerator().generate(
                    RoutingDecision(RoutingAction.MUTE, {}, (), ())
                ),
                confidence=1.5,
            )

    def test_larger_margin_raises_confidence(
        self, make_decision_context, repo
    ) -> None:
        calibrator = ConfidenceCalibrator()
        context = make_decision_context(repo.get_messages()[0])
        narrow = RoutingDecision(
            RoutingAction.MUTE,
            {RoutingAction.MUTE: 2.0, RoutingAction.DIGEST: 1.9},
            (),
            (),
        )
        wide = RoutingDecision(
            RoutingAction.MUTE,
            {RoutingAction.MUTE: 8.0, RoutingAction.DIGEST: 0.1},
            (),
            (),
        )
        assert calibrator.calibrate(context, wide, RoutingEvidence()) > (
            calibrator.calibrate(context, narrow, RoutingEvidence())
        )

    def test_evidence_raises_confidence(self, make_decision_context, repo) -> None:
        calibrator = ConfidenceCalibrator()
        context = make_decision_context(repo.get_messages()[0])
        decision = RoutingDecision(
            RoutingAction.MUTE, {RoutingAction.MUTE: 3.0}, (), ()
        )
        without = calibrator.calibrate(context, decision, RoutingEvidence())
        with_evidence = calibrator.calibrate(
            context, decision, RoutingEvidence(message_ids=("message_0001",))
        )
        assert with_evidence > without

    def test_scam_decisions_are_confident(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        scams = [r for r in all_results if r.message_type == MessageType.SCAM.value]
        assert min(r.confidence for r in scams) >= 0.7


class TestEdgeCases:
    """The brief's edge-case list, none of which may crash."""

    def _route(self, routing_pipeline: RoutingPipeline, message: Message):
        return routing_pipeline.route(message)

    def test_empty_text(
        self, routing_pipeline: RoutingPipeline, make_message: Callable[..., Message]
    ) -> None:
        result = self._route(routing_pipeline, make_message(text=""))
        assert result.action in set(RoutingAction)

    def test_none_text(
        self, routing_pipeline: RoutingPipeline, make_message: Callable[..., Message]
    ) -> None:
        assert self._route(routing_pipeline, make_message(text=None)).reason.text

    @pytest.mark.parametrize(
        ("media_type", "media_id"), [("voice", "vn_004"), ("image", "img_001")]
    )
    def test_media_without_text(
        self,
        routing_pipeline: RoutingPipeline,
        make_message: Callable[..., Message],
        media_type: str,
        media_id: str,
    ) -> None:
        result = self._route(
            routing_pipeline,
            make_message(text=None, media_type=media_type, media_id=media_id),
        )
        assert result.action in set(RoutingAction)

    def test_unknown_media_id(
        self, routing_pipeline: RoutingPipeline, make_message: Callable[..., Message]
    ) -> None:
        result = self._route(
            routing_pipeline,
            make_message(text="see this", media_type="image", media_id="img_999"),
        )
        assert result.action in set(RoutingAction)

    def test_missing_sender(self, routing_pipeline: RoutingPipeline) -> None:
        message = Message(
            message_id="edge_no_sender",
            user_id=next(iter(routing_pipeline.repository.index.users_by_id)),
            conversation_type="personal",
            group_id=None,
            business_id=None,
            sender_user_id=None,
            created_at=datetime(2026, 7, 25, 12, 0),
            message_text="hello",
            media_type=None,
            media_id=None,
            forwarded_count=0,
        )
        assert self._route(routing_pipeline, message).action in set(RoutingAction)

    def test_unknown_group_and_business(
        self, routing_pipeline: RoutingPipeline
    ) -> None:
        message = Message(
            message_id="edge_unknown_refs",
            user_id=next(iter(routing_pipeline.repository.index.users_by_id)),
            conversation_type="group",
            group_id="group_does_not_exist",
            business_id=None,
            sender_user_id="u_does_not_exist",
            created_at=datetime(2026, 7, 25, 12, 0),
            message_text="hello",
            media_type=None,
            media_id=None,
            forwarded_count=0,
        )
        result = self._route(routing_pipeline, message)
        assert result.action in set(RoutingAction)
        assert result.evidence.formatted()

    def test_user_with_no_history(self, routing_pipeline: RoutingPipeline) -> None:
        repo = routing_pipeline.repository
        without_history = next(
            user_id
            for user_id in repo.index.users_by_id
            if not repo.get_user_history(user_id)
        )
        message = Message(
            message_id="edge_no_history",
            user_id=without_history,
            conversation_type="personal",
            group_id=None,
            business_id=None,
            sender_user_id=next(iter(repo.index.users_by_id)),
            created_at=datetime(2026, 7, 25, 12, 0),
            message_text="hello there",
            media_type=None,
            media_id=None,
            forwarded_count=0,
        )
        result = self._route(routing_pipeline, message)
        assert result.evidence_message_ids == NO_EVIDENCE
        assert result.action in set(RoutingAction)

    def test_extreme_forward_count(
        self, routing_pipeline: RoutingPipeline, make_message: Callable[..., Message]
    ) -> None:
        result = self._route(
            routing_pipeline, make_message(text="fwd", forwarded_count=9999)
        )
        assert result.action is RoutingAction.MUTE

    def test_very_long_text(
        self, routing_pipeline: RoutingPipeline, make_message: Callable[..., Message]
    ) -> None:
        result = self._route(routing_pipeline, make_message(text="urgent " * 2000))
        assert result.action in set(RoutingAction)


class TestPipeline:
    """End-to-end wiring and the output contract."""

    def test_routes_every_message(
        self, all_results: tuple[RoutingResult, ...], repo: DataRepository
    ) -> None:
        assert len(all_results) == len(repo.get_messages())
        assert [r.message_id for r in all_results] == [
            m.message_id for m in repo.get_messages()
        ]

    def test_output_row_has_exactly_the_required_columns(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        for result in all_results:
            assert tuple(result.to_output_row()) == tuple(OUTPUT_COLUMNS)

    def test_output_values_are_serialisable(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        for result in all_results:
            row = result.to_output_row()
            assert row["action"] in {"notify", "digest", "mute"}
            assert isinstance(row["reason"], str) and row["reason"]
            assert 0.0 <= row["confidence"] <= 1.0
            assert isinstance(row["evidence_message_ids"], str)

    def test_message_type_matches_phase_two(
        self, all_results: tuple[RoutingResult, ...], pipeline, repo: DataRepository
    ) -> None:
        """Routing must report the category, never invent a different one."""
        for result in all_results[:25]:
            message = repo.get_message(result.message_id)
            assert message is not None
            expected = pipeline.analyse(message).classification.message_type.value
            assert result.message_type == expected

    def test_is_deterministic(
        self, routing_pipeline: RoutingPipeline, repo: DataRepository
    ) -> None:
        message = repo.get_messages()[0]
        first = routing_pipeline.route(message)
        second = routing_pipeline.route(message)
        assert (first.action, first.confidence) == (second.action, second.confidence)
        assert first.evidence.message_ids == second.evidence.message_ids

    def test_rejects_analysis_without_signals(
        self, repo: DataRepository
    ) -> None:
        from src.pipeline import MessagePipeline

        plain = MessagePipeline(repo, personalize=False)
        routing = RoutingPipeline(repo)
        with pytest.raises(ValueError, match="no routing signals"):
            routing.route_analysis(plain.analyse(repo.get_messages()[0]))

    def test_requires_a_personalising_analysis_pipeline(
        self, repo: DataRepository
    ) -> None:
        from src.pipeline import MessagePipeline

        with pytest.raises(ValueError, match="personalize=True"):
            RoutingPipeline(repo, analysis_pipeline=MessagePipeline(repo, personalize=False))

    def test_load_builds_a_working_pipeline(self, dataset_dir) -> None:
        built = RoutingPipeline.load(dataset_dir)
        assert built.route_all()

    def test_to_dict_includes_the_breakdown(
        self, all_results: tuple[RoutingResult, ...]
    ) -> None:
        payload = all_results[0].to_dict()
        assert "decision_breakdown" in payload
        assert "scores" in payload["decision_breakdown"]


#: Fields shared by SampleMessage and Message, for replaying labelled rows.
_SAMPLE_FIELDS = (
    "message_id", "user_id", "conversation_type", "group_id", "business_id",
    "sender_user_id", "created_at", "message_text", "media_type", "media_id",
    "forwarded_count",
)

#: Regression floor on agreement with the labelled actions. The tuned figure is
#: higher; this guards against a change silently undoing the routing work.
MIN_ACTION_AGREEMENT = 0.85


@pytest.fixture(scope="module")
def labelled_outcomes(
    routing_pipeline: RoutingPipeline,
) -> tuple[tuple[str, str, str], ...]:
    """Return ``(message_id, expected_action, routed_action)`` per labelled row."""
    results = []
    for sample in routing_pipeline.repository.loader.records("sample_messages"):
        message = Message(**{f: getattr(sample, f) for f in _SAMPLE_FIELDS})
        routed = routing_pipeline.route(message)
        results.append((sample.message_id, sample.action, routed.action.value))
    return tuple(results)


class TestAgreementWithLabelledActions:
    """The measurement that matters: agreement with ground-truth actions."""

    def test_agreement_stays_above_floor(self, labelled_outcomes) -> None:
        outcomes = labelled_outcomes
        hits = sum(1 for _, expected, got in outcomes if expected == got)
        agreement = hits / len(outcomes)
        assert agreement >= MIN_ACTION_AGREEMENT, (
            f"agreement fell to {agreement:.1%}; "
            f"misses: {[(m, e, g) for m, e, g in outcomes if e != g]}"
        )

    def test_no_labelled_scam_is_delivered(
        self, routing_pipeline: RoutingPipeline
    ) -> None:
        """Letting a labelled scam through is the worst possible failure."""
        for sample in routing_pipeline.repository.loader.records("sample_messages"):
            if sample.message_type != "scam":
                continue
            message = Message(**{f: getattr(sample, f) for f in _SAMPLE_FIELDS})
            assert routing_pipeline.route(message).action is RoutingAction.MUTE

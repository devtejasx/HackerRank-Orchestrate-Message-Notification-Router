"""Tests for the personalisation engine, statistics provider and pipeline wiring."""

from __future__ import annotations

import pytest

from src.classifier.enums import MessageType
from src.data.repository import DataRepository
from src.personalization.base import SignalCalculator, SignalContext
from src.personalization.engine import DEFAULT_CALCULATORS, PersonalizationEngine
from src.personalization.interaction_stats import (
    InteractionScope,
    InteractionStatsProvider,
)
from src.personalization.normalization import NEUTRAL, Contribution
from src.personalization.signal_models import RoutingSignals
from src.pipeline import MessagePipeline

#: Every signal the engine must produce, in declaration order.
EXPECTED_SIGNALS = (
    "sender_priority",
    "business_priority",
    "group_priority",
    "relationship_strength",
    "historical_importance",
    "engagement_modifier",
    "fatigue_modifier",
    "risk_modifier",
    "trust_modifier",
    "urgency_modifier",
)


class TestInteractionStats:
    """Scoped statistics, and the caching that keeps a full pass linear."""

    def test_user_scope_matches_phase_two_rates(
        self, repo: DataRepository, pipeline: MessagePipeline
    ) -> None:
        """The scoped helper must agree with Phase 2 on the user-wide numbers."""
        provider = InteractionStatsProvider(repo)
        for message in repo.get_messages()[:15]:
            features = pipeline.analyse(message).features
            stats = provider.for_user(message.user_id)
            assert stats.total == features.history.total_interactions
            assert stats.open_rate == pytest.approx(features.history.open_rate)
            assert stats.reply_rate == pytest.approx(features.history.reply_rate)

    def test_absent_counterparty_yields_empty_stats(
        self, repo: DataRepository
    ) -> None:
        provider = InteractionStatsProvider(repo)
        empty = provider.for_sender("u_001", None)
        assert empty.total == 0
        assert empty.has_history is False
        assert empty.engagement_trend == NEUTRAL
        assert empty.scope is InteractionScope.SENDER

    def test_results_are_cached(self, repo: DataRepository) -> None:
        provider = InteractionStatsProvider(repo)
        user_id = next(iter(repo.index.history_by_user))
        assert provider.for_user(user_id) is provider.for_user(user_id)

    def test_scoping_excludes_other_recipients(self, repo: DataRepository) -> None:
        provider = InteractionStatsProvider(repo)
        sender_id = next(iter(repo.index.history_by_sender))
        recipients = {
            record.user_id for record in repo.get_sender_history(sender_id)
        }
        scoped_total = sum(
            provider.for_sender(user_id, sender_id).total for user_id in recipients
        )
        assert scoped_total == len(repo.get_sender_history(sender_id))

    def test_rates_are_bounded(self, repo: DataRepository) -> None:
        provider = InteractionStatsProvider(repo)
        for user_id in list(repo.index.history_by_user)[:20]:
            stats = provider.for_user(user_id)
            for rate in (
                stats.open_rate, stats.reply_rate, stats.dismiss_rate,
                stats.report_rate, stats.mute_rate, stats.engagement,
                stats.rejection, stats.engagement_trend,
            ):
                assert 0.0 <= rate <= 1.0


class TestEngine:
    """Composition, validation and output shape."""

    def test_runs_every_default_calculator(
        self, engine: PersonalizationEngine
    ) -> None:
        assert len(engine.calculators) == len(DEFAULT_CALCULATORS) == 10

    def test_produces_every_required_signal(
        self, engine: PersonalizationEngine, repo: DataRepository, pipeline
    ) -> None:
        analysis = pipeline.analyse(repo.get_messages()[0])
        signals = engine.compute(analysis.features, analysis.classification)
        assert tuple(s.name for s in signals.all_signals) == EXPECTED_SIGNALS

    def test_output_identifies_message_and_recipient(
        self, engine: PersonalizationEngine, repo: DataRepository, pipeline
    ) -> None:
        message = repo.get_messages()[0]
        analysis = pipeline.analyse(message)
        signals = engine.compute(analysis.features, analysis.classification)
        assert signals.message_id == message.message_id
        assert signals.user_id == message.user_id

    def test_rejects_mismatched_inputs(
        self, engine: PersonalizationEngine, repo: DataRepository, pipeline
    ) -> None:
        """Personalising the wrong message must fail loudly, not silently."""
        first, second = repo.get_messages()[:2]
        with pytest.raises(ValueError, match="different messages"):
            engine.compute(
                pipeline.analyse(first).features,
                pipeline.analyse(second).classification,
            )

    def test_rejects_duplicate_calculators(self, repo: DataRepository) -> None:
        doubled = [factory() for factory in DEFAULT_CALCULATORS] + [
            DEFAULT_CALCULATORS[0]()
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            PersonalizationEngine(repo, doubled)

    def test_rejects_an_incomplete_calculator_set(self, repo: DataRepository) -> None:
        with pytest.raises(ValueError, match="required signal"):
            PersonalizationEngine(repo, [DEFAULT_CALCULATORS[0]()])

    def test_accepts_a_custom_calculator(self, repo: DataRepository, pipeline) -> None:
        """A calculator can be swapped without touching the engine."""

        class LoudTrust(SignalCalculator):
            name = "trust_modifier"

            def contributions(self, context: SignalContext):  # noqa: ARG002
                return (Contribution("always", 1.0, 1.0, high_reason="Custom rule."),)

            def confidence(self, context: SignalContext) -> float:  # noqa: ARG002
                return 1.0

        replacements = [
            LoudTrust() if factory.name == "trust_modifier" else factory()
            for factory in DEFAULT_CALCULATORS
        ]
        engine = PersonalizationEngine(repo, replacements)
        analysis = pipeline.analyse(repo.get_messages()[0])
        signals = engine.compute(analysis.features, analysis.classification)
        assert signals.trust_modifier.score == 1.0
        assert signals.trust_modifier.reasons == ("Custom rule.",)

    def test_compute_many_reuses_the_cache(
        self, engine: PersonalizationEngine, repo: DataRepository, pipeline
    ) -> None:
        analyses = [pipeline.analyse(m) for m in repo.get_messages()[:10]]
        results = engine.compute_many(
            (a.features, a.classification) for a in analyses
        )
        assert len(results) == 10
        assert [r.message_id for r in results] == [a.message_id for a in analyses]


@pytest.fixture(scope="module")
def all_signals(pipeline: MessagePipeline) -> tuple[RoutingSignals, ...]:
    """Routing signals for every message in the dataset, computed once."""
    return tuple(
        analysis.routing
        for analysis in pipeline.analyse_all()
        if analysis.routing is not None
    )


class TestSignalsOverTheDataset:
    """Properties that must hold for every message in the dataset."""

    def test_every_message_gets_signals(
        self, all_signals: tuple[RoutingSignals, ...], repo: DataRepository
    ) -> None:
        assert len(all_signals) == len(repo.get_messages())

    def test_all_scores_and_confidences_are_bounded(
        self, all_signals: tuple[RoutingSignals, ...]
    ) -> None:
        for signals in all_signals:
            for signal in signals.all_signals:
                assert 0.0 <= signal.score <= 1.0
                assert 0.0 <= signal.confidence <= 1.0
                assert -1.0 <= signal.signed_strength <= 1.0

    def test_inapplicable_signals_are_neutral_with_no_confidence(
        self, all_signals: tuple[RoutingSignals, ...]
    ) -> None:
        """A signal that cannot apply must say nothing, not guess."""
        for signals in all_signals:
            for signal in signals.all_signals:
                if signal.confidence == 0.0:
                    assert signal.signed_strength == 0.0

    def test_every_signal_varies_across_the_dataset(
        self, all_signals: tuple[RoutingSignals, ...]
    ) -> None:
        """A signal with one constant value would carry no information."""
        for name in EXPECTED_SIGNALS:
            scores = {
                signals.by_name(name).score  # type: ignore[union-attr]
                for signals in all_signals
            }
            assert len(scores) > 1, f"{name} never varies"

    def test_reasons_are_produced(
        self, all_signals: tuple[RoutingSignals, ...]
    ) -> None:
        assert all(signals.reasons for signals in all_signals)

    def test_no_routing_decision_is_present(
        self, all_signals: tuple[RoutingSignals, ...]
    ) -> None:
        """Phase 3 must not name a routing action anywhere in its output."""
        actions = {"notify", "digest", "mute"}
        for signals in all_signals:
            assert not actions & {s.name for s in signals.all_signals}
            assert not any(
                reason.strip().lower().rstrip(".") in actions
                for reason in signals.reasons
            )

    def test_scam_messages_are_consistently_suppressed(
        self, pipeline: MessagePipeline
    ) -> None:
        """The clearest end-to-end property Phase 4 will rely on."""
        for analysis in pipeline.analyse_all():
            if analysis.classification.message_type is not MessageType.SCAM:
                continue
            assert analysis.routing is not None
            assert analysis.routing.risk_modifier.signed_strength < 0.0


class TestPipelineIntegration:
    """Phase 3 wiring, and Phase 2 compatibility."""

    def test_routing_is_attached_by_default(
        self, pipeline: MessagePipeline, repo: DataRepository
    ) -> None:
        analysis = pipeline.analyse(repo.get_messages()[0])
        assert isinstance(analysis.routing, RoutingSignals)

    def test_personalisation_can_be_disabled(self, repo: DataRepository) -> None:
        """Existing Phase 2 behaviour must remain reachable unchanged."""
        plain = MessagePipeline(repo, personalize=False)
        analysis = plain.analyse(repo.get_messages()[0])
        assert analysis.routing is None
        assert plain.engine is None
        assert set(analysis.to_dict()) == {"features", "classification"}

    def test_to_dict_includes_routing_when_present(
        self, pipeline: MessagePipeline, repo: DataRepository
    ) -> None:
        payload = pipeline.analyse(repo.get_messages()[0]).to_dict()
        assert "routing" in payload
        assert len(payload["routing"]["signals"]) == 10  # type: ignore[index]

    def test_signals_match_the_analysed_message(
        self, pipeline: MessagePipeline, repo: DataRepository
    ) -> None:
        for message in repo.get_messages()[:20]:
            analysis = pipeline.analyse(message)
            assert analysis.routing is not None
            assert analysis.routing.message_id == message.message_id

    def test_custom_engine_is_used(
        self, repo: DataRepository, engine: PersonalizationEngine
    ) -> None:
        built = MessagePipeline(repo, engine=engine)
        assert built.engine is engine

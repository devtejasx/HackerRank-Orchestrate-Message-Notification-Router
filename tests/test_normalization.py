"""Tests for :mod:`src.personalization.normalization` and the signal records."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.personalization.normalization import (
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    NEUTRAL,
    Contribution,
    blend,
    days_between,
    decay,
    evidence_confidence,
    explain,
    one_sided,
    saturating,
    trend_score,
)
from src.personalization.signal_models import (
    RoutingSignal,
    RoutingSignals,
    SignalPolarity,
)


class TestSaturating:
    """Diminishing-returns mapping for counts."""

    def test_zero_and_negative_map_to_zero(self) -> None:
        assert saturating(0, 5) == 0.0
        assert saturating(-3, 5) == 0.0

    def test_half_point_gives_a_half(self) -> None:
        assert saturating(5, 5) == pytest.approx(0.5)

    def test_is_monotonic_and_bounded(self) -> None:
        values = [saturating(n, 5) for n in range(0, 100, 7)]
        assert values == sorted(values)
        assert all(0.0 <= v < 1.0 for v in values)

    def test_rejects_non_positive_half_point(self) -> None:
        with pytest.raises(ValueError):
            saturating(1, 0)


class TestDecay:
    """Exponential recency decay."""

    def test_now_is_one_and_half_life_is_a_half(self) -> None:
        assert decay(0, 14) == 1.0
        assert decay(14, 14) == pytest.approx(0.5)
        assert decay(28, 14) == pytest.approx(0.25)

    def test_never_happened_is_zero(self) -> None:
        assert decay(None, 14) == 0.0

    def test_future_timestamps_do_not_exceed_one(self) -> None:
        assert decay(-10, 14) == 1.0


class TestBlend:
    """Weighted mean, and what happens with nothing to blend."""

    def test_is_a_weighted_mean(self) -> None:
        contributions = [
            Contribution("a", 1.0, 3.0),
            Contribution("b", 0.0, 1.0),
        ]
        assert blend(contributions) == pytest.approx(0.75)

    def test_result_is_always_bounded(self) -> None:
        contributions = [Contribution(f"c{i}", i / 10, i + 1) for i in range(11)]
        assert 0.0 <= blend(contributions) <= 1.0

    def test_empty_returns_the_default(self) -> None:
        assert blend([]) == NEUTRAL
        assert blend([], default=0.0) == 0.0

    def test_omitting_an_input_renormalises_the_rest(self) -> None:
        """Dropping an unavailable contribution must not drag the score down."""
        full = blend([Contribution("a", 0.8, 1.0), Contribution("b", 0.8, 1.0)])
        partial = blend([Contribution("a", 0.8, 1.0)])
        assert full == pytest.approx(partial)

    def test_rejects_out_of_range_value(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            Contribution("bad", 1.5, 1.0)

    def test_rejects_non_positive_weight(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Contribution("bad", 0.5, 0.0)


class TestOneSided:
    """Rescaling so absence of a phenomenon stays neutral."""

    def test_absence_is_neutral(self) -> None:
        assert one_sided(0.0) == NEUTRAL

    def test_certainty_is_one(self) -> None:
        assert one_sided(1.0) == 1.0

    def test_never_falls_below_neutral(self) -> None:
        assert all(one_sided(v / 10) >= NEUTRAL for v in range(11))


class TestExplain:
    """Reasons come from the same contributions that produced the score."""

    def test_reports_high_and_low_reasons(self) -> None:
        contributions = [
            Contribution("hi", 0.9, 1.0, high_reason="high fired"),
            Contribution("lo", 0.1, 1.0, low_reason="low fired"),
        ]
        assert set(explain(contributions)) == {"high fired", "low fired"}

    def test_ignores_middling_contributions(self) -> None:
        middling = Contribution("mid", NEUTRAL, 1.0, high_reason="h", low_reason="l")
        assert explain([middling]) == ()

    def test_threshold_boundaries(self) -> None:
        assert Contribution("h", HIGH_THRESHOLD, 1.0, high_reason="h").reason == "h"
        assert Contribution("l", LOW_THRESHOLD, 1.0, low_reason="l").reason == "l"

    def test_orders_by_distance_from_neutral(self) -> None:
        contributions = [
            Contribution("mild", 0.7, 1.0, high_reason="mild"),
            Contribution("strong", 1.0, 1.0, high_reason="strong"),
        ]
        assert explain(contributions)[0] == "strong"

    def test_deduplicates_and_limits(self) -> None:
        contributions = [
            Contribution(f"c{i}", 1.0, 1.0, high_reason="same") for i in range(3)
        ]
        assert explain(contributions) == ("same",)
        assert len(explain([Contribution(f"c{i}", 1.0, 1.0, high_reason=f"r{i}")
                            for i in range(5)], limit=2)) == 2


class TestMisc:
    """Remaining primitives."""

    def test_evidence_confidence_grows_with_sample(self) -> None:
        assert evidence_confidence(0, 5) == 0.0
        assert evidence_confidence(5, 5) == pytest.approx(0.5)
        assert evidence_confidence(50, 5) > 0.9

    def test_days_between(self) -> None:
        assert days_between(datetime(2026, 7, 1), datetime(2026, 7, 15)) == 14.0
        assert days_between(None, datetime(2026, 7, 15)) is None

    def test_days_between_never_negative(self) -> None:
        assert days_between(datetime(2026, 7, 20), datetime(2026, 7, 1)) == 0.0

    def test_trend_score(self) -> None:
        assert trend_score(0.5, 0.5) == NEUTRAL
        assert trend_score(1.0, 0.0) == 1.0
        assert trend_score(0.0, 1.0) == 0.0


class TestRoutingSignal:
    """The signal record's contract."""

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            RoutingSignal("x", 1.2, 0.5)
        with pytest.raises(ValueError, match="outside"):
            RoutingSignal("x", 0.5, -0.1)

    def test_signed_strength_respects_polarity(self) -> None:
        boost = RoutingSignal("b", 1.0, 1.0, SignalPolarity.BOOST)
        suppress = RoutingSignal("s", 1.0, 1.0, SignalPolarity.SUPPRESS)
        assert boost.signed_strength == pytest.approx(1.0)
        assert suppress.signed_strength == pytest.approx(-1.0)

    def test_signed_strength_scales_with_confidence(self) -> None:
        certain = RoutingSignal("a", 1.0, 1.0)
        unsure = RoutingSignal("a", 1.0, 0.25)
        assert unsure.signed_strength == pytest.approx(certain.signed_strength * 0.25)

    def test_zero_confidence_cannot_move_anything(self) -> None:
        assert RoutingSignal("a", 1.0, 0.0).signed_strength == 0.0

    def test_neutral_detection(self) -> None:
        assert RoutingSignal("a", NEUTRAL, 1.0).is_neutral is True
        assert RoutingSignal("a", 0.95, 1.0).is_neutral is False

    def test_reason_joins_or_reports_absence(self) -> None:
        assert RoutingSignal("a", 0.5, 0.5, reasons=("x", "y")).reason == "x; y."
        assert "No distinguishing" in RoutingSignal("a", 0.5, 0.5).reason

    def test_to_dict_is_json_friendly(self) -> None:
        payload = RoutingSignal("a", 0.5, 0.5, reasons=("x",)).to_dict()
        assert payload["name"] == "a"
        assert payload["polarity"] == "boost"
        assert payload["reasons"] == ["x"]


def _signal(name: str, score: float = NEUTRAL, confidence: float = 0.5) -> RoutingSignal:
    """Build a signal for container tests."""
    return RoutingSignal(name, score, confidence, reasons=(f"{name} reason",))


class TestRoutingSignals:
    """The container's aggregation behaviour."""

    @pytest.fixture
    def signals(self) -> RoutingSignals:
        names = (
            "sender_priority", "business_priority", "group_priority",
            "relationship_strength", "historical_importance",
            "engagement_modifier", "fatigue_modifier", "risk_modifier",
            "trust_modifier", "urgency_modifier",
        )
        return RoutingSignals(
            message_id="msg_1",
            user_id="u_1",
            **{name: _signal(name) for name in names},
        )

    def test_exposes_ten_signals(self, signals: RoutingSignals) -> None:
        assert len(signals.all_signals) == 10

    def test_identifiers_are_not_signals(self, signals: RoutingSignals) -> None:
        assert all(s.name not in ("msg_1", "u_1") for s in signals.all_signals)

    def test_lookup_by_name(self, signals: RoutingSignals) -> None:
        assert signals.by_name("risk_modifier") is not None
        assert signals.by_name("nope") is None

    def test_collects_every_reason(self, signals: RoutingSignals) -> None:
        assert len(signals.reasons) == 10

    def test_partitions_by_direction(self, signals: RoutingSignals) -> None:
        assert signals.boosting == ()
        assert signals.suppressing == ()

    def test_to_dict_round_trips(self, signals: RoutingSignals) -> None:
        payload = signals.to_dict()
        assert payload["message_id"] == "msg_1"
        assert len(payload["signals"]) == 10

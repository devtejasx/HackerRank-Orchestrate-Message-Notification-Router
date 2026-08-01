"""Tests for each routing-signal calculator, exercised independently.

Every calculator is constructed and run on its own, which is the point of
keeping them independent: a change to one cannot quietly break another's test.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.data.models import Message
from src.personalization.base import SignalCalculator
from src.personalization.business_priority import (
    BusinessPriorityCalculator,
    transaction_likelihood,
)
from src.personalization.engagement import EngagementCalculator
from src.personalization.fatigue import FatigueCalculator
from src.personalization.group_priority import GroupPriorityCalculator
from src.personalization.historical_importance import HistoricalImportanceCalculator
from src.personalization.normalization import NEUTRAL
from src.personalization.relationship import RelationshipStrengthCalculator
from src.personalization.risk import RiskCalculator
from src.personalization.sender_priority import SenderPriorityCalculator
from src.personalization.signal_models import SignalPolarity
from src.personalization.trust import TrustCalculator
from src.personalization.urgency import UrgencyCalculator

ALL_CALCULATORS: tuple[type[SignalCalculator], ...] = (
    SenderPriorityCalculator,
    BusinessPriorityCalculator,
    GroupPriorityCalculator,
    RelationshipStrengthCalculator,
    HistoricalImportanceCalculator,
    EngagementCalculator,
    FatigueCalculator,
    RiskCalculator,
    TrustCalculator,
    UrgencyCalculator,
)


class TestUniversalContract:
    """Guarantees every calculator must honour, whatever it measures."""

    @pytest.mark.parametrize(
        "factory", ALL_CALCULATORS, ids=[c.__name__ for c in ALL_CALCULATORS]
    )
    def test_produces_a_valid_signal_for_every_message(
        self, factory: type[SignalCalculator], repo, make_context
    ) -> None:
        calculator = factory()
        for message in repo.get_messages():
            signal = calculator.calculate(make_context(message))
            assert signal.name == calculator.name
            assert 0.0 <= signal.score <= 1.0
            assert 0.0 <= signal.confidence <= 1.0
            assert all(isinstance(reason, str) for reason in signal.reasons)

    @pytest.mark.parametrize(
        "factory", ALL_CALCULATORS, ids=[c.__name__ for c in ALL_CALCULATORS]
    )
    def test_is_deterministic(
        self, factory: type[SignalCalculator], repo, make_context
    ) -> None:
        calculator = factory()
        context = make_context(repo.get_messages()[0])
        assert calculator.calculate(context) == calculator.calculate(context)

    @pytest.mark.parametrize(
        "factory", ALL_CALCULATORS, ids=[c.__name__ for c in ALL_CALCULATORS]
    )
    def test_reasons_are_capped(
        self, factory: type[SignalCalculator], repo, make_context
    ) -> None:
        calculator = factory()
        for message in repo.get_messages()[:20]:
            signal = calculator.calculate(make_context(message))
            assert len(signal.reasons) <= calculator.max_reasons

    def test_one_sided_signals_never_fall_below_neutral(
        self, repo, make_context
    ) -> None:
        """Absence of risk or urgency must not push priority downward."""
        for factory in (RiskCalculator, UrgencyCalculator):
            calculator = factory()
            for message in repo.get_messages():
                assert calculator.calculate(make_context(message)).score >= NEUTRAL


class TestSenderPriority:
    """Part 1: sender standing."""

    def test_neutral_without_an_individual_sender(
        self, make_context, make_message: Callable[..., Message]
    ) -> None:
        context = make_context(make_message(conversation_type="business"))
        signal = SenderPriorityCalculator().calculate(context)
        assert signal.score == NEUTRAL
        assert signal.confidence == 0.0

    def test_unknown_sender_scores_low_with_low_confidence(
        self, make_context, make_message: Callable[..., Message]
    ) -> None:
        context = make_context(
            make_message(conversation_type="personal", known_sender=False)
        )
        signal = SenderPriorityCalculator().calculate(context)
        assert signal.score < NEUTRAL
        assert signal.confidence < 0.5
        assert any("not messaged" in r or "unknown" in r.lower() for r in signal.reasons)

    def test_known_sender_beats_unknown(
        self, make_context, make_message: Callable[..., Message]
    ) -> None:
        calculator = SenderPriorityCalculator()
        known = calculator.calculate(
            make_context(make_message(conversation_type="personal", known_sender=True))
        )
        unknown = calculator.calculate(
            make_context(make_message(conversation_type="personal", known_sender=False))
        )
        assert known.confidence > unknown.confidence

    def test_polarity_is_boosting(self) -> None:
        assert SenderPriorityCalculator.polarity is SignalPolarity.BOOST


class TestBusinessPriority:
    """Part 2: business standing."""

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("confirmed_travel_booking", 0.9),
            ("recent_food_order", 0.9),
            ("active_bank_account", 0.9),
            ("abandoned_travel_search", 0.2),
            ("doctor_search", 0.4),
        ],
    )
    def test_transaction_likelihood_separates_commitment_from_browsing(
        self, reason: str, expected: float
    ) -> None:
        value = transaction_likelihood(reason)
        assert (value > 0.6) == (expected > 0.5), f"{reason} scored {value}"

    def test_unrecognised_reason_is_neutral(self) -> None:
        assert transaction_likelihood("something_entirely_unseen") == NEUTRAL

    def test_stale_qualifier_lowers_a_commitment(self) -> None:
        assert transaction_likelihood("old_sale_subscription") < transaction_likelihood(
            "active_sale_subscription"
        )

    def test_neutral_without_a_business(
        self, make_context, make_message: Callable[..., Message]
    ) -> None:
        context = make_context(make_message(conversation_type="group"))
        signal = BusinessPriorityCalculator().calculate(context)
        assert signal.score == NEUTRAL
        assert signal.confidence == 0.0

    def test_known_verified_business_scores_above_neutral(
        self, make_context, make_message: Callable[..., Message]
    ) -> None:
        context = make_context(
            make_message(conversation_type="business", known_business=True)
        )
        signal = BusinessPriorityCalculator().calculate(context)
        assert signal.score > NEUTRAL
        assert any("Verified" in reason for reason in signal.reasons)

    def test_relationship_raises_confidence(
        self, make_context, make_message: Callable[..., Message]
    ) -> None:
        calculator = BusinessPriorityCalculator()
        known = calculator.calculate(
            make_context(make_message(conversation_type="business", known_business=True))
        )
        unknown = calculator.calculate(
            make_context(
                make_message(conversation_type="business", known_business=False)
            )
        )
        assert known.score > unknown.score


class TestGroupPriority:
    """Part 3: group standing."""

    def test_neutral_without_a_group(
        self, make_context, make_message: Callable[..., Message]
    ) -> None:
        context = make_context(make_message(conversation_type="personal"))
        signal = GroupPriorityCalculator().calculate(context)
        assert signal.score == NEUTRAL
        assert signal.confidence == 0.0

    def test_muted_group_scores_lower_than_unmuted(self, repo, make_context) -> None:
        """Muting is the heaviest single factor, so it must be visible."""
        calculator = GroupPriorityCalculator()
        scored = []
        for message in repo.get_messages():
            if message.group_id is None:
                continue
            membership = repo.get_group_member(message.group_id, message.user_id)
            if membership is None:
                continue
            signal = calculator.calculate(make_context(message))
            scored.append((membership.group_muted_by_user, signal.score))

        muted = [score for is_muted, score in scored if is_muted]
        unmuted = [score for is_muted, score in scored if not is_muted]
        assert muted and unmuted
        assert sum(muted) / len(muted) < sum(unmuted) / len(unmuted)

    def test_mute_produces_an_explanation(self, repo, make_context) -> None:
        calculator = GroupPriorityCalculator()
        for message in repo.get_messages():
            if message.group_id is None:
                continue
            membership = repo.get_group_member(message.group_id, message.user_id)
            if membership is None or not membership.group_muted_by_user:
                continue
            signal = calculator.calculate(make_context(message))
            assert any("muted" in reason.lower() for reason in signal.reasons)
            return
        pytest.skip("no muted group membership in the dataset")


class TestHistoricalImportance:
    """Part 4: how comparable messages fared before."""

    def test_reports_absence_of_comparable_history(
        self, make_context, make_message: Callable[..., Message]
    ) -> None:
        context = make_context(
            make_message(conversation_type="personal", known_sender=False)
        )
        signal = HistoricalImportanceCalculator().calculate(context)
        assert signal.confidence == 0.0
        assert any("No comparable" in reason for reason in signal.reasons)

    def test_confidence_tracks_comparable_volume(self, repo, make_context) -> None:
        calculator = HistoricalImportanceCalculator()
        signals = [
            calculator.calculate(make_context(message))
            for message in repo.get_messages()
        ]
        assert max(s.confidence for s in signals) > 0.5
        assert min(s.confidence for s in signals) == 0.0


class TestEngagement:
    """Part 5: the user's overall receptiveness."""

    def test_reuses_phase_two_rates(self, repo, make_context) -> None:
        """The signal must be built from Phase 2's history block, not a copy."""
        context = make_context(repo.get_messages()[0])
        contributions = {
            c.name: c.value for c in EngagementCalculator().contributions(context)
        }
        assert contributions["open_rate"] == context.features.history.open_rate
        assert contributions["reply_rate"] == context.features.history.reply_rate

    def test_engaged_users_score_above_disengaged(self, repo, make_context) -> None:
        calculator = EngagementCalculator()
        scored = [
            (context.features.history.open_rate, calculator.calculate(context).score)
            for context in (make_context(m) for m in repo.get_messages())
        ]
        high = [s for rate, s in scored if rate >= 0.7]
        low = [s for rate, s in scored if rate <= 0.3]
        assert high and low
        assert sum(high) / len(high) > sum(low) / len(low)


class TestFatigue:
    """Part 6: notification load."""

    def test_polarity_is_suppressing(self) -> None:
        assert FatigueCalculator.polarity is SignalPolarity.SUPPRESS

    def test_quiet_hours_raises_fatigue(self, repo, make_context) -> None:
        calculator = FatigueCalculator()
        inside, outside = [], []
        for message in repo.get_messages():
            context = make_context(message)
            score = calculator.calculate(context).score
            target = inside if context.features.context.in_quiet_hours else outside
            target.append(score)
        assert inside and outside
        assert sum(inside) / len(inside) > sum(outside) / len(outside)

    def test_quiet_hours_produces_an_explanation(self, repo, make_context) -> None:
        calculator = FatigueCalculator()
        for message in repo.get_messages():
            context = make_context(message)
            if not context.features.context.in_quiet_hours:
                continue
            signal = calculator.calculate(context)
            assert any("quiet hours" in r.lower() for r in signal.reasons)
            return
        pytest.skip("no message lands inside quiet hours")

    def test_outside_quiet_hours_does_not_read_as_unfatigued(
        self, repo, make_context
    ) -> None:
        """Regression: 'outside quiet hours' once dragged the score to zero."""
        calculator = FatigueCalculator()
        outside = [
            calculator.calculate(context).score
            for context in (make_context(m) for m in repo.get_messages())
            if not context.features.context.in_quiet_hours
        ]
        assert min(outside) > 0.2


class TestTrust:
    """Part 7: sender standing, independent of priority."""

    def test_verified_business_scores_above_unverified(
        self, make_context, make_message: Callable[..., Message]
    ) -> None:
        signal = TrustCalculator().calculate(
            make_context(make_message(conversation_type="business", known_business=True))
        )
        assert signal.score > NEUTRAL
        assert any("Verified" in reason for reason in signal.reasons)

    def test_unknown_domain_is_neutral_not_suspicious(self, repo, make_context) -> None:
        """Absence of domain data must not be read as evidence of wrongdoing."""
        calculator = TrustCalculator()
        for message in repo.get_messages():
            if message.business_id is None:
                continue
            business = repo.get_business(message.business_id)
            if business is None or business.sender_domain_matches_official is not None:
                continue
            contributions = {
                c.name: c.value for c in calculator.contributions(make_context(message))
            }
            assert contributions["business_domain"] == NEUTRAL
            return
        pytest.skip("no business with an unknown domain")

    def test_confidence_grows_with_applicable_components(
        self, make_context, make_message: Callable[..., Message]
    ) -> None:
        calculator = TrustCalculator()
        business = calculator.calculate(
            make_context(make_message(conversation_type="business"))
        )
        personal = calculator.calculate(
            make_context(make_message(conversation_type="personal"))
        )
        assert business.confidence > personal.confidence


class TestRisk:
    """Part 8: risk, reusing the Phase 2 verdict without re-classifying."""

    def test_polarity_is_suppressing(self) -> None:
        assert RiskCalculator.polarity is SignalPolarity.SUPPRESS

    def test_confidence_is_the_classifier_confidence(self, repo, make_context) -> None:
        calculator = RiskCalculator()
        for message in repo.get_messages()[:20]:
            context = make_context(message)
            signal = calculator.calculate(context)
            assert signal.confidence == context.classification.confidence

    def test_risk_verdict_always_suppresses(self, repo, make_context) -> None:
        """Every scam or spam verdict must argue for holding the message back."""
        calculator = RiskCalculator()
        risky = [
            calculator.calculate(context)
            for context in (make_context(m) for m in repo.get_messages())
            if context.classification.message_type.is_risk
        ]
        assert risky
        assert all(signal.score > NEUTRAL for signal in risky)
        assert all(signal.signed_strength < 0.0 for signal in risky)

    def test_risk_score_tracks_classifier_certainty(self, repo, make_context) -> None:
        """A scam the classifier is unsure of must not score like a certain one.

        The floor is the classifier's own confidence, so a 51%-confident scam
        lands near 0.6 while an unmistakable one approaches 0.95. Asserting a
        fixed threshold instead would just be a magic number.
        """
        calculator = RiskCalculator()
        pairs = [
            (context.classification.confidence, calculator.calculate(context).score)
            for context in (make_context(m) for m in repo.get_messages())
            if context.classification.message_type.is_risk
        ]
        assert pairs
        assert all(score >= confidence for confidence, score in pairs)

        confident = [s for c, s in pairs if c >= 0.8]
        tentative = [s for c, s in pairs if c < 0.6]
        assert confident and tentative
        assert min(confident) > max(tentative)

    def test_clean_message_is_neutral_not_boosting(self, repo, make_context) -> None:
        """Regression: a clean message once produced a large priority boost."""
        calculator = RiskCalculator()
        clean = [
            calculator.calculate(context)
            for context in (make_context(m) for m in repo.get_messages())
            if not context.classification.message_type.is_risk
        ]
        assert clean
        assert all(signal.signed_strength <= 0.0 for signal in clean)

    def test_states_the_verdict_once(self, repo, make_context) -> None:
        calculator = RiskCalculator()
        for message in repo.get_messages():
            context = make_context(message)
            if not context.classification.message_type.is_risk:
                continue
            reasons = calculator.calculate(context).reasons
            headlines = [r for r in reasons if r.startswith("Classified as")]
            assert len(headlines) == 1
            return
        pytest.skip("no risky message in the dataset")


class TestUrgency:
    """The tenth signal: time criticality."""

    def test_urgent_verdict_scores_above_neutral(self, repo, make_context) -> None:
        from src.classifier.enums import MessageType

        calculator = UrgencyCalculator()
        urgent = [
            calculator.calculate(context).score
            for context in (make_context(m) for m in repo.get_messages())
            if context.classification.message_type is MessageType.URGENT
        ]
        assert urgent
        assert all(score > NEUTRAL for score in urgent)

    def test_non_urgent_never_suppresses(self, repo, make_context) -> None:
        calculator = UrgencyCalculator()
        for message in repo.get_messages():
            assert calculator.calculate(make_context(message)).signed_strength >= 0.0

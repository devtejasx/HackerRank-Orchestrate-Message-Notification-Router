"""Tests for :mod:`src.classifier` - rules, classification and confidence.

Includes a passing case for every one of the eleven categories the classifier
may emit, so no category can silently become unreachable.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.classifier.confidence import ConfidenceModel, score_confidence
from src.classifier.enums import KeywordCategory, MessageType
from src.classifier.message_classifier import MessageClassifier
from src.classifier.rules import DEFAULT_WEIGHTS, Weights, collect_signals
from src.data.models import Message

#: One construction per category, exercising the signal that should win.
CATEGORY_CASES: tuple[tuple[MessageType, dict[str, object]], ...] = (
    (
        MessageType.SCAM,
        {
            "text": "Security alert: your account will be blocked. Verify now at "
                    "account-login.in and reply with the 6 digit OTP.",
            "conversation_type": "personal",
        },
    ),
    (
        MessageType.SPAM,
        # An unsolicited business the recipient has no relationship with,
        # sending media with no text. Matches the one labelled spam example.
        {"text": None, "conversation_type": "business", "known_business": False,
         "media_type": "voice", "media_id": "vn_005"},
    ),
    (
        MessageType.URGENT,
        {"text": "Quick heads-up, the tanker cannot wait, 20 mins max. Critical.",
         "conversation_type": "group"},
    ),
    (
        MessageType.PAYMENT,
        {"text": "Sharing the invoice. The balance amount is due, please settle it.",
         "conversation_type": "personal"},
    ),
    (
        MessageType.EVENT,
        {"text": "Cultural night is scheduled for Sunday. Please RSVP for the party.",
         "conversation_type": "group"},
    ),
    (
        MessageType.PROMOTION,
        {"text": "Mega sale! Flat 50% off with coupon TRY50. Shop now for the best "
                 "price on every deal.",
         "conversation_type": "business"},
    ),
    (
        MessageType.BUSINESS_UPDATE,
        {"text": "Your order has been dispatched and is out for delivery. "
                 "Tracking details are in the app.",
         "conversation_type": "business"},
    ),
    (
        MessageType.FORWARD,
        {"text": "Fwd as received, please share this with everyone. Very viral.",
         "conversation_type": "group", "forwarded_count": 12},
    ),
    (
        MessageType.GREETING,
        {"text": "Good morning everyone! Happy birthday and best wishes, stay blessed.",
         "conversation_type": "group"},
    ),
    (
        MessageType.PERSONAL,
        {"text": "Reached home and had dinner, we can talk tomorrow morning okay?",
         "conversation_type": "personal"},
    ),
    (
        MessageType.UNKNOWN,
        # A one-to-one message from someone with no prior contact and no
        # distinctive content: not enough evidence to call it anything.
        {"text": "Hi, I found your number on the volunteer sheet.",
         "conversation_type": "personal", "known_sender": False},
    ),
)


@pytest.fixture(scope="module")
def classifier() -> MessageClassifier:
    """A classifier with default tuning."""
    return MessageClassifier()


def _classify(pipeline, message: Message):
    """Extract features and classify in one step."""
    return pipeline.analyse(message).classification


class TestEveryCategoryIsReachable:
    """Each of the eleven categories must be produced by at least one message."""

    @pytest.mark.parametrize(
        ("expected", "overrides"),
        CATEGORY_CASES,
        ids=[case[0].value for case in CATEGORY_CASES],
    )
    def test_category(
        self,
        pipeline,
        make_message: Callable[..., Message],
        expected: MessageType,
        overrides: dict[str, object],
    ) -> None:
        result = _classify(pipeline, make_message(**overrides))
        assert result.message_type is expected

    def test_all_categories_covered_by_the_suite(self) -> None:
        """Guards against a category losing its test as cases are edited."""
        assert {case[0] for case in CATEGORY_CASES} == set(MessageType)


class TestClassificationContract:
    """Shape and guarantees of the returned verdict."""

    def test_returns_exactly_one_type(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        result = _classify(pipeline, make_message())
        assert isinstance(result.message_type, MessageType)

    def test_confidence_is_bounded(self, pipeline) -> None:
        for analysis in pipeline.analyse_all():
            assert 0.0 <= analysis.classification.confidence <= 1.0

    def test_reason_names_the_verdict(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        result = _classify(pipeline, make_message(text="Good morning everyone!"))
        assert result.message_type.value in result.classification_reason

    def test_matched_keywords_are_reported(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        result = _classify(pipeline, make_message(text="Please share the OTP now"))
        assert "otp" in result.matched_keywords

    def test_is_deterministic(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        message = make_message(text="Flat 50% off, shop now")
        first = _classify(pipeline, message)
        second = _classify(pipeline, message)
        assert (first.message_type, first.confidence) == (
            second.message_type,
            second.confidence,
        )

    def test_runner_up_and_margin(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        result = _classify(
            pipeline,
            make_message(text="Verify now, share your OTP, account will be blocked"),
        )
        assert result.margin > 0
        assert result.runner_up is not result.message_type

    def test_to_dict_is_json_friendly(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        flat = _classify(pipeline, make_message()).to_dict()
        assert isinstance(flat["message_type"], str)
        assert isinstance(flat["scores"], dict)

    def test_is_risk_flag(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        scam = _classify(
            pipeline,
            make_message(
                text="Verify now and reply with the 6 digit OTP or account "
                     "will be blocked",
                conversation_type="personal",
            ),
        )
        assert scam.is_risk is True
        greeting = _classify(pipeline, make_message(text="Good morning everyone!"))
        assert greeting.is_risk is False


class TestSafetyPrecedence:
    """Risk categories must not be masked by benign ones."""

    def test_credential_request_outranks_payment_language(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        result = _classify(
            pipeline,
            make_message(
                text="Your payment is due. Verify now and share the OTP to release it.",
                conversation_type="personal",
            ),
        )
        assert result.message_type is MessageType.SCAM

    def test_protective_advisory_is_not_a_scam(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        """Negation must stop a safety warning from reading as the threat."""
        result = _classify(
            pipeline,
            make_message(
                text="Safety advisory: the brand will never ask for OTP or "
                     "payment details on calls.",
                conversation_type="business",
            ),
        )
        assert result.message_type is not MessageType.SCAM

    def test_nothing_urgent_is_not_urgent(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        result = _classify(
            pipeline,
            make_message(
                text="Reached home, phone is charging. Talk tomorrow. Nothing urgent.",
                conversation_type="personal",
            ),
        )
        assert result.message_type is not MessageType.URGENT

    def test_known_verified_sender_is_not_impersonation(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        """A verified brand the user deals with is not a scam by domain alone."""
        result = _classify(
            pipeline,
            make_message(
                text="Your order has been dispatched, tracking is in the app.",
                conversation_type="business",
            ),
        )
        assert result.message_type is not MessageType.SCAM


class TestRules:
    """Signal collection and weighting."""

    def test_signals_carry_reasons(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        features = pipeline.extractor.extract(make_message(text="Good morning!"))
        signals = collect_signals(features)
        assert signals
        assert all(signal.reason for signal in signals)
        assert all(signal.weight > 0 for signal in signals)

    def test_keyword_contribution_is_capped(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        """A keyword-dense message must not swamp every other signal."""
        dense = make_message(
            text="otp password pin cvv login code verify now kyc suspended "
                 "jackpot lottery winner"
        )
        features = pipeline.extractor.extract(dense)
        scam_signals = [
            s
            for s in collect_signals(features)
            if s.message_type is MessageType.SCAM
            and s.reason.startswith("matched scam")
        ]
        cap = DEFAULT_WEIGHTS.keyword_cap * DEFAULT_WEIGHTS.scam_keyword
        assert scam_signals[0].weight <= cap

    def test_weights_are_injectable(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        """Behaviour must be tunable without editing rule code."""
        message = make_message(text="Good morning everyone!")
        features = pipeline.extractor.extract(message)

        default_result = MessageClassifier().classify(features)
        assert default_result.message_type is MessageType.GREETING

        muted = MessageClassifier(Weights(greeting_keyword=0.01)).classify(features)
        assert muted.message_type is not MessageType.GREETING

    def test_low_evidence_falls_back_to_unknown(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        strict = MessageClassifier(Weights(minimum_commit_score=99.0))
        features = pipeline.extractor.extract(make_message(text="Good morning!"))
        assert strict.classify(features).message_type is MessageType.UNKNOWN


class TestConfidence:
    """Confidence responds to margin, evidence and ambiguity."""

    def test_bounded_by_the_model(self, pipeline, make_message) -> None:
        model = ConfidenceModel()
        features = pipeline.extractor.extract(make_message())
        for scores in ({}, {MessageType.SCAM: 99.0}, {MessageType.PERSONAL: 0.1}):
            value = score_confidence(features, scores, MessageType.SCAM, model)
            assert model.floor <= value <= model.ceiling

    def test_larger_margin_raises_confidence(self, pipeline, make_message) -> None:
        features = pipeline.extractor.extract(make_message())
        narrow = score_confidence(
            features,
            {MessageType.SCAM: 2.0, MessageType.PERSONAL: 1.9},
            MessageType.SCAM,
        )
        wide = score_confidence(
            features,
            {MessageType.SCAM: 8.0, MessageType.PERSONAL: 0.1},
            MessageType.SCAM,
        )
        assert wide > narrow

    def test_ambiguity_lowers_confidence(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        """No text and no keywords is less certain than a keyword-backed call."""
        silent = pipeline.extractor.extract(
            make_message(text=None, media_type="voice", media_id="vn_004")
        )
        spoken = pipeline.extractor.extract(
            make_message(text="Good morning everyone, happy birthday!")
        )
        scores = {MessageType.PERSONAL: 3.0, MessageType.EVENT: 1.0}
        assert score_confidence(silent, scores, MessageType.PERSONAL) < score_confidence(
            spoken, scores, MessageType.PERSONAL
        )

    def test_obvious_scam_is_highly_confident(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        result = _classify(
            pipeline,
            make_message(
                text="Security alert: account will be blocked. Verify now at "
                     "fake-login.in and reply with the 6 digit OTP.",
                conversation_type="personal",
            ),
        )
        assert result.message_type is MessageType.SCAM
        assert result.confidence >= 0.85

    def test_unknown_verdict_still_scores(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        features = pipeline.extractor.extract(make_message(text=None))
        value = score_confidence(features, {}, MessageType.UNKNOWN)
        assert 0.0 < value <= ConfidenceModel().ceiling

    def test_keyword_family_mapping_covers_keyword_backed_types(self) -> None:
        """Types with a dedicated keyword family must map to one."""
        from src.classifier.confidence import _KEYWORD_FAMILY_FOR_TYPE

        assert _KEYWORD_FAMILY_FOR_TYPE[MessageType.SCAM] is KeywordCategory.SCAM
        assert MessageType.PERSONAL not in _KEYWORD_FAMILY_FOR_TYPE

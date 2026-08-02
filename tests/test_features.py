"""Tests for :mod:`src.features` - text, context, historical and keyword blocks."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import datetime

import pytest

from src.classifier.enums import KeywordCategory
from src.data.models import Message
from src.data.repository import DataRepository
from src.features.extractor import FeatureExtractor
from src.features.feature_models import MessageFeatures
from src.features.text_features import extract_text_features


class TestTextFeatures:
    """Body-derived features."""

    def test_counts_and_flags(self) -> None:
        text = "URGENT!! Pay Rs 500 at pay-now.in or mail me@x.com. Call 9876543210"
        features = extract_text_features(text)

        assert features.length == len(text)
        assert features.word_count > 0
        assert features.digit_count == 13  # "500" plus the ten-digit number
        assert features.uppercase_count == 9  # URGENT, P, R, C
        assert features.contains_url is True
        assert features.contains_email is True
        assert features.contains_phone_number is True
        assert features.contains_currency is True
        assert features.is_empty is False

    def test_empty_text_is_total(self) -> None:
        """Voice notes have no body; every count must still be defined."""
        for value in ("", None, "   "):
            features = extract_text_features(value)
            assert features.is_empty is True
            assert features.length == 0 or value == "   "
            assert features.word_count == 0
            assert features.tokens == ()
            assert features.uppercase_ratio == 0.0
            assert features.lexical_diversity == 0.0
            assert features.contains_url is False

    def test_unique_token_count(self) -> None:
        features = extract_text_features("go go go home")
        assert features.word_count == 4
        assert features.unique_token_count == 2

    def test_lexical_diversity(self) -> None:
        assert extract_text_features("a b c d").lexical_diversity == pytest.approx(1.0)
        assert extract_text_features("a a a a").lexical_diversity == pytest.approx(0.25)

    def test_is_shouty_needs_length(self) -> None:
        assert extract_text_features("OK").is_shouty is False
        assert extract_text_features("BUY NOW LIMITED TIME OFFER!!").is_shouty is True

    def test_normalized_text_and_domains(self) -> None:
        features = extract_text_features("Visit WWW.Foo.COM now")
        assert features.normalized_text == "visit www.foo.com now"
        assert features.domains == ("foo.com",)


class TestContextFeatures:
    """Conversation and sender context, resolved through the repository."""

    def test_group_context(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        features = pipeline.extractor.extract(make_message(conversation_type="group"))
        context = features.context

        assert context.is_group is True
        assert context.is_personal is False
        assert context.group_exists is True
        assert context.sender_exists is True
        assert context.group_size is not None
        assert context.sender_is_admin is True

    def test_business_context(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        features = pipeline.extractor.extract(
            make_message(conversation_type="business")
        )
        context = features.context

        assert context.is_business is True
        assert context.business_exists is True
        assert context.business_verified is True
        assert context.business_domain_matches is True
        assert context.is_trusted_business is True
        assert context.has_domain_mismatch is False

    def test_personal_context_has_no_group_or_business(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        context = pipeline.extractor.extract(
            make_message(conversation_type="personal")
        ).context
        assert context.group_exists is False
        assert context.business_exists is False
        assert context.business_age_days is None
        assert context.business_domain_matches is None

    def test_quiet_hours_detection(
        self, pipeline, repo: DataRepository, make_message: Callable[..., Message]
    ) -> None:
        """A window such as 22:00-07:00 wraps midnight and must still match."""
        message = make_message(conversation_type="personal")
        user = repo.get_user(message.user_id)
        assert user is not None and user.quiet_hours is not None
        start, _ = user.quiet_hours

        inside = make_message(
            conversation_type="personal",
            user_id=message.user_id,
            created_at=datetime(2026, 7, 25, start.hour, (start.minute + 10) % 60),
        )
        assert pipeline.extractor.extract(inside).context.in_quiet_hours is True

    def test_notification_load_is_available_despite_date_gap(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        """Same-day summaries never overlap the message dates; averages do."""
        context = pipeline.extractor.extract(make_message()).context
        assert context.notification_load is None
        assert context.avg_daily_notifications > 0
        assert 0.0 <= context.notification_dismiss_rate <= 1.0

    def test_media_flags(
        self, silent_pipeline, make_message: Callable[..., Message]
    ) -> None:
        """A voice note nothing could transcribe: attached, but with no text.

        Uses the silent pipeline deliberately. With speech-to-text on, the
        transcript *becomes* the body and `is_empty_text` is false - which is
        the point of the integration, and is covered in `test_media.py`.
        """
        features = silent_pipeline.extractor.extract(
            make_message(text=None, media_type="voice", media_id="vn_004")
        )
        assert features.has_media is True
        assert features.contains_attachment is True
        assert features.media_type == "voice"
        assert features.is_empty_text is True

    def test_transcribed_media_flags(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        """The same voice note as shipped: still attached, no longer silent."""
        features = pipeline.extractor.extract(
            make_message(text=None, media_type="voice", media_id="vn_004")
        )
        assert features.has_media is True
        assert features.media_type == "voice"
        assert features.is_empty_text is False
        assert features.has_derived_text is True


class TestHistoricalFeatures:
    """Recipient-scoped history and engagement."""

    def test_rates_are_bounded(self, pipeline) -> None:
        for analysis in pipeline.analyse_all():
            history = analysis.features.history
            for rate in (
                history.open_rate,
                history.reply_rate,
                history.dismiss_rate,
                history.report_rate,
                history.mute_rate,
                history.user_engagement,
                history.business_engagement,
                history.group_engagement,
            ):
                assert 0.0 <= rate <= 1.0

    def test_counts_are_recipient_scoped(
        self, pipeline, repo: DataRepository, make_message: Callable[..., Message]
    ) -> None:
        """History counts must not include other users' messages."""
        message = make_message(conversation_type="group")
        history = pipeline.extractor.extract(message).history

        group_total = len(repo.get_group_history(message.group_id))
        assert history.group_message_count <= group_total

        expected = sum(
            1
            for record in repo.get_group_history(message.group_id)
            if record.user_id == message.user_id
        )
        assert history.group_message_count == expected

    def test_no_history_is_distinguishable_from_disengagement(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        history = pipeline.extractor.extract(make_message()).history
        assert history.has_history == (history.total_interactions > 0)

    def test_business_relationship_flags(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        history = pipeline.extractor.extract(
            make_message(conversation_type="business")
        ).history
        assert history.has_business_relationship is True
        assert history.allows_promotions in (True, False)

    def test_caching_produces_identical_results(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        """Memoised aggregates must not drift between calls."""
        message = make_message()
        first = pipeline.extractor.extract(message).history
        second = pipeline.extractor.extract(message).history
        assert first == second


class TestKeywordFeatures:
    """Keyword block behaviour."""

    def test_matches_are_grouped(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        features = pipeline.extractor.extract(
            make_message(text="Good morning! Verify now and share your OTP.")
        )
        assert features.keywords.has(KeywordCategory.GREETING)
        assert features.keywords.has(KeywordCategory.SCAM)
        assert features.keywords.count(KeywordCategory.SCAM) >= 2
        assert "otp" in features.keywords.words(KeywordCategory.SCAM)

    def test_no_text_yields_no_keywords(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        features = pipeline.extractor.extract(
            make_message(text=None, media_type="image", media_id="img_001")
        )
        assert features.keywords.total_matches == 0
        assert features.keywords.categories == ()

    def test_all_keywords_flattens(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        features = pipeline.extractor.extract(make_message(text="50% off sale, hurry"))
        assert set(features.matched_keywords) == set(features.keywords.all_keywords)


class TestFeatureRecord:
    """The composed record itself."""

    def test_extract_all_covers_every_message(
        self, repo: DataRepository, pipeline
    ) -> None:
        features = pipeline.extractor.extract_all()
        assert len(features) == len(repo.get_messages())
        assert {f.message_id for f in features} == {
            m.message_id for m in repo.get_messages()
        }

    def test_records_are_immutable_and_hashable(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        features = pipeline.extractor.extract(make_message())
        assert len({features}) == 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            features.message_id = "other"  # type: ignore[misc]

    def test_to_dict_is_flat_and_json_friendly(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        flat = pipeline.extractor.extract(make_message()).to_dict()
        assert flat["message_id"] == "msg_test"
        assert "text_length" in flat
        assert "context_is_group" in flat
        assert "history_open_rate" in flat
        assert isinstance(flat["keywords"], dict)

    def test_extractor_reuses_one_matcher(
        self, repo: DataRepository
    ) -> None:
        extractor = FeatureExtractor(repo)
        assert extractor.matcher is extractor.matcher

    def test_features_need_no_further_repository_access(
        self, pipeline, make_message: Callable[..., Message]
    ) -> None:
        """The record must be self-contained for later phases."""
        features: MessageFeatures = pipeline.extractor.extract(make_message())
        assert features.conversation_type
        assert features.created_at is not None
        assert features.text.normalized_text is not None

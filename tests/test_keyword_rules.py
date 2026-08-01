"""Tests for :mod:`src.classifier.keyword_rules` and the vocabulary itself."""

from __future__ import annotations

import pytest

from src.classifier.enums import KeywordCategory
from src.classifier.keyword_rules import (
    DEFAULT_KEYWORDS,
    KeywordMatcher,
    default_matcher,
)

#: Categories the brief requires the engine to recognise.
REQUIRED_CATEGORIES = {
    KeywordCategory.URGENT,
    KeywordCategory.PAYMENT,
    KeywordCategory.PROMOTION,
    KeywordCategory.EVENT,
    KeywordCategory.GREETING,
    KeywordCategory.FORWARD,
    KeywordCategory.SPAM,
    KeywordCategory.SCAM,
}

#: Vocabulary explicitly named in the brief, which must never be dropped.
REQUIRED_KEYWORDS = {
    KeywordCategory.URGENT: ("urgent", "emergency", "asap", "immediately",
                             "help", "critical", "important"),
    KeywordCategory.PAYMENT: ("invoice", "bill", "payment", "upi", "due",
                              "paid", "refund"),
    KeywordCategory.PROMOTION: ("sale", "offer", "discount", "coupon", "deal",
                                "cashback", "buy"),
    KeywordCategory.EVENT: ("meeting", "birthday", "party", "school", "exam",
                            "conference", "event"),
    KeywordCategory.GREETING: ("good morning", "good evening", "happy birthday",
                               "congratulation", "welcome"),
    KeywordCategory.FORWARD: ("share", "viral"),
    KeywordCategory.SPAM: ("limited offer", "buy now", "subscribe", "free gift"),
    KeywordCategory.SCAM: ("lottery", "claim prize", "otp", "verify account",
                           "crypto", "investment", "bank account",
                           "refund immediately", "winner", "click here"),
}


@pytest.fixture(scope="module")
def matcher() -> KeywordMatcher:
    """A matcher over the default dictionaries."""
    return default_matcher()


class TestVocabulary:
    """Invariants the dictionaries must hold."""

    def test_all_required_categories_present(self) -> None:
        assert set(DEFAULT_KEYWORDS) >= REQUIRED_CATEGORIES

    @pytest.mark.parametrize("category", sorted(REQUIRED_KEYWORDS))
    def test_required_keywords_are_matchable(
        self, matcher: KeywordMatcher, category: KeywordCategory
    ) -> None:
        """Every keyword named in the brief must actually match its own text."""
        for keyword in REQUIRED_KEYWORDS[category]:
            matched = matcher.match_by_category(f"please note {keyword} today")
            assert category in matched, f"{category.value}:{keyword} did not match"

    def test_no_within_category_subsumption(self) -> None:
        """No entry may contain another from the same category.

        Otherwise one phrase counts twice and inflates confidence.
        """
        offenders = []
        for category, phrases in DEFAULT_KEYWORDS.items():
            normalised = [phrase.replace("-", " ") for phrase in phrases]
            offenders.extend(
                (category.value, inner, outer)
                for outer in normalised
                for inner in normalised
                if outer != inner and f" {inner} " in f" {outer} "
            )
        assert offenders == []

    def test_no_duplicate_entries(self) -> None:
        for category, phrases in DEFAULT_KEYWORDS.items():
            assert len(set(phrases)) == len(phrases), category.value

    def test_entries_are_lowercase(self) -> None:
        for phrases in DEFAULT_KEYWORDS.values():
            assert all(phrase == phrase.lower() for phrase in phrases)


class TestMatching:
    """Matching behaviour: inflection, phrases and boundaries."""

    def test_matches_are_case_insensitive(self, matcher: KeywordMatcher) -> None:
        assert KeywordCategory.SCAM in matcher.match_by_category("Share your OTP")
        assert KeywordCategory.SCAM in matcher.match_by_category("share your otp")

    def test_inflection_is_tolerated(self, matcher: KeywordMatcher) -> None:
        assert matcher.match_by_category("great offers")[KeywordCategory.PROMOTION] == (
            "offer",
        )
        assert KeywordCategory.GREETING in matcher.match_by_category("congratulations!")

    @pytest.mark.parametrize(
        ("text", "absent"),
        [
            ("a billion dollars", KeywordCategory.PAYMENT),
            ("that was helpful", KeywordCategory.URGENT),
            ("a classic novel", KeywordCategory.EVENT),
        ],
    )
    def test_does_not_match_longer_words(
        self, matcher: KeywordMatcher, text: str, absent: KeywordCategory
    ) -> None:
        assert absent not in matcher.match_by_category(text)

    def test_hyphenated_surface_forms_match(self, matcher: KeywordMatcher) -> None:
        assert KeywordCategory.URGENT in matcher.match_by_category("quick heads-up")

    def test_multi_word_phrase_tolerates_extra_spacing(
        self, matcher: KeywordMatcher
    ) -> None:
        assert KeywordCategory.GREETING in matcher.match_by_category("good    morning")

    def test_each_entry_reported_once(self, matcher: KeywordMatcher) -> None:
        matched = matcher.match_by_category("otp otp otp")
        assert matched[KeywordCategory.SCAM].count("otp") == 1

    def test_empty_text_yields_nothing(self, matcher: KeywordMatcher) -> None:
        assert matcher.match("") == ()
        assert matcher.match(None) == ()

    def test_count_and_categories(self, matcher: KeywordMatcher) -> None:
        text = "Verify now and share your OTP"
        assert matcher.count(text, KeywordCategory.SCAM) >= 2
        assert KeywordCategory.SCAM in matcher.categories(text)


class TestNegationHandling:
    """Negated keywords must not register."""

    def test_protective_advisory_is_not_a_scam(self, matcher: KeywordMatcher) -> None:
        """A message warning that a brand never asks for OTP is the opposite."""
        matched = matcher.match_by_category(
            "The brand says they never ask for OTP or payment details on calls."
        )
        assert KeywordCategory.SCAM not in matched

    def test_nothing_urgent_is_not_urgent(self, matcher: KeywordMatcher) -> None:
        matched = matcher.match_by_category("We can talk tomorrow. Nothing urgent.")
        assert KeywordCategory.URGENT not in matched

    def test_plain_use_still_matches(self, matcher: KeywordMatcher) -> None:
        assert KeywordCategory.SCAM in matcher.match_by_category("please share the OTP")

    def test_mixed_use_still_matches(self, matcher: KeywordMatcher) -> None:
        """A term used plainly somewhere counts, even if negated elsewhere."""
        text = "We never ask for OTP. Now send the OTP to confirm."
        assert KeywordCategory.SCAM in matcher.match_by_category(text)


class TestConfigurability:
    """Dictionaries are injectable data, not hardcoded behaviour."""

    def test_custom_dictionary_is_used(self) -> None:
        custom = KeywordMatcher({KeywordCategory.EVENT: ("standup",)})
        assert KeywordCategory.EVENT in custom.match_by_category("daily standup at 9")
        assert custom.match_by_category("good morning") == {}

    def test_keywords_property_reflects_source(self) -> None:
        custom = KeywordMatcher({KeywordCategory.SPAM: ("blah",)})
        assert custom.keywords == {KeywordCategory.SPAM: ("blah",)}

"""Tests for :mod:`src.utils.text_utils`."""

from __future__ import annotations

import pytest

from src.utils import text_utils as tu


class TestNormalisation:
    """Normalisation, tokenising and whitespace handling."""

    def test_clean_whitespace_collapses_runs(self) -> None:
        assert tu.clean_whitespace("  a \n\n b\tc  ") == "a b c"

    def test_normalize_text_casefolds_and_collapses(self) -> None:
        assert tu.normalize_text("  HeLLo   World \n ") == "hello world"

    def test_normalize_text_applies_nfkc(self) -> None:
        """Full-width lookalikes must fold onto ASCII so matching still works."""
        assert tu.normalize_text("ＯＴＰ") == "otp"

    def test_normalize_text_keeps_punctuation(self) -> None:
        """Keyword phrases such as '% off' and 't&c apply' depend on it."""
        assert tu.normalize_text("50% OFF!") == "50% off!"

    def test_strip_punctuation(self) -> None:
        assert tu.strip_punctuation("a.b,c!") == "a b c"

    @pytest.mark.parametrize("value", ["", None, "   "])
    def test_empty_inputs_are_total(self, value: object) -> None:
        assert tu.normalize_text(value) == ""
        assert tu.tokenize(value) == ()
        assert tu.extract_urls(value) == ()
        assert tu.count_emojis(value) == 0

    def test_tokenize_keeps_contractions(self) -> None:
        assert tu.tokenize("Don't buy NOW!! 50% off") == (
            "don't", "buy", "now", "50", "off",
        )

    def test_unique_tokens(self) -> None:
        assert tu.unique_tokens(("a", "b", "a")) == frozenset({"a", "b"})


class TestCounting:
    """Character and structure counts."""

    def test_counts(self) -> None:
        text = "Hello WORLD 123!!"
        assert tu.count_words(text) == 3
        assert tu.count_digits(text) == 3
        assert tu.count_uppercase(text) == 6
        assert tu.count_punctuation(text) == 2

    def test_uppercase_ratio_is_defined_without_letters(self) -> None:
        assert tu.uppercase_ratio("123 !!") == 0.0

    def test_uppercase_ratio(self) -> None:
        assert tu.uppercase_ratio("AAbb") == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("One. Two! Three?", 3),
            ("no terminator", 1),
            ("", 0),
            ("Dear Customer\n\nYour update is ready.\n\nTeam", 3),
        ],
    )
    def test_count_sentences(self, text: str, expected: int) -> None:
        assert tu.count_sentences(text) == expected

    def test_sentence_count_ignores_dots_inside_domains(self) -> None:
        """A URL must not be split into several sentences."""
        assert tu.count_sentences("Pay at amazonpay-delivery.in now. Then reply!") == 2

    def test_count_emojis_ignores_modifiers(self) -> None:
        assert tu.count_emojis("hi 😀😀") == 2
        assert tu.count_emojis("no emoji here") == 0


class TestExtraction:
    """URL, email, phone and UPI extraction."""

    def test_extracts_explicit_and_bare_urls(self) -> None:
        urls = tu.extract_urls("see https://x.io/a and www.foo.com and bare.in")
        assert urls == ("https://x.io/a", "www.foo.com", "bare.in")

    def test_bare_domain_detection_catches_scam_links(self) -> None:
        """Scam messages here use schemeless lookalike domains."""
        assert "amazonpay-delivery.in" in tu.extract_urls(
            "Pay the fee at amazonpay-delivery.in to release the package"
        )

    def test_email_is_not_reported_as_a_url(self) -> None:
        assert tu.extract_urls("write to me@example.co.in") == ()

    def test_abbreviations_are_not_urls(self) -> None:
        assert tu.extract_urls("Meeting at 3 p.m. today, e.g. tomorrow") == ()

    def test_extract_domains_strips_scheme_and_www(self) -> None:
        assert tu.extract_domains("https://www.Foo.com/path?x=1") == ("foo.com",)

    def test_extract_emails(self) -> None:
        assert tu.extract_emails("a@b.com and a@b.com again") == ("a@b.com",)

    @pytest.mark.parametrize(
        "text",
        ["call +91 98765 43210.", "Ring 9876543210 now", "dial +1-555-123-4567"],
    )
    def test_extract_phone_numbers(self, text: str) -> None:
        assert tu.extract_phone_numbers(text)

    def test_short_digit_runs_are_not_phone_numbers(self) -> None:
        assert tu.extract_phone_numbers("order 12345 costs 99") == ()

    def test_upi_handle_is_distinguished_from_email(self) -> None:
        assert tu.extract_upi_handles("pay arun@okaxis not me@x.com") == ("arun@okaxis",)


class TestMoneySignals:
    """Currency and payment-instrument detection."""

    @pytest.mark.parametrize("text", ["₹500 due", "Rs 500", "costs 20 dollars", "$5"])
    def test_contains_currency(self, text: str) -> None:
        assert tu.contains_currency(text) is True

    def test_no_currency(self) -> None:
        assert tu.contains_currency("see you at five") is False

    @pytest.mark.parametrize(
        "text", ["pay via UPI", "scan the QR code", "send to arun@okaxis", "share CVV"]
    )
    def test_contains_payment_symbol(self, text: str) -> None:
        assert tu.contains_payment_symbol(text) is True

    def test_no_payment_symbol(self) -> None:
        assert tu.contains_payment_symbol("lunch was great") is False


class TestNegation:
    """Negation detection, which several classifier rules depend on."""

    def test_detects_cue_before_match(self) -> None:
        text = "the brand will never ask for otp"
        assert tu.is_negated(text, text.index("otp")) is True

    def test_detects_adjacent_cue(self) -> None:
        text = "nothing urgent"
        assert tu.is_negated(text, text.index("urgent")) is True

    def test_no_cue_means_not_negated(self) -> None:
        text = "please share the otp now"
        assert tu.is_negated(text, text.index("otp")) is False

    def test_match_at_start_is_never_negated(self) -> None:
        assert tu.is_negated("otp please", 0) is False

    def test_cue_outside_window_is_ignored(self) -> None:
        text = "no one told me but you should send the otp"
        assert tu.is_negated(text, text.index("otp")) is False

    def test_term_beginning_with_a_cue_does_not_negate_itself(self) -> None:
        """'no time' and 'cannot wait' start with cue words but are not negated."""
        for phrase in ("no time", "cannot wait", "don't miss"):
            assert tu.is_negated(phrase, 0) is False

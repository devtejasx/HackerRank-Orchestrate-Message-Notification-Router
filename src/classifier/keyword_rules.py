"""Configurable keyword dictionaries and the matcher that applies them.

The vocabularies below start from the categories required by the brief and are
extended with phrasing observed in the labelled examples. That extension is
deliberate: in ``sample_messages.csv`` not one message labelled ``urgent``
contains the word "urgent" - urgency is expressed as time pressure ("20 mins
max", "before EOD", "leaving 15 mins early"). Matching only the literal word
would miss every real case.

Dictionaries are data, not code. Pass a different mapping to
:class:`KeywordMatcher` to retune without touching the engine.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from src.classifier.enums import KeywordCategory
from src.utils.text_utils import is_negated, normalize_text

__all__ = [
    "DEFAULT_KEYWORDS",
    "KeywordMatch",
    "KeywordMatcher",
    "default_matcher",
]


@dataclass(frozen=True, slots=True)
class KeywordMatch:
    """One keyword hit.

    Attributes:
        category: Lexical family the keyword belongs to.
        keyword: The dictionary entry that matched, not the surface form.
    """

    category: KeywordCategory
    keyword: str

    def __str__(self) -> str:
        return f"{self.category.value}:{self.keyword}"


# --------------------------------------------------------------------------- #
# Dictionaries
# --------------------------------------------------------------------------- #

#: Keyword families. Entries are matched case-insensitively on whitespace-
#: normalised text, with tolerance for regular inflection (``offer`` matches
#: ``offers``; ``bill`` does not match ``billion``).
#:
#: Within a category no entry may be a substring of another, so a single
#: phrase cannot be counted twice and inflate confidence. Deliberate overlap
#: *across* categories is fine and meaningful: "buy now" is both promotional
#: and spammy. ``tests/test_keyword_rules.py`` enforces the within-category
#: rule.
DEFAULT_KEYWORDS: Final[Mapping[KeywordCategory, tuple[str, ...]]] = {
    KeywordCategory.URGENT: (
        # Required vocabulary.
        "urgent", "emergency", "asap", "immediately", "help", "critical",
        "important",
        # Time pressure, which is how urgency actually appears in the data.
        "right now", "right away", "at once", "hurry", "heads up",
        "last minute", "before eod", "by eod", "end of day", "today itself",
        "cannot wait", "can not wait", "no time", "running late", "deadline",
        "expires today", "expiring today", "within the hour", "mins max",
        "minutes max", "act fast", "time sensitive", "priority",
    ),
    KeywordCategory.PAYMENT: (
        "invoice", "bill", "payment", "upi", "due", "paid", "refund",
        "pay", "amount", "balance", "transaction", "receipt",
        "outstanding", "instalment", "installment", "emi",
        "settle", "transfer", "reimbursement", "charge", "fee",
    ),
    KeywordCategory.PROMOTION: (
        "sale", "offer", "discount", "coupon", "deal", "cashback", "buy",
        "% off", "percent off", "flat off", "promo",
        "shop now", "order now", "book now", "grab",
        # Deliberately absent: "tap below". It is a generic call to action
        # used just as often by safety advisories and order updates as by
        # marketing, so it separates nothing.
        "exclusive", "lowest price", "best price", "new arrival", "launch",
        "membership", "t&c apply", "terms apply",
        "starting at", "starts at", "per person", "selling",
        "dm if interested",
    ),
    KeywordCategory.EVENT: (
        "meeting", "birthday", "party", "school", "exam", "conference",
        "event", "appointment", "schedule", "reminder", "rsvp", "invite",
        "invitation", "ceremony", "function", "gathering", "picnic",
        "cultural night", "annual day", "sports day", "parent teacher",
        "pta", "bus", "session", "workshop",
        "webinar", "class", "tournament", "trip", "outing",
        # Deliberately absent: "match" and "pickup". Both are far more common
        # in ordinary chatter ("watching the match", "pickup near Gate 2")
        # than in scheduling, and both cost accuracy on the labelled examples.
    ),
    KeywordCategory.GREETING: (
        "good morning", "good evening", "good night", "good afternoon",
        "happy birthday", "congratulation", "welcome",
        "happy anniversary", "happy new year", "happy diwali", "happy holi",
        "eid mubarak", "merry christmas", "best wishes", "many happy returns",
        "stay blessed", "stay positive", "keep smiling", "good vibes",
        "have a great day", "god bless", "blessings",
    ),
    KeywordCategory.FORWARD: (
        # "forward" rather than "forwarded" so the inflection group also
        # covers "forwarding", "forwards" and "pls forward to".
        "forward", "share", "viral", "fwd", "as received",
        "spread the word", "copy paste", "received on whatsapp",
    ),
    KeywordCategory.TRANSACTIONAL: (
        "order", "delivery", "delivered", "dispatched", "shipped",
        "out for delivery", "tracking", "track your", "booking", "booked",
        "confirmed", "confirmation", "prescription", "claim status",
        "your account", "statement", "feedback", "survey", "rate your",
        "ticket", "reservation", "check in", "boarding", "refill",
        "renewal", "expiry", "policy", "warranty", "service request",
    ),
    KeywordCategory.SPAM: (
        "limited offer", "buy now", "subscribe", "free gift",
        "limited time", "limited period", "act now", "hurry up",
        "dont miss", "don't miss", "last chance", "expire soon",
        "expires soon", "no cost", "absolutely free",
        "special price for you", "only for you", "100% free",
        # Deliberately absent: "unsubscribe". It appears in the compliance
        # footer of legitimate marketing ("Reply STOP to unsubscribe"), so it
        # marks a lawful sender rather than a spammy one.
    ),
    KeywordCategory.SCAM: (
        # Required vocabulary.
        "lottery", "claim prize", "otp", "verify account", "crypto",
        "investment", "bank account", "refund immediately", "winner",
        "click here",
        # Credential and account-takeover phrasing seen in the labelled scams.
        "login code", "verification code", "cvv", "pin",
        "password", "verify now", "verify identity",
        "will be blocked", "temporarily blocked", "suspended", "reactivate",
        "kyc", "security alert", "support alert", "unusual activity",
        "prize money", "jackpot", "you have won",
        "guaranteed returns", "double your money", "work from home income",
        # Fee-for-release patterns. "fee" alone is a PAYMENT keyword; these
        # add scam-specific weight on top of it.
        "processing fee", "customs fee", "clearance fee", "delivery fee",
        "reattempt fee",
        "reply with the", "6 digit", "six digit", "gift card",
    ),
}


# --------------------------------------------------------------------------- #
# Matcher
# --------------------------------------------------------------------------- #

#: Regular inflections tolerated on the final word of a keyword.
_INFLECTION: Final[str] = r"(?:s|es|ed|ing)?"

#: Accepted between the words of a multi-word phrase, so a dictionary entry
#: written with spaces also matches the hyphenated surface form.
_WORD_SEPARATOR: Final[str] = r"[\s\-_]+"

#: Characters that must not directly precede or follow a match. Used instead of
#: ``\b`` because several keywords start or end with punctuation ("% off").
_LEFT_GUARD: Final[str] = r"(?<![\w])"
_RIGHT_GUARD: Final[str] = r"(?![\w])"


def _keyword_pattern(keyword: str) -> str:
    """Return the regex source matching one keyword phrase.

    Inner whitespace becomes a flexible separator that also accepts hyphens
    and underscores, so ``heads up`` matches ``heads-up``. A trailing
    inflection group absorbs regular plurals and gerunds without
    over-matching longer words.

    Args:
        keyword: A dictionary entry, already lowercase.

    Returns:
        Regex source, not yet compiled.
    """
    words = keyword.replace("-", " ").split()
    body = _WORD_SEPARATOR.join(re.escape(word) for word in words)

    # Only attach a left guard when the keyword starts with a word character,
    # so entries like "% off" still match.
    left = _LEFT_GUARD if keyword[:1].isalnum() else ""
    right = _RIGHT_GUARD if keyword[-1:].isalnum() else ""
    inflection = _INFLECTION if keyword[-1:].isalpha() else ""
    return f"{left}{body}{inflection}{right}"


class KeywordMatcher:
    """Matches keyword dictionaries against message text.

    Patterns are compiled once per instance, so reuse the same matcher across
    messages rather than constructing one per call.

    Args:
        keywords: Category to keyword-phrase mapping. Defaults to
            :data:`DEFAULT_KEYWORDS`.

    Example:
        >>> KeywordMatcher().categories("Verify now, share OTP")
        (<KeywordCategory.SCAM: 'scam'>,)
    """

    def __init__(
        self, keywords: Mapping[KeywordCategory, Sequence[str]] | None = None
    ) -> None:
        source = keywords if keywords is not None else DEFAULT_KEYWORDS
        self._keywords: Mapping[KeywordCategory, tuple[str, ...]] = {
            category: tuple(phrases) for category, phrases in source.items()
        }
        self._patterns: dict[KeywordCategory, tuple[tuple[str, re.Pattern[str]], ...]] = {
            category: tuple(
                (phrase, re.compile(_keyword_pattern(phrase), re.IGNORECASE))
                for phrase in phrases
            )
            for category, phrases in self._keywords.items()
        }

    @property
    def keywords(self) -> Mapping[KeywordCategory, tuple[str, ...]]:
        """The dictionaries this matcher was built with."""
        return self._keywords

    def match(self, text: object) -> tuple[KeywordMatch, ...]:
        """Return every keyword hit in ``text``.

        A hit is discarded when every one of its occurrences is negated, so
        "we will never ask for OTP" does not register as a scam and "nothing
        urgent" does not register as urgent. A term that appears both negated
        and plainly still counts.

        Args:
            text: Raw message text. Missing or empty text yields no matches.

        Returns:
            Matches ordered by category then dictionary order. Each dictionary
            entry is reported at most once, however often it occurs.
        """
        normalized = normalize_text(text)
        if not normalized:
            return ()

        matches: list[KeywordMatch] = []
        for category, patterns in self._patterns.items():
            for phrase, pattern in patterns:
                if self._has_unnegated_hit(normalized, pattern):
                    matches.append(KeywordMatch(category, phrase))
        return tuple(matches)

    @staticmethod
    def _has_unnegated_hit(text: str, pattern: re.Pattern[str]) -> bool:
        """Whether ``pattern`` occurs at least once without a preceding negation."""
        return any(
            not is_negated(text, found.start()) for found in pattern.finditer(text)
        )

    def match_by_category(
        self, text: object
    ) -> Mapping[KeywordCategory, tuple[str, ...]]:
        """Return matched keywords grouped by category.

        Categories with no hits are omitted, so the mapping doubles as the set
        of categories present.
        """
        grouped: dict[KeywordCategory, list[str]] = {}
        for hit in self.match(text):
            grouped.setdefault(hit.category, []).append(hit.keyword)
        return {category: tuple(words) for category, words in grouped.items()}

    def categories(self, text: object) -> tuple[KeywordCategory, ...]:
        """Return the distinct categories present in ``text``."""
        return tuple(self.match_by_category(text))

    def count(self, text: object, category: KeywordCategory) -> int:
        """Return how many distinct keywords of ``category`` appear."""
        return len(self.match_by_category(text).get(category, ()))


def default_matcher() -> KeywordMatcher:
    """Return a matcher over :data:`DEFAULT_KEYWORDS`.

    Kept as a function rather than a module-level singleton so tests and
    future phases can hold independent instances.
    """
    return KeywordMatcher()


def all_keywords(keywords: Mapping[KeywordCategory, Sequence[str]] | None = None) -> Iterable[str]:
    """Yield every keyword phrase across every category, for diagnostics."""
    source = keywords if keywords is not None else DEFAULT_KEYWORDS
    for phrases in source.values():
        yield from phrases

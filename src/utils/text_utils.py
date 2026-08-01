"""Reusable text analysis primitives.

Every function is pure, total and side-effect free: given ``None`` or an empty
string it returns an empty result rather than raising. All regexes are
compiled once at import.

Deliberately dependency-free (stdlib only) so later phases can reuse these
from any context.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Final

from src.utils.helpers import safe_text

__all__ = [
    "clean_whitespace",
    "normalize_text",
    "strip_punctuation",
    "tokenize",
    "unique_tokens",
    "count_words",
    "count_sentences",
    "count_digits",
    "count_uppercase",
    "uppercase_ratio",
    "count_punctuation",
    "count_emojis",
    "extract_urls",
    "extract_domains",
    "extract_emails",
    "extract_phone_numbers",
    "contains_currency",
    "contains_payment_symbol",
    "extract_upi_handles",
]

# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #

_WHITESPACE_RUN: Final = re.compile(r"\s+")
_PUNCTUATION: Final = re.compile(r"[^\w\s]", re.UNICODE)
_TOKEN: Final = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
#: Sentence terminators. A ``.`` only ends a sentence when followed by
#: whitespace or end-of-text, so dots inside ``x.co.in`` do not split it.
_SENTENCE_END: Final = re.compile(r"[.!?]+(?=\s|$)|\n+")

_EXPLICIT_URL: Final = re.compile(r"\bhttps?://[^\s<>\"')]+", re.IGNORECASE)
_WWW_URL: Final = re.compile(r"\bwww\.[^\s<>\"')]+", re.IGNORECASE)

#: Top-level domains recognised in bare (schemeless) links. Scam messages in
#: this dataset routinely use bare lookalike domains such as
#: ``amazonpay-delivery.in``, so bare-domain detection is not optional.
_KNOWN_TLDS: Final[tuple[str, ...]] = (
    "com", "in", "net", "org", "co", "io", "me", "info", "biz", "app",
    "site", "online", "shop", "store", "live", "link", "click", "xyz",
    "top", "vip", "club", "win", "pw", "icu", "cc", "ru", "uk", "us",
)

_BARE_DOMAIN: Final = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:" + "|".join(_KNOWN_TLDS) + r")\b"
    r"(?:/[^\s<>\"')]*)?",
    re.IGNORECASE,
)

_EMAIL: Final = re.compile(
    r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE
)

#: UPI virtual payment address, e.g. ``name@okhdfcbank``. Distinguished from an
#: email by the absence of a dot in the handle suffix.
_UPI_HANDLE: Final = re.compile(r"\b[a-z0-9._-]{2,}@[a-z]{2,}\b", re.IGNORECASE)

#: Phone numbers: optional country code, then 10+ digits with common
#: separators. The guards only prevent clipping a longer digit run, so a
#: number followed by punctuation still matches.
_PHONE: Final = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,13}\d(?!\d)")

#: Currency symbols and spelled-out currency words.
_CURRENCY_SYMBOLS: Final[frozenset[str]] = frozenset("₹$€£¥₩₽")
_CURRENCY_WORDS: Final = re.compile(
    r"\b(?:rs|inr|usd|eur|gbp|rupees?|dollars?|euros?|paise)\b", re.IGNORECASE
)

#: Markers that a message is moving money, independent of the words used.
_PAYMENT_MARKERS: Final = re.compile(
    r"\b(?:upi|imps|neft|rtgs|vpa|qr\s?code|net\s?banking|debit\s?card|"
    r"credit\s?card|account\s?number|ifsc|cvv|wallet)\b",
    re.IGNORECASE,
)

#: Unicode blocks that hold emoji. Variation selectors and skin-tone modifiers
#: are excluded so a single glyph is never counted twice.
_EMOJI: Final = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001faff"
    "\U0001f1e6-\U0001f1ff"
    "☀-⛿"
    "✀-➿"
    "⬀-⯿"
    "←-⇿"
    "⭐❤"
    "]"
)

#: Stripped before emoji counting so modifiers do not inflate the count.
_EMOJI_MODIFIERS: Final = re.compile("[︎️‍\U0001f3fb-\U0001f3ff]")


def _text_of(value: object) -> str:
    """Return ``value`` as a string, mapping any missing value to ``""``."""
    return safe_text(value, default="") or ""


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def clean_whitespace(value: object) -> str:
    """Collapse every whitespace run to a single space and strip the ends.

    Args:
        value: Any value; missing values become ``""``.

    Returns:
        The cleaned single-line string.
    """
    return _WHITESPACE_RUN.sub(" ", _text_of(value)).strip()


def normalize_text(value: object) -> str:
    """Return a casefolded, whitespace-normalised form for matching.

    Applies NFKC normalisation so lookalike Unicode forms collapse onto their
    ASCII equivalents, then casefolds. Punctuation is preserved, because
    keyword phrases and payment markers depend on it.

    Args:
        value: Any value; missing values become ``""``.

    Returns:
        The normalised string, suitable as a keyword-matching target.
    """
    text = _text_of(value)
    if not text:
        return ""
    return clean_whitespace(unicodedata.normalize("NFKC", text).casefold())


def strip_punctuation(value: object) -> str:
    """Replace every punctuation character with a space."""
    return clean_whitespace(_PUNCTUATION.sub(" ", _text_of(value)))


def tokenize(value: object) -> tuple[str, ...]:
    """Split text into lowercase alphanumeric word tokens.

    Contractions are kept whole (``don't`` stays one token). Punctuation and
    emoji are dropped.

    Args:
        value: Any value; missing values yield an empty tuple.

    Returns:
        Tokens in order of appearance.
    """
    normalized = normalize_text(value)
    if not normalized:
        return ()
    return tuple(match.group(0) for match in _TOKEN.finditer(normalized))


def unique_tokens(tokens: Iterable[str]) -> frozenset[str]:
    """Return the distinct tokens in ``tokens``."""
    return frozenset(tokens)


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #


def count_words(value: object) -> int:
    """Return the number of word tokens."""
    return len(tokenize(value))


def count_sentences(value: object) -> int:
    """Return the number of sentences.

    Sentences are delimited by ``.``, ``!``, ``?`` or a newline. Text with no
    terminator still counts as one sentence; empty text counts as zero.
    """
    text = _text_of(value).strip()
    if not text:
        return 0
    fragments = [part for part in _SENTENCE_END.split(text) if part.strip()]
    return len(fragments) or 1


def count_digits(value: object) -> int:
    """Return the number of digit characters."""
    return sum(1 for char in _text_of(value) if char.isdigit())


def count_uppercase(value: object) -> int:
    """Return the number of uppercase letters."""
    return sum(1 for char in _text_of(value) if char.isupper())


def uppercase_ratio(value: object) -> float:
    """Return uppercase letters as a fraction of all letters.

    Returns:
        A value in ``[0.0, 1.0]``; ``0.0`` when the text holds no letters, so
        the result is always defined.
    """
    text = _text_of(value)
    letters = sum(1 for char in text if char.isalpha())
    if letters == 0:
        return 0.0
    return count_uppercase(text) / letters


def count_punctuation(value: object) -> int:
    """Return the number of punctuation characters."""
    return len(_PUNCTUATION.findall(_text_of(value)))


def count_emojis(value: object) -> int:
    """Return the number of emoji characters.

    Variation selectors, zero-width joiners and skin-tone modifiers are
    stripped first so they never inflate the count. Note this counts emoji
    *characters*, not grapheme clusters: a ZWJ sequence such as a
    profession emoji counts once per component. That is intentional - the
    count exists as a decorativeness signal, not as a rendering measure.
    """
    text = _EMOJI_MODIFIERS.sub("", _text_of(value))
    return len(_EMOJI.findall(text))


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract_urls(value: object) -> tuple[str, ...]:
    """Return every URL, including schemeless links.

    Detects ``https://x.com/y``, ``www.x.com`` and bare ``x.com`` forms.
    Email addresses are excluded so an address is never reported as a link.

    Args:
        value: Any value; missing values yield an empty tuple.

    Returns:
        Distinct URLs in order of first appearance.
    """
    text = _text_of(value)
    if not text:
        return ()

    emails = set(extract_emails(text))
    email_spans = [text.lower().find(email.lower()) for email in emails]

    found: list[str] = []
    for pattern in (_EXPLICIT_URL, _WWW_URL, _BARE_DOMAIN):
        for match in pattern.finditer(text):
            candidate = match.group(0).rstrip(".,;:!?)")
            if not candidate:
                continue
            # Skip anything sitting inside an email address.
            if any(
                start >= 0 and start <= match.start() < start + len(email)
                for start, email in zip(email_spans, emails, strict=False)
            ):
                continue
            if any(candidate.lower() in seen.lower() for seen in found):
                continue
            found.append(candidate)
    return tuple(found)


def extract_domains(value: object) -> tuple[str, ...]:
    """Return the bare host of every URL, lowercased and without ``www.``.

    Useful for comparing a link against a business's official domain.
    """
    hosts: list[str] = []
    for url in extract_urls(value):
        host = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        host = host.split("/", 1)[0].split("?", 1)[0].removeprefix("www.").lower()
        if host and host not in hosts:
            hosts.append(host)
    return tuple(hosts)


def extract_emails(value: object) -> tuple[str, ...]:
    """Return every distinct email address, in order of first appearance."""
    text = _text_of(value)
    if not text:
        return ()
    return tuple(dict.fromkeys(match.group(0) for match in _EMAIL.finditer(text)))


def extract_upi_handles(value: object) -> tuple[str, ...]:
    """Return every UPI virtual payment address (e.g. ``name@okaxis``).

    An ``@`` token whose suffix contains a dot is an email, not a UPI handle,
    and is excluded.
    """
    text = _text_of(value)
    if not text:
        return ()
    emails = {email.lower() for email in extract_emails(text)}
    handles = [
        match.group(0)
        for match in _UPI_HANDLE.finditer(text)
        if match.group(0).lower() not in emails and "." not in match.group(0).split("@")[1]
    ]
    return tuple(dict.fromkeys(handles))


def extract_phone_numbers(value: object) -> tuple[str, ...]:
    """Return every plausible phone number.

    Matches 10-to-14 digit runs with optional country code and common
    separators. Digit runs embedded in words or decimals are ignored.
    """
    text = _text_of(value)
    if not text:
        return ()
    numbers = []
    for match in _PHONE.finditer(text):
        candidate = match.group(0).strip()
        digits = sum(1 for char in candidate if char.isdigit())
        if 10 <= digits <= 14 and candidate not in numbers:
            numbers.append(candidate)
    return tuple(numbers)


# --------------------------------------------------------------------------- #
# Money signals
# --------------------------------------------------------------------------- #


def contains_currency(value: object) -> bool:
    """Return whether the text names an amount of money.

    True for currency symbols (``₹``, ``$``) and for spelled-out currency
    words (``rs``, ``inr``, ``rupees``).
    """
    text = _text_of(value)
    if not text:
        return False
    if any(char in _CURRENCY_SYMBOLS for char in text):
        return True
    return _CURRENCY_WORDS.search(text) is not None


def contains_payment_symbol(value: object) -> bool:
    """Return whether the text references a payment instrument or rail.

    Broader than :func:`contains_currency`: covers UPI handles, card and
    account references, QR codes and bank transfer rails, which can appear
    without any amount being named.
    """
    text = _text_of(value)
    if not text:
        return False
    if _PAYMENT_MARKERS.search(text) is not None:
        return True
    return bool(extract_upi_handles(text))

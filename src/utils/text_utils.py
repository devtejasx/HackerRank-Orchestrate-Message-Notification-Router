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
    "NEGATION_CUES",
    "NEGATION_WINDOW",
    "is_negated",
    "contains_clock_time",
    "extract_shortened_links",
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
#:
#: The second row is the URL-shortener TLDs. They matter more than their
#: obscurity suggests: a shortener hides the destination, which is precisely
#: why phishing uses one, so ``bit.ly/verify-quick`` is a stronger signal than
#: most full domains rather than a weaker one. Their absence meant the single
#: most recognisable shortener in the world parsed as ordinary prose.
#:
#: Widening this list was checked against every message body in the dataset
#: (537 of them, across messages, history and the labelled examples): it
#: produces exactly one new match, and that match is a real scam link.
_KNOWN_TLDS: Final[tuple[str, ...]] = (
    "com", "in", "net", "org", "co", "io", "me", "info", "biz", "app",
    "site", "online", "shop", "store", "live", "link", "click", "xyz",
    "top", "vip", "club", "win", "pw", "icu", "cc", "ru", "uk", "us",
    # Shorteners: bit.ly, ow.ly, cutt.ly, buff.ly, rebrand.ly; goo.gl;
    # is.gd; rb.gy.
    "ly", "gl", "gd", "gy",
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
        # Lowercase before stripping the prefix, so an uppercase "WWW." is
        # removed too.
        host = host.split("/", 1)[0].split("?", 1)[0].lower().removeprefix("www.")
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


# --------------------------------------------------------------------------- #
# Negation
# --------------------------------------------------------------------------- #

#: Words that flip the meaning of a term appearing shortly after them.
#:
#: This matters more than it looks. A safety advisory reading "we will never
#: ask for OTP" contains the single strongest scam keyword in the vocabulary
#: while being the opposite of a scam, and a message signing off "nothing
#: urgent" is the opposite of urgent. Without this guard both inverted.
NEGATION_CUES: Final[frozenset[str]] = frozenset(
    {
        "not", "no", "never", "nothing", "none", "without", "avoid", "beware",
        "dont", "doesnt", "didnt", "wont", "cant", "isnt", "arent", "wasnt",
        "shouldnt", "wouldnt", "neither", "nor", "ignore", "disregard",
        "don't", "doesn't", "didn't", "won't", "can't", "isn't", "aren't",
        "wasn't", "shouldn't", "wouldn't",
    }
)

#: How many preceding words are searched for a negation cue. Four covers
#: "never ask for OTP" without reaching back into an unrelated clause.
NEGATION_WINDOW: Final[int] = 4


def is_negated(text: str, match_start: int, window: int = NEGATION_WINDOW) -> bool:
    """Return whether a negation cue precedes the match at ``match_start``.

    Only text *before* the match is inspected, so a term that itself begins
    with a cue word - "no time", "cannot wait", "don't miss" - is never
    treated as negating itself.

    The search also stops at the previous sentence boundary. Negation does not
    reach across a full stop, and letting it do so inverts the meaning of the
    text that follows: "Pay today to avoid account lock. Scan the QR and send
    a screenshot" reads, to a four-word window that ignores the full stop, as
    though the QR were being warned against rather than pushed.

    Args:
        text: The text the match was found in.
        match_start: Character offset where the matched term begins.
        window: How many preceding words to inspect.

    Returns:
        ``True`` when the term should be read as negated.
    """
    if match_start <= 0:
        return False
    before = text[:match_start]
    sentence_breaks = list(_SENTENCE_END.finditer(before))
    if sentence_breaks:
        before = before[sentence_breaks[-1].end() :]
    # Fold the typographic apostrophe so "don't" tokenises as one word.
    preceding = _TOKEN.findall(before.replace("’", "'"))
    return any(word.lower() in NEGATION_CUES for word in preceding[-window:])


#: An explicit clock time: "7 PM", "9 AM to 11 AM", "7:35", "6.15 a.m.".
#:
#: Hours are bounded to 0-23 and minutes to 0-59 so a price ("Rs 11,000") or a
#: quantity ("1200 sqft") cannot read as a time. The bare-hour form requires an
#: am/pm marker for the same reason.
_CLOCK_TIME: Final = re.compile(
    r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\s*(?:a\.?m\.?|p\.?m\.?)?\b"
    r"|\b(?:[01]?\d|2[0-3])\s*(?:a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)


#: Link-shortening services.
#:
#: A shortener exists to hide where a link goes. That is convenient in a tweet
#: and disqualifying in a message asking someone to log in or pay: the
#: recipient cannot see the destination before clicking, which is exactly the
#: property phishing needs. Legitimate senders in this dataset link to their
#: own named domains or tell the reader to open the app.
_SHORTENER_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "bit.ly", "bitly.com", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
        "is.gd", "buff.ly", "cutt.ly", "rebrand.ly", "rb.gy", "shorturl.at",
        "t.ly", "s.id", "tiny.cc", "shorte.st", "adf.ly", "bl.ink", "lnkd.in",
    }
)


def extract_shortened_links(value: object) -> tuple[str, ...]:
    """Return the link-shortener domains referenced in the text.

    Args:
        value: Any value; missing values yield an empty tuple.

    Returns:
        Matching domains in order of first appearance, deduplicated.
    """
    found = [
        domain
        for domain in extract_domains(value)
        # removeprefix, not lstrip: lstrip takes a character set, so it would
        # eat the leading letters of any domain built from w/., and "www." is
        # already stripped upstream in any case.
        if domain.casefold().removeprefix("www.") in _SHORTENER_DOMAINS
    ]
    return tuple(dict.fromkeys(found))


def contains_clock_time(value: object) -> bool:
    """Return whether the text names an explicit time of day.

    A stand-in for scheduling language that no vocabulary can cover: "7 PM sync
    is still on" and "fire alarm test tomorrow 9 AM to 11 AM" announce a
    scheduled thing without using a single scheduling *word*.
    """
    text = _text_of(value)
    return bool(text) and _CLOCK_TIME.search(text) is not None


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

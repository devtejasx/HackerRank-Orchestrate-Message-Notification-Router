"""Body-only feature extraction.

Pure functions of the message text, with no repository access. Text is parsed
once and every count is derived from that single pass rather than re-scanning
the string per metric.
"""

from __future__ import annotations

from src.features.feature_models import TextFeatures
from src.utils import text_utils
from src.utils.helpers import safe_text

__all__ = ["extract_text_features"]


def extract_text_features(message_text: object) -> TextFeatures:
    """Build :class:`TextFeatures` from a raw message body.

    Voice-note messages carry no text; that is represented by
    :attr:`TextFeatures.is_empty` with all counts at zero rather than by
    ``None``, so downstream arithmetic never needs a guard.

    Args:
        message_text: Raw text, possibly ``None`` or empty.

    Returns:
        A fully populated, immutable feature block.
    """
    raw = safe_text(message_text, default="") or ""

    tokens = text_utils.tokenize(raw)
    urls = text_utils.extract_urls(raw)
    emails = text_utils.extract_emails(raw)
    phone_numbers = text_utils.extract_phone_numbers(raw)

    return TextFeatures(
        length=len(raw),
        word_count=len(tokens),
        sentence_count=text_utils.count_sentences(raw),
        digit_count=text_utils.count_digits(raw),
        uppercase_count=text_utils.count_uppercase(raw),
        uppercase_ratio=text_utils.uppercase_ratio(raw),
        punctuation_count=text_utils.count_punctuation(raw),
        emoji_count=text_utils.count_emojis(raw),
        contains_url=bool(urls),
        contains_email=bool(emails),
        contains_phone_number=bool(phone_numbers),
        contains_currency=text_utils.contains_currency(raw),
        contains_payment_symbol=text_utils.contains_payment_symbol(raw),
        is_empty=not raw.strip(),
        normalized_text=text_utils.normalize_text(raw),
        tokens=tokens,
        unique_token_count=len(text_utils.unique_tokens(tokens)),
        urls=urls,
        domains=text_utils.extract_domains(raw),
        emails=emails,
        phone_numbers=phone_numbers,
    )

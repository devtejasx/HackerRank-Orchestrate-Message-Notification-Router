"""Closed vocabularies for Phase 2 classification.

``MessageType`` is the canonical list of categories the classifier may emit.
:data:`src.config.MESSAGE_TYPES` mirrors it as a plain tuple for the schema
layer, which must not depend on the classifier package; a test asserts the two
never drift apart.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["KeywordCategory", "MessageType"]


class MessageType(StrEnum):
    """The category a message is assigned. Exactly one is returned."""

    PERSONAL = "personal"
    URGENT = "urgent"
    EVENT = "event"
    PAYMENT = "payment"
    BUSINESS_UPDATE = "business_update"
    PROMOTION = "promotion"
    GREETING = "greeting"
    FORWARD = "forward"
    SPAM = "spam"
    SCAM = "scam"
    UNKNOWN = "unknown"

    @property
    def is_risk(self) -> bool:
        """Whether this category represents a safety risk to the recipient."""
        return self in (MessageType.SCAM, MessageType.SPAM)


class KeywordCategory(StrEnum):
    """Lexical families the keyword engine detects.

    A category is evidence, not a verdict: ``PAYMENT`` keywords appear in both
    legitimate invoices and payment scams, so the classifier weighs them
    against context rather than mapping them one-to-one onto a
    :class:`MessageType`.
    """

    URGENT = "urgent"
    PAYMENT = "payment"
    PROMOTION = "promotion"
    EVENT = "event"
    GREETING = "greeting"
    FORWARD = "forward"
    SPAM = "spam"
    SCAM = "scam"


#: Categories whose presence indicates risk regardless of sender.
RISK_CATEGORIES: frozenset[KeywordCategory] = frozenset(
    {KeywordCategory.SCAM, KeywordCategory.SPAM}
)

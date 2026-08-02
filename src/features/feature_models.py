"""Immutable feature records produced by Phase 2.

:class:`MessageFeatures` is composed of four focused blocks rather than one
flat record with fifty fields:

* :class:`TextFeatures` - derived from the message body alone.
* :class:`ContextFeatures` - who sent it, through what conversation, to whom.
* :class:`HistoricalFeatures` - what this user has done with similar messages.
* :class:`KeywordFeatures` - which lexical families the body triggers.

Composition keeps each block independently testable and lets later phases
depend on just the slice they need. Convenience properties on
:class:`MessageFeatures` expose the most-used fields directly, so callers are
never forced to reach through three levels.

Every record is frozen and hashable, so features can be cached and shared.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from src.classifier.enums import KeywordCategory

__all__ = [
    "ContextFeatures",
    "HistoricalFeatures",
    "KeywordFeatures",
    "MediaFeatures",
    "MessageFeatures",
    "TextFeatures",
]

# --------------------------------------------------------------------------- #
# Thresholds used by derived properties. Named so no bare number appears in a
# comparison, and grouped so they are tunable in one place.
# --------------------------------------------------------------------------- #

#: Uppercase ratio at or above which text counts as shouting.
SHOUTY_UPPERCASE_RATIO: float = 0.6

#: Minimum characters before the shouting test applies, so a short "OK" is not
#: treated as shouting.
SHOUTY_MIN_LENGTH: int = 20

#: Dismiss rate at or above which past reception counts as negative.
NEGATIVE_RECEPTION_RATE: float = 0.5


@dataclass(frozen=True, slots=True)
class TextFeatures:
    """Everything derivable from the message body alone.

    Attributes:
        length: Character count of the raw text.
        word_count: Number of word tokens.
        sentence_count: Number of sentences.
        digit_count: Number of digit characters.
        uppercase_count: Number of uppercase letters.
        uppercase_ratio: Uppercase letters over all letters, in ``[0, 1]``.
        punctuation_count: Number of punctuation characters.
        emoji_count: Number of emoji characters.
        contains_url: Whether any link is present, including bare domains.
        contains_email: Whether an email address is present.
        contains_phone_number: Whether a phone number is present.
        contains_currency: Whether an amount of money is named.
        contains_payment_symbol: Whether a payment rail or instrument appears.
        is_empty: Whether the message carries no text at all.
        normalized_text: Casefolded, whitespace-normalised body.
        tokens: Word tokens in order.
        unique_token_count: Number of distinct tokens.
        urls: Extracted links.
        domains: Hosts of the extracted links.
        emails: Extracted email addresses.
        phone_numbers: Extracted phone numbers.
    """

    length: int
    word_count: int
    sentence_count: int
    digit_count: int
    uppercase_count: int
    uppercase_ratio: float
    punctuation_count: int
    emoji_count: int
    contains_url: bool
    contains_email: bool
    contains_phone_number: bool
    contains_currency: bool
    contains_payment_symbol: bool
    is_empty: bool
    normalized_text: str
    tokens: tuple[str, ...]
    unique_token_count: int
    urls: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()
    phone_numbers: tuple[str, ...] = ()

    @property
    def is_shouty(self) -> bool:
        """Whether the text is mostly uppercase, a common promotional tell.

        Requires a minimum length so a short "OK" is not treated as shouting.
        """
        return self.uppercase_ratio >= SHOUTY_UPPERCASE_RATIO and self.length >= SHOUTY_MIN_LENGTH

    @property
    def lexical_diversity(self) -> float:
        """Distinct tokens over total tokens, in ``[0, 1]``.

        ``0.0`` for empty text. Low values suggest repetitive, templated copy.
        """
        if self.word_count == 0:
            return 0.0
        return self.unique_token_count / self.word_count


@dataclass(frozen=True, slots=True)
class ContextFeatures:
    """Who sent the message, through what conversation, and to whom.

    Attributes:
        conversation_type: ``personal``, ``group`` or ``business``.
        is_personal: Whether this is a one-to-one chat.
        is_group: Whether this arrived through a group.
        is_business: Whether a business account sent it.
        media_type: ``image``, ``voice`` or ``None``.
        has_media: Whether an attachment is present.
        forwarded_count: Platform forwarding signal.
        sender_exists: Whether ``sender_user_id`` resolves to a known user.
        group_exists: Whether ``group_id`` resolves to a known group.
        business_exists: Whether ``business_id`` resolves to a known business.
        business_verified: Whether the business carries a verified badge.
        business_age_days: Age of the business account.
        business_domain_matches: Whether the sending domain equals the official
            one. ``None`` when either domain is unknown.
        business_reports_30d: Reports raised against the business recently.
        group_size: Declared member count of the group.
        group_message_volume_30d: Messages seen in the group recently.
        sender_is_admin: Whether the sender administers this group.
        recipient_is_admin: Whether the recipient administers this group.
        group_muted: Whether the recipient muted this group.
        in_quiet_hours: Whether the message landed inside the recipient's
            do-not-disturb window.
        notification_load: Notifications the recipient received on the day the
            message arrived. Almost always ``None`` in the shipped dataset:
            ``daily_notification_summary`` covers 2026-07-04 to 07-17 while
            incoming messages fall in 2026-07-18 to 07-31, so the two ranges
            do not overlap. Use :attr:`avg_daily_notifications` instead.
        notifications_dismissed_today: Same-day dismissals, with the same
            coverage caveat.
        avg_daily_notifications: Mean notifications per day across every day
            recorded for this recipient. This is the usable notification-load
            signal and is what a fatigue-aware router should read.
        avg_daily_dismissed: Mean dismissals per day for this recipient.
        notification_dismiss_rate: Dismissed over sent across the recorded
            window, in ``[0, 1]``.
        recent_activity: Recipient's messages opened in the last 30 days.
    """

    conversation_type: str
    is_personal: bool
    is_group: bool
    is_business: bool
    media_type: str | None
    has_media: bool
    forwarded_count: int
    sender_exists: bool
    group_exists: bool
    business_exists: bool
    business_verified: bool
    business_age_days: int | None
    business_domain_matches: bool | None
    business_reports_30d: int | None
    group_size: int | None
    group_message_volume_30d: int | None
    sender_is_admin: bool
    recipient_is_admin: bool
    group_muted: bool
    in_quiet_hours: bool
    notification_load: int | None
    notifications_dismissed_today: int | None
    avg_daily_notifications: float
    avg_daily_dismissed: float
    notification_dismiss_rate: float
    recent_activity: int

    @property
    def is_trusted_business(self) -> bool:
        """Whether the sender is a verified business using its official domain."""
        return (
            self.business_exists
            and self.business_verified
            and self.business_domain_matches is True
        )

    @property
    def has_domain_mismatch(self) -> bool:
        """Whether a business is sending from a domain that is not its own.

        A strong impersonation signal. ``False`` when the domain is unknown,
        so absence of data never reads as evidence of wrongdoing.
        """
        return self.business_exists and self.business_domain_matches is False


@dataclass(frozen=True, slots=True)
class HistoricalFeatures:
    """What this recipient has previously done with comparable messages.

    Rates are computed over the recipient's own historical interactions and
    are ``0.0`` when there is no history, with :attr:`has_history` available to
    tell "no history" apart from "history, all negative".

    Attributes:
        sender_message_count: Past messages from this sender to this user.
        group_message_count: Past messages seen in this group.
        business_message_count: Past messages from this business to this user.
        total_interactions: Historical messages received by this user.
        open_rate: Fraction of past messages the user opened.
        reply_rate: Fraction the user replied to.
        dismiss_rate: Fraction whose notification the user dismissed.
        report_rate: Fraction the user reported.
        mute_rate: Fraction after which the user muted the conversation.
        user_engagement: Overall engagement of this user, in ``[0, 1]``.
        business_engagement: Engagement with this business, in ``[0, 1]``.
        group_engagement: Engagement with this group, in ``[0, 1]``.
        has_business_relationship: Whether a prior relationship is recorded.
        allows_promotions: Whether the user accepts promotions from this
            business. ``None`` when there is no relationship.
        opted_out_of_promotions: Whether the user explicitly opted out.
    """

    sender_message_count: int
    group_message_count: int
    business_message_count: int
    total_interactions: int
    open_rate: float
    reply_rate: float
    dismiss_rate: float
    report_rate: float
    mute_rate: float
    user_engagement: float
    business_engagement: float
    group_engagement: float
    has_business_relationship: bool
    allows_promotions: bool | None
    opted_out_of_promotions: bool

    @property
    def has_history(self) -> bool:
        """Whether any historical interaction exists for this user."""
        return self.total_interactions > 0

    @property
    def is_negatively_received(self) -> bool:
        """Whether the user usually dismisses or reports messages like this."""
        return self.dismiss_rate >= NEGATIVE_RECEPTION_RATE or self.report_rate > 0.0


@dataclass(frozen=True, slots=True)
class KeywordFeatures:
    """Which lexical families the message body triggers.

    Attributes:
        matches: Category to matched dictionary entries. Categories with no
            hits are absent.
    """

    matches: Mapping[KeywordCategory, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the mapping so the record is genuinely immutable."""
        object.__setattr__(self, "matches", dict(self.matches))

    def __hash__(self) -> int:
        return hash(tuple(sorted((k, v) for k, v in self.matches.items())))

    def has(self, category: KeywordCategory) -> bool:
        """Whether any keyword of ``category`` matched."""
        return bool(self.matches.get(category))

    def count(self, category: KeywordCategory) -> int:
        """Number of distinct keywords of ``category`` that matched."""
        return len(self.matches.get(category, ()))

    def words(self, category: KeywordCategory) -> tuple[str, ...]:
        """Matched keywords of ``category``, or an empty tuple."""
        return self.matches.get(category, ())

    @property
    def categories(self) -> tuple[KeywordCategory, ...]:
        """Categories with at least one match, in detection order."""
        return tuple(self.matches)

    @property
    def all_keywords(self) -> tuple[str, ...]:
        """Every matched keyword across every category."""
        return tuple(word for words in self.matches.values() for word in words)

    @property
    def total_matches(self) -> int:
        """Total number of matched keywords."""
        return len(self.all_keywords)


@dataclass(frozen=True, slots=True)
class MediaFeatures:
    """What is attached to the message, and what could be read out of it.

    Present on every message, empty on most. It exists so multimodal handling
    has a home in the feature record *before* there is a model to fill it: with
    no OCR or speech-to-text installed, :attr:`derived_text` is empty and
    :attr:`derived_from` is ``"none"``, which is a fact worth recording rather
    than an absence to be inferred.

    :attr:`is_registered` and :attr:`file_exists` are kept apart on purpose. An
    id missing from the registry is a dataset defect; a registry entry pointing
    at a file that is not on disk is a packaging defect. They call for different
    fixes and the router should be able to report which one it hit.

    Attributes:
        media_type: Raw ``media_type`` cell, preserved even when unrecognised.
        media_id: Raw ``media_id`` cell.
        is_registered: Whether ``media_id`` was found in a media registry.
        file_exists: Whether the registered path is a readable file.
        derived_text: Transcript or caption recovered from the attachment.
            Empty until a provider is installed.
        derived_from: Name of the provider that produced ``derived_text``.
        derived_confidence: That provider's confidence, in ``[0, 1]``.
        derived_language: BCP-47 tag when the provider reports one.
    """

    media_type: str | None = None
    media_id: str | None = None
    is_registered: bool = False
    file_exists: bool = False
    derived_text: str = ""
    derived_from: str = "none"
    derived_confidence: float = 0.0
    derived_language: str | None = None

    #: The "no attachment" value. Shared, so text-only messages allocate nothing.
    NONE: ClassVar[MediaFeatures]

    @property
    def has_attachment(self) -> bool:
        """Whether the message carries an attachment at all."""
        return self.media_id is not None

    @property
    def has_derived_text(self) -> bool:
        """Whether a provider recovered usable text from the attachment."""
        return bool(self.derived_text.strip())

    @property
    def is_unreadable(self) -> bool:
        """Whether an attachment is present but could not be located on disk.

        True for both dataset and packaging defects. Routing treats such a
        message as media-with-no-content, which is also how it treats an
        attachment no installed model can read.
        """
        return self.has_attachment and not self.file_exists


MediaFeatures.NONE = MediaFeatures()


@dataclass(frozen=True, slots=True)
class MessageFeatures:
    """The complete Phase 2 feature record for one incoming message.

    This is the object Phase 3 (personalisation), Phase 4 (routing) and
    Phase 5 (evidence retrieval) consume. It is self-contained: no repository
    call is needed to read any field.

    Attributes:
        message_id: Identifier of the incoming message.
        user_id: Recipient of the message.
        sender_user_id: Sending user, when the sender is a person.
        group_id: Originating group, when applicable.
        business_id: Sending business, when applicable.
        created_at: When the message arrived.
        text: Body-derived features.
        context: Conversation and sender features.
        history: Recipient's historical behaviour.
        keywords: Lexical matches.
        media: Attachment provenance and any text recovered from it. Defaults
            to :data:`MediaFeatures.NONE`, so every construction predating the
            multimodal seam keeps working unchanged.
    """

    message_id: str
    user_id: str
    sender_user_id: str | None
    group_id: str | None
    business_id: str | None
    created_at: dt.datetime
    text: TextFeatures
    context: ContextFeatures
    history: HistoricalFeatures
    keywords: KeywordFeatures
    media: "MediaFeatures" = field(default_factory=lambda: MediaFeatures.NONE)

    # -- Frequently used shortcuts, so callers need not chain attributes -- #

    @property
    def conversation_type(self) -> str:
        """``personal``, ``group`` or ``business``."""
        return self.context.conversation_type

    @property
    def media_type(self) -> str | None:
        """``image``, ``voice`` or ``None``."""
        return self.context.media_type

    @property
    def has_media(self) -> bool:
        """Whether an attachment is present."""
        return self.context.has_media

    @property
    def contains_attachment(self) -> bool:
        """Alias of :attr:`has_media`, matching the Phase 2 brief's wording."""
        return self.context.has_media

    @property
    def forwarded_count(self) -> int:
        """Platform forwarding signal."""
        return self.context.forwarded_count

    @property
    def is_empty_text(self) -> bool:
        """Whether the message carries no text (typical of voice notes)."""
        return self.text.is_empty

    @property
    def matched_keywords(self) -> tuple[str, ...]:
        """Every matched keyword across every category."""
        return self.keywords.all_keywords

    @property
    def has_derived_text(self) -> bool:
        """Whether any of :attr:`text` was recovered from an attachment.

        When true, :attr:`text` and :attr:`keywords` cover the transcript or
        caption as well as the typed body. Consumers that must weigh recovered
        text differently from typed text should read this, not guess from
        :attr:`has_media`.
        """
        return self.media.has_derived_text

    @property
    def has_unreadable_media(self) -> bool:
        """Whether an attachment is present that could not be read at all."""
        return self.media.is_unreadable

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-friendly view for logging and debugging.

        Nested blocks are prefixed with their name (``text_length``,
        ``context_is_group``), so the result is unambiguous and stable.
        """
        flat: dict[str, Any] = {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "sender_user_id": self.sender_user_id,
            "group_id": self.group_id,
            "business_id": self.business_id,
            "created_at": self.created_at.isoformat(),
        }
        for prefix, block in (
            ("text", self.text),
            ("context", self.context),
            ("history", self.history),
            ("media", self.media),
        ):
            for spec in fields(block):
                flat[f"{prefix}_{spec.name}"] = getattr(block, spec.name)
        flat["keywords"] = {
            category.value: list(words) for category, words in self.keywords.matches.items()
        }
        return flat

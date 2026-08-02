"""What an attachment is, and what a model managed to read out of it.

Two records, deliberately separate:

* :class:`MediaAttachment` - the *file*. Resolved from ``media_id`` through the
  ``images`` / ``voice_notes`` registries. Knowing this needs the repository
  but no model.
* :class:`MediaContent` - the *content*. Text recovered from that file by OCR,
  speech-to-text, or anything else. Producing this needs a model but no
  repository.

Keeping them apart is what makes the seam work: the resolver half is
implemented and tested today, and a provider can be dropped into the other half
later without touching a single caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Self

__all__ = ["MediaModality", "MediaAttachment", "MediaContent"]


class MediaModality(StrEnum):
    """The kinds of attachment the dataset carries.

    Values match ``messages.media_type`` exactly, so a raw cell converts with
    :meth:`from_value` and nothing else needs a lookup table.
    """

    IMAGE = "image"
    VOICE = "voice"

    @classmethod
    def from_value(cls, value: object) -> Self | None:
        """Return the modality named by ``value``, or ``None`` if unrecognised.

        Unrecognised is a real case: a future dataset may carry ``video``, and
        an unknown modality must degrade to "no derived content" rather than
        raise.
        """
        if isinstance(value, cls):
            return value
        text = str(value).strip().casefold() if value is not None else ""
        try:
            return cls(text)
        except ValueError:
            return None


@dataclass(frozen=True, slots=True)
class MediaAttachment:
    """One attachment, located on disk where possible.

    Attributes:
        media_id: The ``media_id`` cell from the message.
        modality: Which kind of media it is, or ``None`` when the
            ``media_type`` cell held something unrecognised.
        file_path: Absolute path to the binary, or ``None`` when the id is not
            in the registry.
        exists: Whether that path is actually a readable file. A registry entry
            can outlive the file it points at.
    """

    media_id: str
    modality: MediaModality | None
    file_path: Path | None = None
    exists: bool = False

    @property
    def is_registered(self) -> bool:
        """Whether ``media_id`` was found in the media registry."""
        return self.file_path is not None

    @property
    def is_readable(self) -> bool:
        """Whether a provider could actually open this attachment."""
        return self.exists and self.file_path is not None


@dataclass(frozen=True, slots=True)
class MediaContent:
    """Text recovered from an attachment, with its provenance.

    Attributes:
        text: The recovered transcript or caption. Empty when nothing was
            recovered, which is the default until a provider is installed.
        provider: Name of whatever produced it, carried through to diagnostics
            so a decision can always be traced back to its source.
        confidence: The provider's own confidence in ``text``, in ``[0, 1]``.
            Routing may discount weakly-recovered text; it must never be
            treated as being as reliable as text the sender actually typed.
        language: BCP-47 tag when the provider reports one.
    """

    text: str = ""
    provider: str = "none"
    confidence: float = 0.0
    language: str | None = None

    #: Shared "nothing was recovered" value, so the common path allocates nothing.
    EMPTY: ClassVar[MediaContent]

    @property
    def has_text(self) -> bool:
        """Whether any usable text was recovered."""
        return bool(self.text.strip())

    def to_dict(self) -> dict[str, object]:
        """Return a flat, JSON-friendly view."""
        return {
            "text": self.text,
            "provider": self.provider,
            "confidence": self.confidence,
            "language": self.language,
        }


MediaContent.EMPTY = MediaContent()

"""Multimodal seam: attachments in, recovered text out.

The router treats a voice transcript exactly like a typed body. This package is
the boundary where the model that produces one plugs in.

    message.media_id
      -> MediaResolver          registry lookup, then filesystem check
      -> MediaUnderstanding     speech-to-text (Whisper) / OCR (not implemented)
      -> TranscriptCache        keyed by media_id, persists between runs
      -> MediaFeatures          provenance + recovered text
      -> FeatureExtractor       recovered text joins the typed body

**Voice is implemented.** :class:`~src.media.whisper.WhisperTranscriber` wraps
``faster-whisper`` and is enabled by default through
:func:`~src.media.understanding.default_understanding`, which falls back to
cached transcripts and then to nothing at all, so the pipeline runs identically
whether or not the dependency is present.

**Images are not.** An image contributes no derived text. The seam is
identical, so adding OCR is the same one-class change Whisper was; see
:mod:`src.media.understanding`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.media.cache import CachingUnderstanding, TranscriptCache
from src.media.content import MediaAttachment, MediaContent, MediaModality
from src.media.understanding import (
    CompositeUnderstanding,
    MediaUnderstanding,
    NullUnderstanding,
    SafeUnderstanding,
    default_understanding,
)
from src.media.whisper import WhisperTranscriber, is_whisper_available

if TYPE_CHECKING:
    from src.media.resolver import MediaResolver


def __getattr__(name: str) -> Any:
    """Resolve :class:`~src.media.resolver.MediaResolver` on first access.

    The resolver is the one member of this package that reaches into
    :mod:`src.features`, whose extractor imports the resolver straight back.
    Importing it eagerly here therefore makes ``import src.media`` fail or
    succeed depending on which package the process happened to touch first -
    a real trap, and one that only bites the caller who imports this package
    directly.

    Deferring that single name keeps ``from src.media import MediaResolver``
    working exactly as before while making every import order safe. Everything
    else in this package is a leaf and is imported normally above.
    """
    if name == "MediaResolver":
        from src.media.resolver import MediaResolver

        return MediaResolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CachingUnderstanding",
    "CompositeUnderstanding",
    "MediaAttachment",
    "MediaContent",
    "MediaModality",
    "MediaResolver",
    "MediaUnderstanding",
    "NullUnderstanding",
    "SafeUnderstanding",
    "TranscriptCache",
    "WhisperTranscriber",
    "default_understanding",
    "is_whisper_available",
]

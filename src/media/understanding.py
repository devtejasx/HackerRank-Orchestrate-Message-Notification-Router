"""The plug point for OCR and speech-to-text.

This module is the *abstraction*: one narrow interface,
:class:`MediaUnderstanding`, that the pipeline calls on every media message.
It deliberately knows nothing about any particular model.

Speech-to-text is implemented against it in :mod:`src.media.whisper` and is
enabled by default via :func:`default_understanding`. Image OCR is not
implemented; an image still contributes no derived text.

A provider is two methods and no framework:

.. code-block:: python

    class TesseractOCR:
        name = "tesseract"

        def supports(self, modality: MediaModality) -> bool:
            return modality is MediaModality.IMAGE

        def understand(self, attachment: MediaAttachment) -> MediaContent:
            return MediaContent(
                text=pytesseract.image_to_string(attachment.file_path),
                provider=self.name,
                confidence=0.6,
            )

    pipeline = RoutingPipeline.load(
        understanding=CompositeUnderstanding(WhisperTranscriber(), TesseractOCR()),
    )

That is the whole integration. No feature, rule, classifier or writer changes,
because recovered text flows into text and keyword extraction the same way a
typed body does - see :class:`~src.media.resolver.MediaResolver`. Adding
Whisper required exactly this and nothing more.

Two guarantees make that safe to rely on:

* **Unreadable attachments never reach a provider.** The resolver checks the
  registry and the filesystem first.
* **A provider that raises cannot fail the run.** :class:`SafeUnderstanding`
  wraps every provider, so a model that crashes on one file costs that file's
  derived text and nothing else. The contract requires a prediction for every
  message, including the ones a model choked on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src import config
from src.media.content import MediaAttachment, MediaContent, MediaModality

__all__ = [
    "MediaUnderstanding",
    "NullUnderstanding",
    "CompositeUnderstanding",
    "SafeUnderstanding",
    "default_understanding",
]

_LOGGER = config.get_logger("media")


@runtime_checkable
class MediaUnderstanding(Protocol):
    """Recovers text from an attachment.

    Implementations must be safe to reuse across the whole dataset: the
    pipeline constructs one and calls it per message, so an expensive model
    should be loaded once in ``__init__`` rather than per call.
    """

    #: Short identifier recorded on every :class:`MediaContent` this produces.
    name: str

    def supports(self, modality: MediaModality) -> bool:
        """Whether this provider handles ``modality`` at all."""
        ...

    def understand(self, attachment: MediaAttachment) -> MediaContent:
        """Recover text from one readable attachment.

        Called only for attachments where :attr:`MediaAttachment.is_readable`
        is true and :meth:`supports` returned true, so an implementation may
        assume the file is present.
        """
        ...


class NullUnderstanding:
    """The default: recovers nothing from anything.

    Not a placeholder to be deleted - it is the correct provider for a run with
    no model installed, and it keeps the media path exercised (and tested) even
    when no OCR or speech-to-text is available.
    """

    name = "none"

    def supports(self, modality: MediaModality) -> bool:  # noqa: ARG002
        """Always ``False``; nothing is understood without a model."""
        return False

    def understand(self, attachment: MediaAttachment) -> MediaContent:  # noqa: ARG002
        """Always the empty content."""
        return MediaContent.EMPTY


class CompositeUnderstanding:
    """Dispatches to the first delegate that supports the modality.

    The natural shape once there is more than one model: a speech-to-text
    provider and an OCR provider handle disjoint modalities and neither needs
    to know the other exists.

    Args:
        *providers: Delegates, tried in order.
    """

    def __init__(self, *providers: MediaUnderstanding) -> None:
        self._providers = providers
        self.name = "+".join(p.name for p in providers) or NullUnderstanding.name

    @property
    def providers(self) -> tuple[MediaUnderstanding, ...]:
        """The delegates, in dispatch order."""
        return self._providers

    def supports(self, modality: MediaModality) -> bool:
        """Whether any delegate handles ``modality``."""
        return any(p.supports(modality) for p in self._providers)

    def understand(self, attachment: MediaAttachment) -> MediaContent:
        """Return the first non-empty result from a supporting delegate."""
        if attachment.modality is None:
            return MediaContent.EMPTY
        for provider in self._providers:
            if not provider.supports(attachment.modality):
                continue
            content = provider.understand(attachment)
            if content.has_text:
                return content
        return MediaContent.EMPTY


class SafeUnderstanding:
    """Wraps a provider so a model failure degrades instead of aborting.

    A speech-to-text model that raises on a truncated file, or an OCR binary
    that is not installed, must cost one message its derived text - not cost
    the run its output.csv. Every failure is logged once with the media id, so
    a silent degradation is still a visible one.

    Args:
        inner: The provider to guard.
    """

    def __init__(self, inner: MediaUnderstanding) -> None:
        self._inner = inner
        self.name = inner.name

    @property
    def inner(self) -> MediaUnderstanding:
        """The wrapped provider."""
        return self._inner

    def supports(self, modality: MediaModality) -> bool:
        """Delegate, treating a failure as "not supported"."""
        try:
            return self._inner.supports(modality)
        except Exception:  # noqa: BLE001 - a broken provider must not end the run
            _LOGGER.exception("Media provider %r failed on supports()", self.name)
            return False

    def understand(self, attachment: MediaAttachment) -> MediaContent:
        """Delegate, treating a failure as "nothing recovered"."""
        try:
            return self._inner.understand(attachment)
        except Exception:  # noqa: BLE001 - see class docstring
            _LOGGER.exception(
                "Media provider %r failed on %s; continuing without derived text",
                self.name,
                attachment.media_id,
            )
            return MediaContent.EMPTY


def default_understanding(
    *, transcribe: bool = True, model_size: str | None = None
) -> MediaUnderstanding:
    """Return the provider used when a caller supplies none.

    Resolves to the best transcription available in this environment, in
    descending order:

    1. **Cached transcripts plus a live model** - the normal case once
       ``faster-whisper`` is installed. Voice notes already transcribed are
       served from ``transcripts.json``; new ones are transcribed and recorded.
    2. **Cached transcripts alone** - ``faster-whisper`` is absent, but the
       committed cache still covers this dataset's voice notes, so routing
       keeps the benefit of transcription without the dependency.
    3. **Nothing** - no model and no cache. Voice notes route on sender context
       alone, exactly as they did before Whisper was integrated, and the
       confidence column discounts them for it.

    Every step degrades; none raises. Image OCR is still unimplemented at every
    level, so an image contributes no derived text regardless.

    Args:
        transcribe: Set ``False`` to force level 3 and skip audio entirely.
        model_size: Whisper weights to load. Defaults to
            :data:`~src.media.whisper.DEFAULT_MODEL_SIZE`.

    Returns:
        A provider that is always safe to call.
    """
    # Imported here rather than at module scope: this module is the abstraction
    # and must not depend on any particular implementation of it.
    from src.media.cache import CachingUnderstanding, TranscriptCache
    from src.media.whisper import DEFAULT_MODEL_SIZE, WhisperTranscriber, is_whisper_available

    if not transcribe:
        return NullUnderstanding()

    cache = TranscriptCache()
    if is_whisper_available():
        transcriber = WhisperTranscriber(model_size or DEFAULT_MODEL_SIZE)
        _LOGGER.debug("Transcription enabled via %s", transcriber.name)
        return CachingUnderstanding(transcriber, cache)

    if len(cache):
        _LOGGER.info(
            "faster-whisper is not installed; using %d cached transcript(s) from %s",
            len(cache),
            cache.path.name,
        )
        return CachingUnderstanding(NullUnderstanding(), cache)

    _LOGGER.info(
        "No transcription available (faster-whisper not installed, no cached "
        "transcripts); voice notes will route on context alone"
    )
    return NullUnderstanding()

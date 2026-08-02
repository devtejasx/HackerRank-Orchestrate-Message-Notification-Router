"""Transcripts, remembered between runs.

:class:`~src.media.resolver.MediaResolver` already avoids transcribing the same
attachment twice *within* a run. This adds the other half: a JSON file on disk
keyed by ``media_id``, so a second run of ``python main.py`` costs nothing at
all rather than several seconds per voice note.

:class:`CachingUnderstanding` is itself a
:class:`~src.media.understanding.MediaUnderstanding`, so it composes with the
existing abstraction rather than sitting beside it - the resolver cannot tell
the difference between a cached transcriber and a bare one.

Entries record a fingerprint of the audio alongside the transcript. Reusing a
transcript for a file that has since changed would be worse than not caching at
all, because the error would be silent and would survive every later run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from src import config
from src.media.content import MediaAttachment, MediaContent, MediaModality
from src.media.understanding import MediaUnderstanding

__all__ = ["TranscriptCache", "CachingUnderstanding", "DEFAULT_CACHE_PATH"]

_LOGGER = config.get_logger("media.cache")

#: Where transcripts are kept when no path is given.
#:
#: Inside the repository rather than a temp directory, deliberately: it makes
#: the cache a reviewable artefact that travels with the project, so a grader
#: without ``faster-whisper`` installed still gets the transcripts this system
#: was built against. Delete it, or pass ``--refresh-transcripts``, to rebuild
#: from the audio.
DEFAULT_CACHE_PATH: Final[Path] = config.PROJECT_ROOT / "transcripts.json"

#: Bytes of audio hashed for the fingerprint. Hashing the head of the file
#: alongside its size distinguishes any realistic replacement without reading
#: megabytes back off disk for every lookup.
_FINGERPRINT_BYTES: Final[int] = 65_536

#: Schema marker, so a future change to the entry shape can invalidate cleanly
#: instead of being misread as valid data.
_CACHE_VERSION: Final[int] = 1


def fingerprint(path: Path) -> str:
    """Return a short content fingerprint for ``path``.

    Size plus a digest of the first :data:`_FINGERPRINT_BYTES`. Deliberately
    not the modification time, which changes on every clone and would make the
    cache useless to anyone but its author.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            head = handle.read(_FINGERPRINT_BYTES)
    except OSError as exc:
        _LOGGER.debug("Could not fingerprint %s: %s", path, exc)
        return ""
    return f"{size}:{hashlib.sha256(head).hexdigest()[:16]}"


class TranscriptCache:
    """A ``media_id -> transcript`` store backed by one JSON file.

    Args:
        path: File to read and write. Defaults to :data:`DEFAULT_CACHE_PATH`.

    Example:
        >>> cache = TranscriptCache()                     # doctest: +SKIP
        >>> cache.get("vn_001", "12345:ab...")            # doctest: +SKIP
        MediaContent(text='Had dinner, call when free...', ...)
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_CACHE_PATH
        self._entries: dict[str, dict[str, object]] = {}
        self._dirty = False
        self._loaded = False

    @property
    def path(self) -> Path:
        """The file this cache reads and writes."""
        return self._path

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._entries)

    def get(self, media_id: str, audio_fingerprint: str) -> MediaContent | None:
        """Return the cached transcript for ``media_id``, if it is still valid.

        Args:
            media_id: The attachment's id.
            audio_fingerprint: Fingerprint of the file as it is now. An entry
                recorded against a different fingerprint is ignored, so
                replacing the audio silently re-transcribes it.

        Returns:
            The transcript, or ``None`` on a miss or a stale entry.
        """
        self._ensure_loaded()
        entry = self._entries.get(media_id)
        if entry is None:
            return None
        stored = entry.get("fingerprint")
        if stored and audio_fingerprint and stored != audio_fingerprint:
            _LOGGER.info("Audio for %s changed; discarding its cached transcript", media_id)
            return None
        return MediaContent(
            text=str(entry.get("text", "")),
            provider=str(entry.get("provider", "cache")),
            confidence=float(entry.get("confidence", 0.0) or 0.0),
            language=entry.get("language") or None,  # type: ignore[arg-type]
        )

    def put(self, media_id: str, content: MediaContent, audio_fingerprint: str) -> None:
        """Record a transcript. Held in memory until :meth:`save`."""
        self._ensure_loaded()
        self._entries[media_id] = {
            "text": content.text,
            "provider": content.provider,
            "confidence": content.confidence,
            "language": content.language,
            "fingerprint": audio_fingerprint,
        }
        self._dirty = True

    def save(self) -> bool:
        """Write the cache to disk if anything changed.

        Entries are sorted by key so the committed file has a stable diff.

        Returns:
            Whether a write actually happened.
        """
        if not self._dirty:
            return False
        payload = {
            "version": _CACHE_VERSION,
            "transcripts": dict(sorted(self._entries.items())),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_name(f"{self._path.name}.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self._path)
        except OSError as exc:
            # A read-only checkout must not fail a run; the transcripts are
            # already in memory and this run is unaffected.
            _LOGGER.warning("Could not write the transcript cache at %s: %s", self._path, exc)
            return False

        _LOGGER.info("Saved %d transcript(s) to %s", len(self._entries), self._path)
        self._dirty = False
        return True

    def clear(self) -> None:
        """Forget every entry, forcing re-transcription on the next run."""
        self._ensure_loaded()
        if self._entries:
            self._entries.clear()
            self._dirty = True

    def _ensure_loaded(self) -> None:
        """Read the file once, tolerating absence and corruption alike."""
        if self._loaded:
            return
        self._loaded = True
        if not self._path.is_file():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOGGER.warning("Ignoring an unreadable transcript cache at %s: %s", self._path, exc)
            return

        if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
            _LOGGER.warning("Ignoring a transcript cache written by another version")
            return
        transcripts = payload.get("transcripts")
        if isinstance(transcripts, dict):
            self._entries = {
                key: value
                for key, value in transcripts.items()
                if isinstance(value, dict)
            }
            _LOGGER.debug("Loaded %d cached transcript(s)", len(self._entries))


class CachingUnderstanding:
    """Serves transcripts from a cache, delegating only on a miss.

    The cache is consulted *before* the delegate is asked whether it supports
    the modality, which is what lets a checkout with no ``faster-whisper``
    installed still route on transcripts recorded earlier. On a miss with no
    usable delegate, the result is simply an empty transcript - the same
    graceful degradation as having no provider at all.

    Args:
        inner: The provider that does the real work.
        cache: Where to look first. A default one is built when omitted.
        save_on_write: Whether to persist after each new transcript. Off for
            tests that must not touch the shared cache file.
    """

    def __init__(
        self,
        inner: MediaUnderstanding,
        cache: TranscriptCache | None = None,
        *,
        save_on_write: bool = True,
    ) -> None:
        self._inner = inner
        self._cache = cache if cache is not None else TranscriptCache()
        self._save_on_write = save_on_write
        # Named for what it can actually do, so the CLI's one-line media
        # summary distinguishes "transcribing" from "replaying transcripts
        # somebody else produced" - a difference that matters when reading a
        # run's output and wondering where the text came from.
        self.name = (
            f"{inner.name}+cache" if inner.name != "none" else "transcript-cache"
        )

    @property
    def inner(self) -> MediaUnderstanding:
        """The wrapped provider."""
        return self._inner

    @property
    def cache(self) -> TranscriptCache:
        """The transcript store in use."""
        return self._cache

    def supports(self, modality: MediaModality) -> bool:
        """Voice is supported whenever a cache exists, model or not.

        Other modalities defer to the delegate, so wrapping an OCR provider in
        a transcript cache does not accidentally claim to handle images it
        cannot read.
        """
        if modality is MediaModality.VOICE and len(self._cache):
            return True
        return self._inner.supports(modality)

    def understand(self, attachment: MediaAttachment) -> MediaContent:
        """Return the cached transcript, or produce and record one."""
        audio_fingerprint = (
            fingerprint(attachment.file_path) if attachment.file_path else ""
        )
        cached = self._cache.get(attachment.media_id, audio_fingerprint)
        if cached is not None:
            return cached

        if attachment.modality is None or not self._inner.supports(attachment.modality):
            return MediaContent.EMPTY

        content = self._inner.understand(attachment)
        if not content.has_text:
            # Not cached. A failure is usually environmental - a model that was
            # not installed yet - and caching it would make that permanent.
            return content

        self._cache.put(attachment.media_id, content, audio_fingerprint)
        if self._save_on_write:
            self._cache.save()
        return content

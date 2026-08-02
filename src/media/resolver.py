"""Turns a message's attachment into a feature block.

The resolver does the half of multimodal handling that needs no model:

1. Read ``media_type`` / ``media_id`` off the message.
2. Look the id up in the ``images`` / ``voice_notes`` registry.
3. Check the referenced file is actually on disk.
4. Hand a readable attachment to the configured
   :class:`~src.media.understanding.MediaUnderstanding` provider.
5. Record what came back, including that nothing did.

Steps 1-3 and 5 are live today. Step 4 is a call into the default provider,
which returns nothing, so behaviour with no model installed is identical to
having no media path at all - but the wiring, the feature block, the
diagnostics and the tests all already exist.

Results are memoised by media id. The dataset reuses attachments across
messages (``img_008`` appears three times), and a real transcription model is
far too expensive to run twice on the same file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src import config
from src.data.models import MessageRecord
from src.data.repository import DataRepository
from src.media.content import MediaAttachment, MediaContent, MediaModality
from src.media.understanding import (
    MediaUnderstanding,
    SafeUnderstanding,
    default_understanding,
)
from src.utils.helpers import resolve_dataset_path

if TYPE_CHECKING:
    from src.features.feature_models import MediaFeatures

__all__ = ["MediaResolver"]

_LOGGER = config.get_logger("media.resolver")


def _media_features() -> type[MediaFeatures]:
    """Return the ``MediaFeatures`` class, imported on first use.

    This module and :mod:`src.features.extractor` genuinely need each other at
    runtime: the extractor constructs a resolver, and the resolver returns the
    extractor's feature block. Importing that block at module scope closes the
    loop, so whether ``import src.media`` works depends on which package the
    process touched first.

    Deferring the import to first call breaks the loop from the media side,
    which leaves feature extraction untouched. Module lookup after the first
    call is a dictionary hit, so the per-message cost is nil.
    """
    from src.features.feature_models import MediaFeatures

    return MediaFeatures


class MediaResolver:
    """Resolves and, where possible, reads the attachment on a message.

    Args:
        repo: Loaded repository, used for the media registries.
        understanding: Provider that recovers text. Defaults to
            :func:`~src.media.understanding.default_understanding`, which
            recovers nothing. Always wrapped in
            :class:`~src.media.understanding.SafeUnderstanding`.

    Example:
        >>> resolver = MediaResolver(repo)                  # doctest: +SKIP
        >>> resolver.resolve(message).has_derived_text      # doctest: +SKIP
        False
    """

    def __init__(
        self,
        repo: DataRepository,
        understanding: MediaUnderstanding | None = None,
    ) -> None:
        self._repo = repo
        inner = understanding if understanding is not None else default_understanding()
        self._understanding = (
            inner if isinstance(inner, SafeUnderstanding) else SafeUnderstanding(inner)
        )
        self._cache: dict[str, MediaContent] = {}

    @property
    def understanding(self) -> MediaUnderstanding:
        """The provider in use, for inspection and diagnostics."""
        return self._understanding

    def resolve(self, message: MessageRecord) -> MediaFeatures:
        """Build the media feature block for one message.

        Args:
            message: The message being analysed.

        Returns:
            :data:`MediaFeatures.NONE` for a message with no attachment, so
            the common text-only path allocates nothing.
        """
        features = _media_features()
        if message.media_id is None:
            return features.NONE

        attachment = self.attachment(message.media_id, message.media_type)
        content = self._content_for(attachment)
        return features(
            media_type=message.media_type,
            media_id=attachment.media_id,
            is_registered=attachment.is_registered,
            file_exists=attachment.exists,
            derived_text=content.text,
            derived_from=content.provider,
            derived_confidence=content.confidence,
            derived_language=content.language,
        )

    def attachment(self, media_id: str, media_type: object) -> MediaAttachment:
        """Locate one attachment on disk.

        Args:
            media_id: Value of the message's ``media_id`` cell.
            media_type: Value of its ``media_type`` cell.

        Returns:
            An attachment whose ``file_path`` is ``None`` when the id is absent
            from the registry, and whose ``exists`` is ``False`` when the
            registry points at a file that is not there. Both are recorded
            rather than raised: an unresolvable attachment is a routable
            message, just one with less to go on.
        """
        modality = MediaModality.from_value(media_type)
        record = self._repo.get_media(modality.value, media_id) if modality else None
        if record is None:
            if modality is not None:
                _LOGGER.debug("Unregistered %s attachment: %s", modality, media_id)
            return MediaAttachment(media_id=media_id, modality=modality)

        path = resolve_dataset_path(record.file_path, self._repo.loader.dataset_dir)
        return MediaAttachment(
            media_id=media_id,
            modality=modality,
            file_path=path,
            exists=path.is_file(),
        )

    def _content_for(self, attachment: MediaAttachment) -> MediaContent:
        """Recover text from ``attachment``, memoised by media id.

        Skips the provider entirely when there is nothing to read, so a model
        is never handed a path that does not exist.
        """
        if not attachment.is_readable or attachment.modality is None:
            return MediaContent.EMPTY

        cached = self._cache.get(attachment.media_id)
        if cached is not None:
            return cached

        content = (
            self._understanding.understand(attachment)
            if self._understanding.supports(attachment.modality)
            else MediaContent.EMPTY
        )
        self._cache[attachment.media_id] = content
        return content

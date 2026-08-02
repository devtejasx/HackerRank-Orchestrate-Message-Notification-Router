"""Multimodal seam: attachments in, recovered text out.

The router already treats an image caption or a voice transcript exactly like a
typed body - it just has no model to produce one. This package is the boundary
where such a model plugs in.

    message.media_id
      -> MediaResolver          registry lookup, then filesystem check
      -> MediaUnderstanding     OCR / speech-to-text  (NullUnderstanding today)
      -> MediaFeatures          provenance + recovered text
      -> FeatureExtractor       recovered text joins the typed body

Everything except the middle step is implemented and tested. Installing
Tesseract or Whisper means writing one class with two methods and passing it to
:meth:`~src.routing.pipeline.RoutingPipeline.load`; see
:mod:`src.media.understanding` for a worked example and the guarantees the
pipeline makes to a provider.
"""

from __future__ import annotations

from src.media.content import MediaAttachment, MediaContent, MediaModality
from src.media.resolver import MediaResolver
from src.media.understanding import (
    CompositeUnderstanding,
    MediaUnderstanding,
    NullUnderstanding,
    SafeUnderstanding,
    default_understanding,
)

__all__ = [
    "CompositeUnderstanding",
    "MediaAttachment",
    "MediaContent",
    "MediaModality",
    "MediaResolver",
    "MediaUnderstanding",
    "NullUnderstanding",
    "SafeUnderstanding",
    "default_understanding",
]

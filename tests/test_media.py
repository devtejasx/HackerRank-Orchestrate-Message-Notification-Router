"""The multimodal seam.

Two things are under test here, and the second matters more than the first.

1. The resolver half - registry lookup, filesystem check, caching, and the
   failure modes (unregistered id, missing file, unknown modality).
2. The *claim* that installing OCR or speech-to-text needs no other change.
   :class:`FakeTranscriber` is written exactly as a real provider would be:
   two methods, no framework, no hook registration. If routing a voice note
   changes when it is passed to ``RoutingPipeline.load(understanding=...)``,
   then Whisper will change it too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.models import Message
from src.data.repository import DataRepository
from src.features.feature_models import MediaFeatures
from src.media import (
    CompositeUnderstanding,
    MediaAttachment,
    MediaContent,
    MediaModality,
    MediaResolver,
    NullUnderstanding,
    SafeUnderstanding,
)
from src.pipeline import MessagePipeline
from src.routing.pipeline import RoutingPipeline


# --------------------------------------------------------------------------- #
# Providers a real integration would look like
# --------------------------------------------------------------------------- #


class FakeTranscriber:
    """Stands in for Whisper. Deliberately as small as the real thing would be."""

    name = "fake-whisper"

    def __init__(self, text: str = "please pay the rent invoice by tonight") -> None:
        self._text = text
        self.calls: list[str] = []

    def supports(self, modality: MediaModality) -> bool:
        return modality is MediaModality.VOICE

    def understand(self, attachment: MediaAttachment) -> MediaContent:
        self.calls.append(attachment.media_id)
        return MediaContent(
            text=self._text, provider=self.name, confidence=0.8, language="en"
        )


class FakeOcr:
    """Stands in for Tesseract."""

    name = "fake-ocr"

    def supports(self, modality: MediaModality) -> bool:
        return modality is MediaModality.IMAGE

    def understand(self, attachment: MediaAttachment) -> MediaContent:  # noqa: ARG002
        return MediaContent(text="LIMITED TIME OFFER", provider=self.name, confidence=0.6)


class ExplodingProvider:
    """A model that crashes. The run must survive it."""

    name = "exploding"

    def supports(self, modality: MediaModality) -> bool:  # noqa: ARG002
        return True

    def understand(self, attachment: MediaAttachment) -> MediaContent:  # noqa: ARG002
        raise RuntimeError("model failed to load")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def voice_message(repo: DataRepository) -> Message:
    """An incoming voice note whose file is actually on disk."""
    for message in repo.get_messages():
        if message.media_type == "voice" and repo.get_media_path(message) is not None:
            return message
    pytest.skip("dataset has no resolvable voice note")


@pytest.fixture(scope="module")
def image_message(repo: DataRepository) -> Message:
    """An incoming image message whose file is actually on disk."""
    for message in repo.get_messages():
        if message.media_type == "image" and repo.get_media_path(message) is not None:
            return message
    pytest.skip("dataset has no resolvable image")


def _with_media(message: Message, **overrides: object) -> Message:
    """Return a copy of ``message`` with the media cells replaced."""
    return Message(**{**message.to_dict(), **overrides})


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


class TestMediaModality:
    def test_recognises_dataset_values(self) -> None:
        assert MediaModality.from_value("image") is MediaModality.IMAGE
        assert MediaModality.from_value("voice") is MediaModality.VOICE

    @pytest.mark.parametrize("value", [None, "", "video", "IMAGE ", 7])
    def test_unknown_values_never_raise(self, value: object) -> None:
        # A future dataset may add a modality; that must degrade, not crash.
        assert MediaModality.from_value(value) in (None, MediaModality.IMAGE)


class TestResolution:
    def test_message_without_media_resolves_to_the_shared_empty_block(
        self, repo: DataRepository
    ) -> None:
        text_only = next(m for m in repo.get_messages() if m.media_id is None)
        assert MediaResolver(repo).resolve(text_only) is MediaFeatures.NONE

    def test_registered_attachment_is_located_on_disk(
        self, repo: DataRepository, image_message: Message
    ) -> None:
        media = MediaResolver(repo).resolve(image_message)
        assert media.is_registered is True
        assert media.file_exists is True
        assert media.is_unreadable is False

    def test_unregistered_id_is_recorded_not_raised(
        self, repo: DataRepository, image_message: Message
    ) -> None:
        media = MediaResolver(repo).resolve(_with_media(image_message, media_id="img_zzz"))
        assert media.is_registered is False
        assert media.is_unreadable is True
        assert media.has_derived_text is False

    def test_unknown_modality_is_recorded_not_raised(
        self, repo: DataRepository, image_message: Message
    ) -> None:
        media = MediaResolver(repo).resolve(_with_media(image_message, media_type="video"))
        assert media.media_type == "video"
        assert media.is_registered is False

    def test_registry_entry_pointing_at_a_missing_file(
        self, repo: DataRepository, image_message: Message, tmp_path: Path
    ) -> None:
        # A registry that outlives its files is a packaging defect, not a crash.
        stripped = DataRepository.load(_dataset_without_media(repo, tmp_path))
        message = next(
            m for m in stripped.get_messages() if m.message_id == image_message.message_id
        )
        media = MediaResolver(stripped).resolve(message)
        assert media.is_registered is True
        assert media.file_exists is False
        assert media.is_unreadable is True


def _dataset_without_media(repo: DataRepository, tmp_path: Path) -> Path:
    """Copy the dataset but leave the binaries behind."""
    import shutil

    destination = tmp_path / "no_media"
    shutil.copytree(
        repo.loader.dataset_dir,
        destination,
        ignore=shutil.ignore_patterns("media"),
    )
    return destination


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


class TestProviders:
    def test_null_provider_recovers_nothing(
        self, repo: DataRepository, voice_message: Message
    ) -> None:
        media = MediaResolver(repo, NullUnderstanding()).resolve(voice_message)
        assert media.has_derived_text is False
        assert media.derived_from == "none"
        assert NullUnderstanding().supports(MediaModality.VOICE) is False

    def test_the_shipped_default_does_transcribe(
        self, repo: DataRepository, voice_message: Message
    ) -> None:
        """What `python main.py` actually does with a voice note.

        Passes whether or not faster-whisper is installed here, because the
        committed transcript cache covers this dataset either way - which is
        the point of that cache.
        """
        media = MediaResolver(repo).resolve(voice_message)
        assert media.has_derived_text is True
        assert media.derived_language in (None, "en")

    def test_provider_output_lands_on_the_feature_block(
        self, repo: DataRepository, voice_message: Message
    ) -> None:
        media = MediaResolver(repo, FakeTranscriber()).resolve(voice_message)
        assert media.has_derived_text is True
        assert media.derived_from == "fake-whisper"
        assert media.derived_confidence == pytest.approx(0.8)
        assert media.derived_language == "en"

    def test_provider_is_not_called_for_modalities_it_declines(
        self, repo: DataRepository, image_message: Message
    ) -> None:
        transcriber = FakeTranscriber()
        MediaResolver(repo, transcriber).resolve(image_message)
        assert transcriber.calls == []

    def test_provider_is_not_called_for_unreadable_attachments(
        self, repo: DataRepository, voice_message: Message
    ) -> None:
        transcriber = FakeTranscriber()
        resolver = MediaResolver(repo, transcriber)
        resolver.resolve(_with_media(voice_message, media_id="vn_zzz"))
        assert transcriber.calls == []

    def test_results_are_cached_per_media_id(
        self, repo: DataRepository, voice_message: Message
    ) -> None:
        # Attachments repeat across messages and transcription is expensive.
        transcriber = FakeTranscriber()
        resolver = MediaResolver(repo, transcriber)
        for _ in range(3):
            resolver.resolve(voice_message)
        assert transcriber.calls == [voice_message.media_id]

    def test_composite_dispatches_by_modality(
        self, repo: DataRepository, voice_message: Message, image_message: Message
    ) -> None:
        both = CompositeUnderstanding(FakeTranscriber(), FakeOcr())
        resolver = MediaResolver(repo, both)
        assert resolver.resolve(voice_message).derived_from == "fake-whisper"
        assert resolver.resolve(image_message).derived_from == "fake-ocr"

    def test_a_crashing_provider_costs_one_transcript_not_the_run(
        self, repo: DataRepository, voice_message: Message
    ) -> None:
        media = MediaResolver(repo, ExplodingProvider()).resolve(voice_message)
        assert media.has_derived_text is False
        assert media.is_registered is True

    def test_providers_are_wrapped_defensively_exactly_once(
        self, repo: DataRepository
    ) -> None:
        already = SafeUnderstanding(FakeTranscriber())
        assert MediaResolver(repo, already).understanding is already


# --------------------------------------------------------------------------- #
# The integration claim
# --------------------------------------------------------------------------- #


class TestPluggingInAModelChangesRouting:
    """Whisper must need no change beyond one constructor argument."""

    def test_recovered_text_reaches_text_and_keyword_features(
        self, repo: DataRepository, voice_message: Message
    ) -> None:
        without = (
            MessagePipeline(repo, understanding=NullUnderstanding())
            .analyse(voice_message)
            .features
        )
        with_model = (
            MessagePipeline(repo, understanding=FakeTranscriber())
            .analyse(voice_message)
            .features
        )

        assert without.text.is_empty is True
        assert without.has_derived_text is False
        assert with_model.text.is_empty is False
        assert with_model.has_derived_text is True
        assert "rent" in with_model.text.normalized_text
        # The transcript is matched by the keyword vocabulary exactly as a
        # typed body would be - that is the whole point of the seam.
        assert "invoice" in with_model.matched_keywords

    def test_typed_body_is_preserved_alongside_the_transcript(
        self, repo: DataRepository, image_message: Message
    ) -> None:
        typed = image_message.message_text or ""
        features = (
            MessagePipeline(repo, understanding=FakeOcr()).analyse(image_message).features
        )
        assert typed.casefold()[:20] in features.text.normalized_text
        assert "limited time offer" in features.text.normalized_text

    def test_installing_a_model_is_one_argument_to_load(
        self, repo: DataRepository, voice_message: Message
    ) -> None:
        baseline = RoutingPipeline(repo, understanding=NullUnderstanding()).route(
            voice_message
        )
        with_model = RoutingPipeline(repo, understanding=FakeTranscriber()).route(
            voice_message
        )

        # Same contract, different evidence base.
        assert with_model.message_id == baseline.message_id
        assert 0.0 <= with_model.confidence <= 1.0
        assert with_model.to_output_row().keys() == baseline.to_output_row().keys()

    def test_the_whole_dataset_still_routes_with_a_model_installed(
        self, repo: DataRepository
    ) -> None:
        results = RoutingPipeline(
            repo, understanding=CompositeUnderstanding(FakeTranscriber(), FakeOcr())
        ).route_all()
        assert len(results) == len(repo.get_messages())
        assert all(0.0 <= r.confidence <= 1.0 for r in results)

    def test_a_crashing_model_still_produces_every_prediction(
        self, repo: DataRepository
    ) -> None:
        results = RoutingPipeline(repo, understanding=ExplodingProvider()).route_all()
        assert len(results) == len(repo.get_messages())

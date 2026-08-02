"""Speech-to-text: the transcriber, the cache, and the equivalence claim.

Nothing here loads real Whisper weights or touches the network. The model is
an injected fake, which keeps the suite fast, offline and deterministic while
still exercising the exact code path the real model runs through -
``WhisperTranscriber`` treats any object with a ``transcribe`` method the same
way.

The most important section is :class:`TestVoiceEqualsText`. The whole design
rests on one claim: a transcribed voice note is indistinguishable to the rest
of the pipeline from a message whose sender typed the same words. If that ever
stops being true, the integration has grown a special case, and these tests
fail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.data.models import Message
from src.data.repository import DataRepository
from src.media.cache import CachingUnderstanding, TranscriptCache, fingerprint
from src.media.content import MediaAttachment, MediaContent, MediaModality
from src.media.resolver import MediaResolver
from src.media.understanding import NullUnderstanding, default_understanding
from src.media.whisper import (
    WhisperTranscriber,
    WhisperUnavailableError,
    is_whisper_available,
)
from src.pipeline import MessagePipeline
from src.routing.pipeline import RoutingPipeline


# --------------------------------------------------------------------------- #
# Fakes standing in for faster-whisper
# --------------------------------------------------------------------------- #


@dataclass
class FakeSegment:
    """One segment of a fake transcription."""

    text: str
    start: float = 0.0
    end: float = 2.0
    avg_logprob: float = -0.2


@dataclass
class FakeInfo:
    """The metadata faster-whisper returns alongside the segments."""

    language: str = "en"
    language_probability: float = 0.99


class FakeModel:
    """A stand-in ``WhisperModel``. Records how it was called."""

    def __init__(self, *segments: FakeSegment, info: FakeInfo | None = None) -> None:
        self._segments = segments or (FakeSegment("the tanker cannot wait, 20 mins max"),)
        self._info = info or FakeInfo()
        self.calls: list[object] = []
        self.kwargs: list[dict[str, object]] = []

    def transcribe(self, source: object, **kwargs: object):
        self.calls.append(source)
        self.kwargs.append(kwargs)
        return iter(self._segments), self._info


class ExplodingModel:
    """A model that fails the way a corrupt file makes a real one fail."""

    def transcribe(self, source: object, **kwargs: object):  # noqa: ARG002
        raise RuntimeError("Invalid data found when processing input")


def _stub_decoder(_path: Path) -> object:
    """Stand in for audio decoding, for tests using a path that does not exist.

    A fake model ignores what it is handed, so the samples need not be real -
    but without this the transcriber would try to decode ``x.mp3`` for real and
    fail before reaching the model under test.
    """
    return "fake-samples"


def _transcriber(*segments: FakeSegment, model: object | None = None) -> WhisperTranscriber:
    """A transcriber wired to a fake model and a decoder that never touches disk."""
    return WhisperTranscriber(
        model=model if model is not None else FakeModel(*segments),
        decoder=_stub_decoder,
    )


#: A voice attachment whose path is never actually read; see :func:`_stub_decoder`.
FAKE_ATTACHMENT = MediaAttachment("vn_x", MediaModality.VOICE, Path("x.mp3"), exists=True)


@pytest.fixture
def voice_message(repo: DataRepository) -> Message:
    """An incoming voice note whose audio is on disk."""
    for message in repo.get_messages():
        if message.media_type == "voice" and repo.get_media_path(message) is not None:
            return message
    pytest.skip("dataset has no resolvable voice note")


@pytest.fixture
def isolated_cache(tmp_path: Path) -> TranscriptCache:
    """A cache in a throwaway location, so the committed one is never touched."""
    return TranscriptCache(tmp_path / "transcripts.json")


# --------------------------------------------------------------------------- #
# The transcriber
# --------------------------------------------------------------------------- #


class TestWhisperTranscriber:
    def test_transcribes_voice_only(self) -> None:
        transcriber = WhisperTranscriber(model=FakeModel())
        assert transcriber.supports(MediaModality.VOICE) is True
        assert transcriber.supports(MediaModality.IMAGE) is False

    def test_produces_text_with_provenance(
        self, repo: DataRepository, voice_message: Message
    ) -> None:
        model = FakeModel(FakeSegment("please call now, dad is unwell"))
        media = MediaResolver(repo, WhisperTranscriber(model=model)).resolve(voice_message)

        assert media.derived_text == "please call now, dad is unwell"
        assert media.derived_from.startswith("faster-whisper:")
        assert media.derived_language == "en"
        assert 0.0 < media.derived_confidence <= 0.95

    def test_joins_multiple_segments(self) -> None:
        content = _transcriber(
            FakeSegment("first part."), FakeSegment("second part.")
        ).understand(FAKE_ATTACHMENT)
        assert content.text == "first part. second part."

    def test_confidence_never_claims_certainty(self) -> None:
        # A transcript is never as reliable as text the sender typed.
        content = _transcriber(FakeSegment("crystal clear", avg_logprob=0.0)).understand(
            FAKE_ATTACHMENT
        )
        assert content.confidence <= 0.95

    def test_poor_audio_scores_lower_than_clean_audio(self) -> None:
        clean = _transcriber(FakeSegment("hi", avg_logprob=-0.1))
        noisy = _transcriber(FakeSegment("hi", avg_logprob=-2.0))
        assert (
            noisy.understand(FAKE_ATTACHMENT).confidence
            < clean.understand(FAKE_ATTACHMENT).confidence
        )

    def test_looping_is_disabled(self) -> None:
        # Whisper repeats itself on repetitive audio when previous text is fed
        # back as context, and two of this dataset's voice notes are robocalls.
        model = FakeModel()
        _transcriber(model=model).understand(FAKE_ATTACHMENT)
        assert model.kwargs[0]["condition_on_previous_text"] is False

    def test_model_is_loaded_lazily(self) -> None:
        # Constructing must be free: a dataset with no voice notes should never
        # pay for the weights.
        transcriber = WhisperTranscriber("nonexistent-model-name")
        assert transcriber.model_size == "nonexistent-model-name"


class TestGracefulFailure:
    """Every failure yields an empty transcript. None raises."""

    ATTACHMENT = FAKE_ATTACHMENT

    def test_a_model_that_raises(self) -> None:
        content = _transcriber(model=ExplodingModel()).understand(self.ATTACHMENT)
        assert content is MediaContent.EMPTY or not content.has_text

    def test_silence_transcribes_to_nothing(self) -> None:
        content = _transcriber(FakeSegment("   ")).understand(self.ATTACHMENT)
        assert content.has_text is False

    def test_attachment_with_no_path(self) -> None:
        content = _transcriber().understand(MediaAttachment("vn_x", MediaModality.VOICE))
        assert content.has_text is False

    def test_a_decoder_that_cannot_read_the_file(self) -> None:
        def broken(_path: Path) -> object:
            raise WhisperUnavailableError("unsupported codec")

        content = WhisperTranscriber(model=FakeModel(), decoder=broken).understand(
            self.ATTACHMENT
        )
        assert content.has_text is False

    def test_a_file_that_is_not_audio(self, tmp_path: Path) -> None:
        # The real decoder, a real path, and bytes that are not audio - the
        # corrupt-file case, end to end.
        corrupt = tmp_path / "corrupt.mp3"
        corrupt.write_bytes(b"this is not an mp3" * 100)
        content = WhisperTranscriber(model=FakeModel()).understand(
            MediaAttachment("vn_bad", MediaModality.VOICE, corrupt, exists=True)
        )
        assert content.has_text is False

    def test_missing_audio_never_reaches_the_model(
        self, repo: DataRepository, voice_message: Message
    ) -> None:
        model = FakeModel()
        resolver = MediaResolver(repo, WhisperTranscriber(model=model))
        resolver.resolve(Message(**{**voice_message.to_dict(), "media_id": "vn_zzz"}))
        assert model.calls == []

    def test_an_unavailable_library_is_reported_once(self) -> None:
        # A missing install should not be retried per file across a whole run.
        transcriber = WhisperTranscriber("base", decoder=_stub_decoder)
        transcriber._load_failed = True  # noqa: SLF001 - simulating a failed load
        for _ in range(3):
            assert transcriber.understand(self.ATTACHMENT).has_text is False

    def test_the_whole_dataset_routes_with_a_broken_model(
        self, repo: DataRepository
    ) -> None:
        results = RoutingPipeline(
            repo, understanding=WhisperTranscriber(model=ExplodingModel())
        ).route_all()
        assert len(results) == len(repo.get_messages())
        assert all(0.0 <= r.confidence <= 1.0 for r in results)

    def test_availability_probe_never_raises(self) -> None:
        assert isinstance(is_whisper_available(), bool)


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #


class TestTranscriptCache:
    def test_round_trips_through_disk(self, isolated_cache: TranscriptCache) -> None:
        isolated_cache.put("vn_001", MediaContent("hello", "test", 0.5, "en"), "fp")
        isolated_cache.save()

        reloaded = TranscriptCache(isolated_cache.path)
        content = reloaded.get("vn_001", "fp")
        assert content is not None
        assert content.text == "hello"
        assert content.confidence == pytest.approx(0.5)
        assert content.language == "en"

    def test_a_miss_returns_none(self, isolated_cache: TranscriptCache) -> None:
        assert isolated_cache.get("vn_nothing", "fp") is None

    def test_changed_audio_invalidates_the_entry(
        self, isolated_cache: TranscriptCache
    ) -> None:
        # Serving a stale transcript would be a silent, permanent error.
        isolated_cache.put("vn_001", MediaContent("old", "test", 0.5), "fingerprint-a")
        assert isolated_cache.get("vn_001", "fingerprint-b") is None

    def test_transcription_happens_once_across_runs(
        self, repo: DataRepository, voice_message: Message, isolated_cache: TranscriptCache
    ) -> None:
        model = FakeModel()
        for _ in range(2):
            provider = CachingUnderstanding(
                WhisperTranscriber(model=model), isolated_cache
            )
            MediaResolver(repo, provider).resolve(voice_message)
        assert len(model.calls) == 1

    def test_a_second_process_reuses_the_file(
        self, repo: DataRepository, voice_message: Message, isolated_cache: TranscriptCache
    ) -> None:
        first = CachingUnderstanding(
            WhisperTranscriber(model=FakeModel(FakeSegment("recorded once"))),
            isolated_cache,
        )
        MediaResolver(repo, first).resolve(voice_message)

        # A fresh cache object over the same file, and a model that would fail
        # if it were consulted at all.
        second = CachingUnderstanding(
            WhisperTranscriber(model=ExplodingModel()),
            TranscriptCache(isolated_cache.path),
        )
        media = MediaResolver(repo, second).resolve(voice_message)
        assert media.derived_text == "recorded once"

    def test_failures_are_not_cached(
        self, repo: DataRepository, voice_message: Message, isolated_cache: TranscriptCache
    ) -> None:
        # A failure is usually environmental. Caching it would make a missing
        # install permanent even after it is fixed.
        failing = CachingUnderstanding(
            WhisperTranscriber(model=ExplodingModel()), isolated_cache
        )
        MediaResolver(repo, failing).resolve(voice_message)
        assert len(isolated_cache) == 0

    def test_cached_transcripts_work_without_any_model(
        self, repo: DataRepository, voice_message: Message, isolated_cache: TranscriptCache
    ) -> None:
        # The path a checkout without faster-whisper installed takes.
        isolated_cache.put(
            voice_message.media_id,
            MediaContent("from the cache", "faster-whisper:base", 0.7, "en"),
            fingerprint(repo.get_media_path(voice_message)),
        )
        provider = CachingUnderstanding(NullUnderstanding(), isolated_cache)
        media = MediaResolver(repo, provider).resolve(voice_message)
        assert media.derived_text == "from the cache"

    def test_a_corrupt_cache_file_is_ignored_not_fatal(self, tmp_path: Path) -> None:
        path = tmp_path / "transcripts.json"
        path.write_text("{not json at all", encoding="utf-8")
        assert len(TranscriptCache(path)) == 0

    def test_a_cache_from_another_version_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "transcripts.json"
        path.write_text(json.dumps({"version": 999, "transcripts": {"a": {}}}), "utf-8")
        assert len(TranscriptCache(path)) == 0

    def test_clear_forces_retranscription(
        self, repo: DataRepository, voice_message: Message, isolated_cache: TranscriptCache
    ) -> None:
        model = FakeModel()
        provider = CachingUnderstanding(WhisperTranscriber(model=model), isolated_cache)
        MediaResolver(repo, provider).resolve(voice_message)
        isolated_cache.clear()
        MediaResolver(repo, provider).resolve(voice_message)
        assert len(model.calls) == 2

    def test_saving_is_skipped_when_nothing_changed(
        self, isolated_cache: TranscriptCache
    ) -> None:
        assert isolated_cache.save() is False

    def test_the_shipped_cache_covers_the_dataset(self, repo: DataRepository) -> None:
        """Every voice note in the shipped dataset has a committed transcript.

        This is what lets a checkout with no faster-whisper installed still
        route on speech. If it fails, `python main.py --refresh-transcripts`
        needs re-running and the result committing.
        """
        cache = TranscriptCache()
        missing = [
            m.media_id
            for m in repo.get_messages()
            if m.media_type == "voice"
            and cache.get(m.media_id, fingerprint(repo.get_media_path(m))) is None
        ]
        assert missing == []


class TestDefaultResolution:
    def test_transcription_can_be_switched_off(self) -> None:
        provider = default_understanding(transcribe=False)
        assert provider.supports(MediaModality.VOICE) is False

    def test_the_default_can_handle_voice(self) -> None:
        # Via a live model or the committed cache; either satisfies the claim.
        assert default_understanding().supports(MediaModality.VOICE) is True

    def test_images_are_still_unimplemented(self) -> None:
        assert default_understanding().supports(MediaModality.IMAGE) is False


# --------------------------------------------------------------------------- #
# The claim the whole design rests on
# --------------------------------------------------------------------------- #


class TestVoiceEqualsText:
    """A transcribed voice note must route as if the sender had typed it.

    This is the regression guard on the integration. Feature extraction,
    classification, personalisation and routing were not modified to support
    speech; a transcript simply becomes the message body. So for any given
    text, routing a voice note that transcribes to it must produce the same
    decision as routing a text message containing it.
    """

    #: Texts chosen to exercise different branches of the classifier, so the
    #: equivalence is not demonstrated on one easy category.
    TRANSCRIPTS = [
        "please call now dad is unwell and we are going to the clinic",
        "your bank account will be blocked today share the otp you received",
        "had dinner call when free nothing urgent",
        "today's school pickup will be from gate 2 instead of the main gate",
        "limited time offer flat 50 percent off, shop now before the sale ends",
    ]

    @staticmethod
    def _as_text_message(voice: Message, transcript: str) -> Message:
        """The same envelope, with the transcript typed in and no attachment."""
        return Message(
            **{
                **voice.to_dict(),
                "message_text": transcript,
                "media_type": None,
                "media_id": None,
            }
        )

    @pytest.mark.parametrize("transcript", TRANSCRIPTS)
    def test_classification_matches(
        self, repo: DataRepository, voice_message: Message, transcript: str
    ) -> None:
        spoken = MessagePipeline(
            repo, understanding=WhisperTranscriber(model=FakeModel(FakeSegment(transcript)))
        ).analyse(voice_message)
        typed = MessagePipeline(repo, understanding=NullUnderstanding()).analyse(
            self._as_text_message(voice_message, transcript)
        )
        assert spoken.classification.message_type is typed.classification.message_type

    @pytest.mark.parametrize("transcript", TRANSCRIPTS)
    def test_routing_action_matches(
        self, repo: DataRepository, voice_message: Message, transcript: str
    ) -> None:
        spoken = RoutingPipeline(
            repo, understanding=WhisperTranscriber(model=FakeModel(FakeSegment(transcript)))
        ).route(voice_message)
        typed = RoutingPipeline(repo, understanding=NullUnderstanding()).route(
            self._as_text_message(voice_message, transcript)
        )
        assert spoken.action is typed.action
        assert spoken.message_type == typed.message_type

    @pytest.mark.parametrize("transcript", TRANSCRIPTS)
    def test_extracted_text_features_match(
        self, repo: DataRepository, voice_message: Message, transcript: str
    ) -> None:
        # The transcript *is* the body: same tokens, same keywords, same
        # length. Anything else would mean a special case crept in.
        spoken = MessagePipeline(
            repo, understanding=WhisperTranscriber(model=FakeModel(FakeSegment(transcript)))
        ).analyse(voice_message).features
        typed = MessagePipeline(repo, understanding=NullUnderstanding()).analyse(
            self._as_text_message(voice_message, transcript)
        ).features

        assert spoken.text.normalized_text == typed.text.normalized_text
        assert spoken.text.tokens == typed.text.tokens
        assert spoken.keywords.matches == typed.keywords.matches

    def test_a_typed_caption_is_preserved_alongside_the_transcript(
        self, repo: DataRepository, voice_message: Message
    ) -> None:
        # Transcription adds to the body; it never replaces what was typed.
        captioned = Message(**{**voice_message.to_dict(), "message_text": "see below"})
        features = (
            MessagePipeline(
                repo,
                understanding=WhisperTranscriber(model=FakeModel(FakeSegment("the rest"))),
            )
            .analyse(captioned)
            .features
        )
        assert "see below" in features.text.normalized_text
        assert "the rest" in features.text.normalized_text

    def test_only_voice_messages_change_decision(
        self, repo: DataRepository
    ) -> None:
        """No text message may be routed differently because of transcription."""
        silent = RoutingPipeline(repo, understanding=NullUnderstanding()).route_all()
        spoken = RoutingPipeline(repo).route_all()

        voice_ids = {m.message_id for m in repo.get_messages() if m.media_type == "voice"}
        moved = {
            a.message_id
            for a, b in zip(silent, spoken, strict=True)
            if (a.action, a.message_type) != (b.action, b.message_type)
        }
        assert moved <= voice_ids

    def test_text_messages_may_still_gain_evidence_from_transcribed_history(
        self, repo: DataRepository
    ) -> None:
        """The one way transcription legitimately reaches a text message.

        ``message_history.csv`` contains voice notes too, and the evidence
        engine classifies history with the same classifier it uses on incoming
        messages. Once those historical voice notes are transcribed they become
        classifiable, so they become eligible as evidence for a text message of
        the same category - which is an improvement, not a leak.

        The guard is that this may only ever change *evidence* and the
        confidence that follows from it. If it moved an action or a category,
        the test above fails.
        """
        silent = RoutingPipeline(repo, understanding=NullUnderstanding()).route_all()
        spoken = RoutingPipeline(repo).route_all()

        voice_ids = {m.message_id for m in repo.get_messages() if m.media_type == "voice"}
        changed_rows = {
            a.message_id
            for a, b in zip(silent, spoken, strict=True)
            if a.to_output_row() != b.to_output_row()
        }
        for message_id in changed_rows - voice_ids:
            before = next(r for r in silent if r.message_id == message_id)
            after = next(r for r in spoken if r.message_id == message_id)
            assert before.action is after.action
            assert before.message_type == after.message_type

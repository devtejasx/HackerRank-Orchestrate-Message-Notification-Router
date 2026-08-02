"""Speech-to-text for voice notes, via ``faster-whisper``.

This is a :class:`~src.media.understanding.MediaUnderstanding` implementation
and nothing more. It is the only module in the project that knows what a
speech recognition model is; everything downstream sees a transcript that is
indistinguishable from text the sender typed, because
:func:`~src.features.extractor._analysable_text` joins it to the body before
feature extraction runs.

    voice note -> WhisperTranscriber -> MediaContent -> MediaFeatures
        -> FeatureExtractor -> classifier -> personalisation -> routing

Nothing in that chain past the second arrow was changed to add this.

Three things are deliberate:

* **The model loads lazily.** Constructing a transcriber is free; the weights
  are fetched and loaded on the first voice note that actually needs them. A
  dataset with no voice notes therefore never pays for the model, and a
  misconfigured install fails at the point of use where it can be caught,
  rather than at import where it would take the process with it.
* **Every failure returns an empty transcript.** Missing file, unreadable
  audio, unsupported codec, model that will not load, model that raises
  mid-decode - all of them degrade to "nothing recovered". The submission
  contract requires a prediction for every message, and a message whose audio
  could not be read is still a message to be routed.
* **Audio decoding is injectable.** By default the file path is handed to
  ``faster-whisper``, which decodes it with PyAV. Environments where PyAV is
  unavailable can supply their own decoder instead of losing transcription
  entirely; see :func:`samples_from_soundfile`.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

from src import config
from src.media.content import MediaAttachment, MediaContent, MediaModality
from src.utils.helpers import clamp

__all__ = [
    "WhisperTranscriber",
    "WhisperUnavailableError",
    "is_whisper_available",
    "samples_from_soundfile",
    "samples_from_ffmpeg",
    "best_available_decoder",
    "DEFAULT_MODEL_SIZE",
]

_LOGGER = config.get_logger("media.whisper")

#: Model weights used when a caller names none.
#:
#: ``base`` rather than ``tiny`` or ``small``: the voice notes here are short,
#: conversational and often decide between "this can wait" and "this cannot",
#: which is exactly the distinction ``tiny`` blurs. ``base`` runs comfortably
#: on CPU in this dataset's scale and is a far better accuracy trade than the
#: seconds it costs.
DEFAULT_MODEL_SIZE: Final[str] = "base"

#: Sample rate Whisper expects when it is handed raw samples.
SAMPLE_RATE: Final[int] = 16_000

#: Decoder signature: a readable path in, mono 16 kHz float32 samples out.
AudioDecoder = Callable[[Path], Any]


class WhisperUnavailableError(RuntimeError):
    """Raised internally when ``faster-whisper`` cannot be used.

    Never propagates out of :meth:`WhisperTranscriber.understand`; it is caught
    there and turned into an empty transcript.
    """


def is_whisper_available() -> bool:
    """Whether ``faster-whisper`` can be imported in this environment.

    Cheap enough to call before building a pipeline, and used by
    :func:`~src.media.understanding.default_understanding` to decide whether
    transcription is possible at all. Importing may still succeed while the
    *model* fails to download; that is handled separately, at first use.
    """
    try:
        import faster_whisper  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - a broken install is "unavailable"
        _LOGGER.debug("faster-whisper is not usable here: %s", exc)
        return False
    return True


class WhisperTranscriber:
    """Turns a voice note into text with ``faster-whisper``.

    Args:
        model_size: Whisper weights to load, e.g. ``tiny``, ``base``,
            ``small``. Defaults to :data:`DEFAULT_MODEL_SIZE`.
        device: ``cpu`` or ``cuda``.
        compute_type: ctranslate2 quantisation. ``int8`` keeps CPU inference
            fast at negligible cost for speech this short.
        language: Force a language instead of detecting one. ``None`` detects.
        beam_size: Decoding beam width.
        model: A pre-built ``WhisperModel``, mainly for tests. Supplying one
            skips loading entirely.
        decoder: Optional callable turning a path into mono 16 kHz float32
            samples. Audio is then decoded with this and the samples handed to
            Whisper, bypassing ``faster-whisper``'s own PyAV-based decoding.
            Omit it to let :func:`best_available_decoder` choose: ``None`` when
            PyAV works, a fallback when it does not.

    Example:
        >>> from src.routing.pipeline import RoutingPipeline        # doctest: +SKIP
        >>> RoutingPipeline.load(understanding=WhisperTranscriber())  # doctest: +SKIP
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 5,
        model: Any | None = None,
        decoder: AudioDecoder | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._model = model
        self._decoder = decoder if decoder is not None else best_available_decoder()
        self._load_failed = False
        self.name = f"faster-whisper:{model_size}"

    @property
    def model_size(self) -> str:
        """The weights this transcriber loads."""
        return self._model_size

    def supports(self, modality: MediaModality) -> bool:
        """Voice notes only. Images are somebody else's job."""
        return modality is MediaModality.VOICE

    def understand(self, attachment: MediaAttachment) -> MediaContent:
        """Transcribe one voice note.

        Args:
            attachment: A readable voice attachment. The resolver has already
                confirmed the file exists.

        Returns:
            The transcript with its provenance, or
            :data:`MediaContent.EMPTY` if anything at all went wrong. This
            method does not raise.
        """
        if attachment.file_path is None:
            return MediaContent.EMPTY
        try:
            return self._transcribe(attachment.file_path)
        except WhisperUnavailableError as exc:
            _LOGGER.warning("Transcription unavailable for %s: %s", attachment.media_id, exc)
        except Exception:  # noqa: BLE001 - corrupt audio, bad codec, model bug
            _LOGGER.exception(
                "Could not transcribe %s; continuing without a transcript",
                attachment.media_id,
            )
        return MediaContent.EMPTY

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _transcribe(self, path: Path) -> MediaContent:
        """Run the model over one file and package the result."""
        model = self._ensure_model()
        source = self._decoder(path) if self._decoder is not None else str(path)

        segments, info = model.transcribe(
            source,
            language=self._language,
            beam_size=self._beam_size,
            # Whisper otherwise feeds each segment's text back in as context,
            # which makes it loop on repetitive audio - and two of this
            # dataset's voice notes are marketing robocalls that repeat
            # verbatim. Left on, their transcripts came back with the same
            # paragraph twice, inflating every length-derived feature.
            condition_on_previous_text=False,
        )
        # `segments` is a generator; nothing is decoded until it is consumed.
        collected = list(segments)
        text = " ".join(segment.text.strip() for segment in collected).strip()

        if not text:
            _LOGGER.info("%s contained no recognisable speech", path.name)
            return MediaContent.EMPTY

        return MediaContent(
            text=text,
            provider=self.name,
            confidence=_transcript_confidence(collected, info),
            language=getattr(info, "language", None),
        )

    def _ensure_model(self) -> Any:
        """Load the model once, on first use.

        Raises:
            WhisperUnavailableError: If the library or the weights cannot be
                loaded. Remembered, so a run with many voice notes reports the
                problem once rather than retrying per file.
        """
        if self._model is not None:
            return self._model
        if self._load_failed:
            raise WhisperUnavailableError("model previously failed to load")

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            self._load_failed = True
            raise WhisperUnavailableError(f"faster-whisper is not importable: {exc}") from exc

        _LOGGER.info(
            "Loading Whisper %s (%s, %s)", self._model_size, self._device, self._compute_type
        )
        try:
            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        except Exception as exc:  # noqa: BLE001 - no weights, no disk, no network
            self._load_failed = True
            raise WhisperUnavailableError(f"could not load Whisper weights: {exc}") from exc
        return self._model


def _transcript_confidence(segments: Sequence[Any], info: Any) -> float:
    """Score how much to trust a transcript, in ``[0, 1]``.

    Whisper reports a mean token log-probability per segment. Exponentiating it
    turns it back into a per-token probability, which is the closest thing the
    model offers to "how sure was I". That is combined with the detected
    language probability, because a transcript of audio Whisper could not even
    place in a language deserves less weight.

    The result is capped below 1.0. A transcript is never as reliable as text
    the sender actually typed, and routing should not be able to treat it as
    though it were.
    """
    if not segments:
        return 0.0

    total_duration = sum(
        max(getattr(s, "end", 0.0) - getattr(s, "start", 0.0), 0.0) for s in segments
    )
    if total_duration > 0:
        weighted = sum(
            math.exp(getattr(s, "avg_logprob", -1.0))
            * max(getattr(s, "end", 0.0) - getattr(s, "start", 0.0), 0.0)
            for s in segments
        )
        acoustic = weighted / total_duration
    else:
        acoustic = sum(math.exp(getattr(s, "avg_logprob", -1.0)) for s in segments) / len(
            segments
        )

    language_probability = float(getattr(info, "language_probability", 1.0) or 1.0)
    return round(clamp(acoustic * language_probability, 0.0, 0.95), 4)


def samples_from_soundfile(path: Path) -> Any:
    """Decode ``path`` to mono 16 kHz float32 samples using ``soundfile``.

    An alternative to ``faster-whisper``'s built-in decoding, for environments
    where PyAV cannot be loaded - a locked-down Windows host blocking its DLL,
    or a minimal container without it. Pass it as ``decoder=`` and
    transcription works with no other change:

    .. code-block:: python

        WhisperTranscriber(decoder=samples_from_soundfile)

    Raises:
        WhisperUnavailableError: If ``soundfile`` or ``numpy`` is missing.
    """
    try:
        import numpy as np
        import soundfile as sf
    except Exception as exc:  # noqa: BLE001
        raise WhisperUnavailableError(
            f"the soundfile decoder needs soundfile and numpy: {exc}"
        ) from exc

    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:  # stereo -> mono
        data = data.mean(axis=1)
    if rate == SAMPLE_RATE:
        return data

    # Linear interpolation. Cruder than a polyphase filter, but speech
    # recognition tolerates it well and it avoids a scipy dependency for what
    # is already a fallback path.
    target_length = int(round(len(data) * SAMPLE_RATE / float(rate)))
    if target_length <= 0:
        raise WhisperUnavailableError(f"{path.name} decoded to no audio")
    return np.interp(
        np.linspace(0, len(data) - 1, target_length),
        np.arange(len(data)),
        data,
    ).astype("float32")


def samples_from_ffmpeg(path: Path) -> Any:
    """Decode ``path`` to mono 16 kHz float32 samples by shelling out to ffmpeg.

    The most capable of the decoders here, and the one to reach for when
    :func:`samples_from_soundfile` fails. libsndfile's MP3 support is
    effectively Layer III only, and this dataset contains Layer I and Layer II
    files that it reports metadata for but cannot decode - a failure that looks
    like corruption and is not.

    Finds ffmpeg on ``PATH``, or falls back to the static binary that
    ``imageio-ffmpeg`` ships, so it needs no system-level install.

    Raises:
        WhisperUnavailableError: If no ffmpeg is available, or it cannot decode
            the file - which for genuinely corrupt audio is the correct answer.
    """
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise WhisperUnavailableError(f"the ffmpeg decoder needs numpy: {exc}") from exc

    executable = _find_ffmpeg()
    if executable is None:
        raise WhisperUnavailableError("no ffmpeg binary found")

    completed = subprocess.run(  # noqa: S603 - fixed argv, path is not shell-interpreted
        [
            executable, "-nostdin", "-loglevel", "error",
            "-i", str(path),
            "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
        ],
        capture_output=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", "replace").strip()[:200]
        raise WhisperUnavailableError(f"ffmpeg could not decode {path.name}: {detail}")
    return np.frombuffer(completed.stdout, dtype="float32")


def _find_ffmpeg() -> str | None:
    """Locate an ffmpeg executable, preferring one already on ``PATH``."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except Exception:  # noqa: BLE001
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def best_available_decoder() -> AudioDecoder | None:
    """Return a decoder for environments where Whisper cannot decode audio itself.

    ``faster-whisper`` decodes with PyAV, which is the right default and needs
    no help. But PyAV is a compiled extension, and there are real environments
    where it will not load - a locked-down Windows host blocking its DLL, or a
    slim container built without it. Losing transcription entirely to that
    would be a poor trade when ffmpeg or libsndfile is sitting right there.

    Returns:
        ``None`` when PyAV works, so the normal path is untouched. Otherwise
        the most capable fallback available, or ``None`` if there is none.
    """
    try:
        import av  # noqa: F401

        return None
    except Exception:  # noqa: BLE001 - PyAV unusable; find a substitute
        _LOGGER.debug("PyAV is unavailable; looking for a fallback audio decoder")

    if _find_ffmpeg() is not None:
        _LOGGER.info("Using ffmpeg to decode audio (PyAV is unavailable)")
        return samples_from_ffmpeg
    try:
        import soundfile  # noqa: F401
    except Exception:  # noqa: BLE001
        _LOGGER.warning("No usable audio decoder; voice notes cannot be transcribed")
        return None
    _LOGGER.info("Using soundfile to decode audio (PyAV is unavailable)")
    return samples_from_soundfile

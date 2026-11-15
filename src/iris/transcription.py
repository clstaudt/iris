"""Speech-to-text transcription using Whisper.

Two engines sit behind one interface. On Apple Silicon, MLX Whisper decodes on
the GPU: several times faster than CPU decoding, and it leaves the CPU idle.
Everywhere else faster-whisper (CTranslate2) runs, on CUDA when available.
"""

import platform
import sys
import threading
import warnings
from collections.abc import Callable, Collection
from pathlib import Path

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import collect_chunks, get_speech_timestamps

from iris.engines import RecognitionEngine, register_recognition_engine

try:
    import mlx.core as mx
    import mlx_whisper
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.transcribe import ModelHolder

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000

# Segments Whisper itself rates as more likely silence than speech are dropped
NO_SPEECH_THRESHOLD = 0.6

# Hugging Face repos per model size and engine
MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
# The sizes offered for choosing, smallest first
WHISPER_SIZES = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")

FASTER_WHISPER_REPOS = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# Whisper tells the language from the sound, and an accent fools it: a German
# speaker's "La sinagoga è aperta al pubblico." was heard as English, and
# Whisper, writing English, wrote "The synagogue is open for the public.", a
# translation. In a conversation, what is said is written in the
# conversation's language unless Whisper finds another this many times
# likelier: one of the user's languages (Iris's languages, as chosen in
# Settings)...
SWITCH = 2.0
# ...or another
SWITCH_ELSEWHERE = 10.0


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def resolve_engine(engine: str = "auto") -> str:
    """Pick "mlx" or "faster-whisper"; MLX only where it can run."""
    mlx_usable = HAS_MLX and is_apple_silicon()
    if engine in ("auto", "mlx"):
        return "mlx" if mlx_usable else "faster-whisper"
    return engine


def resolve_device(engine: str, device: str = "auto") -> str:
    if engine == "mlx":
        return "gpu"
    if device == "auto":
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    return device


def resolve_model(model: str, engine: str, device: str) -> str:
    """Resolve "auto" to the best model the hardware decodes in about two
    seconds: large-v3-turbo on a GPU, small on a CPU."""
    model = {"turbo": "large-v3-turbo"}.get(model, model)
    if model != "auto":
        return model
    return "large-v3-turbo" if engine == "mlx" or device == "cuda" else "small"


def folder_engine(folder: Path) -> str | None:
    """Which engine a local Whisper model folder is for: CTranslate2 models
    (faster-whisper) have a model.bin, MLX conversions their weights."""
    if (folder / "model.bin").exists():
        return "faster-whisper"
    if (folder / "weights.safetensors").exists() or (folder / "weights.npz").exists():
        return "mlx"
    return None


def model_repo(model: str, engine: str) -> str:
    """The Hugging Face repo holding this model for this engine."""
    if engine == "mlx":
        return MLX_REPOS.get(model, model)
    return FASTER_WHISPER_REPOS.get(model, f"Systran/faster-whisper-{model}")


def language_heard(
    likelihood: dict[str, float],
    conversation: str | None,
    languages: Collection[str] | None = None,
) -> str:
    """The language to write what was said in.

    Args:
        likelihood: How likely Whisper finds each language, by its code
        conversation: The language the conversation is in; None before the
            user has said anything
        languages: The user's languages; None counts every language as one

    Returns:
        The likeliest language, or the conversation's when the likeliest is
        not SWITCH times likelier than it (SWITCH_ELSEWHERE times for a
        language not the user's)
    """
    likeliest = max(likelihood, key=likelihood.__getitem__)
    if conversation is None:
        return likeliest
    switch = SWITCH if languages is None or likeliest in languages else SWITCH_ELSEWHERE
    if likelihood[likeliest] >= switch * likelihood.get(conversation, 0.0):
        return likeliest
    return conversation


class WhisperTranscriber:
    """Transcribes audio to text using Whisper model."""

    def __init__(
        self,
        model: str = "auto",
        device: str = "auto",
        engine: str = "auto",
    ):
        """
        Initialize Whisper transcriber.

        Args:
            model: Whisper model size (tiny, base, small, medium, large-v3,
                large-v3-turbo) or "auto" for the best one the hardware runs fast
            device: Device for faster-whisper (cpu, cuda, or auto)
            engine: "auto", "mlx" or "faster-whisper"; MLX needs Apple Silicon
                and falls back to faster-whisper elsewhere
        """
        # A local model folder instead of a size: its contents pick the engine
        self.model_path: Path | None = None
        candidate = Path(model).expanduser()
        if model != "auto" and candidate.is_dir():
            engine = folder_engine(candidate)
            if engine is None:
                raise ValueError(
                    f"{candidate} holds no Whisper model "
                    "(expected MLX weights or a faster-whisper model.bin)"
                )
            if engine == "mlx" and resolve_engine("mlx") != "mlx":
                raise ValueError("MLX models need an Apple Silicon Mac with mlx-whisper")
            self.model_path = candidate

        self.engine = resolve_engine(engine)
        self.device = resolve_device(self.engine, device)
        if self.model_path:
            self.model_name = self.model_path.name
        else:
            self.model_name = resolve_model(model, self.engine, self.device)
        self._transcription_lock = threading.Lock()

        if self.engine == "mlx":
            local = str(self.model_path) if self.model_path else None
            self.repo = local or model_repo(self.model_name, "mlx")
            self.whisper = None
            # Download and load the weights now rather than on the first turn
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            mlx_whisper.transcribe(silence, path_or_hf_repo=self.repo)
        else:
            self.repo = model_repo(self.model_name, "faster-whisper")
            self.whisper = WhisperModel(
                str(self.model_path) if self.model_path else self.model_name,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8",
            )

    def _segments(
        self,
        audio_path: str,
        conversation: str | None = None,
        languages: Collection[str] | None = None,
    ) -> tuple[list[tuple[str, float]], str]:
        """Decode speech into (text, no_speech_prob) segments and a language.

        Both engines see only the speech: Whisper invents words for noise
        ("You", "Thank you."), so the Silero VAD cuts non-speech first.
        Whisper writes in the language it takes the speech to be in, or the
        conversation's when it is unsure (language_heard).
        """
        if self.engine == "faster-whisper":
            segments, info = self.whisper.transcribe(
                audio_path,
                beam_size=5,
                language=None,  # Auto-detect language
                vad_filter=True,
            )
            language = info.language
            if conversation and info.all_language_probs:
                language = language_heard(dict(info.all_language_probs), conversation, languages)
            if language != info.language:
                # Nothing is decoded before the segments are read
                segments, _ = self.whisper.transcribe(
                    audio_path, beam_size=5, language=language, vad_filter=True
                )
            return [(s.text, s.no_speech_prob) for s in segments], language

        audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
        speech = get_speech_timestamps(audio)
        if not speech:
            return [], ""
        chunks, _ = collect_chunks(audio, speech)
        speech = np.concatenate(chunks)
        language = None  # Auto-detect language
        if conversation:
            language = language_heard(self._mlx_likelihood(speech), conversation, languages)
        result = mlx_whisper.transcribe(speech, path_or_hf_repo=self.repo, language=language)
        segments = [(s["text"], s.get("no_speech_prob", 0.0)) for s in result["segments"]]
        return segments, result.get("language", "")

    def _mlx_likelihood(self, speech: np.ndarray) -> dict[str, float]:
        """How likely Whisper finds each language in the speech, from up to
        its first 30 seconds, as mlx_whisper.transcribe finds it (and keeps
        to itself) when it is given no language."""
        model = ModelHolder.get_model(self.repo, mx.float16)
        mel = log_mel_spectrogram(speech, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        first = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, likelihood = model.detect_language(first)
        return likelihood

    def transcribe(
        self,
        audio_path: str,
        conversation: str | None = None,
        languages: Collection[str] | None = None,
    ) -> tuple[str, str]:
        """
        Transcribe audio file using Whisper with automatic language detection.

        Args:
            audio_path: Path to audio file
            conversation: The language the conversation is in, kept unless
                Whisper clearly hears another (language_heard); None to go
                by what Whisper hears alone
            languages: The user's languages: Whisper must be surer of
                another to leave the conversation's for it; None for all alike

        Returns:
            Tuple of (transcribed_text, detected_language_code).
            The language code is an ISO 639-1 code such as 'en', 'fr', 'ja'.
        """
        with self._transcription_lock:
            segments, language = self._segments(audio_path, conversation, languages)
            text = " ".join(
                text.strip() for text, no_speech in segments if no_speech < NO_SPEECH_THRESHOLD
            ).strip()
            return text, language

    def transcribe_streaming(
        self,
        audio_path: str,
        on_partial: Callable[[str], None],
    ) -> tuple[str, str]:
        """
        Transcribe audio with streaming partial results.

        This transcribes the audio and calls the callback with partial
        text as segments are processed.

        Args:
            audio_path: Path to audio file
            on_partial: Callback called with accumulated text after each segment

        Returns:
            Tuple of (final_transcribed_text, detected_language_code)
        """
        with self._transcription_lock:
            segments, language = self._segments(audio_path)

            accumulated_text = []
            for text, no_speech in segments:
                if no_speech >= NO_SPEECH_THRESHOLD:
                    continue
                accumulated_text.append(text.strip())
                on_partial(" ".join(accumulated_text).strip())

            return " ".join(accumulated_text).strip(), language


class StreamingTranscriber:
    """Handles streaming transcription from audio chunks."""

    def __init__(self, transcriber: WhisperTranscriber):
        """
        Initialize streaming transcriber.

        Args:
            transcriber: WhisperTranscriber instance to use
        """
        self.transcriber = transcriber
        self.current_text = ""
        self._lock = threading.Lock()
        self._chunk_files: list[str] = []

    def process_chunk(self, chunk_path: str) -> str:
        """
        Process an audio chunk and update the current transcription.

        Args:
            chunk_path: Path to audio chunk file

        Returns:
            Current accumulated transcription text
        """
        with self._lock:
            self._chunk_files.append(chunk_path)

            # Transcribe the cumulative audio (the chunk contains all audio so far)
            text = self.transcriber.transcribe(chunk_path)
            self.current_text = text

            return text

    def get_final_text(self, final_audio_path: str) -> str:
        """
        Get final transcription from complete audio.

        Args:
            final_audio_path: Path to complete audio file

        Returns:
            Final transcribed text
        """
        with self._lock:
            # Clean up chunk files
            for chunk_file in self._chunk_files:
                Path(chunk_file).unlink(missing_ok=True)
            self._chunk_files.clear()

            # Transcribe final complete audio for best accuracy
            self.current_text = self.transcriber.transcribe(final_audio_path)
            return self.current_text

    def reset(self):
        """Reset the streaming transcriber state."""
        with self._lock:
            self.current_text = ""
            for chunk_file in self._chunk_files:
                Path(chunk_file).unlink(missing_ok=True)
            self._chunk_files.clear()


def _whisper(options: dict) -> WhisperTranscriber:
    """Whisper with the recognition settings: model size or folder, device
    and backend ("engine": auto, mlx or faster-whisper)."""
    return WhisperTranscriber(
        options.get("model") or "auto",
        options.get("device") or "auto",
        engine=options.get("engine") or "auto",
    )


register_recognition_engine(RecognitionEngine("whisper", "Whisper", _whisper))

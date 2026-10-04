"""Wake word detection: is this utterance addressed to the assistant?

While the assistant is dormant, every utterance the microphone picks up is
checked here before anything else happens. The check is two cheap stages:
the Silero VAD (shipped with faster-whisper) drops anything that is not
speech, and Whisper tiny transcribes what is. On Apple Silicon tiny runs on
the GPU through MLX in about 30-40 ms per utterance, so staying dormant costs
next to nothing; elsewhere it runs on the CPU through faster-whisper.

The tiny model is loaded separately from the conversation model: MLX's
transcribe() keeps a single model in memory and would swap weights on every
switch between the two.
"""

import difflib
import re
import threading

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import collect_chunks, get_speech_timestamps

from iris.transcription import HAS_MLX, SAMPLE_RATE, model_repo, resolve_engine

if HAS_MLX:
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import DecodingOptions, decode
    from mlx_whisper.load_models import load_model

# Greetings that may come before the wake word ("Hey Iris", "Ciao Iris")
GREETINGS = ("hey", "hi", "hello", "ok", "okay", "ciao", "salut", "bonjour")

# How alike a heard word must be to the wake word ("Iris" heard as "Irus")
SIMILARITY = 0.75


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def match_wake_word(text: str, wake_word: str) -> tuple[bool, str]:
    """Check whether an utterance opens with the wake word.

    Only the start counts, as when calling someone by name: "Iris, what
    time is it?" wakes, "the irises bloomed" does not. One greeting may
    precede the wake word.

    Returns:
        (woken, remainder) where remainder is what was said after the wake
        word, e.g. "what time is it"
    """
    wake = _words(wake_word)
    heard = _words(text)
    if not wake or not heard:
        return False, ""
    # Read the opening both with and without a greeting: the wake phrase may
    # start with one itself ("Okay Jarvis")
    starts = (0, 1) if heard[0] in GREETINGS else (0,)
    for start in starts:
        candidate = heard[start : start + len(wake)]
        if len(candidate) < len(wake):
            continue
        similarity = difflib.SequenceMatcher(None, " ".join(candidate), " ".join(wake)).ratio()
        if similarity >= SIMILARITY:
            return True, " ".join(heard[start + len(wake) :])
    return False, ""


# Words that may come before or after a sleep word without changing it
# ("Okay, that's all, thanks")
SLEEP_FILLERS = ("ok", "okay", "so", "well", "alright", "right", "thanks", "thank", "you")


def _similar(a: list[str], b: list[str]) -> bool:
    """Same words, each allowing for a mishearing ("thats" for "that's")."""
    return len(a) == len(b) and all(
        difflib.SequenceMatcher(None, x, y).ratio() >= SIMILARITY for x, y in zip(a, b)
    )


def match_sleep_word(text: str, sleep_word: str, wake_word: str = "") -> bool:
    """Check whether an utterance asks the assistant to go to sleep.

    Only an utterance that is just the sleep word counts, optionally with the
    wake word ("That's all, Iris") and a filler at either end ("Okay, that's
    all, thanks"): "that's all I wanted to know about Lisbon" is a request.
    """
    heard, sleep, wake = _words(text), _words(sleep_word), _words(wake_word)
    if not heard or not sleep:
        return False
    remaining, n = [], 0
    while n < len(heard):
        if wake and _similar(heard[n : n + len(wake)], wake):
            n += len(wake)
            continue
        remaining.append(heard[n])
        n += 1
    while len(remaining) > len(sleep) and remaining[0] in SLEEP_FILLERS:
        remaining.pop(0)
    while len(remaining) > len(sleep) and remaining[-1] in SLEEP_FILLERS:
        remaining.pop()
    return _similar(remaining, sleep)


class WakeWordDetector:
    """Listens for the wake word with the smallest Whisper model."""

    def __init__(self, engine: str = "auto"):
        self.engine = resolve_engine(engine)
        self._lock = threading.Lock()
        if self.engine == "mlx":
            self._model = load_model(model_repo("tiny", "mlx"), dtype=mx.float16)
            self._decode_mlx(np.zeros(SAMPLE_RATE, dtype=np.float32), None)  # compile now
        else:
            self._model = WhisperModel("tiny", device="cpu", compute_type="int8")

    def _decode_mlx(self, audio: np.ndarray, prompt: str | None) -> str:
        mel = log_mel_spectrogram(audio, n_mels=self._model.dims.n_mels)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2)
        options = DecodingOptions(prompt=prompt, without_timestamps=True, fp16=True)
        return decode(self._model, mel, options).text

    def transcribe(self, audio: np.ndarray, wake_word: str) -> str:
        """What tiny heard, or "" when the clip holds no speech."""
        speech = get_speech_timestamps(audio)
        if not speech:
            return ""
        chunks, _ = collect_chunks(audio, speech)
        speech_audio = np.concatenate(chunks)
        with self._lock:
            # The wake word as prompt makes tiny spell the name the same way
            if self.engine == "mlx":
                return self._decode_mlx(speech_audio, wake_word)
            segments, _ = self._model.transcribe(
                speech_audio, beam_size=1, initial_prompt=wake_word, vad_filter=False
            )
            return " ".join(s.text.strip() for s in segments)

    def heard(self, audio: np.ndarray, wake_word: str) -> tuple[bool, str]:
        """(woken, remainder) for one utterance; see match_wake_word."""
        return match_wake_word(self.transcribe(audio, wake_word), wake_word)

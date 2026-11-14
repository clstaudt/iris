"""The user's voice: what the tests say, synthesized once and kept.

The user speaks with a voice of its own (a Kokoro voice unlike Iris's), at
the microphone's rate. A part of an utterance can be a pause, given in
seconds, for a user who stops mid-sentence or thinks aloud.
"""

import hashlib
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf

from iris.tts import SpeechGenerator
from irisdev.voicetest.acoustics import RATE, to_rate

# The user's voice, not one an Iris persona uses by default
USER_ENGINE = "kokoro"
USER_VOICE = "am_michael"
# How loud the user arrives at the microphone (peak); speech at arm's length
LEVEL = 0.5

Part = Union[str, float]


class UserVoice:
    """Synthesizes what the user says, keeping each utterance in a cache."""

    def __init__(self, cache: Path):
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        # A synthesizer for each speed spoken at
        self._generators: dict[float, SpeechGenerator] = {}

    def _speak(self, text: str, speed: float = 1.0) -> np.ndarray:
        named = text if speed == 1.0 else f"{text}@{speed}"
        key = hashlib.sha1(f"{USER_ENGINE}:{USER_VOICE}:{named}".encode()).hexdigest()[:16]
        path = self.cache / f"{key}.wav"
        if not path.exists():
            if speed not in self._generators:
                self._generators[speed] = SpeechGenerator(USER_ENGINE, USER_VOICE, speed)
            generated = self._generators[speed].generate(text, language="en")
            if not generated:
                raise RuntimeError(f"Could not synthesize the user saying: {text}")
            audio, rate = sf.read(generated, dtype="float32")
            Path(generated).unlink(missing_ok=True)
            sf.write(path, to_rate(audio, rate), RATE)
        audio, _ = sf.read(path, dtype="float32")
        return audio

    def utterance(self, parts: list[Part], speed: float = 1.0) -> np.ndarray:
        """The parts said one after the other, a number being a pause; at a
        speed above 1, briskly."""
        pieces = []
        for part in parts:
            if isinstance(part, (int, float)):
                pieces.append(np.zeros(int(part * RATE), dtype=np.float32))
            else:
                pieces.append(_trim(self._speak(part, speed)))
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        return audio * (LEVEL / peak) if peak else audio


def _trim(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Without the silence before and after the speech, so pauses are as given."""
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if len(loud) == 0:
        return audio
    return audio[max(0, loud[0] - RATE // 50) : loud[-1] + RATE // 20]

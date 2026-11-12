"""The user's voice: what the tests say, synthesized once and kept.

The user speaks with a voice of its own (by default a Kokoro voice unlike
Iris's), at the microphone's rate. A part of an utterance can be a pause,
given in seconds, for a user who stops mid-sentence or thinks aloud.

Other users are other voices (VOICES): British and American, an Indian and
a French voice speaking English, a Pocket TTS voice close to Iris's own,
and voices speaking each language Iris speaks: French, Italian, Spanish,
Portuguese, Japanese, Chinese, Hindi, German and Dutch. Kokoro is called
directly with the language to speak, so a French voice can speak English
with its accent; Spanish, Portuguese and Hindi, which kokoro-onnx does not
take by name, it speaks from espeak's phonemes, as Iris does. German and
Dutch are spoken by Pocket TTS, cloning a voice from a prompt (not Pocket's
default voices, which Iris uses).

Said as pieces, a sentence with pauses ends each piece with a sentence's
fall in pitch. Spliced, it is synthesized whole and the pauses are cut into
it where its pieces meet, found by Whisper's word timestamps: the user then
stops mid-sentence and the voice says so.
"""

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from kokoro_onnx.config import SUPPORTED_LANGUAGES as KOKORO_ONNX_LANGUAGES
from pocket_tts import TTSModel

from iris.tts import KOKORO_MODEL_FILE, KOKORO_VOICES_FILE, POCKET_LANGUAGES, find_kokoro_models
from irisdev.voicetest.acoustics import RATE, in_room, room, to_rate

logger = logging.getLogger(__name__)

# The user's voice, not one an Iris persona uses by default
USER_ENGINE = "kokoro"
USER_VOICE = "am_michael"
# How loud the user arrives at the microphone (peak); speech at arm's length
LEVEL = 0.5
# Kokoro's language codes by Whisper's
KOKORO_LANGUAGES = {
    "en": "en-us",
    "fr": "fr-fr",
    "it": "it",
    "es": "es",
    "ja": "ja",
    "zh": "cmn",
    "pt": "pt-br",
    "hi": "hi",
}
# Pocket TTS clones a voice from this, said by a Kokoro voice, for German and Dutch
CLONE_PROMPT = "Hello, I am reading this short passage so my voice can be recognized clearly."
# Pocket TTS's own voices (each speaks every language it has) that stand in
# for a woman's and a man's cloned voice where it cannot clone
CLONE_STAND_INS = {"f": "eve", "m": "michael"}
# Whisper for finding where the pieces of a spliced sentence meet
SPLICE_MODEL = "base"

Part = Union[str, float]

# The voices the scenarios speak with: engine, voice and language
VOICES = {
    "michael": {"engine": "kokoro", "voice": "am_michael", "language": "en"},
    "emma": {"engine": "kokoro", "voice": "bf_emma", "language": "en"},
    "nicole": {"engine": "kokoro", "voice": "af_nicole", "language": "en"},
    "indian_en": {"engine": "kokoro", "voice": "hf_alpha", "language": "en"},
    "french_en": {"engine": "kokoro", "voice": "ff_siwis", "language": "en"},
    "pocket_female": {"engine": "kyutai", "voice": "anna", "language": "en"},
    "fr": {"engine": "kokoro", "voice": "ff_siwis", "language": "fr"},
    "it": {"engine": "kokoro", "voice": "im_nicola", "language": "it"},
    "es": {"engine": "kokoro", "voice": "em_alex", "language": "es"},
    "ja": {"engine": "kokoro", "voice": "jm_kumo", "language": "ja"},
    "pt": {"engine": "kokoro", "voice": "pm_alex", "language": "pt"},
    "zh": {"engine": "kokoro", "voice": "zf_xiaobei", "language": "zh"},
    "hi": {"engine": "kokoro", "voice": "hm_omega", "language": "hi"},
    "de": {"engine": "kyutai", "voice": "clone:af_nicole", "language": "de"},
    "nl": {"engine": "kyutai", "voice": "clone:am_adam", "language": "nl"},
    # Another person in the room, for asides not from the user
    "heart": {"engine": "kokoro", "voice": "af_heart", "language": "en"},
}


def kokoro_language(voice: str, language: str) -> str:
    """Kokoro's code for speaking a language; British voices speak British
    English."""
    if language == "en" and voice.startswith("b"):
        return "en-gb"
    return KOKORO_LANGUAGES.get(language, "en-us")


class UserVoice:
    """Synthesizes what the user says, keeping each utterance in a cache."""

    # Engines are shared between voices: each loads once
    _kokoro: Optional[Kokoro] = None
    _pocket: dict = {}
    _whisper: Optional[WhisperModel] = None

    def __init__(self, cache: Path, engine: str = USER_ENGINE, voice: str = USER_VOICE,
                 language: str = "en", speed: float = 1.0):
        """
        Args:
            cache: Where utterances are kept
            engine: "kokoro" or "kyutai" (Pocket TTS)
            voice: A voice of that engine; for Pocket TTS "clone:<kokoro
                voice>" clones the voice of a Kokoro voice
            language: What it speaks (Whisper's code)
            speed: How fast it speaks
        """
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.voice = voice
        self.language = language
        self.speed = speed

    @classmethod
    def named(cls, cache: Path, name: str, speed: float = 1.0) -> "UserVoice":
        """One of VOICES."""
        if name not in VOICES:
            raise ValueError(f"No voice called {name!r}; there are {', '.join(VOICES)}")
        return cls(cache, speed=speed, **VOICES[name])

    def _key(self, text: str) -> str:
        spec = f"{self.engine}:{self._spoken_voice()}:{self.language}:{self.speed}:{text}"
        return hashlib.sha1(spec.encode()).hexdigest()[:16]

    def _spoken_voice(self) -> str:
        """The voice the user speaks with: a cloned one is Pocket TTS's own
        voice for the language where cloning cannot be had (see _pocket_voice)."""
        if self.engine == "kyutai" and self.voice.startswith("clone:"):
            return self._pocket_voice()[2]
        return self.voice

    def _speak(self, text: str) -> np.ndarray:
        path = self.cache / f"{self._key(text)}.wav"
        if not path.exists():
            audio, rate = self._synthesize(text)
            if audio is None or not len(audio):
                raise RuntimeError(f"Could not synthesize the user saying: {text}")
            sf.write(path, to_rate(audio, rate), RATE)
        audio, _ = sf.read(path, dtype="float32")
        return audio

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if self.engine == "kokoro":
            kokoro, code = self._kokoro_engine(), kokoro_language(self.voice, self.language)
            if code in KOKORO_ONNX_LANGUAGES:
                audio, rate = kokoro.create(text, voice=self.voice, speed=self.speed, lang=code)
            else:
                # Named only to pass kokoro-onnx's check: the phonemes say it all
                audio, rate = kokoro.create(text, voice=self.voice, speed=self.speed, lang="en-us",
                                            phonemes=kokoro.tokenizer.phonemize(text, code))
            return np.asarray(audio, dtype=np.float32), int(rate)
        if self.engine == "kyutai":
            model, state, _ = self._pocket_voice()
            return model.generate_audio(state, text).numpy().astype(np.float32), int(model.sample_rate)
        raise ValueError(f"The user cannot speak with the engine {self.engine!r}")

    @classmethod
    def _kokoro_engine(cls) -> Kokoro:
        if cls._kokoro is None:
            folder = find_kokoro_models()
            if folder is None:
                raise RuntimeError("The Kokoro voice model is not on this computer")
            cls._kokoro = Kokoro(str(folder / KOKORO_MODEL_FILE), str(folder / KOKORO_VOICES_FILE))
        return cls._kokoro

    def _pocket_voice(self):
        """Pocket TTS for the language, the voice's state, and the voice it
        is: a named voice, or one cloned from a Kokoro voice reading
        CLONE_PROMPT. Pocket TTS clones only with weights that are gated on
        Hugging Face; without them, a cloned voice is one of Pocket TTS's own
        voices instead (CLONE_STAND_INS), never the one Iris answers in."""
        language = POCKET_LANGUAGES.get(self.language, "english")
        key = (language, self.voice)
        if key not in self._pocket:
            model = TTSModel.load_model(language=language)
            voice = prompt = self.voice
            if voice.startswith("clone:") and not getattr(model, "has_voice_cloning", True):
                voice = prompt = CLONE_STAND_INS[voice.split(":", 1)[1][1:2]]
                logger.warning("Pocket TTS cannot clone voices here: the user speaks %s as %s", language, voice)
            elif voice.startswith("clone:"):
                speaker = UserVoice(self.cache, "kokoro", self.voice.split(":", 1)[1], "en")
                prompt = str(self.cache / f"prompt-{speaker._key(CLONE_PROMPT)}.wav")
                if not Path(prompt).exists():
                    sf.write(prompt, speaker._speak(CLONE_PROMPT), RATE)
            try:
                state = model.get_state_for_audio_prompt(prompt)
            except Exception as error:
                raise RuntimeError(f"Pocket TTS cannot speak with the voice {self.voice!r} here: {error}")
            self._pocket[key] = (model, state, voice)
        return self._pocket[key]

    def utterance(self, parts: list[Part], level: Optional[float] = None,
                  room_response: Optional[tuple] = None, splice: bool = False,
                  unsaid: Optional[list[str]] = None) -> np.ndarray:
        """The parts said one after the other, a number being a pause.

        Args:
            level: The peak it arrives at (LEVEL by default)
            room_response: (rt60, drr): said across a room
            splice: Synthesize the sentence whole and cut the pauses into it
            unsaid: Words that end the sentence but are not said: it is
                synthesized whole and cut off where they begin
        """
        texts = [p for p in parts if isinstance(p, str)]
        if unsaid:
            audio = self._spliced(list(parts) + [0.0] + list(unsaid), keep=len(texts))
        elif splice and len(texts) > 1:
            audio = self._spliced(parts)
        else:
            pieces = []
            for part in parts:
                if isinstance(part, (int, float)):
                    pieces.append(np.zeros(int(part * RATE), dtype=np.float32))
                else:
                    pieces.append(_trim(self._speak(part)))
            audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        if room_response:
            audio = in_room(audio, room(*room_response))
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        level = LEVEL if level is None else level
        return (audio * (level / peak)).astype(np.float32) if peak else audio

    def _spliced(self, parts: list[Part], keep: Optional[int] = None) -> np.ndarray:
        """The parts synthesized as one sentence, with the pauses cut in
        where the parts meet; kept with the utterance. Given keep, only so
        many text parts are kept, and the rest is cut off."""
        spec = "|".join(str(p) for p in parts) + (f"|keep {keep}" if keep is not None else "")
        path = self.cache / f"splice-{self._key(spec)}.wav"
        if path.exists():
            audio, _ = sf.read(path, dtype="float32")
            return audio
        texts = [p for p in parts if isinstance(p, str)]
        whole = _trim(self._speak(" ".join(t.strip() for t in texts)))
        cuts = self._boundaries(whole, [len(t.split()) for t in texts])
        pieces, start, cut_index = [], 0, 0
        for index, part in enumerate(parts):
            if isinstance(part, str):
                continue
            if cut_index < len(cuts) and any(isinstance(p, str) for p in parts[:index]):
                cut = cuts[cut_index]
                pieces.append(whole[start:cut])
                start = cut
                cut_index += 1
                if keep is not None and cut_index >= keep:
                    break
            pieces.append(np.zeros(int(part * RATE), dtype=np.float32))
        else:
            pieces.append(whole[start:])
        audio = _trim(np.concatenate(pieces).astype(np.float32)) if keep is not None else (
            np.concatenate(pieces).astype(np.float32))
        sf.write(path, audio, RATE)
        return audio

    def _boundaries(self, audio: np.ndarray, counts: list[int]) -> list[int]:
        """Sample positions between the parts: midway between the last word
        of one and the first of the next, by Whisper's word timestamps."""
        if UserVoice._whisper is None:
            UserVoice._whisper = WhisperModel(SPLICE_MODEL, device="cpu", compute_type="int8")
        with tempfile.NamedTemporaryFile(suffix=".wav") as file:
            sf.write(file.name, audio, RATE)
            segments, _ = UserVoice._whisper.transcribe(file.name, word_timestamps=True,
                                                        language=self.language)
            found = [w for segment in segments for w in (segment.words or [])]
        if not found:
            raise RuntimeError("Whisper found no words to splice at")
        total = sum(counts)
        cuts, said = [], 0
        for count in counts[:-1]:
            said += count
            index = min(len(found) - 1, max(1, round(said / total * len(found)))) - 1
            end = found[index].end
            following = found[index + 1].start if index + 1 < len(found) else end
            cuts.append(int(round((end + following) / 2 * RATE)))
        return cuts


def _trim(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Without the silence before and after the speech, so pauses are as given."""
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if len(loud) == 0:
        return audio
    return audio[max(0, loud[0] - RATE // 50) : loud[-1] + RATE // 20]

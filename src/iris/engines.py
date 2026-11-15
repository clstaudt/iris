"""The model slots Iris runs, and the engines that can fill them.

Speech synthesis and speech recognition are slots. Iris ships engines for
them (Kyutai's Pocket TTS and Kokoro; Whisper), and plugins can register
more. An engine registers under an id; personas and settings name it by
that id.

This module only keeps the registries. The engines live with their code
(iris.tts, iris.transcription) or in plugins, which register them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Where an engine registered by Iris itself comes from
BUILT_IN = "built-in"


@dataclass(frozen=True)
class Step:
    """A step of making a voice ready to speak a language, as the app shows
    it: "download" with what is downloaded ("model", the language's or the
    engine's model; "voice", the voice's own files for it; "cloning", the
    model that makes a cloned voice), or "make" (a cloned voice made for the
    language from its sample)."""

    stage: str
    what: str = ""


class SpeechEngine:
    """Turns text into speech. Subclass it and register the class.

    The speech generator makes one instance per engine and sets its voice
    (and speed, where the engine takes one) before each piece of text.
    """

    #: Key personas name the engine by, e.g. "kyutai"
    id = ""
    #: Shown in the app, e.g. "Kyutai"
    name = ""
    #: Whisper language codes it speaks, with one voice or another
    languages: frozenset[str] = frozenset()
    #: Whether its voices take a speed
    takes_speed = False

    voice = ""
    speed = 1.0

    def __init__(self, models_dir: str | Path | None = None):
        """
        Args:
            models_dir: Folder of model files chosen by the user, for engines
                that load their models from one
        """

    @classmethod
    def available(cls, models_dir: str | Path | None = None) -> bool:
        """Whether it can speak now (its models are in place)."""
        return True

    @classmethod
    def voices(cls, models_dir: str | Path | None = None) -> list[dict]:
        """Its voices, as {"id", "name", "language"} (the language a voice
        was made for, as shown). "languages", those it speaks, may be left
        out: voice_languages() tells them."""
        return []

    @classmethod
    def voice_language(cls, voice: str) -> str:
        """The Whisper code of the language a voice was made for."""
        return "en"

    @classmethod
    def voice_languages(cls, voice: str) -> frozenset[str]:
        """The languages a voice speaks properly: natively, correctly
        pronounced. A persona's replies in any other language are shown and
        not spoken; no other voice speaks them in its place. Every voice
        speaks all of the engine's languages, unless the engine says
        otherwise."""
        return cls.languages

    @classmethod
    def ready(cls, voice: str, language: str, models_dir: str | Path | None = None) -> bool:
        """Whether the voice can speak the language now, with nothing to
        download or make first. Speaking never fetches what is missing: a
        reply in a language the voice is not ready for is shown, while
        make_ready() gets it, with progress."""
        return cls.available(models_dir)

    @classmethod
    def make_ready(
        cls,
        voice: str,
        language: str,
        models_dir: str | Path | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        on_step: Callable[[Step], None] | None = None,
    ):
        """Download, and make, what speaking the language in the voice needs.
        on_step is told each step as it starts, on_progress the bytes of a
        download (done, total). Raises when it cannot be had."""

    def prepare(self):
        """Load what its first reply needs, when it is the persona's engine.

        What another language needs may load when it is first spoken.
        """

    def generate(self, text: str, language: str = "en") -> Optional[str]:
        """Speak text in the current voice into a WAV file.

        Returns:
            The file's path, or None if speech could not be generated
        """
        raise NotImplementedError


@dataclass(frozen=True)
class RecognitionEngine:
    """Turns recorded speech into text.

    create(options) returns the recognizer: an object whose
    transcribe(audio_path) returns (text, Whisper language code). The options
    are the recognition settings, "model", "device" and "engine" (for
    Whisper: its backend), which an engine may ignore. The recognizer may
    also have model_name, device and model_path for the app to show.
    """

    id: str
    name: str
    create: Callable[[dict], object]
    source: str = BUILT_IN


_speech: dict[str, type[SpeechEngine]] = {}
_speech_sources: dict[str, str] = {}
_recognition: dict[str, RecognitionEngine] = {}


def register_speech_engine(engine: type[SpeechEngine], source: str = BUILT_IN):
    """Make a speech engine available to personas under its id."""
    if not engine.id:
        raise ValueError(f"{engine.__name__} has no id")
    if _speech.get(engine.id, engine) is not engine:
        raise ValueError(f"A speech engine '{engine.id}' is already registered")
    _speech[engine.id] = engine
    _speech_sources[engine.id] = source


def speech_engines() -> dict[str, type[SpeechEngine]]:
    """The registered speech engines by id, in the order they came."""
    return dict(_speech)


def speech_engine(engine_id: str) -> type[SpeechEngine]:
    """The speech engine of this id.

    Raises:
        ValueError: If none is registered under it
    """
    try:
        return _speech[engine_id]
    except KeyError:
        raise ValueError(f"No speech engine '{engine_id}'") from None


def speech_languages() -> frozenset[str]:
    """The languages Iris talks in, as Whisper codes: those any of its voices
    speaks. A conversation follows the user into each of them, whatever the
    persona's voice speaks."""
    return frozenset().union(*(engine.languages for engine in _speech.values()))


def voice_speaks(engine_id: str, voice: str, language: str) -> bool:
    """Whether a voice of this engine speaks a language properly."""
    engine = _speech.get(engine_id)
    return engine is not None and language in engine.voice_languages(voice)


def register_recognition_engine(engine: RecognitionEngine):
    """Make a speech recognition engine available under its id."""
    if not engine.id:
        raise ValueError("The recognition engine has no id")
    if engine.id in _recognition and _recognition[engine.id] != engine:
        raise ValueError(f"A recognition engine '{engine.id}' is already registered")
    _recognition[engine.id] = engine


def recognition_engines() -> dict[str, RecognitionEngine]:
    """The registered recognition engines by id, in the order they came."""
    return dict(_recognition)


def recognition_engine(engine_id: str) -> RecognitionEngine:
    """The recognition engine of this id.

    Raises:
        ValueError: If none is registered under it
    """
    try:
        return _recognition[engine_id]
    except KeyError:
        raise ValueError(f"No recognition engine '{engine_id}'") from None


def describe(
    speech: Optional[str] = None,
    recognition: Optional[str] = None,
    models_dir: str | Path | None = None,
) -> dict[str, list[dict]]:
    """Each slot's engines for the app: {"stt": [...], "tts": [...]}, each
    engine as {id, label, active, available, source}.

    Args:
        speech: The speech engine in use
        recognition: The recognition engine in use
        models_dir: The voice model folder chosen by the user
    """
    return {
        "stt": [
            {
                "id": engine.id,
                "label": engine.name,
                "active": engine.id == recognition,
                "available": True,
                "source": engine.source,
            }
            for engine in _recognition.values()
        ],
        "tts": [
            {
                "id": engine.id,
                "label": engine.name,
                "active": engine.id == speech,
                "available": engine.available(models_dir),
                "source": _speech_sources.get(engine.id, BUILT_IN),
            }
            for engine in _speech.values()
        ],
    }

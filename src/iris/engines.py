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


class SpeechEngine:
    """Turns text into speech. Subclass it and register the class.

    The speech generator makes one instance per engine and sets its voice
    (and speed, where the engine takes one) before each piece of text.
    """

    #: Key personas name the engine by, e.g. "kyutai"
    id = ""
    #: Shown in the app, e.g. "Kyutai"
    name = ""
    #: Whisper language codes it speaks
    languages: frozenset[str] = frozenset()
    #: Whether its voices take a speed
    takes_speed = False
    #: Engines whose voices it borrows for languages it lacks: a reply in
    #: such a language is spoken by the first of them that speaks it
    borrows: tuple[str, ...] = ()

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
        """Its voices, as {"id", "name", "language"} (the language as shown)."""
        return []

    @classmethod
    def voice_language(cls, voice: str) -> str:
        """The Whisper code of the language a voice was made for."""
        return "en"

    @classmethod
    def default_voice(cls, language: str) -> Optional[str]:
        """Its voice for a language, used when another engine borrows it."""
        return None

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


def speaking_engine(engine_id: str, language: str) -> Optional[tuple[str, Optional[str]]]:
    """Who speaks a reply in this language for a persona on this engine.

    Returns:
        (engine_id, None) when the engine speaks the language itself,
        (other_id, voice) when it borrows another engine's voice for it, or
        None when neither can
    """
    engine = _speech.get(engine_id)
    if engine is None:
        return None
    if language in engine.languages:
        return engine_id, None
    for other_id in engine.borrows:
        other = _speech.get(other_id)
        if other is not None and language in other.languages:
            voice = other.default_voice(language)
            if voice:
                return other_id, voice
    return None


def spoken_languages(engine_id: str) -> frozenset[str]:
    """The languages a persona on this engine speaks, with its own voice or
    one it borrows."""
    engine = _speech.get(engine_id)
    if engine is None:
        return frozenset()
    candidates = set(engine.languages)
    for other_id in engine.borrows:
        other = _speech.get(other_id)
        if other is not None:
            candidates |= other.languages
    return frozenset(lang for lang in candidates if speaking_engine(engine_id, lang))


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

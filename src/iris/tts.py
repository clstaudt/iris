"""Text-to-speech generation and pipeline using Pocket TTS or kokoro-onnx."""

import logging
import os
import queue
import re
import tempfile
import threading
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro_onnx import Kokoro
from kokoro_onnx import tokenizer as kokoro_tokenizer
from phonemizer.backend import EspeakBackend
from phonemizer.separator import default_separator
from phonemizer.utils import list2str, str2list
from pocket_tts import TTSModel
from pocket_tts.default_parameters import DEFAULT_VOICE_FOR_LANGUAGE

from iris.audio import get_filter
from iris.engines import (
    SpeechEngine,
    register_speech_engine,
    speaking_engine,
    speech_engine,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class KeptEspeak:
    """Stands in for the phonemizer module in Kokoro's tokenizer.

    phonemizer.phonemize() loads espeak afresh on every call, which takes
    about 1.5 s on an M3: three times what the voice model needs for a
    sentence. This phonemizes the same way, with one espeak per language,
    loaded once and kept.
    """

    def __init__(self):
        self._backends: dict[tuple, EspeakBackend] = {}
        self._lock = threading.Lock()

    def phonemize(
        self, text: str, language: str, preserve_punctuation: bool, with_stress: bool
    ) -> str:
        with self._lock:
            key = (language, preserve_punctuation, with_stress)
            backend = self._backends.get(key)
            if backend is None:
                backend = EspeakBackend(
                    language, preserve_punctuation=preserve_punctuation, with_stress=with_stress
                )
                self._backends[key] = backend
            lines = [line.strip(os.linesep) for line in str2list(text)]
            lines = [line for line in lines if line.strip()]
            if not lines:
                return ""
            return list2str(
                backend.phonemize(lines, separator=default_separator, strip=False, njobs=1)
            )


kokoro_tokenizer.phonemizer = KeptEspeak()

# Kokoro model download URLs
KOKORO_MODEL_URL = (
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"
)

# Mapping from Whisper language codes to Kokoro TTS language/voice settings.
# Kokoro supports: en-us, en-gb, fr-fr, it, ja, cmn
# A Kokoro voice name starts with its language ("af_bella": American English,
# "bm_lewis": British English, "ff_siwis": French...). A persona keeps its own
# voice for languages that voice speaks, and borrows the default otherwise.
LANGUAGE_VOICE_MAP = {
    "en": {"code": "en-us", "default_voice": "af_bella", "voice_prefixes": "ab"},
    "fr": {"code": "fr-fr", "default_voice": "ff_siwis", "voice_prefixes": "f"},
    "it": {"code": "it", "default_voice": "if_sara", "voice_prefixes": "i"},
    "ja": {"code": "ja", "default_voice": "jf_alpha", "voice_prefixes": "j"},
    "zh": {"code": "cmn", "default_voice": "zf_xiaoni", "voice_prefixes": "z"},
}


def get_language_info(whisper_lang: str) -> dict | None:
    """Get Kokoro language info for a Whisper language code.

    Args:
        whisper_lang: Whisper language code (e.g. 'en', 'fr', 'ja', 'zh')

    Returns:
        Dict with 'code' and 'default_voice', or None if unsupported.
    """
    return LANGUAGE_VOICE_MAP.get(whisper_lang)


def is_language_supported(whisper_lang: str) -> bool:
    """Check if a Whisper language code is supported by Kokoro TTS."""
    return whisper_lang in LANGUAGE_VOICE_MAP


KOKORO_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"
# Where Iris keeps the voice model when it downloads it
DEFAULT_KOKORO_DIR = Path.home() / ".iris" / "models"


def kokoro_files_in(folder: Path) -> bool:
    """Whether a folder holds both Kokoro files."""
    return (folder / KOKORO_MODEL_FILE).exists() and (folder / KOKORO_VOICES_FILE).exists()


def find_kokoro_models() -> Path | None:
    """The first of the usual places that holds the Kokoro files, if any."""
    candidates = []
    if env_dir := os.environ.get("IRIS_MODELS_DIR"):
        candidates.append(Path(env_dir))
    candidates += [
        DEFAULT_KOKORO_DIR,
        Path.home() / ".local" / "share" / "iris" / "models",
        Path.cwd() / "models",
    ]
    return next((folder for folder in candidates if kokoro_files_in(folder)), None)


class KokoroGenerator(SpeechEngine):
    """Generates speech audio using kokoro-onnx."""

    id = "kokoro"
    name = "Kokoro"
    languages = frozenset(LANGUAGE_VOICE_MAP)
    takes_speed = True

    def __init__(
        self,
        voice: str = "af_bella",
        speed: float = 1.25,
        models_dir: str | Path | None = None,
    ):
        """
        Initialize kokoro TTS generator.

        Args:
            voice: Voice to use for generation
            speed: Speech speed multiplier
            models_dir: Folder holding the Kokoro files; None looks in the
                usual places (IRIS_MODELS_DIR, ~/.iris/models, ...)
        """
        self.voice = voice
        self.speed = speed

        # A folder chosen by the user, or the usual locations
        self.models_dir = Path(models_dir).expanduser() if models_dir else None
        self.model_path, self.voices_path = self._locate_models()

        # Initialize Kokoro
        self.kokoro = Kokoro(model_path=str(self.model_path), voices_path=str(self.voices_path))

    def _locate_models(self) -> tuple[Path, Path]:
        """The Kokoro files, in the chosen folder or the first of the usual
        places that holds them.

        Raises:
            FileNotFoundError: If they are not there. The app downloads them
                during setup.
        """
        folder = self.models_dir or find_kokoro_models()
        if folder is None or not kokoro_files_in(folder):
            raise FileNotFoundError(
                f"{folder or DEFAULT_KOKORO_DIR} does not hold "
                f"{KOKORO_MODEL_FILE} and {KOKORO_VOICES_FILE}"
            )
        return folder / KOKORO_MODEL_FILE, folder / KOKORO_VOICES_FILE

    @classmethod
    def available(cls, models_dir: str | Path | None = None) -> bool:
        """Kokoro speaks once its files are downloaded (or chosen)."""
        folder = Path(models_dir).expanduser() if models_dir else find_kokoro_models()
        return folder is not None and kokoro_files_in(folder)

    @classmethod
    def voices(cls, models_dir: str | Path | None = None) -> list[dict]:
        """Kokoro's voices come from its voices file, so there are none
        until it is downloaded."""
        folder = Path(models_dir).expanduser() if models_dir else find_kokoro_models()
        if folder is None or not (folder / KOKORO_VOICES_FILE).exists():
            return []
        with np.load(folder / KOKORO_VOICES_FILE) as voices:
            names = sorted(voices.files)
        return [
            {
                "id": name,
                "name": name.split("_", 1)[-1].replace("_", " ").title(),
                "language": KOKORO_VOICE_LANGUAGES.get(name[:1], ("", "Other"))[1],
            }
            for name in names
        ]

    @classmethod
    def voice_language(cls, voice: str) -> str:
        """A Kokoro voice name starts with its language."""
        return KOKORO_VOICE_LANGUAGES.get(voice[:1], ("en",))[0]

    @classmethod
    def default_voice(cls, language: str) -> str | None:
        info = get_language_info(language)
        return info["default_voice"] if info else None

    def voice_for(self, language: str) -> tuple[str, str]:
        """Pick (Kokoro language code, voice) for a reply in this language.

        The persona's voice is used whenever it speaks the language, so
        personas sound different from each other; for a language it cannot
        speak, that language's default voice takes over.
        """
        lang_info = get_language_info(language)
        if lang_info is None:
            # Unsupported language – fall back to English with the own voice
            kokoro_lang, voice = "en-us", self.voice
        elif self.voice[:1] in lang_info["voice_prefixes"]:
            kokoro_lang, voice = lang_info["code"], self.voice
        else:
            kokoro_lang, voice = lang_info["code"], lang_info["default_voice"]
        # British voices are given British pronunciation
        if kokoro_lang == "en-us" and voice.startswith("b"):
            kokoro_lang = "en-gb"
        return kokoro_lang, voice

    def generate(self, text: str, language: str = "en") -> str | None:
        """
        Generate audio file from text using kokoro-onnx Python API.

        Args:
            text: Text to convert to speech
            language: Whisper language code (e.g. 'en', 'fr', 'ja').
                Selects the Kokoro language and an appropriate voice.
                Falls back to English if unsupported.

        Returns:
            Path to generated WAV file, or None if generation failed
        """
        kokoro_lang, voice = self.voice_for(language)

        temp_audio = None
        try:
            # Create temporary file for output
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio.close()

            # Generate audio using kokoro-onnx Python API
            audio_data, sample_rate = self.kokoro.create(
                text=text, voice=voice, speed=self.speed, lang=kokoro_lang
            )

            # Save audio to file
            sf.write(temp_audio.name, audio_data, sample_rate)

            # Verify file was created and has content
            if not Path(temp_audio.name).exists():
                raise RuntimeError("Failed to create output file")

            file_size = Path(temp_audio.name).stat().st_size
            if file_size == 0:
                raise RuntimeError("Created empty audio file")

            return temp_audio.name

        except Exception:
            logger.exception("Kokoro could not speak")
            return None


def heard_part(text: str, fraction: float) -> str:
    """The words of a spoken text that were heard out by the time this
    fraction of its audio had played (judged by length: speech goes through
    a text at a fairly even pace)."""
    end = int(len(text) * max(0.0, min(1.0, fraction)))
    if end >= len(text):
        return text
    if not text[end].isspace():
        # Cut in the middle of a word, which was not heard out
        end = max(text.rfind(" ", 0, end), text.rfind("\n", 0, end))
        if end < 0:
            return ""
    return text[:end].rstrip()


# Whisper language codes Pocket TTS speaks, and the model for each. Japanese
# and Chinese are left to Kokoro.
POCKET_LANGUAGES = {
    "en": "english",
    "fr": "french",
    "it": "italian",
    "de": "german",
    "es": "spanish",
    "pt": "portuguese",
    "nl": "dutch",
}
# Pocket TTS's English voices; each further language has one voice of its own
POCKET_ENGLISH_VOICES = (
    "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette",
    "eponine", "eve", "fantine", "george", "jane", "jean", "javert", "marius",
    "mary", "michael", "paul", "peter_yearsley", "stuart_bell", "vera",
)  # fmt: skip
# A Kokoro voice name starts with its language, then f or m for female or male
KOKORO_VOICE_NAME = re.compile(r"[a-z][fm]_\w+")
KOKORO_VOICE_LANGUAGES = {
    "a": ("en", "American English"),
    "b": ("en", "British English"),
    "e": ("es", "Spanish"),
    "f": ("fr", "French"),
    "h": ("hi", "Hindi"),
    "i": ("it", "Italian"),
    "j": ("ja", "Japanese"),
    "p": ("pt", "Portuguese"),
    "z": ("zh", "Chinese"),
}
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "nl": "Dutch",
}


def engine_speaks(engine: str, language: str) -> bool:
    """Whether a persona on this engine can speak its replies in a language,
    with its own voice or one it borrows (Kyutai personas leave the
    languages Pocket TTS lacks to Kokoro)."""
    return speaking_engine(engine, language) is not None


def voice_language(engine: str, voice: str) -> str:
    """The Whisper language code of the language a voice was made for."""
    try:
        return speech_engine(engine).voice_language(voice)
    except ValueError:
        return "en"


def list_voices(engine: str, models_dir: str | Path | None = None) -> list[dict]:
    """An engine's voices as {id, name, language}."""
    return speech_engine(engine).voices(models_dir)


# Characters Pocket TTS stumbles over: the word after a dash is garbled, and
# words joined by a non-breaking hyphen are dropped
POCKET_REPLACEMENTS = (
    (re.compile(r"\s*\u2014\s*|\s+\u2013\s+"), ", "),
    (re.compile(r"[\u2010\u2011\u2013]"), "-"),
)
SENTENCE_END = re.compile(r"(?<=[.!?\u2026])\s+")
# A period that ends an abbreviation, not a sentence: "z. B.", "e.g.", "Dr."
ABBREVIATION_END = re.compile(
    r"(?:^|[\s.])(?:\w|Dr|Mr|Mrs|Ms|Prof|St|Nr|ca|vs|etc)\.$", re.IGNORECASE
)


def pocket_sentences(text: str) -> list[str]:
    """Text as Pocket TTS takes it best: one sentence at a time.

    Its models are trained on single sentences. Given several at once, they
    often stop after the first and drop the rest, the German model especially.
    """
    for pattern, replacement in POCKET_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    sentences: list[str] = []
    for part in SENTENCE_END.split(text.strip()):
        part = part.strip(" ,")
        if not part:
            continue
        # After an abbreviation, or before lower case, the sentence goes on
        if sentences and (part[0].islower() or ABBREVIATION_END.search(sentences[-1])):
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return sentences


class PocketGenerator(SpeechEngine):
    """Generates speech audio with Kyutai's Pocket TTS, on the CPU.

    Pocket TTS has one model per language: English loads right away, the
    others the first time a reply is in that language.
    """

    id = "kyutai"
    name = "Kyutai"
    languages = frozenset(POCKET_LANGUAGES)
    # Japanese and Chinese are left to Kokoro
    borrows = ("kokoro",)

    def __init__(self, voice: str = "alba", models_dir: str | Path | None = None):
        """
        Args:
            voice: A Pocket TTS voice, or an audio file of a voice to clone
            models_dir: Unused: Pocket TTS fetches its models itself
        """
        self.voice = voice
        self._models: dict[str, TTSModel] = {}
        self._voice_states: dict[tuple[str, str], object] = {}
        self._model("english")

    def _model(self, pocket_lang: str) -> TTSModel:
        if pocket_lang not in self._models:
            self._models[pocket_lang] = TTSModel.load_model(language=pocket_lang)
        return self._models[pocket_lang]

    def _voice_state(self, pocket_lang: str, voice: str):
        key = (pocket_lang, voice)
        if key not in self._voice_states:
            model = self._model(pocket_lang)
            self._voice_states[key] = model.get_state_for_audio_prompt(voice)
        return self._voice_states[key]

    @classmethod
    def voices(cls, models_dir: str | Path | None = None) -> list[dict]:
        names = list(POCKET_ENGLISH_VOICES) + [
            DEFAULT_VOICE_FOR_LANGUAGE[pocket_lang]
            for code, pocket_lang in POCKET_LANGUAGES.items()
            if code != "en"
        ]
        return [
            {
                "id": name,
                "name": name.replace("_", " ").title(),
                "language": LANGUAGE_NAMES[cls.voice_language(name)],
            }
            for name in names
        ]

    @classmethod
    def voice_language(cls, voice: str) -> str:
        """Each language beyond English has a voice of its own."""
        for code, pocket_lang in POCKET_LANGUAGES.items():
            if DEFAULT_VOICE_FOR_LANGUAGE.get(pocket_lang) == voice:
                return code
        return "en"

    @classmethod
    def default_voice(cls, language: str) -> str | None:
        if language == "en":
            return "alba"
        pocket_lang = POCKET_LANGUAGES.get(language)
        return DEFAULT_VOICE_FOR_LANGUAGE.get(pocket_lang) if pocket_lang else None

    def voice_for(self, language: str) -> tuple[str, str]:
        """Pick (Pocket model language, voice) for a reply in this language.

        English uses the voice itself; another language uses the voice made
        for it, unless the voice is cloned from an audio file.
        """
        pocket_lang = POCKET_LANGUAGES.get(language, "english")
        voice = self.voice
        if pocket_lang != "english" and not Path(voice).exists():
            voice = DEFAULT_VOICE_FOR_LANGUAGE[pocket_lang]
        return pocket_lang, voice

    def generate(self, text: str, language: str = "en") -> str | None:
        """
        Generate a WAV file from text.

        Args:
            text: Text to convert to speech
            language: Whisper language code (e.g. 'en', 'fr')

        Returns:
            Path to generated WAV file, or None if generation failed
        """
        try:
            pocket_lang, voice = self.voice_for(language)
            model = self._model(pocket_lang)
            state = self._voice_state(pocket_lang, voice)
            sentences = pocket_sentences(text)
            if not sentences:
                return None
            audio = torch.cat([model.generate_audio(state, sentence) for sentence in sentences])

            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio.close()
            sf.write(temp_audio.name, audio.numpy(), model.sample_rate)
            return temp_audio.name

        except Exception:
            logger.exception("Pocket TTS could not speak")
            return None


register_speech_engine(PocketGenerator)
register_speech_engine(KokoroGenerator)


class SpeechGenerator:
    """Speaks with the persona's engine and voice.

    The engines are those registered in iris.engines: Kyutai's Pocket TTS
    and Kokoro, and any a plugin adds. Each loads the first time it is
    needed. A reply in a language the persona's engine lacks is spoken by an
    engine it borrows from (Kyutai personas by Kokoro, for Japanese and
    Chinese).
    """

    def __init__(
        self,
        engine: str = "kyutai",
        voice: str = "alba",
        speed: float = 1.0,
        models_dir: str | Path | None = None,
    ):
        """
        Args:
            engine: Id of a registered speech engine
            voice: A voice of that engine
            speed: Speech speed multiplier, for engines that take one
            models_dir: Voice model folder chosen by the user (Kokoro's)
        """
        self.engine = engine
        self.voice = voice
        self.speed = speed
        self.models_dir = Path(models_dir).expanduser() if models_dir else None
        self._engines: dict[str, SpeechEngine] = {}
        # Replies and previews may ask for speech at the same time
        self._lock = threading.Lock()
        self.prepare()

    def speaks(self, language: str) -> bool:
        """Whether replies in this language can be spoken in it."""
        return engine_speaks(self.engine, language)

    def prepare(self):
        """Load the current engine now, so its first reply is not delayed."""
        with self._lock:
            self._engine(self.engine)

    def _engine(self, engine: str) -> SpeechEngine:
        if engine not in self._engines:
            self._engines[engine] = speech_engine(engine)(models_dir=self.models_dir)
        return self._engines[engine]

    def generate(
        self,
        text: str,
        language: str = "en",
        engine: str | None = None,
        voice: str | None = None,
        speed: float | None = None,
    ) -> str | None:
        """
        Generate a WAV file in the persona's voice, or in the one given.

        Args:
            text: Text to convert to speech
            language: Whisper language code (e.g. 'en', 'fr', 'ja')
            engine, voice, speed: Instead of the persona's, e.g. for a preview

        Returns:
            Path to generated WAV file, or None if generation failed
        """
        engine = engine or self.engine
        voice = voice or self.voice
        speed = speed or self.speed
        # Another engine takes over, with its voice for that language
        speaker = speaking_engine(engine, language)
        if speaker is not None and speaker[1] is not None:
            engine, voice = speaker
        with self._lock:
            try:
                generator = self._engine(engine)
            except Exception:
                logger.exception("Could not load the voice")
                return None
            generator.voice = voice
            if generator.takes_speed:
                generator.speed = speed
            return generator.generate(text, language=language)


# Clips of the voice remembered, to recognize its echo in a turn
SPOKEN_KEPT = 20


class SpeechPipeline:
    """
    Parallel speech generation and playback pipeline.

    Generation and playback happen in parallel for maximum responsiveness.
    Text is queued for generation, which produces audio files that are
    then queued for playback.
    """

    def __init__(self, generator, player):
        """
        Initialize speech pipeline.

        Args:
            generator: KokoroGenerator instance for TTS
            player: AudioPlayer instance for playback
        """
        self.generator = generator
        self.player = player

        # Current language for TTS (set by assistant on each turn)
        self.current_language = "en"

        # Optional callback receiving playback level (0.0-1.0) for output meters
        self.on_output_level = None
        # Optional callback for the moment the voice starts saying a reply
        self.on_reply_audio = None
        self._reply_heard = False
        # What the voice said lately, as (when it stopped being heard, text),
        # so a turn the microphone made of it can be told from the user's
        self._spoken: deque[tuple[float, str]] = deque(maxlen=SPOKEN_KEPT)

        # Queues for pipeline
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        # Flags for status tracking
        self.is_speaking_flag = threading.Event()
        self.is_generating_flag = threading.Event()
        self._status_lock = threading.Lock()

        # stop() starts a new epoch; work queued in an older one is dropped
        # wherever it is, even audio still being generated when stop() came
        self._epoch = 0
        self._generating_epoch: int | None = None

        # What has been said of the current reply: the texts played to their
        # end, and the one playing now
        self._said: list[str] = []
        self._playing: str | None = None

        # Worker threads
        self.generation_worker_thread = None
        self.playback_worker_thread = None

        # Start the pipeline
        self._start_pipeline()

    def _start_pipeline(self):
        """
        Start pipeline: generation thread → audio queue → playback thread.
        """

        def generation_worker():
            """Generate TTS audio files from text queue with lower priority."""
            # Lower priority of generation thread to prevent audio underruns
            try:
                import os

                # Increase niceness significantly (lower priority) on Unix systems
                os.nice(10)
            except (AttributeError, OSError):
                pass  # Not available on all systems

            while True:
                item = self.text_queue.get()
                if item is None:  # Shutdown signal
                    self.audio_queue.put(None)
                    break

                # Unpack text, language, the epoch it was queued in, and the
                # voice it is spoken in when not the persona's (a preview)
                text, language, epoch, sound = item
                if epoch != self._epoch:  # stopped before its turn came
                    self.text_queue.task_done()
                    continue

                # Set flag to indicate TTS generation in progress
                self._generating_epoch = epoch
                self.is_generating_flag.set()

                # Generate audio file in the requested language
                voice = {k: v for k, v in (sound or {}).items() if k != "audio_filter"}
                audio_file = self.generator.generate(text, language=language, **voice)

                # Clear flag when done generating
                self.is_generating_flag.clear()
                self._generating_epoch = None

                # Put generated audio in playback queue, unless stop() came
                # while it was being generated
                if audio_file and epoch == self._epoch:
                    self.audio_queue.put((audio_file, epoch, text, sound))
                elif audio_file:
                    Path(audio_file).unlink(missing_ok=True)
                self.text_queue.task_done()

        def playback_worker():
            """Play audio files from audio queue sequentially."""
            while True:
                item = self.audio_queue.get()
                if item is None:  # Shutdown signal
                    break
                audio_file, epoch, text, sound = item

                # Play the audio, unless it belongs to a stopped epoch; the
                # player checks again while playing, so a stop() racing the
                # start of the clip still silences it
                with self._status_lock:
                    playing = epoch == self._epoch
                    # A preview in another voice is not part of the reply
                    first = playing and not sound and not self._reply_heard
                    if playing and not sound:
                        self._playing = text
                        self._reply_heard = True
                if first and self.on_reply_audio:
                    self.on_reply_audio()
                if playing:
                    self.is_speaking_flag.set()
                    effect = {}
                    if sound and "audio_filter" in sound:
                        name = sound["audio_filter"]
                        effect["audio_filter"] = get_filter(name) if name else None
                    self.player.play(
                        audio_file,
                        on_level=self.on_output_level,
                        cancelled=lambda: epoch != self._epoch,
                        **effect,
                    )
                    self.is_speaking_flag.clear()
                    with self._status_lock:
                        # Played to its end, unless stop() came first and
                        # kept the part that was heard
                        if self._playing is not None and epoch == self._epoch:
                            self._said.append(text)
                        self._playing = None
                        if not sound:
                            # When it stopped being heard: the stream
                            # outlives the clip by up to a block
                            heard = getattr(self.player, "heard_until", None)
                            ended = time.monotonic() if heard is None else min(time.monotonic(), heard)
                            self._spoken.append((ended, text))

                # Cleanup
                try:
                    Path(audio_file).unlink(missing_ok=True)
                except Exception:
                    pass

                self.audio_queue.task_done()

        # Start both threads
        self.generation_worker_thread = threading.Thread(target=generation_worker, daemon=True)
        self.playback_worker_thread = threading.Thread(target=playback_worker, daemon=True)
        self.generation_worker_thread.start()
        self.playback_worker_thread.start()

    def more_to_say(self) -> bool:
        """Whether more of what it speaks is still to come: text waiting to
        be spoken, being synthesized, or synthesized and waiting to play."""
        with self._status_lock:
            generating = (
                self.is_generating_flag.is_set() and self._generating_epoch == self._epoch
            )
            return not self.text_queue.empty() or generating or not self.audio_queue.empty()

    @property
    def is_active(self) -> bool:
        """Check if pipeline has pending work or is currently processing."""
        with self._status_lock:
            # Audio still being generated for a stopped epoch will be dropped:
            # it does not count as speaking
            generating = (
                self.is_generating_flag.is_set() and self._generating_epoch == self._epoch
            )
            return (
                not self.text_queue.empty()
                or generating
                or not self.audio_queue.empty()
                or self.is_speaking_flag.is_set()
            )

    def queue_speech(self, text: str, language: str | None = None, sound: dict | None = None):
        """
        Add text to speech pipeline (non-blocking).

        Args:
            text: Text to speak
            language: Whisper language code; self.current_language if None
            sound: Voice to speak in instead of the persona's, e.g. for a
                preview: engine, voice, speed and audio_filter, each optional
        """
        language = language or self.current_language
        self.text_queue.put((text, language, self._epoch, sound))

    def begin_reply(self):
        """Start keeping track of what is said of a new reply."""
        with self._status_lock:
            self._said = []
            self._reply_heard = False

    def spoken_since(self, moment: float) -> str:
        """What the voice was saying at or after this moment (time.monotonic()):
        clips still heard then, and the one playing now."""
        with self._status_lock:
            texts = [text for ended, text in self._spoken if ended >= moment]
            if self._playing:
                texts.append(self._playing)
        return " ".join(texts)

    def said(self) -> str:
        """What has been said of the current reply so far, as far as it was
        heard when stop() cut it."""
        with self._status_lock:
            return " ".join(self._said)

    def wait_for_speech(self):
        """Wait for all speech generation and playback to finish."""
        self.text_queue.join()  # Wait for generation
        self.audio_queue.join()  # Wait for playback

    def pause(self):
        """Hold speech mid-sentence; queued text keeps being synthesized."""
        self.player.pause()

    def resume(self):
        """Continue speech from where pause() held it."""
        self.player.resume()

    def stop(self):
        """Stop the speech pipeline immediately: what plays, what waits, and
        what is still being generated."""
        with self._status_lock:
            self._epoch += 1
            # Of the clip playing now, the words heard before the stop count
            # as said
            if self._playing is not None:
                part = heard_part(self._playing, self.player.progress())
                if part:
                    self._said.append(part)
                self._playing = None
        # Stop any currently playing audio
        self.player.stop()

        # Clear queues
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
                self.text_queue.task_done()
            except Exception:
                pass
        while not self.audio_queue.empty():
            try:
                item = self.audio_queue.get_nowait()
                if item is not None:
                    Path(item[0]).unlink(missing_ok=True)
                self.audio_queue.task_done()
            except Exception:
                pass

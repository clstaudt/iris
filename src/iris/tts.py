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
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pocket_tts.models.tts_model as pocket_model_module
import pocket_tts.modules.text_conditioner as pocket_text_module
import pocket_tts.utils.config as pocket_config_module
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro
from kokoro_onnx import tokenizer as kokoro_tokenizer
from kokoro_onnx.config import SUPPORTED_LANGUAGES as KOKORO_ONNX_LANGUAGES
from kokoro_onnx.config import VOCAB as KOKORO_VOCAB
from misaki import zh as misaki_zh
from phonemizer.backend import EspeakBackend
from phonemizer.separator import default_separator
from phonemizer.utils import list2str, str2list
from pocket_tts import TTSModel
from pocket_tts.default_parameters import DEFAULT_VOICE_FOR_LANGUAGE
from pykakasi import kakasi

from iris import cloning, downloads, pocket
from iris.audio import SpeakerError, get_filter, voice_log
from iris.cloning import cloned_voices, made_for, voice_folder, voice_state
from iris.downloads import HubFile, Progress
from iris.engines import (
    SpeechEngine,
    Step,
    register_speech_engine,
    speech_engine,
    speech_languages,
    voice_speaks,
)
from iris.events import brief
from iris.numbers import numbers_in_words

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class KeptEspeak:
    """Stands in for the phonemizer module in Kokoro's tokenizer.

    phonemizer.phonemize() loads espeak afresh on every call, which takes
    about 1.5 s on an M3: three times what the voice model needs for a
    sentence. This phonemizes the same way, with one espeak per language,
    loaded once and kept.

    Where espeak reads a word as English ("football" in French), it marks
    the switch, "(en)fˈʊtbɔːl(fr)". Kokoro would say the marks ("en
    football fr"), so they are dropped, as Kokoro's own pipeline drops them.
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
                    language,
                    preserve_punctuation=preserve_punctuation,
                    with_stress=with_stress,
                    language_switch="remove-flags",
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

# Kokoro v1.0 was trained on misaki's Mandarin, which writes some sounds as
# symbols of their own. kokoro-onnx's table lacks them, but Kokoro keeps them
# on token ids the table gives to capital letters, which espeak never writes:
# each is passed as the symbol the table has on its id.
KOKORO_V1_IDS = {"̃": 17, "ʣ": 18, "ʥ": 19, "ʦ": 20, "ʨ": 21, "ᵝ": 22, "ꭧ": 23}
_ON_ID = {token: symbol for symbol, token in KOKORO_VOCAB.items()}
KOKORO_V1_SYMBOLS = str.maketrans(
    {s: _ON_ID[t] for s, t in KOKORO_V1_IDS.items() if s not in KOKORO_VOCAB and t in _ON_ID}
)
# Words in Latin letters inside Mandarin, said by espeak's English
LATIN_WORDS = re.compile(r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*")


class Readers:
    """Read Japanese and Mandarin for Kokoro. Each reader is made the first
    time it is needed (a fraction of a second), and used by one thread at a
    time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._kana = None
        self._mandarin = None

    def kana(self, text: str) -> str:
        """Japanese as espeak can read it: its numbers in kanji, and every
        kanji in hiragana. espeak reads no kanji: it says "Chinese letter",
        in English, for each."""
        with self._lock:
            if self._kana is None:
                self._kana = kakasi()
            return "".join(item["hira"] for item in self._kana.convert(numbers_in_words(text, "ja")))

    def mandarin(self, text: str, english) -> str:
        """Kokoro's own phonemes for Mandarin (misaki's, numbers read in
        Chinese), and english(words) for words in Latin letters."""
        with self._lock:
            if self._mandarin is None:
                self._mandarin = misaki_zh.ZHG2P()
            parts, last = [], 0
            for match in LATIN_WORDS.finditer(text):
                parts += [self._mandarin(text[last : match.start()])[0], english(match[0])]
                last = match.end()
            parts.append(self._mandarin(text[last:])[0])
        return " ".join(p.strip() for p in parts if p.strip()).translate(KOKORO_V1_SYMBOLS)


READERS = Readers()


def kokoro_speech(tokenizer, text: str, language: str, code: str) -> dict:
    """What Kokoro is given to say a text: the text, the language kokoro-onnx
    is to read it in, and phonemes when they are made here.

    kokoro-onnx reads numbers with English rules before espeak does, in any
    language ("16,50 €" as 1650): French, Italian, Spanish and Portuguese
    numbers are given in words. Japanese is given in kana, and Mandarin as
    misaki's phonemes, Kokoro's own: espeak reads neither well. Spanish,
    Portuguese and Hindi, which kokoro-onnx refuses by name, are given as
    espeak's phonemes, named as English only to pass its check.

    Args:
        tokenizer: kokoro-onnx's tokenizer, for espeak's phonemes
        language: Whisper code of the text's language
        code: Kokoro's code for it (KOKORO_LANGUAGES)
    """
    if language in ("fr", "it", "es", "pt"):
        text = numbers_in_words(text, language)
    if language == "ja":
        return {"text": READERS.kana(text), "lang": code}
    if language == "zh":
        english = lambda words: tokenizer.phonemize(words, "en-us")  # noqa: E731
        return {"text": text, "lang": "en-us", "phonemes": READERS.mandarin(text, english)}
    if code in KOKORO_ONNX_LANGUAGES:
        return {"text": text, "lang": code}
    return {"text": text, "lang": "en-us", "phonemes": tokenizer.phonemize(text, code)}

# Kokoro model download URLs
KOKORO_MODEL_URL = (
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"
)

# Whisper codes of the languages Kokoro speaks, and the language its phonemes
# for each are made in (espeak's, misaki's for Mandarin). Kokoro has no
# German or Dutch voices. Which of them a voice speaks properly is
# KOKORO_VOICES_SPEAK.
KOKORO_LANGUAGES = {
    "en": "en-us",
    "fr": "fr-fr",
    "it": "it",
    "es": "es",
    "pt": "pt-br",
    "hi": "hi",
    "ja": "ja",
    "zh": "cmn",
}


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
    languages = frozenset(KOKORO_LANGUAGES)
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
    def make_ready(
        cls,
        voice: str,
        language: str,
        models_dir: str | Path | None = None,
        on_progress: Progress | None = None,
        on_step: Callable[[Step], None] | None = None,
    ):
        """Its model and voices, one download for every language (a folder
        chosen instead is used as it is)."""
        if cls.available(models_dir):
            return
        if models_dir:
            raise FileNotFoundError(f"{models_dir} does not hold Kokoro's model and voices")
        if on_step:
            on_step(Step("download", "model"))
        downloads.download_files(
            [
                (KOKORO_MODEL_URL, DEFAULT_KOKORO_DIR / KOKORO_MODEL_FILE),
                (KOKORO_VOICES_URL, DEFAULT_KOKORO_DIR / KOKORO_VOICES_FILE),
            ],
            on_progress or (lambda done, total: None),
        )

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
    def voice_languages(cls, voice: str) -> frozenset[str]:
        """As measured (KOKORO_VOICES_SPEAK); a voice not measured is taken
        to speak the language it was made for and English."""
        if voice in KOKORO_VOICES_SPEAK:
            return KOKORO_VOICES_SPEAK[voice]
        return frozenset({cls.voice_language(voice), "en"})

    def code_for(self, language: str) -> str | None:
        """Kokoro's code for a reply in this language in the persona's
        voice, or None when the voice does not speak it. British voices
        are given British English."""
        if language not in self.voice_languages(self.voice):
            return None
        code = KOKORO_LANGUAGES[language]
        if code == "en-us" and self.voice.startswith("b"):
            code = "en-gb"
        return code

    def generate(self, text: str, language: str = "en") -> str | None:
        """
        Generate audio file from text using kokoro-onnx Python API.

        Args:
            text: Text to convert to speech
            language: Whisper language code (e.g. 'en', 'fr', 'ja'), which
                the voice speaks

        Returns:
            Path to generated WAV file, or None if generation failed or the
            voice does not speak the language
        """
        kokoro_lang, voice = self.code_for(language), self.voice
        if kokoro_lang is None:
            logger.warning("The Kokoro voice %s does not speak %s", voice, language)
            return None

        temp_audio = None
        try:
            # Create temporary file for output
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio.close()

            # Generate audio using kokoro-onnx Python API
            spoken = kokoro_speech(self.kokoro.tokenizer, text, language, kokoro_lang)
            audio_data, sample_rate = self.kokoro.create(voice=voice, speed=self.speed, **spoken)

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


# Characters that stand alone, as Japanese and Chinese (and Korean) put no
# spaces between words
STANDS_ALONE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


def heard_part(text: str, fraction: float) -> str:
    """The words of a spoken text that were heard out by the time this
    fraction of its audio had played (judged by length: speech goes through
    a text at a fairly even pace)."""
    end = int(len(text) * max(0.0, min(1.0, fraction)))
    if end >= len(text):
        return text

    def inside_a_word(at: int) -> bool:
        before, after = text[at - 1], text[at]
        spaced = not (STANDS_ALONE.match(before) or STANDS_ALONE.match(after))
        return spaced and not before.isspace() and not after.isspace()

    # Cut in the middle of a word, which was not heard out: back to its start
    while end > 0 and inside_a_word(end):
        end -= 1
    return text[:end].rstrip()


# Whisper language codes Pocket TTS speaks, and the model for each. It has
# none for Hindi, Japanese and Chinese.
POCKET_LANGUAGES = pocket.LANGUAGES


def _from_the_cache(file_path) -> Path:
    """Pocket TTS's own way to its files, which would download what is
    missing, unseen: from the Hugging Face cache only. Iris downloads what
    a voice needs for a language beforehand, with progress (iris.readiness).

    Raises:
        LocalEntryNotFoundError: The file is not downloaded
    """
    path = str(file_path)
    if path.startswith("hf://"):
        file = HubFile.parse(path)
        return Path(
            hf_hub_download(
                file.repo, file.filename, revision=file.revision, local_files_only=True
            )
        )
    if path.startswith(("http://", "https://")):
        raise FileNotFoundError(f"Pocket TTS does not download {path} while Iris speaks")
    return Path(path)


# Every module of Pocket TTS that fetches what it loads (the model and its
# weights, the tokenizer, a config, a voice) takes its files from the cache
for _module in (pocket_model_module, pocket_text_module, pocket_config_module):
    _module.download_if_necessary = _from_the_cache
# Pocket TTS's English voices; each further language has one voice of its
# own. Every voice speaks every language (see PocketGenerator), but these.
POCKET_ENGLISH_VOICES = (
    "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette",
    "eponine", "eve", "fantine", "george", "jane", "jean", "javert", "marius",
    "mary", "michael", "paul", "peter_yearsley", "stuart_bell", "vera",
)  # fmt: skip
# The languages of Pocket TTS a voice does not speak properly, as measured
# (see KOKORO_VOICES_SPEAK). Through French's model, about half of the
# voices made for another language keep so much of their accent that
# Whisper now and then hears English. The other languages were measured with
# Anna, Alba, Charles, Juergen and Estelle, who all speak them.
POCKET_VOICES_LACK = {
    "alba": frozenset({"fr"}),
    "azelma": frozenset({"fr"}),
    "bill_boerst": frozenset({"fr"}),
    "cosette": frozenset({"fr"}),
    "eponine": frozenset({"fr"}),
    "george": frozenset({"fr"}),
    "jean": frozenset({"fr"}),
    "lola": frozenset({"fr"}),
    "marius": frozenset({"fr"}),
    "paul": frozenset({"fr"}),
    "peter_yearsley": frozenset({"fr"}),
    "rafael": frozenset({"fr"}),
}
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
# The languages each Kokoro voice speaks properly, as measured: Whisper
# (small) heard the language in 3 of 4 of the voice's sentences, and about as
# many of their words as from the voices made for the language
# (tests/unit/fixtures/voice_languages.json, which tests/unit/
# test_multilingual.py holds these tables to). Each voice speaks the
# language it was made for but zf_xiaoni (its Mandarin half understood), and
# English but jf_alpha (heard as Japanese one time in two); most speak
# Spanish, Portuguese, Italian or Mandarin, few French or Hindi, and only the
# Japanese voices Japanese.
KOKORO_VOICES_SPEAK = {  # fmt: skip
    "af_alloy": frozenset({"en", "es", "pt", "zh"}),
    "af_aoede": frozenset({"en", "es", "pt", "zh"}),
    "af_bella": frozenset({"en", "es", "pt", "zh"}),
    "af_heart": frozenset({"en", "es", "pt", "zh"}),
    "af_jessica": frozenset({"en", "it", "es", "pt"}),
    "af_kore": frozenset({"en", "es", "pt", "zh"}),
    "af_nicole": frozenset({"en", "es", "pt", "zh"}),
    "af_nova": frozenset({"en", "it", "es", "pt", "zh"}),
    "af_river": frozenset({"en", "es", "pt", "zh"}),
    "af_sarah": frozenset({"en", "es", "pt", "zh"}),
    "af_sky": frozenset({"en", "it", "es", "pt", "hi"}),
    "am_adam": frozenset({"en", "it", "es", "pt", "zh"}),
    "am_echo": frozenset({"en", "it", "es", "pt", "zh"}),
    "am_eric": frozenset({"en", "it", "es", "pt", "zh"}),
    "am_fenrir": frozenset({"en", "it", "es", "pt", "zh"}),
    "am_liam": frozenset({"en", "es", "pt", "zh"}),
    "am_michael": frozenset({"en", "it", "es", "pt", "zh"}),
    "am_onyx": frozenset({"en", "it", "es", "pt", "zh"}),
    "am_puck": frozenset({"en", "it", "es", "pt", "zh"}),
    "am_santa": frozenset({"en", "fr", "it", "es", "pt", "hi", "zh"}),
    "bf_alice": frozenset({"en", "it", "es"}),
    "bf_emma": frozenset({"en", "it", "es", "zh"}),
    "bf_isabella": frozenset({"en", "it", "es", "zh"}),
    "bf_lily": frozenset({"en", "it", "pt"}),
    "bm_daniel": frozenset({"en", "fr", "it", "es", "pt"}),
    "bm_fable": frozenset({"en", "it", "es", "pt", "zh"}),
    "bm_george": frozenset({"en", "fr", "it", "es", "pt", "zh"}),
    "bm_lewis": frozenset({"en", "it", "es", "pt", "zh"}),
    "ef_dora": frozenset({"en", "it", "es", "pt", "hi", "zh"}),
    "em_alex": frozenset({"en", "it", "es", "pt", "hi", "zh"}),
    "em_santa": frozenset({"en", "it", "es", "pt", "zh"}),
    "ff_siwis": frozenset({"en", "fr", "es", "zh"}),
    "hf_alpha": frozenset({"en", "it", "es", "hi", "zh"}),
    "hf_beta": frozenset({"en", "it", "es", "pt", "hi"}),
    "hm_omega": frozenset({"en", "es", "pt", "hi", "zh"}),
    "hm_psi": frozenset({"en", "it", "es", "pt", "hi", "zh"}),
    "if_sara": frozenset({"en", "fr", "it", "pt", "hi", "zh"}),
    "im_nicola": frozenset({"en", "it", "pt", "zh"}),
    "jf_alpha": frozenset({"it", "es", "pt", "ja", "zh"}),
    "jf_gongitsune": frozenset({"en", "it", "es", "pt", "hi", "ja", "zh"}),
    "jf_nezumi": frozenset({"en", "it", "es", "pt", "ja", "zh"}),
    "jf_tebukuro": frozenset({"en", "it", "es", "pt", "hi", "ja", "zh"}),
    "jm_kumo": frozenset({"en", "it", "es", "pt", "ja", "zh"}),
    "pf_dora": frozenset({"en", "it", "es", "pt", "hi", "zh"}),
    "pm_alex": frozenset({"en", "it", "es", "pt", "hi", "zh"}),
    "pm_santa": frozenset({"en", "it", "es", "pt", "zh"}),
    "zf_xiaobei": frozenset({"en", "es", "pt", "zh"}),
    "zf_xiaoni": frozenset({"en", "pt"}),
    "zf_xiaoxiao": frozenset({"en", "it", "es", "pt", "zh"}),
    "zf_xiaoyi": frozenset({"en", "it", "es", "pt", "zh"}),
}
# A word in each language a voice speaks, said (and not played) to load what
# speaking the language needs
GREETINGS = {
    "en": "Hello.",
    "fr": "Bonjour.",
    "it": "Ciao.",
    "de": "Hallo.",
    "es": "Hola.",
    "pt": "Olá.",
    "nl": "Hallo.",
    "hi": "नमस्ते।",
    "ja": "こんにちは。",
    "zh": "你好。",
}
# Iris's languages, in the order the app lists them
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "nl": "Dutch",
    "hi": "Hindi",
    "ja": "Japanese",
    "zh": "Chinese",
}


def in_order(languages) -> list[str]:
    """Whisper codes in the order the app lists Iris's languages."""
    order = list(LANGUAGE_NAMES)
    return sorted(languages, key=lambda code: (order.index(code) if code in order else 99, code))


def voice_languages(engine: str, voice: str) -> list[str]:
    """The languages a voice speaks properly, as Whisper codes in order."""
    try:
        return in_order(speech_engine(engine).voice_languages(voice))
    except ValueError:
        return []


def voice_ready(engine: str, voice: str, language: str, models_dir=None) -> bool:
    """Whether a voice is ready to speak a language now: it speaks the
    language properly, and what it needs for it is downloaded (and, for a
    cloned voice, made). Speaking never fetches it; iris.readiness does,
    with progress."""
    if not voice_speaks(engine, voice, language):
        return False
    try:
        return speech_engine(engine).ready(voice, language, models_dir)
    except ValueError:
        return False


def voice_language(engine: str, voice: str) -> str:
    """The Whisper language code of the language a voice was made for."""
    try:
        return speech_engine(engine).voice_language(voice)
    except ValueError:
        return "en"


def list_voices(engine: str, models_dir: str | Path | None = None) -> list[dict]:
    """An engine's voices as {id, name, language, languages}: the language
    each was made for, as shown, and the Whisper codes of those it speaks
    properly, the one it was made for first."""
    found = speech_engine(engine)
    listed = []
    for voice in found.voices(models_dir):
        spoken = voice.get("languages") or voice_languages(engine, voice["id"])
        own = found.voice_language(voice["id"])
        listed.append({**voice, "languages": sorted(spoken, key=lambda code: code != own)})
    return listed


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


def pocket_files(voices=("eve",), language: str = "english") -> list[str]:
    """The files Pocket TTS loads for a language and these voices of it, as
    hf:// paths, so they can be fetched with progress before it loads them.

    Its model comes in two forms: one that clones voices, which Hugging Face
    hands out only to an account that accepted its terms, and the open one
    without cloning that Pocket TTS falls back to. The open one is fetched.
    """
    known = set(PocketGenerator.catalogue())
    return pocket.model_paths(language) + [
        pocket.voice_path(language, voice) for voice in voices if voice in known
    ]


class PocketGenerator(SpeechEngine):
    """Generates speech audio with Kyutai's Pocket TTS, on the CPU.

    Pocket TTS has one model per language: English loads when the persona
    speaks with Pocket TTS, the others the first time a reply is in that
    language. A voice speaks each of them through the language's model,
    given the voice's own prompt, and stays itself: Whisper hears the
    language word for word, at the voice's own pitch. A few voices keep too
    much of their accent in French (POCKET_VOICES_LACK).
    """

    id = "kyutai"
    name = "Kyutai"
    languages = frozenset(POCKET_LANGUAGES)

    @classmethod
    def voice_languages(cls, voice: str) -> frozenset[str]:
        return cls.languages - POCKET_VOICES_LACK.get(voice, frozenset())

    @classmethod
    def catalogue(cls) -> list[str]:
        """Pocket TTS's own voices: the English ones, and one for each
        further language."""
        return list(POCKET_ENGLISH_VOICES) + [
            DEFAULT_VOICE_FOR_LANGUAGE[pocket_lang]
            for code, pocket_lang in POCKET_LANGUAGES.items()
            if code != "en"
        ]

    @classmethod
    def missing(cls, voice: str, language: str) -> list[HubFile]:
        """The files speaking a language in a voice still needs from Hugging
        Face: the language's model (in either form: the one that clones
        speaks as well, once it is here) and, for one of Pocket TTS's own
        voices, the voice as the model reads it. A cloned voice is made for
        the language besides (made_for)."""
        pocket_lang = POCKET_LANGUAGES[language]
        cached = cloning.KYUTAI.cached
        model = pocket.model_files(pocket_lang, clones=True)
        if not all(cached(file) for file in model):
            model = pocket.model_files(pocket_lang)
        files = list(model)
        if voice_folder(voice) is None:
            files.append(pocket.voice_file(pocket_lang, voice))
        return [file for file in files if not cached(file)]

    @classmethod
    def ready(cls, voice: str, language: str, models_dir: str | Path | None = None) -> bool:
        pocket_lang = POCKET_LANGUAGES.get(language)
        if pocket_lang is None:
            return False
        folder = voice_folder(voice)
        if folder is not None and pocket_lang not in made_for(folder):
            return False
        try:
            return not cls.missing(voice, language)
        except Exception:
            return False

    @classmethod
    def make_ready(
        cls,
        voice: str,
        language: str,
        models_dir: str | Path | None = None,
        on_progress: Progress | None = None,
        on_step: Callable[[Step], None] | None = None,
    ):
        """Download the language's model and the voice for it; a cloned
        voice is made for the language from its sample, with the language's
        cloning model (downloaded first), unless it is made already.

        Raises:
            CloneAccessError: A cloned voice to make, without Hugging Face
                access to the cloning model
            FileNotFoundError: A cloned voice whose sample is gone
        """
        on_progress = on_progress or (lambda done, total: None)
        on_step = on_step or (lambda step: None)
        pocket_lang = POCKET_LANGUAGES[language]
        folder = voice_folder(voice)
        if folder is not None and pocket_lang not in made_for(folder):
            if not cloning.weights_cached(pocket_lang):
                on_step(Step("download", "cloning"))
            cloning.make_language(
                folder, pocket_lang, on_progress, on_making=lambda: on_step(Step("make"))
            )
        missing = cls.missing(voice, language)
        if missing:
            weights = {file for file in pocket.model_files(pocket_lang)}
            on_step(Step("download", "model" if weights & set(missing) else "voice"))
            cloning.KYUTAI.fetch(missing, on_progress)

    def voice_changed(self, voice: str):
        """A cloned voice made again: what was loaded of it goes."""
        folder = voice_folder(voice)
        key = str(folder) if folder is not None else voice
        for loaded in [k for k in self._voice_states if k[1] == key]:
            del self._voice_states[loaded]

    def __init__(self, voice: str = "eve", models_dir: str | Path | None = None):
        """
        Args:
            voice: A Pocket TTS voice, an audio file of a voice to clone, or
                a voice cloned in the app ("clone:<name>")
            models_dir: Unused: Pocket TTS fetches its models itself
        """
        self.voice = voice
        self._models: dict[str, TTSModel] = {}
        self._voice_states: dict[tuple[str, str], object] = {}

    def prepare(self):
        """English, which a conversation starts in, loads at once."""
        self._model("english")

    def _model(self, pocket_lang: str) -> TTSModel:
        if pocket_lang not in self._models:
            self._models[pocket_lang] = TTSModel.load_model(language=pocket_lang)
        return self._models[pocket_lang]

    def _voice_state(self, pocket_lang: str, voice: str):
        key = (pocket_lang, voice)
        if key not in self._voice_states:
            folder = voice_folder(voice)
            if folder is not None:
                # Made by the language's cloning model the first time, kept
                self._voice_states[key] = voice_state(self._models, pocket_lang, folder)
            else:
                model = self._model(pocket_lang)
                self._voice_states[key] = model.get_state_for_audio_prompt(voice)
        return self._voice_states[key]

    @property
    def voice(self) -> str:
        """The voice it speaks with: one of Pocket TTS's own, an audio file
        to clone, or the folder of a voice cloned in the app, which personas
        name "clone:<name>" (iris.cloning). Like an audio file, a cloned
        voice speaks every language as itself."""
        return self._voice

    @voice.setter
    def voice(self, voice: str):
        folder = voice_folder(voice)
        self._voice = str(folder) if folder is not None else voice

    @classmethod
    def voices(cls, models_dir: str | Path | None = None) -> list[dict]:
        names = cls.catalogue()
        return [
            {
                "id": name,
                "name": name.replace("_", " ").title(),
                "language": LANGUAGE_NAMES[cls.voice_language(name)],
            }
            for name in names
        ] + cloned_voices()

    @classmethod
    def voice_language(cls, voice: str) -> str:
        """Each language beyond English has a voice of its own."""
        for code, pocket_lang in POCKET_LANGUAGES.items():
            if DEFAULT_VOICE_FOR_LANGUAGE.get(pocket_lang) == voice:
                return code
        return "en"

    def generate(self, text: str, language: str = "en") -> str | None:
        """
        Generate a WAV file from text.

        Args:
            text: Text to convert to speech
            language: Whisper language code (e.g. 'en', 'fr'), spoken with
                the voice through the language's model

        Returns:
            Path to generated WAV file, or None if generation failed or
            Pocket TTS has no model for the language

        Raises:
            Exception: The language's model or the voice could not be
                loaded (offline, at its first use), which the caller tells
        """
        pocket_lang, voice = POCKET_LANGUAGES.get(language), self.voice
        if pocket_lang is None:
            logger.warning("Pocket TTS does not speak %s", language)
            return None
        model = self._model(pocket_lang)
        state = self._voice_state(pocket_lang, voice)
        try:
            # Only the English model reads digits ("am Gas im November"),
            # and it drops the cents of an amount of money
            text = numbers_in_words(text, language if pocket_lang != "english" else "en")
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
    """Speaks with the persona's engine and voice, in every language.

    The engines are those registered in iris.engines: Kyutai's Pocket TTS
    and Kokoro, and any a plugin adds. Each loads the first time it is
    needed. The voice never changes within a persona: a reply in a language
    the persona's voice does not speak properly is not spoken by another
    voice or engine in its place (the assistant shows it instead).
    """

    # The voice model folder chosen by the user, if any
    models_dir: Path | None = None

    def __init__(
        self,
        engine: str = "kyutai",
        voice: str = "eve",
        speed: float = 1.0,
        models_dir: str | Path | None = None,
        prepare: bool = True,
    ):
        """
        Args:
            engine: Id of a registered speech engine
            voice: A voice of that engine
            speed: Speech speed multiplier, for engines that take one
            models_dir: Voice model folder chosen by the user (Kokoro's)
            prepare: Load the engine now (prepare() raises if it cannot be
                loaded); False leaves that to the caller
        """
        self.engine = engine
        self.voice = voice
        self.speed = speed
        self.models_dir = Path(models_dir).expanduser() if models_dir else None
        self._engines: dict[str, SpeechEngine] = {}
        # Replies and previews may ask for speech at the same time
        self._lock = threading.Lock()
        # Told (a message for the user) when an engine, or its model for a
        # language, cannot be loaded; once, until it speaks again
        self.on_failure: Callable[[str], None] | None = None
        self._failing: set[tuple[str, str]] = set()
        # (engine, voice, language) prepared for their first reply
        self._prepared: set[tuple[str, str, str]] = set()
        if prepare:
            self.prepare()

    def speaks(self, language: str) -> bool:
        """Whether the persona's voice speaks this language properly. A
        reply in another is shown, not spoken."""
        return voice_speaks(self.engine, self.voice, language)

    def ready(self, language: str) -> bool:
        """Whether the persona's voice is ready to speak this language now:
        nothing to download or make first (voice_ready)."""
        return voice_ready(self.engine, self.voice, language, self.models_dir)

    def voice_changed(self, voice: str):
        """A voice made again from a new sample: what each engine keeps
        loaded of it goes, and it is prepared afresh."""
        with self._lock:
            for engine in self._engines.values():
                engine.voice_changed(voice)
            self._prepared = {key for key in self._prepared if key[1] != voice}

    def languages(self) -> frozenset[str]:
        """The languages a reply may be in, as Whisper codes: all Iris talks
        in, whether the persona's voice speaks them or not."""
        return speech_languages()

    def prepare(self):
        """Load the current engine now, so its first reply is not delayed."""
        with self._lock:
            self._engine(self.engine).prepare()

    def prepare_language(self, language: str):
        """Load what a reply in this language needs before it comes, while
        the model thinks: the language's model, voice or phonemes. The first
        reply in a language would otherwise wait seconds for them (Pocket
        TTS's model and the voice's prompt for it, espeak's Japanese for
        Kokoro). Once per voice and language, and only in a language the
        voice speaks."""
        engine, voice = self.engine, self.voice
        if not self.speaks(language) or (engine, voice, language) in self._prepared:
            return
        # What is not here is not fetched for it (iris.readiness does that)
        if not self.ready(language):
            return
        with self._lock:
            try:
                generator = self._engine(engine)
                generator.voice = voice
                path = generator.generate(GREETINGS.get(language, GREETINGS["en"]), language=language)
            except Exception:
                logger.exception("Could not prepare the voice for %s", language)
                return
        if path:
            Path(path).unlink(missing_ok=True)
        self._prepared.add((engine, voice, language))

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
            Path to generated WAV file, or None if generation failed or the
            voice does not speak the language
        """
        engine = engine or self.engine
        voice = voice or self.voice
        speed = speed or self.speed
        # No other voice speaks in its place
        if not voice_speaks(engine, voice, language):
            logger.warning("The voice %s:%s does not speak %s", engine, voice, language)
            return None
        with self._lock:
            try:
                generator = self._engine(engine)
                generator.voice = voice
                if generator.takes_speed:
                    generator.speed = speed
                path = generator.generate(text, language=language)
            except Exception as e:
                logger.exception("Could not load the voice")
                self._failed(engine, language, e)
                return None
            self._failing.discard((engine, language))
            return path

    def _failed(self, engine: str, language: str, error: Exception):
        """Tell that a voice could not be loaded, once until it speaks: a
        reply in that language is then silent, which the user would
        otherwise not know the reason of."""
        if (engine, language) in self._failing:
            return
        self._failing.add((engine, language))
        if self.on_failure is not None:
            name = LANGUAGE_NAMES.get(language, language)
            self.on_failure(
                f"The voice for {name} could not be loaded, so Iris does not speak {name} "
                f"for now. {brief(error)}"
            )


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
        # Optional callback told when the speaker could not be opened (the
        # SpeakerError): once, until a clip plays again
        self.on_speaker_error = None
        self._speaker_failed = False
        # The speaker failed during this reply: the rest of it is not spoken
        self._muted = False
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
        # Of the piece of the reply played last, for the gap before the next:
        # when it was heard to its end, and when the player was done with it
        # (None at the start of a reply, or once a piece was cut short)
        self._heard_to: tuple[float, float] | None = None

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
            """Generate TTS audio files from the text queue, at the priority
            of the rest of the core. It used to call os.nice(10), which on
            macOS lowers the whole process, the audio's threads too, and
            stacked with every pipeline made (a preview, then Iris's own)
            to the lowest priority there is. Synthesis runs far ahead of
            playback without it (irisdev.speech_timing measures that)."""
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
                ready = time.monotonic()

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
                    heard_to, self._heard_to = self._heard_to, None
                if first and self.on_reply_audio:
                    self.on_reply_audio()
                if playing:
                    self.is_speaking_flag.set()
                    effect = {}
                    if sound and "audio_filter" in sound:
                        name = sound["audio_filter"]
                        effect["audio_filter"] = get_filter(name) if name else None
                    played = True
                    try:
                        self.player.play(
                            audio_file,
                            on_level=self.on_output_level,
                            cancelled=lambda: epoch != self._epoch,
                            **effect,
                        )
                    except SpeakerError as error:
                        played = False
                        self._speaker_lost(error)
                    except Exception:
                        played = False
                        logger.exception("Playback failed")
                    else:
                        self._speaker_failed = False
                    self.is_speaking_flag.clear()
                    heard = getattr(self.player, "heard", None)
                    began, ended = heard if isinstance(heard, tuple) else (None, None)
                    with self._status_lock:
                        # Played to its end, unless stop() came first and
                        # kept the part that was heard
                        if played and self._playing is not None and epoch == self._epoch:
                            self._said.append(text)
                        self._playing = None
                        if played and not sound:
                            self._spoken.append((time.monotonic(), text))
                        if played and not sound and epoch == self._epoch and ended is not None:
                            self._heard_to = (ended, time.monotonic())
                    if played and not sound and heard_to is not None and began is not None:
                        self._log_gap(began - heard_to[0], ready - heard_to[1])

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

    @staticmethod
    def _log_gap(gap: float, waited: float):
        """Tell the voice log how long the voice was silent between two
        pieces of a reply, and how much of that it waited for the second to
        be synthesized."""
        waiting = f", {waited:.2f} s of it waiting for the piece to be synthesized"
        voice_log.debug(
            "gap: %.2f s between two pieces of the reply%s", gap, waiting if waited >= 0.01 else ""
        )

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

    def _speaker_lost(self, error: SpeakerError):
        """The speaker could not be opened, even with the devices read anew:
        the rest of the reply is not spoken (it is shown all the same), and
        the user is told, once until a clip plays again. The next reply, or
        a preview, tries the speaker again."""
        logger.warning("The speaker could not be opened: %s", error)
        self.stop()
        with self._status_lock:
            self._muted = True
            told, self._speaker_failed = self._speaker_failed, True
        if not told and self.on_speaker_error:
            self.on_speaker_error(error)

    def queue_speech(self, text: str, language: str | None = None, sound: dict | None = None):
        """
        Add text to speech pipeline (non-blocking).

        Args:
            text: Text to speak
            language: Whisper language code; self.current_language if None
            sound: Voice to speak in instead of the persona's, e.g. for a
                preview: engine, voice, speed and audio_filter, each optional
        """
        if self._muted:
            return
        language = language or self.current_language
        self.text_queue.put((text, language, self._epoch, sound))

    def queue_audio(self, audio_file: str | Path):
        """Play an audio file as it is, after what is queued, outside any
        reply: no voice says it, and no effect is put on it (a sample heard
        before a voice is cloned of it). The file is removed once played,
        or once stop() drops it."""
        self.audio_queue.put((str(audio_file), self._epoch, "", {"audio_filter": ""}))

    def begin_reply(self):
        """Start keeping track of what is said of a new reply."""
        with self._status_lock:
            self._said = []
            self._reply_heard = False
            self._muted = False
            self._heard_to = None

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

    def resume(self, rewind: bool = True):
        """Continue speech from where pause() held it: a little before it
        (rewind), or exactly there (see AudioPlayer.resume)."""
        self.player.resume(rewind)

    def stop(self):
        """Stop the speech pipeline immediately: what plays, what waits, and
        what is still being generated."""
        with self._status_lock:
            self._epoch += 1
            self._heard_to = None
            # What comes next is spoken again, should the speaker have failed
            self._muted = False
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

"""A persona in a file of its own, to take to another computer or give to
someone, and back: exported and imported from Settings, Personas.

A persona file (Noam.irispersona) is a zip archive of:

    persona.yaml                the persona, as its file in ~/.iris/personas
                                holds it, with the version of this format
    voice/voice.yaml            the persona's cloned voice, if it speaks with
    voice/sample.wav            one: its name, its sample, and the voice made
    voice/<language>.safetensors  of it for each language so far

A cloned voice travels with the languages made of it, so that it speaks them
at once where it goes, without access to Kyutai's gated cloning model; and
with its sample, to be made for others there, and cloned again. A voice of
Kyutai's own or of Kokoro's is only named.

A persona file comes from anywhere. It is imported only when it holds what
Iris writes and nothing else: no other file, no path that leads elsewhere,
no link, nothing too large; its YAML read with safe_load, each setting
checked; the sample a WAV of a sample's length, and each language's voice a
voice state as Pocket TTS keeps one. The voice is put together in a folder
of its own first, and goes into the library only once all of it is sound,
before the persona is saved. A file refused leaves nothing behind.

Imported, the persona is named as it was, unless another persona has the
name: then "Noam 2"; its voice keeps its own name in the same way, under a
folder of its own. The voice is made ready for Iris's other languages as any
persona's voice is (iris.readiness), where the user sees it.
"""

import contextlib
import io
import logging
import math
import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path

import numpy as np
import safetensors
import soundfile as sf
import torch
import yaml

from iris import cloning, engines, personas, pocket
from iris.audio import AUDIO_FILTERS
from iris.tts import KOKORO_VOICE_NAME, PocketGenerator

logger = logging.getLogger(__name__)

# The version of the format this Iris writes, and the newest it reads
FORMAT = 1
EXTENSION = ".irispersona"
PERSONA_FILE = "persona.yaml"
VOICE_DIR = "voice/"

# At most: the file, all it holds, the YAML files, the sample (30 s at
# 24 kHz is 1.4 MB), and a language's voice (one of a 30 s sample is 19 MB)
MAX_FILE = 256_000_000
MAX_TOTAL = 256_000_000
MAX_TEXT = 64_000
MAX_SAMPLE = 4_000_000
MAX_STATE = 48_000_000
# A name, wake word or sleep word, in characters, and the speed
MAX_WORDS = 100
SPEEDS = (0.5, 2.0)
# The sample's length, in seconds
SAMPLE_SECONDS = (1.0, cloning.MAX_SAMPLE + 1.0)

NOT_A_PERSONA = "This is not an Iris persona file"
DAMAGED = "This persona file is damaged"
UNREADABLE = "The persona in this file cannot be read"
VOICE_DAMAGED = "The voice in this persona file is damaged"
TOO_LARGE = "This persona file is too large"
STRANGE = "This persona file holds files Iris does not take"
NEWER = "This persona file is from a newer version of Iris"


class PersonaFileError(ValueError):
    """A persona file that cannot be written or imported, with why, for the
    app."""


def _allowed() -> dict[str, int]:
    """The files a persona file may hold, with the most bytes of each."""
    allowed = {
        PERSONA_FILE: MAX_TEXT,
        VOICE_DIR + cloning.INFO_FILE: MAX_TEXT,
        VOICE_DIR + cloning.SAMPLE_FILE: MAX_SAMPLE,
    }
    for pocket_lang in pocket.LANGUAGES.values():
        allowed[f"{VOICE_DIR}{pocket_lang}.safetensors"] = MAX_STATE
    return allowed


# ------------------------------------------------------------------- export


def export_persona(persona: personas.Persona, path: Path):
    """Write a persona, as saved, to a persona file, with its cloned voice
    if it speaks with one. The file appears once it is written whole.

    Raises:
        PersonaFileError: Its cloned voice is not here
        OSError: The file cannot be written
    """
    data = {"format": FORMAT, **personas.file_data(persona.to_dict())}
    folder = cloning.voice_folder(persona.voice)
    if folder is not None and cloning.voice_missing(persona.voice):
        raise PersonaFileError(f"The voice {cloning.voice_name(folder)} cannot be found")
    part = path.with_name(f".{path.name}.part")
    try:
        with zipfile.ZipFile(part, "w") as archive:
            archive.writestr(PERSONA_FILE, personas.dump(data), zipfile.ZIP_DEFLATED)
            if folder is not None:
                info = yaml.safe_dump({"name": cloning.voice_name(folder)}, allow_unicode=True)
                archive.writestr(VOICE_DIR + cloning.INFO_FILE, info, zipfile.ZIP_DEFLATED)
                archive.write(
                    folder / cloning.SAMPLE_FILE,
                    VOICE_DIR + cloning.SAMPLE_FILE,
                    zipfile.ZIP_DEFLATED,
                )
                # A voice state hardly shrinks
                for pocket_lang in cloning.made_for(folder):
                    name = f"{pocket_lang}.safetensors"
                    archive.write(folder / name, VOICE_DIR + name, zipfile.ZIP_STORED)
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


# ------------------------------------------------------------------- import


def import_persona(path: Path) -> str:
    """Add the persona of a persona file, with its cloned voice if it has
    one, once all of it is checked.

    Returns:
        The persona's id

    Raises:
        PersonaFileError: The file is not one Iris takes, with why
        OSError: It cannot be read, or the persona not written
    """
    if path.stat().st_size > MAX_FILE:
        raise PersonaFileError(TOO_LARGE)
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, EOFError, ValueError) as e:
        raise PersonaFileError(NOT_A_PERSONA) from e
    with archive:
        try:
            entries = _entries(archive)
            settings = _settings(_read(archive, entries.pop(PERSONA_FILE)))
            voice = settings["voice"]
            if voice.startswith(cloning.PREFIX) != bool(entries):
                raise PersonaFileError(STRANGE if entries else VOICE_DAMAGED)
            # A name of its own, the wake word that is its name with it
            personas.reload_personas()
            name = personas.unused_name(settings["name"])
            if settings["wake_word"].casefold() == settings["name"].casefold():
                settings["wake_word"] = ""
            settings["name"] = name
            if not entries:
                return personas.save_persona(settings)
            folder = _add_voice(archive, entries, voice[len(cloning.PREFIX) :])
        except (zipfile.BadZipFile, EOFError, zlib.error) as e:
            raise PersonaFileError(DAMAGED) from e
    settings["voice"] = cloning.PREFIX + folder.name
    try:
        return personas.save_persona(settings)
    except BaseException:
        logger.info("The imported persona could not be saved: removing its voice %s", folder)
        shutil.rmtree(folder, ignore_errors=True)
        raise


def _entries(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """The files a persona file holds, by name, once each is one Iris
    writes, in no more bytes than it may have."""
    allowed = _allowed()
    entries: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        # Only the names Iris writes: no other path, none with "..", none
        # that starts at the root, and no link
        name = info.filename
        link = (info.external_attr >> 16) & 0o170000 == 0o120000
        if link or name not in allowed or name in entries:
            raise PersonaFileError(STRANGE)
        if info.flag_bits & 0x1 or info.compress_type not in (
            zipfile.ZIP_STORED,
            zipfile.ZIP_DEFLATED,
        ):
            raise PersonaFileError(NOT_A_PERSONA)
        total += info.file_size
        if info.file_size > allowed[name] or total > MAX_TOTAL:
            raise PersonaFileError(TOO_LARGE)
        entries[name] = info
    if PERSONA_FILE not in entries:
        raise PersonaFileError(NOT_A_PERSONA)
    return entries


def _copy(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target) -> int:
    """Copy a file the archive holds to target, a binary file, never more
    bytes than it says it holds (its checksum is checked at its end)."""
    copied = 0
    with archive.open(info) as source:
        while chunk := source.read(1 << 20):
            copied += len(chunk)
            if copied > info.file_size:
                raise PersonaFileError(TOO_LARGE)
            target.write(chunk)
    return copied


def _read(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    """A text file the archive holds (of MAX_TEXT bytes at most)."""
    kept = io.BytesIO()
    _copy(archive, info, kept)
    try:
        return kept.getvalue().decode("utf-8")
    except UnicodeDecodeError as e:
        raise PersonaFileError(UNREADABLE) from e


def _yaml(text: str) -> dict:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise PersonaFileError(UNREADABLE) from e
    if not isinstance(data, dict):
        raise PersonaFileError(UNREADABLE)
    return data


def _text(data: dict, key: str, required: bool = False, line: bool = True) -> str:
    """A setting that is text: a line of a few words (line), or any."""
    value = data.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise PersonaFileError(UNREADABLE)
    if line and (len(value) > MAX_WORDS or not value.isprintable()):
        raise PersonaFileError(UNREADABLE)
    return value.strip()


def _settings(text: str) -> dict:
    """The persona a persona file holds, as the persona editor saves one,
    once each setting is checked."""
    data = _yaml(text)
    version = data.get("format")
    if type(version) is not int or version < 1:
        raise PersonaFileError(NOT_A_PERSONA)
    if version > FORMAT:
        raise PersonaFileError(NEWER)
    settings = {
        "name": _text(data, "name", required=True),
        "engine": _text(data, "engine", required=True),
        "voice": _text(data, "voice", required=True),
        "audio_filter": _text(data, "audio_filter"),
        "wake_word": _text(data, "wake_word"),
        "sleep_word": _text(data, "sleep_word"),
        "description": _text(data, "description", line=False),
    }
    speed = data.get("speed", 1.0)
    if type(speed) not in (int, float) or not math.isfinite(speed):
        raise PersonaFileError(UNREADABLE)
    if not SPEEDS[0] <= speed <= SPEEDS[1]:
        raise PersonaFileError(UNREADABLE)
    settings["speed"] = float(speed)
    if settings["audio_filter"] and settings["audio_filter"] not in AUDIO_FILTERS:
        raise PersonaFileError(UNREADABLE)
    engine, voice = settings["engine"], settings["voice"]
    if engine not in engines.speech_engines():
        raise PersonaFileError(f"The persona speaks with {engine}, which this Iris does not have")
    if not _known_voice(engine, voice):
        raise PersonaFileError(UNREADABLE)
    return settings


def _known_voice(engine: str, voice: str) -> bool:
    """Whether a persona file names a voice Iris has, or its own cloned one:
    never a path or an address, which Pocket TTS would read or fetch."""
    if engine == PocketGenerator.id:
        if voice.startswith(cloning.PREFIX):
            return True
        return voice in PocketGenerator.catalogue()
    if engine == "kokoro":
        return bool(KOKORO_VOICE_NAME.fullmatch(voice))
    listed = engines.speech_engine(engine).voices()
    return any(listed_voice.get("id") == voice for listed_voice in listed)


def _add_voice(archive: zipfile.ZipFile, entries: dict[str, zipfile.ZipInfo], base: str) -> Path:
    """Put the cloned voice of a persona file together in a folder of its
    own, check all of it, and add it to the library.

    Returns:
        Its folder
    """
    if not {VOICE_DIR + cloning.INFO_FILE, VOICE_DIR + cloning.SAMPLE_FILE} <= set(entries):
        raise PersonaFileError(VOICE_DAMAGED)
    try:
        known = _yaml(_read(archive, entries.pop(VOICE_DIR + cloning.INFO_FILE)))
        name = _text(known, "name", required=True)
    except PersonaFileError as e:
        raise PersonaFileError(VOICE_DAMAGED) from e
    library = cloning.voices_dir()
    there = library.exists()
    library.mkdir(parents=True, exist_ok=True)
    made = Path(tempfile.mkdtemp(prefix=".import-", dir=library))
    try:
        for entry, info in entries.items():
            with open(made / entry[len(VOICE_DIR) :], "wb") as target:
                _copy(archive, info, target)
        _check_sample(made / cloning.SAMPLE_FILE)
        for state in made.glob("*.safetensors"):
            _check_state(state)
        return cloning.add_voice(made, base, name, personas.voice_in_use)
    except BaseException:
        shutil.rmtree(made, ignore_errors=True)
        if not there:
            # Unless a voice cloned meanwhile is in it
            with contextlib.suppress(OSError):
                library.rmdir()
        raise


def _check_sample(path: Path):
    """That a voice's sample is one as Iris keeps them: a WAV file of one
    channel, of a sample's length, its sound all numbers."""
    try:
        info = sf.info(str(path))
        sound_ok = info.format == "WAV" and info.channels == 1 and info.samplerate >= 8000
        seconds = info.frames / info.samplerate if info.samplerate else 0.0
        if not sound_ok or not SAMPLE_SECONDS[0] <= seconds <= SAMPLE_SECONDS[1]:
            raise ValueError("not a sample")
        audio, _ = sf.read(str(path), dtype="float32")
        if not np.all(np.isfinite(audio)):
            raise ValueError("not a sample")
    except Exception as e:
        raise PersonaFileError(VOICE_DAMAGED) from e


def _check_state(path: Path):
    """That a language's voice is a voice state as Pocket TTS exports one
    and reads it: safetensors, each tensor its module's ("<module>/<name>"),
    the attention caches of its layers among them, all of it there, and its
    numbers finite."""
    try:
        caches = 0
        with safetensors.safe_open(str(path), framework="pt") as state:
            for key in state.keys():
                module, _, name = key.partition("/")
                if not module or not name or "/" in name:
                    raise ValueError(f"{key} is not a module's")
                tensor = state.get_tensor(key)
                if tensor.is_floating_point():
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError(f"{key} is not all numbers")
                    caches += name == "cache"
                elif tensor.dtype not in (torch.int64, torch.int32):
                    raise ValueError(f"{key} is not a voice state's")
        if not caches:
            raise ValueError("no attention cache")
    except Exception as e:
        raise PersonaFileError(VOICE_DAMAGED) from e

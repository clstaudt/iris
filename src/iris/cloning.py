"""Voices cloned from a sample of someone speaking, for Kyutai's Pocket TTS.

Pocket TTS clones a voice from a few seconds of speech, but only with the
form of its models that Kyutai hands out on Hugging Face to an account that
accepted its terms (kyutai/pocket-tts). The open form Iris fetches otherwise
(kyutai/pocket-tts-without-voice-cloning) is the same model with its audio
encoder zeroed: it speaks with voices made beforehand, and cannot make one.
A voice made with the first speaks with the second just as well, so the
cloning model is needed only to make a voice, once per language.

A cloned voice lives in a folder of its own under ~/.iris/voices
(IRIS_VOICES_DIR): its sample, trimmed and levelled (sample.wav), its name
(voice.yaml), and the voice each language's model made of the sample
(english.safetensors, french.safetensors...). English is made when the voice
is cloned; another language the first time the voice speaks it, with that
language's cloning model, fetched then. Personas name the voice
"clone:<folder>". It is the same voice in every language.

Cloning goes through KYUTAI: the Hugging Face login, the access to the gated
repo, and the model loaded from it. Tests put a stand-in in its place.
"""

import base64
import binascii
import contextlib
import io
import logging
import os
import re
import shutil
import threading
from collections.abc import Callable
from math import gcd
from pathlib import Path

import httpx
import numpy as np
import scipy.signal as signal
import soundfile as sf
import yaml
from huggingface_hub import auth_check, get_token, login, whoami
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from pocket_tts import TTSModel, export_model_state
from pocket_tts.utils.config import CONFIGS_DIR as POCKET_CONFIGS

from iris import downloads
from iris.audio import Recording
from iris.downloads import HubFile, Progress
from iris.events import brief
from iris.turns import FRAME, SPEECH_THRESHOLD, SpeechDetector
from iris.turns import SAMPLE_RATE as VAD_RATE

try:
    # Decodes what soundfile cannot (M4A, AAC); faster-whisper brings it
    import av

    HAS_AV = True
except ImportError:
    HAS_AV = False

logger = logging.getLogger(__name__)

# The gated repo, and the page where its terms are accepted
CLONING_REPO = "kyutai/pocket-tts"
MODEL_PAGE = f"https://huggingface.co/{CLONING_REPO}"

# How personas name a cloned voice, and the folder names it takes
PREFIX = "clone:"
FOLDER_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
SAMPLE_FILE = "sample.wav"
INFO_FILE = "voice.yaml"
# The language a cloned voice is listed under in the persona editor
LISTED_AS = "Cloned"

# Where cloned voices are kept, unless IRIS_VOICES_DIR names another folder
DEFAULT_DIR = Path.home() / ".iris" / "voices"

# The sample: Pocket TTS's rate, the speech it needs at least (it clones
# from 3 to 5 s, better from 10 to 30 s) and at most, the silence kept
# around the speech and in place of a long pause, and its level
RATE = 24000
MIN_SPEECH = 5.0
MAX_SAMPLE = 30.0
EDGE = 0.15
LONG_PAUSE = 0.7
KEPT_PAUSE = 0.35
# Speech goes on a little after the VAD's last frame of it (word endings)
HANGOVER = 0.2
TARGET_DB = -20.0
PEAK_DB = -1.0
MAX_GAIN_DB = 30.0
# A file chosen to clone from, at most
MAX_FILE_BYTES = 50_000_000

# What cloning can do here, as access() tells it
LOGIN, TERMS, READY = "login", "terms", "ready"


class CloneAccessError(RuntimeError):
    """The cloning model cannot be had: no Hugging Face login, or the account
    has not accepted Kyutai's terms. Its message is for the app."""


class SampleError(ValueError):
    """A sample no voice can be made from, with why, for the app."""


# ---------------------------------------------------------------- the voices


def voices_dir() -> Path:
    """The folder cloned voices are kept in."""
    return Path(os.environ.get("IRIS_VOICES_DIR") or DEFAULT_DIR)


def voice_folder(voice: str) -> Path | None:
    """The folder of a cloned voice, named as personas name it
    ("clone:<folder>") or given as the folder itself; None for any other
    voice."""
    voice = str(voice)
    if voice.startswith(PREFIX):
        name = voice[len(PREFIX) :]
        return voices_dir() / name if FOLDER_NAME.fullmatch(name) else None
    path = Path(voice)
    if path.is_absolute() and path.parent == voices_dir() and FOLDER_NAME.fullmatch(path.name):
        return path
    return None


def is_cloned(voice: str) -> bool:
    """Whether a voice is one cloned in the app."""
    return voice_folder(voice) is not None


def voice_name(folder: Path) -> str:
    """A cloned voice's name, as the editor shows it."""
    try:
        info = yaml.safe_load((folder / INFO_FILE).read_text()) or {}
        return str(info.get("name") or "").strip() or folder.name.title()
    except (OSError, yaml.YAMLError, AttributeError):
        return folder.name.title()


def describe(folder: Path) -> dict:
    """A cloned voice as the editor lists voices: {id, name, language}."""
    return {"id": PREFIX + folder.name, "name": voice_name(folder), "language": LISTED_AS}


def cloned_voices() -> list[dict]:
    """The voices cloned here, by folder name."""
    folder = voices_dir()
    if not folder.is_dir():
        return []
    return [
        describe(path)
        for path in sorted(folder.iterdir())
        if FOLDER_NAME.fullmatch(path.name) and (path / SAMPLE_FILE).is_file()
    ]


def slug(name: str) -> str:
    """A folder name for a voice named this (as personas are named)."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "voice"


# Folders of voices forgotten while the core runs are not taken again: a
# voice loaded under that folder may still be kept in memory
_forgotten: set[str] = set()


def make_voice(sample: np.ndarray, name: str) -> Path:
    """A folder for a new voice, named for the persona, holding its sample
    (at RATE) and its name. Another voice of that name gets a number."""
    base = slug(name)
    shown = name.strip() or "Cloned voice"
    folder, n = voices_dir() / base, 2
    while folder.exists() or folder.name in _forgotten:
        folder, n = voices_dir() / f"{base}-{n}", n + 1
    folder.mkdir(parents=True)
    if n > 2:
        shown = f"{shown} {n - 1}"
    sf.write(folder / SAMPLE_FILE, sample, RATE, subtype="PCM_16")
    (folder / INFO_FILE).write_text(yaml.safe_dump({"name": shown}, allow_unicode=True))
    return folder


def forget_voice(voice: str) -> bool:
    """Remove a cloned voice's folder. Returns whether one was removed."""
    folder = voice_folder(voice)
    if folder is None or not folder.is_dir():
        return False
    shutil.rmtree(folder, ignore_errors=True)
    _forgotten.add(folder.name)
    return True


# ------------------------------------------------------- Kyutai's gated model


class Kyutai:
    """Kyutai's cloning model as Hugging Face hands it out: the account logged
    in on this computer, its access to the gated repo, the model's files in
    the Hugging Face cache, and the model loaded from them."""

    def token(self) -> str | None:
        return get_token()

    def whoami(self) -> dict:
        """The account the token belongs to; raises if it is refused."""
        return whoami()

    def check_access(self):
        """Raises GatedRepoError unless the account may fetch the repo."""
        auth_check(CLONING_REPO)

    def login(self, token: str):
        """Keep the token as Hugging Face keeps it (its own token file, not
        Iris's settings); raises if it is refused. What huggingface_hub
        prints on the way is not passed on."""
        with contextlib.redirect_stdout(io.StringIO()):
            login(token=token, add_to_git_credential=False)

    def cached(self, file: HubFile) -> bool:
        return downloads.files_cached([file])

    def fetch(self, file: HubFile, on_progress: Progress):
        downloads.download_hub_files([file], on_progress)

    def load_model(self, pocket_lang: str):
        """The language's model; Pocket TTS loads the cloning form when its
        weights are in the cache (or can be fetched), the open one if not."""
        return TTSModel.load_model(language=pocket_lang)


KYUTAI = Kyutai()


def cloning_weights(pocket_lang: str) -> HubFile:
    """The file of a language's model that clones (Pocket TTS's own config
    names it, at a fixed revision)."""
    config = yaml.safe_load((POCKET_CONFIGS / f"{pocket_lang}.yaml").read_text())
    return HubFile.parse(config["weights_path"])


def weights_cached(pocket_lang: str = "english") -> bool:
    try:
        return KYUTAI.cached(cloning_weights(pocket_lang))
    except Exception:
        return False


def _status(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def _refused(error: Exception) -> bool:
    """Whether Hugging Face turned the account away from the gated repo."""
    return isinstance(error, RepositoryNotFoundError) or (
        isinstance(error, HfHubHTTPError) and _status(error) in (401, 403)
    )


def _offline(error: Exception) -> str:
    """Why Hugging Face could not be asked, in a few words."""
    if isinstance(error, HfHubHTTPError):
        status = _status(error)
        return f"Hugging Face answered with an error ({status})" if status else brief(error, 120)
    if isinstance(error, (httpx.TransportError, OSError)) or "offline" in str(error).lower():
        return "No connection to Hugging Face"
    return brief(error, 120)


def access(pocket_lang: str = "english") -> dict:
    """Whether voices can be cloned on this computer.

    Returns:
        {"state": "login" (no Hugging Face login, or its token is refused),
        "terms" (logged in, but the account has not accepted Kyutai's
        terms), or "ready"; "cached": whether the language's cloning model
        is downloaded; "error": what kept the state from being told, e.g.
        no connection, else ""}. Offline, a downloaded model is ready.
    """
    cached = weights_cached(pocket_lang)
    result = {"state": LOGIN, "cached": cached, "error": ""}

    def unknown(error: Exception) -> dict:
        """Hugging Face could not be asked: a downloaded model clones still."""
        if cached:
            return {**result, "state": READY}
        return {**result, "state": TERMS, "error": _offline(error)}

    if not KYUTAI.token():
        return result
    try:
        KYUTAI.whoami()
    except Exception as e:
        if isinstance(e, HfHubHTTPError) and _status(e) == 401:
            return {**result, "error": "Hugging Face did not accept the token"}
        return unknown(e)
    try:
        KYUTAI.check_access()
    except Exception as e:
        return {**result, "state": TERMS} if _refused(e) else unknown(e)
    return {**result, "state": READY}


def connect(token: str) -> dict:
    """Log in to Hugging Face with an access token, then tell access() as
    it is now. The token is kept by huggingface_hub alone, and neither
    returned nor logged, nor is any error that could hold it."""
    token = (token or "").strip()
    if not token:
        return {**access(), "error": "Paste an access token"}
    try:
        KYUTAI.login(token)
    except Exception as e:
        logger.warning("Logging in to Hugging Face failed (%s)", type(e).__name__)
        refused = isinstance(e, ValueError) or _status(e) in (401, 403)
        error = "Hugging Face did not accept the token" if refused else _offline(e)
        return {"state": LOGIN, "cached": weights_cached(), "error": error}
    return access()


def fetch_weights(pocket_lang: str, on_progress: Progress | None = None):
    """Download a language's cloning model unless it is in the cache.

    Raises:
        CloneAccessError: Hugging Face does not hand it out to this account
    """
    file = cloning_weights(pocket_lang)
    if KYUTAI.cached(file):
        return
    try:
        KYUTAI.fetch(file, on_progress or (lambda done, total: None))
    except Exception as e:
        if _refused(e):
            raise CloneAccessError(
                f"Kyutai's cloning model for {pocket_lang.title()} needs a Hugging Face login "
                "that accepted its terms"
            ) from e
        raise


def cloning_model(pocket_lang: str):
    """A language's model in the form that clones, fetched first if need be.

    Raises:
        CloneAccessError: It cannot be had with the access there is
    """
    fetch_weights(pocket_lang)
    model = KYUTAI.load_model(pocket_lang)
    if not getattr(model, "has_voice_cloning", False):
        raise CloneAccessError(
            f"Kyutai's cloning model for {pocket_lang.title()} could not be loaded"
        )
    return model


def _save_state(state: dict, path: Path):
    """Keep a voice as Pocket TTS exports one, under a temporary name first,
    so an interrupted write never leaves half a voice."""
    part = path.with_name(path.name + ".part")
    export_model_state(state, part)
    part.replace(path)


def voice_state(models: dict, pocket_lang: str, folder: Path):
    """A cloned voice for a language's model, as Pocket TTS speaks with it.

    Made once per language, with the language's cloning model, and kept in
    the voice's folder; from then on read from there, with whichever form of
    the model is loaded. The cloning model then replaces the open one in
    models (the generator's loaded models by language): both speak alike.

    Raises:
        FileNotFoundError: The voice's sample is gone
        CloneAccessError: The voice is new to the language, and its cloning
            model cannot be had
    """
    made = folder / f"{pocket_lang}.safetensors"
    model = models.get(pocket_lang)
    if made.is_file():
        if model is None:
            model = models[pocket_lang] = KYUTAI.load_model(pocket_lang)
        return model.get_state_for_audio_prompt(str(made))
    sample = folder / SAMPLE_FILE
    if not sample.is_file():
        raise FileNotFoundError(f"The cloned voice {voice_name(folder)} is gone")
    if model is None or not getattr(model, "has_voice_cloning", False):
        try:
            model = cloning_model(pocket_lang)
        except CloneAccessError as e:
            raise CloneAccessError(
                f"{voice_name(folder)} is a cloned voice: speaking {pocket_lang.title()} needs "
                "Kyutai's cloning model, and Hugging Face access to it. Connect again in the "
                "persona's voice."
            ) from e
        models[pocket_lang] = model
    state = model.get_state_for_audio_prompt(sample)
    _save_state(state, made)
    return state


# ---------------------------------------------------------------- the sample


def read_audio(data: bytes) -> tuple[np.ndarray, int]:
    """An audio file's sound, mixed to one channel, and its rate: WAV, FLAC,
    OGG, MP3 and AIFF through soundfile, the rest (M4A, AAC) through PyAV.

    Raises:
        SampleError: It cannot be read
    """
    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return audio.mean(axis=1), int(rate)
    except Exception:
        pass
    if HAS_AV:
        try:
            return _decode(data)
        except Exception:
            pass
    raise SampleError("Iris cannot read this file as audio")


def _decode(data: bytes) -> tuple[np.ndarray, int]:
    """Decode with PyAV (FFmpeg), to one channel at the file's own rate."""
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        rate = int(stream.rate or 48000)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks = []
        for frame in container.decode(stream):
            chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(frame)]
        chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(None)]
    if not chunks:
        raise ValueError("no audio")
    return np.concatenate(chunks).astype(np.float32), rate


def _resample(audio: np.ndarray, rate: int, to: int) -> np.ndarray:
    if rate == to:
        return audio.astype(np.float32)
    common = gcd(int(rate), int(to))
    return signal.resample_poly(audio, to // common, rate // common).astype(np.float32)


def speech_frames(audio: np.ndarray, rate: int, detector=None) -> np.ndarray:
    """Which of the VAD's frames (FRAME samples at 16 kHz, 32 ms) hold
    speech, carried on a little after each (word endings).

    Args:
        detector: Speech probability of a frame; Silero VAD if None
    """
    heard = _resample(audio, rate, VAD_RATE)
    detector = detector or SpeechDetector()
    count = len(heard) // FRAME
    speech = np.array(
        [detector(heard[i * FRAME : (i + 1) * FRAME]) >= SPEECH_THRESHOLD for i in range(count)],
        dtype=bool,
    )
    hang = int(round(HANGOVER * VAD_RATE / FRAME))
    carried = speech.copy()
    for i in np.flatnonzero(speech):
        carried[i : i + hang + 1] = True
    return carried


def prepare_sample(audio: np.ndarray, rate: int, detector=None) -> np.ndarray:
    """A sample as a voice is best cloned from: at RATE, from the first word
    to the last with a little silence around, long pauses shortened, at
    most MAX_SAMPLE long, and levelled.

    Args:
        audio: The recording or file, one channel
        rate: Its rate
        detector: Speech probability of a frame; Silero VAD if None

    Raises:
        SampleError: It holds less than MIN_SPEECH of speech
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(audio) or not np.all(np.isfinite(audio)):
        raise SampleError("Nothing was recorded")
    audio = audio - float(np.mean(audio))
    speech = speech_frames(audio, rate, detector)
    frame_seconds = FRAME / VAD_RATE
    spoken = float(speech.sum()) * frame_seconds
    if spoken < MIN_SPEECH:
        raise SampleError(
            f"Too short: {spoken:.0f} s of speech came through, "
            f"and a voice needs {MIN_SPEECH:.0f} s"
        )
    sample = _resample(audio, rate, RATE)

    def at(frame: int) -> int:
        """Where a VAD frame starts in the sample."""
        return int(round(frame * frame_seconds * RATE))

    # The speech, in runs of frames, with what lies between them
    runs, start = [], None
    for i, on in enumerate(np.append(speech, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    edge, kept = int(EDGE * RATE), int(KEPT_PAUSE * RATE)
    pieces, first, last = [], at(runs[0][0]), at(runs[-1][1])
    begin = max(0, first - edge)
    for (_, end), (following, _) in zip(runs, runs[1:]):
        gap = at(following) - at(end)
        if gap > LONG_PAUSE * RATE:
            pieces.append(sample[begin : at(end) + kept // 2])
            begin = at(following) - kept // 2
    pieces.append(sample[begin : min(len(sample), last + edge)])
    sample = np.concatenate([_faded(piece) for piece in pieces])
    if len(sample) > MAX_SAMPLE * RATE:
        sample = _faded(sample[: int(MAX_SAMPLE * RATE)])

    # Speech at TARGET_DB, its peaks below PEAK_DB, quiet ones raised at most MAX_GAIN_DB
    loud = np.sqrt(np.mean(sample**2)) if len(sample) else 0.0
    if loud <= 0:
        raise SampleError("Nothing was recorded")
    gain = min(10 ** ((TARGET_DB - 20 * np.log10(loud)) / 20), 10 ** (MAX_GAIN_DB / 20))
    sample = sample * gain
    peak = float(np.max(np.abs(sample)))
    ceiling = 10 ** (PEAK_DB / 20)
    if peak > ceiling:
        sample = sample * (ceiling / peak)
    return sample.astype(np.float32)


def _faded(piece: np.ndarray, seconds: float = 0.01) -> np.ndarray:
    """A piece with its ends faded in and out, so a cut does not click."""
    piece = piece.astype(np.float32).copy()
    n = min(len(piece) // 2, int(seconds * RATE))
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        piece[:n] *= ramp
        piece[-n:] *= ramp[::-1]
    return piece


# ------------------------------------------------- what the persona editor asks


class Cloning:
    """Voice cloning as the persona editor goes through it (the voice.*
    methods): access to the cloning model and a login for it, its download,
    a sample recorded or chosen, the voice made of it, and a voice dropped
    again. Pushes clone.progress {done, total} while the model downloads,
    and clone.level {level, seconds} while a sample is recorded."""

    def __init__(
        self,
        emit: Callable[[str, dict], None],
        assistant: Callable[[], object | None] = lambda: None,
        interrupt: Callable[[], object] = lambda: None,
        in_use: Callable[[str], bool] = lambda voice: False,
    ):
        """
        Args:
            emit: Pushes an event to the app
            assistant: The running assistant, if any: its microphone is
                recorded from, and it does not listen meanwhile
            interrupt: Stops what Iris says, before a recording
            in_use: Whether a persona speaks with a voice
        """
        self._emit = emit
        self._assistant = assistant
        self._interrupt = interrupt
        self._in_use = in_use
        self._lock = threading.Lock()
        self._fetching = threading.Lock()
        self._recording: Recording | None = None
        # The assistant held from listening while a sample is recorded, and
        # whether it was held already (the user typing)
        self._holding = None
        self._held_before = False
        self._sample: np.ndarray | None = None

    def access(self) -> dict:
        return access()

    def connect(self, token: str) -> dict:
        return connect(token)

    def fetch(self) -> dict:
        """Download the English cloning model, which a voice is first made
        with, unless it is there."""
        with self._fetching:
            try:
                fetch_weights("english", self._progress)
            except CloneAccessError as e:
                return {"ok": False, "error": str(e), "access": True}
            except Exception as e:
                return {"ok": False, "error": _offline(e)}
        return {"ok": True}

    def _progress(self, done: int, total: int):
        self._emit("clone.progress", {"done": done, "total": total})

    def record(self, on: bool, device: int | None = None, keep: bool = True) -> dict:
        """Start recording a sample, or stop. Iris does not listen meanwhile,
        and stops speaking. Stopped, the sample is prepared (or, keep false,
        dropped): {"ok", "seconds"} or {"ok": False, "error"}."""
        with self._lock:
            if on:
                return self._start(device)
            recording, self._recording = self._recording, None
            if recording is None:
                return {"ok": False, "error": "Nothing was recorded"}
            audio, rate = recording.stop()
            self._release()
            return self._take(audio, rate) if keep else {"ok": True}

    def _start(self, device: int | None) -> dict:
        if self._recording is not None:
            self._recording.stop()
            self._recording = None
            self._release()
        assistant = self._assistant()
        if assistant is not None:
            device = assistant.recorder.device
            self._holding, self._held_before = assistant, assistant.hold_event.is_set()
            assistant.hold_event.set()
            self._interrupt()
        recording = Recording(
            lambda data: self._emit("clone.level", data), device=device, on_end=self._release
        )
        try:
            recording.start()
        except Exception as e:
            self._release()
            return {"ok": False, "error": f"The microphone could not be opened: {brief(e)}"}
        self._recording = recording
        return {"ok": True}

    def _release(self):
        """Let Iris listen again, unless it was held before the recording."""
        assistant, self._holding = self._holding, None
        if assistant is not None and not self._held_before:
            assistant.hold_event.clear()

    def use_file(self, data: str, name: str = "") -> dict:
        """Prepare a sample from an audio file, sent as base64."""
        try:
            raw = base64.b64decode(data or "", validate=True)
        except (binascii.Error, ValueError):
            return {"ok": False, "error": "Iris cannot read this file as audio"}
        if len(raw) > MAX_FILE_BYTES:
            return {"ok": False, "error": "The file is too large: choose a recording under 50 MB"}
        try:
            audio, rate = read_audio(raw)
        except SampleError as e:
            return {"ok": False, "error": str(e)}
        with self._lock:
            return self._take(audio, rate)

    def _take(self, audio: np.ndarray, rate: int) -> dict:
        try:
            sample = prepare_sample(audio, rate)
        except SampleError as e:
            return {"ok": False, "error": str(e)}
        self._sample = sample
        return {"ok": True, "seconds": round(len(sample) / RATE, 1)}

    def clone(self, name: str) -> dict:
        """Make a voice of the sample prepared last, named for the persona:
        its folder, and its English made with the cloning model.

        Returns:
            {"ok", "voice": {id, name, language}}, or {"ok": False, "error",
            "access": True when it is the cloning model that cannot be had}
        """
        with self._lock:
            if self._sample is None:
                return {"ok": False, "error": "Record a sample or choose one first"}
            folder = make_voice(self._sample, name)
            try:
                voice_state({"english": cloning_model("english")}, "english", folder)
            except CloneAccessError as e:
                shutil.rmtree(folder, ignore_errors=True)
                return {"ok": False, "error": str(e), "access": True}
            except Exception as e:
                logger.exception("The voice could not be made")
                shutil.rmtree(folder, ignore_errors=True)
                return {"ok": False, "error": f"The voice could not be made: {brief(e)}"}
            self._sample = None
        return {"ok": True, "voice": describe(folder)}

    def forget(self, voice: str) -> dict:
        """Drop a cloned voice no persona speaks with (one made in the editor
        and not kept). One a persona speaks with stays."""
        if not is_cloned(voice) or self._in_use(voice):
            return {"ok": True, "forgotten": False}
        return {"ok": True, "forgotten": forget_voice(voice)}

    def replaced(self, previous: str, voice: str):
        """A persona saved with a new cloned voice in place of the one cloned
        before (cloned again): the old one goes, unless a persona speaks
        with it."""
        if previous != voice and is_cloned(previous) and is_cloned(voice):
            self.forget(previous)

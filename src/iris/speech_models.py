"""The speech models Iris downloads before it can listen and speak, and a
fetcher that brings them in the background, one after the other.

The voices come first, so they can be heard while the rest arrives: Pocket
TTS's English model with the voices of the personas that speak with it, and
Kokoro's model and voices. Then Smart Turn, which hears when the user has
finished, and the Whisper model that recognizes speech; with a wake word set,
also the smallest Whisper, which listens for it.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from iris import downloads
from iris.downloads import HubFile, Progress
from iris.personas import load_all_personas
from iris.transcription import model_repo, resolve_device, resolve_engine, resolve_model
from iris.tts import (
    DEFAULT_KOKORO_DIR,
    KOKORO_MODEL_FILE,
    KOKORO_MODEL_URL,
    KOKORO_VOICES_FILE,
    KOKORO_VOICES_URL,
    find_kokoro_models,
    pocket_files,
)
from iris.turns import SMART_TURN_FILE, SMART_TURN_REPO

# Rough download sizes in MB, shown before a download knows them exactly
WHISPER_MB = {
    "tiny": 75,
    "base": 145,
    "small": 480,
    "medium": 1500,
    "large-v3": 3100,
    "large-v3-turbo": 1600,
}
KOKORO_MB = 335
KYUTAI_MB = 220
KYUTAI_VOICE_MB = 6
TURN_MB = 9


@dataclass
class SpeechModel:
    """A model to download: what it is for, which one, and how to fetch it."""

    id: str
    #: What it does for Iris, e.g. "Speech recognition"
    label: str
    #: Which model, e.g. "Whisper large-v3-turbo"
    detail: str
    size_mb: int | None
    #: Shown under the orb while it downloads
    progress_label: str
    is_cached: Callable[[], bool]
    fetch: Callable[[Progress], object]
    #: What it is in a sentence, e.g. "speech recognition model"
    what: str = "speech model"
    #: Iris starts without it (without Smart Turn a silence ends a turn)
    optional: bool = False


def whisper(model: str = "auto", engine: str = "auto", device: str = "auto") -> SpeechModel | None:
    """The Whisper model that recognizes speech; None for a model folder,
    which is used as it is."""
    if Path(model).expanduser().is_dir():
        return None
    resolved = resolve_engine(engine)
    size = resolve_model(model, resolved, resolve_device(resolved, device))
    repo = model_repo(size, resolved)
    return SpeechModel(
        "stt",
        "Speech recognition",
        f"Whisper {size}",
        WHISPER_MB.get(size),
        "Downloading speech recognition model",
        lambda: downloads.is_cached(repo),
        lambda progress: downloads.download(repo, progress),
        what="speech recognition model",
    )


def wake(engine: str = "auto") -> SpeechModel:
    """The smallest Whisper, which listens for the wake word while Iris sleeps."""
    repo = model_repo("tiny", resolve_engine(engine))
    return SpeechModel(
        "wake",
        "Wake word",
        "Whisper tiny",
        WHISPER_MB["tiny"],
        "Downloading wake word model",
        lambda: downloads.is_cached(repo),
        lambda progress: downloads.download(repo, progress),
        what="wake word model",
    )


def turn() -> SpeechModel:
    """Smart Turn, which hears when the user has finished speaking."""
    return SpeechModel(
        "turn",
        "Turn detection",
        "Smart Turn v3",
        TURN_MB,
        "Downloading turn detection model",
        lambda: downloads.is_cached(SMART_TURN_REPO, SMART_TURN_FILE),
        lambda progress: downloads.download(SMART_TURN_REPO, progress, filename=SMART_TURN_FILE),
        what="turn detection model",
        optional=True,
    )


def kokoro(folder: str | None = None) -> SpeechModel | None:
    """Kokoro's model and voices; None when a folder holding them was chosen."""
    if folder:
        return None
    return SpeechModel(
        "kokoro",
        "Kokoro voices",
        "Kokoro v1.0",
        KOKORO_MB,
        "Downloading voice model",
        lambda: find_kokoro_models() is not None,
        lambda progress: downloads.download_files(
            [
                (KOKORO_MODEL_URL, DEFAULT_KOKORO_DIR / KOKORO_MODEL_FILE),
                (KOKORO_VOICES_URL, DEFAULT_KOKORO_DIR / KOKORO_VOICES_FILE),
            ],
            progress,
        ),
        what="Kokoro voice model",
    )


def kyutai_voices() -> list[str]:
    """The Pocket TTS voices of the personas, fetched with its model."""
    voices = {p.voice for p in load_all_personas().values() if p.engine == "kyutai"}
    return sorted(voices) or ["anna"]


def kyutai(voices: list[str] | None = None) -> SpeechModel:
    """Pocket TTS's English model and these voices of it (the personas')."""
    voices = kyutai_voices() if voices is None else voices

    def files() -> list[HubFile]:
        return [HubFile.parse(path) for path in pocket_files(voices)]

    return SpeechModel(
        "kyutai",
        "Kyutai voices",
        "Pocket TTS",
        KYUTAI_MB + KYUTAI_VOICE_MB * len(voices),
        "Downloading voice model",
        lambda: downloads.files_cached(files()),
        lambda progress: downloads.download_hub_files(files(), progress),
        what="Kyutai voice model",
    )


def needed(
    whisper_model: str = "auto",
    whisper_engine: str = "auto",
    whisper_device: str = "auto",
    recognition_engine: str = "whisper",
    wake_word: str | None = None,
    tts_dir: str | None = None,
) -> list[SpeechModel]:
    """The speech models Iris needs with these choices, in the order they are
    fetched: the voices first. A model folder chosen instead of a download
    leaves its model out, and so does another recognition engine than
    Whisper, which fetches its own."""
    models = [kyutai(), kokoro(tts_dir), turn()]
    if recognition_engine == "whisper":
        models.append(whisper(whisper_model or "auto", whisper_engine, whisper_device))
        if wake_word:
            models.append(wake(whisper_engine))
    return [model for model in models if model is not None]


def failure(error: Exception) -> str:
    """Why a download failed, in a few words for the app."""
    text = " ".join(str(error).split()) or type(error).__name__
    lowered = text.lower()
    if any(
        word in lowered
        for word in ("connection", "resolve", "timed out", "timeout", "network", "offline")
    ):
        return "No connection"
    return text if len(text) <= 120 else text[:117] + "..."


class Fetcher:
    """Brings the speech models in the background, one after the other, and
    keeps each one's state for the app: waiting, downloading (with the bytes
    so far), done, or failed (with why). Started again, it fetches what is
    still missing, the failed ones included, and takes up models added since
    (a wake word chosen later) between two downloads."""

    def __init__(
        self,
        lock: threading.RLock,
        on_change: Callable[[list[dict], bool], None],
        on_progress: Callable[[SpeechModel | None, int, int], None] | None = None,
    ):
        """
        Args:
            lock: Held while fetching; whatever else downloads models holds
                it too, so no file is fetched twice at once
            on_change: Told every state (as states() gives them) and whether
                it is still fetching, whenever one changes
            on_progress: Told the model downloading and its bytes, done and
                total; the model is None once all are fetched
        """
        self._lock = lock
        self._on_change = on_change
        self._on_progress = on_progress
        self._guard = threading.Lock()
        self._choose: Callable[[], list[SpeechModel]] = list
        self._models: list[SpeechModel] = []
        self._states: dict[str, dict] = {}
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def start(self, choose: Callable[[], list[SpeechModel]]) -> list[dict]:
        """Fetch what choose() names and is missing; returns the states."""
        with self._guard:
            self._choose = choose
            self._refresh(retry=True)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
        self._changed()
        return self.states()

    def states(self) -> list[dict]:
        """Each model, in the order it is fetched, as the app shows it."""
        with self._guard:
            return [
                {
                    "id": model.id,
                    "label": model.label,
                    "detail": model.detail,
                    "size_mb": model.size_mb,
                    **self._states[model.id],
                }
                for model in self._models
            ]

    def describe(self, models: list[SpeechModel]) -> list[dict]:
        """These models' states without fetching them: what is known of
        them, else whether they are downloaded already."""
        with self._guard:
            known = dict(self._states)
        return [
            {
                "id": model.id,
                "label": model.label,
                "detail": model.detail,
                "size_mb": model.size_mb,
                **known.get(model.id, self._first_state(model)),
            }
            for model in models
        ]

    @staticmethod
    def _first_state(model: SpeechModel) -> dict:
        try:
            cached = model.is_cached()
        except Exception:
            cached = False
        return {"state": "done" if cached else "waiting", "done": 0, "total": 0, "error": ""}

    def _refresh(self, retry: bool = False):
        """Take up the models chosen now (called with the guard held)."""
        self._models = self._choose()
        for model in self._models:
            state = self._states.get(model.id)
            if state is None or (retry and state["state"] == "failed"):
                self._states[model.id] = self._first_state(model)

    def _changed(self):
        self._on_change(self.states(), self.busy)

    def _set(self, model: SpeechModel, **state):
        with self._guard:
            self._states[model.id].update(state)

    def _progress(self, model: SpeechModel, done: int, total: int):
        self._set(model, done=done, total=total)
        if self._on_progress:
            self._on_progress(model, done, total)
        self._changed()

    def _next(self) -> SpeechModel | None:
        with self._guard:
            self._refresh()
            model = next(
                (m for m in self._models if self._states[m.id]["state"] == "waiting"), None
            )
            if model is None:
                self._thread = None
            else:
                self._states[model.id].update(state="downloading", done=0, total=0, error="")
            return model

    def _run(self):
        with self._lock:
            while (model := self._next()) is not None:
                self._changed()
                try:
                    if not model.is_cached():
                        model.fetch(lambda done, total, m=model: self._progress(m, done, total))
                    self._set(model, state="done")
                except Exception as error:
                    self._set(model, state="failed", error=failure(error))
        if self._on_progress:
            self._on_progress(None, 0, 0)
        self._changed()

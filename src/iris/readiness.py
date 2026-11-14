"""Making voices ready to speak a language, where the user sees it.

A voice speaks a language once what it needs is here: for one of Kyutai's
voices, the language's model and the voice as that model reads it; for a
cloned voice, besides, its version for the language, made from its sample
with the language's cloning model; for Kokoro's, its model and voices.
Speaking never fetches or makes any of it (iris.tts loads Pocket TTS's
files from the cache only): a reply in a language its voice is not ready
for is shown, and spoken from the moment the voice is ready.

Voices are made ready here instead, one job after the other: ahead, for the
languages chosen in the app, for every persona's voice; for a voice chosen
or cloned since; for a voice played that is not here yet; and when a
conversation comes to a language its voice is not ready for. Every job is
shown while it runs (voice.job events): what it downloads or makes, its
progress, what it is for, a download that has stalled (no byte for
STALL_AFTER seconds), and when it is done or failed, to retry. The user
never waits on a download without knowing it.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from iris import cloning, engines
from iris.engines import Step
from iris.events import brief
from iris.tts import LANGUAGE_NAMES, voice_ready

logger = logging.getLogger(__name__)

# A download that has received nothing for this long is said to have stalled
STALL_AFTER = 20.0
# How often the stall is looked for, in seconds
WATCH_EVERY = 1.0
# Jobs done or failed kept for the app, the latest
KEPT = 40

# Why a job was started, for where the app shows it: the conversation shows
# its own in the transcript; every running job shows under the orb
CONVERSATION = "conversation"
LANGUAGES = "languages"
VOICES = "voices"
PREVIEW = "preview"
CLONE = "clone"

QUEUED, DOWNLOADING, MAKING, READY, FAILED = "queued", "downloading", "making", "ready", "failed"


@dataclass
class Job:
    """Making one voice ready for one language."""

    engine: str
    voice: str
    language: str
    #: The voice as the app names it
    name: str
    #: The personas that speak with it, by name
    personas: list[str]
    origin: str
    state: str = QUEUED
    step: Step = field(default_factory=lambda: Step("download", "model"))
    done: int = 0
    total: int = 0
    stalled: bool = False
    error: str = ""
    #: When the last byte came, or the step began (time.monotonic())
    moved: float = field(default_factory=time.monotonic)
    #: Asked to be tried again while it still runs (a stall)
    again: bool = False
    #: Set once it is ready or has failed
    finished: threading.Event = field(default_factory=threading.Event)

    @property
    def id(self) -> str:
        return key(self.engine, self.voice, self.language)

    @property
    def language_name(self) -> str:
        return LANGUAGE_NAMES.get(self.language, self.language)

    def title(self) -> str:
        """What is going on, e.g. "Downloading the German voice model"."""
        language = self.language_name
        engine_name = engines.speech_engines().get(self.engine)
        engine_name = engine_name.name if engine_name else self.engine
        whose = _possessive(self.name)
        if self.state == READY:
            return f"{whose} voice is ready for {language}"
        if self.state == FAILED:
            if self.step.stage == "make":
                return f"{whose} voice could not be made for {language}"
            return f"{whose} voice for {language} could not be downloaded"
        if self.step.stage == "make":
            return f"Making {whose} voice for {language}"
        if self.step.what == "cloning":
            return f"Downloading Kyutai's cloning model for {language}"
        if self.step.what == "voice":
            return f"Downloading {whose} voice for {language}"
        if not self.language:
            return f"Downloading the {engine_name} voice model"
        return f"Downloading the {language} voice model"

    def purpose(self) -> str:
        """What it is for, e.g. "Iris will speak German once it's done"."""
        who = _together(self.personas) or self.name
        if self.state in (READY, FAILED):
            return ""
        if self.origin == PREVIEW and not self.personas:
            return f"{self.name} plays once it's done"
        if not self.language:
            return f"{who} will speak once it's done"
        return f"{who} will speak {self.language_name} once it's done"

    def describe(self) -> dict:
        return {
            "id": self.id,
            "engine": self.engine,
            "voice": self.voice,
            "language": self.language,
            "name": self.name,
            "personas": list(self.personas),
            "origin": self.origin,
            "state": self.state,
            "stage": self.step.stage,
            "what": self.step.what,
            "done": self.done,
            "total": self.total,
            "stalled": self.stalled,
            "error": self.error,
            "title": self.title(),
            "purpose": self.purpose(),
        }


def key(engine: str, voice: str, language: str) -> str:
    return f"{engine}|{voice}|{language}"


def _possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _together(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def failure(error: Exception) -> str:
    """Why a job failed, in a few words for the app."""
    if isinstance(error, cloning.CloneAccessError):
        return str(error)
    text = brief(error, 160)
    lowered = text.lower()
    offline = ("connection", "resolve", "timed out", "network", "offline")
    if any(word in lowered for word in offline):
        return "No connection"
    return text


class Readier:
    """Makes voices ready, one job at a time, and tells the app each one's
    state as it changes (voice.job {job}).

    A job is asked for by voice and language: one that is ready already
    needs none; one asked for again while it waits or runs is the same job
    (a conversation that comes to need it makes it the conversation's); one
    that failed is tried again."""

    def __init__(
        self,
        emit: Callable[[str, dict], None],
        lock: threading.RLock,
        models_dir: Callable[[], str | None] = lambda: None,
        stall_after: float = STALL_AFTER,
    ):
        """
        Args:
            emit: Pushes an event to the app
            lock: Held while a job runs; whatever else downloads models
                holds it too, so no file is fetched twice at once
            models_dir: The voice model folder chosen in the app, if any
            stall_after: Seconds without a byte before a download is said
                to have stalled
        """
        self._emit = emit
        self._lock = lock
        self._models_dir = models_dir
        self._stall_after = stall_after
        self._guard = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._queue: list[Job] = []
        self._wake = threading.Condition(self._guard)
        self._worker: threading.Thread | None = None
        self._watcher: threading.Thread | None = None

    # ---------------------------------------------------------------- asking

    def ready(self, engine: str, voice: str, language: str) -> bool:
        return voice_ready(engine, voice, language, self._models_dir())

    def request(
        self,
        engine: str,
        voice: str,
        language: str,
        name: str,
        personas: list[str] | None = None,
        origin: str = VOICES,
    ) -> dict | None:
        """Make a voice ready for a language, unless it is.

        Returns:
            The job (as describe() gives it), or None when it is ready
        """
        engine_class = engines.speech_engines().get(engine)
        if engine_class is None or not engines.voice_speaks(engine, voice, language):
            return None
        if self.ready(engine, voice, language):
            return None
        # Kokoro's one download serves every language
        if not self._per_language(engine):
            language = ""
        with self._guard:
            job = self._jobs.get(key(engine, voice, language))
            if job is not None and job.state not in (READY, FAILED):
                if origin == CONVERSATION:
                    job.origin = CONVERSATION
                if personas:
                    job.personas = sorted(set(job.personas) | set(personas))
                found = job
            else:
                found = Job(engine, voice, language, name, sorted(set(personas or [])), origin)
                self._jobs[found.id] = found
                self._queue.append(found)
                self._start()
                self._wake.notify_all()
        self._changed(found)
        return found.describe()

    def retry(self, job_id: str) -> dict | None:
        """Try a failed job again; one that has stalled is tried again as
        soon as its download gives up (a connection silent for 10 s, see
        iris.downloads)."""
        with self._guard:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state == FAILED:
                fresh = Job(job.engine, job.voice, job.language, job.name, job.personas, job.origin)
                self._jobs[fresh.id] = fresh
                self._queue.append(fresh)
                self._start()
                self._wake.notify_all()
                job = fresh
            elif job.state != READY:
                job.again = True
        self._changed(job)
        return job.describe()

    def jobs(self) -> list[dict]:
        """Every job the app should know of: those waiting and running, and
        the latest done or failed."""
        with self._guard:
            return [job.describe() for job in self._jobs.values()]

    def find(self, engine: str, voice: str, language: str) -> Job | None:
        if not self._per_language(engine):
            language = ""
        with self._guard:
            return self._jobs.get(key(engine, voice, language))

    def wait(self, job_id: str, timeout: float | None = None) -> bool:
        """Wait for a job to end; whether it made the voice ready."""
        with self._guard:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        job.finished.wait(timeout)
        return job.state == READY

    @staticmethod
    def _per_language(engine: str) -> bool:
        return engine != "kokoro"

    # ---------------------------------------------------------------- running

    def _start(self):
        """Start the worker and the stall watch, unless they run (guard held)."""
        if self._worker is None:
            self._worker = threading.Thread(target=self._work, daemon=True)
            self._worker.start()
        if self._watcher is None:
            self._watcher = threading.Thread(target=self._watch, daemon=True)
            self._watcher.start()

    def _next(self) -> Job | None:
        with self._guard:
            if not self._queue:
                self._worker = None
                return None
            return self._queue.pop(0)

    def _work(self):
        while (job := self._next()) is not None:
            self._run(job)

    def _run(self, job: Job):
        engine = engines.speech_engines().get(job.engine)
        with self._lock:
            # Ready meanwhile (another job fetched the same model)
            language = job.language or "en"
            if engine is not None and self.ready(job.engine, job.voice, language):
                self._finish(job, READY)
                return
            self._set(job, state=DOWNLOADING, moved=time.monotonic())
            try:
                if engine is None:
                    raise ValueError(f"No speech engine {job.engine}")
                engine.make_ready(
                    job.voice,
                    language,
                    self._models_dir(),
                    on_progress=lambda done, total: self._progress(job, done, total),
                    on_step=lambda step: self._step(job, step),
                )
            except Exception as error:
                logger.warning("Could not make %s ready for %s: %s", job.voice, language, error)
                again = job.again
                self._finish(job, FAILED, failure(error))
                if again:
                    self.retry(job.id)
                return
        self._finish(job, READY)

    def _step(self, job: Job, step: Step):
        state = MAKING if step.stage == "make" else DOWNLOADING
        self._set(job, step=step, state=state, done=0, total=0, moved=time.monotonic())

    def _progress(self, job: Job, done: int, total: int):
        with self._guard:
            moved = done != job.done or total != job.total
            job.done, job.total = done, total
            if moved:
                job.moved = time.monotonic()
                job.stalled = False
        self._changed(job)

    def _set(self, job: Job, **changes):
        with self._guard:
            for name, value in changes.items():
                setattr(job, name, value)
            if "stalled" not in changes:
                job.stalled = False
        self._changed(job)

    def _finish(self, job: Job, state: str, error: str = ""):
        with self._guard:
            job.state, job.error, job.stalled, job.again = state, error, False, False
            # The latest done or failed are kept, for an app that comes later
            ended = [j for j in self._jobs.values() if j.state in (READY, FAILED)]
            for old in ended[:-KEPT]:
                self._jobs.pop(old.id, None)
        job.finished.set()
        self._changed(job)

    def _watch(self):
        """Say when a download has received nothing for a while."""
        while True:
            time.sleep(WATCH_EVERY)
            stalled = []
            with self._guard:
                running = [j for j in self._jobs.values() if j.state == DOWNLOADING]
                if not running and self._worker is None:
                    self._watcher = None
                    return
                for job in running:
                    quiet = time.monotonic() - job.moved
                    if not job.stalled and quiet >= self._stall_after:
                        job.stalled = True
                        stalled.append(job)
            for job in stalled:
                logger.warning("The download for %s has stalled", job.id)
                self._changed(job)

    def _changed(self, job: Job):
        self._emit("voice.job", job.describe())

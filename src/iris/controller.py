"""GUI-facing controller: runs the VoiceAssistant off the main thread and
exposes commands (start/stop/interrupt/send_text) and a single effective
state stream for external UIs such as the Electron shell."""

import importlib.metadata
import itertools
import logging
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

from iris import cloning, downloads, engines, readiness, servers, speech_models
from iris.assistant import VoiceAssistant
from iris.attachments import AttachmentError, prepare_all
from iris.audio import AUDIO_FILTERS, AudioPlayer, AudioRecorder, SoundCheck, get_filter
from iris.events import EventBus, brief
from iris.llm import ConversationManager, effective_host, normalize_host, ollama_client
from iris.personas import (
    delete_persona,
    find_persona,
    get_persona,
    list_personas,
    persona_stamp,
    reload_personas,
    save_persona,
    shipped,
    speaking_with,
    voice_in_use,
    voice_users,
)
from iris.llm_client import is_openai, resolve_host
from iris.roles import delete_role, find_role, reload_roles, save_role
from iris.plugins import load as load_plugins
from iris.plugins import plugin_dir
from iris.transcription import (
    WHISPER_SIZES,
    WhisperTranscriber,
    folder_engine,
    model_repo,
    resolve_device,
    resolve_engine,
    resolve_model,
)
from iris.speech_models import KOKORO_MB, WHISPER_MB
from iris.tts import (
    KOKORO_MODEL_FILE,
    KOKORO_VOICES_FILE,
    LANGUAGE_NAMES,
    POCKET_LANGUAGES,
    PocketGenerator,
    SpeechGenerator,
    SpeechPipeline,
    find_kokoro_models,
    in_order,
    kokoro_files_in,
    list_voices,
    voice_language,
    voice_languages,
    voice_ready,
)

logger = logging.getLogger(__name__)

# How long a voice played that is not here yet is waited for, in seconds
PREVIEW_WAIT = 900.0


def _shown_path(folder: Path) -> str:
    """A folder as the app shows it, the home folder as ~."""
    try:
        return str(Path("~") / folder.relative_to(Path.home()))
    except ValueError:
        return str(folder)

# What a voice says when previewed, in its own language
PREVIEW_LINES = {
    "en": "Hi, I'm {name}. This is how I sound when we talk.",
    "fr": "Bonjour, je suis {name}. Voilà comment je parle.",
    "it": "Ciao, sono {name}. Ecco come parlo.",
    "es": "Hola, soy {name}. Así es como hablo.",
    "de": "Hallo, ich bin {name}. So klinge ich, wenn wir uns unterhalten.",
    "pt": "Olá, eu sou {name}. É assim que eu falo.",
    "nl": "Hallo, ik ben {name}. Zo klink ik als we praten.",
    "ja": "こんにちは、{name}です。こんな声で話します。",
    "zh": "你好，我是{name}。这就是我说话的声音。",
    "hi": "नमस्ते, मैं {name} हूँ। मेरी आवाज़ ऐसी है।",
}

# When several status flags are set simultaneously, the earliest match wins.
# Tool activity comes first: "thinking" stays set while a tool runs, and the
# specific activity is what the user needs to see.
FLAG_STATE_PRIORITY = (
    "looking",
    "capturing_screen",
    "searching",
    "browsing",
    "summarizing",
    "understanding",
    "thinking",
    "listening",
    "dormant",
)

# How long "speaking" survives a gap in playback, in seconds.
SPEAKING_HOLD = 0.6
# Seconds between looks at the file of the persona in use, which may change
# on disk while Iris runs
PERSONA_CHECK = 1.0

# Seconds a voice preview before Iris runs is followed at most: its engine
# loading, then the line said
PREVIEW_LIMIT = 90.0

# The model that answers about images when the conversation's model cannot
# see, if the server has it; else the first one that can see
VISION_MODEL = "qwen3-vl:8b"

# Events that make up the visible conversation. Kept here so a UI that
# reloads (or attaches late) can rebuild its transcript; streaming chunks are
# left out because chat.done carries the full reply.
TRANSCRIPT_EVENTS = ("chat.user", "chat.done", "chat.continued", "tool.use", "error", "notice")
TRANSCRIPT_LIMIT = 500
# Turn timings kept for the app
TIMINGS_KEPT = 20

# The assistant's loop, should it fail, is started again after a moment;
# failing more than this often within the window, it is left stopped (and
# told), until Resume or the microphone starts it again
LOOP_RESTARTS = 3
LOOP_RESTART_WINDOW = 60.0
LOOP_RESTART_DELAY = 0.2

# Seconds a reply stopped by opening another thread is given to end, so
# what was said of it is kept in the thread it was said in
REPLY_END_WAIT = 5.0

# Iris starts in the thread talked in last, unless told which one
LAST_THREAD = object()


def silent_microphone_message() -> str:
    """What to say when the microphone delivers nothing but exact zeros.

    On macOS that is the system withholding it from the app, which is asked
    for it under a name that differs in the app (Iris) and in development
    (the app that started the dev server). Elsewhere it is a muted or
    missing microphone.
    """
    if sys.platform == "darwin":
        if getattr(sys, "frozen", False):  # the packaged app's core
            return (
                "The microphone returns pure silence: macOS does not let Iris use it. "
                "Enable it for Iris under System Settings > Privacy & Security > "
                "Microphone, then reopen Iris."
            )
        return (
            "The microphone returns pure silence: macOS does not let this app use it. "
            "In development the permission belongs to the app that launched the dev "
            "server (Cursor, VS Code, Terminal or iTerm), not to Electron. Enable it "
            "under System Settings > Privacy & Security > Microphone, then relaunch "
            "that app completely."
        )
    return (
        "The microphone returns pure silence. Check that it is connected and not "
        "muted, and that Iris may use it."
    )


class AssistantController:
    """Owns a VoiceAssistant instance and translates it into bus events.

    The assistant's run_loop blocks on microphone input, so all lifecycle
    work happens on daemon threads. Commands return immediately; results and
    state changes arrive as bus events.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.assistant: VoiceAssistant | None = None
        # Plugins shipped, installed and in the user's folder; their engines
        # are registered from here on
        self.plugins = load_plugins()

        self._loop_thread: threading.Thread | None = None
        self._poller: threading.Thread | None = None
        self._start_lock = threading.Lock()
        # Whether the assistant's loop runs, and whether it failed again and
        # again and was left stopped; both changed under the start lock
        self._looping = False
        self._loop_failed = False
        # Held while recognition or the voice that did not load is loaded
        # again, so switching it on twice loads (and downloads) it once
        self._reloading = {"recognition": threading.Lock(), "voice": threading.Lock()}

        # Typed turns. Owned here rather than by the assistant so messages sent
        # while the models are still loading are answered once they are ready.
        self._turns: queue.Queue = queue.Queue()
        self._turn_ids = itertools.count(1)

        # Server chosen in the UI, as resolve_host keeps it; None follows OLLAMA_HOST
        self._ollama_host: str | None = None

        self._flags: dict = {}
        self._state_lock = threading.Lock()
        self._last_state = "idle"
        self._speaking_until = 0.0
        self._poller_stop = threading.Event()
        # The file of the persona in use as it was taken on, and when it
        # was last looked at
        self._persona_stamp: tuple | None = None
        self._persona_checked = 0.0

        # Silent-mic watchdog: a real microphone always has a noise floor, so
        # a stream of exact zeros means the OS is denying mic access (on macOS,
        # TCC denies silently when the host app lacks the microphone grant).
        self._zero_level_since: float | None = None
        self._silence_warned = False

        self._transcript: list[dict] = []
        self._transcript_lock = threading.Lock()
        # The latest turns' timings, for an app that attaches later
        self._timings: deque = deque(maxlen=TIMINGS_KEPT)

        # The first-run walkthrough, before Iris runs: the speech models
        # downloading meanwhile (held while anything downloads, so no file is
        # fetched twice), the choices it made so far, the microphone's sound
        # check, and a voice of its own for previews
        self._fetching = threading.RLock()
        self.fetcher = speech_models.Fetcher(
            self._fetching, self._models_changed, self._model_progress
        )
        self._walkthrough: dict = {}
        self._sound_check: SoundCheck | None = None
        self._preview: SpeechPipeline | None = None
        self._preview_lock = threading.Lock()
        # Voices cloned in the persona editor; a sample is recorded from
        # Iris's microphone, while it does not listen
        self.cloning = cloning.Cloning(
            bus.emit, lambda: self.assistant, self.interrupt, voice_in_use, self._voice_changed
        )
        # What plays a sample before Iris runs
        self._sample_player: AudioPlayer | None = None
        # Iris's languages, as chosen in the app: every persona's voice is
        # made ready for them ahead. They do not hold a conversation to them
        self.languages: list[str] = ["en"]
        # Voices made ready to speak a language, where the user sees it; the
        # preview last asked for, played once its voice is
        self.readier = readiness.Readier(bus.emit, self._fetching, self._tts_dir)
        self._preview_wanted: dict | None = None

        bus.subscribe(self._on_bus_event)

    # ------------------------------------------------------------------
    # Effective state computation (flags + speech pipeline activity)
    # ------------------------------------------------------------------

    def _on_bus_event(self, event_type: str, data: dict):
        if event_type in TRANSCRIPT_EVENTS:
            with self._transcript_lock:
                self._transcript.append({"type": event_type, "data": dict(data)})
                del self._transcript[:-TRANSCRIPT_LIMIT]
        elif event_type == "memory.forgotten":
            # Nor does a window that reloads show the conversation again
            with self._transcript_lock:
                self._transcript.clear()
        if event_type == "turn.timing":
            self._timings.append(dict(data))
        if event_type == "flags":
            with self._state_lock:
                self._flags = data.get("flags", {})
        elif event_type == "audio.level":
            self._watch_silence(data.get("rms", 0.0))

    def _watch_silence(self, rms: float):
        """Warn when the mic delivers only exact zeros for several seconds."""
        if rms > 0:
            self._zero_level_since = None
            self._silence_warned = False
            return
        if self._zero_level_since is None:
            self._zero_level_since = time.monotonic()
        elif not self._silence_warned and time.monotonic() - self._zero_level_since > 4:
            self._silence_warned = True
            self.bus.emit("error", {"message": silent_microphone_message()})

    def _flag_state(self) -> str:
        with self._state_lock:
            flags = dict(self._flags)
        for name in FLAG_STATE_PRIORITY:
            if flags.get(name):
                return name
        return "idle"

    def _publish_state(self, state: str):
        with self._state_lock:
            if state == self._last_state:
                return
            self._last_state = state
        self.bus.emit("state", {"state": state})

    def _speaking_poller(self):
        """Recompute the effective state at 10 Hz.

        Speaking is not a StatusManager flag (it derives from
        SpeechPipeline.is_active), so it must be polled. It also drops out
        briefly between sentences, hence the hold: without it the UI flickers
        back to thinking on every gap.
        """
        while not self._poller_stop.is_set():
            if self.assistant is not None:
                now = time.monotonic()
                if self.assistant.speech_pipeline.is_active:
                    self._speaking_until = now + SPEAKING_HOLD

                if self.assistant.suspend_event.is_set():
                    # Everything holds while the conversation is paused
                    state = "paused"
                elif now < self._speaking_until:
                    state = "speaking"
                else:
                    state = self._flag_state()
                # A closed mic only explains an otherwise idle assistant: typed
                # turns are still answered with the mic off, and that work shows
                if state == "idle":
                    if self._listening_paused():
                        state = "paused"
                    elif self.assistant.pause_event.is_set():
                        state = "muted"
                    elif self.assistant.hold_event.is_set():
                        state = "typing"
                self._publish_state(state)
                if now - self._persona_checked >= PERSONA_CHECK:
                    self._persona_checked = now
                    self._follow_persona_file()
            self._poller_stop.wait(0.1)

    def _follow_persona_file(self):
        """The file of the persona in use changed on disk while Iris ran
        (edited by hand, or replaced by the user's own copy): Iris takes it
        on, voice, personality and words, so it never speaks with another
        voice than the editor shows. Not while a reply is given: that one
        is heard to its end in its own voice, and the next takes the new."""
        assistant = self.assistant
        try:
            stamp = persona_stamp(assistant.persona.id)
            if stamp == self._persona_stamp or assistant.replying:
                return
            reload_personas()
            if find_persona(assistant.persona.id) is None:
                # Gone without a file to fall back to: Iris goes on as it is
                self._persona_stamp = stamp
                return
            self.set_persona(assistant.persona.id)
        except Exception:
            logger.exception("The persona's file could not be taken on")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def start(self, options: dict | None = None) -> dict:
        """Build the assistant and start its run_loop on a background thread.

        greet: Iris says hello once it listens (the end of the walkthrough).
        """
        options = dict(options or {})
        greet = bool(options.pop("greet", False))
        with self._start_lock:
            if self._loop_thread and self._loop_thread.is_alive():
                return {"ok": True, "already_running": True}
            self._release_first_run()
            if greet:
                self.greet()
            if self._loop_failed and self.assistant is not None:
                # The assistant left stopped after failing gives way to the new one
                self.assistant.stop_event.set()
                threading.Thread(
                    target=self._shut_down, args=(self.assistant,), daemon=True
                ).start()
            self._ollama_host = normalize_host(options.get("ollama_host"))
            self._loop_failed = False
            self._publish_state("starting")
            self._loop_thread = threading.Thread(
                target=self._run, args=(options,), daemon=True
            )
            self._loop_thread.start()
            self._start_poller()
        return {"ok": True}

    @staticmethod
    def _shut_down(assistant):
        try:
            assistant.shut_down()
        except Exception:
            logger.exception("The assistant before could not be shut down")

    def _start_poller(self):
        """Start computing the effective state, ending the poller before."""
        if self._poller is not None and self._poller.is_alive():
            self._poller_stop.set()
            self._poller.join(timeout=1.0)
        self._poller_stop.clear()
        self._poller = threading.Thread(target=self._speaking_poller, daemon=True)
        self._poller.start()

    def _loop_ended(self):
        """No loop runs: nothing polls the state, which is idle. Called
        under the start lock, so a command never shows a state of its own
        (paused, muted) after this."""
        self._poller_stop.set()
        if self._poller is not None and self._poller is not threading.current_thread():
            self._poller.join(timeout=1.0)
        self._publish_state("idle")

    def _revive(self):
        """Start the assistant's loop again if it was left stopped after
        failing again and again, so that Pause, Resume and the microphone
        act on a live loop."""
        with self._start_lock:
            if not self._loop_failed or self._looping or self.assistant is None:
                return
            # It is past its last use of the lock: it ends at once
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=1.0)
            self._loop_failed = False
            self._looping = True
            logger.info("Starting the assistant's loop again")
            self._loop_thread = threading.Thread(target=self._supervise, daemon=True)
            self._loop_thread.start()
            self._start_poller()

    def _pick_vision_model(self, preferred: str = VISION_MODEL) -> str | None:
        """Resolve the model used for image prompts.

        Returns None when no separate model is needed or available, which keeps
        the conversation on the primary model instead of switching to one that
        is not installed.
        """
        installed = ConversationManager.list_available_models(self._ollama_host)
        names = {m["name"] for m in installed}
        if preferred in names:
            return preferred
        for model in installed:
            if model["vision"]:
                return model["name"]
        return None

    def _run(self, options: dict):
        kwargs = dict(options)
        kwargs["ollama_host"] = self._ollama_host
        persona = get_persona(kwargs.pop("persona", "iris"))
        # The wake word is one switch; the words are the persona's. A word
        # given as it is (a test's) is used as given.
        wake_on = kwargs.pop("wake_on", None)
        if wake_on is not None:
            kwargs["wake_word"] = persona.wake_word if wake_on else None
        kwargs.setdefault("sleep_word", persona.sleep_word)
        # {plugin id: enabled} chosen in the UI; the others follow their default
        kwargs["tools"] = kwargs.pop("tools", {}) or {}
        # Memory's own switch; and amnesia, kept for every thread before each
        # had its own, for the thread Iris starts in
        remember = bool(kwargs.pop("remember", True))
        amnesia = bool(kwargs.pop("amnesia", False))
        # The thread to go on in: an id, or None for a new one with the
        # persona; without one, the thread talked in last
        thread = kwargs.pop("thread", LAST_THREAD)
        # The app still shows the conversation when it starts a core again
        # after one stopped: the recent turns are not shown a second time
        restore = bool(kwargs.pop("restore_thread", True))
        languages = kwargs.pop("languages", None)
        # Model folders chosen in Settings; empty means download as usual
        if stt_path := kwargs.pop("stt_model_path", ""):
            kwargs["whisper_model"] = stt_path
        if tts_dir := kwargs.pop("tts_models_dir", ""):
            kwargs["tts_models_dir"] = tts_dir

        if "vision_model" not in kwargs:
            kwargs["vision_model"] = self._pick_vision_model()

        try:
            self._fetch_models(kwargs)
        except Exception as e:
            self.bus.emit("error", {"message": f"Could not download the speech models: {e}"})

        try:
            # The persona's file as taken on, before Iris runs to compare it
            self._persona_stamp = persona_stamp(persona.id)
            self.assistant = VoiceAssistant(
                persona=persona,
                event_bus=self.bus,
                turn_queue=self._turns,
                plugins=self.plugins.plugins,
                **kwargs,
            )
            self.assistant.on_voice_not_ready = self._voice_not_ready
            self._enter_first_thread(thread)
            if not remember:
                self.assistant.conversation.set_remember(False)
            if amnesia:
                self.assistant.conversation.set_amnesia(True)
            self.bus.emit("config", self.assistant.config_info())
            if restore:
                self._restore_thread()
            self.bus.emit("threads.changed", {})
        except Exception as e:
            self.bus.emit("error", {"message": f"Failed to start assistant: {e}"})
            with self._start_lock:
                self._loop_ended()
            return

        # Every persona's voice is made ready for Iris's languages
        try:
            self.set_languages(self.languages if languages is None else languages)
        except Exception:
            logger.exception("Could not make the voices ready for Iris's languages")

        # Diagnostics only — must never prevent the assistant from running
        try:
            if self._warn_if_ollama_unreachable():
                self._warn_if_model_missing()
        except Exception:
            pass

        with self._start_lock:
            self._looping = True
        self._supervise()

    def _supervise(self):
        """Run the assistant's loop until Iris is stopped.

        The loop survives what goes wrong in it; should it fail all the same,
        it is started again, so that Pause, Resume and the microphone always
        steer a live loop. Failing more than LOOP_RESTARTS times within
        LOOP_RESTART_WINDOW, it is left stopped, the error is told, and Iris
        is idle until Resume or the microphone starts it again.
        """
        failures: deque[float] = deque()
        gave_up = False
        try:
            while True:
                try:
                    self.assistant.run_loop()
                    return
                except Exception as e:
                    logger.exception("The assistant's loop failed")
                    if self.assistant.stop_event.is_set():
                        return
                    now = time.monotonic()
                    failures.append(now)
                    while now - failures[0] > LOOP_RESTART_WINDOW:
                        failures.popleft()
                    if len(failures) > LOOP_RESTARTS:
                        gave_up = True
                        self.bus.emit("error", {"message": f"Assistant loop failed: {e}"})
                        return
                    time.sleep(LOOP_RESTART_DELAY)
        finally:
            with self._start_lock:
                self._looping = False
                self._loop_failed = gave_up
                self._loop_ended()

    def stop(self) -> dict:
        """Request shutdown. Takes effect after the current listen/turn ends."""
        if self.assistant is not None:
            self.assistant.stop_event.set()
            self.assistant.speech_pipeline.stop()
        self._poller_stop.set()
        self._publish_state("idle")
        return {"ok": True}

    def interrupt(self) -> dict:
        """Barge-in: stop speaking and cancel the current response."""
        if self.assistant is not None:
            self.assistant.interrupt_event.set()
            self.assistant.speech_pipeline.stop()
        return {"ok": True}

    def _show(self, state: str):
        """Show a state a command brought about at once, rather than at the
        poller's next look; only while a loop runs to be in it."""
        with self._start_lock:
            if self._looping:
                self._publish_state(state)

    def pause(self) -> dict:
        """Pause the conversation: the mic closes, speech holds mid-sentence,
        the reply stops streaming and queued turns wait. Models stay loaded.
        A loop left stopped after failing is started again, paused."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.suspend_event.set()
        self.assistant.speech_pipeline.pause()
        self._revive()
        self._show("paused")
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def resume(self) -> dict:
        """Continue the conversation where pause() held it, starting the
        loop again if it was left stopped after failing."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.suspend_event.clear()
        self._resume_listening()
        self.assistant.speech_pipeline.resume()
        self._revive()
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def _listening_paused(self) -> bool:
        """Listening is paused, as on opening another thread."""
        return bool(getattr(self.assistant, "listening_paused", False))

    def _resume_listening(self):
        if hasattr(self.assistant, "listening_paused"):
            self.assistant.listening_paused = False

    def set_mic(self, on: bool) -> dict:
        """Switch the microphone. Off still answers typed messages. Without
        speech recognition, which did not load at the start, it is loaded
        first, and the microphone opens once it is there. A loop left
        stopped after failing is started again."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        if on and self.assistant.transcriber is None:
            threading.Thread(target=self._load_recognition, daemon=True).start()
            self._revive()
            return {"ok": True}
        if on:
            self.assistant.pause_event.clear()
            # Switched on, it listens: also where opening a thread paused it
            self._resume_listening()
        else:
            self.assistant.pause_event.set()
        self._revive()
        if not on:
            self._show("muted")
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def send_text(self, text: str, attachments: list | None = None) -> dict:
        """Queue a typed message as the next turn of the conversation.

        Typed and spoken turns are taken one at a time by the assistant's loop,
        so a message sent mid-reply waits for that reply to finish. Its
        chat.user event (carrying the returned id) is emitted when it starts.
        Attachments ({name, mime, data: base64}) are read here, so a file iris
        cannot use is refused before the turn is queued.
        """
        text = (text or "").strip()
        try:
            files = prepare_all(attachments or [])
        except AttachmentError as e:
            return {"ok": False, "error": str(e)}
        if not text and not files:
            return {"ok": False, "error": "Empty message"}
        turn_id = f"t{next(self._turn_ids)}"
        self._turns.put({"text": text, "id": turn_id, "attachments": files})
        return {"ok": True, "id": turn_id}

    def hold_listening(self, on: bool) -> dict:
        """Close the mic while the user types; reopen it when they stop."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        if on:
            self.assistant.hold_event.set()
        else:
            self.assistant.hold_event.clear()
        return {"ok": True}

    def set_voice_output(self, enabled: bool) -> dict:
        """Switch spoken replies on or off. Switching off cuts current speech.
        A voice that did not load is tried again first, and speaks once it
        is there."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        if enabled and self.assistant.voice_error is not None:
            threading.Thread(target=self._load_voice, daemon=True).start()
            return {"ok": True}
        self.assistant.speak_replies = bool(enabled)
        if not enabled:
            self.assistant.speech_pipeline.stop()
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def _load_recognition(self):
        """Try again to load speech recognition (downloading it first), and
        open the microphone once it is there."""
        if not self._reloading["recognition"].acquire(blocking=False):
            return
        try:
            self._reload_recognition()
        finally:
            self._reloading["recognition"].release()

    def _reload_recognition(self):
        assistant = self.assistant
        options = assistant.recognition_options
        try:
            if assistant.recognition_engine == "whisper":
                self._fetch_stt(
                    options.get("model", "auto"),
                    whisper_engine=options.get("engine", "auto"),
                    whisper_device=options.get("device", "auto"),
                )
        except Exception:
            pass  # loading says why
        finally:
            self.bus.emit("setup.progress", {"label": "", "done": 0, "total": 0})
        if assistant.load_recognition():
            assistant.pause_event.clear()
        self.bus.emit("config", assistant.config_info())

    def _load_voice(self):
        """Try again to load the voice, and speak replies once it is there."""
        if not self._reloading["voice"].acquire(blocking=False):
            return
        try:
            assistant = self.assistant
            if assistant.load_voice():
                assistant.speak_replies = True
            self.bus.emit("config", assistant.config_info())
        finally:
            self._reloading["voice"].release()

    def set_barge_in(self, enabled: bool) -> dict:
        """Whether speaking interrupts the voice (with echo cancellation)."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.barge_in = bool(enabled)
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def set_tool(self, plugin_id: str, enabled: bool) -> dict:
        """Switch a plugin's tools (camera, screen, web, ...) on or off."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        if not self.assistant.tools.set_enabled(plugin_id, bool(enabled)):
            return {"ok": False, "error": f"Tool '{plugin_id}' is not available"}
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def plugins_list(self) -> dict:
        """The plugins found, what each brings, and those that failed to load."""
        tools = self.assistant.tools if self.assistant else None
        return {
            "plugins": [
                {
                    "id": plugin.id,
                    "name": plugin.name,
                    "source": plugin.source,
                    "available": plugin.available(),
                    "switchable": plugin.switchable,
                    "enabled_by_default": plugin.enabled_by_default,
                    "enabled": tools.enabled(plugin.id) if tools else None,
                    "tools": tools.names_of(plugin.id) if tools else [],
                    "speech_engines": [e.id for e in plugin.speech_engines()],
                    "recognition_engines": [e.id for e in plugin.recognition_engines()],
                }
                for plugin in self.plugins.plugins
            ],
            "failures": self.plugins.failures,
            "folder": str(plugin_dir()),
        }

    def engines_list(self) -> dict:
        """Each slot's engines, with the ones in use marked."""
        assistant = self.assistant
        return engines.describe(
            speech=assistant.tts_generator.engine if assistant else None,
            recognition=assistant.recognition_engine if assistant else None,
            models_dir=assistant.tts_generator.models_dir if assistant else None,
        )

    def set_wake_word(self, word: str) -> dict:
        """Listen only for this word until addressed ("" listens to everything)."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        try:
            self.assistant.set_wake_word(word)
        except Exception as e:
            return {"ok": False, "error": f"Wake word detection unavailable: {e}"}
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def set_wake(self, on: bool) -> dict:
        """Switch the wake word on (the persona's) or off."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        return self.set_wake_word(self.assistant.persona.wake_word if on else "")

    def set_wake_options(self, window=None, sleep_word=None) -> dict:
        """How long Iris stays awake after a turn, and a sleep word for now
        (the persona's is set with set_persona_words)."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.set_wake_options(window, sleep_word)
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def inject_audio(self, path: str) -> dict:
        """Debug: answer a recorded utterance as if the mic had heard it."""
        self._turns.put({"audio": path})
        return {"ok": True}

    def _restore_thread(self):
        """Show the recent part of the thread Iris is in: after a restart,
        or once it opened another thread.

        The turns are marked as restored: they come once the models are
        loaded, after what the app has said meanwhile (a microphone it may
        not use, say), and go above it.
        """
        memory = getattr(self.assistant, "memory", None)
        if memory is None:
            return
        for turn in memory.store.recent_turns(self.assistant.conversation.thread):
            if turn["role"] == "user":
                self.bus.emit(
                    "chat.user", {"text": turn["text"], "source": turn["source"], "restored": True}
                )
            elif turn["role"] == "assistant":
                self.bus.emit(
                    "chat.done", {"text": turn["text"], "interrupted": False, "restored": True}
                )

    # ------------------------------------------------------------------
    # Threads: a conversation with one persona, and its own memory
    # ------------------------------------------------------------------

    def _enter_first_thread(self, wanted):
        """Go on in the thread asked for at the start: the one talked in
        last, unless the app names one (that is still there), or None for a
        new thread with the persona Iris started with."""
        assistant = self.assistant
        if getattr(assistant, "memory", None) is None:
            return
        store = assistant.memory.store
        # The persona as its file says now
        reload_personas()
        found = None
        if wanted is not None:
            found = store.thread(wanted) if wanted is not LAST_THREAD else None
            found = found or store.last_thread()
        if found is None:
            return
        persona = self._thread_persona(found)
        if persona != assistant.persona.id:
            self.set_persona(persona, note=False)
        assistant.conversation.open_thread(found["id"], found["role"], found["amnesia"])

    @staticmethod
    def _thread_persona(thread: dict) -> str:
        """The persona a thread is with: Iris once its own is gone."""
        return thread["persona"] if find_persona(thread["persona"]) else "iris"

    def threads_list(self) -> dict:
        """The threads, the one talked in last first, and the one open (None
        for a new thread) while Iris runs."""
        memory = getattr(self.assistant, "memory", None)
        if memory is None:
            return {"threads": []}
        threads = []
        for thread in memory.store.threads():
            persona = self._thread_persona(thread)
            role = find_role(thread["role"])
            threads.append(
                {
                    **thread,
                    "persona": persona,
                    "persona_name": get_persona(persona).name,
                    "role": role.id if role else None,
                    "role_name": role.name if role else "",
                }
            )
        return {"threads": threads, "open": self.assistant.conversation.thread}

    def open_thread(self, thread_id) -> dict:
        """Go on in another thread, with its persona and its memory."""
        memory = getattr(self.assistant, "memory", None)
        if memory is None:
            return {"ok": False, "error": "Memory is off"}
        thread = memory.store.thread(int(thread_id))
        if thread is None:
            return {"ok": False, "error": f"No thread {thread_id}"}
        # The thread's persona as its file says now: it may have been edited
        # while another one spoke, which no look at the file in use noticed
        reload_personas()
        if thread["id"] == self.assistant.conversation.thread:
            return {"ok": True, "id": thread["id"]}  # open already: nothing changes
        self._go_to_thread(
            thread["id"], self._thread_persona(thread), thread["role"], thread["amnesia"]
        )
        self._switched()
        return {"ok": True, "id": thread["id"]}

    def new_thread(self, persona: str, role: str | None = None) -> dict:
        """Go on in a new thread with this persona, and this role if one is
        given. It is stored with its first turn; until then another persona
        or role can be chosen for it. Neither changes afterwards."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        reload_personas()  # the persona as its file says now
        if persona not in list_personas():
            return {"ok": False, "error": f"Unknown persona: {persona}"}
        if role and find_role(role) is None:
            return {"ok": False, "error": f"Unknown role: {role}"}
        # Leaving a thread for a new one is heard; choosing another persona
        # or role for the new one, before anything is said in it, is not
        leaving = self.assistant.conversation.thread is not None
        self._go_to_thread(None, persona, role or None)
        self._switched(chime=leaving)
        return {"ok": True}

    def forget_thread(self, thread_id) -> dict:
        """Delete a thread with its memory. Forgetting the open one leaves a
        new thread, with the same persona and role."""
        memory = getattr(self.assistant, "memory", None)
        if memory is None:
            return {"ok": False, "error": "Memory is off"}
        thread_id = int(thread_id)
        conversation = self.assistant.conversation
        if conversation.thread == thread_id:
            self._go_to_thread(None, self.assistant.persona.id, conversation.role)
        memory.store.forget_thread(thread_id)
        self.bus.emit("memory.updated", {})
        self.bus.emit("threads.changed", {})
        return {"ok": True}

    def _go_to_thread(
        self,
        thread_id: int | None,
        persona: str,
        role: str | None = None,
        amnesia: bool = False,
    ):
        """Leave the thread Iris is in for another (None: a new one). A reply
        underway stops, as Stop does, and is kept in the thread it was given
        in; messages waiting for it are dropped. Iris takes on the thread's
        persona, its role and its memory, and the app is shown the thread's
        recent turns."""
        assistant = self.assistant
        if assistant.replying:
            self.interrupt()
            deadline = time.monotonic() + REPLY_END_WAIT
            while assistant.replying and time.monotonic() < deadline:
                time.sleep(0.02)
        self._drop_waiting_turns()
        if persona != assistant.persona.id:
            self.set_persona(persona, note=False)
        assistant.conversation.open_thread(thread_id, role, amnesia)
        with self._transcript_lock:
            self._transcript = []
        self.bus.emit("thread.opened", {"id": thread_id})
        self._restore_thread()
        self.bus.emit("config", assistant.config_info())
        self.bus.emit("threads.changed", {})

    def _switched(self, chime: bool = True):
        """Another thread was opened, or a new one: a soft chime says so, and
        listening pauses, shown as Pause shows it, until Resume, so Iris does
        not take what is said around it for the thread. Typed messages are
        still answered."""
        if chime and hasattr(self.assistant, "thread_chime"):
            self.assistant.thread_chime()
        if not hasattr(self.assistant, "listening_paused"):
            return
        self.assistant.listening_paused = True
        self._show("paused")
        self.bus.emit("config", self.assistant.config_info())

    def _drop_waiting_turns(self):
        """Typed messages not answered yet belong to the thread left."""
        kept = []
        while True:
            try:
                turn = self._turns.get_nowait()
            except queue.Empty:
                break
            if "text" not in turn:
                kept.append(turn)
        for turn in kept:
            self._turns.put(turn)

    def memory_topics(self) -> dict:
        """The topics Iris remembers of the open thread, most recently
        discussed first, and how many turns it keeps in all, in every
        thread, filed under a topic or not yet."""
        memory = getattr(self.assistant, "memory", None)
        if memory is None:
            return {"topics": [], "turns": 0}
        thread = self.assistant.conversation.thread
        counts = memory.store.turn_counts(thread)
        return {
            "topics": [
                {
                    "id": t.id,
                    "title": t.title,
                    "gist": t.gist,
                    "updated": t.updated,
                    "turns": counts.get(t.id, 0),
                }
                for t in memory.store.topics(thread)
            ],
            "turns": memory.store.turn_count(),
        }

    def memory_topic(self, topic_id: int) -> dict:
        """One topic of the open thread with its summary and every turn filed
        under it."""
        memory = getattr(self.assistant, "memory", None)
        thread = self.assistant.conversation.thread if memory else None
        topic = memory.store.topic(int(topic_id), thread) if thread is not None else None
        if topic is None:
            return {"ok": False, "error": f"No topic {topic_id}"}
        return {
            "ok": True,
            "id": topic.id,
            "title": topic.title,
            "gist": topic.gist,
            "summary": topic.summary,
            "updated": topic.updated,
            "turns": topic.turns,
        }

    def set_remember(self, on: bool) -> dict:
        """Switch memory on or off, for every thread: off, nothing is
        written, and memory is neither in the prompt nor offered as tools."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.conversation.set_remember(on)
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def set_amnesia(self, on: bool) -> dict:
        """While on, nothing of the open thread is written to memory; kept
        with the thread."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.conversation.set_amnesia(on)
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def memory_forget(self, topic_id: int) -> dict:
        """Delete a topic and everything filed under it."""
        memory = getattr(self.assistant, "memory", None)
        if memory is None:
            return {"ok": False, "error": "Memory is off"}
        memory.store.forget(int(topic_id))
        self.bus.emit("memory.updated", {})
        return {"ok": True}

    def memory_forget_all(self) -> dict:
        """Forget everything: every thread and all that is remembered, and
        the conversation, in the model's context and in the app (see
        VoiceAssistant.forget_everything). Iris goes on in a new thread,
        with the same persona."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.forget_everything()
        return {"ok": True}

    def _progress(self, model: str, label: str):
        return lambda done, total: self.bus.emit(
            "setup.progress", {"model": model, "label": label, "done": done, "total": total}
        )

    def _fetch(self, model: speech_models.SpeechModel | None):
        """Download a speech model unless it is there, with progress under
        the orb. Waits while the walkthrough's downloads run, so no file is
        fetched twice at once."""
        if model is None:
            return
        with self._fetching:
            if not model.is_cached():
                model.fetch(self._progress(model.id, model.progress_label))

    def _fetch_stt(self, whisper_model: str, wake_word: str | None = None, **engine):
        """Download the speech recognition model unless it is cached or a
        local folder was chosen (used as it is)."""
        self._fetch(
            speech_models.whisper(
                whisper_model,
                engine.get("whisper_engine", "auto"),
                engine.get("whisper_device", "auto"),
            )
        )
        if wake_word:
            self._fetch(speech_models.wake(engine.get("whisper_engine", "auto")))

    def _fetch_turn(self):
        """Download Smart Turn, which hears when the user has finished
        speaking, unless it is cached. Without it a turn ends after a silence."""
        try:
            self._fetch(speech_models.turn())
        except Exception:
            pass

    def _fetch_tts(self, tts_dir: str | None):
        """Download the voice model unless a folder was chosen or the files
        are already in one of the usual places."""
        self._fetch(speech_models.kokoro(tts_dir))

    def _fetch_models(self, kwargs: dict):
        """Download the speech models that are neither found nor pointed to,
        with visible progress, the voices first.

        On a fresh install the first launch fetches large models (1.6 GB for
        large-v3-turbo, a few hundred MB for each voice engine); without this
        the UI would only say "Loading models". A model folder chosen in
        Settings is used as it is and never downloaded. Usually the
        walkthrough has fetched them already; while it still is, this waits
        for it, then fetches whatever failed there.
        """
        # Each is tried whatever became of the others (offline, all fail):
        # Iris starts with what it has and says what it could not fetch.
        # Without Smart Turn a silence ends a turn, which needs no word.
        try:
            for model in speech_models.needed(
                whisper_model=kwargs.get("whisper_model", "auto"),
                whisper_engine=kwargs.get("whisper_engine", "auto"),
                whisper_device=kwargs.get("whisper_device", "auto"),
                recognition_engine=kwargs.get("recognition_engine", "whisper"),
                # A sleep word wakes Iris by the wake word, on or off
                wake_word=kwargs.get("wake_word") or kwargs.get("sleep_word"),
                tts_dir=kwargs.get("tts_models_dir"),
            ):
                try:
                    self._fetch(model)
                except Exception as e:
                    if not model.optional:
                        message = f"Could not download the {model.what}: {brief(e)}"
                        self.bus.emit("error", {"message": message})
        finally:
            self.bus.emit("setup.progress", {"label": "", "done": 0, "total": 0})

    def setup_info(self) -> dict:
        """For the first-launch walkthrough: what would be downloaded, roughly
        how much, and what is already on this machine."""
        engine = resolve_engine("auto")
        model = resolve_model("auto", engine, resolve_device(engine))
        return {
            "stt": {
                "engine": engine,
                "model": model,
                "cached": downloads.is_cached(model_repo(model, engine)),
                "size_mb": WHISPER_MB.get(model),
            },
            "tts": {"found": str(find_kokoro_models() or ""), "size_mb": KOKORO_MB},
            "ollama_url": effective_host(self._ollama_host),
            "models": self.fetcher.describe(speech_models.needed()),
        }

    # ------------------------------------------------------------------
    # The first-run walkthrough: what it needs before Iris runs
    # ------------------------------------------------------------------

    def setup_download(self, options: dict | None = None) -> dict:
        """Fetch the speech models Iris will need, in the background, while
        the walkthrough goes on; asked again, retry what failed. Progress
        comes as setup.models events (each model's state) and setup.progress
        (the one downloading, for the orb).

        options: the choices made so far that change what is needed:
        whisper_model, wake_word, and model folders (stt_path, tts_dir) to
        use instead of downloads.
        """
        options = dict(options or {})
        self._walkthrough = options

        def choose():
            return speech_models.needed(
                whisper_model=options.get("stt_path") or options.get("whisper_model") or "auto",
                wake_word=options.get("wake_word") or None,
                tts_dir=options.get("tts_dir") or None,
            )

        return {"ok": True, "models": self.fetcher.start(choose)}

    def _models_changed(self, models: list[dict], busy: bool):
        self.bus.emit("setup.models", {"models": models, "busy": busy})

    def _model_progress(self, model, done: int, total: int):
        if model is None:
            self.bus.emit("setup.progress", {"label": "", "done": 0, "total": 0})
        else:
            self._progress(model.id, model.progress_label)(done, total)

    @staticmethod
    def probe_server(url: str) -> dict:
        """A server's address as Iris would keep it, and its models, without
        switching to it: the walkthrough lets the user choose first."""
        host = resolve_host(url)
        try:
            ollama_client(host, timeout=8).list()
        except Exception as e:
            where = host or "the default server"
            return {"ok": False, "error": f"Cannot reach the server at {where}: {e}"}
        return {
            "ok": True,
            "host": host or "",
            "models": ConversationManager.list_available_models(host),
        }

    @staticmethod
    def find_servers() -> dict:
        """Model servers with the OpenAI-compatible API answering on this
        computer at their usual ports (besides Ollama, the default one)."""
        return {"servers": servers.discover()}

    def sound_check(self, on: bool, device: int | None = None) -> dict:
        """Listen to the microphone on its own, before Iris does, so the
        walkthrough can show its level and whether a voice comes through
        (mic.check events). Off closes it again."""
        if self._sound_check is not None:
            self._sound_check.stop()
            self._sound_check = None
        if not on:
            return {"ok": True}
        check = SoundCheck(lambda data: self.bus.emit("mic.check", data), device=device)
        try:
            check.start()
        except Exception as e:
            return {"ok": False, "error": f"The microphone could not be opened: {e}"}
        self._sound_check = check
        return {"ok": True, "device": check.device_name()}

    def greet(self) -> dict:
        """Have Iris say hello as its next turn, in the persona's own words:
        the end of the walkthrough. Kept until Iris runs, if it does not yet."""
        self._turns.put({"greet": True})
        return {"ok": True}

    def _release_first_run(self):
        """Iris starts: what the walkthrough used of the microphone and the
        speaker is given back."""
        self.sound_check(False)
        with self._preview_lock:
            preview, self._preview = self._preview, None
        if preview is not None:
            preview.stop()
            preview.text_queue.put(None)  # its workers end

    @staticmethod
    def check_model_paths(stt_path: str | None = None, tts_dir: str | None = None) -> dict:
        """Whether chosen model folders hold what they should."""
        if stt_path:
            folder = Path(stt_path).expanduser()
            if not folder.is_dir() or folder_engine(folder) is None:
                return {
                    "ok": False,
                    "error": f"{folder} holds no Whisper model "
                    "(an MLX conversion or a faster-whisper model folder)",
                }
        if tts_dir and not kokoro_files_in(Path(tts_dir).expanduser()):
            return {
                "ok": False,
                "error": f"{tts_dir} does not hold {KOKORO_MODEL_FILE} and {KOKORO_VOICES_FILE}",
            }
        return {"ok": True}

    def set_model_paths(self, stt_path: str | None = None, tts_dir: str | None = None) -> dict:
        """Use model folders the user already has ("" = download as usual).

        A folder without the expected files is refused right away; loading
        the new model takes seconds, so it happens in the background and the
        running assistant switches over when it is ready.
        """
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        checked = self.check_model_paths(stt_path, tts_dir)
        if not checked["ok"]:
            return checked

        def load():
            assistant = self.assistant
            try:
                if stt_path is not None:
                    model = stt_path or "auto"
                    self._fetch_stt(model)
                    assistant.transcriber = WhisperTranscriber(model)
                if tts_dir is not None:
                    self._fetch_tts(tts_dir)
                    generator = SpeechGenerator(
                        assistant.tts_generator.engine,
                        assistant.tts_generator.voice,
                        assistant.tts_generator.speed,
                        models_dir=tts_dir or None,
                    )
                    generator.on_failure = assistant.tts_generator.on_failure
                    assistant.tts_generator = generator
                    assistant.speech_pipeline.generator = generator
                self.bus.emit("config", assistant.config_info())
            except Exception as e:
                self.bus.emit("error", {"message": f"Could not load the model: {e}"})
            finally:
                self.bus.emit("setup.progress", {"label": "", "done": 0, "total": 0})

        threading.Thread(target=load, daemon=True).start()
        return {"ok": True}

    def stt_models(self) -> dict:
        """Whisper's sizes for this computer's engine: roughly how big each
        is, whether it is downloaded, and which one is in use."""
        transcriber = self.assistant.transcriber if self.assistant else None
        engine = getattr(transcriber, "engine", None) or resolve_engine("auto")
        folder = str(getattr(transcriber, "model_path", None) or "")
        current = "" if folder else getattr(transcriber, "model_name", "")
        return {
            "engine": engine,
            "folder": folder,
            "models": [
                {
                    "id": size,
                    "size_mb": WHISPER_MB.get(size),
                    "cached": downloads.is_cached(model_repo(size, engine)),
                    "active": size == current,
                }
                for size in WHISPER_SIZES
            ],
        }

    def set_stt_model(self, model: str) -> dict:
        """Recognize speech with another Whisper size. It is downloaded if
        needed and loaded in the background; the running assistant switches
        over when it is ready."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        if model not in WHISPER_SIZES:
            return {"ok": False, "error": f"There is no Whisper model called {model}"}
        if self.assistant.recognition_engine != "whisper":
            return {"ok": False, "error": "The recognition engine in use is not Whisper"}

        def load():
            assistant = self.assistant
            try:
                self._fetch_stt(model)
                assistant.transcriber = engines.recognition_engine("whisper").create(
                    {"model": model, "device": "auto", "engine": "auto"}
                )
                self.bus.emit("config", assistant.config_info())
            except Exception as e:
                self.bus.emit("error", {"message": f"Could not load the model: {e}"})
            finally:
                self.bus.emit("setup.progress", {"label": "", "done": 0, "total": 0})

        threading.Thread(target=load, daemon=True).start()
        return {"ok": True}

    def set_turn_detection(self, mode: str) -> dict:
        """End turns when Smart Turn hears the user has finished
        ("smart_turn"), or after a silence ("silence")."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        if mode not in ("smart_turn", "silence"):
            return {"ok": False, "error": f"Unknown turn detection: {mode}"}

        def load():
            assistant = self.assistant
            try:
                if mode == "smart_turn":
                    self._fetch_turn()
                if not assistant.set_turn_detection(mode):
                    self.bus.emit(
                        "error",
                        {"message": "Smart Turn could not be loaded, so a turn ends after a silence."},
                    )
                self.bus.emit("config", assistant.config_info())
            finally:
                self.bus.emit("setup.progress", {"label": "", "done": 0, "total": 0})

        threading.Thread(target=load, daemon=True).start()
        return {"ok": True}

    def _warn_if_ollama_unreachable(self) -> bool:
        """Say plainly what is missing when the server does not answer, e.g. on
        a computer where Ollama is not installed yet. Returns whether one answers."""
        if getattr(self.assistant.conversation, "reachable", True):
            return True
        url = effective_host(self._ollama_host)
        if is_openai(self._ollama_host):
            message = f"Can't reach the server at {url}. Start it, or set another server in Settings."
        else:
            message = (
                f"Can't reach Ollama at {url}. Install it from ollama.com and start "
                f"it, then download a model, e.g.: ollama pull qwen3:1.7b. Or set "
                f"another server in Settings."
            )
        self.bus.emit(
            "error",
            {
                "message": message,
                # Retracted by the UI once a reachable server is set
                "kind": "ollama_unreachable",
            },
        )
        return False

    def _warn_if_model_missing(self):
        """Report a model the server does not have up front rather than on
        the first reply."""
        available = [
            m["name"] for m in ConversationManager.list_available_models(self._ollama_host)
        ]
        model = self.assistant.conversation.model
        if available and model not in available:
            if is_openai(self._ollama_host):
                message = f"The server has no model '{model}'. Pick another one in Settings."
            else:
                message = (
                    f"Ollama model '{model}' is not installed. Pick another one "
                    f"in Settings, or install it with: ollama pull {model}"
                )
            self.bus.emit(
                "error",
                {
                    "message": message,
                    # Lets the UI retract the warning once another model is chosen
                    "kind": "model_missing",
                },
            )

    def list_models(self) -> dict:
        """List the server's models usable for conversation."""
        return {"models": ConversationManager.list_available_models(self._ollama_host)}

    @staticmethod
    def check_servers(urls: list) -> dict:
        """Which of these model servers answer ("" is the default one)."""
        return {"servers": servers.check([str(url) for url in urls or []])}

    def set_ollama_host(self, url: str) -> dict:
        """Point the assistant at another server ("" = default): Ollama, or
        any server with the OpenAI-compatible API.

        The server is asked what it is (resolve_host) and for its models
        first, so an unreachable URL is reported instead of breaking the next
        reply.
        """
        host = resolve_host(url)
        try:
            ollama_client(host, timeout=8).list()
        except Exception as e:
            where = host or "the default server"
            return {"ok": False, "error": f"Cannot reach the server at {where}: {e}"}
        self._ollama_host = host
        # A server answers now: the "can't reach Ollama" warning no longer holds
        with self._transcript_lock:
            self._transcript = [
                e for e in self._transcript if e["data"].get("kind") != "ollama_unreachable"
            ]
        if self.assistant is not None:
            self.assistant.conversation.set_host(host)
            # The model that sees is one of this server's, or there is none:
            # the last server's would not be found here
            self.assistant.conversation.vision_model = self._pick_vision_model()
            self.bus.emit("config", self.assistant.config_info())
            try:
                self._warn_if_model_missing()
            except Exception:
                pass
        return {"ok": True, "host": host or ""}

    def set_model(self, name: str) -> dict:
        """Switch the LLM model. Conversation history is preserved."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        if not name:
            return {"ok": False, "error": "No model given"}
        try:
            self.assistant.conversation.set_model(name)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        # The UI retracts its missing-model warning; keep history consistent
        with self._transcript_lock:
            self._transcript = [
                e for e in self._transcript if e["data"].get("kind") != "model_missing"
            ]
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True, "model": name}

    def list_input_devices(self) -> dict:
        """List available microphones."""
        return {"devices": AudioRecorder.list_input_devices()}

    def set_input_device(self, index: int | None) -> dict:
        """Select the microphone. Takes effect on the next listen."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.recorder.set_device(index)
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True, "input_device": self.assistant.recorder.current_device_name()}

    def list_personas(self) -> dict:
        """Built-in and user personas, re-scanned so new files appear."""
        names = reload_personas()
        return {"personas": list(names), "names": names}

    def set_persona(self, name: str, note: bool = True) -> dict:
        """Hot-swap the persona (prompt, voice, speed, audio filter): the
        open thread's, edited or gone, or another thread's as it is opened
        (with no note: what the thread holds was said by its own persona)."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        if name not in list_personas():
            return {"ok": False, "error": f"Unknown persona: {name}"}

        persona = get_persona(name)
        assistant = self.assistant
        previous = assistant.persona.name
        assistant.persona = persona
        assistant.conversation.persona = persona
        # The history still holds the old persona's replies, which small
        # models happily continue; tell the model the voice has changed hands
        if note and previous != persona.name:
            assistant.conversation.add_note(
                f"You are now {persona.name}. Earlier replies in this conversation "
                f"were given by {previous}; do not continue {previous}'s name or manner."
            )
        assistant.set_persona_words(persona.wake_word, persona.sleep_word)
        generator = assistant.tts_generator
        generator.engine = persona.engine
        generator.voice = persona.voice
        generator.speed = persona.speed
        # The conversation keeps its language, also one the new voice does
        # not speak: its replies in it are shown
        if assistant.conversation_language not in generator.languages():
            assistant.conversation_language = "en"
        assistant.player.audio_filter = (
            get_filter(persona.audio_filter) if persona.audio_filter else None
        )

        # Load the persona's speech engine now rather than at its first reply
        def prepare():
            try:
                generator.prepare()
            except Exception as e:
                self.bus.emit("error", {"message": f"Could not load the voice: {e}"})

        threading.Thread(target=prepare, daemon=True).start()
        self._ready_personas(readiness.VOICES, {persona.id})
        self._persona_stamp = persona_stamp(persona.id)
        self.bus.emit("config", assistant.config_info())
        return {"ok": True, "persona": name}

    def persona_voices(self) -> dict:
        """For the persona editor: the speech engines (names, whether they
        take a speed) and their voices, each with the languages it speaks
        properly; the languages Iris talks in; and the audio filters."""
        models_dir = self.assistant.tts_generator.models_dir if self.assistant else None
        registered = engines.speech_engines()
        return {
            "engines": {engine: list_voices(engine, models_dir) for engine in registered},
            "engine_info": [
                {"id": engine.id, "name": engine.name, "takes_speed": engine.takes_speed}
                for engine in registered.values()
            ],
            "languages": [
                {"code": code, "name": LANGUAGE_NAMES.get(code, code)}
                for code in in_order(engines.speech_languages())
            ],
            "filters": list(AUDIO_FILTERS),
        }

    def persona_get(self, persona_id: str) -> dict:
        """A persona's settings, for editing."""
        reload_personas()
        persona = find_persona(persona_id)
        if persona is None:
            return {"ok": False, "error": f"Unknown persona: {persona_id}"}
        return {
            "ok": True,
            "id": persona.id,
            **persona.to_dict(),
            # Shipped with Iris: it can be reset once edited, but not deleted
            "shipped": shipped(persona.id),
            "edited": shipped(persona.id) and not persona.builtin,
        }

    def persona_save(self, settings: dict) -> dict:
        """Save a persona edited in the UI (a new one has no id). Saving the
        current persona applies the changes right away."""
        if not (settings.get("name") or "").strip():
            return {"ok": False, "error": "The persona needs a name"}
        if settings.get("engine") not in engines.speech_engines() or not settings.get("voice"):
            return {"ok": False, "error": "The persona needs a voice"}
        before = find_persona(settings.get("id") or "")
        # A cloned voice that is not here is not given to a persona. One
        # that speaks with it already (its voice removed outside Iris, or by
        # an Iris that dropped voices) still saves its other changes, its
        # wake word say; the conversation tells to choose another voice
        voice = settings["voice"]
        if cloning.voice_missing(voice) and (before is None or before.voice != voice):
            name = cloning.voice_name(cloning.voice_folder(voice))
            error = f"The voice {name} cannot be found: choose another voice"
            return {"ok": False, "error": error}
        persona_id = save_persona(settings, settings.get("id") or None)
        if self.assistant is not None and self.assistant.persona.id == persona_id:
            self.set_persona(persona_id)
        else:
            # Its voice is made ready for Iris's languages
            self._ready_personas(readiness.VOICES, {persona_id})
        return {"ok": True, "id": persona_id, **self.list_personas()}

    def set_persona_words(
        self,
        persona_id: str | None = None,
        wake_word: str | None = None,
        sleep_word: str | None = None,
    ) -> dict:
        """Give a persona (the current one, without an id) its wake word or
        sleep word, as set beside the wake word switch. None leaves a word
        as it is; an empty wake word is the persona's name again, an empty
        sleep word none. Nothing is written when nothing changes, so a
        persona that ships with Iris is copied only when it is changed."""
        if not persona_id:
            persona_id = self.assistant.persona.id if self.assistant else "iris"
        reload_personas()
        persona = find_persona(persona_id)
        if persona is None:
            return {"ok": False, "error": f"Unknown persona: {persona_id}"}
        settings = persona.to_dict()
        if wake_word is not None:
            settings["wake_word"] = wake_word.strip() or persona.name
        if sleep_word is not None:
            settings["sleep_word"] = sleep_word.strip()
        if settings == persona.to_dict():
            return {"ok": True, "id": persona.id, **self.list_personas()}
        return self.persona_save({**settings, "id": persona.id})

    def adopt_wake_words(
        self, persona_id: str | None, wake_word: str | None, sleep_word: str | None
    ) -> dict:
        """Iris kept one wake word and one sleep word for every persona
        before they became each persona's own. The ones a user set become
        those of the persona in use, so they go on working; whether the
        wake word was on (it was when there was one) stays the switch."""
        wake_word, sleep_word = (wake_word or "").strip(), (sleep_word or "").strip()
        words = {}
        if wake_word:
            words["wake_word"] = wake_word
        if sleep_word:
            words["sleep_word"] = sleep_word
        # A persona gone since is Iris, as Iris starts with it
        persona = get_persona(persona_id or "iris")
        result = self.set_persona_words(persona.id, **words) if words else {"ok": True}
        return {**result, "wake_on": bool(wake_word)}

    def persona_delete(self, persona_id: str) -> dict:
        """Delete the user's persona, or reset an edited one that ships with
        Iris. When it was the current persona, Iris takes over."""
        still_there = delete_persona(persona_id)
        current = self.assistant.persona.id if self.assistant else None
        if current == persona_id:
            current = persona_id if still_there else "iris"
            self.set_persona(current)
        return {"ok": True, "persona": current, **self.list_personas()}

    # ------------------------------------------------------------------
    # Iris's languages, and the voices made ready for them
    # ------------------------------------------------------------------

    def set_languages(self, codes) -> dict:
        """Iris's languages, as chosen in the app (English always): every
        persona's voice is made ready for each of them, ahead, with progress
        (voice.job events). They never hold a conversation to them: Iris
        follows the user into any of its languages, and a voice not ready
        for one is made ready when a reply comes to it."""
        wanted = {code for code in (codes or []) if code in LANGUAGE_NAMES} | {"en"}
        self.languages = in_order(wanted)
        return {"ok": True, "languages": self.languages, "jobs": self._ready_personas()}

    def _ready_personas(self, origin: str = readiness.LANGUAGES, only: set | None = None) -> list:
        """Make the personas' voices (those of only, by id, if given) ready
        for Iris's languages. Returns the jobs that does."""
        jobs = []
        for (engine, voice), speakers in voice_users().items():
            if only is not None and not any(persona.id in only for persona in speakers):
                continue
            names = [persona.name for persona in speakers]
            for language in self.languages:
                job = self.readier.request(
                    engine, voice, language, self._voice_name(engine, voice), names, origin
                )
                if job is not None:
                    jobs.append(job)
        return jobs

    def _voice_not_ready(self, engine: str, voice: str, language: str):
        """A reply came to a language the persona's voice is not ready for:
        it is made ready now, with its progress in the conversation. One that
        failed stays failed until the user retries it."""
        found = self.readier.find(engine, voice, language)
        if found is not None and found.state == readiness.FAILED:
            return
        persona = self.assistant.persona.name if self.assistant else ""
        self.readier.request(
            engine,
            voice,
            language,
            self._voice_name(engine, voice),
            [persona] if persona else [],
            readiness.CONVERSATION,
        )

    def _tts_dir(self) -> str | None:
        """The voice model folder chosen in the app, if any."""
        if self.assistant is not None:
            folder = self.assistant.tts_generator.models_dir
            return str(folder) if folder else None
        return self._walkthrough.get("tts_dir") or None

    @staticmethod
    def _voice_name(engine: str, voice: str) -> str:
        """A voice as the app names it."""
        folder = cloning.voice_folder(voice)
        if folder is not None:
            return cloning.voice_name(folder)
        name = voice.split("_", 1)[-1] if engine == "kokoro" else voice
        return name.replace("_", " ").title()

    def _voice_changed(self, voice: str):
        """A cloned voice made again from a new sample: what was loaded of
        it goes, in Iris's voice and in the walkthrough's."""
        generators = []
        if self.assistant is not None:
            generators.append(self.assistant.tts_generator)
        with self._preview_lock:
            if self._preview is not None:
                generators.append(self._preview.generator)
        for generator in generators:
            generator.voice_changed(voice)

    def voice_list(self) -> dict:
        """For the Voices page and the persona editor: every engine's voices,
        as persona_voices() lists them, each with the personas that speak
        with it and the languages it is ready for; a cloned voice with its
        folder and the languages it is made for so far. With Iris's
        languages, the jobs making voices ready, and the voice a persona
        speaks with once its cloned voice is deleted."""
        listing = self.persona_voices()
        users = voice_users()
        models_dir = self._tts_dir()
        for engine, voices in listing["engines"].items():
            for voice in voices:
                speakers = users.get((engine, voice["id"]), [])
                voice["personas"] = [{"id": p.id, "name": p.name} for p in speakers]
                voice["ready"] = [
                    code
                    for code in voice["languages"]
                    if voice_ready(engine, voice["id"], code, models_dir)
                ]
                folder = cloning.voice_folder(voice["id"]) if engine == PocketGenerator.id else None
                if folder is not None:
                    made = set(cloning.made_for(folder))
                    voice["cloned"] = {
                        "folder": _shown_path(folder),
                        "made": [code for code, lang in POCKET_LANGUAGES.items() if lang in made],
                    }
        listing["chosen"] = list(self.languages)
        listing["jobs"] = self.readier.jobs()
        listing["fallback"] = cloning.FALLBACK_VOICE
        return listing

    def voice_make_ready(self, engine: str, voice: str, languages=None) -> dict:
        """Make a voice ready for languages (Iris's, without any), from the
        Voices page; one that failed is tried again."""
        codes = list(languages) if languages else list(self.languages)
        names = [persona.name for persona in speaking_with(voice)]
        jobs = []
        for code in codes:
            found = self.readier.find(engine, voice, code)
            if found is not None and found.state == readiness.FAILED:
                job = self.readier.retry(found.id)
            else:
                job = self.readier.request(
                    engine, voice, code, self._voice_name(engine, voice), names, readiness.VOICES
                )
            if job is not None:
                jobs.append(job)
        return {"ok": True, "jobs": jobs}

    def voice_retry(self, job_id: str) -> dict:
        job = self.readier.retry(job_id)
        return {"ok": job is not None, "job": job}

    def voice_clone(self, params: dict) -> dict:
        """Make a cloned voice of a sample prepared (see Cloning.clone), once
        the user confirms it, then for the languages chosen with it (Iris's
        own when none are named), English always: those jobs come back, for
        the clone to finish once they have. A language whose cloning model
        is not here downloads it first, shown as its job."""
        result = self.cloning.clone(
            params.get("name", ""), params.get("replace", ""), params.get("sample", "")
        )
        if not result.get("ok"):
            return result
        voice = result["voice"]["id"]
        named = (params.get("name") or "").strip()
        names = [persona.name for persona in speaking_with(voice)] or ([named] if named else [])
        chosen = params.get("languages")
        codes = in_order({"en", *(self.languages if chosen is None else chosen)})
        jobs = []
        for code in codes:
            job = self.readier.request(
                "kyutai", voice, code, result["voice"]["name"], names, readiness.CLONE
            )
            if job is not None:
                jobs.append(job)
        return {**result, "jobs": jobs}

    def voice_sample_play(self, sample: str = "", on: bool = True) -> dict:
        """Play a sample prepared for cloning, as the voice would be made of
        it, cutting off whatever is being said; or stop it (on false). It
        goes through the speech pipeline, as a preview does, so the
        microphone does not take it for the user; before Iris runs, it
        plays on the speaker alone."""
        if not on:
            if self.assistant is not None:
                self.assistant.speech_pipeline.stop()
            if self._sample_player is not None:
                self._sample_player.stop()
            return {"ok": True}
        path = self.cloning.sample_file(sample)
        if path is None:
            return {"ok": False, "error": "Record a sample or choose one first"}
        if self.assistant is not None:
            pipeline = self.assistant.speech_pipeline
            pipeline.stop()
            pipeline.queue_audio(path)
            return {"ok": True}
        if self._sample_player is not None:
            self._sample_player.stop()
        player = self._sample_player = AudioPlayer()

        def play():
            self._publish_state("speaking")
            try:
                player.play(
                    str(path),
                    on_level=lambda level: self.bus.emit("audio.output", {"rms": level}),
                    audio_filter=None,
                )
            except Exception as e:
                self.bus.emit("error", {"message": f"The sample could not be played: {brief(e)}"})
            finally:
                path.unlink(missing_ok=True)
                # Unless Iris has started meanwhile, which says what it is doing
                if self.assistant is None:
                    self._publish_state("idle")

        threading.Thread(target=play, daemon=True).start()
        return {"ok": True}

    def voice_delete(self, voice: str, reassign: bool = False) -> dict:
        """Delete a cloned voice, from the Voices page. One that personas
        speak with is deleted only when they are to speak with another
        (reassign): Kyutai's own default voice, which the app names when it
        asks. The persona in use takes it at once."""
        folder = cloning.voice_folder(voice)
        if folder is None or not folder.is_dir():
            return {"ok": False, "error": "There is no such cloned voice"}
        speakers = speaking_with(voice)
        if speakers and not reassign:
            names = [persona.name for persona in speakers]
            verb = "speaks" if len(names) == 1 else "speak"
            return {
                "ok": False,
                "error": f"{readiness._together(names)} {verb} with this voice",
                "personas": [{"id": p.id, "name": p.name} for p in speakers],
            }
        for persona in speakers:
            self.persona_save(
                {
                    **persona.to_dict(),
                    "id": persona.id,
                    "engine": PocketGenerator.id,
                    "voice": cloning.FALLBACK_VOICE,
                }
            )
        cloning.forget_voice(voice)
        return {"ok": True, **self.list_personas()}

    def persona_preview(self, settings: dict) -> dict:
        """Say a line in a persona's voice, as edited, cutting off whatever is
        being said. It goes through the speech pipeline, so the mic stays
        closed while it plays. Before Iris runs (in the walkthrough) a voice
        of its own says it, once that engine's model is downloaded."""
        engine, voice = settings.get("engine"), settings.get("voice")
        if engine not in engines.speech_engines() or not voice:
            return {"ok": False, "error": "No voice chosen"}
        if cloning.voice_missing(voice):
            name = cloning.voice_name(cloning.voice_folder(voice))
            return {"ok": False, "error": f"The voice {name} cannot be found"}
        # In the voice's own language, or the first it speaks
        language = voice_language(engine, voice)
        spoken = voice_languages(engine, voice)
        if language not in spoken:
            language = "en" if "en" in spoken or not spoken else spoken[0]
        name = (settings.get("name") or "").strip() or "Iris"
        line = PREVIEW_LINES.get(language, PREVIEW_LINES["en"]).format(name=name)
        sound = {
            "engine": engine,
            "voice": voice,
            "speed": float(settings.get("speed") or 1.0),
            "audio_filter": settings.get("audio_filter") or "",
        }
        self._preview_wanted = dict(settings)
        if self.assistant is None and not self._voice_downloaded(engine):
            return {"ok": False, "error": "The voice is still downloading"}
        # A voice not here yet is downloaded first, where the user sees it,
        # and plays once it is (unless another was asked for meanwhile)
        if not self.readier.ready(engine, voice, language):
            users = [persona.name for persona in speaking_with(voice)]
            job = self.readier.request(
                engine, voice, language, self._voice_name(engine, voice), users, readiness.PREVIEW
            )
            if job is not None:
                threading.Thread(
                    target=self._preview_once_ready, args=(job["id"], dict(settings)), daemon=True
                ).start()
                return {"ok": True, "job": job}
        if self.assistant is None:
            return self._preview_before_start(line, language, sound)
        pipeline = self.assistant.speech_pipeline
        pipeline.stop()
        pipeline.queue_speech(line, language=language, sound=sound)
        return {"ok": True}

    def _preview_once_ready(self, job_id: str, settings: dict):
        if self.readier.wait(job_id, PREVIEW_WAIT) and self._preview_wanted == settings:
            self.persona_preview(settings)

    def _voice_downloaded(self, engine: str) -> bool:
        """Whether a speech engine's model is here, to preview its voices
        before Iris runs (an engine a plugin adds fetches its own)."""
        if engine == "kyutai":
            return speech_models.kyutai().is_cached()
        if engine == "kokoro":
            return bool(self._walkthrough.get("tts_dir")) or speech_models.kokoro().is_cached()
        return True

    def _preview_before_start(self, line: str, language: str, sound: dict) -> dict:
        """Say a preview in a voice of the walkthrough's own, while Iris does
        not run yet. Its engine loads on a thread, so no request waits for
        it; the state is speaking while it plays, as it is with Iris."""
        if not self._voice_downloaded(sound["engine"]):
            return {"ok": False, "error": "The voice is still downloading"}
        tts_dir = self._walkthrough.get("tts_dir") or None

        def speak():
            try:
                with self._preview_lock:
                    if self._preview is None:
                        generator = SpeechGenerator(
                            sound["engine"], sound["voice"], sound["speed"], models_dir=tts_dir
                        )
                        self._preview = SpeechPipeline(generator, AudioPlayer())
                        self._preview.on_output_level = lambda level: self.bus.emit(
                            "audio.output", {"rms": level}
                        )
                    pipeline = self._preview
                pipeline.stop()
                pipeline.queue_speech(line, language=language, sound=sound)
            except Exception as e:
                self.bus.emit("error", {"message": f"Could not load the voice: {e}"})
                return
            self._publish_state("speaking")
            deadline = time.monotonic() + PREVIEW_LIMIT
            while pipeline.is_active and time.monotonic() < deadline:
                time.sleep(0.05)
            # Unless Iris has started meanwhile, which says what it is doing
            if self._preview is pipeline:
                self._publish_state("idle")

        threading.Thread(target=speak, daemon=True).start()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Roles: a focus a thread can be given
    # ------------------------------------------------------------------

    @staticmethod
    def roles_list() -> dict:
        """Every role, by name, re-read so files added by hand show up."""
        return {"roles": [role.to_dict() for role in reload_roles()]}

    def role_save(self, settings: dict) -> dict:
        """Save a role edited in the app (a new one has no id). Its threads
        have the new focus from their next turn."""
        if not (settings.get("name") or "").strip():
            return {"ok": False, "error": "The role needs a name"}
        role_id = save_role(settings, settings.get("id") or None)
        if self.assistant is not None:
            self.bus.emit("config", self.assistant.config_info())
        self.bus.emit("threads.changed", {})
        return {"ok": True, "id": role_id, **self.roles_list()}

    def role_delete(self, role_id: str) -> dict:
        """Delete the user's role, or reset an edited one that ships with
        Iris. The threads of a role that is gone go on without one."""
        if not delete_role(role_id):
            memory = getattr(self.assistant, "memory", None)
            if memory is not None:
                memory.store.clear_role(role_id)
                if self.assistant.conversation.role == role_id:
                    self.assistant.conversation.role = None
        if self.assistant is not None:
            self.bus.emit("config", self.assistant.config_info())
        self.bus.emit("threads.changed", {})
        return {"ok": True, **self.roles_list()}

    def history(self) -> dict:
        """The conversation so far, as the events that built it."""
        with self._transcript_lock:
            return {"events": list(self._transcript)}

    def info(self) -> dict:
        try:
            version = importlib.metadata.version("iris")
        except importlib.metadata.PackageNotFoundError:
            version = "dev"
        with self._state_lock:
            state = self._last_state
        names = reload_personas()
        result = {
            "version": version,
            "personas": list(names),
            "persona_names": names,
            "state": state,
            "running": bool(self._loop_thread and self._loop_thread.is_alive()),
            "timings": list(self._timings),
        }
        if self.assistant is not None:
            result["config"] = self.assistant.config_info()
        return result

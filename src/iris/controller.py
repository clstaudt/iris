"""GUI-facing controller: runs the VoiceAssistant off the main thread and
exposes commands (start/stop/interrupt/send_text) and a single effective
state stream for external UIs such as the Electron shell."""

import importlib.metadata
import itertools
import queue
import threading
import time
from collections import deque
from pathlib import Path

from iris import downloads, engines, servers
from iris.assistant import VoiceAssistant
from iris.attachments import AttachmentError, prepare_all
from iris.audio import AUDIO_FILTERS, AudioRecorder, get_filter
from iris.events import EventBus
from iris.llm import ConversationManager, effective_host, normalize_host, ollama_client
from iris.personas import (
    delete_persona,
    find_persona,
    get_persona,
    list_personas,
    reload_personas,
    save_persona,
    shipped,
)
from iris.llm_client import is_openai, resolve_host
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
from iris.tts import (
    DEFAULT_KOKORO_DIR,
    KOKORO_MODEL_FILE,
    KOKORO_MODEL_URL,
    KOKORO_VOICES_FILE,
    KOKORO_VOICES_URL,
    SpeechGenerator,
    engine_speaks,
    find_kokoro_models,
    kokoro_files_in,
    list_voices,
    voice_language,
)
from iris.turns import SMART_TURN_FILE, SMART_TURN_REPO

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

# Rough download sizes for the first-launch setup, in MB
APPROX_MODEL_MB = {
    "tiny": 75,
    "base": 145,
    "small": 480,
    "medium": 1500,
    "large-v3": 3100,
    "large-v3-turbo": 1600,
}
KOKORO_MB = 335

# Events that make up the visible conversation. Kept here so a UI that
# reloads (or attaches late) can rebuild its transcript; streaming chunks are
# left out because chat.done carries the full reply.
TRANSCRIPT_EVENTS = ("chat.user", "chat.done", "chat.continued", "tool.use", "error")
TRANSCRIPT_LIMIT = 500
# Turn timings kept for the app
TIMINGS_KEPT = 20


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
        self._start_lock = threading.Lock()

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

        # Silent-mic watchdog: a real microphone always has a noise floor, so
        # a stream of exact zeros means the OS is denying mic access (on macOS,
        # TCC denies silently when the host app lacks the microphone grant).
        self._zero_level_since: float | None = None
        self._silence_warned = False

        self._transcript: list[dict] = []
        self._transcript_lock = threading.Lock()
        # The latest turns' timings, for an app that attaches later
        self._timings: deque = deque(maxlen=TIMINGS_KEPT)

        bus.subscribe(self._on_bus_event)

    # ------------------------------------------------------------------
    # Effective state computation (flags + speech pipeline activity)
    # ------------------------------------------------------------------

    def _on_bus_event(self, event_type: str, data: dict):
        if event_type in TRANSCRIPT_EVENTS:
            with self._transcript_lock:
                self._transcript.append({"type": event_type, "data": dict(data)})
                del self._transcript[:-TRANSCRIPT_LIMIT]
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
            self.bus.emit(
                "error",
                {
                    "message": (
                        "Microphone returns pure silence — macOS is denying audio "
                        "capture. In dev mode the permission belongs to the app that "
                        "launched the dev server (Cursor, VS Code, Terminal or iTerm), "
                        "not to Electron. Enable it under System Settings > Privacy & "
                        "Security > Microphone, then relaunch that app completely."
                    )
                },
            )

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
                    if self.assistant.pause_event.is_set():
                        state = "muted"
                    elif self.assistant.hold_event.is_set():
                        state = "typing"
                self._publish_state(state)
            self._poller_stop.wait(0.1)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def start(self, options: dict | None = None) -> dict:
        """Build the assistant and start its run_loop on a background thread."""
        with self._start_lock:
            if self._loop_thread and self._loop_thread.is_alive():
                return {"ok": True, "already_running": True}
            self._poller_stop.clear()
            self._ollama_host = normalize_host((options or {}).get("ollama_host"))
            self._publish_state("starting")
            self._loop_thread = threading.Thread(
                target=self._run, args=(options or {},), daemon=True
            )
            self._loop_thread.start()
            threading.Thread(target=self._speaking_poller, daemon=True).start()
        return {"ok": True}

    def _pick_vision_model(self, preferred: str) -> str | None:
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
        persona_name = kwargs.pop("persona", "iris")
        # {plugin id: enabled} chosen in the UI; the others follow their default
        kwargs["tools"] = kwargs.pop("tools", {}) or {}
        amnesia = bool(kwargs.pop("amnesia", False))
        # Model folders chosen in Settings; empty means download as usual
        if stt_path := kwargs.pop("stt_model_path", ""):
            kwargs["whisper_model"] = stt_path
        if tts_dir := kwargs.pop("tts_models_dir", ""):
            kwargs["tts_models_dir"] = tts_dir

        if "vision_model" not in kwargs:
            kwargs["vision_model"] = self._pick_vision_model("qwen3-vl:8b")

        try:
            self._fetch_models(kwargs)
        except Exception as e:
            self.bus.emit("error", {"message": f"Could not download the speech models: {e}"})

        try:
            self.assistant = VoiceAssistant(
                persona=get_persona(persona_name),
                event_bus=self.bus,
                turn_queue=self._turns,
                plugins=self.plugins.plugins,
                **kwargs,
            )
            self.assistant.conversation.set_amnesia(amnesia)
            self.bus.emit("config", self.assistant.config_info())
            self._restore_thread()
        except Exception as e:
            self.bus.emit("error", {"message": f"Failed to start assistant: {e}"})
            self._publish_state("idle")
            return

        # Diagnostics only — must never prevent the assistant from running
        try:
            if self._warn_if_ollama_unreachable():
                self._warn_if_model_missing()
        except Exception:
            pass

        try:
            self.assistant.run_loop()
        except Exception as e:
            self.bus.emit("error", {"message": f"Assistant loop failed: {e}"})
        finally:
            self._poller_stop.set()
            self._publish_state("idle")

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

    def pause(self) -> dict:
        """Pause the conversation: the mic closes, speech holds mid-sentence,
        the reply stops streaming and queued turns wait. Models stay loaded."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.suspend_event.set()
        self.assistant.speech_pipeline.pause()
        self._publish_state("paused")
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def resume(self) -> dict:
        """Continue the conversation where pause() held it."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.suspend_event.clear()
        self.assistant.speech_pipeline.resume()
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

    def set_mic(self, on: bool) -> dict:
        """Switch the microphone. Off still answers typed messages."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        if on:
            self.assistant.pause_event.clear()
        else:
            self.assistant.pause_event.set()
            self._publish_state("muted")
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
        """Switch spoken replies on or off. Switching off cuts current speech."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.speak_replies = bool(enabled)
        if not enabled:
            self.assistant.speech_pipeline.stop()
        self.bus.emit("config", self.assistant.config_info())
        return {"ok": True}

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

    def set_wake_options(self, window=None, sleep_word=None) -> dict:
        """How long Iris stays awake after a turn, and its sleep word."""
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
        """Show the recent part of the continuous conversation after a restart."""
        memory = getattr(self.assistant, "memory", None)
        if memory is None:
            return
        for turn in memory.store.recent_turns():
            if turn["role"] == "user":
                self.bus.emit("chat.user", {"text": turn["text"], "source": turn["source"]})
            elif turn["role"] == "assistant":
                self.bus.emit("chat.done", {"text": turn["text"], "interrupted": False})

    def memory_topics(self) -> dict:
        """The topics Iris remembers, most recently discussed first."""
        memory = getattr(self.assistant, "memory", None)
        if memory is None:
            return {"topics": []}
        counts = memory.store.turn_counts()
        return {
            "topics": [
                {
                    "id": t.id,
                    "title": t.title,
                    "gist": t.gist,
                    "updated": t.updated,
                    "turns": counts.get(t.id, 0),
                }
                for t in memory.store.topics()
            ]
        }

    def memory_topic(self, topic_id: int) -> dict:
        """One topic with its summary and every turn filed under it."""
        memory = getattr(self.assistant, "memory", None)
        topic = memory.store.topic(int(topic_id)) if memory else None
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

    def set_amnesia(self, on: bool) -> dict:
        """While on, nothing of the conversation is written to memory."""
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

    def _progress(self, model: str, label: str):
        return lambda done, total: self.bus.emit(
            "setup.progress", {"model": model, "label": label, "done": done, "total": total}
        )

    def _fetch_stt(self, whisper_model: str, wake_word: str | None = None, **engine):
        """Download the speech recognition model unless it is cached or a
        local folder was chosen (used as it is)."""
        if Path(whisper_model).expanduser().is_dir():
            return
        resolved = resolve_engine(engine.get("whisper_engine", "auto"))
        device = resolve_device(resolved, engine.get("whisper_device", "auto"))
        repo = model_repo(resolve_model(whisper_model, resolved, device), resolved)
        if not downloads.is_cached(repo):
            downloads.download(repo, self._progress("stt", "Downloading speech recognition model"))
        if wake_word:
            tiny = model_repo("tiny", resolved)
            if not downloads.is_cached(tiny):
                downloads.download(tiny, self._progress("wake", "Downloading wake word model"))

    def _fetch_turn(self):
        """Download Smart Turn, which hears when the user has finished
        speaking, unless it is cached. Without it a turn ends after a silence."""
        if downloads.is_cached(SMART_TURN_REPO, SMART_TURN_FILE):
            return
        try:
            downloads.download(
                SMART_TURN_REPO,
                self._progress("turn", "Downloading turn detection model"),
                filename=SMART_TURN_FILE,
            )
        except Exception:
            pass

    def _fetch_tts(self, tts_dir: str | None):
        """Download the voice model unless a folder was chosen or the files
        are already in one of the usual places."""
        if tts_dir or find_kokoro_models() is not None:
            return
        downloads.download_files(
            [
                (KOKORO_MODEL_URL, DEFAULT_KOKORO_DIR / KOKORO_MODEL_FILE),
                (KOKORO_VOICES_URL, DEFAULT_KOKORO_DIR / KOKORO_VOICES_FILE),
            ],
            self._progress("tts", "Downloading voice model"),
        )

    def _fetch_models(self, kwargs: dict):
        """Download the speech models that are neither found nor pointed to,
        with visible progress.

        On a fresh install the first launch fetches large models (1.6 GB for
        large-v3-turbo, 335 MB for the voice); without this the UI would only
        say "Loading models". A model folder chosen in Settings is used as it
        is and never downloaded.
        """
        try:
            # Whisper's models; another recognition engine fetches its own
            if kwargs.get("recognition_engine", "whisper") == "whisper":
                self._fetch_stt(
                    kwargs.get("whisper_model", "auto"),
                    kwargs.get("wake_word"),
                    whisper_engine=kwargs.get("whisper_engine", "auto"),
                    whisper_device=kwargs.get("whisper_device", "auto"),
                )
            self._fetch_turn()
            self._fetch_tts(kwargs.get("tts_models_dir"))
        finally:
            self.bus.emit("setup.progress", {"label": "", "done": 0, "total": 0})

    def setup_info(self) -> dict:
        """For the first-launch setup: what would be downloaded, roughly how
        much, and what is already on this machine."""
        engine = resolve_engine("auto")
        model = resolve_model("auto", engine, resolve_device(engine))
        return {
            "stt": {
                "engine": engine,
                "model": model,
                "cached": downloads.is_cached(model_repo(model, engine)),
                "size_mb": APPROX_MODEL_MB.get(model),
            },
            "tts": {"found": str(find_kokoro_models() or ""), "size_mb": KOKORO_MB},
            "ollama_url": effective_host(self._ollama_host),
        }

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
                    "size_mb": APPROX_MODEL_MB.get(size),
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
            if self.assistant.conversation.vision_model:
                conversation = self.assistant.conversation
                conversation.vision_model = (
                    self._pick_vision_model("qwen3-vl:8b") or conversation.vision_model
                )
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

    def set_persona(self, name: str) -> dict:
        """Hot-swap the persona (prompt, voice, speed, audio filter)."""
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
        if previous != persona.name:
            assistant.conversation.add_note(
                f"You are now {persona.name}. Earlier replies in this conversation "
                f"were given by {previous}; do not continue {previous}'s name or manner."
            )
        generator = assistant.tts_generator
        generator.engine = persona.engine
        generator.voice = persona.voice
        generator.speed = persona.speed
        # A language the new voice cannot speak is not kept up
        if not generator.speaks(assistant.conversation_language):
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
        self.bus.emit("config", assistant.config_info())
        return {"ok": True, "persona": name}

    def persona_voices(self) -> dict:
        """For the persona editor: the speech engines (names, whether they
        take a speed) and their voices, and the audio filters."""
        models_dir = self.assistant.tts_generator.models_dir if self.assistant else None
        registered = engines.speech_engines()
        return {
            "engines": {engine: list_voices(engine, models_dir) for engine in registered},
            "engine_info": [
                {"id": engine.id, "name": engine.name, "takes_speed": engine.takes_speed}
                for engine in registered.values()
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
        persona_id = save_persona(settings, settings.get("id") or None)
        if self.assistant is not None and self.assistant.persona.id == persona_id:
            self.set_persona(persona_id)
        return {"ok": True, "id": persona_id, **self.list_personas()}

    def persona_delete(self, persona_id: str) -> dict:
        """Delete the user's persona, or reset an edited one that ships with
        Iris. When it was the current persona, Iris takes over."""
        still_there = delete_persona(persona_id)
        current = self.assistant.persona.id if self.assistant else None
        if current == persona_id:
            current = persona_id if still_there else "iris"
            self.set_persona(current)
        return {"ok": True, "persona": current, **self.list_personas()}

    def persona_preview(self, settings: dict) -> dict:
        """Say a line in a persona's voice, as edited, cutting off whatever is
        being said. It goes through the speech pipeline, so the mic stays
        closed while it plays."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        engine, voice = settings.get("engine"), settings.get("voice")
        if engine not in engines.speech_engines() or not voice:
            return {"ok": False, "error": "No voice chosen"}
        # In the voice's own language, when Iris can speak it with that engine
        language = voice_language(engine, voice)
        if not engine_speaks(engine, language):
            language = "en"
        name = (settings.get("name") or "").strip() or "Iris"
        pipeline = self.assistant.speech_pipeline
        pipeline.stop()
        pipeline.queue_speech(
            PREVIEW_LINES.get(language, PREVIEW_LINES["en"]).format(name=name),
            language=language,
            sound={
                "engine": engine,
                "voice": voice,
                "speed": float(settings.get("speed") or 1.0),
                "audio_filter": settings.get("audio_filter") or "",
            },
        )
        return {"ok": True}

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

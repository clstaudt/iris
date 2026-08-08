"""GUI-facing controller: runs the VoiceAssistant off the main thread and
exposes commands (start/stop/interrupt/send_text) and a single effective
state stream for external UIs such as the Electron shell."""

import importlib.metadata
import threading
import time

from iris.assistant import VoiceAssistant
from iris.audio import AudioRecorder, get_filter
from iris.events import EventBus
from iris.llm import ConversationManager
from iris.personas import get_persona, list_personas

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
)

# How long "speaking" survives a gap in playback, in seconds.
SPEAKING_HOLD = 0.6


class AssistantController:
    """Owns a VoiceAssistant instance and translates it into bus events.

    The assistant's run_loop blocks on microphone input, so all lifecycle
    work happens on daemon threads. Commands return immediately; results and
    state changes arrive as bus events.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.assistant: VoiceAssistant | None = None

        self._loop_thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._chat_lock = threading.Lock()

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

        bus.subscribe(self._on_bus_event)

    # ------------------------------------------------------------------
    # Effective state computation (flags + speech pipeline activity)
    # ------------------------------------------------------------------

    def _on_bus_event(self, event_type: str, data: dict):
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

                if self.assistant.pause_event.is_set():
                    state = "paused"
                elif now < self._speaking_until:
                    state = "speaking"
                else:
                    state = self._flag_state()
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
        installed = ConversationManager.list_available_models()
        names = {m["name"] for m in installed}
        if preferred in names:
            return preferred
        for model in installed:
            if model["vision"]:
                return model["name"]
        return None

    def _run(self, options: dict):
        kwargs = dict(options)
        persona_name = kwargs.pop("persona", "iris")

        # The GUI has no CLI flags, so tools are on by default; each one
        # degrades to a warning if its optional dependencies are missing.
        kwargs.setdefault("enable_vision", True)
        kwargs.setdefault("enable_screen_capture", True)
        kwargs.setdefault("enable_web_search", True)
        if "vision_model" not in kwargs:
            kwargs["vision_model"] = self._pick_vision_model("qwen3-vl:8b")

        try:
            self.assistant = VoiceAssistant(
                persona=get_persona(persona_name),
                event_bus=self.bus,
                **kwargs,
            )
            self.bus.emit("config", self.assistant.config_info())
        except Exception as e:
            self.bus.emit("error", {"message": f"Failed to start assistant: {e}"})
            self._publish_state("idle")
            return

        # Diagnostics only — must never prevent the assistant from running
        try:
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
        """Pause listening. Aborts an in-progress recording within ~50 ms;
        models stay loaded so resume is instant."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.pause_event.set()
        self._publish_state("paused")
        return {"ok": True}

    def resume(self) -> dict:
        """Resume listening after a pause."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        self.assistant.pause_event.clear()
        return {"ok": True}

    def send_text(self, text: str) -> dict:
        """Send a typed message through the same chat path as voice input."""
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "Empty message"}
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}

        self.bus.emit(
            "chat.user",
            {"text": text, "language": self.assistant.conversation_language},
        )

        def work():
            with self._chat_lock:
                self.assistant.chat(
                    text, response_language=self.assistant.conversation_language
                )

        threading.Thread(target=work, daemon=True).start()
        return {"ok": True}

    def _warn_if_model_missing(self):
        """Report a missing Ollama model up front rather than on first reply."""
        available = [m["name"] for m in ConversationManager.list_available_models()]
        model = self.assistant.conversation.model
        if available and model not in available:
            self.bus.emit(
                "error",
                {
                    "message": (
                        f"Ollama model '{model}' is not installed. Pick another one "
                        f"from the model menu, or install it with: ollama pull {model}"
                    )
                },
            )

    def list_models(self) -> dict:
        """List Ollama models usable for conversation."""
        return {"models": ConversationManager.list_available_models()}

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

    def set_persona(self, name: str) -> dict:
        """Hot-swap the persona (prompt, voice, speed, audio filter)."""
        if self.assistant is None:
            return {"ok": False, "error": "Assistant not running"}
        if name not in list_personas():
            return {"ok": False, "error": f"Unknown persona: {name}"}

        persona = get_persona(name)
        assistant = self.assistant
        assistant.persona = persona
        assistant.conversation.persona = persona
        assistant.tts_generator.voice = persona.voice
        assistant.tts_generator.speed = persona.speed
        assistant.player.audio_filter = (
            get_filter(persona.audio_filter) if persona.audio_filter else None
        )
        self.bus.emit("config", assistant.config_info())
        return {"ok": True, "persona": name}

    def info(self) -> dict:
        try:
            version = importlib.metadata.version("iris")
        except importlib.metadata.PackageNotFoundError:
            version = "dev"
        with self._state_lock:
            state = self._last_state
        result = {
            "version": version,
            "personas": list_personas(),
            "state": state,
            "running": bool(self._loop_thread and self._loop_thread.is_alive()),
        }
        if self.assistant is not None:
            result["config"] = self.assistant.config_info()
        return result

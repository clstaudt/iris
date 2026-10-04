"""GUI-facing controller: runs the VoiceAssistant off the main thread and
exposes commands (start/stop/interrupt/send_text) and a single effective
state stream for external UIs such as the Electron shell."""

import importlib.metadata
import itertools
import queue
import threading
import time

from iris.assistant import VoiceAssistant
from iris.audio import AudioRecorder, get_filter
from iris.events import EventBus
from iris.llm import ConversationManager, normalize_host, ollama_client
from iris.personas import get_persona, list_personas, reload_personas

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

# Events that make up the visible conversation. Kept here so a UI that
# reloads (or attaches late) can rebuild its transcript; streaming chunks are
# left out because chat.done carries the full reply.
TRANSCRIPT_EVENTS = ("chat.user", "chat.done", "tool.use", "error")
TRANSCRIPT_LIMIT = 500


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

        # Typed turns. Owned here rather than by the assistant so messages sent
        # while the models are still loading are answered once they are ready.
        self._turns: queue.Queue = queue.Queue()
        self._turn_ids = itertools.count(1)

        # Ollama server chosen in the UI; None follows OLLAMA_HOST
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

        bus.subscribe(self._on_bus_event)

    # ------------------------------------------------------------------
    # Effective state computation (flags + speech pipeline activity)
    # ------------------------------------------------------------------

    def _on_bus_event(self, event_type: str, data: dict):
        if event_type in TRANSCRIPT_EVENTS:
            with self._transcript_lock:
                self._transcript.append({"type": event_type, "data": dict(data)})
                del self._transcript[:-TRANSCRIPT_LIMIT]
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
        # {group: enabled} chosen in the UI; applied once the tools exist
        tool_choices = kwargs.pop("tools", {}) or {}
        amnesia = bool(kwargs.pop("amnesia", False))

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
                turn_queue=self._turns,
                **kwargs,
            )
            if self.assistant.tools is not None:
                for group, enabled in tool_choices.items():
                    self.assistant.tools.set_group(group, bool(enabled))
            self.assistant.conversation.set_amnesia(amnesia)
            self.bus.emit("config", self.assistant.config_info())
            self._restore_thread()
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

    def send_text(self, text: str) -> dict:
        """Queue a typed message as the next turn of the conversation.

        Typed and spoken turns are taken one at a time by the assistant's loop,
        so a message sent mid-reply waits for that reply to finish. Its
        chat.user event (carrying the returned id) is emitted when it starts.
        """
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "Empty message"}
        turn_id = f"t{next(self._turn_ids)}"
        self._turns.put({"text": text, "id": turn_id})
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

    def set_tool(self, group: str, enabled: bool) -> dict:
        """Switch a tool group (camera, screen, web) on or off."""
        if self.assistant is None or self.assistant.tools is None:
            return {"ok": False, "error": "Assistant not running"}
        if not self.assistant.tools.set_group(group, bool(enabled)):
            return {"ok": False, "error": f"Tool '{group}' is not available"}
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

    def _warn_if_model_missing(self):
        """Report a missing Ollama model up front rather than on first reply."""
        available = [
            m["name"] for m in ConversationManager.list_available_models(self._ollama_host)
        ]
        model = self.assistant.conversation.model
        if available and model not in available:
            self.bus.emit(
                "error",
                {
                    "message": (
                        f"Ollama model '{model}' is not installed. Pick another one "
                        f"in Settings, or install it with: ollama pull {model}"
                    ),
                    # Lets the UI retract the warning once another model is chosen
                    "kind": "model_missing",
                },
            )

    def list_models(self) -> dict:
        """List Ollama models usable for conversation."""
        return {"models": ConversationManager.list_available_models(self._ollama_host)}

    def set_ollama_host(self, url: str) -> dict:
        """Point the assistant at another Ollama server ("" = default).

        The server is asked for its models first, so an unreachable URL is
        reported instead of breaking the next reply.
        """
        host = normalize_host(url)
        try:
            ollama_client(host, timeout=8).list()
        except Exception as e:
            where = host or "the default server"
            return {"ok": False, "error": f"Cannot reach Ollama at {where}: {e}"}
        self._ollama_host = host
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
        assistant.tts_generator.voice = persona.voice
        assistant.tts_generator.speed = persona.speed
        assistant.player.audio_filter = (
            get_filter(persona.audio_filter) if persona.audio_filter else None
        )
        self.bus.emit("config", assistant.config_info())
        return {"ok": True, "persona": name}

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
        }
        if self.assistant is not None:
            result["config"] = self.assistant.config_info()
        return result

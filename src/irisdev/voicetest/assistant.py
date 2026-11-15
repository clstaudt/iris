"""The assistant under test.

A test talks to an assistant through a small interface (AssistantUnderTest):
it starts it on an endpoint of a space, reads what it reports, presses Stop,
pauses it, changes its settings and reads or seeds its conversation.
IrisUnderTest is Iris's: the controller of the app, with its real
recognition, turn detection, echo canceller, language model and voice,
running in this process. Another assistant is tested by implementing the
same interface.
"""

import copy
import logging
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Protocol

from iris.audio import voice_log
from iris.controller import AssistantController
from iris.events import EventBus
from iris.turns import SAMPLE_RATE, SmartTurn
from irisdev.voicetest.acoustics import Endpoint
from irisdev.voicetest.isolation import isolate

# States in which the assistant is working on a turn
BUSY = frozenset(
    {"understanding", "thinking", "speaking", "looking", "capturing_screen", "searching", "browsing", "summarizing"}
)
# The assistant has finished once it has been quiet and idle this long
QUIET = 1.5
# How long a setting may take to show in the assistant's config
CONFIG_WAIT = 30.0
# The tools Iris is held with unless a conversation switches them on
DEFAULT_TOOLS = {"camera": False, "screen": False, "web": False}


def wait_until(condition, timeout: float, step: float = 0.05) -> bool:
    """Wait for a condition; whether it came true within the timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(step)
    return bool(condition())


class Log:
    """Every event an assistant reported, with when it came."""

    def __init__(self):
        self.events: list[tuple[float, str, dict]] = []
        self._by_kind: dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def __call__(self, kind: str, data: dict):
        event = (time.monotonic(), kind, dict(data))
        with self._lock:
            self.events.append(event)
            self._by_kind[kind].append(event)

    def since(self, start: float, kind: Optional[str] = None, end: float = float("inf")):
        with self._lock:
            events = self.events if kind is None else self._by_kind.get(kind, [])
            return [(at, k, d) for at, k, d in events if start <= at <= end]

    def last(self, kind: str) -> Optional[dict]:
        with self._lock:
            found = self._by_kind.get(kind)
            return found[-1][2] if found else None

    def last_before(self, moment: float, kind: str) -> Optional[tuple]:
        """The last event of a kind before a moment."""
        with self._lock:
            found = [e for e in self._by_kind.get(kind, []) if e[0] < moment]
        return found[-1] if found else None


class AssistantUnderTest(Protocol):
    """What a test needs of an assistant.

    It reports what happens as events in its log, as Iris does: "state"
    {state}, "flags" {flags}, "chat.user" {text, language, source} for each
    turn it heard, "chat.continued" when the user went on with a turn it had
    taken, "chat.done" {text, interrupted} for each reply, "tool.use" {tool,
    label, detail}, "wake", "wake.window" {remaining}, "sleep", "config"
    {...}, "turn.timing" {stages, components} and "error" {message}. Iris
    also logs what its voice did over speech (iris.voice) as "voice"
    {message}: each hold, what it was judged to be, where the voice went on.
    """

    name: str
    endpoint: Endpoint
    log: Log

    def start(self) -> None: ...
    def close(self) -> None: ...
    def interrupt(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def set_wake_word(self, word: str) -> None: ...
    def send_text(self, text: str) -> None: ...
    def clear_conversation(self) -> None: ...
    def configure(self, name: str, value) -> bool: ...
    def history(self) -> list[dict]: ...
    def seed_history(self, messages: list[dict]) -> None: ...


# Smart Turn's verdicts, logged by every Iris that watches them -----------------

_ORIGINAL_JUDGE = SmartTurn.__call__
_WATCHERS: list = []


def _judge_and_log(self, audio):
    """SmartTurn.__call__, logging each verdict to the Iris it judges for."""
    started = time.perf_counter()
    probability = _ORIGINAL_JUDGE(self, audio)
    took = time.perf_counter() - started
    watchers = list(_WATCHERS)
    mine = [w for w in watchers if w.judges_with(self)] or watchers
    for watcher in mine:
        watcher.log("turn.verdict", {"probability": round(float(probability), 3),
                                     "seconds": round(len(audio) / SAMPLE_RATE, 2), "took": round(took, 3)})
    return probability


def watch_verdicts(watcher):
    """Log Smart Turn's verdicts to this watcher's log (watcher.log, and
    watcher.judges_with(smart_turn)). The class is patched, not an instance:
    a listen() already waiting holds on to the instance it started with."""
    if SmartTurn.__call__ is not _judge_and_log:
        SmartTurn.__call__ = _judge_and_log
    if watcher not in _WATCHERS:
        _WATCHERS.append(watcher)


def unwatch_verdicts(watcher):
    if watcher in _WATCHERS:
        _WATCHERS.remove(watcher)
    if not _WATCHERS:
        SmartTurn.__call__ = _ORIGINAL_JUDGE


class _VoiceLines(logging.Handler):
    """Iris's voice log (iris.voice), as "voice" {message} events in a log."""

    def __init__(self, log: Log):
        super().__init__(logging.INFO)
        self.log = log

    def emit(self, record: logging.LogRecord):
        self.log("voice", {"message": record.getMessage()})


class IrisUnderTest:
    """Iris, as the app runs it, on an endpoint of a space.

    The voice log (iris.voice) is one per process: with two Irises in one
    (self-talk), each logs the other's lines too.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        llm_host: str,
        llm_model: str,
        persona: str = "iris",
        tools: Optional[dict] = None,
        start_timeout: float = 600.0,
        home: Optional[Path] = None,
        stub=None,
    ):
        """
        Args:
            endpoint: Its microphone and speaker
            llm_host: The language model's server
            llm_model: The model
            persona: Who it is (one shipped with Iris)
            tools: {tool: on}; all off by default, so replies do not depend
                on the camera, the screen or the web
            home: The folder its personas, memory and plugins are kept in,
                instead of the user's (a new temporary one if not given)
            stub: The scripted model it talks to, if it does: its requests
                are logged as "stub.request" {kind, index}
        """
        self.home = isolate(home)
        self.name = persona
        self.endpoint = endpoint
        self.log = Log()
        self.persona = persona
        self.stub = stub
        if stub is not None:
            stub.on_request = lambda kind, record: self.log("stub.request", {"kind": kind, "index": record.index})
        self.options = {
            "persona": persona,
            "llm_model": llm_model,
            "ollama_host": llm_host,
            "memory": False,
            "tools": dict(tools) if tools is not None else dict(DEFAULT_TOOLS),
            "speak_replies": True,
            "barge_in": True,
            "turn_detection": "smart_turn",
            "audio": endpoint,
        }
        self.start_timeout = start_timeout
        # Memories it was started with (set_memory)
        self._memories = 0
        self.controller: Optional[AssistantController] = None
        # The Whisper size it started with
        self.default_stt = ""
        self._voice_lines = _VoiceLines(self.log)

    @property
    def assistant(self):
        return self.controller.assistant

    def state(self) -> str:
        last = self.log.last("state")
        return last.get("state", "") if last else ""

    def start(self):
        bus = EventBus()
        bus.subscribe(self.log)
        self.controller = AssistantController(bus)
        watch_verdicts(self)
        voice_log.addHandler(self._voice_lines)
        if not voice_log.isEnabledFor(logging.INFO):
            voice_log.setLevel(logging.INFO)
        self.controller.start(dict(self.options))
        if not wait_until(lambda: self.state() in ("listening", "dormant"), self.start_timeout, 0.2):
            errors = [d.get("message") for _, _, d in self.log.since(0, "error")]
            raise RuntimeError(f"{self.name} did not start listening: {errors or self.state()}")
        if self.assistant.echo is None:
            raise RuntimeError("No echo canceller: interrupting by voice cannot be tested")
        self._count_model_rounds()
        self.default_stt = self.config().get("whisper_model", "")
        # Load what loads on first use, so the first turn is measured fairly
        warm = self.assistant.tts_generator.generate("Hello.", language="en")
        if warm:
            Path(warm).unlink(missing_ok=True)

    def restart(self):
        """A new Iris on the same endpoint and log: nothing of the last one's
        conversation, echo canceller or loaded voice carries over, except
        what it keeps in its memory."""
        self.close()
        self.start()

    def set_memory(self, on: bool):
        """Start Iris again with long-term memory on, as the app runs it, in
        a memory of its own (a new one each time), or off."""
        if on:
            self._memories += 1
            os.environ["IRIS_MEMORY_DB"] = str(self.home / f"memory-{self._memories}.db")
        self.options["memory"] = on
        self.restart()

    def _count_model_rounds(self):
        """Log each request to the model as "model.ask" {round}, to count the
        rounds a reply took (a real model's included)."""
        conversation = self.assistant.conversation
        ask = conversation._ask

        def counted(model, messages, **extra):
            rounds = sum(1 for m in messages if m.get("role") == "tool")
            self.log("model.ask", {"tools": bool(extra.get("tools")), "tool_results": rounds})
            return ask(model, messages, **extra)

        conversation._ask = counted

    def judges_with(self, smart_turn) -> bool:
        recorder = getattr(self.controller.assistant, "recorder", None) if self.controller else None
        return getattr(recorder, "turn_end", None) is smart_turn

    def warm_up(self, wav_path: str):
        """Recognize a recording once, so the recognizer is loaded."""
        self.assistant.transcriber.transcribe(wav_path)

    def transcribe(self, wav_path: str) -> tuple[str, str]:
        """What Iris's own recognizer hears in a recording, and its language."""
        return self.assistant.transcriber.transcribe(wav_path)

    def close(self):
        unwatch_verdicts(self)
        voice_log.removeHandler(self._voice_lines)
        if self.controller is not None:
            self.controller.stop()
            wait_until(lambda: self.state() == "idle", 10)

    def interrupt(self):
        self.controller.interrupt()

    def pause(self):
        self.controller.pause()

    def resume(self):
        self.controller.resume()

    def set_wake_word(self, word: str):
        self.controller.set_wake_word(word)

    def send_text(self, text: str):
        self.controller.send_text(text)

    def clear_conversation(self):
        """A new conversation: nothing said before, and in English again."""
        self.assistant.conversation.conversation_history.clear()
        self.assistant.conversation_language = "en"
        self.assistant.conversation.language = "en"

    def seed_memory(self, topics: list[dict]):
        """Topics remembered from earlier conversations ({title, gist,
        summary}), as filing leaves them: listed in the memory index, their
        turns long out of the recent conversation."""
        store = self.assistant.memory.store
        for topic in topics:
            store.create_topic(topic["title"], topic["gist"], topic["summary"])

    def speaking(self) -> bool:
        """Whether it is still speaking its reply (optional for an assistant
        under test: without it, a pause between two sentences can pass for
        the end of the reply)."""
        return self.assistant.speech_pipeline.is_active

    def history(self) -> list[dict]:
        """A copy of the conversation as the model will be given it."""
        return copy.deepcopy(self.assistant.conversation.conversation_history)

    def seed_history(self, messages: list[dict]):
        self.assistant.conversation.conversation_history.extend(copy.deepcopy(messages))

    def config(self) -> dict:
        return self.assistant.config_info()

    def stt_downloaded(self) -> list[str]:
        """The Whisper sizes on this computer (switching to another would
        download it first)."""
        return [m["id"] for m in self.controller.stt_models().get("models", []) if m.get("cached")]

    def voice_queued(self) -> bool:
        """Whether more of the reply is still to be played than the clip
        playing now: text waiting to be synthesized, being synthesized, or
        audio waiting to be played."""
        pipeline = self.assistant.speech_pipeline
        with pipeline._status_lock:
            generating = pipeline.is_generating_flag.is_set() and pipeline._generating_epoch == pipeline._epoch
        return generating or not pipeline.text_queue.empty() or not pipeline.audio_queue.empty()

    def refresh_model(self):
        """Read what the model is again (the scripted model's changed): its
        context length and capabilities, forgetting tools it refused, and
        which model on the server sees, as Iris picks it at its start."""
        conversation = self.assistant.conversation
        conversation._capability_cache.clear()
        conversation._no_tools.clear()
        conversation.max_context_tokens = conversation._get_model_context_length()
        conversation.vision_model = self.controller._pick_vision_model()

    def configure(self, name: str, value, wait: float = CONFIG_WAIT) -> bool:
        """Change a setting as the app does, and wait for the "config" event
        that shows it. Settings: "tool" (id, on), "persona", "turn_detection",
        "stt_model", "wake_word", "wake_options" {window, sleep_word}, "mic",
        "model" and "memory" (which starts Iris again). Returns whether it
        showed in time."""
        since = time.monotonic()
        answer, shows = self._change(name, value)
        if not answer.get("ok", False):
            self.log("error", {"message": f"Could not set {name} to {value!r}: {answer.get('error')}"})
            return False
        return wait_until(lambda: any(shows(d) for _, _, d in self.log.since(since, "config")), wait, 0.1)

    def _change(self, name: str, value):
        """Change a setting: the controller's answer, and a test of whether
        a config shows the change."""
        controller = self.controller
        if name == "memory":
            self.set_memory(bool(value))
            return {"ok": True}, lambda config: config.get("memory") == bool(value)
        if name == "tool":
            plugin, on = value

            def shows(config):
                return (config.get("tools", {}).get(plugin) or {}).get("enabled") == bool(on)

            return controller.set_tool(plugin, bool(on)), shows
        if name == "wake_options":
            window, sleep_word = value.get("window"), value.get("sleep_word")

            def shows(config):
                window_shown = window is None or config.get("wake_window") == max(5.0, float(window))
                word_shown = sleep_word is None or config.get("sleep_word") == sleep_word.strip()
                return window_shown and word_shown

            return controller.set_wake_options(window, sleep_word), shows
        setters = {
            "persona": ("set_persona", "persona_id", value),
            "turn_detection": ("set_turn_detection", "turn_detection", value),
            "stt_model": ("set_stt_model", "whisper_model", value),
            "wake_word": ("set_wake_word", "wake_word", value or ""),
            "mic": ("set_mic", "mic_on", bool(value)),
            "model": ("set_model", "llm_model", value),
        }
        if name not in setters:
            raise ValueError(f"No setting called {name!r}")
        setter, key, shown = setters[name]

        def shows(config):
            return config.get(key) == shown

        return getattr(controller, setter)(shown), shows


def busy(assistant: AssistantUnderTest) -> bool:
    last = assistant.log.last("state")
    return bool(last) and last.get("state") in BUSY


def finished(assistant: AssistantUnderTest, quiet: float = QUIET) -> bool:
    """The assistant is idle, has nothing more to say, and its speaker has
    been silent for a while. A reply written whole can still be on its way
    to the speaker: Iris shows no state while its first clip is synthesized."""
    speaker = assistant.endpoint.speaker
    speaking = getattr(assistant, "speaking", None)
    if busy(assistant) or speaker.playing() or (speaking is not None and speaking()):
        return False
    last = speaker.last_sound()
    return last is None or time.monotonic() - last >= quiet

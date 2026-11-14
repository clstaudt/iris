"""The assistant under test.

A test talks to an assistant through a small interface (AssistantUnderTest):
it starts it on an endpoint of a space, reads what it reports, presses Stop,
sets a wake word and clears its conversation. IrisUnderTest is Iris's: the
controller of the app, with its real recognition, turn detection, echo
canceller, language model and voice, running in this process. Another
assistant is tested by implementing the same interface.
"""

import threading
import time
from pathlib import Path
from typing import Optional, Protocol

from iris.controller import AssistantController
from iris.events import EventBus
from irisdev.voicetest.acoustics import Endpoint
from irisdev.voicetest.isolation import isolate

# States in which the assistant is working on a turn
BUSY = frozenset(
    {"understanding", "thinking", "speaking", "looking", "capturing_screen", "searching", "browsing", "summarizing"}
)
# The assistant has finished once it has been quiet and idle this long
QUIET = 1.5


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
        self._lock = threading.Lock()

    def __call__(self, kind: str, data: dict):
        with self._lock:
            self.events.append((time.monotonic(), kind, dict(data)))

    def since(self, start: float, kind: Optional[str] = None, end: float = float("inf")):
        with self._lock:
            return [(at, k, d) for at, k, d in self.events if start <= at <= end and (kind is None or k == kind)]

    def last(self, kind: str) -> Optional[dict]:
        found = self.since(0, kind)
        return found[-1][2] if found else None


class AssistantUnderTest(Protocol):
    """What a test needs of an assistant.

    It reports what happens as events in its log, as Iris does: "state"
    {state}, "chat.user" {text} for each turn it heard, "chat.continued"
    when the user went on with a turn it had taken, "chat.done" {text,
    interrupted} for each reply, "turn.timing" {stages} and "error"
    {message}.
    """

    name: str
    endpoint: Endpoint
    log: Log

    def start(self) -> None: ...
    def close(self) -> None: ...
    def interrupt(self) -> None: ...
    def set_wake_word(self, word: str) -> None: ...
    def send_text(self, text: str) -> None: ...
    def clear_conversation(self) -> None: ...


class IrisUnderTest:
    """Iris, as the app runs it, on an endpoint of a space."""

    def __init__(
        self,
        endpoint: Endpoint,
        llm_host: str,
        llm_model: str,
        persona: str = "iris",
        tools: Optional[dict] = None,
        start_timeout: float = 600.0,
        home: Optional[Path] = None,
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
        """
        self.home = isolate(home)
        self.name = persona
        self.endpoint = endpoint
        self.log = Log()
        bus = EventBus()
        bus.subscribe(self.log)
        self.controller = AssistantController(bus)
        self.options = {
            "persona": persona,
            "llm_model": llm_model,
            "ollama_host": llm_host,
            "memory": False,
            "tools": tools if tools is not None else {"camera": False, "screen": False, "web": False},
            "speak_replies": True,
            "barge_in": True,
            "turn_detection": "smart_turn",
            "audio": endpoint,
        }
        self.start_timeout = start_timeout

    @property
    def assistant(self):
        return self.controller.assistant

    def state(self) -> str:
        last = self.log.last("state")
        return last.get("state", "") if last else ""

    def start(self):
        self.controller.start(dict(self.options))
        if not wait_until(lambda: self.state() in ("listening", "dormant"), self.start_timeout, 0.2):
            errors = [d.get("message") for _, _, d in self.log.since(0, "error")]
            raise RuntimeError(f"{self.name} did not start listening: {errors or self.state()}")
        if self.assistant.echo is None:
            raise RuntimeError("No echo canceller: interrupting by voice cannot be tested")
        # Load what loads on first use, so the first turn is measured fairly
        self.assistant.tts_generator.generate("Hello.", language="en")

    def warm_up(self, wav_path: str):
        """Recognize a recording once, so the recognizer is loaded."""
        self.assistant.transcriber.transcribe(wav_path)

    def close(self):
        self.controller.stop()
        wait_until(lambda: self.state() == "idle", 10)

    def interrupt(self):
        self.controller.interrupt()

    def set_wake_word(self, word: str):
        self.controller.set_wake_word(word)

    def send_text(self, text: str):
        self.controller.send_text(text)

    def clear_conversation(self):
        self.assistant.conversation.conversation_history.clear()

    def speaking(self) -> bool:
        """Whether it is still speaking its reply (optional for an assistant
        under test: without it, a pause between two sentences can pass for
        the end of the reply)."""
        return self.assistant.speech_pipeline.is_active


def busy(assistant: AssistantUnderTest) -> bool:
    last = assistant.log.last("state")
    return bool(last) and last.get("state") in BUSY


def finished(assistant: AssistantUnderTest, quiet: float = QUIET) -> bool:
    """The assistant is idle and its speaker has been silent for a while."""
    speaker = assistant.endpoint.speaker
    if busy(assistant) or speaker.playing():
        return False
    last = speaker.last_sound()
    return last is None or time.monotonic() - last >= quiet

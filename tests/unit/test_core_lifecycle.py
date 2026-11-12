"""The core's own part in starting, failing and stopping: a start that
fails is reported and leaves Iris idle, ready to be started again; a loop
that fails is reported; a core the app starts again after one stopped does
not show the conversation a second time; and the core does not outlive the
app that started it."""

import os
import signal
import threading
import time
from types import SimpleNamespace

from iris import rpc_server
from iris.controller import AssistantController
from iris.events import EventBus


class Recorded:
    def __init__(self, bus: EventBus):
        self.events: list[tuple[str, dict]] = []
        bus.subscribe(lambda kind, data: self.events.append((kind, dict(data))))

    def of(self, kind: str) -> list[dict]:
        return [d for k, d in self.events if k == kind]


def _wait_for(condition, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return bool(condition())


class Assistant:
    """Stands in for VoiceAssistant; its run_loop may fail."""

    fails = None
    turns = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stop_event = threading.Event()
        self.speech_pipeline = SimpleNamespace(is_active=False, stop=lambda: None)
        self.suspend_event = threading.Event()
        self.pause_event = threading.Event()
        self.hold_event = threading.Event()
        self.interrupt_event = threading.Event()
        self.conversation = SimpleNamespace(set_amnesia=lambda on: None)
        self.memory = SimpleNamespace(store=SimpleNamespace(recent_turns=lambda: list(Assistant.turns)))

    def config_info(self):
        return {}

    def run_loop(self):
        if Assistant.fails is not None:
            raise Assistant.fails
        self.stop_event.wait(5)


def _controller(monkeypatch, assistant=Assistant):
    monkeypatch.setattr("iris.controller.VoiceAssistant", assistant)
    monkeypatch.setattr(AssistantController, "_fetch_models", lambda self, kwargs: None)
    monkeypatch.setattr(AssistantController, "_pick_vision_model", lambda self, preferred=None: None)
    monkeypatch.setattr(AssistantController, "_warn_if_ollama_unreachable", lambda self: False)
    bus = EventBus()
    return AssistantController(bus), Recorded(bus)


def test_a_start_that_fails_is_reported_and_iris_can_be_started_again(monkeypatch):
    attempts = []

    def failing(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise OSError("cannot open the memory database")
        return Assistant(**kwargs)

    controller, seen = _controller(monkeypatch, failing)
    Assistant.fails = None
    assert controller.start({})["ok"]
    assert _wait_for(lambda: seen.of("error"))
    assert seen.of("error") == [{"message": "Failed to start assistant: cannot open the memory database"}]
    assert _wait_for(lambda: not controller.info()["running"])
    assert seen.of("state")[-1] == {"state": "idle"}

    # Nothing is left half started: starting again works
    again = controller.start({})
    assert again["ok"] and "already_running" not in again
    assert _wait_for(lambda: controller.info()["running"])
    assert len(attempts) == 2
    controller.stop()


def test_a_loop_that_fails_is_reported(monkeypatch):
    controller, seen = _controller(monkeypatch)
    Assistant.fails = RuntimeError("the audio thread died")
    try:
        controller.start({})
        assert _wait_for(lambda: seen.of("error"))
        assert seen.of("error") == [{"message": "Assistant loop failed: the audio thread died"}]
        assert _wait_for(lambda: seen.of("state")[-1] == {"state": "idle"})
        assert not controller.info()["running"]
    finally:
        Assistant.fails = None


def test_a_core_started_again_does_not_show_the_conversation_twice(monkeypatch):
    """The app keeps showing the conversation when the core stopped and was
    started again; the new core does not send its recent turns again."""
    Assistant.turns = [
        {"role": "user", "text": "Hello", "source": "voice"},
        {"role": "assistant", "text": "Hi there."},
    ]
    try:
        controller, seen = _controller(monkeypatch)
        controller.start({})
        assert _wait_for(lambda: len(seen.of("chat.done")) == 1)
        assert seen.of("chat.user") == [{"text": "Hello", "source": "voice"}]
        controller.stop()
        assert _wait_for(lambda: not controller.info()["running"])

        again, seen_again = _controller(monkeypatch)
        again.start({"restore_thread": False})
        assert _wait_for(lambda: again.info()["running"])
        time.sleep(0.2)
        assert not seen_again.of("chat.user") and not seen_again.of("chat.done")
        # The option is the controller's, not the assistant's
        assert "restore_thread" not in again.assistant.kwargs
        again.stop()
    finally:
        Assistant.turns = []


def test_the_core_ends_itself_when_the_app_that_started_it_is_gone(monkeypatch):
    """A core whose parent died is taken in by the system (its parent id
    changes): it must not linger holding the microphone."""
    parents = iter([4242] + [4242] * 3 + [1] * 100)
    killed = []
    monkeypatch.setattr(rpc_server.os, "getppid", lambda: next(parents))
    monkeypatch.setattr(rpc_server.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    rpc_server._start_parent_watchdog(interval=0.01)
    assert _wait_for(lambda: killed, 2)
    assert killed == [(os.getpid(), signal.SIGTERM)]


def test_the_core_stops_serving_when_the_app_closes_its_input():
    lines = iter(['{"jsonrpc": "2.0", "id": 1, "method": "app.info"}\n', ""])
    written = []
    done = threading.Event()

    def serve():
        rpc_server.serve(lambda: next(lines), {"app.info": lambda params: {"ok": True}}, written.append)
        done.set()

    threading.Thread(target=serve, daemon=True).start()
    assert done.wait(5)
    assert written == [{"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}]

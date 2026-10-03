"""Unit tests for AssistantController lifecycle using a fake assistant."""

import threading
import time

from iris.controller import AssistantController
from iris.events import EventBus


class FakePipeline:
    @property
    def is_active(self):
        return False

    def stop(self):
        pass


class FakeConversation:
    model = "fake-llm"

    def set_model(self, name):
        self.model = name


class FakeAssistant:
    """Stands in for VoiceAssistant without loading any models."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.conversation = FakeConversation()
        self.speech_pipeline = FakePipeline()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.interrupt_event = threading.Event()
        self.conversation_language = "en"

    def run_loop(self):
        while not self.stop_event.is_set():
            self.stop_event.wait(0.05)

    def config_info(self):
        return {
            "persona": "Iris",
            "llm_model": "fake-llm",
            "whisper_model": "fake-whisper",
            "whisper_device": "cpu",
            "tts_voice": "af_fake",
            "input_device": "Fake Microphone",
        }


def _wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_tool_activity_outranks_thinking():
    """"thinking" stays set while a tool runs, so it must not mask the tool."""
    controller = AssistantController(EventBus())
    controller._flags = {"thinking": True, "looking": True}
    assert controller._flag_state() == "looking"


def test_start_pause_resume_stop(monkeypatch):
    monkeypatch.setattr("iris.controller.VoiceAssistant", FakeAssistant)

    bus = EventBus()
    states = []
    bus.subscribe(lambda event_type, data: states.append(data.get("state")) if event_type == "state" else None)

    controller = AssistantController(bus)
    try:
        assert controller.start({})["ok"] is True
        assert _wait_for(lambda: controller.assistant is not None)
        assert _wait_for(lambda: controller.info()["running"])

        # Pause → paused state is published
        assert controller.pause()["ok"] is True
        assert _wait_for(lambda: "paused" in states)
        assert controller.assistant.pause_event.is_set()

        # Resume → back to a non-paused state
        assert controller.resume()["ok"] is True
        assert not controller.assistant.pause_event.is_set()

        # Starting again while running is a no-op
        assert controller.start({})["already_running"] is True

        # Model switching goes through to the conversation manager
        assert controller.set_model("other-llm")["ok"] is True
        assert controller.assistant.conversation.model == "other-llm"
    finally:
        controller.stop()
    assert _wait_for(lambda: not controller.info()["running"])

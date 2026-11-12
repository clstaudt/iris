"""What Iris says when the system keeps it from a device, or a file it
starts with is broken: a microphone that gives only silence (macOS denies
the app silently), a persona file that cannot be read, and a persona whose
voice engine is not there. Iris says so, in words that fit the system it
runs on, and goes on."""

import sys
from types import SimpleNamespace

import pytest

from iris import controller as controller_module
from iris import personas
from iris.assistant import VoiceAssistant
from iris.controller import SILENT, SILENT_MAC, AssistantController
from iris.events import EventBus
from iris.personas import Persona
from iris.tts import SpeechGenerator


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(controller_module, "time", SimpleNamespace(monotonic=clock.monotonic))
    return clock


def _errors(bus_events) -> list[str]:
    return [d["message"] for k, d in bus_events if k == "error"]


def _controller():
    bus = EventBus()
    events = []
    bus.subscribe(lambda kind, data: events.append((kind, data)))
    return AssistantController(bus), events


@pytest.mark.parametrize("platform, message", [("darwin", SILENT_MAC), ("linux", SILENT), ("win32", SILENT)])
def test_a_microphone_that_gives_only_silence_is_reported_once(clock, monkeypatch, platform, message):
    monkeypatch.setattr(sys, "platform", platform)
    controller, events = _controller()
    for _ in range(3):
        controller.bus.emit("audio.level", {"rms": 0.0})
        clock.now += 1.0
    assert _errors(events) == []  # three seconds: not yet
    for _ in range(5):
        controller.bus.emit("audio.level", {"rms": 0.0})
        clock.now += 1.0
    assert _errors(events) == [message]
    if platform != "darwin":
        assert "macOS" not in message and "System Settings" not in message


def test_a_microphone_that_hears_something_is_not_silent(clock):
    controller, events = _controller()
    for n in range(20):
        # Quiet, but a real microphone's noise floor
        controller.bus.emit("audio.level", {"rms": 0.0 if n % 3 else 0.001})
        clock.now += 1.0
    assert _errors(events) == []


def test_silence_after_sound_again_is_reported_again(clock):
    controller, events = _controller()
    for level in [0.0] * 6 + [0.01] + [0.0] * 6:
        controller.bus.emit("audio.level", {"rms": level})
        clock.now += 1.0
    assert len(_errors(events)) == 2


def test_a_persona_file_that_cannot_be_read_is_left_out(tmp_path, monkeypatch):
    monkeypatch.setattr(personas, "USER_DIR", tmp_path)
    (tmp_path / "broken.yaml").write_text("name: [never closed\n")
    (tmp_path / "half.yaml").write_text("name: Half\n")  # no description, no voice
    (tmp_path / "empty.yaml").write_text("")
    names = personas.reload_personas()
    assert "iris" in names
    assert not {"broken", "half", "empty"} & set(names)
    assert personas.get_persona("broken").id == "iris"


def test_a_persona_whose_engine_is_not_there_is_answered_in_text(monkeypatch):
    """A persona file naming an engine no plugin brings any more: its voice
    cannot be loaded, which Iris says, and it replies in text."""
    ghost = Persona(name="Ghost", description="You are Ghost.", voice="whisper", engine="no-such-engine")
    assistant = object.__new__(VoiceAssistant)
    assistant.event_bus = EventBus()
    events = []
    assistant.event_bus.subscribe(lambda kind, data: events.append((kind, data)))
    assistant.speak_replies = True
    assistant.tts_generator = SpeechGenerator(ghost.engine, ghost.voice, prepare=False)
    assert assistant.load_voice() is False
    assert assistant.speak_replies is False
    [message] = _errors(events)
    assert message.startswith("The voice could not be loaded, so replies are shown as text.")

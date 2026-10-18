"""A user who goes on with a turn Smart Turn took for finished takes it back:
the answer begun is dropped and the whole turn answered."""

import threading
import time

from iris.assistant import VoiceAssistant
from iris.events import EventBus


class Flags:
    status_listening = False

    def __init__(self):
        self.status_lock = threading.Lock()


class Pipeline:
    def __init__(self):
        self.is_speaking_flag = threading.Event()
        self.stops = 0

    def stop(self):
        self.stops += 1


class Recorder:
    """Listens on: the user goes on after the first turn only."""

    def __init__(self, whole_turn):
        self.whole_turn = whole_turn
        self.calls = []

    def listen_on(self, turn, on_speech, until, should_stop=None, on_level=None):
        self.calls.append(turn)
        if len(self.calls) == 1:
            on_speech()
            return self.whole_turn
        while not until():
            time.sleep(0.001)
        return None


def _assistant(tmp_path, announce=True):
    assistant = object.__new__(VoiceAssistant)
    assistant.interrupt_event = threading.Event()
    for name in ("pause_event", "suspend_event", "hold_event", "stop_event"):
        setattr(assistant, name, threading.Event())
    assistant.status_manager = Flags()
    assistant.speech_pipeline = Pipeline()
    assistant.event_bus = EventBus()
    assistant._last_voice = 0.0
    first, whole = tmp_path / "first.wav", tmp_path / "whole.wav"
    first.write_bytes(b"")
    whole.write_bytes(b"")
    assistant.recorder = Recorder(str(whole))
    answered = []

    def handle_heard(path, dormant):
        assistant._announced = announce
        answered.append((path, dormant, assistant.interrupt_event.is_set()))

    assistant._handle_heard = handle_heard
    return assistant, str(first), str(whole), answered


def test_the_whole_turn_is_answered(tmp_path):
    assistant, first, whole, answered = _assistant(tmp_path)
    events = []
    assistant.event_bus.subscribe(lambda kind, data: events.append(kind))

    assistant._answer_listening(first, dormant=False)

    # The first answer was cancelled as soon as the user went on, the whole
    # turn answered afresh
    assert answered == [(first, False, True), (whole, False, False)]
    assert assistant.speech_pipeline.stops == 1
    assert events.count("chat.continued") == 1
    assert not assistant.interrupt_event.is_set()


def test_a_turn_not_shown_yet_needs_no_taking_back(tmp_path):
    assistant, first, _, _ = _assistant(tmp_path, announce=False)
    events = []
    assistant.event_bus.subscribe(lambda kind, data: events.append(kind))
    assistant._answer_listening(first, dormant=False)
    assert "chat.continued" not in events

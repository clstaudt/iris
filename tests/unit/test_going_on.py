"""While a turn is answered the microphone listens on. A user who goes on
with a turn Smart Turn took for finished takes it back: the answer begun is
dropped and the whole turn answered. A user who speaks over the voice (with
echo cancellation) interrupts it: what they say is the next turn."""

import threading
import time

from iris.assistant import VoiceAssistant
from iris.events import EventBus


class Flags:
    status_listening = False

    def __init__(self):
        self.status_lock = threading.Lock()


class Pipeline:
    def __init__(self, said=""):
        self.is_speaking_flag = threading.Event()
        self.stops = 0
        self._said = said

    def stop(self):
        self.stops += 1

    def pause(self):
        pass

    def resume(self):
        pass

    @property
    def is_active(self):
        return self.is_speaking_flag.is_set()

    def said(self):
        return self._said


class Recorder:
    """Listens on: the user speaks after the first turn only."""

    def __init__(self, whole_turn, new_turn):
        self.whole_turn, self.new_turn = whole_turn, new_turn
        self.calls = []

    def listen_on(self, turn, on_speech, until, should_stop=None, on_level=None, **kw):
        self.calls.append(kw.get("speaking") is not None)
        if len(self.calls) == 1:
            return self.whole_turn if on_speech() else self.new_turn
        while not until():
            time.sleep(0.001)
        return None


def _assistant(tmp_path, announce=True, said="", barge_in=False):
    assistant = object.__new__(VoiceAssistant)
    assistant.interrupt_event = threading.Event()
    for name in ("pause_event", "suspend_event", "hold_event", "stop_event"):
        setattr(assistant, name, threading.Event())
    assistant.status_manager = Flags()
    assistant.speech_pipeline = Pipeline(said)
    assistant.event_bus = EventBus()
    assistant._last_voice = 0.0
    assistant.barge_in = barge_in
    assistant.echo = object() if barge_in else None
    paths = {}
    for name in ("first", "whole", "new"):
        paths[name] = tmp_path / f"{name}.wav"
        paths[name].write_bytes(b"")
    assistant.recorder = Recorder(str(paths["whole"]), str(paths["new"]))
    answered = []

    def handle_heard(path, dormant):
        assistant._announced = announce
        answered.append((path, assistant.interrupt_event.is_set()))

    assistant._handle_heard = handle_heard
    return assistant, {k: str(v) for k, v in paths.items()}, answered


def _events(assistant):
    events = []
    assistant.event_bus.subscribe(lambda kind, data: events.append(kind))
    return events


def test_the_whole_turn_is_answered(tmp_path):
    assistant, paths, answered = _assistant(tmp_path)
    events = _events(assistant)

    assistant._answer_listening(paths["first"], dormant=False)

    # The first answer was cancelled as soon as the user went on, the whole
    # turn answered afresh
    assert answered == [(paths["first"], True), (paths["whole"], False)]
    assert assistant.speech_pipeline.stops == 1
    assert events.count("chat.continued") == 1
    assert not assistant.interrupt_event.is_set()


def test_a_turn_not_shown_yet_needs_no_taking_back(tmp_path):
    assistant, paths, _ = _assistant(tmp_path, announce=False)
    events = _events(assistant)
    assistant._answer_listening(paths["first"], dormant=False)
    assert "chat.continued" not in events


def test_without_echo_cancellation_listening_ends_with_the_voice(tmp_path):
    assistant, paths, _ = _assistant(tmp_path)
    assistant._answer_listening(paths["first"], dormant=False)
    assert assistant.recorder.calls == [False, False]


def test_speaking_over_the_voice_is_the_next_turn(tmp_path):
    assistant, paths, answered = _assistant(tmp_path, said="Lisbon in May is", barge_in=True)
    events = _events(assistant)

    assistant._answer_listening(paths["first"], dormant=False)

    # The voice was stopped; what the user said is a turn of its own, and the
    # answer it interrupted stays in the conversation as far as it was heard
    assert assistant.recorder.calls == [True, True]
    assert answered == [(paths["first"], True), (paths["new"], False)]
    assert assistant.speech_pipeline.stops == 1
    assert "chat.continued" not in events


def test_barge_in_switched_off(tmp_path):
    assistant, paths, _ = _assistant(tmp_path, barge_in=True)
    assistant.barge_in = False
    assistant._answer_listening(paths["first"], dormant=False)
    assert assistant.recorder.calls == [False, False]


"""Unit tests for wake word matching, detection and the dormant/awake flow."""

import threading
import time

import numpy as np
import pytest

from iris import assistant as assistant_module
from iris import wakeword
from iris.assistant import DEFAULT_WAKE_WINDOW, VoiceAssistant
from iris.events import EventBus
from iris.wakeword import WakeWordDetector, match_sleep_word, match_wake_word


@pytest.mark.parametrize(
    "heard, woken, remainder",
    [
        ("Iris, what time is it?", True, "what time is it"),
        ("Hey Iris.", True, ""),
        ("Ciao Iris, come stai?", True, "come stai"),
        ("iris what's the weather", True, "what's the weather"),
        ("Irus, are you there?", True, "are you there"),  # tiny mishearing the name
        ("The irises in the garden bloomed.", False, ""),
        ("Paris is lovely.", False, ""),
        ("I read the news today.", False, ""),
        ("Tell Iris I said hi.", False, ""),  # mentioned, not addressed
        ("", False, ""),
    ],
)
def test_only_an_utterance_that_opens_with_the_name_wakes(heard, woken, remainder):
    assert match_wake_word(heard, "Iris") == (woken, remainder)


def test_multi_word_wake_phrases():
    assert match_wake_word("Hey computer, lights on", "computer") == (True, "lights on")
    assert match_wake_word("my computer is slow", "computer") == (False, "")
    assert match_wake_word("Okay Jarvis, status", "okay jarvis") == (True, "status")


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeTiny:
    def __init__(self, *args, **kwargs):
        self.prompts = []

    def transcribe(self, audio, beam_size, initial_prompt, vad_filter):
        self.prompts.append(initial_prompt)
        return [FakeSegment(" Iris,"), FakeSegment(" what time is it?")], None


@pytest.fixture
def cpu_detector(monkeypatch):
    monkeypatch.setattr(wakeword, "WhisperModel", FakeTiny)
    monkeypatch.setattr(
        wakeword, "collect_chunks", lambda audio, chunks: ([audio[:100]], [])
    )
    return WakeWordDetector("faster-whisper")


def test_detector_prompts_tiny_with_the_wake_word(cpu_detector, monkeypatch):
    monkeypatch.setattr(wakeword, "get_speech_timestamps", lambda audio: [{"start": 0, "end": 100}])
    assert cpu_detector.heard(np.zeros(16000), "Iris") == (True, "what time is it")
    assert cpu_detector._model.prompts == ["Iris"]


def test_detector_skips_whisper_without_speech(cpu_detector, monkeypatch):
    monkeypatch.setattr(wakeword, "get_speech_timestamps", lambda audio: [])
    assert cpu_detector.heard(np.zeros(16000), "Iris") == (False, "")
    assert cpu_detector._model.prompts == []


class FakeDetector:
    def __init__(self, result):
        self.result = result

    def heard(self, audio, word):
        return self.result


def _bare_assistant(monkeypatch, detector_result):
    a = object.__new__(VoiceAssistant)
    a.wake_word = "Iris"
    a.wake_detector = FakeDetector(detector_result)
    a.awake_until = 0.0
    a.wake_window = DEFAULT_WAKE_WINDOW
    a.sleep_word = "That's all"
    a._was_dormant = True
    a._last_window_emit = 0.0
    a.pause_event = threading.Event()
    a.suspend_event = threading.Event()
    a.event_bus = EventBus()
    a.speak_replies = True
    a.turns = []
    a.chimes = 0
    a._run_voice_turn = lambda path: a.turns.append(path)
    a._chime = lambda: setattr(a, "chimes", a.chimes + 1)
    monkeypatch.setattr(assistant_module, "decode_audio", lambda path, sampling_rate: np.zeros(10))
    return a


def test_dormant_assistant_ignores_what_is_not_addressed_to_it(monkeypatch):
    a = _bare_assistant(monkeypatch, (False, ""))
    assert a.dormant
    a._handle_heard("/clip.wav", dormant=True)
    assert a.turns == [] and a.dormant


def test_name_with_a_request_wakes_and_answers_at_once(monkeypatch):
    a = _bare_assistant(monkeypatch, (True, "what time is it"))
    woke = []
    a.event_bus.subscribe(lambda t, d: woke.append(d) if t == "wake" else None)
    a._handle_heard("/clip.wav", dormant=True)
    assert a.turns == ["/clip.wav"] and a.chimes == 0
    assert woke == [{"word": "Iris"}]
    assert not a.dormant
    assert a.awake_until == pytest.approx(time.monotonic() + DEFAULT_WAKE_WINDOW, abs=1)


def test_bare_name_chimes_and_waits_for_the_request(monkeypatch):
    a = _bare_assistant(monkeypatch, (True, ""))
    a._handle_heard("/clip.wav", dormant=True)
    assert a.turns == [] and a.chimes == 1 and not a.dormant


def test_awake_assistant_answers_without_the_name(monkeypatch):
    a = _bare_assistant(monkeypatch, (False, ""))
    a._stay_awake()
    a._handle_heard("/clip.wav", dormant=False)
    assert a.turns == ["/clip.wav"]


def test_falls_dormant_when_the_window_passes(monkeypatch):
    a = _bare_assistant(monkeypatch, (False, ""))
    a.awake_until = time.monotonic() - 0.1
    assert a.dormant
    a.wake_word = None  # without a wake word it is never dormant
    assert not a.dormant



# -- going back to sleep -----------------------------------------------------


@pytest.mark.parametrize(
    "heard, asleep",
    [
        ("That's all, Iris.", True),
        ("Iris, that's all.", True),
        ("Okay, that's all, thanks.", True),
        ("thats all", True),
        ("That's all I wanted to know about Lisbon.", False),
        ("Thanks, Iris.", False),
        ("That's awful, Iris.", False),
    ],
)
def test_only_the_sleep_word_itself_puts_iris_to_sleep(heard, asleep):
    assert match_sleep_word(heard, "That's all", "Iris") is asleep


class Transcribes:
    def __init__(self, text):
        self.text = text

    def transcribe(self, path):
        return self.text, "en"


class QuietStatus:
    status_lock = threading.Lock()
    status_understanding = False

    def update_status_bar(self, *args):
        pass


class QuietDisplay:
    def show_user_message(self, text):
        pass


def _awake_assistant(monkeypatch, heard):
    """Runs the real path from a recorded utterance to the reply."""
    a = _bare_assistant(monkeypatch, (False, ""))
    del a._run_voice_turn  # use the real one
    a._turn_active = threading.Event()
    a.transcriber = Transcribes(heard)
    a.status_manager = QuietStatus()
    a.display_manager = QuietDisplay()
    a.speech_pipeline = None
    a.conversation_language = "en"
    a.replies = []
    a.chat = lambda text, response_language, source=None: a.replies.append(text)
    a._stay_awake()
    return a


def test_the_sleep_word_sends_iris_to_sleep_without_a_reply(monkeypatch):
    a = _awake_assistant(monkeypatch, "That's all, Iris.")
    a._handle_heard("/clip.wav", dormant=False)
    assert a.replies == []
    assert a.dormant  # and the finished turn does not wake it again


def test_anything_else_is_answered_and_keeps_iris_awake(monkeypatch):
    a = _awake_assistant(monkeypatch, "That's all I wanted to know about Lisbon.")
    a._handle_heard("/clip.wav", dormant=False)
    assert a.replies == ["That's all I wanted to know about Lisbon."]
    assert not a.dormant


def test_the_window_is_adjustable_and_announced(monkeypatch):
    a = _bare_assistant(monkeypatch, (False, ""))
    windows = []
    def record(event_type, data):
        if event_type == "wake.window":
            windows.append(data["remaining"])

    a.event_bus.subscribe(record)
    a.wake_window = 60.0
    a._stay_awake()
    assert windows == [60.0]
    assert a.awake_until == pytest.approx(time.monotonic() + 60.0, abs=1)


def test_speaking_keeps_iris_awake(monkeypatch):
    a = _bare_assistant(monkeypatch, (False, ""))
    a.awake_until = time.monotonic() + 1  # about to fall asleep
    a._keep_awake_while_speaking()
    assert a.awake_until == pytest.approx(time.monotonic() + DEFAULT_WAKE_WINDOW, abs=1)


def test_falling_asleep_chimes_once(monkeypatch):
    a = _bare_assistant(monkeypatch, (False, ""))
    played, events = [], []
    monkeypatch.setattr(assistant_module.sd, "play", lambda data, rate: played.append(len(data)))
    a.event_bus.subscribe(lambda t, d: events.append(t))

    a._note_dormancy(False)  # awake
    a._note_dormancy(True)  # the window ran out
    a._note_dormancy(True)  # still asleep: no second chime
    assert len(played) == 1
    assert events.count("sleep") == 1

    a.speak_replies = False  # Voice off: asleep without a sound
    a._note_dormancy(False)
    a._note_dormancy(True)
    assert len(played) == 1

    a.speak_replies = True
    a.pause_event.set()  # mic off: it was not listening, so no chime either
    a._note_dormancy(False)
    a._note_dormancy(True)
    assert len(played) == 1
    assert events.count("sleep") == 3  # but the UI always learns it

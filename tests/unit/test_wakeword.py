"""Unit tests for wake word matching, detection and the dormant/awake flow."""

import time

import numpy as np
import pytest

from iris import assistant as assistant_module
from iris import wakeword
from iris.assistant import WAKE_WINDOW, VoiceAssistant
from iris.events import EventBus
from iris.wakeword import WakeWordDetector, match_wake_word


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
    assert a.awake_until == pytest.approx(time.monotonic() + WAKE_WINDOW, abs=1)


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

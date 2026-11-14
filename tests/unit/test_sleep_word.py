"""Unit tests for going to sleep: the sleep word, which always puts Iris to
sleep, with the wake word on or off, and sleeping when idle, which can be
switched off."""

import math
import time

import pytest

from iris import assistant as assistant_module
from iris.assistant import _window
from iris.personas import Persona
from iris.wakeword import match_sleep_word

from tests.unit.test_wakeword import _awake_assistant


class Hears:
    """A wake word detector that hears what it is told, and notes the word
    it listened for."""

    def __init__(self, result=(False, "")):
        self.result = result
        self.words = []

    def heard(self, audio, word):
        self.words.append(word)
        return self.result


def _iris(monkeypatch, heard, wake_on=True, sleep_word="That's all", persona=None):
    """An awake Iris that hears this next, as the real path takes it."""
    a = _awake_assistant(monkeypatch, heard)
    a.persona = persona or Persona("Iris", "Warm.", voice="alba", sleep_word=sleep_word)
    a.sleep_word = sleep_word or None
    a.wake_detector = Hears()
    if not wake_on:
        a.wake_word = None
    return a


# The sleep word ---------------------------------------------------------------


def test_with_the_wake_word_on_the_sleep_word_puts_iris_to_sleep(monkeypatch):
    a = _iris(monkeypatch, "That's all, Iris.")
    a._handle_heard("/clip.wav", dormant=False)
    assert a.replies == [] and a.dormant


def test_mid_conversation_the_sleep_word_is_not_given_to_the_model(monkeypatch):
    a = _iris(monkeypatch, "What is the capital of Italy?")
    a._handle_heard("/clip.wav", dormant=False)
    a.transcriber.text = "Okay, that's all, thanks."
    a._handle_heard("/clip.wav", dormant=False)
    assert a.replies == ["What is the capital of Italy?"]
    assert a.dormant


def test_with_the_wake_word_off_the_sleep_word_puts_iris_to_sleep_too(monkeypatch):
    a = _iris(monkeypatch, "That's all, Iris.", wake_on=False)
    a._handle_heard("/clip.wav", dormant=False)
    assert a.replies == [] and a.dormant


def test_asleep_by_the_sleep_word_it_wakes_by_the_persona_s_name(monkeypatch):
    a = _iris(monkeypatch, "That's all.", wake_on=False)
    a._handle_heard("/clip.wav", dormant=False)
    woke = []
    a.event_bus.subscribe(lambda t, d: woke.append(d) if t == "wake" else None)

    # Anything else is not for it
    a._handle_heard("/clip.wav", dormant=True)
    assert a.replies == [] and a.dormant

    a.wake_detector.result = (True, "what is the capital of Spain")
    a.transcriber.text = "Iris, what is the capital of Spain?"
    a._handle_heard("/clip.wav", dormant=True)
    assert a.wake_detector.words == ["Iris", "Iris"]
    assert woke == [{"word": "Iris"}]
    assert a.replies == ["Iris, what is the capital of Spain?"]
    # And then, with the wake word off, it listens to everything again
    a.awake_until = time.monotonic() - 1
    assert not a.dormant


def test_a_persona_with_a_wake_word_of_its_own_wakes_by_that(monkeypatch):
    hal = Persona("HAL", "Calm.", voice="bm_lewis", wake_word="Hey HAL", sleep_word="Goodbye")
    a = _iris(monkeypatch, "Goodbye, HAL.", wake_on=False, sleep_word="Goodbye", persona=hal)
    a._handle_heard("/clip.wav", dormant=False)
    assert a.dormant
    a._handle_heard("/clip.wav", dormant=True)
    assert a.wake_detector.words == ["Hey HAL"]


def test_a_typed_message_wakes_iris_asleep_by_the_sleep_word(monkeypatch):
    a = _iris(monkeypatch, "That's all.", wake_on=False)
    a._handle_heard("/clip.wav", dormant=False)
    a._wake()
    assert not a.dormant


def test_switching_the_wake_word_off_wakes_iris_asleep_by_the_sleep_word(monkeypatch):
    a = _iris(monkeypatch, "That's all.", wake_on=False)
    a._handle_heard("/clip.wav", dormant=False)
    a.set_wake_word("")
    assert not a.dormant


def test_without_a_way_to_wake_it_iris_stays_awake(monkeypatch):
    a = _iris(monkeypatch, "That's all.", wake_on=False)
    a.wake_detector = None

    def fails(engine):
        raise RuntimeError("no model")

    monkeypatch.setattr(assistant_module, "WakeWordDetector", fails)
    errors = []
    a.event_bus.subscribe(lambda t, d: errors.append(d["message"]) if t == "error" else None)
    a._handle_heard("/clip.wav", dormant=False)
    assert not a.dormant and a.replies == []
    assert errors and "cannot sleep" in errors[0]


def test_without_a_sleep_word_the_words_are_answered(monkeypatch):
    a = _iris(monkeypatch, "That's all.", wake_on=False, sleep_word="")
    a._handle_heard("/clip.wav", dormant=False)
    assert a.replies == ["That's all."] and not a.dormant


@pytest.mark.parametrize(
    "heard",
    [
        "That's all I wanted to know about Lisbon.",  # at the start, then a request
        "I told him that's all.",  # inside
        "Is that all, Iris?",
    ],
)
def test_the_sleep_word_counts_only_said_on_its_own(monkeypatch, heard):
    a = _iris(monkeypatch, heard, wake_on=False)
    a._handle_heard("/clip.wav", dormant=False)
    assert a.replies == [heard] and not a.dormant


@pytest.mark.parametrize(
    "heard, sleep_word, wake_word, asleep",
    [
        ("Gute Nacht, Klaus.", "Gute Nacht", "Klaus", True),
        ("Klaus, gute Nacht!", "Gute Nacht", "Klaus", True),
        ("Gute Nacht und bis morgen, Klaus.", "Gute Nacht", "Klaus", False),
        ("Bonne nuit, Iris.", "Bonne nuit", "Iris", True),
        ("おやすみ。", "おやすみ", "Iris", True),
        ("おやすみなさい、また明日。", "おやすみ", "Iris", False),
    ],
)
def test_a_sleep_word_in_another_language(heard, sleep_word, wake_word, asleep):
    assert match_sleep_word(heard, sleep_word, wake_word) is asleep


def test_a_sleep_word_in_another_language_puts_iris_to_sleep(monkeypatch):
    klaus = Persona("Klaus", "Dry.", voice="juergen", sleep_word="Gute Nacht")
    a = _iris(monkeypatch, "Gute Nacht, Klaus.", sleep_word="Gute Nacht", persona=klaus)
    a.wake_word = None
    a.transcriber.transcribe = lambda path: ("Gute Nacht, Klaus.", "de")
    a._handle_heard("/clip.wav", dormant=False)
    assert a.replies == [] and a.dormant


# Sleeping when idle -------------------------------------------------------------


def test_the_window_is_none_or_at_least_the_shortest():
    assert _window(0) == 0.0
    assert _window(None) == 0.0
    assert _window(2) == 5.0
    assert _window(60) == 60.0


def test_without_a_window_iris_stays_awake_until_told(monkeypatch):
    a = _iris(monkeypatch, "What is the capital of Italy?")
    windows = []
    a.event_bus.subscribe(lambda t, d: windows.append(d) if t == "wake.window" else None)
    a.set_wake_options(window=0)
    a._handle_heard("/clip.wav", dormant=False)
    a._keep_awake_while_speaking()
    assert a.awake_until == math.inf and not a.dormant
    assert windows == []  # nothing to count down
    # The sleep word still puts it to sleep
    a.transcriber.text = "That's all."
    a._handle_heard("/clip.wav", dormant=False)
    assert a.dormant


def test_switching_the_window_on_while_awake_starts_it(monkeypatch):
    a = _iris(monkeypatch, "Hello.")
    a.set_wake_options(window=0)
    a.set_wake_options(window=60)
    assert a.awake_until == pytest.approx(time.monotonic() + 60, abs=1)


def test_asleep_a_new_window_does_not_wake_iris(monkeypatch):
    a = _iris(monkeypatch, "Hello.")
    a.awake_until = 0.0
    a.set_wake_options(window=0)
    assert a.dormant


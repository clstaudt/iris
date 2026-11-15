"""Unit tests for the soft drone Iris sounds while a reply is long in
coming: none for a reply that comes quickly, fading in after a few seconds
of silence and held, never over the voice, ended on the speaker once the
reply is stopped, paused or Iris goes to sleep, and never taken by the
microphone for the user.

Silent: the clock is the test's, moved on by it, and the speaker a stand-in
that notes what it is given; the model is scripted.
"""

import functools
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from iris import assistant as assistant_module
from iris import audio as audio_module
from iris.assistant import (
    DRONE_FADE,
    DRONE_LEVEL,
    DRONE_RENEW,
    DRONE_SECONDS,
    THINKING_DRONE_AFTER,
    ThinkingDrone,
    VoiceAssistant,
    _sleep_chime,
    _thinking_drone,
    _thread_chime,
    _wake_chime,
)
from iris.audio import AudioRecorder
from iris.controller import AssistantController
from iris.echo import HAS_ECHO_CANCELLATION, EchoCanceller
from iris.events import EventBus
from iris.llm import ConversationManager
from iris.plugin import PluginContext
from iris.tools import Tools
from iris.tts import SpeechPipeline
from iris.turns import FRAME, SAMPLE_RATE
from irisdev.voicetest.acoustics import in_room, room, to_rate
from tests.unit.test_sleep_tool import HeldPlayer, InstantGenerator, Status, _wait
from tests.unit.test_turn_taking import FakeStream, Verdict, _turn_file

RATE = 24000


class Clock:
    """The test's clock: it moves on only when told, and counts how often it
    was read, so a test can wait for the drone to have looked at the time."""

    def __init__(self):
        self.now = 100.0
        self.reads = 0
        self._read = threading.Condition()

    def __call__(self) -> float:
        with self._read:
            self.reads += 1
            self._read.notify_all()
            return self.now

    def advance(self, seconds: float):
        self.now += seconds

    def looked(self, times: int = 2) -> bool:
        """Wait until the clock was read this many more times: the drone
        looked at the wait at least once since it was last moved on."""
        with self._read:
            target = self.reads + times
            return self._read.wait_for(lambda: self.reads >= target, timeout=5.0)


def _run(drone: ThinkingDrone, clock: Clock, seconds: float, step: float = 0.25):
    """Let so much time pass, the drone looking at it every step."""
    for _ in range(round(seconds / step)):
        clock.advance(step)
        drone.tick()


def _drone(heard=lambda: False, may_sound=lambda: True):
    """A drone on the test's clock: (drone, clock, when clips started, how
    often it was hushed)."""
    clock = Clock()
    played, hushed = [], []
    drone = ThinkingDrone(
        lambda: played.append(clock.now),
        lambda: hushed.append(clock.now),
        heard,
        may_sound,
        clock=clock,
    )
    return drone, clock, played, hushed


# -- when it sounds ------------------------------------------------------------------


def test_the_drone_fades_in_after_a_few_seconds_of_silence_and_holds():
    drone, clock, played, hushed = _drone()
    start = clock.now
    _run(drone, clock, THINKING_DRONE_AFTER - 0.25)
    assert played == []  # nothing for a reply that comes within it
    _run(drone, clock, 30.0)
    assert played == [start + THINKING_DRONE_AFTER] and hushed == []


def test_a_wait_longer_than_a_clip_has_the_next_take_over_as_it_fades_out():
    drone, clock, played, hushed = _drone()
    start = clock.now + THINKING_DRONE_AFTER
    _run(drone, clock, THINKING_DRONE_AFTER + 2 * DRONE_RENEW + 1.0)
    assert played == [start, start + DRONE_RENEW, start + 2 * DRONE_RENEW]
    assert hushed == []


def test_the_voice_ends_the_drone_and_the_silence_counts_from_its_end():
    speaking = [False]
    drone, clock, played, hushed = _drone(heard=lambda: speaking[0])
    _run(drone, clock, THINKING_DRONE_AFTER)
    assert len(played) == 1
    speaking[0] = True  # "Let me look that up."
    _run(drone, clock, 20.0)
    # The voice took the speaker: nothing here stops it, nor plays over it
    assert len(played) == 1 and hushed == []
    speaking[0] = False  # a tool runs
    ended = clock.now
    _run(drone, clock, THINKING_DRONE_AFTER)
    assert played[1:] == [ended + THINKING_DRONE_AFTER]


def test_it_ends_on_the_speaker_once_it_may_not_sound_and_comes_back_after():
    allowed = [True]
    drone, clock, played, hushed = _drone(may_sound=lambda: allowed[0])
    _run(drone, clock, THINKING_DRONE_AFTER)
    allowed[0] = False  # paused, or going to sleep
    _run(drone, clock, 0.25)
    assert len(played) == 1 and len(hushed) == 1
    _run(drone, clock, 20.0)
    assert len(played) == 1 and len(hushed) == 1
    allowed[0] = True
    _run(drone, clock, THINKING_DRONE_AFTER)
    assert len(played) == 2


def test_stopped_it_is_silent_and_starts_no_clip():
    drone, clock, played, hushed = _drone()
    drone.stop()  # before it sounded: nothing to stop
    assert hushed == []
    drone, clock, played, hushed = _drone()
    _run(drone, clock, THINKING_DRONE_AFTER)
    drone.stop()
    assert len(hushed) == 1
    _run(drone, clock, 30.0)
    assert len(played) == 1 and len(hushed) == 1


def test_stopped_while_the_voice_has_the_speaker_it_leaves_the_voice_alone():
    speaking = [False]
    drone, clock, played, hushed = _drone(heard=lambda: speaking[0])
    _run(drone, clock, THINKING_DRONE_AFTER)
    speaking[0] = True
    drone.stop()  # the thinking is over, the voice speaks on
    assert hushed == []
    speaking[0] = False
    drone.stop()  # once it has spoken: a clip it never took over from ends
    assert len(hushed) == 1


# -- the sound ------------------------------------------------------------------------


def _loudest(sound: np.ndarray) -> float:
    """The RMS of its loudest 50 ms."""
    window = RATE // 20
    return max(
        float(np.sqrt(np.mean(sound[i : i + window] ** 2)))
        for i in range(0, len(sound) - window + 1, window)
    )


def test_the_drone_is_faint_soft_and_ends_low_wherever_it_is_cut():
    drone = _thinking_drone()
    assert len(drone) == round(DRONE_SECONDS * RATE)
    assert np.abs(drone).max() <= DRONE_LEVEL + 1e-6
    for chime in (_wake_chime(), _sleep_chime(), _thread_chime()):
        assert np.abs(drone).max() <= np.abs(chime).max() / 10
        assert _loudest(drone) <= _loudest(chime) / 8
    # No attack: it swells in over seconds, from silence
    assert drone[0] == 0.0
    assert np.abs(drone[: RATE // 4]).max() < 0.05 * DRONE_LEVEL
    assert np.abs(drone[: int(DRONE_FADE * RATE) // 2]).max() < 0.6 * DRONE_LEVEL
    assert np.abs(np.diff(drone)).max() < 0.004
    # Where the next clip takes over, it has nearly faded out, even when it
    # is heard 0.3 s later than played
    late = int((DRONE_RENEW - 0.3) * RATE)
    assert np.abs(drone[late:]).max() < 0.25 * DRONE_LEVEL
    # Warm: next to nothing of it above 1.2 kHz
    power = np.abs(np.fft.rfft(drone)) ** 2
    frequencies = np.fft.rfftfreq(len(drone), 1 / RATE)
    assert power[frequencies > 1200].sum() < 1e-4 * power.sum()


CANCELLERS = [
    pytest.param(False, id="alone"),
    pytest.param(
        True,
        id="through the echo canceller",
        marks=pytest.mark.skipif(not HAS_ECHO_CANCELLATION, reason="no echo canceller here"),
    ),
]


@pytest.mark.parametrize("canceller", CANCELLERS)
def test_the_microphone_does_not_take_the_drone_for_the_user(monkeypatch, canceller):
    """While the model thinks, the microphone listens on for the user going
    on (AudioRecorder.listen_on), with Silero VAD as Iris runs it: twelve
    seconds of the drone, heard through the room as loud as it is played,
    quieter and louder, start no turn and interrupt nothing, with or
    without the echo canceller (told of nothing, as it is not of the
    drone)."""
    drone = to_rate(_thinking_drone()[: 12 * RATE], RATE)
    pause = np.zeros(SAMPLE_RATE, dtype=np.float32)
    heard = [pause]
    for gain in (0.3, 1.0, 3.0):
        heard += [in_room(drone * gain, room(0.4, 9.0)), pause]
    recording = np.concatenate(heard)
    noise = np.random.default_rng(7).standard_normal(len(recording)).astype(np.float32)
    monkeypatch.setattr(FakeStream, "recording", recording + 0.0005 * noise)
    monkeypatch.setattr(audio_module.sd, "InputStream", FakeStream)
    recorder = AudioRecorder(turn_end=Verdict(0.9), echo=EchoCanceller() if canceller else None)
    looks = [0]

    def all_heard() -> bool:
        looks[0] += 1
        return looks[0] > len(recording) // FRAME + 5

    spoke = []
    earlier = _turn_file()
    path = recorder.listen_on(
        earlier,
        on_speech=lambda: spoke.append(True) or True,
        until=lambda: False,
        should_stop=all_heard,
        speaking=(lambda: False) if canceller else None,
    )
    Path(earlier).unlink()
    assert path is None and spoke == []


# -- while Iris answers --------------------------------------------------------------


class Speaker:
    """The speaker as the chimes and the drone reach it: what it was given,
    and how often it was stopped."""

    def __init__(self):
        self.played: list[np.ndarray] = []
        self.stops = 0

    def play(self, audio, sample_rate, **options) -> float:
        self.played.append(np.asarray(audio))
        return 0.0

    def stop(self):
        self.stops += 1


class Voice(HeldPlayer):
    """The voice: each clip plays until released; stopping it stops the
    speaker, as AudioPlayer does."""

    def __init__(self):
        super().__init__()
        self.audio = Speaker()

    def stop(self):
        self.audio.stop()

    def resume(self, rewind=True):
        pass


class Model:
    """A model that thinks until the test lets it say the next part of its
    reply."""

    def __init__(self, *parts):
        self.parts = [(threading.Event(), text) for text in parts]

    def go_on(self):
        next(ready for ready, _ in self.parts if not ready.is_set()).set()

    def finish(self):
        for ready, _ in self.parts:
            ready.set()

    def chat(self, model, messages, stream=False, tools=None, **options):
        def chunks():
            for ready, text in self.parts:
                if not ready.wait(10):
                    return
                yield {"message": {"content": text}}

        return chunks()


def _assistant(*parts, wake_word=None) -> VoiceAssistant:
    """An assistant with the real reply stream and speech pipeline, awake,
    the model saying these parts of its reply."""
    a = object.__new__(VoiceAssistant)
    a.event_bus = EventBus()
    a.status_manager = Status()
    a.speak_replies = True
    a.memory = None
    a.wake_word = wake_word
    a.wake_window = 30.0
    a.awake_until = 0.0
    a._last_window_emit = 0.0
    a.interrupt_event = threading.Event()
    a.suspend_event = threading.Event()
    a.stop_event = threading.Event()
    a.player = Voice()
    a.speech_pipeline = SpeechPipeline(InstantGenerator(), a.player)
    a.tts_generator = SimpleNamespace(
        languages=lambda: frozenset({"en"}),
        speaks=lambda language: True,
        ready=lambda language: True,
        prepare_language=lambda language: None,
    )
    a.tools = Tools([], PluginContext())
    with patch("iris.llm.ollama") as mock_ollama:
        mock_ollama.show.return_value = MagicMock(modelinfo={}, details=MagicMock(family="test"))
        mock_ollama.list.return_value = []
        a.conversation = ConversationManager(model="test", tools=a.tools)
    a.conversation.client = Model(*parts)
    if wake_word:
        a._stay_awake()
    return a


@pytest.mark.parametrize(
    "stop",
    [lambda a: a.interrupt_event.set(), lambda a: a.stop_event.set()],
    ids=["the reply stopped", "Iris shutting down"],
)
def test_the_drone_may_not_sound_once_stopped(stop):
    a = _assistant()
    assert a._may_think_aloud()
    stop(a)
    assert not a._may_think_aloud()


@pytest.fixture
def answering(monkeypatch):
    """Start answering with an assistant as _assistant makes it, its drone
    on the test's clock: (assistant, clock, model, the turn)."""
    clock = Clock()
    on_the_clock = functools.partial(ThinkingDrone, clock=clock)
    monkeypatch.setattr(assistant_module, "ThinkingDrone", on_the_clock)
    made = []

    def start(*parts, wake_word=None):
        a = _assistant(*parts, wake_word=wake_word)
        model = a.conversation.client
        turn = threading.Thread(target=a.chat, args=("What's the weather?",), daemon=True)
        turn.start()
        made.append((a, model, turn))
        # The wait is counted from when the drone began to look at it
        assert _wait(lambda: clock.reads > 0)
        return a, clock, model, turn

    yield start
    for a, model, turn in made:
        model.finish()
        a.player.release.set()
        turn.join(5)


def test_a_reply_that_comes_quickly_has_no_drone(answering):
    a, clock, model, turn = answering("It's sunny.")
    clock.advance(THINKING_DRONE_AFTER - 0.5)
    assert clock.looked()
    model.go_on()
    a.player.release.set()
    turn.join(5)
    assert not turn.is_alive()
    clock.advance(THINKING_DRONE_AFTER * 3)
    assert a.player.audio.played == []


def test_a_long_wait_has_a_drone_that_never_sounds_over_the_voice(answering):
    a, clock, model, turn = answering("Let me look that up.\n\n", "It's sunny all day.")
    speaker = a.player.audio
    clock.advance(THINKING_DRONE_AFTER)
    assert _wait(lambda: len(speaker.played) == 1)  # it fades in after a few seconds
    assert np.array_equal(speaker.played[0], _thinking_drone())
    clock.advance(DRONE_RENEW / 2)
    assert clock.looked()
    assert len(speaker.played) == 1  # one clip, held

    model.go_on()  # its first sentence: the voice says it, which ends the drone
    assert a.player.playing.wait(3)
    stops = speaker.stops
    clock.advance(DRONE_RENEW)
    assert clock.looked()
    # Nothing over the voice, and the voice not stopped
    assert len(speaker.played) == 1 and speaker.stops == stops

    a.player.release.set()  # said; the tool it looked up runs
    assert _wait(lambda: not a.speech_pipeline.is_active)
    clock.advance(THINKING_DRONE_AFTER)
    assert _wait(lambda: len(speaker.played) == 2)  # a few seconds after the voice

    model.go_on()  # the rest of the reply
    turn.join(5)
    assert not turn.is_alive()
    clock.advance(DRONE_RENEW)
    assert len(speaker.played) == 2  # the thinking is over


def _stop(a):
    """Stop, as the app's button does, and as opening another thread does
    before it chimes."""
    controller = AssistantController(EventBus())
    controller.assistant = a
    controller.interrupt()


def _speak_over(a):
    """The user speaks while the reply is thought out (as in
    VoiceAssistant._answer_listening)."""
    a.interrupt_event.set()
    a.speech_pipeline.stop()
    a.speech_pipeline.resume()


@pytest.mark.parametrize("stop", [_stop, _speak_over], ids=["Stop", "the user speaking"])
def test_a_stopped_reply_ends_the_drone_at_once(answering, stop):
    a, clock, model, turn = answering("It's sunny.")
    speaker = a.player.audio
    clock.advance(THINKING_DRONE_AFTER)
    assert _wait(lambda: len(speaker.played) == 1)
    stops = speaker.stops
    stop(a)
    assert speaker.stops > stops  # the drone is cut on the speaker
    clock.advance(THINKING_DRONE_AFTER)
    turn.join(5)
    assert not turn.is_alive()
    clock.advance(THINKING_DRONE_AFTER * 3)
    assert len(speaker.played) == 1


@pytest.mark.parametrize(
    "quiet",
    [
        VoiceAssistant._end_conversation,
        VoiceAssistant._go_to_sleep,
        lambda a: a.suspend_event.set(),
        lambda a: setattr(a, "speak_replies", False),
    ],
    ids=["the sleep tool", "put to sleep", "paused", "the voice off"],
)
def test_the_drone_ends_on_the_speaker_once_it_may_not_sound(answering, quiet):
    a, clock, model, turn = answering("Good night.", wake_word="Ada")
    speaker = a.player.audio
    clock.advance(THINKING_DRONE_AFTER)
    assert _wait(lambda: len(speaker.played) == 1)
    quiet(a)
    assert _wait(lambda: speaker.stops == 1)
    for _ in range(3):
        clock.advance(THINKING_DRONE_AFTER)
        assert clock.looked()
    assert len(speaker.played) == 1 and speaker.stops == 1


class Unspeakable:
    """A voice that cannot say the piece it is given, once let go."""

    def __init__(self):
        self.release = threading.Event()

    def generate(self, text, language="en"):
        self.release.wait(10)
        return None


def test_a_drone_the_voice_never_took_over_from_ends_with_the_turn(answering):
    a, clock, model, turn = answering("Here it is.\n\n")
    voice = a.speech_pipeline.generator = Unspeakable()
    speaker = a.player.audio
    clock.advance(THINKING_DRONE_AFTER)
    assert _wait(lambda: len(speaker.played) == 1)
    model.go_on()  # a piece for the voice: the drone leaves it the speaker
    assert _wait(lambda: not a.status_manager.status_thinking)
    assert speaker.stops == 0
    voice.release.set()  # which says nothing after all
    turn.join(5)
    assert not turn.is_alive()
    assert speaker.stops == 1

"""Unit tests for the low ping Iris sounds now and then while a reply is
long in coming: none for a reply that comes quickly, the first after a few
seconds of silence, then one every few seconds, never over the voice, none
once the reply is stopped or Iris goes to sleep, and none the microphone
takes for the user: the echo canceller is told of it once it has learned
the room, and not before.

Silent: the clock is the test's, moved on by it, and the speaker a stand-in
that notes what it is given; the model is scripted.
"""

import functools
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from iris import assistant as assistant_module
from iris import audio as audio_module
from iris.assistant import (
    THINKING_PING_LEVEL,
    THINKING_TONE_AFTER,
    THINKING_TONE_EVERY,
    ThinkingTones,
    VoiceAssistant,
    _sleep_chime,
    _thinking_ping,
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
from iris.turns import FRAME, SAMPLE_RATE, SpeechDetector, TurnTracker
from irisdev.voicetest.acoustics import in_room, room, to_rate
from tests.unit.test_echo import _voice
from tests.unit.test_sleep_tool import HeldPlayer, InstantGenerator, Status, _wait
from tests.unit.test_turn_taking import FakeStream, Verdict, _turn_file

needs_canceller = pytest.mark.skipif(not HAS_ECHO_CANCELLATION, reason="no echo canceller here")


class Clock:
    """The test's clock: it moves on only when told, and counts how often it
    was read, so a test can wait for the tones to have looked at the time."""

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
        """Wait until the clock was read this many more times: the tones
        looked at the wait at least once since it was last moved on."""
        with self._read:
            target = self.reads + times
            return self._read.wait_for(lambda: self.reads >= target, timeout=5.0)


def _run(tones: ThinkingTones, clock: Clock, seconds: float, step: float = 0.25):
    """Let so much time pass, the tones looking at it every step."""
    for _ in range(round(seconds / step)):
        clock.advance(step)
        tones.tick()


def _tones(heard=lambda: False, may_sound=lambda: True):
    clock = Clock()
    played = []
    tones = ThinkingTones(lambda: played.append(clock.now), heard, may_sound, clock=clock)
    return tones, clock, played


# -- when a tone is due ----------------------------------------------------------


def test_the_first_tone_comes_after_a_few_seconds_of_silence_then_one_every_few():
    tones, clock, played = _tones()
    start = clock.now
    _run(tones, clock, THINKING_TONE_AFTER - 0.25)
    assert played == []  # nothing for a reply that comes within it
    _run(tones, clock, 30.0 - (THINKING_TONE_AFTER - 0.25))
    expected = list(np.arange(start + THINKING_TONE_AFTER, start + 30.0, THINKING_TONE_EVERY))
    assert played == expected and len(played) == 5


def test_silence_is_counted_from_when_the_voice_was_last_heard():
    speaking = [False]
    tones, clock, played = _tones(heard=lambda: speaking[0])
    _run(tones, clock, 2.0)
    speaking[0] = True  # "Let me look that up."
    _run(tones, clock, 20.0)
    assert played == []
    speaking[0] = False  # a tool runs
    ended = clock.now
    _run(tones, clock, THINKING_TONE_AFTER)
    assert played == [ended + THINKING_TONE_AFTER]


def test_no_tone_while_one_may_not_sound_nor_once_stopped():
    allowed = [False]
    tones, clock, played = _tones(may_sound=lambda: allowed[0])
    _run(tones, clock, 20.0)
    assert played == []
    allowed[0] = True
    _run(tones, clock, THINKING_TONE_AFTER)
    assert len(played) == 1
    tones.stop()
    _run(tones, clock, 30.0)
    assert len(played) == 1


# -- the ping ----------------------------------------------------------------------


def _loudest(sound: np.ndarray, rate: int = 24000) -> float:
    """The RMS of its loudest 50 ms."""
    window = rate // 20
    return max(
        float(np.sqrt(np.mean(sound[i : i + window] ** 2)))
        for i in range(0, len(sound) - window + 1, window // 4)
    )


def test_the_ping_is_low_soft_and_gives_way_to_the_next_without_a_click():
    ping = _thinking_ping()
    assert ping is _thinking_ping()  # made once
    _thinking_ping.cache_clear()
    assert np.array_equal(ping, _thinking_ping())  # the same every time
    assert abs(np.abs(ping).max() - THINKING_PING_LEVEL) < 1e-6
    for chime in (_wake_chime(), _sleep_chime(), _thread_chime()):
        assert np.abs(ping).max() <= np.abs(chime).max() / 4
        assert _loudest(ping) <= _loudest(chime) / 3
    # It swells in, with no edge to click
    assert abs(ping[0]) < 1e-9 and np.abs(np.diff(ping)).max() < 0.002
    # It ends faded out just before the next one is due
    assert THINKING_TONE_EVERY - 0.2 < len(ping) / 24000 < THINKING_TONE_EVERY
    assert np.abs(ping[-24:]).max() < 1e-4
    # Low and round: next to nothing of it above 1 kHz
    power = np.abs(np.fft.rfft(ping)) ** 2
    frequencies = np.fft.rfftfreq(len(ping), 1 / 24000)
    assert power[frequencies > 1000].sum() < 1e-4 * power.sum()


def test_the_voice_cuts_its_tail_low():
    """The voice starting cuts the ping wherever it is: at most at its
    peak, 30 dB under the voice's (0.85), and a few seconds in, far lower."""
    ping = _thinking_ping()
    assert np.abs(ping).max() <= 0.85 * 10 ** (-30 / 20)
    assert np.abs(ping[3 * 24000 :]).max() < 0.25 * THINKING_PING_LEVEL


def _turns(x: np.ndarray) -> int:
    """How many turns Iris's turn tracker, with Silero VAD, starts on this
    16 kHz audio: listening starts afresh after each, as on_speech would."""
    detector = SpeechDetector()
    tracker, turns = TurnTracker(detector, None, 3.5), 0
    for i in range(0, len(x) - FRAME + 1, FRAME):
        tracker.feed(x[i : i + FRAME])
        if tracker.started:
            turns += 1
            detector.reset()
            tracker = TurnTracker(detector, None, 3.5)
    return turns


class Room:
    """A room with Iris's speaker and microphone in it: what is played into
    it is heard back on demand. The speaker plays one clip at a time, the
    microphone hears it through the room 60 ms later, over a noise floor,
    and the echo canceller cleans what it hears."""

    def __init__(self, echo: EchoCanceller):
        self.echo = echo
        self.start = time.monotonic()
        self.clips: list[tuple[int, np.ndarray, int]] = []  # (from, clip, rate)
        self.heard_to = 0  # samples heard and cleaned so far
        self.response = room(0.4, 9.0)
        self.noise = np.random.default_rng(7)

    def play(self, audio, sample_rate, **options) -> float:
        """The speaker, as the assistant plays on it: the clip is heard from
        where the microphone has got to, and that moment is told back as its
        latency."""
        self.clips.append((self.heard_to, np.asarray(audio), sample_rate))
        return self.start + self.heard_to / SAMPLE_RATE - time.monotonic()

    def say(self, voice: np.ndarray):
        """The voice (16 kHz), told to the canceller as the player tells it."""
        started = time.monotonic()
        self.echo.played(started + self.play(voice, SAMPLE_RATE), voice, SAMPLE_RATE)

    def _speaker(self, until: int) -> np.ndarray:
        speaker = np.zeros(until, dtype=np.float32)
        for begin, clip, rate in self.clips:
            clip = to_rate(clip, rate)[: max(0, until - begin)]
            speaker[begin:] = 0.0  # a new clip ends the one before
            speaker[begin : begin + len(clip)] = clip
        return speaker

    def listen(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """The next seconds the microphone hears: (what it heard, what the
        canceller left of it)."""
        a = self.heard_to
        b = a + int(seconds * SAMPLE_RATE) // 480 * 480
        delayed = np.concatenate([np.zeros(960, dtype=np.float32), self._speaker(b)])[:b]
        heard = in_room(delayed, self.response)[a:b]
        heard = (heard + self.noise.standard_normal(len(heard)) * 0.0005).astype(np.float32)
        cleaned = np.concatenate([
            self.echo.clean(heard[i : i + 480], self.start + (a + i) / SAMPLE_RATE)
            for i in range(0, len(heard), 480)
        ])
        self.heard_to = b
        return heard, cleaned


def _energy(x: np.ndarray) -> float:
    return float(np.mean(x**2))


@needs_canceller
def test_the_canceller_is_told_of_the_ping_once_it_has_learned_the_room():
    """Told of the ping before it has found the voice's way to the
    microphone, the canceller lets more of it through, and more like speech,
    than untold: so it is told of it only once it has heard enough of the
    voice. Either way the speech detector hears nobody in it."""
    a = object.__new__(VoiceAssistant)
    a.echo = EchoCanceller()
    listening = Room(a.echo)
    a.player = SimpleNamespace(audio=listening)
    listening.listen(1.0)

    # Fresh: the ping is not told
    a._play_thinking_ping()
    assert not a.echo._played
    heard, cleaned = listening.listen(THINKING_TONE_EVERY)
    assert _turns(heard) == 0 and _turns(cleaned) == 0

    # Once it has heard the voice for a while (this one is voiced half the
    # time), it is told, and takes it out
    listening.say(_voice(24.0))
    listening.listen(25.0)
    assert a.echo.learned
    a._play_thinking_ping()
    assert abs(len(a.echo._played[-1][1]) - len(_thinking_ping()) * 2 / 3) <= 1
    heard, cleaned = listening.listen(THINKING_TONE_EVERY)
    assert _turns(cleaned) == 0
    assert _energy(cleaned) < _energy(heard) / 10


@pytest.mark.parametrize("canceller", [
    pytest.param(False, id="alone"),
    pytest.param(True, id="through an echo canceller not told of it", marks=needs_canceller),
])
def test_the_microphone_does_not_take_the_ping_for_the_user(monkeypatch, canceller):
    """While the model thinks, the microphone listens on for the user going
    on (AudioRecorder.listen_on), with Silero VAD as Iris runs it: a ping
    every few seconds, heard through the room as loud as it is played,
    quieter and louder, starts no turn and interrupts nothing, on the
    microphone alone and through an echo canceller that has not learned
    the room yet."""
    ping = to_rate(_thinking_ping(), 24000)
    speaker = [np.zeros(SAMPLE_RATE, dtype=np.float32)]
    for gain in (0.3, 1.0, 3.0):
        clip = np.zeros(int(THINKING_TONE_EVERY * SAMPLE_RATE), dtype=np.float32)
        clip[: len(ping)] = ping * gain
        speaker.append(clip)
    speaker = np.concatenate(speaker)
    recording = in_room(speaker, room(0.4, 9.0))[: len(speaker)]
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
    """The speaker as the chimes and pings reach it: what it was given, and
    how often it was stopped."""

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
def test_no_ping_may_sound_once_stopped(stop):
    a = _assistant()
    assert a._may_think_aloud()
    stop(a)
    assert not a._may_think_aloud()


@pytest.fixture
def answering(monkeypatch):
    """Start answering with an assistant as _assistant makes it, its tones
    on the test's clock: (assistant, clock, model, the turn)."""
    clock = Clock()
    on_the_clock = functools.partial(ThinkingTones, clock=clock)
    monkeypatch.setattr(assistant_module, "ThinkingTones", on_the_clock)
    made = []

    def start(*parts, wake_word=None):
        a = _assistant(*parts, wake_word=wake_word)
        model = a.conversation.client
        turn = threading.Thread(target=a.chat, args=("What's the weather?",), daemon=True)
        turn.start()
        made.append((a, model, turn))
        # The wait is counted from when the tones began
        assert _wait(lambda: clock.reads > 0)
        return a, clock, model, turn

    yield start
    for a, model, turn in made:
        model.finish()
        a.player.release.set()
        turn.join(5)


def test_a_reply_that_comes_quickly_has_no_tone(answering):
    a, clock, model, turn = answering("It's sunny.")
    clock.advance(THINKING_TONE_AFTER - 0.5)
    assert clock.looked()
    model.go_on()
    a.player.release.set()
    turn.join(5)
    assert not turn.is_alive()
    clock.advance(THINKING_TONE_EVERY * 3)
    assert a.player.audio.played == []


def test_a_long_wait_has_tones_that_never_sound_over_the_voice(answering):
    a, clock, model, turn = answering("Let me look that up.\n\n", "It's sunny all day.")
    tones = a.player.audio.played
    clock.advance(THINKING_TONE_AFTER)
    assert _wait(lambda: len(tones) == 1)  # the first after a few seconds
    assert np.array_equal(tones[0], _thinking_ping())
    clock.advance(THINKING_TONE_EVERY)
    assert _wait(lambda: len(tones) == 2)  # and again while the wait goes on

    model.go_on()  # its first sentence: the voice says it
    assert a.player.playing.wait(3)
    clock.advance(THINKING_TONE_EVERY * 3)
    assert clock.looked()
    assert len(tones) == 2  # never over the voice

    a.player.release.set()  # said; the tool it looked up runs
    assert _wait(lambda: not a.speech_pipeline.is_active)
    clock.advance(THINKING_TONE_AFTER)
    assert _wait(lambda: len(tones) == 3)  # a few seconds after the voice

    model.go_on()  # the rest of the reply
    turn.join(5)
    assert not turn.is_alive()
    clock.advance(THINKING_TONE_EVERY * 3)
    assert len(tones) == 3  # the thinking is over


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
def test_a_stopped_reply_ends_its_tones_at_once(answering, stop):
    a, clock, model, turn = answering("It's sunny.")
    tones = a.player.audio.played
    clock.advance(THINKING_TONE_AFTER)
    assert _wait(lambda: len(tones) == 1)
    stops = a.player.audio.stops
    stop(a)
    assert a.player.audio.stops > stops  # the tone playing is cut
    clock.advance(THINKING_TONE_EVERY)  # another would be due
    turn.join(5)
    assert not turn.is_alive()
    clock.advance(THINKING_TONE_EVERY * 3)
    assert len(tones) == 1


@pytest.mark.parametrize(
    "sleep",
    [VoiceAssistant._end_conversation, VoiceAssistant._go_to_sleep],
    ids=["the sleep tool", "put to sleep"],
)
def test_no_tone_once_iris_goes_to_sleep(answering, sleep):
    a, clock, model, turn = answering("Good night.", wake_word="Iris")
    tones = a.player.audio.played
    clock.advance(THINKING_TONE_AFTER)
    assert _wait(lambda: len(tones) == 1)
    sleep(a)
    for _ in range(3):
        clock.advance(THINKING_TONE_EVERY)
        assert clock.looked()
    assert len(tones) == 1


@pytest.mark.parametrize(
    "quiet",
    [lambda a: a.suspend_event.set(), lambda a: setattr(a, "speak_replies", False)],
    ids=["paused", "the voice off"],
)
def test_no_tone_while_paused_or_with_the_voice_off(answering, quiet):
    a, clock, model, turn = answering("It's sunny.")
    quiet(a)
    for _ in range(3):
        clock.advance(THINKING_TONE_EVERY)
        assert clock.looked()
    assert a.player.audio.played == []
    a.suspend_event.clear()
    a.speak_replies = True
    clock.advance(THINKING_TONE_AFTER)
    assert _wait(lambda: len(a.player.audio.played) == 1)  # once it may again

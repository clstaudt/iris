"""Echo cancellation: what the player plays is taken out of the microphone."""

import time

import numpy as np
import pytest
from scipy.signal import fftconvolve

from iris import audio as audio_module
from iris.audio import AudioPlayer
from iris.echo import HAS_ECHO_CANCELLATION, RATE, STEP, EchoCanceller

pytestmark = pytest.mark.skipif(not HAS_ECHO_CANCELLATION, reason="no echo canceller here")


def _voice(seconds, seed=1):
    """Voiced, syllable-paced sound with pitch movement: speech enough for AEC3."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    pitch = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / RATE
    voice = sum(np.sin(k * phase) / k for k in range(1, 15))
    syllables = np.clip(np.sin(2 * np.pi * 3.3 * t + rng.uniform(0, 6)), 0, None)
    return (0.1 * voice * syllables).astype(np.float32)


class TestReference:
    def test_what_was_played_at_the_time(self):
        canceller = EchoCanceller()
        clip = np.arange(RATE, dtype=np.float32) / RATE  # 1 s ramp at 16 kHz
        canceller.played(10.0, clip, RATE)
        assert np.allclose(canceller._reference(10.5, STEP), clip[RATE // 2 : RATE // 2 + STEP])
        # Silence before and after it
        assert not canceller._reference(9.0, STEP).any()
        assert not canceller._reference(11.5, STEP).any()

    def test_other_rates_are_brought_to_16_khz(self):
        canceller = EchoCanceller()
        canceller.played(0.0, np.ones(24000, dtype=np.float32), 24000)
        assert abs(len(canceller._played[0][1]) - RATE) <= 1

    def test_nothing_after_a_stop(self):
        canceller = EchoCanceller()
        canceller.played(10.0, np.ones(RATE, dtype=np.float32), RATE)
        canceller.played(11.5, np.ones(RATE, dtype=np.float32), RATE)
        canceller.cut(10.25)
        assert canceller._reference(10.1, STEP).all()
        assert not canceller._reference(10.3, STEP).any()
        assert not canceller._reference(11.6, STEP).any()


def test_the_voice_is_taken_out_and_the_user_left_in():
    played = _voice(6.0)
    rng = np.random.default_rng(0)
    room = rng.standard_normal(int(0.2 * RATE)) * np.exp(-np.arange(int(0.2 * RATE)) / 600)
    room[0] = 1.0
    room /= np.sqrt(np.sum(room**2))
    echo = fftconvolve(played, room)[: len(played)].astype(np.float32)
    # The speaker's sound reaches the microphone 30 ms after it was played
    delay = int(0.03 * RATE)
    mic = np.zeros(len(played) + delay, dtype=np.float32)
    mic[delay:] += echo

    canceller = EchoCanceller()
    canceller.played(0.0, played, RATE)
    block = 3 * STEP
    cleaned = np.concatenate(
        [
            canceller.clean(mic[i : i + block], i / RATE)
            for i in range(0, len(mic) - block + 1, block)
        ]
    )
    settled = slice(3 * RATE, len(cleaned))
    reduction = 10 * np.log10(np.mean(mic[settled] ** 2) / np.mean(cleaned[settled] ** 2))
    assert reduction > 15  # dB

    # Where nothing was played, the user is heard as loud as before (a few
    # milliseconds later, and without rumble)
    canceller = EchoCanceller()
    user = _voice(2.0, seed=7)
    kept = np.concatenate(
        [
            canceller.clean(user[i : i + block], 100 + i / RATE)
            for i in range(0, len(user) - block + 1, block)
        ]
    )
    loss = 10 * np.log10(np.mean(user[RATE // 2 :] ** 2) / np.mean(kept[RATE // 2 :] ** 2))
    assert abs(loss) < 3  # dB


def test_the_player_says_when_its_sound_leaves_the_speaker(monkeypatch):
    registered = []

    class Echo:
        def played(self, start, audio, rate):
            registered.append((start, len(audio), rate))

        def cut(self, at):
            registered.append(("cut", at))

    class Stream:
        latency = 0.1
        active = False

    monkeypatch.setattr(audio_module.sd, "play", lambda *a, **k: None)
    monkeypatch.setattr(audio_module.sd, "stop", lambda: None)
    monkeypatch.setattr(audio_module.sd, "get_stream", lambda: Stream())
    player = AudioPlayer()
    player.echo = Echo()
    before = time.monotonic()
    player._play_resumable(np.zeros(2400, dtype=np.float32), 24000, None)
    (start, length, rate), = registered
    assert before + 0.1 <= start <= time.monotonic() + 0.1
    assert (length, rate) == (2400, 24000)

    player.stop()
    assert registered[-1][0] == "cut"


def _clean_all(canceller, mic):
    block = 3 * STEP
    for i in range(0, len(mic) - block + 1, block):
        canceller.clean(mic[i : i + block], i / RATE)


def _echo_of(played, delay):
    """The voice reaching the microphone delay seconds after it was played."""
    rng = np.random.default_rng(0)
    room = rng.standard_normal(int(0.2 * RATE)) * np.exp(-np.arange(int(0.2 * RATE)) / 600)
    room[0] = 1.0
    room /= np.sqrt(np.sum(room**2))
    echo = fftconvolve(played, room)[: len(played)].astype(np.float32)
    mic = np.zeros(len(played) + int(delay * RATE), dtype=np.float32)
    mic[int(delay * RATE) :] += echo
    return mic


def test_it_says_when_it_is_taking_the_voice_out():
    played = _voice(8.0)
    canceller = EchoCanceller()
    canceller.played(0.0, played, RATE)
    assert canceller.suppression is None and canceller.cancelling is False
    _clean_all(canceller, _echo_of(played, 0.03))
    assert canceller.cancelling is True
    assert canceller.suppression > 15


def test_and_when_the_voice_came_out_late_it_says_it_is_not():
    """Playback that stalled (a busy computer): the voice reaches the
    microphone far later than the canceller was told, and comes through."""
    played = _voice(8.0)
    canceller = EchoCanceller()
    canceller.played(0.0, played, RATE)
    _clean_all(canceller, _echo_of(played, 1.0))
    assert canceller.settled
    assert not canceller.cancelling
    assert canceller.suppression < 10


def test_the_user_speaking_over_the_voice_leaves_it_cancelling():
    played = _voice(8.0)
    canceller = EchoCanceller()
    canceller.played(0.0, played, RATE)
    mic = _echo_of(played, 0.03)
    # Half a second of a louder user, in the middle: two windows of it are
    # spared, more are not
    user = 3 * _voice(0.5, seed=7)
    mic[4 * RATE : 4 * RATE + len(user)] += user
    _clean_all(canceller, mic)
    assert canceller.cancelling is True


def test_but_a_user_speaking_on_over_it_is_not_spared():
    played = _voice(8.0)
    canceller = EchoCanceller()
    canceller.played(0.0, played, RATE)
    mic = _echo_of(played, 0.03)
    user = 3 * _voice(1.5, seed=7)
    mic[6 * RATE : 6 * RATE + len(user)] += user
    _clean_all(canceller, mic)
    assert canceller.cancelling is False


def test_with_nothing_to_cancel_it_counts_as_cancelling():
    """Headphones: the voice does not reach the microphone, only the room's
    noise does, before the voice and while it plays."""
    played = _voice(8.0)
    canceller = EchoCanceller()
    canceller.played(1.0, played, RATE)
    rng = np.random.default_rng(5)
    mic = (rng.standard_normal(9 * RATE) * 0.003).astype(np.float32)
    _clean_all(canceller, mic)
    assert canceller.settled
    assert canceller.suppression is None
    assert canceller.cancelling is True

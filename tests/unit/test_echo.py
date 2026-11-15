"""Echo cancellation: what the player plays is taken out of the microphone."""

import time

import numpy as np
import pytest
from scipy.signal import fftconvolve

from iris import audio as audio_module
from iris.audio import AudioPlayer
from iris.echo import HAS_ECHO_CANCELLATION, RATE, STEP, EchoCanceller
from iris.turns import HOLD_CANCELLED, HOLD_OUT_OF_RESIDUE, HOLD_RESIDUAL, HOLD_STANDS_OUT, HOLD_WITHIN_ECHO

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


def test_the_users_speech_before_the_voice_is_not_taken_for_the_rooms_noise():
    """The user spoke on for 2 s, then the voice answered, its echo through
    the room 0.8 s late: more than the canceller finds, so it comes through.
    That is judged against the room's noise, not the user's speech."""
    voice = _voice(6.0)
    canceller = EchoCanceller()
    canceller.played(3.0, voice, RATE)
    rng = np.random.default_rng(5)
    mic = (rng.standard_normal(10 * RATE) * 0.003).astype(np.float32)
    t = np.arange(int(2.2 * RATE)) / RATE
    user = 0.05 * sum(np.sin(2 * np.pi * k * 180 * t) / k for k in range(1, 10))
    mic[RATE // 2 : RATE // 2 + len(user)] += user.astype(np.float32)
    echo = _echo_of(voice, 0.8)
    mic[3 * RATE : 3 * RATE + len(echo)] += echo[: len(mic) - 3 * RATE]
    _clean_all(canceller, mic)
    assert canceller.settled
    assert canceller.cancelling is False


def _replies_with_a_new_delay(user_at=None, user_gain=3.0):
    """Two replies of the voice, the second's sound coming out 0.1 s later
    than the first's (its stream woke with a delay of its own), cleaned from
    a cold canceller on: for each 128 ms of each reply, how what was heard
    stood against the voice's echo, and whether the user spoke in it.
    user_at: (reply, seconds into it) the user speaks for 1.5 s, louder."""
    rng = np.random.default_rng(3)
    replies = [(0.5, 0.03), (10.0, 0.13)]
    voices = [_voice(6.0, seed=1), _voice(6.0, seed=2)]
    mic = (rng.standard_normal(17 * RATE) * 0.001).astype(np.float32)
    for (start, delay), voice in zip(replies, voices):
        echo = 0.5 * _echo_of(voice, delay)
        mic[int(start * RATE) : int(start * RATE) + len(echo)] += echo
    user = None
    if user_at is not None:
        reply, into = user_at
        begins = replies[reply][0] + into
        speech = user_gain * _voice(1.5, seed=7)
        mic[int(begins * RATE) : int(begins * RATE) + len(speech)] += speech
        user = (begins + 0.05, begins + 1.5)
    canceller = EchoCanceller()
    for (start, _), voice in zip(replies, voices):
        canceller.played(start, voice, RATE)
    block, span = 3 * STEP, 0.128
    weighed = []
    for i in range(0, len(mic) - block + 1, block):
        canceller.clean(mic[i : i + block], i / RATE)
        end = (i + block) / RATE
        if any(start <= end - span and end <= start + 6.0 for start, _ in replies):
            speaking = user is not None and user[0] <= end - span and end <= user[1]
            weighed.append((end, canceller.against_echo(end - span, end), speaking, canceller.settled))
    return weighed


def test_from_a_replys_first_word_its_echo_does_not_stand_out():
    """From the start of each reply, on a cold canceller too (its first 3 s,
    before it has learned the room) and once its sound comes out later than
    before (the canceller lets it through while it finds it again): the
    voice's echo is never heard standing out above what it could be, nor
    out of what the canceller left of it just before, so it cannot hold the
    voice."""
    weighed = _replies_with_a_new_delay()
    known = [levels for _, levels, _, _ in weighed if levels is not None]
    # Known from the first quarter second of voice on, cold or not
    assert len(known) > 0.9 * len(weighed)
    assert any(levels is not None for _, levels, _, settled in weighed if not settled)
    assert max(levels.heard for levels in known) < HOLD_STANDS_OUT
    cancelled = [levels for levels in known if levels.before is not None and levels.before <= -HOLD_CANCELLED]
    assert cancelled
    assert not [levels for levels in cancelled if levels.left - levels.before >= HOLD_OUT_OF_RESIDUE
                and (levels.left <= HOLD_RESIDUAL or levels.heard <= HOLD_WITHIN_ECHO)]


def test_a_user_louder_than_the_voice_stands_out_at_once():
    for reply, into in ((0, 1.0), (1, 0.3)):
        weighed = _replies_with_a_new_delay(user_at=(reply, into))
        over = [levels.heard for _, levels, speaking, _ in weighed if speaking and levels is not None]
        assert over and max(over[:4]) >= HOLD_STANDS_OUT


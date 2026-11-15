"""Tests for the playback level envelope used by the output level meter."""

import threading
import time

import numpy as np
import pytest

from iris import audio as audio_module
from iris.audio import AudioPlayer


def test_emit_levels_tracks_amplitude():
    sample_rate = 16000
    # Half a second of silence followed by half a second of a loud tone
    quiet = np.zeros(sample_rate // 2, dtype=np.float32)
    t = np.linspace(0, 0.5, sample_rate // 2, dtype=np.float32)
    loud = (0.8 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio = np.concatenate([quiet, loud])

    levels = []
    stop = threading.Event()
    # A high rate keeps the test fast while still walking the whole envelope
    AudioPlayer._emit_levels(audio, sample_rate, levels.append, stop, rate=1000)

    assert levels, "expected level callbacks"
    assert all(0.0 <= level <= 1.0 for level in levels)
    # Silence first, signal later
    assert levels[0] == 0.0
    assert max(levels) > 0.5


def test_emit_levels_stops_when_requested():
    sample_rate = 16000
    audio = np.zeros(sample_rate * 5, dtype=np.float32)

    levels = []
    stop = threading.Event()
    stop.set()  # already stopped: must return without emitting
    AudioPlayer._emit_levels(audio, sample_rate, levels.append, stop, rate=1000)

    assert levels == []


def test_emit_levels_handles_audio_shorter_than_one_frame():
    levels = []
    AudioPlayer._emit_levels(
        np.zeros(10, dtype=np.float32), 16000, levels.append, threading.Event()
    )
    assert levels == []


class Clock:
    """The audio module's clock, moved on by the test alone: the time heard
    is what the test says, however slowly the threads run. Sleeps are real,
    so the threads still take turns."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


def _wait_for(condition, timeout=10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return bool(condition())


def _device(monkeypatch):
    """A speaker that plays until the test says the clip is over, on the
    test's clock; the clips it was given."""
    played = []

    class FakeStream:
        latency = 0.0
        active = True

    stream = FakeStream()

    def fake_play(segment, rate, **kwargs):
        played.append(segment.copy())
        stream.active = True

    def fake_stop():
        stream.active = False

    clock = Clock()
    monkeypatch.setattr(audio_module, "time", clock)
    monkeypatch.setattr(audio_module.sd, "play", fake_play)
    monkeypatch.setattr(audio_module.sd, "stop", fake_stop)
    monkeypatch.setattr(audio_module.sd, "get_stream", lambda: stream)
    return clock, stream, played


def _held(monkeypatch, audio, sample_rate, heard: float) -> float:
    """Play a clip, pause it once so much of it was heard, and resume it;
    where the rest was played from, in seconds."""
    clock, stream, played = _device(monkeypatch)
    player = AudioPlayer()
    worker = threading.Thread(target=player._play_resumable, args=(audio, sample_rate, None))
    worker.start()
    assert _wait_for(lambda: played)
    clock.now += heard
    player.pause()
    assert _wait_for(lambda: not stream.active)  # the device is stopped
    # However long the pause lasts, it is not counted as heard
    clock.now += 5.0
    assert len(played) == 1 and player.paused

    player.resume()
    assert _wait_for(lambda: len(played) == 2)
    stream.active = False  # the remainder finishes
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert len(played) == 2
    return (len(audio) - len(played[1])) / sample_rate


def test_paused_playback_resumes_where_it_was_cut(monkeypatch):
    """A pause stops the device; resume plays the rest, slightly rewound."""
    sample_rate = 16000
    audio = np.linspace(0, 1, sample_rate * 2, dtype=np.float32)  # 2 s, a ramp
    resumed_at = _held(monkeypatch, audio, sample_rate, heard=0.8)
    # Restarted a little before the cut, never after it
    assert resumed_at == pytest.approx(0.8 - audio_module.RESUME_REWIND, abs=0.001)


def test_playback_waits_until_the_stream_has_really_finished(monkeypatch):
    """The stream outlives the clip by its buffers; returning before it is
    inactive let the next clip's sd.play() cut off the last words."""
    import time as time_module

    from iris import audio as audio_module

    sample_rate = 16000
    audio = np.zeros(sample_rate // 10, dtype=np.float32)  # a 0.1 s clip
    started = []

    class SlowStream:
        latency = 0.0

        @property
        def active(self):
            # Still draining its buffers a second after the clip's own length
            return time_module.monotonic() - started[0] < 1.0

    stream = SlowStream()
    def fake_play(*args, **kwargs):
        started.append(time_module.monotonic())

    monkeypatch.setattr(audio_module.sd, "play", fake_play)
    monkeypatch.setattr(audio_module.sd, "stop", lambda: None)
    monkeypatch.setattr(audio_module.sd, "get_stream", lambda: stream)

    AudioPlayer()._play_resumable(audio, sample_rate, None)
    assert time_module.monotonic() - started[0] >= 1.0


def test_a_clip_held_early_does_not_start_over(monkeypatch):
    """A hold soon after a clip began goes back at most half of what was
    heard: resuming from its start repeated the first words at every hold."""
    sample_rate = 16000
    audio = np.linspace(0, 1, sample_rate * 2, dtype=np.float32)
    # Held 0.3 s in, less than the rewind
    assert _held(monkeypatch, audio, sample_rate, heard=0.3) == pytest.approx(0.15, abs=0.001)

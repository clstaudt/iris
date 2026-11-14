"""Tests for the playback level envelope used by the output level meter."""

import threading
import time

import numpy as np

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


def test_paused_playback_resumes_where_it_was_cut(monkeypatch):
    """A pause stops the device; resume plays the rest, slightly rewound."""
    import time as time_module

    from iris import audio as audio_module

    sample_rate = 16000
    audio = np.linspace(0, 1, sample_rate * 2, dtype=np.float32)  # 2 s, a ramp
    played = []

    class FakeStream:
        latency = 0.0

        def __init__(self):
            self.active = True

    stream = FakeStream()

    def fake_play(segment, rate, **kwargs):
        played.append(segment.copy())
        stream.active = True

    def fake_stop():
        stream.active = False

    monkeypatch.setattr(audio_module.sd, "play", fake_play)
    monkeypatch.setattr(audio_module.sd, "stop", fake_stop)
    monkeypatch.setattr(audio_module.sd, "get_stream", lambda: stream)

    player = AudioPlayer()
    worker = threading.Thread(target=player._play_resumable, args=(audio, sample_rate, None))
    worker.start()
    time_module.sleep(0.8)  # ~0.8 s "heard"
    player.pause()
    time_module.sleep(0.2)
    assert len(played) == 1 and player.paused

    player.resume()
    time_module.sleep(0.2)
    stream.active = False  # the remainder finishes
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(played) == 2
    resumed_at = len(audio) - len(played[1])
    heard = 0.8 - audio_module.RESUME_REWIND
    # Restarted a little before the cut, never after it
    assert abs(resumed_at / sample_rate - heard) < 0.15


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

    monkeypatch.setattr(audio_module.sd, "play", fake_play)
    monkeypatch.setattr(audio_module.sd, "stop", fake_stop)
    monkeypatch.setattr(audio_module.sd, "get_stream", lambda: stream)

    player = AudioPlayer()
    worker = threading.Thread(target=player._play_resumable, args=(audio, sample_rate, None))
    worker.start()
    time.sleep(0.3)  # held 0.3 s in, less than the rewind
    player.pause()
    time.sleep(0.2)
    player.resume()
    time.sleep(0.2)
    stream.active = False
    worker.join(timeout=2)

    resumed_at = (len(audio) - len(played[1])) / sample_rate
    assert 0.1 < resumed_at < 0.3

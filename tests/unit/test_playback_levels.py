"""Tests for the playback level envelope used by the output level meter."""

import threading

import numpy as np

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

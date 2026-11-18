"""The end-to-end test's core: what its virtual speaker tells the test."""

import time

import numpy as np

from irisdev import e2e_core
from irisdev.voicetest.acoustics import RATE, Clip, Speaker


def _voice(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_the_speaker_tells_the_clips_cut_off_before_their_end(monkeypatch):
    """A reply's voice cut off (by the next clip, or Stop) is told apart from
    one played to its end, however short: a one-word reply is a reply."""
    speaker = Speaker()
    monkeypatch.setattr(e2e_core.LAPTOP, "speaker", speaker)
    # From two seconds ago, over the next minute
    span = {"since": (time.time() - 2) * 1000, "until": (time.time() + 60) * 1000}
    # A word heard to its end a second ago
    speaker.clips.append(Clip(_voice(0.4), time.monotonic() - 1))
    assert e2e_core.speaker(span)["cut"] == 0
    # Clips far longer than the test, so each is still heard when cut:
    # the next clip cuts the one being heard
    speaker.play(_voice(30), RATE)
    speaker.play(_voice(30), RATE)
    assert e2e_core.speaker(span)["cut"] == 1
    # Stop cuts the one playing
    speaker.stop()
    told = e2e_core.speaker(span)
    assert (told["clips"], told["cut"]) == (3, 2)

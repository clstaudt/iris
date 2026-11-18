"""The end-to-end test's core: what its virtual speaker tells the test."""

import time

import numpy as np

from irisdev import e2e_core
from irisdev.voicetest.acoustics import RATE


def _voice(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_the_speaker_tells_the_clips_cut_off_before_their_end():
    """A reply's voice cut off (by the next clip, or Stop) is told apart from
    one played to its end, however short: a one-word reply is a reply."""
    # Over the next ten seconds, without waiting for them
    span = {"since": time.time() * 1000, "until": (time.time() + 10) * 1000}
    speaker = e2e_core.LAPTOP.speaker
    speaker.play(_voice(0.4), RATE)
    assert e2e_core.speaker(span)["cut"] == 0
    # The next clip cuts the one still being heard
    speaker.play(_voice(2.0), RATE)
    assert e2e_core.speaker(span)["cut"] == 1
    # Stop cuts the one playing
    speaker.stop()
    told = e2e_core.speaker(span)
    assert (told["clips"], told["cut"]) == (2, 2)

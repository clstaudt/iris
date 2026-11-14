"""The microphone stays open from one listen to the next, so nothing said
between two listeners is lost, and the VAD starts afresh after long quiet."""

import time
from types import SimpleNamespace

import numpy as np

from iris.audio import TAKE_UP_WITHIN, AudioRecorder
from iris.echo import STEP
from iris.turns import FRAME, SAMPLE_RATE, VAD_REFRESH, TurnTracker

BLOCK = 3 * STEP


class Stream:
    """An input stream the test feeds block by block."""

    def __init__(self, callback, **options):
        self.callback = callback
        self.open = False

    def __enter__(self):
        self.open = True
        return self

    def __exit__(self, *exc):
        self.open = False

    def feed(self, value: float, blocks: int = 1, age: float = 0.01):
        """Blocks of a constant value, one per block's duration as a real
        microphone delivers them, each captured age seconds before."""
        for _ in range(blocks):
            time.sleep(BLOCK / SAMPLE_RATE)
            now = time.monotonic()
            info = SimpleNamespace(currentTime=now, inputBufferAdcTime=now - age)
            self.callback(np.full((BLOCK, 1), value, dtype=np.float32), BLOCK, info, None)


class Device:
    def __init__(self):
        self.streams: list[Stream] = []

    def input_stream(self, **options):
        stream = Stream(**options)
        self.streams.append(stream)
        return stream


def _frames(frames, count):
    return [frames.get(timeout=0.1) for _ in range(count)]


def test_what_is_said_between_two_listeners_reaches_the_second():
    device = Device()
    recorder = AudioRecorder(audio=device)
    with recorder._microphone() as frames:
        stream = device.streams[0]
        stream.feed(0.1, blocks=4)
        _frames(frames, 3)
    # Nobody listens: the user goes on
    stream.feed(0.5, blocks=10)
    with recorder._microphone(go_on=True) as frames:
        heard = np.concatenate(_frames(frames, 4))
    # The same stream, and all said in between (of the first listener's
    # audio only its last part frame, under 32 ms, is left out)
    assert stream.open and len(device.streams) == 1
    assert np.isclose(heard, 0.5).all()


def test_a_listen_long_after_the_last_starts_now():
    device = Device()
    recorder = AudioRecorder(audio=device)
    with recorder._microphone() as frames:
        stream = device.streams[0]
        stream.feed(0.1, blocks=4)
        _frames(frames, 3)
    # Iris's voice, say, while nobody listened
    stream.feed(0.7, blocks=10, age=TAKE_UP_WITHIN + 0.2)
    time.sleep(TAKE_UP_WITHIN + 0.1)
    with recorder._microphone() as frames:
        stream.feed(0.2, blocks=4)
        heard = np.concatenate(_frames(frames, 3))
    assert np.isclose(heard, 0.2).all()


def test_closing_the_microphone_and_opening_it_again():
    device = Device()
    recorder = AudioRecorder(audio=device)
    with recorder._microphone():
        pass
    recorder.close_microphone()
    assert not device.streams[0].open
    with recorder._microphone():
        pass
    assert len(device.streams) == 2 and device.streams[1].open
    recorder.set_device(None)
    assert not device.streams[1].open


class Detector:
    """A VAD that rates every frame as given, and counts its resets."""

    def __init__(self, probability=0.0):
        self.probability = probability
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, frame):
        return self.probability


def test_the_vad_starts_afresh_after_long_quiet():
    detector = Detector(0.0)
    tracker = TurnTracker(detector, finished=None, max_silence=3.0)
    frame = np.zeros(FRAME, dtype=np.float32)
    frames = round(VAD_REFRESH * 2.5 / (FRAME / SAMPLE_RATE))
    for _ in range(frames):
        tracker.feed(frame)
    assert detector.resets == 2


def test_not_while_speech_comes_and_goes():
    detector = Detector(0.4)  # above the silence threshold, below speech
    tracker = TurnTracker(detector, finished=None, max_silence=3.0)
    frame = np.zeros(FRAME, dtype=np.float32)
    for _ in range(round(VAD_REFRESH * 2 / (FRAME / SAMPLE_RATE))):
        tracker.feed(frame)
    assert detector.resets == 0

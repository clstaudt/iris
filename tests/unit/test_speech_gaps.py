"""The pieces of a reply follow one another as closely as the sentences in
one.

Each clip plays on an output stream of its own, block by block, its last
block filled up with silence. The clip is over once its last sample has been
heard, as PortAudio tells: the next one starts then, on a stream of its own,
while the stream before plays out its silence alone and ends by itself.
Waiting for that silence (up to a block, 1.4 s at 24 kHz) and for the stream
to close was a pause of up to 1.5 s between every two pieces of a reply. A
clip still heard is cut by the next, and Stop cuts the clip at once.

The pipeline synthesizes the next piece while one plays, and tells the voice
log how long the voice was silent between two pieces and how much of that it
waited for synthesis; the speaker tells it where it ran out of audio.

sounddevice's stream is a stand-in that the test calls back at the moments it
chooses, on a clock of the test's: nothing is opened, nothing sounds, and no
test depends on how fast it runs.
"""

import logging
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice as sd
import soundfile as sf

from iris import audio, tts
from iris.audio import CLOSE_WITHIN, AudioPlayer, SoundDevice
from iris.tts import SpeechPipeline

RATE = 24000
BLOCK = RATE // 2
# How long after the device asks for a block its first sample is heard
LATENCY = 0.05


class Clock:
    """time, its monotonic() moved on by the test alone; sleeps are real."""

    def __init__(self, now: float = 100.0):
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


class Stream:
    """sounddevice.OutputStream, the device asking for a block whenever the
    test says (ask)."""

    def __init__(self, streams: list, samplerate=None, callback=None, blocksize=0, **options):
        self.samplerate, self.callback = samplerate, callback
        self.blocksize = blocksize or BLOCK
        self.latency = LATENCY
        self.started = self.aborted = self.closed = self.finished = False
        # PortAudio's clock when the callback runs
        self.time = 0.0
        streams.append(self)

    def start(self):
        self.started = True

    @property
    def active(self) -> bool:
        return self.started and not (self.finished or self.aborted or self.closed)

    def abort(self, ignore_errors=True):
        self.aborted = True

    def close(self, ignore_errors=True):
        self.closed = True

    def ask(self, at: float, late: float = 0.0, underflow: bool = False):
        """The device asks for the next block at this moment (on both
        clocks), its first sample heard LATENCY later; the callback runs
        late seconds after that. Returns the block, or None once the stream
        ended by itself."""
        self.time = at + late
        block = np.zeros((self.blocksize, 1), dtype=np.float32)
        info = SimpleNamespace(currentTime=at, outputBufferDacTime=at + LATENCY)
        try:
            self.callback(block, self.blocksize, info, SimpleNamespace(output_underflow=underflow))
        except sd.CallbackAbort:
            self.finished = True
            return None
        return block


@pytest.fixture
def speaker(monkeypatch):
    """The computer's speaker as SoundDevice plays on it, its streams those
    the clips were played on, on the test's clock."""
    streams: list[Stream] = []
    clock = Clock()
    monkeypatch.setattr(audio, "time", clock)
    monkeypatch.setattr(audio.sd, "OutputStream", lambda **options: Stream(streams, **options))
    monkeypatch.setattr(audio, "system_defaults", lambda: None)
    monkeypatch.setattr(audio, "_PORTAUDIO", audio._PortAudio())
    return SimpleNamespace(device=SoundDevice(), streams=streams, clock=clock)


def _clip(seconds: float) -> np.ndarray:
    return np.full(int(seconds * RATE), 0.1, dtype=np.float32)


def _played_out(speaker, seconds: float) -> Stream:
    """Play a clip and have the device take all of it, the clock at the
    moment its last sample is heard; its stream."""
    clock = speaker.clock
    speaker.device.play(_clip(seconds), RATE, blocksize=BLOCK)
    stream, clip = speaker.streams[-1], audio._PORTAUDIO.clip
    while clip.ends is None:
        stream.ask(clock.now)
        clock.now += BLOCK / RATE
    clock.now = clip.ends
    return stream


def _wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return bool(condition())


# The speaker -------------------------------------------------------------------


def test_a_clip_is_over_once_its_last_sample_is_heard_not_its_silence(speaker):
    device, clock = speaker.device, speaker.clock
    device.play(_clip(1.25), RATE, blocksize=BLOCK)  # two blocks and a half
    stream = speaker.streams[0]
    for at in (100.0, 100.5):
        clock.now = at
        stream.ask(at)
        assert device.playing()
    clock.now = 101.0
    last = stream.ask(101.0)
    assert np.count_nonzero(last) == BLOCK // 2  # the rest of the block is silence

    # Its last sample is heard a quarter of a second after the block's first
    clock.now = 101.0 + LATENCY + 0.25 - 0.01
    assert device.playing()
    clock.now += 0.02
    assert not device.playing()
    assert stream.active  # still playing the silence after the clip


def test_the_next_clip_starts_at_once_and_nothing_of_the_last_is_cut(speaker):
    device, clock = speaker.device, speaker.clock
    first = _played_out(speaker, 1.25)
    device.play(_clip(2.0), RATE, blocksize=BLOCK)
    second = speaker.streams[1]
    assert second.started
    assert first.active and not (first.aborted or first.closed)
    second.ask(clock.now)

    # The first ends by itself after its silence; its stream is closed while
    # the second has long to play
    assert first.ask(clock.now) is None
    assert device.playing()
    assert first.closed
    assert second.active and not second.closed


def test_a_clip_still_heard_is_cut_by_the_next(speaker):
    device, clock = speaker.device, speaker.clock
    device.play(_clip(2.0), RATE, blocksize=BLOCK)
    first = speaker.streams[0]
    first.ask(clock.now)
    device.play(_clip(1.0), RATE, blocksize=BLOCK)
    assert first.aborted and first.closed
    assert speaker.streams[1].active


def test_a_stream_that_ended_is_closed_only_while_the_clip_has_long_to_play(speaker):
    """Closing a stream takes a moment: near the end of the clip playing
    now it would have its end noticed late."""
    device, clock = speaker.device, speaker.clock
    first = _played_out(speaker, 1.25)
    device.play(_clip(CLOSE_WITHIN - 0.1), RATE, blocksize=BLOCK)
    second = speaker.streams[1]
    second.ask(clock.now)
    assert first.ask(clock.now) is None
    assert device.playing()
    assert not first.closed

    # Stop ends the clip at once; the stream that only played silence is
    # closed once nothing plays
    device.stop()
    assert second.aborted and second.closed
    assert not first.closed
    assert not device.playing()
    assert first.closed


def test_the_player_goes_on_once_the_clip_has_been_heard(speaker):
    clock = speaker.clock
    player = AudioPlayer(audio=speaker.device)
    worker = threading.Thread(target=player._play_resumable, args=(_clip(1.25), RATE, None))
    worker.start()
    assert _wait_for(lambda: speaker.streams)
    stream = speaker.streams[0]
    stream.ask(100.0)  # all of it: the player's blocks are large

    clock.now = 100.0 + LATENCY + 1.25 + 0.01
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert stream.active  # the player did not wait for the silence after the clip
    assert player.heard == (pytest.approx(100.0 + LATENCY), pytest.approx(clock.now))


def _dropouts(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("dropout")]


def test_a_block_asked_for_too_late_is_a_dropout_in_the_voice_log(speaker, caplog):
    caplog.set_level(logging.DEBUG, logger="iris.voice")
    device, clock = speaker.device, speaker.clock
    device.play(_clip(1.5), RATE, blocksize=BLOCK)
    stream = speaker.streams[0]
    stream.ask(100.0)
    stream.ask(100.5, late=LATENCY + 0.03)  # its first sample was due 30 ms before
    stream.ask(101.0, underflow=True)
    assert _dropouts(caplog) == []  # told once the clip is over

    clock.now = 102.0
    assert not device.playing()
    assert _dropouts(caplog) == [
        "dropout: the speaker had no audio for a block of the clip (30 ms late)",
        "dropout: the speaker had no audio for a block of the clip (PortAudio's output underflow)",
    ]
    device.stop()
    assert len(_dropouts(caplog)) == 2  # once


def test_a_clip_on_time_tells_of_no_dropout(speaker, caplog):
    caplog.set_level(logging.DEBUG, logger="iris.voice")
    _played_out(speaker, 1.25)
    assert not speaker.device.playing()
    assert _dropouts(caplog) == []


# The pipeline ------------------------------------------------------------------


class Voice:
    """Synthesizes each piece at once, but those it is told to hold until
    released; knows which file holds which piece."""

    def __init__(self, held=()):
        self.held = set(held)
        self.release = threading.Event()
        self.texts: dict[str, str] = {}
        self.synthesized = threading.Event()

    def generate(self, text, language="en", **options):
        if text in self.held:
            assert self.release.wait(5)
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(path, np.zeros(240, dtype=np.float32), RATE)
        self.texts[path] = text
        if text == "Two.":
            self.synthesized.set()
        return path


def _spoken(pipeline: SpeechPipeline, *pieces):
    """Speak the pieces and wait until the pipeline is done with them."""
    for piece in pieces:
        pipeline.queue_speech(piece)
    done = threading.Thread(target=pipeline.wait_for_speech, daemon=True)
    done.start()
    done.join(10)
    assert not done.is_alive()


def test_the_next_piece_is_synthesized_while_one_plays():
    voice = Voice()

    class Player:
        heard = (None, None)
        ahead = None

        def play(self, audio_file, **options):
            if voice.texts[audio_file] == "One.":
                Player.ahead = voice.synthesized.wait(5)

    pipeline = SpeechPipeline(voice, Player())
    pipeline.begin_reply()
    _spoken(pipeline, "One.", "Two.")
    assert Player.ahead


def test_each_gap_between_two_pieces_is_in_the_voice_log(monkeypatch, caplog):
    """With how much of it the voice waited for the next piece to be
    synthesized: none when it was ready."""
    caplog.set_level(logging.DEBUG, logger="iris.voice")
    clock = Clock(0.0)
    monkeypatch.setattr(tts, "time", clock)
    voice = Voice(held={"Two."})
    # When each piece begins and ends being heard
    heard = {"One.": (1.0, 3.0), "Two.": (3.6, 5.0), "Three.": (5.1, 6.0)}

    class Player:
        def __init__(self):
            self.heard = (None, None)

        def play(self, audio_file, **options):
            self.heard = heard[voice.texts[audio_file]]
            clock.now = self.heard[1]

    pipeline = SpeechPipeline(voice, Player())
    pipeline.begin_reply()
    for piece in ("One.", "Two.", "Three."):
        pipeline.queue_speech(piece)
    assert _wait_for(lambda: pipeline._heard_to is not None)
    clock.now = 3.5  # the second piece is synthesized half a second after the first was heard
    voice.release.set()
    _spoken(pipeline)

    gaps = [r.getMessage() for r in caplog.records if r.getMessage().startswith("gap")]
    assert gaps == [
        "gap: 0.60 s between two pieces of the reply, "
        "0.50 s of it waiting for the piece to be synthesized",
        "gap: 0.10 s between two pieces of the reply",
    ]

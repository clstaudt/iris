"""Unit tests for turn-taking: when the user has finished speaking."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper.vad import get_vad_model

from iris import audio as audio_module
from iris.audio import AudioRecorder
from iris.turns import FRAME, PRE_SPEECH, SAMPLE_RATE, SpeechDetector, TurnTracker

FRAME_SECONDS = FRAME / SAMPLE_RATE


def frames(seconds: float) -> int:
    return round(seconds / FRAME_SECONDS)


class ScriptedSpeech:
    """A VAD whose verdicts are scripted: speech while the frame is loud."""

    def __call__(self, frame):
        return 0.9 if np.abs(frame).max() > 0.5 else 0.05

    def reset(self):
        pass


class Verdict:
    """A Smart Turn that gives the same verdict every time it is asked."""

    def __init__(self, probability):
        self.probability = probability
        self.heard = []

    def __call__(self, audio):
        self.heard.append(len(audio))
        return self.probability


SPEECH = np.full(FRAME, 0.8, dtype=np.float32)
SILENCE = np.zeros(FRAME, dtype=np.float32)


def run(tracker, script):
    """Feed (frame, count) pairs; return the frame number the turn ended at,
    or None."""
    n = 0
    for frame, count in script:
        for _ in range(count):
            n += 1
            if tracker.feed(frame):
                return n
    return None


class TestTurnTracker:
    def test_a_finished_turn_ends_at_the_first_pause(self):
        tracker = TurnTracker(ScriptedSpeech(), Verdict(0.9), max_silence=3.5)
        ended = run(tracker, [(SILENCE, 10), (SPEECH, frames(1.5)), (SILENCE, frames(5))])
        # 0.2 s into the pause, not 3.5 s
        assert ended == 10 + frames(1.5) + frames(0.2)

    def test_an_unfinished_turn_waits_for_the_long_silence(self):
        finished = Verdict(0.1)
        tracker = TurnTracker(ScriptedSpeech(), finished, max_silence=3.5)
        ended = run(tracker, [(SPEECH, frames(1.0)), (SILENCE, frames(5))])
        assert ended == frames(1.0) + frames(3.5)
        assert len(finished.heard) == 1  # asked once for the pause

    def test_each_pause_is_judged_again(self):
        finished = Verdict(0.1)
        tracker = TurnTracker(ScriptedSpeech(), finished, max_silence=3.5)
        run(
            tracker,
            [(SPEECH, frames(1.0)), (SILENCE, frames(1.0)), (SPEECH, frames(1.0)), (SILENCE, 10)],
        )
        assert len(finished.heard) == 2
        assert tracker.verdicts == [0.1, 0.1]

    def test_without_smart_turn_only_silence_ends_a_turn(self):
        tracker = TurnTracker(ScriptedSpeech(), None, max_silence=1.0)
        ended = run(tracker, [(SPEECH, frames(1.0)), (SILENCE, frames(2))])
        assert ended == frames(1.0) + frames(1.0)

    def test_a_click_does_not_start_a_turn(self):
        tracker = TurnTracker(ScriptedSpeech(), Verdict(0.9), max_silence=0.5)
        assert run(tracker, [(SPEECH, 2), (SILENCE, frames(2))]) is None
        assert not tracker.started

    def test_the_turn_starts_just_before_its_first_syllable(self):
        tracker = TurnTracker(ScriptedSpeech(), Verdict(0.9), max_silence=3.5)
        run(tracker, [(SILENCE, frames(3)), (SPEECH, frames(1.0)), (SILENCE, frames(1))])
        kept_before = len(tracker.audio()) / SAMPLE_RATE - 1.0 - 0.2
        assert abs(kept_before - PRE_SPEECH) <= FRAME_SECONDS * 1.5


def _speechlike(seconds=2.0):
    """A voiced, syllable-paced signal: not speech, but it moves the VAD."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    voice = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12))
    syllables = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    signal = 0.1 * voice * syllables
    return signal[: len(signal) // FRAME * FRAME].astype(np.float32)


class TestSpeechDetector:
    def test_frame_by_frame_as_over_a_whole_recording(self):
        signal = np.concatenate([np.zeros(32 * FRAME, np.float32), _speechlike()])
        whole = get_vad_model()(signal).ravel()
        detector = SpeechDetector()
        streamed = [detector(signal[i : i + FRAME]) for i in range(0, len(signal), FRAME)]
        assert np.allclose(whole, streamed, atol=1e-6)

    def test_silence_is_not_speech(self):
        detector = SpeechDetector()
        assert max(detector(SILENCE) for _ in range(20)) < 0.1


class FakeStream:
    """Stands in for sounddevice's InputStream, playing a recording into
    the callback as fast as it is read."""

    recording = np.zeros(0, dtype=np.float32)

    def __init__(self, callback, **kwargs):
        self.callback = callback
        self.blocksize = kwargs["blocksize"]

    def __enter__(self):
        for i in range(0, len(self.recording) - self.blocksize + 1, self.blocksize):
            block = self.recording[i : i + self.blocksize].reshape(-1, 1)
            self.callback(block, self.blocksize, None, None)
        return self

    def __exit__(self, *exc):
        return False


class TestListen:
    def test_records_the_turn_and_no_more(self, monkeypatch):
        monkeypatch.setattr(audio_module, "SpeechDetector", ScriptedSpeech)
        FakeStream.recording = np.concatenate(
            [np.zeros(SAMPLE_RATE * 2), np.full(SAMPLE_RATE, 0.8), np.zeros(SAMPLE_RATE * 4)]
        ).astype(np.float32)
        monkeypatch.setattr(audio_module.sd, "InputStream", FakeStream)

        recorder = AudioRecorder(turn_end=Verdict(0.9))
        levels = []
        path = recorder.listen(on_level=levels.append)
        recorded, rate = sf.read(path)
        Path(path).unlink()
        # Half a second before the speech, the second of speech, the pause
        assert rate == SAMPLE_RATE
        assert abs(len(recorded) / rate - (PRE_SPEECH + 1.0 + 0.2)) < 0.1
        assert levels and all(0.0 <= level <= 1.0 for level in levels)

    def test_stops_when_told(self, monkeypatch):
        monkeypatch.setattr(audio_module, "SpeechDetector", ScriptedSpeech)
        FakeStream.recording = np.zeros(SAMPLE_RATE, dtype=np.float32)
        monkeypatch.setattr(audio_module.sd, "InputStream", FakeStream)
        assert AudioRecorder().listen(should_stop=lambda: True) is None


def _turn_file(seconds=1.0):
    """A turn listen() recorded: quiet, so the VAD takes none of it for speech."""
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(path, np.full(int(SAMPLE_RATE * seconds), 0.01, dtype=np.float32), SAMPLE_RATE)
    return path


class TestListenOn:
    def test_a_user_who_goes_on_gets_the_whole_turn(self, monkeypatch):
        monkeypatch.setattr(audio_module, "SpeechDetector", ScriptedSpeech)
        FakeStream.recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE, 0.8), np.zeros(SAMPLE_RATE * 4)]
        ).astype(np.float32)
        monkeypatch.setattr(audio_module.sd, "InputStream", FakeStream)
        earlier = _turn_file(1.0)
        went_on = []

        path = AudioRecorder(turn_end=Verdict(0.9)).listen_on(
            earlier, on_speech=lambda: went_on.append(True), until=lambda: False
        )
        whole, _ = sf.read(path)
        Path(path).unlink()
        Path(earlier).unlink()
        assert went_on == [True]
        # The earlier second, then the rest of the turn as listen() records it
        assert abs(len(whole) / SAMPLE_RATE - (1.0 + PRE_SPEECH + 1.0 + 0.2)) < 0.1
        assert np.allclose(whole[: SAMPLE_RATE], 0.01, atol=1e-4)

    def test_nothing_more_once_the_answer_is_there(self, monkeypatch):
        monkeypatch.setattr(audio_module, "SpeechDetector", ScriptedSpeech)
        FakeStream.recording = np.zeros(SAMPLE_RATE, dtype=np.float32)
        monkeypatch.setattr(audio_module.sd, "InputStream", FakeStream)
        earlier = _turn_file()
        went_on = []
        path = AudioRecorder().listen_on(
            earlier, on_speech=lambda: went_on.append(True), until=lambda: True
        )
        Path(earlier).unlink()
        assert path is None and went_on == []

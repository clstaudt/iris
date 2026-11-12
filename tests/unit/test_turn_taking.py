"""Unit tests for turn-taking: when the user has finished speaking."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from faster_whisper.vad import get_vad_model

from iris import audio as audio_module
from iris.audio import AudioRecorder
from iris.turns import FRAME, HOLD_BACKOFF, PRE_SPEECH, SAMPLE_RATE, SpeechDetector, TurnTracker

FRAME_SECONDS = FRAME / SAMPLE_RATE


def frames(seconds: float) -> int:
    return round(seconds / FRAME_SECONDS)


class ScriptedSpeech:
    """A VAD whose verdicts are scripted: certain to be speech while the frame
    is loud, sure while it is less loud, not speech while it is quiet. It
    counts the frames it was given, by any instance."""

    frames = 0

    def __call__(self, frame):
        ScriptedSpeech.frames += 1
        loudest = np.abs(frame).max()
        return 0.95 if loudest > 0.5 else 0.75 if loudest > 0.2 else 0.05

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

        def goes_on():
            went_on.append(True)
            return True

        path = AudioRecorder(turn_end=Verdict(0.9)).listen_on(
            earlier, on_speech=goes_on, until=lambda: False
        )
        whole, _ = sf.read(path)
        Path(path).unlink()
        Path(earlier).unlink()
        assert went_on == [True]
        # The earlier second, then the rest of the turn as listen() records it
        assert abs(len(whole) / SAMPLE_RATE - (1.0 + PRE_SPEECH + 1.0 + 0.2)) < 0.1
        assert np.allclose(whole[: SAMPLE_RATE], 0.01, atol=1e-4)

    def test_a_new_turn_leaves_the_earlier_one_out(self, monkeypatch):
        monkeypatch.setattr(audio_module, "SpeechDetector", ScriptedSpeech)
        FakeStream.recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE, 0.8), np.zeros(SAMPLE_RATE * 4)]
        ).astype(np.float32)
        monkeypatch.setattr(audio_module.sd, "InputStream", FakeStream)
        earlier = _turn_file(1.0)
        path = AudioRecorder(turn_end=Verdict(0.9)).listen_on(
            earlier, on_speech=lambda: False, until=lambda: False
        )
        turn, _ = sf.read(path)
        Path(path).unlink()
        Path(earlier).unlink()
        assert abs(len(turn) / SAMPLE_RATE - (PRE_SPEECH + 1.0 + 0.2)) < 0.1

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


class Voice:
    """The voice, as listen_on() holds and releases it, and in which frames."""

    def __init__(self):
        self.holds = 0
        self.releases = 0
        self.held_in: list[int] = []
        self.released_in: list[int] = []

    def hold(self):
        self.holds += 1
        self.held_in.append(ScriptedSpeech.frames)

    def release(self):
        self.releases += 1
        self.released_in.append(ScriptedSpeech.frames)


class TestOverTheVoice:
    """Speech over the voice holds it; in the quiet the user goes on, while
    the voice's echo dies away with it."""

    def _listen(
        self,
        monkeypatch,
        recording,
        answered_after=None,
        settled=True,
        cancelling=None,
        should_stop=None,
        speaking_until=None,
    ):
        """Listen over a voice that speaks throughout, or (speaking_until)
        until so many seconds into the recording."""
        monkeypatch.setattr(audio_module, "SpeechDetector", ScriptedSpeech)
        monkeypatch.setattr(ScriptedSpeech, "frames", 0)
        FakeStream.recording = recording.astype(np.float32)
        monkeypatch.setattr(audio_module.sd, "InputStream", FakeStream)
        recorder = AudioRecorder(turn_end=Verdict(0.9), echo=PassThrough(settled, cancelling))
        voice, spoke, frames = Voice(), [], [0]

        def until():
            frames[0] += 1
            return answered_after is not None and frames[0] > answered_after

        earlier = _turn_file()
        path = recorder.listen_on(
            earlier,
            on_speech=lambda: spoke.append(True) or False,
            until=until,
            should_stop=(lambda: should_stop(voice)) if should_stop else None,
            speaking=(lambda: True) if speaking_until is None else (lambda: ScriptedSpeech.frames < round(speaking_until / FRAME_SECONDS)),
            hold=voice.hold,
            release=voice.release,
        )
        Path(earlier).unlink()
        return path, voice, spoke

    def test_the_user_goes_on_in_the_quiet(self, monkeypatch):
        # Sure to be speech, not certain: it is the going on that tells
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 2, 0.3), np.zeros(SAMPLE_RATE * 4)]
        )
        path, voice, spoke = self._listen(monkeypatch, recording)
        turn, _ = sf.read(path)
        Path(path).unlink()
        assert (voice.holds, voice.releases, spoke) == (1, 0, [True])
        # All of what the user said, from just before it
        assert abs(len(turn) / SAMPLE_RATE - (PRE_SPEECH + 2.0 + 0.2)) < 0.15

    def test_a_stop_that_goes_on_a_moment_while_held_is_the_users(self, monkeypatch):
        # Certain speech, half a second: going on just past the voice's dying
        # away is enough
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE // 2, 0.8), np.zeros(SAMPLE_RATE * 4)]
        )
        path, voice, spoke = self._listen(monkeypatch, recording)
        Path(path).unlink()
        assert (voice.holds, voice.releases, spoke) == (1, 0, [True])

    def test_certain_speech_that_stops_with_the_voice_is_its_echo(self, monkeypatch):
        # An echo canceller that lets the voice through makes the voice
        # certain speech; it stops as the voice is held
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 3 // 10, 0.8), np.zeros(SAMPLE_RATE * 2)]
        )
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(2.5))
        assert path is None
        assert (voice.holds, voice.releases, spoke) == (1, 1, [])

    def test_listening_ended_during_a_hold_lets_the_voice_go_on(self, monkeypatch):
        # Muted, or the user starts typing, while the voice is held
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 2, 0.8), np.zeros(SAMPLE_RATE * 2)]
        )
        path, voice, spoke = self._listen(
            monkeypatch, recording, should_stop=lambda voice: voice.holds > 0
        )
        assert path is None
        assert (voice.holds, voice.releases, spoke) == (1, 1, [])

    def test_speech_heard_while_the_voice_may_not_be_held_is_kept(self, monkeypatch):
        # An echo burst while the canceller leaks starts the back-off; the
        # user begins at 1.9 s, is held once it is over, and goes on: the
        # turn begins with the user's first word
        recording = np.concatenate(
            [
                np.zeros(SAMPLE_RATE),
                np.full(SAMPLE_RATE * 3 // 10, 0.8),
                np.zeros(SAMPLE_RATE * 6 // 10),
                np.full(SAMPLE_RATE * 5 // 2, 0.8),
                np.zeros(SAMPLE_RATE * 4),
            ]
        )
        path, voice, spoke = self._listen(monkeypatch, recording, cancelling=False)
        turn, _ = sf.read(path)
        Path(path).unlink()
        assert (voice.holds, voice.releases, spoke) == (2, 1, [True])
        assert len(turn) / SAMPLE_RATE > 2.5

    def test_speech_that_goes_on_is_the_users_while_the_canceller_leaks(self, monkeypatch):
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 2, 0.8), np.zeros(SAMPLE_RATE * 4)]
        )
        path, voice, spoke = self._listen(monkeypatch, recording, cancelling=False)
        Path(path).unlink()
        assert (voice.holds, voice.releases, spoke) == (1, 0, [True])

    @pytest.mark.parametrize("cancelling", [False, True])
    def test_a_leaking_canceller_does_not_hold_the_voice_at_every_leak(self, monkeypatch, cancelling):
        # Two echo bursts a second apart: while the canceller leaks, the
        # second comes while the voice may not be held, and is over while the
        # voice speaks on, so it cannot be told from its echo
        burst = np.full(SAMPLE_RATE * 3 // 10, 0.8)
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), burst, np.zeros(SAMPLE_RATE * 7 // 10), burst, np.zeros(SAMPLE_RATE * 3)]
        )
        path, voice, spoke = self._listen(
            monkeypatch, recording, answered_after=frames(5.0), cancelling=cancelling
        )
        assert path is None
        holds = 2 if cancelling else 1
        assert (voice.holds, voice.releases, spoke) == (holds, holds, [])
        if cancelling:
            assert voice.held_in[1] - voice.released_in[0] < frames(0.5)

    def test_echo_held_again_soon_holds_the_voice_less_often(self, monkeypatch):
        # A canceller that takes out some of the voice, not enough: bursts of
        # its echo every 1.2 s. The second hold that heard only echo within
        # HOLD_AGAIN starts the back-off, whatever the canceller says
        burst = np.full(SAMPLE_RATE * 3 // 10, 0.8)
        gap = np.zeros(SAMPLE_RATE * 9 // 10)
        recording = np.concatenate([np.zeros(SAMPLE_RATE)] + [burst, gap] * 5 + [np.zeros(SAMPLE_RATE * 2)])
        path, voice, spoke = self._listen(
            monkeypatch, recording, answered_after=frames(8.0), cancelling=True
        )
        assert path is None and spoke == []
        assert voice.holds == voice.releases
        assert voice.holds < 5
        assert voice.held_in[2] - voice.released_in[1] >= frames(HOLD_BACKOFF)

    def test_echo_kept_while_the_voice_may_not_be_held_is_judged_when_it_ends(self, monkeypatch):
        # An echo burst starts the back-off; a second one comes during it, and
        # the voice falls quiet as it ends: it does not go on, so it is echo
        burst = np.full(SAMPLE_RATE * 3 // 10, 0.8)
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), burst, np.zeros(SAMPLE_RATE * 6 // 10), np.full(SAMPLE_RATE // 2, 0.8),
             np.zeros(SAMPLE_RATE * 3)]
        )
        path, voice, spoke = self._listen(
            monkeypatch, recording, answered_after=frames(5.0), cancelling=False, speaking_until=2.65
        )
        assert path is None
        assert (voice.holds, voice.releases, spoke) == (1, 1, [])

    def test_a_user_kept_while_the_voice_may_not_be_held_is_taken_when_it_ends(self, monkeypatch):
        # The same, but the user goes on after the voice fell quiet
        burst = np.full(SAMPLE_RATE * 3 // 10, 0.8)
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), burst, np.zeros(SAMPLE_RATE * 6 // 10), np.full(SAMPLE_RATE * 2, 0.8),
             np.zeros(SAMPLE_RATE * 4)]
        )
        path, voice, spoke = self._listen(monkeypatch, recording, cancelling=False, speaking_until=2.65)
        turn, _ = sf.read(path)
        Path(path).unlink()
        # Not held again: the voice was quiet by then
        assert (voice.holds, voice.releases, spoke) == (1, 1, [True])
        assert len(turn) / SAMPLE_RATE > 2.0

    def test_the_echo_dies_away_and_the_voice_goes_on(self, monkeypatch):
        # A quarter second of echo, sure to be speech, gone once the voice is held
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE // 4, 0.3), np.zeros(SAMPLE_RATE * 2)]
        )
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(2.5))
        assert path is None
        assert (voice.holds, voice.releases, spoke) == (1, 1, [])

    def test_not_listened_to_until_the_canceller_has_learned_the_room(self, monkeypatch):
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 2, 0.8), np.zeros(SAMPLE_RATE)]
        )
        path, voice, spoke = self._listen(
            monkeypatch, recording, answered_after=frames(3.5), settled=False
        )
        assert path is None
        assert (voice.holds, voice.releases, spoke) == (0, 0, [])


class PassThrough:
    """An echo canceller with nothing to cancel; cancelling says whether it
    measurably takes the voice out (None: it does not say)."""

    def __init__(self, settled=False, cancelling=None):
        self.settled = settled
        if cancelling is not None:
            self.cancelling = cancelling

    def clean(self, audio, captured):
        return audio

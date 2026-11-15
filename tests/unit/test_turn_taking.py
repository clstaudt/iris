"""Unit tests for turn-taking: when the user has finished speaking."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from faster_whisper.vad import get_vad_model

from iris import audio as audio_module
from iris.audio import AudioRecorder
from iris.echo import EchoLevels
from iris.turns import (
    FRAME,
    HOLD_TRIGGER,
    HOLD_TRIGGER_AFTER_ECHO,
    HOLD_TRIGGER_NEAR_ECHO,
    PRE_SPEECH,
    SAMPLE_RATE,
    SpeechDetector,
    TurnTracker,
)

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
        # For each release, whether the voice was to go on a little before
        # where it was cut
        self.rewound: list[bool] = []

    def hold(self):
        self.holds += 1
        self.held_in.append(ScriptedSpeech.frames)

    def release(self, rewind):
        self.releases += 1
        self.released_in.append(ScriptedSpeech.frames)
        self.rewound.append(rewind)


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
        echo=None,
        recorder=None,
    ):
        """Listen over a voice that speaks throughout, or (speaking_until)
        until so many seconds into the recording; echo is the canceller, one
        that cannot tell how loud the voice's echo is by default. A recorder
        given listens again (the next reply)."""
        monkeypatch.setattr(audio_module, "SpeechDetector", ScriptedSpeech)
        monkeypatch.setattr(ScriptedSpeech, "frames", 0)
        FakeStream.recording = recording.astype(np.float32)
        monkeypatch.setattr(audio_module.sd, "InputStream", FakeStream)
        echo = echo if echo is not None else PassThrough(settled, cancelling)
        if recorder is None:
            recorder = AudioRecorder(turn_end=Verdict(0.9), echo=echo)
        recorder.echo = echo
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
        # The first burst holds the voice; after a hold for echo the second
        # is too short to hold it again, and ends while the voice speaks on
        assert (voice.holds, voice.releases, spoke) == (1, 1, [])

    def test_echo_holds_the_voice_once_in_a_reply(self, monkeypatch):
        # A canceller that takes out some of the voice, not enough, and
        # cannot tell how loud its echo is: bursts of the echo every 1.2 s.
        # Each hold stops and restarts the voice, so after the first that
        # heard only echo, bursts like these no longer hold it, and it goes
        # on where it was cut: the voice goes on without stuttering
        burst = np.full(SAMPLE_RATE * 3 // 10, 0.8)
        gap = np.zeros(SAMPLE_RATE * 9 // 10)
        recording = np.concatenate([np.zeros(SAMPLE_RATE)] + [burst, gap] * 5 + [np.zeros(SAMPLE_RATE * 2)])
        path, voice, spoke = self._listen(
            monkeypatch, recording, answered_after=frames(8.0), cancelling=True
        )
        assert path is None and spoke == []
        assert (voice.holds, voice.releases, voice.rewound) == (1, 1, [False])
        # A canceller that can tell holds it not even once
        echo = Weighed(lambda t: (2.0, 1.0) if np.abs(_at(recording, t)).max() > 0.2 else (-20.0, -40.0))
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(8.0), echo=echo)
        assert path is None and spoke == []
        assert (voice.holds, voice.releases) == (0, 0)

    def test_after_two_holds_for_echo_the_voice_is_not_held_again(self, monkeypatch):
        # Echo that stops once the voice is held: a short burst holds it,
        # then stretches just past the longer speech a hold takes after echo.
        # The first stretch holds it a second time; after that, the voice
        # goes on to its end
        burst = np.full(SAMPLE_RATE * 3 // 10, 0.8)
        stretch = np.full(SAMPLE_RATE * 11 // 10, 0.8)
        gap = np.zeros(SAMPLE_RATE)
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), burst, gap, stretch, gap, stretch, gap, stretch, np.zeros(SAMPLE_RATE * 2)]
        )
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(10.0), cancelling=True)
        assert spoke == [] and path is None
        assert (voice.holds, voice.releases) == (2, 2)

    def test_after_a_hold_for_echo_a_user_who_speaks_on_still_stops_the_voice(self, monkeypatch):
        # An echo burst holds the voice and is let go; then the user talks
        # over the voice, on and on: that holds it again and is their turn
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 3 // 10, 0.8), np.zeros(SAMPLE_RATE),
             np.full(SAMPLE_RATE * 3, 0.3), np.zeros(SAMPLE_RATE * 4)]
        )
        path, voice, spoke = self._listen(monkeypatch, recording, cancelling=True)
        assert path is not None
        Path(path).unlink()
        assert (voice.holds, voice.releases, spoke) == (2, 1, [True])

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

    def test_the_echo_dies_away_and_the_voice_goes_on_where_it_was_cut(self, monkeypatch):
        # A quarter second of sound sure to be speech, gone once the voice is
        # held, from a canceller that cannot tell how loud the echo is: held
        # for nothing, the voice goes on exactly where it was cut
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE // 4, 0.3), np.zeros(SAMPLE_RATE * 2)]
        )
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(2.5))
        assert path is None
        assert (voice.holds, voice.releases, spoke) == (1, 1, [])
        assert voice.rewound == [False]

    def test_a_back_channel_lets_the_voice_go_on_a_little_before_the_cut(self, monkeypatch):
        # Speech that goes on a little into the hold, too little for a turn:
        # the user spoke over the voice, and hears the cut word again
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 45 // 100, 0.3), np.zeros(SAMPLE_RATE * 2)]
        )
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(2.5))
        assert path is None
        assert (voice.holds, voice.releases, spoke) == (1, 1, [])
        assert voice.rewound == [True]

    def test_the_voices_own_echo_is_not_held(self, monkeypatch):
        # The start of a reply: the canceller finds the voice again and lets
        # it through, sure to be speech for half a second at a time, about as
        # loud as the voice's echo. The voice speaks on without a hold
        burst = np.full(SAMPLE_RATE // 2, 0.8)
        gap = np.zeros(SAMPLE_RATE * 7 // 10)
        recording = np.concatenate([np.zeros(SAMPLE_RATE)] + [burst, gap] * 4 + [np.zeros(SAMPLE_RATE * 2)])
        echo = Weighed(lambda t: (3.0, 1.5) if np.abs(_at(recording, t)).max() > 0.2 else (-20.0, -40.0))
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(6.5), echo=echo)
        assert path is None and spoke == []
        assert (voice.holds, voice.releases) == (0, 0)

    def test_what_the_canceller_leaves_of_the_voice_holds_nothing(self, monkeypatch):
        # Faint, far below the echo: the VAD is sure of it all the same
        recording = np.concatenate([np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 2, 0.3), np.zeros(SAMPLE_RATE * 2)])
        echo = Weighed(lambda t: (-2.0, -25.0) if np.abs(_at(recording, t)).max() > 0.2 else (-20.0, -40.0))
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(4.5), echo=echo)
        assert path is None and spoke == []
        assert (voice.holds, voice.releases) == (0, 0)

    def test_a_user_who_stands_out_above_the_echo_holds_the_voice_at_once(self, monkeypatch):
        recording = np.concatenate([np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 2, 0.3), np.zeros(SAMPLE_RATE * 4)])
        echo = Weighed(lambda t: (14.0, 13.0) if np.abs(_at(recording, t)).max() > 0.2 else (-20.0, -40.0))
        path, voice, spoke = self._listen(monkeypatch, recording, echo=echo)
        Path(path).unlink()
        assert (voice.holds, voice.releases, spoke) == (1, 0, [True])
        assert voice.held_in[0] - frames(1.0) <= frames(HOLD_TRIGGER) + 1

    def test_after_a_hold_for_echo_near_its_level_such_speech_needs_longer_in_later_replies(self, monkeypatch):
        # Echo let through for 0.7 s holds the voice once; in the next reply
        # the same does not, until a hold hears the user
        def burst(seconds):
            recording = np.concatenate([np.zeros(SAMPLE_RATE), np.full(int(SAMPLE_RATE * seconds), 0.8),
                                        np.zeros(SAMPLE_RATE * 2)])
            echo = Weighed(lambda t: (3.0, 1.5) if np.abs(_at(recording, t)).max() > 0.2 else (-20.0, -40.0))
            return recording, echo

        recorder = AudioRecorder(turn_end=Verdict(0.9))
        recording, echo = burst(0.7)
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(3.5), echo=echo,
                                          recorder=recorder)
        assert path is None and (voice.holds, voice.releases, voice.rewound) == (1, 1, [False])
        assert recorder.near_echo_needs == HOLD_TRIGGER_AFTER_ECHO
        recording, echo = burst(0.8)
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(3.5), echo=echo,
                                          recorder=recorder)
        assert path is None and (voice.holds, voice.releases) == (0, 0)
        # The user speaks on, about as loud as the echo: held once they have
        # gone on longer, and the bar is back where it was
        recording, echo = burst(2.5)
        path, voice, spoke = self._listen(monkeypatch, recording, echo=echo, recorder=recorder)
        Path(path).unlink()
        assert (voice.holds, voice.releases, spoke) == (1, 0, [True])
        assert recorder.near_echo_needs == HOLD_TRIGGER_NEAR_ECHO

    def test_a_user_as_loud_as_the_echo_holds_the_voice_once_they_go_on(self, monkeypatch):
        recording = np.concatenate([np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 2, 0.3), np.zeros(SAMPLE_RATE * 4)])
        echo = Weighed(lambda t: (3.0, 2.0) if np.abs(_at(recording, t)).max() > 0.2 else (-20.0, -40.0))
        path, voice, spoke = self._listen(monkeypatch, recording, echo=echo)
        Path(path).unlink()
        assert (voice.holds, voice.releases, spoke) == (1, 0, [True])
        assert abs(voice.held_in[0] - frames(1.0) - frames(HOLD_TRIGGER_NEAR_ECHO)) <= 1

    # Loud speakers: the user's voice reaches the microphone quieter than the
    # voice's echo. It does not stand out above the echo, but the canceller,
    # which left 45 dB less than the echo of the voice alone, leaves it

    def test_a_short_word_quieter_than_loud_speakers_holds_the_voice_and_is_the_users(self, monkeypatch):
        # "Stop" said once, 0.3 s, sure to be speech but not certain
        recording = np.concatenate([np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 3 // 10, 0.3),
                                    np.zeros(SAMPLE_RATE * 4)])
        echo = Weighed(lambda t: (-3.0, -12.0, -45.0) if np.abs(_at(recording, t)).max() > 0.2
                       else (-20.0, -45.0, -45.0))
        path, voice, spoke = self._listen(monkeypatch, recording, echo=echo)
        Path(path).unlink()
        assert (voice.holds, voice.releases, spoke) == (1, 0, [True])
        assert voice.held_in[0] - frames(1.0) <= frames(HOLD_TRIGGER) + 1

    def test_a_word_as_loud_as_a_quiet_moment_of_the_voice_holds_it_and_is_the_users(self, monkeypatch):
        # "Wait" said where the voice is quiet: about as loud as its echo
        # could be, and far above what the canceller left before
        recording = np.concatenate([np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 3 // 10, 0.3),
                                    np.zeros(SAMPLE_RATE * 4)])
        echo = Weighed(lambda t: (-3.0, -4.0, -45.0) if np.abs(_at(recording, t)).max() > 0.2
                       else (-20.0, -45.0, -45.0))
        path, voice, spoke = self._listen(monkeypatch, recording, echo=echo)
        Path(path).unlink()
        assert (voice.holds, voice.releases, spoke) == (1, 0, [True])
        assert voice.held_in[0] - frames(1.0) <= frames(HOLD_TRIGGER) + 1

    def test_a_word_out_of_the_cancellers_residue_holds_the_voice_after_a_hold_for_echo(self, monkeypatch):
        # A burst that stands out above the echo holds the voice, and stops
        # as it is held: echo, after which speech must go on longer to hold
        # the voice. Then the user says "Wait": held at once all the same
        before_word = [np.zeros(32 * FRAME), np.full(frames(HOLD_TRIGGER) * FRAME, 0.8), np.zeros(SAMPLE_RATE)]
        word_at = sum(len(piece) for piece in before_word) / SAMPLE_RATE
        recording = np.concatenate(before_word + [np.full(SAMPLE_RATE * 3 // 10, 0.3), np.zeros(SAMPLE_RATE * 4)])

        def levels(t):
            frame = np.abs(_at(recording, t)).max()
            if frame > 0.5:
                return 10.0, 9.0  # how much the canceller left before is not known
            return (-3.0, -12.0, -45.0) if frame > 0.2 else (-20.0, -45.0, -45.0)

        path, voice, spoke = self._listen(monkeypatch, recording, echo=Weighed(levels))
        Path(path).unlink()
        assert (voice.holds, voice.releases, voice.rewound, spoke) == (2, 1, [False], [True])
        assert voice.held_in[1] - frames(word_at) <= frames(HOLD_TRIGGER) + 1

    def test_the_voice_let_through_at_the_start_of_a_reply_is_not_held(self, monkeypatch):
        # The canceller finds the voice again and lets it through, about as
        # loud as its echo, and left as much just before: nothing stands out
        burst = np.full(SAMPLE_RATE // 2, 0.8)
        gap = np.zeros(SAMPLE_RATE * 7 // 10)
        recording = np.concatenate([np.zeros(SAMPLE_RATE)] + [burst, gap] * 4 + [np.zeros(SAMPLE_RATE * 2)])
        echo = Weighed(lambda t: (3.0, 1.5, -1.0) if np.abs(_at(recording, t)).max() > 0.2 else (-2.0, -3.0, -1.0))
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(6.5), echo=echo)
        assert path is None and spoke == []
        assert (voice.holds, voice.releases) == (0, 0)

    def test_what_the_canceller_leaves_later_in_a_reply_holds_nothing(self, monkeypatch):
        # Faint and sure to be speech, no more than the canceller left of the
        # voice before it
        recording = np.concatenate([np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 2, 0.3), np.zeros(SAMPLE_RATE * 2)])
        echo = Weighed(lambda t: (-2.0, -36.0, -45.0) if np.abs(_at(recording, t)).max() > 0.2
                       else (-20.0, -45.0, -45.0))
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(4.5), echo=echo)
        assert path is None and spoke == []
        assert (voice.holds, voice.releases) == (0, 0)

    def test_the_voice_let_through_for_a_moment_later_in_a_reply_is_not_held(self, monkeypatch):
        # The canceller, which was cancelling, lets the voice through for a
        # moment, about as loud as its echo: far above what it left before,
        # but the voice, not the user
        recording = np.concatenate([np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 3 // 10, 0.8),
                                    np.zeros(SAMPLE_RATE * 2)])
        echo = Weighed(lambda t: (3.0, 0.0, -45.0) if np.abs(_at(recording, t)).max() > 0.2
                       else (-20.0, -45.0, -45.0))
        path, voice, spoke = self._listen(monkeypatch, recording, answered_after=frames(2.5), echo=echo)
        assert path is None and spoke == []
        assert (voice.holds, voice.releases) == (0, 0)

    def test_a_short_word_over_the_echo_is_the_users_though_it_ends_before_the_echo_dies_away(self, monkeypatch):
        # "Stop" said once, louder than the voice's echo could be: over
        # before the voice has died away in the hold, and the user's all the
        # same
        recording = np.concatenate([np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 3 // 10, 0.3),
                                    np.zeros(SAMPLE_RATE * 4)])
        echo = Weighed(lambda t: (12.0, 11.0) if np.abs(_at(recording, t)).max() > 0.2 else (-20.0, -40.0))
        path, voice, spoke = self._listen(monkeypatch, recording, echo=echo)
        Path(path).unlink()
        assert (voice.holds, voice.releases, spoke) == (1, 0, [True])

    def test_not_listened_to_until_the_canceller_has_learned_the_room(self, monkeypatch):
        recording = np.concatenate(
            [np.zeros(SAMPLE_RATE), np.full(SAMPLE_RATE * 2, 0.8), np.zeros(SAMPLE_RATE)]
        )
        path, voice, spoke = self._listen(
            monkeypatch, recording, answered_after=frames(3.5), settled=False
        )
        assert path is None
        assert (voice.holds, voice.releases, spoke) == (0, 0, [])


def _at(recording, t):
    """The frame of the recording t seconds into it."""
    i = max(0, int(t * SAMPLE_RATE))
    return recording[i : i + FRAME] if i < len(recording) else np.zeros(1)


class PassThrough:
    """An echo canceller with nothing to cancel, that cannot tell how loud
    the voice's echo is; cancelling says whether it measurably takes the
    voice out (None: it does not say)."""

    def __init__(self, settled=False, cancelling=None):
        self.settled = settled
        if cancelling is not None:
            self.cancelling = cancelling

    def clean(self, audio, captured):
        return audio


class Weighed(PassThrough):
    """A settled canceller that tells how loud what was heard is against the
    voice's echo: levels(t) gives (heard, left) in dB at t seconds into the
    recording, or (heard, left, before) with what it left of the voice just
    before."""

    def __init__(self, levels):
        super().__init__(settled=True, cancelling=True)
        self.levels = levels
        self._first = None

    def clean(self, audio, captured):
        if self._first is None:
            self._first = captured
        return audio

    def against_echo(self, start, end):
        return EchoLevels(*self.levels((end - FRAME / SAMPLE_RATE) - self._first))

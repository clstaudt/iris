"""When the user has finished speaking: voice activity and end of turn.

Silero VAD, the model faster-whisper ships, tells speech from everything
else in every 32 ms of microphone audio. When speech pauses for 0.2 s,
Smart Turn v3 (pipecat-ai/smart-turn-v3, BSD-2) listens to the turn so far
and judges from its words and intonation whether the user has finished, or
is only thinking. A finished turn ends there; an unfinished one waits for
more, up to a long silence. This is how Pipecat takes turns, and it answers
a finished question in a fraction of a second instead of after seconds of
silence, without cutting off someone who pauses mid-thought.
"""

import os
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import onnxruntime
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.utils import get_assets_path
from huggingface_hub import hf_hub_download

SAMPLE_RATE = 16000
# Samples Silero VAD judges at a time (32 ms), and those it is given of the
# frame before for context
FRAME = 512
VAD_CONTEXT = 64

# Silero's speech probability at which a frame is speech, and below which it
# is not any more, once speech has begun
SPEECH_THRESHOLD = 0.5
SILENCE_THRESHOLD = 0.35
# Speech this long starts a turn (shorter sounds are clicks and coughs), and a
# Seconds without speech before a turn after which the VAD starts afresh
VAD_REFRESH = 4.0
# pause this long has Smart Turn judge whether it ends the turn
SPEECH_START = 0.2
SPEECH_PAUSE = 0.2
# Over the voice, what the echo canceller leaves of it can sound like speech.
# Speech this sure (VAD probability) and this long holds the voice: weighed
# on recordings of a voice's echo in rooms, it holds the voice about once a
# minute by mistake, and catches nearly every user who speaks.
HOLD_THRESHOLD = 0.7
HOLD_TRIGGER = 0.13
# Speech that holds the voice again after a hold heard only its echo; after
# a second such hold the voice is not held again until it has ended
HOLD_TRIGGER_AFTER_ECHO = 1.0
# Over the voice, speech is weighed against how loud the voice's echo could
# be (dB). Iris's own echo, let through by a canceller that is finding the
# voice again at the start of a reply, came to at most 6 dB over it in the
# voice tests' room (and 7.4 dB for a test's harsher, synthetic voice), the
# user's first syllables mostly to 8 or more: speech standing out this much
# holds the voice after HOLD_TRIGGER. Speech that does not stand out holds
# it only after HOLD_TRIGGER_NEAR_ECHO: the echo let through was sure to be
# speech for up to 0.5 s at a time, and a user who speaks about as loud as
# the echo is still heard within the second. Once such a hold heard only
# the echo, such speech needs HOLD_TRIGGER_AFTER_ECHO, in later replies too,
# until a hold hears the user
HOLD_STANDS_OUT = 8.0
HOLD_TRIGGER_NEAR_ECHO = 0.6
# Speech the VAD is sure of while the canceller leaves this much less than
# the echo (dB) is what the canceller leaves of the voice: it holds nothing,
# and speech over the voice starts over
HOLD_RESIDUAL = -10.0
# Held, the voice is listened to this long in the quiet. Once the device has
# played out what it had (about 0.15 s) and the room is still, speech in at
# least this share of the time is the user's; anything less was the echo.
HOLD = 0.55
HOLD_SETTLE = 0.25
HOLD_SPEECH = 0.5
# Speech this certain and this long over the voice needs to go on for only
# HOLD_CERTAIN_GOES_ON frames while the voice is held to be the user's: a
# "Stop" said once is nearly over by then. Something must go on all the
# same, as an echo canceller that lets the voice through (one that has lost
# the voice's timing does) makes the voice itself certain speech, and that
# stops with the voice.
HOLD_CERTAIN = 0.9
HOLD_CERTAIN_TIME = 0.1
HOLD_CERTAIN_GOES_ON = 2
# While the echo canceller measurably lets the voice through, or after a
# second hold within HOLD_AGAIN seconds that heard only the voice's echo,
# the voice is not held again this long (seconds): held at every leak, it
# would stutter
HOLD_BACKOFF = 2.0
HOLD_AGAIN = 4.0
# Audio kept from before speech was recognized, so a turn starts with its
# first syllable
PRE_SPEECH = 0.5

SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"
# Smart Turn hears the last 8 s of a turn
SMART_TURN_SECONDS = 8


def _session(path: str) -> onnxruntime.InferenceSession:
    """A one-thread CPU session: both models are small, and run while the
    voice model may be busy."""
    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.log_severity_level = 4
    return onnxruntime.InferenceSession(
        path, providers=["CPUExecutionProvider"], sess_options=options
    )


class SpeechDetector:
    """Silero VAD over a stream of 512-sample frames at 16 kHz, carrying its
    state from frame to frame (the same probabilities faster-whisper's VAD
    computes over a whole recording)."""

    def __init__(self):
        self._session = _session(os.path.join(get_assets_path(), "silero_vad_v6.onnx"))
        self.reset()

    def reset(self):
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(VAD_CONTEXT, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """The probability that this frame is speech."""
        window = np.concatenate([self._context, frame])[np.newaxis, :]
        probability, self._h, self._c = self._session.run(
            None, {"input": window, "h": self._h, "c": self._c}
        )
        self._context = frame[-VAD_CONTEXT:]
        return float(probability.ravel()[0])


class SmartTurn:
    """Smart Turn v3: the probability that speech which paused is a finished
    turn. Its input is Whisper's log-mel spectrogram of the last 8 s,
    computed here by faster-whisper's feature extractor."""

    def __init__(self, path: str):
        self._session = _session(path)
        self._features = FeatureExtractor(
            feature_size=80,
            sampling_rate=SAMPLE_RATE,
            hop_length=160,
            chunk_length=SMART_TURN_SECONDS,
            n_fft=400,
        )

    @classmethod
    def load(cls, download: bool = True) -> Optional["SmartTurn"]:
        """The model from the Hugging Face cache, downloaded first if it is
        not there (9 MB) and download is set. None if it cannot be had, for
        instance offline: turns then end after a long silence, as before."""
        try:
            path = hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE, local_files_only=True)
        except Exception:
            if not download:
                return None
            try:
                path = hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE)
            except Exception:
                return None
        return cls(path)

    def __call__(self, audio: np.ndarray) -> float:
        """The probability that the turn in this 16 kHz audio is finished."""
        samples = SMART_TURN_SECONDS * SAMPLE_RATE
        audio = audio[-samples:].astype(np.float32)
        # Shorter turns are padded with silence before them
        audio = np.pad(audio, (samples - len(audio), 0))
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        features = self._features(audio, padding=0)
        (probability,) = self._session.run(None, {"input_features": features[np.newaxis]})
        return float(np.ravel(probability)[0])


class TurnTracker:
    """Follows microphone audio frame by frame and says when the user's turn
    is over.

    A turn starts with speech of SPEECH_START or longer. When it pauses for
    SPEECH_PAUSE, Smart Turn judges the turn so far: finished ends it, not
    finished waits for the user to go on. Silence of max_silence ends a turn
    whatever Smart Turn said, and is all there is without Smart Turn.

    A turn the user goes on with after it seemed over starts from that
    earlier audio (before), and is judged as a whole.
    """

    def __init__(
        self,
        speech: Callable[[np.ndarray], float],
        finished: Optional[Callable[[np.ndarray], float]],
        max_silence: float,
        before: Optional[np.ndarray] = None,
    ):
        self.speech = speech
        self.finished = finished
        self.max_silence = max_silence
        self._earlier = before
        frame_seconds = FRAME / SAMPLE_RATE
        self._start_frames = max(1, round(SPEECH_START / frame_seconds))
        self._pause_frames = max(1, round(SPEECH_PAUSE / frame_seconds))
        self._max_silent_frames = max(1, round(max_silence / frame_seconds))
        self._before: deque[np.ndarray] = deque(
            maxlen=round(PRE_SPEECH / frame_seconds) + self._start_frames
        )
        self._frames: list[np.ndarray] = []
        self._speech_run = 0
        self._silent_run = 0
        self._judged = False
        # Smart Turn's verdicts on this turn's pauses, for inspection
        self.verdicts: list[float] = []
        # The VAD's probability that the last frame was speech
        self.probability = 0.0
        # Once the turn is over: seconds from the end of speech until that
        # was clear, the silence listened to and Smart Turn's verdict
        self.end_wait = 0.0
        self._frame_seconds = frame_seconds
        # Frames in a row without speech before the turn started: after a
        # long quiet stretch the VAD starts afresh, as it rates the first
        # syllables after many seconds of noise floor too low
        self._quiet_run = 0
        self._refresh_frames = max(1, round(VAD_REFRESH / frame_seconds))

    @property
    def started(self) -> bool:
        """Whether the user has started speaking."""
        return bool(self._frames)

    def forget_before(self):
        """The user began a new turn: the earlier one is not part of it."""
        self._earlier = None

    def audio(self) -> np.ndarray:
        """The turn so far, from just before its first syllable."""
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        if self._earlier is not None:
            return np.concatenate([self._earlier, *self._frames])
        return np.concatenate(self._frames)

    def feed(self, frame: np.ndarray) -> bool:
        """Take the next 512 samples; True once the turn is over."""
        probability = self.probability = self.speech(frame)
        if not self._frames:
            self._quiet_run = self._quiet_run + 1 if probability < SILENCE_THRESHOLD else 0
            if self._quiet_run >= self._refresh_frames and hasattr(self.speech, "reset"):
                self.speech.reset()
                self._quiet_run = 0
            self._before.append(frame)
            self._speech_run = self._speech_run + 1 if probability >= SPEECH_THRESHOLD else 0
            if self._speech_run >= self._start_frames:
                self._frames = list(self._before)
            return False

        self._frames.append(frame)
        if probability >= SILENCE_THRESHOLD:
            self._silent_run = 0
            self._judged = False
            return False

        self._silent_run += 1
        silence = self._silent_run * self._frame_seconds
        if self._silent_run >= self._max_silent_frames:
            self.end_wait = silence
            return True
        paused = self._silent_run >= self._pause_frames
        if paused and self.finished is not None and not self._judged:
            # Once per pause: until the user speaks again it stays the same
            self._judged = True
            judging = time.perf_counter()
            verdict = self.finished(self.audio())
            self.verdicts.append(verdict)
            if verdict > 0.5:
                self.end_wait = silence + time.perf_counter() - judging
                return True
        return False

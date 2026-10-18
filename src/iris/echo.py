"""Echo cancellation: Iris's own voice taken out of what the microphone hears,
so the user can interrupt it by speaking.

WebRTC's echo canceller (AEC3, through LiveKit's audio processing module)
learns how the voice from the speaker reaches the microphone and subtracts
it. It needs to be told what was played, and when: the player registers
each clip as it starts, and the microphone's audio is cleaned in 10 ms steps
against what was played at the time it was captured.
"""

import atexit
import threading
from collections import deque
from math import gcd

import numpy as np
from scipy.signal import resample_poly

try:
    from livekit.rtc import AudioFrame, AudioProcessingModule

    HAS_ECHO_CANCELLATION = True
except Exception:  # the native library is missing on this platform
    HAS_ECHO_CANCELLATION = False

RATE = 16000
STEP = RATE // 100  # the canceller works in 10 ms steps
# Each step of microphone audio is paired with what the speaker played this
# much later: the canceller then always hears the voice before its echo (it
# finds the delay itself), however far off the devices' reported latencies
LEAD = 0.05
# Played audio kept for the canceller (seconds)
KEEP = 20.0
# Voice heard by the canceller before it has learned the room (seconds)
SETTLE = 3.0


def _int16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class EchoCanceller:
    """Removes what the player played from what the microphone captured."""

    def __init__(self):
        self._apm = AudioProcessingModule(echo_cancellation=True, high_pass_filter=True)
        # LiveKit shuts its library down at exit before this would be freed
        # (and complains); atexit runs this first, as it was registered later
        atexit.register(self._apm._ffi_handle.dispose)
        self._lock = threading.Lock()
        # What was played: (when it started leaving the speaker, its samples
        # at 16 kHz), on the time.monotonic() clock
        self._played: deque[tuple[float, np.ndarray]] = deque()
        # How much of the voice it has cancelled so far (seconds)
        self._voice_heard = 0.0

    @property
    def settled(self) -> bool:
        """Whether it has heard enough of the voice to have learned the room."""
        return self._voice_heard >= SETTLE

    def played(self, start: float, audio: np.ndarray, sample_rate: int):
        """Audio that leaves the speaker from start (time.monotonic()) on."""
        mono = audio.mean(axis=1) if audio.ndim > 1 else audio
        if sample_rate != RATE:
            common = gcd(RATE, sample_rate)
            mono = resample_poly(mono, RATE // common, sample_rate // common)
        with self._lock:
            self._played.append((start, mono.astype(np.float32)))
            # What ended long ago is forgotten
            while self._played and self._ends(self._played[0]) < start - KEEP:
                self._played.popleft()

    @staticmethod
    def _ends(played: tuple[float, np.ndarray]) -> float:
        start, samples = played
        return start + len(samples) / RATE

    def cut(self, at: float):
        """Playback stopped at this time: what would have followed never played."""
        with self._lock:
            self._played = deque(
                (start, samples[: max(0, int((at - start) * RATE))])
                for start, samples in self._played
                if start < at
            )

    def _reference(self, start: float, length: int) -> np.ndarray:
        """What was played from start on, silence where nothing was."""
        reference = np.zeros(length, dtype=np.float32)
        with self._lock:
            for played, samples in self._played:
                offset = int(round((played - start) * RATE))
                begin, end = max(0, offset), min(length, offset + len(samples))
                if begin < end:
                    reference[begin:end] = samples[begin - offset : end - offset]
        return reference

    def clean(self, audio: np.ndarray, captured: float) -> np.ndarray:
        """Microphone audio at 16 kHz (a whole number of 10 ms steps), first
        captured at this time, without what the speaker played into it."""
        cleaned = np.empty_like(audio)
        for i in range(0, len(audio) - STEP + 1, STEP):
            at = captured + i / RATE
            reference = self._reference(at + LEAD, STEP)
            if reference.any():
                self._voice_heard += STEP / RATE
            render = AudioFrame(_int16(reference), RATE, 1, STEP)
            self._apm.process_reverse_stream(render)
            self._apm.set_stream_delay_ms(int(LEAD * 1000))
            capture = AudioFrame(_int16(audio[i : i + STEP]), RATE, 1, STEP)
            self._apm.process_stream(capture)
            cleaned[i : i + STEP] = np.frombuffer(capture.data, dtype=np.int16) / 32768.0
        return cleaned

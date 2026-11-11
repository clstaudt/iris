"""A virtual acoustic space: microphones and speakers without audio devices.

A Space holds endpoints, each a microphone and a speaker as one device would
have them (an assistant's laptop, say). Every microphone hears, in real
time:

- what scripted voices say into the space,
- its own speaker coming back as an echo, delayed and quieter,
- the other endpoints' speakers, as from across a table,
- a faint noise floor (a real microphone is never silent).

Every speaker records each clip it is given: when it started being heard and
when it was cut off. So a test knows exactly when a user stopped speaking,
when the assistant's first word was heard and when its voice fell silent.

An endpoint is an audio device as Iris takes it (iris.audio.SoundDevice):
input_stream, query_devices, play, stop and playing. Any assistant whose
audio goes through such an object can be tested in a space.

Times are on the time.monotonic() clock throughout.
"""

import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

# The microphones' rate
RATE = 16000
# What a virtual speaker reports as its output latency
OUTPUT_LATENCY = 0.05
# Above this a sample counts as sound
AUDIBLE = 0.01


def to_rate(audio: np.ndarray, rate: int, target: int = RATE) -> np.ndarray:
    """Mono float32 audio at the target rate."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate == target:
        return audio.copy()
    divisor = int(np.gcd(rate, target))
    return resample_poly(audio, target // divisor, rate // divisor).astype(np.float32)


def sound_span(audio: np.ndarray, rate: int = RATE) -> tuple[float, float]:
    """Seconds from the start to the first and past the last audible sample."""
    loud = np.flatnonzero(np.abs(audio) > AUDIBLE)
    if len(loud) == 0:
        return 0.0, 0.0
    return loud[0] / rate, (loud[-1] + 1) / rate


def merge(intervals: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    """Intervals joined where they overlap or lie less than gap apart."""
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


@dataclass
class Clip:
    """One clip a speaker was given."""

    audio: np.ndarray  # at RATE
    started: float  # when it started being heard
    stopped: Optional[float] = None  # when it was cut off, if it was

    @property
    def end(self) -> float:
        """When it stopped being heard."""
        natural = self.started + len(self.audio) / RATE
        return natural if self.stopped is None else min(natural, self.stopped)

    def heard(self) -> np.ndarray:
        """The part that was heard."""
        return self.audio[: max(0, int(round((self.end - self.started) * RATE)))]

    def sounds(self, gap: float = 0.25) -> list[tuple[float, float]]:
        """The stretches in which it was audible."""
        heard = self.heard()
        step = RATE // 50  # 20 ms windows
        windows = len(heard) // step
        if windows == 0:
            return []
        loud = np.abs(heard[: windows * step]).reshape(windows, step).max(axis=1) > AUDIBLE
        spans = [
            (float(self.started + i * step / RATE), float(self.started + (i + 1) * step / RATE))
            for i in np.flatnonzero(loud)
        ]
        return merge(spans, gap)


@dataclass
class Utterance:
    """Something a scripted voice said into the space."""

    text: str
    start: float  # its first sample
    speech_start: float  # its first audible sample
    speech_end: float  # past its last audible sample
    audio: np.ndarray = field(repr=False)
    # The endpoints that hear it, with how loud; None for all at full level
    heard_by: Optional[dict] = None


class Speaker:
    """An endpoint's speaker: records every clip instead of playing it."""

    def __init__(self):
        self.clips: list[Clip] = []
        self._lock = threading.Lock()

    def play(self, data, samplerate, **options) -> float:
        now = time.monotonic()
        with self._lock:
            self._cut(now)
            self.clips.append(Clip(to_rate(data, int(samplerate)), now + OUTPUT_LATENCY))
        return OUTPUT_LATENCY

    def stop(self):
        with self._lock:
            self._cut(time.monotonic())

    def _cut(self, now: float):
        last = self.clips[-1] if self.clips else None
        if last is not None and last.stopped is None and last.end > now:
            last.stopped = max(now, last.started)

    def playing(self) -> bool:
        """Whether the last clip still has audio to play."""
        with self._lock:
            return bool(self.clips) and self.clips[-1].end > time.monotonic()

    def clips_between(self, start: float, end: float = float("inf")) -> list[Clip]:
        with self._lock:
            return [c for c in self.clips if c.end > start and c.started < end]

    def sounds(self, start: float = 0.0, end: float = float("inf"), gap: float = 0.25):
        """The stretches between two moments in which the speaker was audible."""
        spans = []
        for clip in self.clips_between(start, end):
            spans += [(max(a, start), min(b, end)) for a, b in clip.sounds() if b > start and a < end]
        return merge(spans, gap)

    def audible(self, at: Optional[float] = None) -> bool:
        """Whether the speaker is heard at this moment (now by default)."""
        at = time.monotonic() if at is None else at
        return any(a <= at < b for a, b in self.sounds(at - 0.3, at + 0.3, gap=0.0))

    def first_sound_after(self, moment: float) -> Optional[float]:
        sounds = self.sounds(moment)
        return sounds[0][0] if sounds else None

    def last_sound(self, start: float = 0.0, end: Optional[float] = None) -> Optional[float]:
        end = time.monotonic() if end is None else end
        sounds = self.sounds(start, end)
        return sounds[-1][1] if sounds else None

    def sound_heard(self, start: float, end: float) -> float:
        """Seconds of audible sound between two moments."""
        return sum(b - a for a, b in self.sounds(start, end, gap=0.0))

    def silent_after(self, moment: float, span: float = 0.3, limit: float = 10.0) -> Optional[float]:
        """The first moment after this one from which the speaker stayed
        silent for a while; None if it did not within the limit. Waits for
        the moments to pass."""
        t = moment
        while t < moment + limit:
            wait = t + span - time.monotonic()
            if wait > 0:
                time.sleep(wait + 0.02)
            if self.sound_heard(t, t + span) == 0:
                return t
            t += 0.02
        return None

    def recording(self, start: float, end: float) -> np.ndarray:
        """What the speaker played between two moments."""
        out = np.zeros(max(0, int((end - start) * RATE)), dtype=np.float32)
        for clip in self.clips_between(start, end):
            _mix(out, start, clip.heard(), clip.started)
        return out


def _mix(block: np.ndarray, block_start: float, audio: np.ndarray, audio_start: float, gain: float = 1.0):
    """Add audio that starts at audio_start into a block starting at block_start."""
    offset = int(round((block_start - audio_start) * RATE))
    a, b = max(0, offset), min(len(audio), offset + len(block))
    if b > a:
        block[a - offset : b - offset] += audio[a:b] * gain


class Endpoint:
    """A microphone and a speaker in the space, as one audio device."""

    def __init__(self, space: "Space", name: str):
        self.space = space
        self.name = name
        self.speaker = Speaker()
        # What the microphone delivered: (first sample's time, block)
        self._heard: list[tuple[float, np.ndarray]] = []

    # The audio device, as iris.audio.SoundDevice --------------------------

    def input_stream(self, samplerate=RATE, blocksize=512, callback=None, **options):
        return _Microphone(self, int(samplerate), int(blocksize or 512), callback)

    def query_devices(self, device=None, kind=None):
        info = {
            "name": f"Virtual microphone ({self.name})",
            "max_input_channels": 1,
            "max_output_channels": 2,
            "default_samplerate": RATE,
            "index": 0,
        }
        return dict(info) if kind is not None or device is not None else [info]

    def play(self, audio, sample_rate, **options) -> float:
        return self.speaker.play(audio, sample_rate, **options)

    def stop(self):
        self.speaker.stop()

    def playing(self) -> bool:
        return self.speaker.playing()

    # Listening back ---------------------------------------------------------

    def recording(self, start: float, end: float) -> np.ndarray:
        """What the microphone heard between two moments."""
        out = np.zeros(max(0, int((end - start) * RATE)), dtype=np.float32)
        for at, block in list(self._heard):
            _mix(out, start, block, at)
        return out


class Space:
    """Endpoints and voices in one acoustic space."""

    def __init__(
        self,
        echo_gain: float = 0.3,
        echo_delay: float = 0.02,
        across_gain: float = 0.6,
        across_delay: float = 0.01,
        noise: float = 0.0005,
        seed: int = 7,
    ):
        """
        Args:
            echo_gain: How loud an endpoint's own speaker comes back into its
                microphone (0.3: a laptop's speakers at a normal volume)
            echo_delay: Seconds from a speaker to its own microphone
            across_gain: How loud another endpoint's speaker is heard
            across_delay: Seconds from another endpoint's speaker
            noise: Level of every microphone's noise floor
        """
        self.echo_gain = echo_gain
        self.echo_delay = echo_delay
        self.across_gain = across_gain
        self.across_delay = across_delay
        self.noise = noise
        self.endpoints: list[Endpoint] = []
        self.utterances: list[Utterance] = []
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)

    def endpoint(self, name: str) -> Endpoint:
        endpoint = Endpoint(self, name)
        self.endpoints.append(endpoint)
        return endpoint

    def say(self, text: str, audio: np.ndarray, delay: float = 0.0,
            heard_by: Optional[dict] = None) -> Utterance:
        """A scripted voice says something, after a delay and after what it
        is still saying. heard_by maps endpoints to how loud they hear it;
        None is all of them, at full level."""
        start = time.monotonic() + delay
        with self._lock:
            if self.utterances:
                last = self.utterances[-1]
                start = max(start, last.start + len(last.audio) / RATE)
            first, past_last = sound_span(audio)
            utterance = Utterance(text, start, start + first, start + past_last, audio, heard_by)
            self.utterances.append(utterance)
        return utterance

    def voice_speaking(self, at: Optional[float] = None) -> bool:
        """Whether a scripted voice is speaking at this moment."""
        at = time.monotonic() if at is None else at
        with self._lock:
            return any(u.speech_start <= at < u.speech_end for u in self.utterances)

    def _block(self, listener: Endpoint, start: float, frames: int) -> np.ndarray:
        """What a microphone hears from a moment on, for so many samples."""
        block = (self._rng.standard_normal(frames) * self.noise).astype(np.float32)
        end = start + frames / RATE
        with self._lock:
            utterances = [u for u in self.utterances if u.start < end]
        for u in utterances:
            gain = 1.0 if u.heard_by is None else u.heard_by.get(listener, 0.0)
            if gain:
                _mix(block, start, u.audio, u.start, gain)
        for endpoint in self.endpoints:
            own = endpoint is listener
            gain = self.echo_gain if own else self.across_gain
            delay = self.echo_delay if own else self.across_delay
            if not gain:
                continue
            for clip in endpoint.speaker.clips_between(start - delay, end):
                _mix(block, start, clip.heard(), clip.started + delay, gain)
        return block


class _Microphone:
    """An input stream delivering what an endpoint's microphone hears, in
    real time, through the callback as sounddevice does."""

    def __init__(self, endpoint: Endpoint, rate: int, blocksize: int, callback):
        if rate != RATE:
            raise ValueError(f"The virtual microphone records at {RATE} Hz")
        self.endpoint = endpoint
        self.blocksize = blocksize
        self.callback = callback
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    def _run(self):
        seconds = self.blocksize / RATE
        start = time.monotonic()
        n = 0
        while not self._stop.is_set():
            block_start = start + n * seconds
            wait = block_start + seconds - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            block = self.endpoint.space._block(self.endpoint, block_start, self.blocksize)
            self.endpoint._heard.append((block_start, block))
            info = SimpleNamespace(currentTime=time.monotonic(), inputBufferAdcTime=block_start)
            if self.callback is not None:
                self.callback(block.reshape(-1, 1), self.blocksize, info, None)
            n += 1

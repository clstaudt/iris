"""A virtual acoustic space: microphones and speakers without audio devices.

A Space holds endpoints, each a microphone and a speaker as one device would
have them (an assistant's laptop, say). Every microphone hears, in real
time:

- what scripted voices say into the space,
- its own speaker coming back as an echo, delayed and quieter,
- the other endpoints' speakers, as from across a table,
- a faint noise floor (a real microphone is never silent),
- sounds placed in the room (Space.add_source): a fan, music, a door.

Every speaker records each clip it is given: when it started being heard and
when it was cut off. So a test knows exactly when a user stopped speaking,
when the assistant's first word was heard and when its voice fell silent.

As a real device's, a speaker's sound does not come out exactly when the
device says: a stream started after a silence (a device waking up) comes
out later than reported, by an amount that differs each time
(Space(wake_latency=...)). An echo canceller then finds the voice in a new
place at the start of every reply, and lets it through while it adapts.

An endpoint is an audio device as Iris takes it (iris.audio.SoundDevice):
input_stream, query_devices, play, stop and playing. Any assistant whose
audio goes through such an object can be tested in a space.

Times are on the time.monotonic() clock throughout.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

# The microphones' rate
RATE = 16000
# What a virtual speaker reports as its output latency
OUTPUT_LATENCY = 0.05
# A speaker whose stream starts after this many seconds of silence wakes up:
# its sound comes out up to Space.wake_latency later than it reports, by an
# amount drawn anew; within a reply its clips follow one another, and keep
# the latency the first of them had
WAKE_IDLE = 2.0
# How much later than reported a waking speaker's sound may come out
# (seconds), unless the space is told otherwise: a Bluetooth speaker's
# latency moves by as much from one stream to the next
WAKE_LATENCY = 0.3
# Above this a sample counts as sound
AUDIBLE = 0.01
# Seconds of what a microphone heard kept for listening back, so a long
# session does not hold all of it
HEARD_KEPT = 900.0
# How far back a speaker looks for its last sound when not told where from
LOOKBACK = 10.0


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


def room(rt60: float, drr: float = 0.0, rate: int = RATE, seed: int = 3) -> np.ndarray:
    """A room's impulse response, synthesized: the direct sound, then a
    tail of reflections dying away exponentially, 60 dB in rt60 seconds.
    drr is the direct-to-reverberant ratio in dB: positive near the
    microphone, negative across the room."""
    length = max(1, int(rt60 * 1.2 * rate))
    t = np.arange(length) / rate
    tail = np.random.default_rng(seed).standard_normal(length) * np.exp(-3 * np.log(10) * t / rt60)
    tail[: max(1, int(0.003 * rate))] = 0.0  # the first reflection comes after the direct sound
    energy = float(np.sum(tail**2))
    if energy > 0:
        tail *= np.sqrt(10 ** (-drr / 10) / energy)
    tail[0] = 1.0
    return tail.astype(np.float32)


def in_room(audio: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Audio as heard in a room, its reverberation ringing on after it."""
    if len(audio) == 0:
        return audio
    return fftconvolve(audio, response).astype(np.float32)


def spans_of(audio: np.ndarray, start: float = 0.0, gap: float = 0.25) -> list[tuple[float, float]]:
    """The stretches of audio (from start, in seconds) in which it is audible."""
    step = RATE // 50  # 20 ms windows
    windows = len(audio) // step
    if windows == 0:
        return []
    loud = np.abs(audio[: windows * step]).reshape(windows, step).max(axis=1) > AUDIBLE
    spans = [(float(start + i * step / RATE), float(start + (i + 1) * step / RATE)) for i in np.flatnonzero(loud)]
    return merge(spans, gap)


def subtract(intervals: list[tuple[float, float]], taken: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of the intervals outside those taken."""
    out = []
    for start, end in intervals:
        pieces = [(start, end)]
        for a, b in taken:
            pieces = [p for s, e in pieces for p in ((s, min(e, a)), (max(s, b), e)) if p[1] > p[0]]
        out += pieces
    return out


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
    # What sounds() found, and the cut it was found for
    _sounds: dict = field(default_factory=dict, repr=False)
    # Its echo in the room, and the cut it was made for
    _echo: Optional[tuple] = field(default=None, repr=False)

    @property
    def end(self) -> float:
        """When it stopped being heard."""
        natural = self.started + len(self.audio) / RATE
        return natural if self.stopped is None else min(natural, self.stopped)

    def heard(self) -> np.ndarray:
        """The part that was heard."""
        return self.audio[: max(0, int(round((self.end - self.started) * RATE)))]

    def sounds(self, gap: float = 0.25) -> list[tuple[float, float]]:
        """The stretches in which it was audible (kept until it is cut)."""
        key = (self.stopped, gap)
        if key not in self._sounds:
            self._sounds = {key: spans_of(self.heard(), self.started, gap)}
        return self._sounds[key]


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
    _spans: Optional[list] = field(default=None, repr=False)

    def spans(self) -> list[tuple[float, float]]:
        """The stretches in which the voice is speaking (pauses between
        words of less than 0.15 s included)."""
        if self._spans is None:
            self._spans = spans_of(self.audio, self.start, gap=0.15)
        return self._spans

    def heard_at(self, endpoint) -> bool:
        return self.heard_by is None or bool(self.heard_by.get(endpoint))


@dataclass
class Source:
    """A sound in the room besides the scripted voices: a fan, music, a door.

    It is mixed into every microphone from its start, once or over and over,
    until it ends or is removed. It is not a voice: it neither waits for
    voices nor counts as one speaking."""

    space: "Space"
    audio: np.ndarray = field(repr=False)
    at: float
    gain: float = 1.0
    loop: bool = False
    name: str = ""
    removed: Optional[float] = None

    @property
    def end(self) -> float:
        natural = float("inf") if self.loop else self.at + len(self.audio) / RATE
        return natural if self.removed is None else min(natural, self.removed)

    def remove(self):
        """Silence it from now on."""
        if self.removed is None:
            self.removed = time.monotonic()

    def mix(self, block: np.ndarray, start: float):
        """Add what of it is heard in a block starting at this moment."""
        if not len(self.audio) or not self.gain:
            return
        frames = len(block)
        index = int(round((start - self.at) * RATE)) + np.arange(frames)
        valid = index >= 0
        if self.loop:
            index = index % len(self.audio)
        else:
            valid &= index < len(self.audio)
        if self.removed is not None:
            valid &= start + np.arange(frames) / RATE < self.removed
        if valid.any():
            block[valid] += self.audio[index[valid]] * self.gain


@dataclass
class Room:
    """The character of a speaker's sound in its own microphone, beyond a
    delayed and quieter copy, in one: the speaker's coloration and
    distortion, the room's reverberation, and a delay the audio device does
    not report. Space(room=Room()) sounds like laptop speakers next to their
    microphone, which an echo canceller removes less easily than the copy.
    """

    reverb: float = 0.4  # seconds for the reverberation to fall by 60 dB (RT60)
    drr: float = 9.0  # direct-to-reverberant ratio, in dB
    drive: Optional[float] = 2.0  # how hard loud passages distort (None for none)
    low_cut: Optional[float] = 180.0  # Hz below which a small speaker gives little
    extra_delay: float = 0.04  # seconds later than the device reports


class Speaker:
    """An endpoint's speaker: records every clip instead of playing it.

    As a sounddevice stream does, it plays in blocks (blocksize): the last
    one is filled up with silence, and the stream is playing until that has
    been played too.

    It always reports OUTPUT_LATENCY. Woken after WAKE_IDLE seconds of
    silence, it plays up to wake_latency later than that, by an amount the
    random generator draws (none if wake_latency is 0); clips that follow
    one another keep it.
    """

    def __init__(self, wake_latency: float = 0.0, rng: Optional[np.random.Generator] = None):
        self.clips: list[Clip] = []
        self._lock = threading.Lock()
        # Until when the last clip's stream plays (its last block played)
        self._stream_end = 0.0
        self.wake_latency = wake_latency
        self._rng = rng if rng is not None else np.random.default_rng()
        # How much later than reported its sound comes out, since it woke
        self.late = 0.0
        # When its last stream stopped playing (None: it never played)
        self._quiet_since: Optional[float] = None

    def play(self, data, samplerate, blocksize=None, **options) -> float:
        now = time.monotonic()
        with self._lock:
            self._cut(now)
            if self._quiet_since is None or now - self._quiet_since >= WAKE_IDLE:
                self.late = float(self._rng.uniform(0.0, self.wake_latency)) if self.wake_latency else 0.0
            clip = Clip(to_rate(data, int(samplerate)), now + OUTPUT_LATENCY + self.late)
            self.clips.append(clip)
            frames = len(data)
            if blocksize:
                frames = -(-frames // int(blocksize)) * int(blocksize)
            self._stream_end = clip.started + frames / int(samplerate)
            self._quiet_since = self._stream_end
        return OUTPUT_LATENCY

    def stop(self):
        with self._lock:
            self._cut(time.monotonic())

    def _cut(self, now: float):
        last = self.clips[-1] if self.clips else None
        if last is not None and last.stopped is None and last.end > now:
            last.stopped = max(now, last.started)
        if self._quiet_since is not None:
            self._quiet_since = min(self._quiet_since, max(now, self.clips[-1].started))

    def playing(self) -> bool:
        """Whether the last clip's stream is still playing, its last block
        filled up with silence (a stopped one is not, even before it was
        heard)."""
        with self._lock:
            last = self.clips[-1] if self.clips else None
            return last is not None and last.stopped is None and self._stream_end > time.monotonic()

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

    def last_sound(self, start: Optional[float] = None, end: Optional[float] = None) -> Optional[float]:
        """When the last sound ended; looked for in the last few seconds
        unless told where from."""
        end = time.monotonic() if end is None else end
        sounds = self.sounds(end - LOOKBACK if start is None else start, end)
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

    def __init__(self, space: "Space", name: str, rng: Optional[np.random.Generator] = None):
        self.space = space
        self.name = name
        self.speaker = Speaker(space.wake_latency, rng)
        # What the microphone delivered, its last HEARD_KEPT seconds:
        # (first sample's time, block)
        self._heard: deque = deque()
        # When input streams were open: [opened, closed or None]
        self._open: list[list] = []
        self._open_lock = threading.Lock()

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
        """What the microphone heard between two moments (within the last
        HEARD_KEPT seconds)."""
        out = np.zeros(max(0, int((end - start) * RATE)), dtype=np.float32)
        for at, block in list(self._heard):
            _mix(out, start, block, at)
        return out

    def _heard_block(self, at: float, block: np.ndarray):
        self._heard.append((at, block))
        while self._heard and self._heard[0][0] < at - HEARD_KEPT:
            self._heard.popleft()

    def _opened(self) -> list:
        span = [time.monotonic(), None]
        with self._open_lock:
            self._open.append(span)
        return span

    def open_spans(self, start: float = 0.0, end: Optional[float] = None) -> list[tuple[float, float]]:
        """When an input stream was open between two moments."""
        end = time.monotonic() if end is None else end
        with self._open_lock:
            spans = [(a, end if b is None else b) for a, b in self._open]
        return merge([(max(a, start), min(b, end)) for a, b in spans if b > start and a < end])

    def deaf_spans(self, start: float, end: float) -> list[tuple[float, float]]:
        """Speech of the scripted voices this microphone hears, between two
        moments, while no input stream was open: what nobody listened to."""
        speech = []
        for utterance in self.space.utterances_between(start, end):
            if utterance.heard_at(self):
                speech += [(max(a, start), min(b, end)) for a, b in utterance.spans() if b > start and a < end]
        return subtract(merge(speech), self.open_spans(start, end))

    def deaf_seconds(self, start: float, end: float) -> float:
        return round(sum(b - a for a, b in self.deaf_spans(start, end)), 3)


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
        echo_room: Optional[float] = None,
        echo_drive: Optional[float] = None,
        echo_drr: float = 3.0,
        echo_low_cut: Optional[float] = None,
        echo_late: float = 0.0,
        room: Optional[Room] = None,
        wake_latency: float = WAKE_LATENCY,
    ):
        """
        Args:
            echo_gain: How loud an endpoint's own speaker comes back into its
                microphone (0.3: a laptop's speakers at a normal volume)
            echo_delay: Seconds from a speaker to its own microphone
            across_gain: How loud another endpoint's speaker is heard
            across_delay: Seconds from another endpoint's speaker
            noise: Level of every microphone's noise floor
            echo_room: The echo rings in a room with this RT60 (seconds),
                on past the moment its clip is cut
            echo_drive: The speaker distorts (tanh) at this drive, as a small
                speaker played loud does
            echo_drr: The echo room's direct-to-reverberant ratio, in dB
            echo_low_cut: Below this frequency (Hz) the speaker gives little,
                as small speakers do; None for a full range
            echo_late: Seconds the echo arrives later than the audio device
                reports, on top of echo_delay
            room: A room to take all of the echo's character from at once
                (overrides echo_room, echo_drive, echo_drr, echo_low_cut and
                echo_late)
            wake_latency: A speaker woken from a silence plays up to this
                many seconds later than it reports, by an amount that
                differs each time (see Speaker); 0 for exactly when it
                reports
        """
        self.echo_gain = echo_gain
        self.echo_delay = echo_delay
        self.across_gain = across_gain
        self.across_delay = across_delay
        self.noise = noise
        self.wake_latency = wake_latency
        self._seed = seed
        self.endpoints: list[Endpoint] = []
        self.utterances: list[Utterance] = []
        self.sources: list[Source] = []
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)
        if room is not None:
            echo_room, echo_drive, echo_drr = room.reverb, room.drive, room.drr
            echo_low_cut, echo_late = room.low_cut, room.extra_delay
        self.echo_drr = echo_drr
        self.echo_late = echo_late
        self.echo_low_cut = echo_low_cut
        self._low_cut = butter(2, echo_low_cut, "highpass", fs=RATE, output="sos") if echo_low_cut else None
        self.set_echo(echo_gain, echo_room, echo_drive)

    def set_echo(self, gain: Optional[float] = None, room_rt60: Optional[float] = None,
                 drive: Optional[float] = None):
        """How the speakers come back into their own microphones from now on."""
        if gain is not None:
            self.echo_gain = gain
        self.echo_room = room_rt60
        self.echo_drive = drive
        self._echo_response = room(room_rt60, self.echo_drr) if room_rt60 else None

    def echo_of(self, clip: Clip) -> np.ndarray:
        """What of a clip comes back into its microphone, before echo_gain:
        coloured below echo_low_cut, distorted at echo_drive and ringing in
        the echo room. Made once per
        cut, as the clip is played."""
        if not self.echo_drive and self._echo_response is None and self._low_cut is None:
            return clip.heard()
        key = (clip.stopped, self.echo_drive, self.echo_room)
        if clip._echo is None or clip._echo[0] != key:
            audio = clip.heard()
            if self._low_cut is not None and len(audio):
                audio = sosfilt(self._low_cut, audio).astype(np.float32)
            if self.echo_drive:
                audio = (np.tanh(self.echo_drive * audio) / np.tanh(self.echo_drive)).astype(np.float32)
            if self._echo_response is not None:
                audio = in_room(audio, self._echo_response)
            clip._echo = (key, audio)
        return clip._echo[1]

    def add_source(self, audio: np.ndarray, at: Optional[float] = None, gain: float = 1.0,
                   loop: bool = False, name: str = "") -> Source:
        """A sound in the room from a moment on (now by default)."""
        source = Source(self, np.asarray(audio, dtype=np.float32), time.monotonic() if at is None else at,
                        gain, loop, name)
        with self._lock:
            self.sources.append(source)
        return source

    def remove_sources(self):
        """Silence every sound placed in the room."""
        with self._lock:
            sources, self.sources = self.sources, []
        for source in sources:
            source.remove()

    def utterances_between(self, start: float, end: float) -> list[Utterance]:
        with self._lock:
            return [u for u in self.utterances if u.start < end and u.start + len(u.audio) / RATE > start]

    def endpoint(self, name: str) -> Endpoint:
        # Each speaker draws its latencies from a generator of its own, so
        # they do not depend on how much noise the microphones drew
        endpoint = Endpoint(self, name, np.random.default_rng([self._seed, len(self.endpoints)]))
        self.endpoints.append(endpoint)
        return endpoint

    def say(self, text: str, audio: np.ndarray, delay: float = 0.0,
            heard_by: Optional[dict] = None, queued: bool = True) -> Utterance:
        """A scripted voice says something, after a delay and after what it
        is still saying (queued; another voice, not queued, speaks over it).
        heard_by maps endpoints to how loud they hear it; None is all of
        them, at full level."""
        start = time.monotonic() + delay
        with self._lock:
            if self.utterances and queued:
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
            utterances = [u for u in self.utterances if u.start < end and u.start + len(u.audio) / RATE > start]
            sources = [s for s in self.sources if s.at < end and s.end > start]
        for u in utterances:
            gain = 1.0 if u.heard_by is None else u.heard_by.get(listener, 0.0)
            if gain:
                _mix(block, start, u.audio, u.start, gain)
        for source in sources:
            source.mix(block, start)
        ringing = len(self._echo_response) / RATE if self._echo_response is not None else 0.0
        for endpoint in self.endpoints:
            own = endpoint is listener
            gain = self.echo_gain if own else self.across_gain
            delay = self.echo_delay + self.echo_late if own else self.across_delay
            if not gain:
                continue
            for clip in endpoint.speaker.clips_between(start - delay - (ringing if own else 0.0), end):
                audio = self.echo_of(clip) if own else clip.heard()
                _mix(block, start, audio, clip.started + delay, gain)
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
        self._span = self.endpoint._opened()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        self._span[1] = time.monotonic()

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
            self.endpoint._heard_block(block_start, block)
            info = SimpleNamespace(currentTime=time.monotonic(), inputBufferAdcTime=block_start)
            if self.callback is not None:
                self.callback(block.reshape(-1, 1), self.blocksize, info, None)
            n += 1

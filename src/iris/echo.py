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
from dataclasses import dataclass
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
# How much quieter the voice must come out of the canceller than it went in
# (dB) for it to count as cancelling. A working canceller takes out 20 dB
# or more; one that has lost the voice's timing takes out little, and the
# voice comes through as speech (listening then holds the voice less often)
CANCELLING = 10.0
# It is judged over the last CANCELLING_WINDOWS stretches of WINDOW seconds
# of voice, all of which must show it but the CANCELLING_SPARED worst: the
# voice that leaks through does so in some stretches only, and the user who
# speaks over the voice (which no canceller takes out) is in a few of them
WINDOW = 0.25
CANCELLING_WINDOWS = 12
CANCELLING_SPARED = 2
# Below this level (RMS) of what was played, the voice counts as silent
PLAYING = 0.01
# A step of the microphone counts towards it while the voice played within
# this long before it was captured (its way through the room and the devices)
HEARD_WITHIN = 0.1
# Where the voice is not heard this much louder than the room's own noise
# (in energy), there is nothing to cancel: a window of it does not count.
# With headphones no window counts, and the canceller counts as cancelling.
ABOVE_NOISE = 4.0
# The room's noise is the quietest WINDOW the microphone heard among the
# last NOISE_WINDOWS (20 s): there is silence between sentences and turns.
# (Not the microphone while nothing played: a voice that comes out later
# than the canceller is told is heard then.)
NOISE_WINDOWS = 80
# What the microphone hears at a moment holds the voice's echo of what was
# played up to ECHO_WITHIN seconds before (a device that plays later than it
# says, the room's reverberation) and up to LEAD after (one that plays
# early). When the voice really comes out is not known, so its echo is
# taken to be as loud as the loudest of that, through the room's coupling
ECHO_WITHIN = 0.5
# The coupling (how loud the microphone hears the voice, against what was
# played) is the median over the last COUPLING_WINDOWS windows of voice
# heard over the room's noise: the user speaking over the voice is in a few
COUPLING_WINDOWS = 40
# Seconds of the microphone's levels kept, for what was heard to be weighed
# against the voice's echo (against_echo)
LEVELS_KEPT = 3.0
# What the canceller left of the voice just before a stretch: in the last
# RESIDUE_STEPS steps before it in which the microphone heard the voice
# (within VOICE_HEARD dB of its echo, and over the room's noise), within
# RESIDUE_WINDOW seconds before it (a pause between sentences has none),
# and up to RESIDUE_GAP before it (where the user's first syllable may be,
# before the VAD is sure of it). Fewer than RESIDUE_FEWEST such steps tell
# nothing (the voice has not reached the microphone yet)
RESIDUE_WINDOW = 2.5
RESIDUE_GAP = 0.15
VOICE_HEARD = -15.0
RESIDUE_STEPS = 50
RESIDUE_FEWEST = 20


@dataclass(frozen=True)
class EchoLevels:
    """What the microphone heard over a stretch, against the voice's echo
    as loud as it could have been then (dB): all of it (heard), and what the
    canceller left of it (left); and what it left of the voice it heard
    just before (before; None if it heard too little of it then).

    Speech louder than the echo is the user's. Speech quieter than the echo
    can be told by what the canceller leaves: of the voice alone, a working
    canceller leaves 30 dB or more below the echo, the user it leaves as
    loud as they are. So speech it leaves far above what it left just
    before is the user's, however loud the voice, unless it is about as
    loud as the echo: a canceller that loses the voice, for a moment or for
    longer, lets it through so."""

    heard: float
    left: float
    before: float | None = None


def _db(energy: float, reference: float) -> float:
    return 10.0 * float(np.log10(max(energy, 1e-12) / reference))


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
        # How much quieter the voice came out, by window (dB), and the energy
        # of the microphone before and after cancelling in the window so far
        self._suppressions: deque[float] = deque(maxlen=CANCELLING_WINDOWS)
        self._window = [0.0, 0.0, 0]
        # The microphone's energy by window, and in the window so far
        self._levels: deque[float] = deque(maxlen=NOISE_WINDOWS)
        self._level = [0.0, 0]
        # The energy of each step of what was played lately: (when it was
        # played, energy)
        self._recent: deque[tuple[float, float]] = deque()
        # The coupling by window of voice, and the energy heard and the most
        # the echo could have had in the window so far
        self._couplings: deque[float] = deque(maxlen=COUPLING_WINDOWS)
        self._coupling = [0.0, 0.0, 0]
        # By step, lately: (when it was captured, its energy, the energy the
        # canceller left of it, the most its echo could have had)
        self._steps: deque[tuple[float, float, float, float]] = deque()

    @property
    def settled(self) -> bool:
        """Whether it has heard enough of the voice to have learned the room."""
        return self._voice_heard >= SETTLE

    @property
    def suppression(self) -> float | None:
        """How much quieter the voice came out than it went in, lately, in
        all stretches but the CANCELLING_SPARED worst (dB); None before any
        voice was heard."""
        if not self._suppressions:
            return None
        ordered = sorted(self._suppressions)
        return ordered[min(CANCELLING_SPARED, len(ordered) - 1)]

    @property
    def coupling(self) -> float | None:
        """How loud the microphone hears the voice: its energy against the
        energy of the loudest of what was played within ECHO_WITHIN before
        (the median over recent windows of voice); None before the voice was
        heard over the room's noise."""
        if not self._couplings:
            return None
        ordered = sorted(self._couplings)
        return ordered[len(ordered) // 2]

    def against_echo(self, start: float, end: float) -> EchoLevels | None:
        """What the microphone heard between two moments (time.monotonic(),
        within the last LEVELS_KEPT seconds), against the voice's echo as
        loud as it could have been then. None if nothing was played then,
        or before the coupling is known."""
        coupling = self.coupling
        if coupling is None:
            return None
        heard = left = bound = 0.0
        # The steps just before in which the voice was heard: (what the
        # canceller left, the most its echo could have had)
        noise = ABOVE_NOISE * min(self._levels) if self._levels else 0.0
        voice = 10.0 ** (VOICE_HEARD / 10.0)
        voiced: list[tuple[float, float]] = []
        for at, energy, cleaned, most in self._steps:
            if start <= at < end:
                heard, left, bound = heard + energy, left + cleaned, bound + most
            elif (start - RESIDUE_WINDOW <= at < start - RESIDUE_GAP and most >= PLAYING**2
                  and energy >= voice * coupling * most and energy > noise):
                voiced.append((cleaned, most))
        echo = coupling * bound
        if bound < PLAYING**2 or echo <= 0.0:
            return None
        before = None
        if len(voiced) >= RESIDUE_FEWEST:
            residue, most = (sum(x) for x in zip(*voiced[-RESIDUE_STEPS:]))
            before = _db(residue, coupling * most)
        return EchoLevels(_db(heard, echo), _db(left, echo), before)

    @property
    def cancelling(self) -> bool:
        """Whether it is taking the voice out, as measured: settled, and the
        voice lately came out at least CANCELLING dB quieter, or was not
        heard over the room's noise at all."""
        suppression = self.suppression
        return self.settled and (suppression is None or suppression >= CANCELLING)

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
            self._listen_to_the_room(audio[i : i + STEP])
            self._weigh(at, audio[i : i + STEP], cleaned[i : i + STEP], reference)
            # What was played reaches the microphone a little later
            lately = self._reference(at - HEARD_WITHIN, int(HEARD_WITHIN * RATE) + STEP)
            if float(np.sqrt(np.mean(lately**2))) >= PLAYING:
                self._measure(audio[i : i + STEP], cleaned[i : i + STEP])
        return cleaned

    def _listen_to_the_room(self, heard: np.ndarray):
        """Count a step of the microphone towards the room's noise."""
        level = self._level
        level[0] += float(np.mean(heard**2))
        level[1] += 1
        if level[1] * STEP >= WINDOW * RATE:
            self._levels.append(level[0] / level[1])
            self._level = [0.0, 0]

    def _weigh(self, at: float, heard: np.ndarray, cleaned: np.ndarray, reference: np.ndarray):
        """Keep a step's levels, and the most its echo could have had: the
        loudest step of what was played from ECHO_WITHIN before it to LEAD
        after (reference is what is played LEAD after it)."""
        recent = self._recent
        recent.append((at + LEAD, float(np.mean(reference**2))))
        while recent and recent[0][0] < at - ECHO_WITHIN:
            recent.popleft()
        most = max(played for _, played in recent)
        energy = float(np.mean(heard**2))
        steps = self._steps
        steps.append((at, energy, float(np.mean(cleaned**2)), most))
        while steps and steps[0][0] < at - LEVELS_KEPT:
            steps.popleft()
        if most < PLAYING**2:
            return
        window = self._coupling
        window[0] += energy
        window[1] += most
        window[2] += 1
        if window[2] * STEP >= WINDOW * RATE:
            noise = min(self._levels) if self._levels else 0.0
            if window[0] / window[2] >= ABOVE_NOISE * noise:
                self._couplings.append(window[0] / window[1])
            self._coupling = [0.0, 0.0, 0]

    def _measure(self, heard: np.ndarray, cleaned: np.ndarray):
        """Count a step of the voice towards how much it is taken out."""
        window = self._window
        window[0] += float(np.mean(heard**2))
        window[1] += float(np.mean(cleaned**2))
        window[2] += 1
        if window[2] * STEP >= WINDOW * RATE:
            noise = min(self._levels) if self._levels else 0.0
            if window[0] > 0.0 and window[0] / window[2] >= ABOVE_NOISE * noise:
                self._suppressions.append(10.0 * float(np.log10(window[0] / max(window[1], 1e-12))))
            self._window = [0.0, 0.0, 0]

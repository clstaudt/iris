"""Audio recording, playback, and filters for voice assistant."""

import ctypes
import logging
import queue
import sys
import tempfile
import threading
import time
import traceback
import warnings
import weakref
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf

from iris.echo import STEP, EchoCanceller
from iris.turns import (
    FRAME,
    HOLD,
    HOLD_AGAIN,
    HOLD_TRIGGER_AFTER_ECHO,
    HOLD_BACKOFF,
    HOLD_CERTAIN,
    HOLD_CERTAIN_GOES_ON,
    HOLD_CERTAIN_TIME,
    HOLD_RESIDUAL,
    HOLD_SETTLE,
    HOLD_SPEECH,
    HOLD_STANDS_OUT,
    HOLD_THRESHOLD,
    HOLD_TRIGGER,
    HOLD_TRIGGER_NEAR_ECHO,
    SAMPLE_RATE,
    SPEECH_THRESHOLD,
    SmartTurn,
    SpeechDetector,
    TurnTracker,
)

# Each hold of the voice, what it was judged to be, and where the voice went
# on: the app writes it to a file of its own (iris.rpc_server)
voice_log = logging.getLogger("iris.voice")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


# ============================================================================
# Audio Filters
# ============================================================================


class AudioFilter(ABC):
    """Base class for audio filters."""

    @abstractmethod
    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply the filter to audio data.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Filtered audio data
        """
        pass


class PhoneFilter(AudioFilter):
    """
    Simulates telephone/phone audio quality.
    Bandpass 300-3400 Hz (typical phone bandwidth) with slight distortion.
    """

    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply phone filter effect."""
        # Bandpass filter: 300-3400 Hz (standard telephone bandwidth)
        sos = signal.butter(4, [300, 3400], btype="band", fs=sample_rate, output="sos")
        audio = signal.sosfilt(sos, audio)

        # Add slight clipping for characteristic phone distortion
        audio = np.clip(audio, -0.8, 0.8)

        # Add very subtle noise for realism
        noise = np.random.normal(0, 0.001, len(audio))
        audio = audio + noise

        return audio


class RadioFilter(AudioFilter):
    """
    Simulates radio/walkie-talkie audio quality.
    Bandpass 300-5000 Hz with compression and light static.
    """

    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply radio filter effect."""
        # Bandpass filter: 300-5000 Hz (radio bandwidth)
        sos = signal.butter(4, [300, 5000], btype="band", fs=sample_rate, output="sos")
        audio = signal.sosfilt(sos, audio)

        # Apply compression using tanh (soft clipping)
        audio = np.tanh(audio * 2.0) * 0.5

        # Add light static noise
        noise = np.random.normal(0, 0.004, len(audio))
        audio = audio + noise

        return audio


class IntercomFilter(AudioFilter):
    """
    Simulates intercom/PA system audio quality.
    Bandpass 400-4000 Hz with moderate compression.
    Good for spaceship computer or announcement system effects.
    """

    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply intercom filter effect."""
        # Bandpass filter: 400-4000 Hz (intercom bandwidth)
        sos = signal.butter(3, [400, 4000], btype="band", fs=sample_rate, output="sos")
        audio = signal.sosfilt(sos, audio)

        # Apply moderate compression
        audio = np.tanh(audio * 1.5) * 0.6

        # Very subtle background hum (like electrical interference)
        t = np.arange(len(audio)) / sample_rate
        hum = np.sin(2 * np.pi * 60 * t) * 0.001  # 60 Hz hum
        audio = audio + hum

        return audio


class RoboticFilter(AudioFilter):
    """
    Simulates robotic/synthetic voice.
    Slight pitch modulation and metallic resonance.
    """

    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply robotic filter effect."""
        # High-pass filter to emphasize higher frequencies
        sos = signal.butter(2, 200, btype="high", fs=sample_rate, output="sos")
        audio = signal.sosfilt(sos, audio)

        # Add subtle ring modulation for metallic quality
        t = np.arange(len(audio)) / sample_rate
        modulation = 1 + 0.1 * np.sin(2 * np.pi * 100 * t)
        audio = audio * modulation

        # Slight compression
        audio = np.tanh(audio * 1.8) * 0.55

        return audio


# Registry of available filters
AUDIO_FILTERS = {
    "phone": PhoneFilter(),
    "radio": RadioFilter(),
    "intercom": IntercomFilter(),
    "robotic": RoboticFilter(),
}


def get_filter(filter_name: str) -> AudioFilter | None:
    """
    Get an audio filter by name.

    Args:
        filter_name: Name of the filter ('phone', 'radio', 'intercom', 'robotic', etc.)

    Returns:
        AudioFilter instance, or None if filter not found
    """
    return AUDIO_FILTERS.get(filter_name.lower())


# AudioPlayer.play() uses the player's own filter unless given another
OWN_FILTER = object()


# ============================================================================
# Audio Recording and Playback
# ============================================================================


class SpeakerError(Exception):
    """The speaker could not be opened, even with the devices read anew."""


class MicrophoneError(Exception):
    """The microphone stopped delivering sound and could not be opened again."""


class _PropertyAddress(ctypes.Structure):
    """CoreAudio's AudioObjectPropertyAddress."""

    _fields_ = [
        ("selector", ctypes.c_uint32),
        ("scope", ctypes.c_uint32),
        ("element", ctypes.c_uint32),
    ]


def _four_cc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


# On macOS, CoreAudio tells which devices are the system's defaults
_core_audio = None
if sys.platform == "darwin":
    try:
        _core_audio = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        _core_audio.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_PropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        _core_audio.AudioObjectGetPropertyData.restype = ctypes.c_int32
    except (OSError, AttributeError):
        _core_audio = None

# kAudioObjectSystemObject, and the properties of its default input and
# output devices (in the global scope, of the main element)
_SYSTEM_OBJECT = 1
_DEFAULT_DEVICES = (_four_cc("dIn "), _four_cc("dOut"))
_SCOPE_GLOBAL = _four_cc("glob")


def system_defaults() -> tuple[int, ...] | None:
    """The system's default input and output devices, as CoreAudio numbers
    them; None where that cannot be told.

    Only macOS needs it: PortAudio's CoreAudio part keeps the defaults it
    found when it was initialized, so it goes on opening a device the user
    switched away from (AirPods connecting), or one that is gone (AirPods
    disconnecting). The default device PortAudio opens elsewhere, the sound
    server's on Linux and the sound mapper on Windows, follows the system's
    choice by itself.
    """
    if _core_audio is None:
        return None
    found = []
    for selector in _DEFAULT_DEVICES:
        address = _PropertyAddress(selector, _SCOPE_GLOBAL, 0)
        device = ctypes.c_uint32(0)
        size = ctypes.c_uint32(ctypes.sizeof(device))
        status = _core_audio.AudioObjectGetPropertyData(
            _SYSTEM_OBJECT, ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(device)
        )
        if status != 0:
            return None
        found.append(device.value)
    return tuple(found)


# Closing a stream takes a moment (0.1 s on macOS): the streams of clips
# played to their end are closed while the clip playing now still has at
# least this long to play (seconds), so its end is not noticed late
CLOSE_WITHIN = 0.5


class _Clip:
    """A clip played on an output stream of its own, as sounddevice.play()
    plays one: block by block, the last block filled up with silence, the
    stream ending by itself after it.

    Its callback learns from PortAudio when the clip's last sample is due at
    the speaker: the clip has played then, and the next one can start at
    once, on a stream of its own, while this one plays out the silence after
    it alone. Waiting for that silence, up to a block (1.4 s at 24 kHz), and
    closing the stream before the next could open, was a pause of up to 1.5 s
    between every two pieces of a reply. Its stream is not cut short: should
    the speaker play later than it says (Bluetooth), nothing of the clip is
    lost.
    """

    def __init__(self, audio: np.ndarray, sample_rate: int, **options):
        data = np.asarray(audio, dtype=np.float32)
        self.data = data.reshape(len(data), -1)
        self.sample_rate = sample_rate
        # Frames handed to the device so far
        self.frame = 0
        # When its last sample is heard (time.monotonic()), once the block
        # holding it went to the device
        self.ends: float | None = None
        # Blocks the device asked for after they were due, and how late
        # (seconds): what the speaker had no audio for
        self.late: list[float] = []
        self._reported = False
        self.stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=self.data.shape[1],
            dtype="float32",
            callback=self._callback,
            prime_output_buffers_using_stream_callback=False,
            **options,
        )
        try:
            self.latency = float(self.stream.latency)
        except Exception:
            self.latency = 0.0
        try:
            self.stream.start()
        except Exception:
            self.stream.close(ignore_errors=True)
            raise

    def _callback(self, outdata, frames, time_info, status):
        entered = time.monotonic()
        count = min(len(self.data) - self.frame, frames)
        if count <= 0:
            raise sd.CallbackAbort  # the block after the last: as sounddevice.play()
        outdata[:count] = self.data[self.frame : self.frame + count]
        outdata[count:] = 0
        self.frame += count
        # How long until this block's first sample is heard
        ahead = time_info.outputBufferDacTime - time_info.currentTime
        if not 0.0 <= ahead < 1.0:  # not told
            ahead = self.latency
        if status.output_underflow:
            self.late.append(0.0)
        elif time_info.outputBufferDacTime > 0.0:
            # Called this long after the block was due (waiting for Python)
            try:
                behind = self.stream.time - time_info.outputBufferDacTime
            except Exception:
                behind = 0.0
            if behind > 0.0:
                self.late.append(behind)
        if self.frame >= len(self.data):
            self.ends = entered + ahead + count / self.sample_rate

    @property
    def active(self) -> bool:
        try:
            return bool(self.stream.active)
        except Exception:
            return False

    def playing(self) -> bool:
        """Whether some of the clip is still to be heard."""
        return self.active and (self.ends is None or time.monotonic() < self.ends)

    def played_out(self) -> bool:
        """Whether all of the clip has been heard: its stream plays silence."""
        return self.ends is not None and time.monotonic() >= self.ends

    def left(self) -> float:
        """Seconds of the clip still to be heard."""
        if self.ends is not None:
            return max(0.0, self.ends - time.monotonic())
        return (len(self.data) - self.frame) / self.sample_rate

    def report(self):
        """Tell the voice log, once, where the speaker ran out of audio."""
        if self._reported:
            return
        self._reported = True
        for behind in self.late:
            voice_log.debug(
                "dropout: the speaker had no audio for a block of the clip%s",
                f" ({behind * 1000:.0f} ms late)" if behind else " (PortAudio's output underflow)",
            )

    def close(self):
        """End its stream at once (its device may be gone)."""
        self.report()
        try:
            self.stream.abort(ignore_errors=True)
            self.stream.close(ignore_errors=True)
        except Exception:
            logger.warning("The speaker's stream did not close cleanly", exc_info=True)


class _PortAudio:
    """PortAudio, which the computer's audio goes through: one per process.

    PortAudio reads the list of devices, and which are the defaults, once:
    when it is initialized. A device that comes or goes later (AirPods
    connecting or disconnecting) changes the devices underneath it, and a
    stream opened on the default it knows then fails ("Internal PortAudio
    error", "Invalid device") or plays on a device the user left. Only
    initializing it again reads the devices anew, and that closes every
    stream, so it is done here, in one place, under the lock that every use
    of PortAudio takes: the player, the microphone and previews each run on
    a thread of their own.
    """

    def __init__(self):
        # PortAudio crashes the process when two threads close a stream at
        # once. Stop reaches the speaker's from several threads (the app's
        # command, the reply's thread, the playback thread), so every use
        # goes through this lock, and the devices are read anew under it.
        self.lock = threading.RLock()
        # The clip playing now, and clips played to their end whose streams
        # still play the silence after them, until they end by themselves
        self.clip: _Clip | None = None
        self.ending: list[_Clip] = []
        # How many times the devices were read anew: a stream opened before
        # the last time is gone
        self.generation = 0
        # The microphone's streams that are open, closed when the devices are
        # read anew
        self.streams: weakref.WeakSet = weakref.WeakSet()
        # Called once the devices were read anew
        self.watchers: list[Callable[[], None]] = []
        # The system's defaults when the devices were read last
        self.defaults = system_defaults()

    def refresh(self, seen: int | None = None) -> bool:
        """Read the devices anew, unless that was done since generation seen
        (for the same change, by another thread).

        What plays stops, and the microphone's streams are closed: their
        captures open them again, on the devices as they are now.

        Returns:
            Whether the devices were read since seen; False if PortAudio
            could not be initialized again
        """
        with self.lock:
            if seen is not None and seen != self.generation:
                return True
            self.silence()
            for stream in list(self.streams):
                stream.drop()
            try:
                sd._terminate()
            except Exception:
                logger.warning("PortAudio did not terminate cleanly", exc_info=True)
            try:
                sd._initialize()
            except Exception:
                logger.exception("PortAudio could not be initialized again")
                return False
            self.generation += 1
            self.defaults = system_defaults()
            logger.info(
                "Read the audio devices anew; the defaults are %s and %s",
                _default_name("input"),
                _default_name("output"),
            )
            for watcher in list(self.watchers):
                try:
                    watcher()
                except Exception:
                    logger.exception("A watcher of the audio devices failed")
            return True

    def follow_system(self) -> bool:
        """Read the devices anew if the system's defaults changed since they
        were last read; only macOS tells (see system_defaults). Returns
        whether they were read anew."""
        defaults = system_defaults()
        with self.lock:
            if defaults is None or defaults == self.defaults:
                return False
            # Tried once for each change, whether it works or not
            self.defaults = defaults
            logger.info("The system's default audio devices changed")
            return self.refresh()

    def play(self, audio: np.ndarray, sample_rate: int, options: dict) -> _Clip:
        """Start playing a clip (under the lock), ending the one before,
        unless all of it has been heard: its stream is left to play the
        silence after it and end by itself."""
        last, self.clip = self.clip, None
        if last is not None and last.played_out():
            self.ending.append(last)
        elif last is not None:
            last.close()
        try:
            self.clip = _Clip(audio, sample_rate, **options)
        except sd.PortAudioError:
            if not self.ending:
                raise
            # Some speakers take one stream at a time: what still plays
            # after the clips before is only silence
            self._close_ending(everything=True)
            self.clip = _Clip(audio, sample_rate, **options)
        return self.clip

    def playing(self) -> bool:
        """Whether some of the clip played last is still to be heard (under
        the lock). The streams of clips played out that have ended are
        closed meanwhile, while it has long to play."""
        clip = self.clip
        if clip is None or clip.left() > CLOSE_WITHIN:
            self._close_ending()
        if clip is None:
            return False
        if clip.playing():
            return True
        clip.report()
        return False

    def stop(self):
        """End the clip playing now at once (under the lock). The streams
        that only play silence are closed later, as Stop and a hold must
        silence the voice without waiting for them."""
        if self.clip is not None:
            self.clip.close()
        self.clip = None

    def silence(self):
        """End every stream of a clip at once (under the lock)."""
        self.stop()
        self._close_ending(everything=True)

    def _close_ending(self, everything: bool = False):
        for clip in list(self.ending):
            if everything or not clip.active:
                clip.close()
                self.ending.remove(clip)


def _default_name(kind: str) -> str | None:
    try:
        return sd.query_devices(kind=kind)["name"]
    except Exception:
        return None


_PORTAUDIO = _PortAudio()


class _InputStream:
    """A stream from one of the computer's microphones, started and closed
    under PortAudio's lock. Reading the devices anew closes it for good
    (gone): its capture opens another."""

    def __init__(self, stream):
        # Opened (not yet started): it is closed when the devices are read
        # anew, started or not
        self._stream = stream
        self._open = True
        self.gone = False
        _PORTAUDIO.streams.add(self)

    def __enter__(self):
        with _PORTAUDIO.lock:
            if self.gone:
                raise sd.PortAudioError("The devices were read anew before the stream started")
            try:
                self._stream.__enter__()
            except Exception:
                # It would not start: closed, not left open in PortAudio
                self.__exit__()
                raise
        return self

    def __exit__(self, *exc):
        with _PORTAUDIO.lock:
            _PORTAUDIO.streams.discard(self)
            self._close()
        return False

    def drop(self):
        """Close it for the devices to be read anew."""
        _PORTAUDIO.streams.discard(self)
        self._close()
        self.gone = True

    def _close(self):
        if not self._open:
            return
        self._open = False
        try:
            self._stream.__exit__(None, None, None)
        except Exception:
            logger.warning("The microphone's stream did not close cleanly", exc_info=True)


class SoundDevice:
    """The computer's microphone and speaker, through sounddevice.

    All of Iris's audio goes through a device: the microphone as an input
    stream, the speaker as an output stream for each clip, one clip heard at
    a time (see _Clip). A test gives Iris a device of its own instead, such
    as a virtual room, and two assistants in one process can each have one.

    The devices may change while Iris runs. A clip the speaker cannot play
    has the devices read anew and is played once more, on the default as it
    is now; a microphone that cannot be opened, or stops delivering, is
    opened again the same way (see AudioRecorder). On macOS a change of the
    system's default devices is followed before each clip, and every second
    while Iris listens (see system_defaults), so it speaks and listens where
    the user chose without first failing.
    """

    @property
    def generation(self) -> int:
        """How many times the devices were read anew."""
        return _PORTAUDIO.generation

    def refresh(self, seen: int | None = None) -> bool:
        """Read the devices anew (see _PortAudio.refresh)."""
        return _PORTAUDIO.refresh(seen)

    def follow_system(self) -> bool:
        """Read the devices anew if the system's defaults changed."""
        return _PORTAUDIO.follow_system()

    def watch(self, watcher: Callable[[], None]):
        """Call watcher, on the thread that does it, each time the devices
        were read anew."""
        with _PORTAUDIO.lock:
            _PORTAUDIO.watchers.append(watcher)

    def unwatch(self, watcher: Callable[[], None]):
        with _PORTAUDIO.lock:
            if watcher in _PORTAUDIO.watchers:
                _PORTAUDIO.watchers.remove(watcher)

    def input_stream(self, **options):
        """An input stream, as sounddevice.InputStream takes its options."""
        with _PORTAUDIO.lock:
            return _InputStream(sd.InputStream(**options))

    def query_devices(self, *args, **kwargs):
        """The audio devices, as sounddevice.query_devices lists them."""
        with _PORTAUDIO.lock:
            return sd.query_devices(*args, **kwargs)

    def play(self, audio: np.ndarray, sample_rate: int, **options) -> float:
        """Start playing a clip, ending the one before unless all of it has
        been heard (see _Clip). The options are sounddevice.OutputStream's.

        A speaker that cannot be opened may be one PortAudio knows from
        before a device change: the devices are read anew and the clip is
        tried once more. The PortAudioError of that second try is raised.

        Returns:
            The output's latency in seconds (0 if it cannot be told)
        """
        with _PORTAUDIO.lock:
            _PORTAUDIO.follow_system()
            seen = _PORTAUDIO.generation
            try:
                clip = _PORTAUDIO.play(audio, sample_rate, options)
            except sd.PortAudioError as error:
                logger.warning("The speaker could not be opened (%s); reading the devices anew",
                               error)
                if not _PORTAUDIO.refresh(seen):
                    raise
                clip = _PORTAUDIO.play(audio, sample_rate, options)
            return clip.latency

    def stop(self):
        """Stop what plays."""
        with _PORTAUDIO.lock:
            _PORTAUDIO.stop()

    def playing(self) -> bool:
        """Whether some of the clip played last is still to be heard: not
        the silence its stream plays after it."""
        with _PORTAUDIO.lock:
            return _PORTAUDIO.playing()


# The computer's own audio, which Iris uses unless given another device
SYSTEM = SoundDevice()


# A device of a test's own (a virtual room) is never read anew: these stand in
# for the SoundDevice methods it lacks
def _generation(audio) -> int:
    return getattr(audio, "generation", 0)


def _refresh(audio, seen: int) -> bool:
    refresh = getattr(audio, "refresh", None)
    return bool(refresh is not None and refresh(seen))


def _follow_system(audio):
    follow = getattr(audio, "follow_system", None)
    if follow is not None:
        follow()


def _db(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.1f} dB"


class AudioRecorder:
    """Records the user's turns from the microphone: from the first syllable
    to where Smart Turn hears the end of what they had to say."""

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: float = 0.02,
        silence_duration: float = 2.5,
        chunk_duration: float = 2.0,
        device: int | None = None,
        turn_end: SmartTurn | None = None,
        echo: EchoCanceller | None = None,
        audio: SoundDevice | None = None,
    ):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Audio sample rate for recording (16 kHz for listen())
            silence_threshold: RMS level above which the input counts as
                sound, for level meters
            silence_duration: Silence that ends a turn whatever Smart Turn
                says, and all that does without it
            chunk_duration: Duration of chunks for streaming transcription
            device: Input device index (None = system default); the
                microphone is found again by its name when the devices change
            turn_end: Smart Turn, which hears when a turn is finished
            echo: Takes the voice out of what listen_on() hears while it speaks
            audio: The device recorded from; the computer's microphone if None
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.chunk_duration = chunk_duration
        self.turn_end = turn_end
        self.echo = echo
        self.audio = audio or SYSTEM
        self._speech: SpeechDetector | None = None  # loaded on first listen()
        self._device_error = False  # Track if we've had a device error
        # Of the last turn heard: seconds from the end of speech until it was
        # clear the turn was over
        self.last_turn_wait: float | None = None
        # The microphone, open from one listen to the next, and how far its
        # audio was listened to (the end of the last block taken)
        self._capture: _Capture | None = None
        self._capture_lock = threading.Lock()
        self._heard_until: float | None = None
        # How long speech over the voice about as loud as its echo must go
        # on to hold it (seconds): longer once such a hold heard only the
        # echo, in the replies after it too, until a hold hears the user
        self.near_echo_needs = HOLD_TRIGGER_NEAR_ECHO
        # When the system's default devices were last looked at
        self._followed = 0.0
        self.set_device(device)

    def check_audio_devices(self) -> tuple[bool, str]:
        """
        Check if audio input devices are available.

        Returns:
            Tuple of (devices_available: bool, message: str)
        """
        try:
            devices = self.audio.query_devices()
            input_devices = [d for d in devices if d["max_input_channels"] > 0]

            if not input_devices:
                return False, "No audio input devices found. Please connect a microphone."

            # Check if default input device is valid
            try:
                default_input = self.audio.query_devices(kind="input")
                if default_input["max_input_channels"] == 0:
                    return False, "Default input device has no input channels."
            except Exception:
                return False, "No default audio input device configured."

            return True, "Audio input device available"

        except Exception as e:
            return False, f"Error checking audio devices: {e}"

    @staticmethod
    def input_device_name() -> str:
        """Return the name of the default audio input device."""
        try:
            with _PORTAUDIO.lock:
                return sd.query_devices(kind="input")["name"]
        except Exception:
            return "Unknown microphone"

    @staticmethod
    def list_input_devices() -> list[dict]:
        """List available audio input devices as {index, name} dicts."""
        try:
            with _PORTAUDIO.lock:
                devices = sd.query_devices()
            return [
                {"index": i, "name": d["name"]}
                for i, d in enumerate(devices)
                if d["max_input_channels"] > 0
            ]
        except Exception:
            return []

    def set_device(self, index: int | None):
        """Select the input device by index (None = system default).

        Takes effect on the next listen() call. The device is kept by its
        name too: when the devices are read anew their indices may change,
        and the microphone is found again by its name, or while it is gone,
        the system's default is listened to."""
        self.device = index
        # The microphone chosen; None for the system's default
        self.device_name = self._name_of(index) if index is not None else None
        self._resolved_at = _generation(self.audio)
        self.close_microphone()

    def _name_of(self, index: int) -> str | None:
        try:
            return self.audio.query_devices(index)["name"]
        except Exception:
            return None

    def _device_index(self) -> int | None:
        """The index of the microphone chosen in the devices as they were
        last read (None: the system's default)."""
        if self.device_name is None:
            return self.device
        generation = _generation(self.audio)
        if generation != self._resolved_at:
            self._resolved_at = generation
            try:
                devices = list(self.audio.query_devices())
            except Exception:
                devices = []
            self.device = next(
                (
                    i
                    for i, d in enumerate(devices)
                    if d["name"] == self.device_name and d["max_input_channels"] > 0
                ),
                None,
            )
            if self.device is None:
                logger.warning("The microphone %s is gone; listening to the default",
                               self.device_name)
        return self.device

    def close_microphone(self):
        """Close the microphone (the mic switched off, the user typing, the
        conversation paused or over); the next listen opens it again."""
        with self._capture_lock:
            capture, self._capture = self._capture, None
            self._heard_until = None
        if capture is not None:
            capture.close()

    def current_device_name(self) -> str:
        """Return the name of the currently selected input device."""
        try:
            device = self._device_index()
            if device is not None:
                return self.audio.query_devices()[device]["name"]
            return self.audio.query_devices(kind="input")["name"]
        except Exception:
            return "Unknown microphone"

    def listen(
        self,
        on_level: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        silence_duration: float | None = None,
    ) -> str | None:
        """
        Record one turn from the microphone.

        The microphone opens at once; the turn starts with speech and ends
        when Smart Turn judges a pause in it the end of the turn, or after
        silence_duration of silence.

        Args:
            on_level: Optional callback called with audio level (0.0-1.0) for visualization
            should_stop: Optional callable; recording aborts (returning None) as soon
                as it returns True. Checked every 50 ms.
            silence_duration: Seconds of silence that end the turn whatever
                Smart Turn says; defaults to the recorder's setting

        Returns:
            Path to temporary audio file, or None if recording failed or was aborted
        """
        silence = self.silence_duration if silence_duration is None else silence_duration
        tracker = TurnTracker(self._detector(), self.turn_end, silence)
        level = self._level_meter(on_level)

        # Start recording immediately
        try:
            with self._microphone(go_on=False) as frames:
                while True:
                    if should_stop and should_stop():
                        return None
                    try:
                        frame = frames.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    level(frame)
                    if tracker.feed(frame):
                        self.last_turn_wait = tracker.end_wait
                        break
        except Exception as e:
            error_str = str(e).lower()
            # Check if this is a device error (no microphone available): one
            # that could not be opened even with the devices read anew, or
            # that stopped delivering and could not be opened again
            if (
                isinstance(e, (MicrophoneError, sd.PortAudioError))
                or "device" in error_str
                or "input" in error_str
                or "invalid" in error_str
            ):
                if not self._device_error:
                    # Only print the error once
                    self._device_error = True
                    print(f"\n[ERROR] Audio device error: {e}")
                    print("No microphone available. Please connect a microphone and restart.")
            else:
                # Other errors - print them
                print(f"Recording error: {e}")
            return None

        audio_data = tracker.audio()

        # Check if we have enough audio (at least 0.3 seconds)
        min_duration = 0.3
        if len(audio_data) / self.sample_rate < min_duration:
            return None

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, audio_data, self.sample_rate)

        return temp_file.name

    def listen_on(
        self,
        turn: str,
        on_speech: Callable[[], bool],
        until: Callable[[], bool],
        should_stop: Callable[[], bool] | None = None,
        on_level: Callable[[float], None] | None = None,
        silence_duration: float | None = None,
        speaking: Callable[[], bool] | None = None,
        hold: Callable[[], None] | None = None,
        release: Callable[[], None] | None = None,
    ) -> str | None:
        """
        Keep listening after a turn seemed over, while it is being answered.

        Smart Turn can take a pause for the end of a turn that was not over.
        Should the user go on before until() is true, on_speech() is called at
        once and the rest of the turn is recorded until it ends, judged as a
        whole with what came before.

        Given speaking, the user is also heard over the voice, with the echo
        canceller taking it out of the microphone. What it leaves of the
        voice can still sound like speech, so speech over the voice holds it
        (hold() pauses it), and the microphone listens in the quiet: the user
        goes on speaking, the voice's echo dies away with it; speech the VAD
        is certain of needs to go on for a moment only. Echo lets the voice
        go on (release()) where it was cut; speech too short for a turn (a
        back-channel), a little before it. Until the canceller has learned
        the room, it lets too much of the voice through for that: speech over
        the voice is not listened to then. Listening ends with the voice
        going on, however it ends.

        The VAD hardly hears how loud speech is, and takes even the faint
        echo a working canceller leaves for speech; at the start of a reply,
        while the canceller finds the voice again (its sound comes out of
        the speaker a little earlier or later each time), much of the voice
        comes through. So speech over the voice is weighed against how loud
        the voice's echo could be (EchoCanceller.against_echo): speech
        standing out above it holds the voice at once; speech about as loud
        as the echo only once it has gone on for a while; and speech the VAD
        is sure of while the canceller leaves far less than the echo is what
        it leaves of the voice, and holds nothing.

        Args:
            turn: Path to the turn recorded by listen()
            on_speech: Called as soon as the user speaks; returns whether that
                goes on with the earlier turn (False: a new turn begins)
            until: Listening ends when this returns True, unless the user has
                spoken by then
            should_stop: Recording aborts (returning None) when this returns True
            on_level: Audio level callback (0.0-1.0) once the user has spoken
            silence_duration: As for listen()
            speaking: Whether the voice is speaking; the user is heard over it
            hold: Pauses the voice
            release: Lets the voice go on after a hold: release(rewind), a
                little before where it was cut, or exactly there

        Returns:
            Path to the turn the user spoke (with the earlier one, if it went
            on with it), or None if the user said nothing
        """
        before, _ = sf.read(turn, dtype="float32")
        silence = self.silence_duration if silence_duration is None else silence_duration
        echo = self.echo if speaking is not None else None

        def new_tracker() -> TurnTracker:
            return TurnTracker(self._detector(), self.turn_end, silence, before=before)

        tracker = new_tracker()
        level = self._level_meter(on_level)
        frame_seconds = FRAME / self.sample_rate
        trigger = round(HOLD_TRIGGER / frame_seconds)
        near_trigger = round(self.near_echo_needs / frame_seconds)
        echo_trigger = round(HOLD_TRIGGER_AFTER_ECHO / frame_seconds)
        certain_frames = round(HOLD_CERTAIN_TIME / frame_seconds)
        backoff = round(HOLD_BACKOFF / frame_seconds)
        again = round(HOLD_AGAIN / frame_seconds)
        died_away = HOLD_SETTLE - frame_seconds / 2
        # Frames the voice must not be speaking for to count as done: between
        # two of its clips it may seem not to for a moment
        done_frames = round(HOLD_SETTLE / frame_seconds)

        # Frames in a row sure, and certain, to be speech, and whether speech
        # was certain since listening began anew
        sure, certain, was_certain = 0, 0, False
        # While speech over the voice is judged: frames left to judge it in,
        # when that began (the voice held, or seen to have ended), whether the
        # voice was held for it, and how many frames after the voice died
        # away were speech
        held, held_at, paused, heard = 0, 0.0, False, 0
        # Frames until the voice may be held again, whether speech heard
        # meanwhile waits to be judged, and when a hold last heard only echo
        calm, waiting, echoed = 0, False, -again
        # Frames in a row the voice was not speaking, and frames taken
        quiet, taken = 0, 0
        # Whether the hold was for speech about as loud as the echo
        held_near = False
        # Frames of speech that hold the voice. Each hold stops and restarts
        # the voice, so holds for echo stutter: after one, it takes longer
        # speech to hold the voice; after a second, it is not held again
        # until it has ended
        hold_after = trigger
        never = 1 << 30
        # How loud what the microphone heard was against the voice's echo
        # (None from a canceller that cannot tell)
        against = getattr(echo, "against_echo", None)
        # Of the speech over the voice sure to be speech now, once it was
        # long enough to hold the voice: the most it stood out above the
        # voice's echo (dB), and for how many frames it was sure
        loudest, lasted = None, 0

        def not_held(frames: int, why: str):
            """Speech over the voice long enough to have held it before it
            was weighed against the echo, which did not hold it."""
            voice_log.info("not held: %.2f s sure to be speech over the voice, %s", frames * frame_seconds, why)

        try:
            with self._microphone(echo, go_on=True) as frames:
                while True:
                    if should_stop and should_stop():
                        return None
                    if not tracker.started and until():
                        return None
                    try:
                        frame = frames.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    # When the frame was captured (time.monotonic())
                    at = frames.first + taken * frame_seconds
                    taken += 1
                    spoke = tracker.started
                    ended = tracker.feed(frame)
                    speech = tracker.probability >= SPEECH_THRESHOLD
                    sure = sure + 1 if tracker.probability >= HOLD_THRESHOLD else 0
                    certain = certain + 1 if tracker.probability >= HOLD_CERTAIN else 0
                    if certain >= certain_frames and echo is not None and echo.settled:
                        was_certain = True
                    calm = max(0, calm - 1)
                    if echo is not None:
                        quiet = 0 if speaking() else quiet + 1
                        if quiet >= done_frames:
                            hold_after = trigger
                    over_voice = echo is not None and quiet < done_frames
                    # The speech's last frames, weighed against the echo
                    levels = None
                    if not sure:
                        if loudest is not None and not held:
                            not_held(lasted, f"at most {loudest:.1f} dB over its echo (holding it takes "
                                             f"{HOLD_STANDS_OUT:.0f} dB, or {near_trigger * frame_seconds:.2f} s)")
                        loudest = None
                    elif over_voice and not held and against is not None and echo.settled:
                        levels = against(at - (min(sure, trigger) - 1) * frame_seconds, at + frame_seconds)
                    if levels is not None and levels.left <= HOLD_RESIDUAL:
                        # The VAD is sure of what the canceller leaves of
                        # the voice: it starts over
                        if sure >= trigger:
                            not_held(sure, f"{-levels.left:.0f} dB below its echo: what the canceller left of it")
                        sure, certain, was_certain, loudest, levels = 0, 0, False, None, None
                    elif levels is not None and sure >= trigger:
                        loudest = levels.heard if loudest is None else max(loudest, levels.heard)
                        lasted = sure
                    # Speech that does not stand out above the echo holds
                    # the voice only once it has gone on for a while
                    bar = hold_after
                    if levels is not None and levels.heard < HOLD_STANDS_OUT:
                        bar = max(hold_after, near_trigger)
                    if held:
                        held -= 1
                        heard += speech and at - held_at >= died_away
                        if held:
                            continue
                        enough = heard * frame_seconds >= HOLD_SPEECH * (HOLD - HOLD_SETTLE)
                        # Speech the VAD was certain of needs to go on for
                        # a moment only. The canceller's own measure cannot
                        # spare it that: the moment it starts letting the
                        # voice through looks to it like the user starting
                        # to speak
                        sure_users = was_certain and heard >= HOLD_CERTAIN_GOES_ON
                        if not ((enough or sure_users) and tracker.started):
                            # The voice's echo, or too little speech for a
                            # turn (a back-channel): the voice goes on, and so
                            # does listening. Held for nothing it goes on
                            # where it was cut; the user who spoke over it
                            # hears the cut word again
                            if paused:
                                release(heard > 0)
                                if not getattr(echo, "cancelling", True) or taken - echoed < again:
                                    calm = backoff
                                echoed = taken
                                hold_after = echo_trigger if hold_after == trigger else never
                                if held_near and not heard:
                                    # The echo went on long enough: from now
                                    # on, in later replies too, such speech
                                    # must go on longer
                                    near_trigger = max(near_trigger, echo_trigger)
                                    self.near_echo_needs = max(self.near_echo_needs, HOLD_TRIGGER_AFTER_ECHO)
                                voice_log.info(
                                    "%s: released after %.2f s of speech while held, going on %s; %s%s",
                                    "aside" if heard else "echo",
                                    heard * frame_seconds,
                                    "a little before where it was cut" if heard else "where it was cut",
                                    "no more holds until the voice ends" if hold_after == never
                                    else f"holds now need {hold_after * frame_seconds:.2f} s",
                                    f", and speech about as loud as the echo {near_trigger * frame_seconds:.2f} s "
                                    "from now on" if held_near and not heard else "",
                                )
                            tracker, sure, certain, was_certain, loudest = new_tracker(), 0, 0, False, None
                            continue
                        voice_log.info(
                            "user: %.2f s of speech while held%s", heard * frame_seconds,
                            " (the voice had ended)" if not paused else "",
                        )
                        if self.near_echo_needs > HOLD_TRIGGER_NEAR_ECHO:
                            self.near_echo_needs = HOLD_TRIGGER_NEAR_ECHO
                            voice_log.info(
                                "speech about as loud as the voice's echo holds it after %.2f s again",
                                HOLD_TRIGGER_NEAR_ECHO,
                            )
                        spoke = False  # the user's: taken up below
                    elif echo is not None and (not spoke or waiting) and (sure >= trigger or tracker.started):
                        if over_voice:
                            if not echo.settled:
                                tracker, sure, certain, was_certain = new_tracker(), 0, 0, False
                                waiting = False
                            elif calm or sure < bar:
                                # Kept, to be judged once the voice may be
                                # held again or has ended; over while the
                                # voice speaks on, it cannot be told from
                                # its echo. After a hold for echo, or about
                                # as loud as the echo, it holds the voice
                                # only once it has gone on longer
                                if ended:
                                    tracker, sure, certain, was_certain = new_tracker(), 0, 0, False
                                    waiting = False
                                    continue
                                waiting = True
                            else:
                                hold()
                                voice_log.info(
                                    "held: %.2f s sure to be speech, %s, canceller %s, suppression %s",
                                    sure * frame_seconds,
                                    "its level against the echo unknown" if levels is None
                                    else f"{levels.heard:.1f} dB over the voice's echo",
                                    "cancelling" if getattr(echo, "cancelling", True) else "leaking",
                                    _db(getattr(echo, "suppression", None)),
                                )
                                held, held_at, paused, heard, waiting = round(HOLD / frame_seconds), at, True, 0, False
                                held_near, loudest = bar > hold_after, None
                            continue
                        if waiting:
                            # The voice ended meanwhile: what was heard is
                            # judged as in a hold, with nothing to hold
                            held, held_at, paused, heard, waiting = round(HOLD / frame_seconds), at, False, 0, False
                            held_near = False
                            continue
                    if tracker.started:
                        if not spoke and not on_speech():
                            tracker.forget_before()
                        level(frame)
                    if ended:
                        self.last_turn_wait = tracker.end_wait
                        break
        except Exception as e:
            print(f"Recording error: {e}")
            return None
        finally:
            if held and paused and release is not None:
                release(True)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, tracker.audio(), self.sample_rate)
        return temp_file.name

    def _detector(self) -> SpeechDetector:
        """The VAD, fresh for a new stretch of audio."""
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"Turns are recorded at {SAMPLE_RATE} Hz")
        if self._speech is None:
            self._speech = SpeechDetector()
        self._speech.reset()
        return self._speech

    @contextmanager
    def _microphone(self, echo: EchoCanceller | None = None, go_on: bool = False) -> Iterator["_Frames"]:
        """The microphone's audio, in 32 ms frames; without what the speaker
        plays, given an echo canceller.

        The microphone stays open when this ends. go_on takes up its audio
        where the last listener stopped, so a turn the user goes on with
        loses nothing said in between; otherwise listening starts now, unless
        the last listener stopped only a moment ago.

        While it is listened to, the system's default devices are followed
        should they change (macOS), which closes the stream for the
        microphone to be opened again where the user chose. A stream that is
        gone, or stops delivering, is opened again (see _reopen).
        """
        with self._capture_lock:
            if self._capture is not None and self._capture.gone:
                self._capture.close()
                self._capture = None
            if self._capture is None:
                # Just opened: everything it has captured is new
                self._capture = self._open_capture()
                self._heard_until = None
            else:
                now = time.monotonic()
                since = self._heard_until
                if since is None or (not go_on and now - since > TAKE_UP_WITHIN):
                    since = now
                self._capture.skip_before(since)
            capture = self._capture

        def heard_until(moment: float):
            self._heard_until = moment

        yield _Frames(echo, capture, on_block=heard_until, reopen=self._reopen, follow=self._follow)

    def _follow(self):
        """Follow the system's default devices, at most once every
        FOLLOW_EVERY seconds (see SoundDevice.follow_system)."""
        now = time.monotonic()
        if now - self._followed >= FOLLOW_EVERY:
            self._followed = now
            _follow_system(self.audio)

    def _open_capture(self) -> "_Capture":
        """Open the microphone chosen, or the system's default (see
        _open_microphone)."""
        return _open_microphone(self.audio, self.sample_rate, self._device_index)

    def _reopen(self, lost: "_Capture") -> "_Capture":
        """Open the microphone again after its stream was lost.

        A stream closed for the devices to be read anew (gone) is opened
        again on them. One that stopped delivering has lost its device: the
        devices are read anew, and the microphone is opened where it is now
        (the one chosen, found by its name, else the new default). Should
        that one deliver nothing either, the microphone is given up
        (MicrophoneError), rather than opened again and again.

        Raises:
            queue.Empty: The microphone was closed meanwhile
            MicrophoneError: The microphone delivers nothing
            sd.PortAudioError: It could not be opened again
        """
        with self._capture_lock:
            if self._capture is not lost:
                if self._capture is None:
                    raise queue.Empty
                return self._capture  # opened again by another listener
            self._capture = None
            self._heard_until = None
            self._capture = _reopen_microphone(
                self.audio, self.sample_rate, self._device_index, lost
            )
            return self._capture

    @staticmethod
    def _level_meter(on_level: Callable[[float], None] | None) -> Callable[[np.ndarray], None]:
        """Reports the level of the frames it is given to on_level, at ~20 Hz."""
        last = [0.0]

        def level(frame: np.ndarray):
            now = time.monotonic()
            if on_level and now - last[0] >= 0.05:
                last[0] = now
                # Scale RMS to a 0-1 range (typical speech is 0.01-0.2);
                # float() keeps it JSON-serializable for GUI consumers
                rms = np.sqrt(np.mean(frame**2))
                on_level(float(min(1.0, rms / 0.15)))

        return level

    def listen_streaming(self, on_chunk: Callable[[str], None]) -> Generator[str, None, str | None]:
        """
        Record audio with streaming chunks for real-time transcription.

        Yields audio file paths for chunks as they become available,
        and returns the final complete audio file path.

        Args:
            on_chunk: Callback called with each chunk's audio file path

        Yields:
            Paths to temporary audio files for each chunk

        Returns:
            Path to final complete audio file, or None if no audio captured
        """
        audio_chunks = []
        silent_chunks = 0
        has_audio = False
        max_silent_chunks = int(self.silence_duration * self.sample_rate / 1024)
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        last_chunk_end = 0
        chunk_lock = threading.Lock()

        def audio_callback(indata, frames, time_info, status):
            """Callback for audio stream."""
            with chunk_lock:
                audio_chunks.append(indata.copy())

        # Start recording immediately
        try:
            with self.audio.input_stream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=1024,
                device=self.device,
            ):
                try:
                    while True:
                        sd.sleep(100)

                        with chunk_lock:
                            if len(audio_chunks) > 0:
                                recent_chunk = audio_chunks[-1]
                                rms = np.sqrt(np.mean(recent_chunk**2))

                                # Track if we've captured any audio above silence
                                if rms > self.silence_threshold:
                                    has_audio = True
                                    silent_chunks = 0
                                else:
                                    silent_chunks += 1

                                # Check if we have a new chunk to process
                                total_samples = sum(len(c) for c in audio_chunks)
                                if has_audio and total_samples - last_chunk_end >= chunk_samples:
                                    # Create chunk file for transcription
                                    chunk_data = np.concatenate(audio_chunks, axis=0)
                                    chunk_file = tempfile.NamedTemporaryFile(
                                        delete=False, suffix=".wav"
                                    )
                                    sf.write(chunk_file.name, chunk_data, self.sample_rate)
                                    last_chunk_end = total_samples
                                    on_chunk(chunk_file.name)

                                # Only stop on silence if we've recorded some audio
                                if has_audio and silent_chunks >= max_silent_chunks:
                                    break

                except KeyboardInterrupt:
                    raise
        except Exception as e:
            error_str = str(e).lower()
            # Check if this is a device error (no microphone available)
            if "device" in error_str or "input" in error_str or "invalid" in error_str:
                if not self._device_error:
                    # Only print the error once
                    self._device_error = True
                    print(f"\n[ERROR] Audio device error: {e}")
                    print("No microphone available. Please connect a microphone and restart.")
            else:
                # Other errors - print them
                print(f"Recording error: {e}")
            return None

        if not audio_chunks:
            return None

        audio_data = np.concatenate(audio_chunks, axis=0)

        # Check if we have enough audio
        min_duration = 0.3
        if len(audio_data) / self.sample_rate < min_duration:
            return None

        # Save final complete audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, audio_data, self.sample_rate)

        return temp_file.name


class _Capture:
    """The microphone kept open from one listen to the next.

    Its audio waits in a queue for whoever listens next, so what is said
    while nobody listens (the moment between two listens, while a turn is
    being recognized) is heard by the next listener instead of being lost.
    Audio older than KEEP_CAPTURED is dropped.
    """

    def __init__(self, audio: SoundDevice, sample_rate: int, device: int | None):
        self.sample_rate = sample_rate
        self._blocks: deque[tuple[np.ndarray, float]] = deque()
        self._ready = threading.Condition()
        self._closed = False
        # The devices' generation it was opened in, whether it has delivered
        # audio, when it last did (or was opened), and whether it was opened
        # because the one before it stopped delivering
        self.generation = _generation(audio)
        self.delivered = False
        self._last_block = time.monotonic()
        self.after_stall = False
        self._stream = audio.input_stream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            # The echo canceller works in 10 ms steps
            blocksize=3 * STEP,
            device=device,
        )
        self._stream.__enter__()

    def _callback(self, indata, frame_count, time_info, status):
        # When the block was captured, on the time.monotonic() clock
        try:
            lag = time_info.currentTime - time_info.inputBufferAdcTime
        except AttributeError:
            lag = 0.0
        if not 0.0 < lag < 1.0:  # not reported
            lag = frame_count / self.sample_rate
        captured = time.monotonic() - lag
        with self._ready:
            self._last_block = time.monotonic()
            self.delivered = True
            self._blocks.append((indata[:, 0].copy(), captured))
            while self._blocks and self._blocks[0][1] < captured - KEEP_CAPTURED:
                self._blocks.popleft()
            self._ready.notify()

    @property
    def gone(self) -> bool:
        """Its stream was closed for the devices to be read anew."""
        return getattr(self._stream, "gone", False)

    def take(self, timeout: float) -> tuple[np.ndarray, float]:
        """The next block and when it was captured; raises queue.Empty if
        none came within timeout, or the microphone was closed, and
        MicrophoneError if its stream is gone or has stopped delivering."""
        with self._ready:
            if not self._blocks and not self._closed:
                self._ready.wait(timeout)
            if self._blocks:
                return self._blocks.popleft()
            if not self._closed and (self.gone or self._stalled()):
                raise MicrophoneError("The microphone stopped delivering sound")
            raise queue.Empty

    def _stalled(self) -> bool:
        # A Bluetooth microphone takes a moment to start
        allowed = STALLED if self.delivered else STARTING
        return time.monotonic() - self._last_block > allowed

    def skip_before(self, moment: float):
        """Drop what was captured before this moment."""
        with self._ready:
            while self._blocks and self._blocks[0][1] < moment:
                self._blocks.popleft()

    def close(self):
        """Close it; a stream whose device is gone may fail to close, which
        is no reason to fail."""
        with self._ready:
            if self._closed:
                return
            self._closed = True
            self._blocks.clear()
            self._ready.notify_all()
        try:
            self._stream.__exit__(None, None, None)
        except Exception:
            logger.warning("The microphone did not close cleanly", exc_info=True)


class _Frames:
    """The microphone's audio in 32 ms frames, as the VAD takes it: cleaned
    of the voice, given an echo canceller, outside the audio callback."""

    def __init__(
        self, echo: EchoCanceller | None, capture: _Capture, on_block=None, reopen=None, follow=None
    ):
        self._echo = echo
        self._capture = capture
        self._on_block = on_block
        # Opens the microphone again when its stream is lost, and follows the
        # system's default devices
        self._reopen = reopen
        self._follow = follow
        self._pending = np.zeros(0, dtype=np.float32)
        # When the first frame was captured (time.monotonic()); the frames
        # follow one another from there
        self.first = 0.0
        self._started = False

    def get(self, timeout: float) -> np.ndarray:
        """The next frame; raises queue.Empty if none came within timeout."""
        if self._follow is not None:
            self._follow()
        while len(self._pending) < FRAME:
            try:
                block, captured = self._capture.take(timeout)
            except MicrophoneError:
                if self._reopen is None:
                    raise
                self._capture = self._reopen(self._capture)
                continue
            if not self._started:
                self.first, self._started = captured, True
            if self._on_block:
                self._on_block(captured + len(block) / self._capture.sample_rate)
            if self._echo is not None:
                block = self._echo.clean(block, captured)
            self._pending = np.concatenate([self._pending, block])
        frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
        return frame


# The sound check: seconds of speech it needs to hear to say a voice comes
# through, seconds of nothing but exact zeros after which the microphone
# counts as silenced (on macOS: access not given), how often it reports, and
# how long a check left open runs before it closes by itself
HEARD_AFTER = 0.3
SILENT_AFTER = 3.0
SOUND_CHECK_REPORT = 0.05
SOUND_CHECK_LIMIT = 300.0


def _open_microphone(
    audio: SoundDevice, sample_rate: int, device: Callable[[], int | None]
) -> "_Capture":
    """Open a capture of the microphone device() names (None: the system's
    default).

    PortAudio may know the devices from before one came or went: when it
    cannot open the microphone, the devices are read anew and it is tried
    once more, device() asked again. The PortAudioError of that second try
    is raised.
    """
    seen = _generation(audio)
    try:
        return _Capture(audio, sample_rate, device())
    except sd.PortAudioError as error:
        logger.warning("The microphone could not be opened (%s); reading the devices anew", error)
        if not _refresh(audio, seen):
            raise
        return _Capture(audio, sample_rate, device())


def _reopen_microphone(
    audio: SoundDevice, sample_rate: int, device: Callable[[], int | None], lost: "_Capture"
) -> "_Capture":
    """Open a capture again after its stream was lost.

    A stream closed for the devices to be read anew (gone) is opened again
    on them. One that stopped delivering has lost its device: the devices
    are read anew, and the microphone is opened where it is now. Should that
    one deliver nothing either, it is given up (MicrophoneError), rather
    than opened again and again.

    Raises:
        MicrophoneError: The microphone delivers nothing
        sd.PortAudioError: It could not be opened again
    """
    lost.close()
    stalled = not lost.gone
    if stalled:
        if lost.after_stall and not lost.delivered:
            raise MicrophoneError("The microphone delivers no sound")
        logger.warning("The microphone stopped delivering sound; reading the devices anew")
        _refresh(audio, lost.generation)
    capture = _open_microphone(audio, sample_rate, device)
    capture.after_stall = stalled
    return capture


class _Chosen:
    """A microphone chosen by its index, found again by its name once the
    devices were read anew, as their indices may change; None for the
    system's default, or for a chosen one that is gone."""

    def __init__(self, audio: SoundDevice, device: int | None):
        self.audio = audio
        self.device = device
        self._seen = _generation(audio)
        self.name = None
        if device is not None:
            try:
                self.name = audio.query_devices(device)["name"]
            except Exception:
                pass

    def __call__(self) -> int | None:
        generation = _generation(self.audio)
        if self.name is not None and generation != self._seen:
            self._seen = generation
            try:
                devices = list(self.audio.query_devices())
            except Exception:
                devices = []
            self.device = next(
                (
                    i
                    for i, d in enumerate(devices)
                    if d["name"] == self.name and d["max_input_channels"] > 0
                ),
                None,
            )
        return self.device


class SoundCheck:
    """The microphone on its own, before Iris listens, for the walkthrough's
    sound check: its level, whether a voice is in it (the VAD Iris hears
    turns with), and whether it gives nothing but silence. Reports about 20
    times a second as {"level": 0-1, "heard": bool, "silent": bool}."""

    def __init__(
        self,
        on_update: Callable[[dict], None],
        device: int | None = None,
        audio: SoundDevice | None = None,
        detector: Callable[[np.ndarray], float] | None = None,
    ):
        """
        Args:
            on_update: Told each report
            device: Input device index (None = system default)
            audio: The device listened to; the computer's microphone if None
            detector: Speech probability of a frame; Silero VAD if None
        """
        self.on_update = on_update
        self.device = device
        self.audio = audio or SYSTEM
        self._detector = detector
        self._stopped = threading.Event()
        self._capture: _Capture | None = None
        self._thread: threading.Thread | None = None

    def device_name(self) -> str:
        """The name of the microphone listened to."""
        try:
            if self.device is not None:
                return self.audio.query_devices()[self.device]["name"]
            return self.audio.query_devices(kind="input")["name"]
        except Exception:
            return "Unknown microphone"

    def start(self):
        """Open the microphone, raising if it cannot be, and listen to it.
        Should the devices change, it is opened again as Iris's microphone
        is (see AudioRecorder._reopen)."""
        self._chosen = _Chosen(self.audio, self.device)
        self._capture = _open_microphone(self.audio, SAMPLE_RATE, self._chosen)
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _reopen(self, lost: "_Capture") -> "_Capture":
        self._capture = _reopen_microphone(self.audio, SAMPLE_RATE, self._chosen, lost)
        return self._capture

    def stop(self):
        """Close the microphone."""
        self._stopped.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def _listen(self):
        frames = _Frames(None, self._capture, reopen=self._reopen)
        detector = self._detector or SpeechDetector()
        started = last = time.monotonic()
        speech, level, zeros_since = 0.0, 0.0, None
        try:
            while not self._stopped.is_set() and time.monotonic() - started < SOUND_CHECK_LIMIT:
                try:
                    frame = frames.get(0.25)
                except queue.Empty:
                    continue
                except (MicrophoneError, sd.PortAudioError):
                    logger.warning("The sound check lost its microphone", exc_info=True)
                    break
                rms = float(np.sqrt(np.mean(frame**2)))
                # The loudest moment since the last report, scaled as the
                # level meter of listen() scales it
                level = max(level, min(1.0, rms / 0.15))
                if detector(frame) >= SPEECH_THRESHOLD:
                    speech += FRAME / SAMPLE_RATE
                now = time.monotonic()
                if rms > 0:
                    zeros_since = None
                elif zeros_since is None:
                    zeros_since = now
                if now - last >= SOUND_CHECK_REPORT:
                    last = now
                    self.on_update(
                        {
                            "level": level,
                            "heard": speech >= HEARD_AFTER,
                            "silent": zeros_since is not None and now - zeros_since >= SILENT_AFTER,
                        }
                    )
                    level = 0.0
        finally:
            self._capture.close()


# A recording made on its own (a sample of a voice to clone): how often it
# reports its level, and how long it runs before it stops by itself
RECORDING_REPORT = 0.05
RECORDING_LIMIT = 60.0


class Recording:
    """The microphone on its own, recording until stopped, at the device's
    own rate (a voice to clone is kept above the 16 kHz Iris listens at).
    Reports about 20 times a second as {"level": 0-1, "seconds": s}, the
    level scaled as the sound check scales it. It stops by itself after
    RECORDING_LIMIT, and tells on_end."""

    def __init__(
        self,
        on_update: Callable[[dict], None],
        device: int | None = None,
        audio: SoundDevice | None = None,
        on_end: Callable[[], None] | None = None,
        limit: float = RECORDING_LIMIT,
    ):
        """
        Args:
            on_update: Told each report
            device: Input device index (None = system default)
            audio: The device recorded from; the computer's microphone if None
            on_end: Told when it stopped by itself
            limit: Seconds it records at most
        """
        self.on_update = on_update
        self.device = device
        self.audio = audio or SYSTEM
        self.on_end = on_end
        self.limit = limit
        self.sample_rate = SAMPLE_RATE
        self._blocks: list[np.ndarray] = []
        self._stopped = threading.Event()
        self._capture: _Capture | None = None
        self._thread: threading.Thread | None = None

    def _device_rate(self) -> int:
        """The rate the device records at by itself; 16 kHz if it cannot tell."""
        try:
            if self.device is not None:
                info = self.audio.query_devices()[self.device]
            else:
                info = self.audio.query_devices(kind="input")
            return int(info["default_samplerate"]) or SAMPLE_RATE
        except Exception:
            return SAMPLE_RATE

    def start(self):
        """Open the microphone, raising if it cannot be, and record. Should
        the microphone be lost (the devices change), the recording ends there
        as at its limit: a sample from two microphones clones worse than a
        shorter one, which the user can record again."""
        self.sample_rate = self._device_rate()
        self._capture = _open_microphone(
            self.audio, self.sample_rate, _Chosen(self.audio, self.device)
        )
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[np.ndarray, int]:
        """Close the microphone; returns what was recorded and its rate."""
        self._stopped.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        blocks = self._blocks
        audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
        return audio, self.sample_rate

    def _record(self):
        capture = self._capture
        last = time.monotonic()
        level, samples, ended = 0.0, 0, False
        try:
            while not self._stopped.is_set():
                if samples >= self.limit * self.sample_rate:
                    ended = True
                    break
                try:
                    block, _ = capture.take(0.25)
                except queue.Empty:
                    continue
                except MicrophoneError:
                    logger.warning("The recording lost its microphone")
                    ended = True
                    break
                self._blocks.append(block)
                samples += len(block)
                rms = float(np.sqrt(np.mean(block**2))) if len(block) else 0.0
                level = max(level, min(1.0, rms / 0.15))
                now = time.monotonic()
                if now - last >= RECORDING_REPORT:
                    last = now
                    self.on_update({"level": level, "seconds": round(samples / self.sample_rate, 2)})
                    level = 0.0
            # Stopped: what the microphone delivered until then is kept too
            while not ended:
                try:
                    self._blocks.append(capture.take(0)[0])
                except (queue.Empty, MicrophoneError):
                    break
        finally:
            capture.close()
        if ended and self.on_end:
            self.on_end()


# Seconds of the microphone's audio kept for the next listener
KEEP_CAPTURED = 30.0
# A microphone that delivers nothing for this long has lost its device (it
# delivers a block every 30 ms), or one just opened, nothing for this long
STALLED = 2.0
STARTING = 5.0
# Seconds between two looks at the system's default devices while listening
FOLLOW_EVERY = 1.0
# A new listen takes up the microphone's audio where the last one stopped if
# that was at most this long ago (a turn following the answer at once);
# after longer, what came in between is not for it (it may be Iris's voice)
TAKE_UP_WITHIN = 0.5

# Frames per playback buffer: large, so generation never starves playback
PLAYBACK_BLOCK = 32768

# On resume, playback restarts this far before the point where it was cut,
# so the listener hears the interrupted word again rather than half of it
RESUME_REWIND = 0.35


def play_sound(audio: np.ndarray, sample_rate: int, **options) -> float:
    """Play on the computer's speaker (see SoundDevice.play)."""
    return SYSTEM.play(audio, sample_rate, **options)


def stop_sound():
    """Stop what plays on the computer's speaker."""
    SYSTEM.stop()


def sound_playing() -> bool:
    """Whether the computer's speaker still has audio to play."""
    return SYSTEM.playing()


class AudioPlayer:
    """Plays audio files with normalization and fade effects."""

    def __init__(self, audio_filter: Optional[AudioFilter] = None, audio: SoundDevice | None = None):
        """
        Initialize audio player.

        Args:
            audio_filter: Optional AudioFilter instance to apply to audio
            audio: The device played on; the computer's speaker if None
        """
        self.audio_filter = audio_filter
        self.audio = audio or SYSTEM
        # Told what is played, so the microphone can hear the user over it
        self.echo: EchoCanceller | None = None
        self._paused = threading.Event()
        self._stopped = threading.Event()
        # Whether playback goes on a little before where a pause cut it
        self._rewind = True
        # Where the clip playing now is: (its length and sample rate, the
        # position its current segment started from, and when that segment
        # started being heard, None while paused)
        self._place: tuple[int, int, int, float | None] | None = None
        # Of the clip played last: when it began to be heard, and when it
        # was heard to its end (None while it plays, or once cut short)
        self.heard: tuple[float | None, float | None] = (None, None)

    def _cut(self):
        """What plays stopped being heard now."""
        if self.echo is not None:
            self.echo.cut(time.monotonic())

    def stop(self):
        """Stop any currently playing audio."""
        self._stopped.set()
        self.audio.stop()
        self._cut()

    def pause(self):
        """Pause playback; play() holds its place until resume()."""
        self._paused.set()

    def resume(self, rewind: bool = True):
        """Continue playback from where pause() cut it: a little before it
        (rewind), so a listener who was interrupted hears the cut word again,
        or exactly there, for a voice held for nothing (its own echo)."""
        self._rewind = rewind
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def progress(self) -> float:
        """How much of the clip playing now has been heard, from 0 to 1."""
        place = self._place
        if place is None:
            return 0.0
        length, sample_rate, position, heard_since = place
        heard = position
        if heard_since is not None:
            heard += max(0.0, time.monotonic() - heard_since) * sample_rate
        return min(1.0, heard / length) if length else 1.0

    @staticmethod
    def _emit_levels(
        audio: np.ndarray,
        sample_rate: int,
        on_level: Callable[[float], None],
        stop: threading.Event,
        rate: int = 15,
    ):
        """Walk the audio's RMS envelope in wall-clock time during playback.

        sounddevice plays the buffer without exposing progress, so the envelope
        is precomputed and replayed on a timer alongside it.
        """
        mono = audio if audio.ndim == 1 else audio.mean(axis=1)
        frame = max(1, int(sample_rate / rate))
        frames = len(mono) // frame
        if frames == 0:
            return
        envelope = np.sqrt(
            np.mean(mono[: frames * frame].reshape(frames, frame) ** 2, axis=1)
        )
        # Playback is normalized to 0.85 peak; scale so speech reaches the top
        envelope = np.minimum(envelope / 0.25, 1.0)
        # Syllables make the raw envelope jump between frames, which reads as
        # flicker on a meter; a short moving average keeps the shape but not
        # the jitter.
        if frames >= 3:
            envelope = np.convolve(envelope, np.ones(3) / 3, mode="same")

        start = time.monotonic()
        while not stop.is_set():
            index = int((time.monotonic() - start) * rate)
            if index >= frames:
                return
            on_level(float(envelope[index]))
            stop.wait(1 / rate)

    def _play_resumable(
        self,
        audio: np.ndarray,
        sample_rate: int,
        on_level: Callable[[float], None] | None,
        cancelled: Callable[[], bool] | None = None,
    ):
        """Play a buffer, honoring pause() and stop() while it plays.

        sounddevice cannot pause a stream, so a pause stops it, estimates how
        far playback got, and the remainder is played on resume. The devices
        read anew while it plays (one came or went) cut it the same way: the
        remainder is played at once, on the device as it is now.

        Raises:
            SpeakerError: The speaker could not be opened
        """
        self._stopped.clear()
        self._place = (len(audio), sample_rate, 0, None)
        self.heard = (None, None)

        def stopped() -> bool:
            return self._stopped.is_set() or (cancelled is not None and cancelled())

        position = 0
        # Where the last cut left the clip: how far it was heard (samples),
        # and how much of its segment was heard (seconds)
        cut: tuple[int, float] | None = None
        while position < len(audio):
            while self._paused.is_set() and not stopped():
                time.sleep(0.05)
            if stopped():
                return
            if cut is not None:
                position = self._going_on(cut, len(audio), sample_rate)
                cut = None
                if position >= len(audio):
                    return

            segment = np.ascontiguousarray(audio[position:])
            if position > 0:
                # Fade the restart in, like the start of every clip
                fade = min(len(segment), int(sample_rate * 0.01))
                ramp = np.linspace(0, 1, fade)
                segment[:fade] *= ramp if segment.ndim == 1 else ramp[:, np.newaxis]

            # Large blocks and a high latency, so the audio callback (which
            # needs Python) runs seldom and is never late
            started = time.monotonic()
            try:
                latency = self.audio.play(
                    segment, sample_rate, blocksize=PLAYBACK_BLOCK, latency="high"
                )
            except sd.PortAudioError as error:
                raise SpeakerError(str(error)) from error
            generation = _generation(self.audio)
            self._place = (len(audio), sample_rate, position, started + latency)
            if self.heard[0] is None:
                self.heard = (started + latency, None)
            if self.echo is not None:
                self.echo.played(started + latency, segment, sample_rate)

            level_thread = None
            level_stop = threading.Event()
            if on_level:
                level_thread = threading.Thread(
                    target=self._emit_levels,
                    args=(segment, sample_rate, on_level, level_stop),
                    daemon=True,
                )
                level_thread.start()

            # Until the clip has been heard to its end (the device tells when,
            # see SoundDevice.playing), or a pause or stop cuts it. The next
            # clip starts then, while the stream of this one plays out the
            # silence after it alone. The deadline is only a safety net
            # against a hung device.
            buffered = latency + 2 * PLAYBACK_BLOCK / sample_rate
            deadline = len(segment) / sample_rate + buffered + 2.0
            while time.monotonic() - started < deadline:
                if self._paused.is_set() or stopped():
                    break
                if not self.audio.playing():
                    break
                time.sleep(0.02)
            elapsed = time.monotonic() - started
            moved = _generation(self.audio) != generation
            cut_by_pause = (self._paused.is_set() or moved) and not stopped()
            if moved and not self._paused.is_set():
                # Played on at once on the device as it is now: the words
                # the change clipped are heard again
                self._rewind = True
            if cut_by_pause or stopped():
                self.audio.stop()
                self._cut()

            if level_thread:
                level_stop.set()
                level_thread.join(timeout=0.5)
                on_level(0.0)

            if not cut_by_pause:
                if not stopped():
                    self.heard = (self.heard[0], started + elapsed)
                return
            # What was heard is the wall time minus the output latency; where
            # it goes on is decided once it does (resume)
            heard = max(0.0, elapsed - latency)
            cut = (position + int(heard * sample_rate), heard)
            self._place = (len(audio), sample_rate, min(cut[0], len(audio)), None)

    def _going_on(self, cut: tuple[int, float], length: int, sample_rate: int) -> int:
        """Where playback goes on after a cut: where the clip was heard to,
        or a little before it (rewind), for the words the cut may have
        clipped, but never further back than half of the segment heard: a
        hold early in a clip must not start it over."""
        heard_to, heard = cut
        back = min(RESUME_REWIND, heard / 2) if self._rewind else 0.0
        position = max(0, heard_to - int(back * sample_rate))
        voice_log.info(
            "resumes at %.2f s of %.2f s, %s (heard %.2f s of this segment)",
            position / sample_rate, length / sample_rate,
            f"{back:.2f} s before where it was cut" if back else "where it was cut", heard,
        )
        return position

    def play(
        self,
        audio_file: str,
        on_level: Callable[[float], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        audio_filter: AudioFilter | None | object = OWN_FILTER,
    ):
        """
        Play pre-generated audio file with optional filter.

        Args:
            audio_file: Path to WAV file to play
            on_level: Optional callback receiving the playback level (0.0-1.0)
                at ~15 Hz while the audio plays, for output level meters
            cancelled: Optional callable; playback ends (or never starts) as
                soon as it returns True

            audio_filter: Filter for this clip instead of the player's own
                (None for none)
        """
        try:
            # Load audio into memory completely before any playback
            audio, sample_rate = sf.read(audio_file, dtype="float32")

            # Pre-process EVERYTHING before starting playback to avoid CPU spikes

            # Apply audio filter if specified
            if audio_filter is OWN_FILTER:
                audio_filter = self.audio_filter
            if audio_filter:
                audio = audio_filter.apply(audio, sample_rate)

            # Normalize audio to prevent clipping and reduce artifacts
            # Scale to 85% of max to give extra headroom
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio * (0.85 / max_val)

            # Apply gentle fade in/out to reduce clicks
            fade_samples = int(sample_rate * 0.01)  # 10ms fade
            if len(audio) > fade_samples * 2:
                # Fade in
                fade_in = np.linspace(0, 1, fade_samples)
                if audio.ndim == 1:
                    audio[:fade_samples] *= fade_in
                else:
                    audio[:fade_samples] *= fade_in[:, np.newaxis]

                # Fade out
                fade_out = np.linspace(1, 0, fade_samples)
                if audio.ndim == 1:
                    audio[-fade_samples:] *= fade_out
                else:
                    audio[-fade_samples:] *= fade_out[:, np.newaxis]

            self._play_resumable(np.ascontiguousarray(audio), sample_rate, on_level, cancelled)

        except SpeakerError:
            raise
        except Exception as e:
            print(f"Error playing audio: {e}")
            traceback.print_exc()

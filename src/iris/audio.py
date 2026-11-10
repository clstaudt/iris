"""Audio recording, playback, and filters for voice assistant."""

import queue
import tempfile
import threading
import time
import warnings
from abc import ABC, abstractmethod
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
    HOLD_CERTAIN,
    HOLD_CERTAIN_TIME,
    HOLD_SETTLE,
    HOLD_SPEECH,
    HOLD_THRESHOLD,
    HOLD_TRIGGER,
    SAMPLE_RATE,
    SPEECH_THRESHOLD,
    SmartTurn,
    SpeechDetector,
    TurnTracker,
)

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
            device: Input device index (None = system default)
            turn_end: Smart Turn, which hears when a turn is finished
            echo: Takes the voice out of what listen_on() hears while it speaks
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.chunk_duration = chunk_duration
        self.device = device
        self.turn_end = turn_end
        self.echo = echo
        self._speech: SpeechDetector | None = None  # loaded on first listen()
        self._device_error = False  # Track if we've had a device error
        # Of the last turn heard: seconds from the end of speech until it was
        # clear the turn was over
        self.last_turn_wait: float | None = None

    @staticmethod
    def check_audio_devices() -> tuple[bool, str]:
        """
        Check if audio input devices are available.

        Returns:
            Tuple of (devices_available: bool, message: str)
        """
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d["max_input_channels"] > 0]

            if not input_devices:
                return False, "No audio input devices found. Please connect a microphone."

            # Check if default input device is valid
            try:
                default_input = sd.query_devices(kind="input")
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
            return sd.query_devices(kind="input")["name"]
        except Exception:
            return "Unknown microphone"

    @staticmethod
    def list_input_devices() -> list[dict]:
        """List available audio input devices as {index, name} dicts."""
        try:
            return [
                {"index": i, "name": d["name"]}
                for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0
            ]
        except Exception:
            return []

    def set_device(self, index: int | None):
        """Select the input device by index (None = system default).

        Takes effect on the next listen() call."""
        self.device = index

    def current_device_name(self) -> str:
        """Return the name of the currently selected input device."""
        try:
            if self.device is not None:
                return sd.query_devices()[self.device]["name"]
            return sd.query_devices(kind="input")["name"]
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
            with self._microphone() as frames:
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
        goes on speaking, the voice's echo dies away with it. Speech more
        certain than any echo the settled canceller leaves is the user's too.
        Echo lets the voice go on (release()). Until the canceller has
        learned the room, it lets too much of the voice through for that:
        speech over the voice is not listened to then.

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
            release: Lets the voice go on after a hold

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
        certain_frames = round(HOLD_CERTAIN_TIME / frame_seconds)
        # Frames in a row sure, and certain, to be speech, and whether speech
        # was certain since listening began anew; while the voice is held,
        # frames left to listen to, and how many after it died away were speech
        sure, certain, was_certain, held, heard = 0, 0, False, 0, 0
        try:
            with self._microphone(echo) as frames:
                while True:
                    if should_stop and should_stop():
                        return None
                    if not tracker.started and until():
                        return None
                    try:
                        frame = frames.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    spoke = tracker.started
                    ended = tracker.feed(frame)
                    sure = sure + 1 if tracker.probability >= HOLD_THRESHOLD else 0
                    certain = certain + 1 if tracker.probability >= HOLD_CERTAIN else 0
                    if certain >= certain_frames and echo is not None and echo.settled:
                        was_certain = True
                    if held:
                        held -= 1
                        settled = HOLD - held * frame_seconds > HOLD_SETTLE
                        heard += settled and tracker.probability >= SPEECH_THRESHOLD
                        if held:
                            continue
                        enough = heard * frame_seconds >= HOLD_SPEECH * (HOLD - HOLD_SETTLE)
                        if not ((enough or was_certain) and tracker.started):
                            # The voice's echo: it goes on, and so does listening
                            release()
                            tracker, sure, certain, was_certain = new_tracker(), 0, 0, False
                            continue
                        spoke = False  # the user's: taken up below
                    elif echo is not None and not spoke and (sure >= trigger or tracker.started):
                        if speaking():
                            if echo.settled:
                                hold()
                                held, heard = round(HOLD / frame_seconds), 0
                            else:
                                tracker, sure, certain, was_certain = new_tracker(), 0, 0, False
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
    def _microphone(self, echo: EchoCanceller | None = None) -> Iterator["_Frames"]:
        """The microphone open, its audio arriving in 32 ms frames; without
        what the speaker plays, given an echo canceller."""
        frames = _Frames(echo)

        def audio_callback(indata, frame_count, time_info, status):
            """Callback for audio stream."""
            # When the block was captured, on the time.monotonic() clock
            try:
                lag = time_info.currentTime - time_info.inputBufferAdcTime
            except AttributeError:
                lag = 0.0
            if not 0.0 < lag < 1.0:  # not reported
                lag = frame_count / self.sample_rate
            frames.put(indata[:, 0].copy(), time.monotonic() - lag)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            # The echo canceller works in 10 ms steps
            blocksize=3 * STEP if echo is not None else FRAME,
            device=self.device,
        ):
            yield frames

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
            with sd.InputStream(
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


class _Frames:
    """The microphone's audio in 32 ms frames, as the VAD takes it: cleaned
    of the voice, given an echo canceller, outside the audio callback."""

    def __init__(self, echo: EchoCanceller | None):
        self._echo = echo
        self._blocks: queue.Queue[tuple[np.ndarray, float]] = queue.Queue()
        self._pending = np.zeros(0, dtype=np.float32)

    def put(self, block: np.ndarray, captured: float):
        self._blocks.put((block, captured))

    def get(self, timeout: float) -> np.ndarray:
        """The next frame; raises queue.Empty if none came within timeout."""
        while len(self._pending) < FRAME:
            block, captured = self._blocks.get(timeout=timeout)
            if self._echo is not None:
                block = self._echo.clean(block, captured)
            self._pending = np.concatenate([self._pending, block])
        frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
        return frame


# Frames per playback buffer: large, so generation never starves playback
PLAYBACK_BLOCK = 32768

# On resume, playback restarts this far before the point where it was cut,
# so the listener hears the interrupted word again rather than half of it
RESUME_REWIND = 0.35


# sounddevice's play() and stop() act on one shared output stream, and
# PortAudio crashes the process when two threads close it at once. Stop
# reaches it from several threads (the app's command, the reply's thread, the
# playback thread), so every use of that stream goes through this lock.
_OUTPUT = threading.RLock()


def play_sound(audio: np.ndarray, sample_rate: int, **options) -> float:
    """Start playing on the shared output stream, ending what played before.

    Returns:
        The stream's output latency in seconds (0 if it cannot be told)
    """
    with _OUTPUT:
        sd.play(audio, sample_rate, **options)
        try:
            return float(sd.get_stream().latency)
        except Exception:
            return 0.0


def stop_sound():
    """Stop whatever plays on the shared output stream."""
    with _OUTPUT:
        sd.stop()


def sound_playing() -> bool:
    """Whether the shared output stream still has audio to play."""
    with _OUTPUT:
        try:
            return bool(sd.get_stream().active)
        except Exception:
            return False


class AudioPlayer:
    """Plays audio files with normalization and fade effects."""

    def __init__(self, audio_filter: Optional[AudioFilter] = None):
        """
        Initialize audio player.

        Args:
            audio_filter: Optional AudioFilter instance to apply to audio
        """
        self.audio_filter = audio_filter
        # Told what is played, so the microphone can hear the user over it
        self.echo: EchoCanceller | None = None
        self._paused = threading.Event()
        self._stopped = threading.Event()
        # Where the clip playing now is: (its length and sample rate, the
        # position its current segment started from, and when that segment
        # started being heard, None while paused)
        self._place: tuple[int, int, int, float | None] | None = None

    def stop(self):
        """Stop any currently playing audio."""
        self._stopped.set()
        stop_sound()
        if self.echo is not None:
            self.echo.cut(time.monotonic())

    def pause(self):
        """Pause playback; play() holds its place until resume()."""
        self._paused.set()

    def resume(self):
        """Continue playback from where pause() cut it."""
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
        far playback got, and the remainder is played on resume.
        """
        self._stopped.clear()
        self._place = (len(audio), sample_rate, 0, None)

        def stopped() -> bool:
            return self._stopped.is_set() or (cancelled is not None and cancelled())

        position = 0
        while position < len(audio):
            while self._paused.is_set() and not stopped():
                time.sleep(0.05)
            if stopped():
                return

            segment = np.ascontiguousarray(audio[position:])
            if position > 0:
                # Fade the restart in, like the start of every clip
                fade = min(len(segment), int(sample_rate * 0.01))
                ramp = np.linspace(0, 1, fade)
                segment[:fade] *= ramp if segment.ndim == 1 else ramp[:, np.newaxis]

            # Play with maximum buffer size and latency for ultra-smooth playback
            # Extra large buffer (32768 samples = ~740ms) prevents any underruns
            started = time.monotonic()
            latency = play_sound(segment, sample_rate, blocksize=PLAYBACK_BLOCK, latency="high")
            self._place = (len(audio), sample_rate, position, started + latency)
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

            # Until the stream has played its last buffer (PortAudio only then
            # reports it inactive), or a pause or stop cuts it. The stream
            # outlives the clip by up to a block plus the device latency; giving
            # up earlier let the next clip's sd.play() cut off the last words.
            # The deadline is only a safety net against a hung device.
            buffered = latency + 2 * PLAYBACK_BLOCK / sample_rate
            deadline = len(segment) / sample_rate + buffered + 2.0
            while time.monotonic() - started < deadline:
                if self._paused.is_set() or stopped():
                    break
                if not sound_playing():
                    break
                time.sleep(0.02)
            elapsed = time.monotonic() - started
            cut_by_pause = self._paused.is_set() and not stopped()
            if cut_by_pause or stopped():
                stop_sound()
                if self.echo is not None:
                    self.echo.cut(time.monotonic())

            if level_thread:
                level_stop.set()
                level_thread.join(timeout=0.5)
                on_level(0.0)

            if not cut_by_pause:
                return
            # What was heard is the wall time minus the output latency
            heard = max(0.0, elapsed - latency - RESUME_REWIND)
            position += int(heard * sample_rate)
            self._place = (len(audio), sample_rate, position, None)

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

        except Exception as e:
            print(f"Error playing audio: {e}")
            import traceback

            traceback.print_exc()

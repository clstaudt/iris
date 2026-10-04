"""Audio recording, playback, and filters for voice assistant."""

import tempfile
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf

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


# ============================================================================
# Audio Recording and Playback
# ============================================================================


class AudioRecorder:
    """Records audio from microphone with silence detection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: float = 0.02,
        silence_duration: float = 2.5,
        chunk_duration: float = 2.0,
        device: int | None = None,
    ):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Audio sample rate for recording
            silence_threshold: RMS threshold for silence detection
            silence_duration: Duration of silence to stop recording
            chunk_duration: Duration of chunks for streaming transcription
            device: Input device index (None = system default)
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.chunk_duration = chunk_duration
        self.device = device
        self._device_error = False  # Track if we've had a device error

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
    ) -> str | None:
        """
        Record audio from microphone until silence is detected.
        Starts recording immediately without waiting for speech threshold.

        Args:
            on_level: Optional callback called with audio level (0.0-1.0) for visualization
            should_stop: Optional callable; recording aborts (returning None) as soon
                as it returns True. Checked every 50 ms.

        Returns:
            Path to temporary audio file, or None if recording failed or was aborted
        """
        audio_chunks = []
        silent_chunks = 0
        has_audio = False
        max_silent_chunks = int(self.silence_duration * self.sample_rate / 1024)

        def audio_callback(indata, frames, time_info, status):
            """Callback for audio stream."""
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
                        sd.sleep(50)  # Faster updates for smoother visualization

                        if should_stop and should_stop():
                            return None

                        if len(audio_chunks) > 0:
                            recent_chunk = audio_chunks[-1]
                            rms = np.sqrt(np.mean(recent_chunk**2))

                            # Notify level callback (normalize to 0-1 range)
                            if on_level:
                                # Scale RMS to a 0-1 range (typical speech is 0.01-0.2)
                                # float() matters: rms is a numpy scalar, which
                                # is not JSON-serializable for GUI consumers
                                normalized_level = float(min(1.0, rms / 0.15))
                                on_level(normalized_level)

                            # Track if we've captured any audio above silence
                            if rms > self.silence_threshold:
                                has_audio = True
                                silent_chunks = 0
                            else:
                                silent_chunks += 1
                                # Only stop on silence if we've recorded some audio
                                if has_audio and silent_chunks >= max_silent_chunks:
                                    break

                except KeyboardInterrupt:
                    # Re-raise to allow proper shutdown in main loop
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

        # Check if we have enough audio (at least 0.3 seconds)
        min_duration = 0.3
        if len(audio_data) / self.sample_rate < min_duration:
            return None

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, audio_data, self.sample_rate)

        return temp_file.name

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


# Frames per playback buffer: large, so generation never starves playback
PLAYBACK_BLOCK = 32768

# On resume, playback restarts this far before the point where it was cut,
# so the listener hears the interrupted word again rather than half of it
RESUME_REWIND = 0.35


class AudioPlayer:
    """Plays audio files with normalization and fade effects."""

    def __init__(self, audio_filter: Optional[AudioFilter] = None):
        """
        Initialize audio player.

        Args:
            audio_filter: Optional AudioFilter instance to apply to audio
        """
        self.audio_filter = audio_filter
        self._paused = threading.Event()
        self._stopped = threading.Event()

    def stop(self):
        """Stop any currently playing audio."""
        self._stopped.set()
        sd.stop()

    def pause(self):
        """Pause playback; play() holds its place until resume()."""
        self._paused.set()

    def resume(self):
        """Continue playback from where pause() cut it."""
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

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
            sd.play(segment, sample_rate, blocksize=PLAYBACK_BLOCK, latency="high")
            started = time.monotonic()
            try:
                latency = float(sd.get_stream().latency)
            except Exception:
                latency = 0.0

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
                try:
                    if not sd.get_stream().active:
                        break
                except Exception:
                    pass
                time.sleep(0.02)
            elapsed = time.monotonic() - started
            cut_by_pause = self._paused.is_set() and not stopped()
            if cut_by_pause or stopped():
                sd.stop()

            if level_thread:
                level_stop.set()
                level_thread.join(timeout=0.5)
                on_level(0.0)

            if not cut_by_pause:
                return
            # What was heard is the wall time minus the output latency
            heard = max(0.0, elapsed - latency - RESUME_REWIND)
            position += int(heard * sample_rate)

    def play(
        self,
        audio_file: str,
        on_level: Callable[[float], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ):
        """
        Play pre-generated audio file with optional filter.

        Args:
            audio_file: Path to WAV file to play
            on_level: Optional callback receiving the playback level (0.0-1.0)
                at ~15 Hz while the audio plays, for output level meters
            cancelled: Optional callable; playback ends (or never starts) as
                soon as it returns True
        """
        try:
            # Load audio into memory completely before any playback
            audio, sample_rate = sf.read(audio_file, dtype="float32")

            # Pre-process EVERYTHING before starting playback to avoid CPU spikes

            # Apply audio filter if specified
            if self.audio_filter:
                audio = self.audio_filter.apply(audio, sample_rate)

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

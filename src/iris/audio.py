"""Audio recording, playback, and filters for voice assistant."""

import tempfile
import threading
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
        sos = signal.butter(4, [300, 3400], btype='band', fs=sample_rate, output='sos')
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
        sos = signal.butter(4, [300, 5000], btype='band', fs=sample_rate, output='sos')
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
        sos = signal.butter(3, [400, 4000], btype='band', fs=sample_rate, output='sos')
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
        sos = signal.butter(2, 200, btype='high', fs=sample_rate, output='sos')
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
    ):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Audio sample rate for recording
            silence_threshold: RMS threshold for silence detection
            silence_duration: Duration of silence to stop recording
            chunk_duration: Duration of chunks for streaming transcription
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.chunk_duration = chunk_duration

    def listen(
        self,
        on_level: Callable[[float], None] | None = None,
        check_paused: Callable[[], bool] | None = None,
    ) -> str | None:
        """
        Record audio from microphone until silence is detected.
        Starts recording immediately without waiting for speech threshold.

        Args:
            on_level: Optional callback called with audio level (0.0-1.0) for visualization
            check_paused: Optional callback that returns True if recording should be paused

        Returns:
            Path to temporary audio file, or None if recording failed
        """
        audio_chunks = []
        silent_chunks = 0
        has_audio = False
        max_silent_chunks = int(self.silence_duration * self.sample_rate / 1024)
        paused = False

        def audio_callback(indata, frames, time_info, status):
            """Callback for audio stream."""
            # Only record if not paused
            if not paused:
                audio_chunks.append(indata.copy())

        # Start recording immediately
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=1024,
            ):
                try:
                    while True:
                        sd.sleep(50)  # Faster updates for smoother visualization

                        # Check if we should pause/unpause
                        if check_paused:
                            new_paused = check_paused()
                            if new_paused != paused:
                                paused = new_paused
                                if paused:
                                    # Reset silence counter when pausing
                                    silent_chunks = 0

                        # Skip processing if paused
                        if paused:
                            if on_level:
                                on_level(0.0)  # Show no activity when paused
                            continue

                        if len(audio_chunks) > 0:
                            recent_chunk = audio_chunks[-1]
                            rms = np.sqrt(np.mean(recent_chunk**2))

                            # Notify level callback (normalize to 0-1 range)
                            if on_level:
                                # Scale RMS to a 0-1 range (typical speech is 0.01-0.2)
                                normalized_level = min(1.0, rms / 0.15)
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

    def listen_streaming(
        self, on_chunk: Callable[[str], None]
    ) -> Generator[str, None, str | None]:
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
                                if (
                                    has_audio
                                    and total_samples - last_chunk_end >= chunk_samples
                                ):
                                    # Create chunk file for transcription
                                    chunk_data = np.concatenate(audio_chunks, axis=0)
                                    chunk_file = tempfile.NamedTemporaryFile(
                                        delete=False, suffix=".wav"
                                    )
                                    sf.write(
                                        chunk_file.name, chunk_data, self.sample_rate
                                    )
                                    last_chunk_end = total_samples
                                    on_chunk(chunk_file.name)

                                # Only stop on silence if we've recorded some audio
                                if has_audio and silent_chunks >= max_silent_chunks:
                                    break

                except KeyboardInterrupt:
                    raise
        except Exception as e:
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


class AudioPlayer:
    """Plays audio files with normalization and fade effects."""
    
    def __init__(self, audio_filter: Optional[AudioFilter] = None):
        """
        Initialize audio player.
        
        Args:
            audio_filter: Optional AudioFilter instance to apply to audio
        """
        self.audio_filter = audio_filter
        self.should_stop = False

    def stop(self):
        """Stop any currently playing audio immediately."""
        self.should_stop = True
        try:
            # More graceful stop to avoid malloc errors
            sd.stop()
        except Exception:
            pass

    def play(self, audio_file: str):
        """
        Play pre-generated audio file with optional filter.

        Args:
            audio_file: Path to WAV file to play
        """
        # Reset stop flag at start of playback
        self.should_stop = False
        
        try:
            # Check if we should stop before even loading
            if self.should_stop:
                return
            
            # Load audio into memory completely before any playback
            audio, sample_rate = sf.read(audio_file, dtype="float32")

            # Check again before processing
            if self.should_stop:
                return
            
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

            # Ensure audio is contiguous in memory for efficient playback
            audio = np.ascontiguousarray(audio)

            # Final check before starting playback
            if self.should_stop:
                return

            # Play with maximum buffer size and latency for ultra-smooth playback
            # Extra large buffer (32768 samples = ~740ms) prevents any underruns
            sd.play(audio, sample_rate, blocksize=32768, latency='high')
            
            # Wait for playback to complete, checking stop flag frequently
            while not self.should_stop:
                try:
                    stream = sd.get_stream()
                    if stream is None or not stream.active:
                        break
                    sd.sleep(30)  # Check every 30ms for fast interrupt response
                except Exception:
                    # Stream already stopped or error
                    break
            
            # If stopped, ensure audio is halted (but check if stream exists first)
            if self.should_stop:
                try:
                    sd.stop()
                except Exception:
                    pass  # Already stopped

        except Exception as e:
            # Silently ignore errors from interrupted playback
            if "PortAudioError" not in str(type(e).__name__):
                print(f"Error playing audio: {e}")
                import traceback
                traceback.print_exc()

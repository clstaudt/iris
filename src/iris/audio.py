"""Audio recording and playback for voice assistant."""

import tempfile
import threading
import warnings
from collections.abc import Callable, Generator
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

warnings.filterwarnings("ignore")


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
        self, on_level: Callable[[float], None] | None = None
    ) -> str | None:
        """
        Record audio from microphone until silence is detected.
        Starts recording immediately without waiting for speech threshold.

        Args:
            on_level: Optional callback called with audio level (0.0-1.0) for visualization

        Returns:
            Path to temporary audio file, or None if recording failed
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
            ):
                try:
                    while True:
                        sd.sleep(50)  # Faster updates for smoother visualization

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

    def stop(self):
        """Stop any currently playing audio."""
        sd.stop()

    def play(self, audio_file: str):
        """
        Play pre-generated audio file.

        Args:
            audio_file: Path to WAV file to play
        """
        try:
            # Load and play the audio
            audio, sample_rate = sf.read(audio_file, dtype="float32")

            # Normalize audio to prevent clipping and reduce artifacts
            # Scale to 90% of max to avoid clipping
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio * (0.9 / max_val)

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

            # Play audio with appropriate blocksize to reduce glitches
            sd.play(audio, sample_rate, blocksize=2048)
            sd.wait()

        except Exception as e:
            print(f"Error playing audio: {e}")
            import traceback

            traceback.print_exc()


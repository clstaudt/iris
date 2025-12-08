"""Audio recording and playback for voice assistant."""

import tempfile
import warnings
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
    ):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Audio sample rate for recording
            silence_threshold: RMS threshold for silence detection
            silence_duration: Duration of silence to stop recording
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

    def listen(self) -> str | None:
        """
        Record audio from microphone until silence is detected.
        Waits for speech to be detected before starting to record.

        Returns:
            Path to temporary audio file, or None if no speech detected
        """
        audio_chunks = []
        silent_chunks = 0
        speech_detected = False
        max_silent_chunks = int(self.silence_duration * self.sample_rate / 1024)
        speech_threshold = (
            self.silence_threshold * 1.5
        )  # Threshold to detect actual speech (lowered for sensitivity)

        def audio_callback(indata, frames, time_info, status):
            """Callback for audio stream."""
            audio_chunks.append(indata.copy())

        # Start recording
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

                        if len(audio_chunks) > 0:
                            recent_chunk = audio_chunks[-1]
                            rms = np.sqrt(np.mean(recent_chunk**2))

                            # Wait for speech to start
                            if not speech_detected:
                                if rms > speech_threshold:
                                    speech_detected = True
                                    silent_chunks = 0
                            else:
                                # Once speech is detected, monitor for silence
                                if rms < self.silence_threshold:
                                    silent_chunks += 1
                                    if silent_chunks >= max_silent_chunks:
                                        break
                                else:
                                    silent_chunks = 0

                except KeyboardInterrupt:
                    # Re-raise to allow proper shutdown in main loop
                    raise
        except Exception as e:
            print(f"Recording error: {e}")
            return None

        # Only save audio if speech was actually detected
        if not speech_detected or not audio_chunks:
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


class AudioPlayer:
    """Plays audio files with normalization and fade effects."""

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


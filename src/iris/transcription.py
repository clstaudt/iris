"""Speech-to-text transcription using Whisper."""

import threading
import warnings
from collections.abc import Callable
from pathlib import Path

from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")


class WhisperTranscriber:
    """Transcribes audio to text using Whisper model."""

    def __init__(self, model: str = "base", device: str = "auto"):
        """
        Initialize Whisper transcriber.

        Args:
            model: Whisper model size (tiny, base, small, medium, large)
            device: Device for Whisper (cpu, cuda, or auto)
        """
        # Auto-detect device
        if device == "auto":
            import torch

            if torch.backends.mps.is_available():
                device = "cpu"  # faster-whisper doesn't support MPS, use CPU
                print("Apple Silicon detected - Whisper will use CPU (optimized)")
            elif torch.cuda.is_available():
                device = "cuda"
                print("CUDA detected - Whisper will use GPU acceleration")
            else:
                device = "cpu"
                print("Using CPU for Whisper")

        # Load Whisper model
        print(f"Loading Whisper model '{model}' on {device}...")
        self.whisper = WhisperModel(
            model,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
        self.device = device
        self._transcription_lock = threading.Lock()
        print("Whisper model loaded")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        with self._transcription_lock:
            segments, info = self.whisper.transcribe(
                audio_path,
                beam_size=5,
                language="en",  # Change to None for auto-detection
            )

            # Collect all segments
            text = " ".join([segment.text for segment in segments]).strip()

            return text

    def transcribe_streaming(
        self,
        audio_path: str,
        on_partial: Callable[[str], None],
    ) -> str:
        """
        Transcribe audio with streaming partial results.

        This transcribes the audio and calls the callback with partial
        text as segments are processed.

        Args:
            audio_path: Path to audio file
            on_partial: Callback called with accumulated text after each segment

        Returns:
            Final complete transcribed text
        """
        with self._transcription_lock:
            segments, info = self.whisper.transcribe(
                audio_path,
                beam_size=5,
                language="en",
            )

            # Stream segments as they're processed
            accumulated_text = []
            for segment in segments:
                accumulated_text.append(segment.text)
                current_text = " ".join(accumulated_text).strip()
                on_partial(current_text)

            return " ".join(accumulated_text).strip()


class StreamingTranscriber:
    """Handles streaming transcription from audio chunks."""

    def __init__(self, transcriber: WhisperTranscriber):
        """
        Initialize streaming transcriber.

        Args:
            transcriber: WhisperTranscriber instance to use
        """
        self.transcriber = transcriber
        self.current_text = ""
        self._lock = threading.Lock()
        self._chunk_files: list[str] = []

    def process_chunk(self, chunk_path: str) -> str:
        """
        Process an audio chunk and update the current transcription.

        Args:
            chunk_path: Path to audio chunk file

        Returns:
            Current accumulated transcription text
        """
        with self._lock:
            self._chunk_files.append(chunk_path)

            # Transcribe the cumulative audio (the chunk contains all audio so far)
            text = self.transcriber.transcribe(chunk_path)
            self.current_text = text

            return text

    def get_final_text(self, final_audio_path: str) -> str:
        """
        Get final transcription from complete audio.

        Args:
            final_audio_path: Path to complete audio file

        Returns:
            Final transcribed text
        """
        with self._lock:
            # Clean up chunk files
            for chunk_file in self._chunk_files:
                Path(chunk_file).unlink(missing_ok=True)
            self._chunk_files.clear()

            # Transcribe final complete audio for best accuracy
            self.current_text = self.transcriber.transcribe(final_audio_path)
            return self.current_text

    def reset(self):
        """Reset the streaming transcriber state."""
        with self._lock:
            self.current_text = ""
            for chunk_file in self._chunk_files:
                Path(chunk_file).unlink(missing_ok=True)
            self._chunk_files.clear()


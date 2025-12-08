"""Speech-to-text transcription using Whisper."""

import warnings

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
        print("Whisper model loaded")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        segments, info = self.whisper.transcribe(
            audio_path,
            beam_size=5,
            language="en",  # Change to None for auto-detection
        )

        # Collect all segments
        text = " ".join([segment.text for segment in segments]).strip()

        return text


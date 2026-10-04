"""Integration tests for Whisper transcription."""

import tempfile
import numpy as np
import soundfile as sf
import pytest
from iris.transcription import WhisperTranscriber


@pytest.fixture(scope="module")
def transcriber():
    """Create WhisperTranscriber - downloads model on first run."""
    return WhisperTranscriber(model="tiny", device="cpu")


def create_silent_audio(duration=1.0, sample_rate=16000):
    """Create a silent audio file for testing."""
    samples = int(duration * sample_rate)
    audio = np.zeros(samples, dtype=np.float32)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp.name, audio, sample_rate)
    return temp.name


@pytest.mark.integration
class TestWhisperTranscriber:
    def test_transcribe_silent_audio(self, transcriber):
        """Transcribing silence returns empty or minimal text."""
        audio_path = create_silent_audio()
        result = transcriber.transcribe(audio_path)
        # Silent audio should produce empty or very short result
        assert len(result) < 50

    def test_device_detected(self, transcriber):
        """Device is set after initialization."""
        # MLX (Apple Silicon) decodes on the GPU; faster-whisper on CPU or CUDA
        assert transcriber.device in ("cpu", "cuda", "gpu")


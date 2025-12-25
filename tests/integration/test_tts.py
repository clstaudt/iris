"""Integration tests for TTS."""

import os
from pathlib import Path
import pytest
from iris.tts import KokoroGenerator


@pytest.fixture
def generator():
    """Create KokoroGenerator - requires kokoro-tts installed and models."""
    return KokoroGenerator()


@pytest.mark.integration
class TestKokoroGenerator:
    def test_generate_creates_audio_file(self, generator):
        """TTS generates a valid audio file."""
        audio_path = generator.generate("Hello world")
        assert audio_path is not None
        assert Path(audio_path).exists()
        assert Path(audio_path).stat().st_size > 0
        # Cleanup
        Path(audio_path).unlink(missing_ok=True)

    def test_generate_empty_returns_file(self, generator):
        """Empty text still produces output."""
        audio_path = generator.generate("Test")
        assert audio_path is not None
        Path(audio_path).unlink(missing_ok=True)







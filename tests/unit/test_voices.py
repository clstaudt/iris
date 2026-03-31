"""Unit tests for VoiceStore."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from iris.voices import VoiceStore, VoiceInfo


def _create_reference_wav(path: Path, duration: float = 5.0, sample_rate: int = 24000):
    """Create a dummy WAV file for testing."""
    samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
    sf.write(str(path), samples, sample_rate)


class TestVoiceStore:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = VoiceStore(voices_dir=Path(self.tmpdir))
        # Create a reference WAV file
        self.ref_path = Path(self.tmpdir) / "ref.wav"
        _create_reference_wav(self.ref_path, duration=5.0)

    def test_save_and_get_voice(self):
        info = self.store.save_voice(
            name="test_voice",
            reference_audio_path=self.ref_path,
            base_voice="af_bella",
            description="A test voice",
        )
        assert info.name == "test_voice"
        assert info.base_voice == "af_bella"
        assert info.description == "A test voice"
        assert info.reference_duration > 0

        retrieved = self.store.get_voice("test_voice")
        assert retrieved is not None
        assert retrieved.name == "test_voice"
        assert retrieved.base_voice == "af_bella"

    def test_list_voices(self):
        assert self.store.list_voices() == []

        self.store.save_voice(name="voice_a", reference_audio_path=self.ref_path, base_voice="af_bella")
        ref2 = Path(self.tmpdir) / "ref2.wav"
        _create_reference_wav(ref2)
        self.store.save_voice(name="voice_b", reference_audio_path=ref2, base_voice="af_bella")

        voices = self.store.list_voices()
        assert len(voices) == 2
        names = [v.name for v in voices]
        assert "voice_a" in names
        assert "voice_b" in names

    def test_delete_voice(self):
        self.store.save_voice(name="to_delete", reference_audio_path=self.ref_path, base_voice="af_bella")
        assert self.store.get_voice("to_delete") is not None

        result = self.store.delete_voice("to_delete")
        assert result is True
        assert self.store.get_voice("to_delete") is None

    def test_delete_nonexistent_voice(self):
        result = self.store.delete_voice("nonexistent")
        assert result is False

    def test_resolve_cloned_voice(self):
        self.store.save_voice(
            name="cloned", reference_audio_path=self.ref_path, base_voice="bm_lewis"
        )

        base_voice, ref_audio = self.store.resolve_voice("cloned")
        assert base_voice == "bm_lewis"
        assert ref_audio is not None
        assert ref_audio.exists()

    def test_resolve_builtin_voice(self):
        base_voice, ref_audio = self.store.resolve_voice("af_bella")
        assert base_voice == "af_bella"
        assert ref_audio is None

    def test_reject_duplicate_name(self):
        self.store.save_voice(name="unique", reference_audio_path=self.ref_path, base_voice="af_bella")

        try:
            ref2 = Path(self.tmpdir) / "ref2.wav"
            _create_reference_wav(ref2)
            self.store.save_voice(name="unique", reference_audio_path=ref2, base_voice="af_bella")
            assert False, "Should have raised FileExistsError"
        except FileExistsError:
            pass

    def test_reject_invalid_name(self):
        try:
            self.store.save_voice(name="bad name!", reference_audio_path=self.ref_path, base_voice="af_bella")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_reject_builtin_voice_name(self):
        try:
            self.store.save_voice(name="af_custom", reference_audio_path=self.ref_path, base_voice="af_bella")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_reject_short_audio(self):
        short_ref = Path(self.tmpdir) / "short.wav"
        _create_reference_wav(short_ref, duration=1.0)
        try:
            self.store.save_voice(name="short_voice", reference_audio_path=short_ref, base_voice="af_bella")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "too short" in str(e)

    def test_reject_long_audio(self):
        long_ref = Path(self.tmpdir) / "long.wav"
        _create_reference_wav(long_ref, duration=60.0)
        try:
            self.store.save_voice(name="long_voice", reference_audio_path=long_ref, base_voice="af_bella")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "too long" in str(e)

    def test_reject_missing_file(self):
        try:
            self.store.save_voice(
                name="missing", reference_audio_path="/nonexistent/path.wav", base_voice="af_bella"
            )
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_reference_wav_stored(self):
        self.store.save_voice(name="stored", reference_audio_path=self.ref_path, base_voice="af_bella")
        voice_dir = Path(self.tmpdir) / "stored"
        assert (voice_dir / "reference.wav").exists()
        assert (voice_dir / "metadata.yaml").exists()

"""Unit tests for picking and running the speech recognition engine."""

import numpy as np
import pytest

from iris import transcription
from iris.transcription import WhisperTranscriber, model_repo, resolve_engine, resolve_model


@pytest.fixture
def apple_silicon(monkeypatch):
    monkeypatch.setattr(transcription, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(transcription, "HAS_MLX", True)


def test_mlx_is_chosen_on_apple_silicon(apple_silicon):
    assert resolve_engine("auto") == "mlx"
    assert resolve_engine("faster-whisper") == "faster-whisper"


def test_elsewhere_faster_whisper_is_used_even_if_mlx_is_asked_for(monkeypatch):
    monkeypatch.setattr(transcription, "is_apple_silicon", lambda: False)
    assert resolve_engine("auto") == "faster-whisper"
    assert resolve_engine("mlx") == "faster-whisper"


def test_mlx_needs_the_package(monkeypatch):
    monkeypatch.setattr(transcription, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(transcription, "HAS_MLX", False)
    assert resolve_engine("auto") == "faster-whisper"


def test_auto_model_is_turbo_on_a_gpu_and_small_on_a_cpu():
    assert resolve_model("auto", "mlx", "gpu") == "large-v3-turbo"
    assert resolve_model("auto", "faster-whisper", "cuda") == "large-v3-turbo"
    assert resolve_model("auto", "faster-whisper", "cpu") == "small"
    assert resolve_model("turbo", "faster-whisper", "cpu") == "large-v3-turbo"
    assert resolve_model("base", "mlx", "gpu") == "base"


def test_each_engine_downloads_its_own_weights():
    assert model_repo("small", "mlx") == "mlx-community/whisper-small-mlx"
    assert model_repo("large-v3-turbo", "mlx") == "mlx-community/whisper-large-v3-turbo"
    assert model_repo("small", "faster-whisper") == "Systran/faster-whisper-small"
    assert model_repo("large-v3-turbo", "faster-whisper").endswith("faster-whisper-large-v3-turbo")


class FakeMlx:
    """Stands in for mlx_whisper; records what it was asked to decode."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def transcribe(self, audio, path_or_hf_repo, **kwargs):
        self.calls.append((len(audio), path_or_hf_repo))
        return self.result


def _mlx_transcriber(monkeypatch, result, speech):
    fake = FakeMlx(result)
    monkeypatch.setattr(transcription, "mlx_whisper", fake, raising=False)
    monkeypatch.setattr(transcription, "decode_audio", lambda path, sampling_rate: np.zeros(32000))
    monkeypatch.setattr(transcription, "get_speech_timestamps", lambda audio: speech)
    monkeypatch.setattr(
        transcription,
        "collect_chunks",
        lambda audio, chunks: ([audio[c["start"] : c["end"]] for c in chunks], []),
    )
    t = WhisperTranscriber(model="small", engine="mlx")
    fake.calls.clear()  # forget the warm-up
    return t, fake


def test_mlx_decodes_only_the_speech(apple_silicon, monkeypatch):
    result = {
        "language": "fr",
        "segments": [
            {"text": " Bonjour", "no_speech_prob": 0.05},
            {"text": " You", "no_speech_prob": 0.92},
        ],
    }
    t, fake = _mlx_transcriber(monkeypatch, result, [{"start": 4000, "end": 12000}])
    assert t.engine == "mlx" and t.device == "gpu"
    assert t.transcribe("/fake.wav") == ("Bonjour", "fr")
    # Only the 8000 samples of speech went to the model
    assert fake.calls == [(8000, "mlx-community/whisper-small-mlx")]


def test_mlx_skips_the_model_when_there_is_no_speech(apple_silicon, monkeypatch):
    t, fake = _mlx_transcriber(monkeypatch, {"segments": []}, [])
    assert t.transcribe("/fake.wav") == ("", "")
    assert fake.calls == []

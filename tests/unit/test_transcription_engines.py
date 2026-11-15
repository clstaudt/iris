"""Unit tests for picking and running the speech recognition engine."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from iris import transcription
from iris.transcription import (
    WhisperTranscriber,
    language_heard,
    model_repo,
    resolve_engine,
    resolve_model,
)


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
        self.languages = []  # the language each was given

    def transcribe(self, audio, path_or_hf_repo, **kwargs):
        self.calls.append((len(audio), path_or_hf_repo))
        self.languages.append(kwargs.get("language"))
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
    fake.languages.clear()
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
    assert t.transcribe("/fake.wav", "it") == ("", "")
    assert fake.calls == []


# ---------------------------------------------------------------------------
# The language what is said is written in
# ---------------------------------------------------------------------------

# Whisper's likelihoods for Italian said with an accent, which it found likelier
# English, and for English said clearly
UNSURE = {"en": 0.52, "it": 0.31, "es": 0.08, "pt": 0.05}
CLEAR = {"en": 0.96, "it": 0.012, "de": 0.01}
# What Whisper writes in each language: in English, a translation
WRITTEN = {
    "it": " Il ristorante è molto popolare.",
    "en": " The restaurant is very popular.",
}


def test_an_unsure_language_keeps_to_the_conversations():
    assert language_heard(UNSURE, "it") == "it"


def test_a_clearly_likelier_language_is_followed():
    assert language_heard(CLEAR, "it") == "en"
    assert language_heard({"de": 0.9, "it": 0.05}, "it") == "de"


def test_the_conversations_language_heard_as_likeliest_is_kept():
    assert language_heard({"it": 0.6, "es": 0.3}, "it") == "it"


def test_a_language_the_user_did_not_choose_must_be_clearer_still():
    """Italian with an accent was taken for Spanish six times likelier."""
    spanish = {"es": 0.6, "it": 0.1, "pt": 0.1}
    assert language_heard(spanish, "it", {"en", "it"}) == "it"
    assert language_heard(spanish, "it", {"en", "it", "es"}) == "es"
    assert language_heard(spanish, "it") == "es"
    # Spanish said clearly is followed all the same
    assert language_heard({"es": 0.93, "it": 0.02}, "it", {"en", "it"}) == "es"


def test_the_first_utterance_is_in_the_likeliest_language():
    """With no conversation yet, Whisper goes by what it hears alone."""
    assert language_heard(UNSURE, None) == "en"
    assert language_heard({"es": 0.6, "it": 0.1}, None, {"en", "it"}) == "es"


class FakeFasterWhisper:
    """Stands in for faster-whisper's model: finds the languages as likely as
    given, and writes in the language it is given, or the likeliest. Nothing
    is decoded until the segments are read, as in faster-whisper."""

    def __init__(self, likelihood):
        self.likelihood = likelihood
        self.asked = []  # the language each transcription was given
        self.decoded = []  # the languages decoded in

    def transcribe(self, audio, beam_size, language, vad_filter):
        self.asked.append(language)
        ranked = sorted(self.likelihood.items(), key=lambda item: -item[1])
        written = language or ranked[0][0]
        info = SimpleNamespace(language=written, all_language_probs=None if language else ranked)

        def segments():
            self.decoded.append(written)
            yield SimpleNamespace(text=WRITTEN[written], no_speech_prob=0.02)

        return segments(), info


def _faster_whisper(monkeypatch, likelihood):
    fake = FakeFasterWhisper(likelihood)
    monkeypatch.setattr(transcription, "WhisperModel", lambda *args, **kwargs: fake)
    return WhisperTranscriber(model="small", device="cpu", engine="faster-whisper"), fake


def test_faster_whisper_writes_an_unsure_utterance_in_the_conversations_language(monkeypatch):
    t, fake = _faster_whisper(monkeypatch, UNSURE)
    assert t.transcribe("/fake.wav", "it") == ("Il ristorante è molto popolare.", "it")
    # English was never decoded
    assert fake.asked == [None, "it"] and fake.decoded == ["it"]


def test_faster_whisper_follows_a_clear_switch(monkeypatch):
    t, fake = _faster_whisper(monkeypatch, CLEAR)
    assert t.transcribe("/fake.wav", "it") == ("The restaurant is very popular.", "en")
    assert fake.asked == [None]


def test_faster_whisper_hears_the_first_utterance_as_before(monkeypatch):
    t, fake = _faster_whisper(monkeypatch, UNSURE)
    assert t.transcribe("/fake.wav") == ("The restaurant is very popular.", "en")
    assert fake.asked == [None]


def _mlx_hearing(monkeypatch, likelihood):
    """MLX Whisper finding the languages as likely as given."""
    result = {"language": "en", "segments": [{"text": WRITTEN["en"], "no_speech_prob": 0.02}]}
    t, fake = _mlx_transcriber(monkeypatch, result, [{"start": 4000, "end": 12000}])
    told = []

    def likelihood_of(self, speech):
        told.append(len(speech))
        return likelihood

    monkeypatch.setattr(WhisperTranscriber, "_mlx_likelihood", likelihood_of)
    return t, fake, told


def test_mlx_writes_an_unsure_utterance_in_the_conversations_language(apple_silicon, monkeypatch):
    t, fake, told = _mlx_hearing(monkeypatch, UNSURE)
    t.transcribe("/fake.wav", "it")
    # Told from the speech alone, and decoded in Italian
    assert told == [8000] and fake.languages == ["it"]


def test_mlx_weighs_the_languages_the_user_chose(apple_silicon, monkeypatch):
    t, fake, _ = _mlx_hearing(monkeypatch, {"es": 0.6, "it": 0.1})
    t.transcribe("/fake.wav", "it", {"en", "it"})
    t.transcribe("/fake.wav", "it", {"en", "it", "es"})
    assert fake.languages == ["it", "es"]


def test_mlx_hears_english_with_an_english_only_model(apple_silicon, monkeypatch):
    """A local model folder may hold one, which tells no language."""
    result = {"language": "en", "segments": [{"text": WRITTEN["en"], "no_speech_prob": 0.02}]}
    t, fake = _mlx_transcriber(monkeypatch, result, [{"start": 4000, "end": 12000}])
    english_only = SimpleNamespace(is_multilingual=False)
    holder = SimpleNamespace(get_model=lambda path, dtype: english_only)
    monkeypatch.setattr(transcription, "ModelHolder", holder, raising=False)
    monkeypatch.setattr(transcription, "mx", SimpleNamespace(float16="float16"), raising=False)
    t.transcribe("/fake.wav", "it", {"en", "it"})
    assert fake.languages == ["en"]


def test_mlx_follows_a_clear_switch(apple_silicon, monkeypatch):
    t, fake, _ = _mlx_hearing(monkeypatch, CLEAR)
    t.transcribe("/fake.wav", "it")
    assert fake.languages == ["en"]


def test_mlx_hears_the_first_utterance_as_before(apple_silicon, monkeypatch):
    t, fake, told = _mlx_hearing(monkeypatch, UNSURE)
    assert t.transcribe("/fake.wav") == ("The restaurant is very popular.", "en")
    # Whisper tells the language itself, as it decodes
    assert told == [] and fake.languages == [None]


# ---------------------------------------------------------------------------
# The language what is said is written in: measured
# ---------------------------------------------------------------------------

# How likely Whisper found each language in sentences said by synthesized
# voices, with each engine's default model (fixtures/whisper_languages.json)
HEARD = json.loads((Path(__file__).parent / "fixtures" / "whisper_languages.json").read_text())
# The user's languages in the measurements
USERS = {"en", "it"}


def _heard(engine: str, said: str, accented: bool | None = None) -> list[dict[str, float]]:
    """Whisper's likelihoods for each clip in a language, said with an accent
    (by a voice made for another language) or not, or either."""
    found = []
    for clip, cell in HEARD[engine]["clips"].items():
        language, voice = clip.split("-")[0], "-".join(clip.split("-")[1:3])
        if language == said and accented in (None, HEARD["made_for"][voice] != said):
            found.append(cell["p"])
    return found


def _wrong(likelihoods, expected: str, conversation: str | None) -> tuple[int, int]:
    """How many Whisper takes for another language by itself, and how many
    when told the conversation's language and the user's."""
    alone = sum(max(p, key=p.get) != expected for p in likelihoods)
    told = sum(language_heard(p, conversation, USERS) != expected for p in likelihoods)
    return alone, told


@pytest.mark.parametrize(
    "engine, alone, told", [("mlx", 38, 16), ("faster-whisper", 58, 38)]
)  # fmt: skip
def test_accented_italian_is_written_in_italian_far_more_often(engine, alone, told):
    """Of 195 Italian sentences said by voices made for other languages,
    Whisper took this many for another language by itself, and this many in
    an Italian conversation."""
    assert _wrong(_heard(engine, "it", accented=True), "it", "it") == (alone, told)


@pytest.mark.parametrize("engine", ["mlx", "faster-whisper"])
def test_a_switch_to_one_of_the_users_languages_is_followed_as_before(engine):
    """English: 48 sentences, by voices made for it and for others."""
    likelihoods = _heard(engine, "en")
    alone, told = _wrong(likelihoods, "en", "it")
    assert told == alone < len(likelihoods) / 10


@pytest.mark.parametrize(
    "engine, alone, told", [("mlx", 2, 2), ("faster-whisper", 6, 6)]
)  # fmt: skip
def test_a_switch_to_another_language_is_followed_as_before(engine, alone, told):
    """German, French, Spanish and Portuguese, which the user did not
    choose: 92 sentences, by voices made for them and for others."""
    each = [_wrong(_heard(engine, said), said, "it") for said in ("de", "fr", "es", "pt")]
    assert tuple(map(sum, zip(*each))) == (alone, told)


@pytest.mark.parametrize(
    "engine, alone, told", [("mlx", 38, 38), ("faster-whisper", 58, 61)]
)  # fmt: skip
def test_switching_to_italian_with_an_accent_from_english(engine, alone, told):
    """Leaving the conversation's English for Italian takes Whisper finding
    Italian twice as likely: faster-whisper loses 3 switches of 195 that it
    heard by itself."""
    assert _wrong(_heard(engine, "it", accented=True), "it", "en") == (alone, told)

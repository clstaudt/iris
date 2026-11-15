"""Unit tests for speech generation with Pocket TTS, and the engine switch."""

import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from iris import tts
from iris.tts import (
    KokoroGenerator,
    PocketGenerator,
    SpeechGenerator,
    SpeechPipeline,
    list_voices,
    pocket_sentences,
    voice_language,
    voice_languages,
)


@pytest.fixture
def mock_tts_model():
    with patch("iris.tts.TTSModel") as mock:
        model = mock.load_model.return_value
        model.sample_rate = 24000
        model.generate_audio.return_value = torch.zeros(100)
        yield mock


@pytest.fixture
def mock_kokoro():
    with patch.object(tts, "Kokoro") as mock, patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        mock.return_value.create.return_value = (np.zeros(100, dtype="float32"), 24000)
        yield mock


def test_english_uses_the_voice(mock_tts_model):
    gen = PocketGenerator(voice="marius")
    result = gen.generate("Hello there.", language="en")

    assert result is not None
    mock_tts_model.load_model.assert_called_with(language="english")
    mock_tts_model.load_model.return_value.get_state_for_audio_prompt.assert_called_with("marius")
    os.unlink(result)


def test_another_language_loads_its_model_with_the_same_voice(mock_tts_model):
    """Spoken by Estelle, the French voice, Anna's French replies were
    another woman's voice."""
    gen = PocketGenerator(voice="anna")
    result = gen.generate("Bonjour.", language="fr")

    assert result is not None
    mock_tts_model.load_model.assert_called_with(language="french")
    mock_tts_model.load_model.return_value.get_state_for_audio_prompt.assert_called_with("anna")
    os.unlink(result)


def test_a_language_without_a_model_is_not_spoken(mock_tts_model):
    gen = PocketGenerator(voice="alba")
    assert gen.generate("こんにちは", language="ja") is None
    assert not mock_tts_model.load_model.return_value.generate_audio.called


def test_models_and_voices_load_once(mock_tts_model):
    gen = PocketGenerator(voice="alba")
    for _ in range(3):
        os.unlink(gen.generate("Hello.", language="en"))

    assert mock_tts_model.load_model.call_count == 1
    assert mock_tts_model.load_model.return_value.get_state_for_audio_prompt.call_count == 1


def test_a_failed_generation_returns_none(mock_tts_model):
    gen = PocketGenerator(voice="alba")
    mock_tts_model.load_model.return_value.generate_audio.side_effect = RuntimeError("boom")
    assert gen.generate("Hello.", language="en") is None


def test_audio_is_written_at_the_model_rate(mock_tts_model):
    mock_tts_model.load_model.return_value.generate_audio.return_value = torch.ones(2400) * 0.1
    gen = PocketGenerator(voice="alba")
    result = gen.generate("Hello.", language="en")

    audio, rate = tts.sf.read(result, dtype="float32")
    assert rate == 24000
    assert np.allclose(audio, 0.1, atol=1e-3)
    os.unlink(result)


# ---------------------------------------------------------------------------
# SpeechGenerator: the persona's engine
# ---------------------------------------------------------------------------


def test_only_the_persona_engine_loads_at_start(mock_tts_model, mock_kokoro):
    SpeechGenerator("kyutai", "alba")
    assert mock_tts_model.load_model.called
    assert not mock_kokoro.called


def test_a_kokoro_persona_speaks_with_kokoro(mock_tts_model, mock_kokoro):
    gen = SpeechGenerator("kokoro", "bm_lewis", speed=1.1)
    os.unlink(gen.generate("Good afternoon.", language="en"))

    kwargs = mock_kokoro.return_value.create.call_args.kwargs
    assert kwargs["voice"] == "bm_lewis"
    assert kwargs["speed"] == 1.1
    assert kwargs["lang"] == "en-gb"
    assert not mock_tts_model.load_model.called


def test_switching_engine_switches_voice(mock_tts_model, mock_kokoro):
    gen = SpeechGenerator("kokoro", "af_sarah")
    gen.engine, gen.voice = "kyutai", "george"
    os.unlink(gen.generate("Hello.", language="en"))

    mock_tts_model.load_model.return_value.get_state_for_audio_prompt.assert_called_with("george")


def test_kyutai_leaves_japanese_to_no_other_engine(mock_tts_model, mock_kokoro):
    """Kokoro's Japanese voice spoke a Kyutai persona's Japanese replies:
    the voice changed in the middle of the conversation."""
    gen = SpeechGenerator("kyutai", "alba")
    assert gen.generate("こんにちは", language="ja") is None

    assert not mock_kokoro.called
    assert not mock_tts_model.load_model.return_value.generate_audio.called


def test_kokoro_leaves_german_to_no_other_engine(mock_tts_model, mock_kokoro):
    gen = SpeechGenerator("kokoro", "af_sarah")
    assert gen.generate("Guten Tag.", language="de") is None

    assert not mock_tts_model.load_model.called
    assert not mock_kokoro.return_value.create.called


def test_a_preview_voice_does_not_change_the_persona(mock_tts_model, mock_kokoro):
    gen = SpeechGenerator("kyutai", "alba")
    os.unlink(gen.generate("Hi.", engine="kokoro", voice="am_adam", speed=0.9))

    assert mock_kokoro.return_value.create.call_args.kwargs["voice"] == "am_adam"
    assert (gen.engine, gen.voice) == ("kyutai", "alba")


# ---------------------------------------------------------------------------
# Voices offered in the persona editor
# ---------------------------------------------------------------------------


POCKET_SEVEN = ["en", "fr", "it", "de", "es", "pt", "nl"]


def test_kyutai_voices_include_one_per_further_language():
    voices = {v["id"]: v for v in list_voices("kyutai")}
    assert voices["anna"] == {
        "id": "anna", "name": "Anna", "language": "English", "languages": POCKET_SEVEN,
    }
    assert voices["bill_boerst"]["name"] == "Bill Boerst"
    assert voices["estelle"]["language"] == "French"


def test_every_kyutai_voice_speaks_pocket_tts_languages_but_french_not_always():
    """Not only the language it was made for: each language's model takes
    the voice's own prompt. In French about half of the voices made for
    another language keep too much of their accent (Alba's French was
    heard as English or Czech one time in five). The language a voice was
    made for is listed first."""
    for voice in list_voices("kyutai"):
        assert set(voice["languages"]) >= set(POCKET_SEVEN) - {"fr"}, voice["id"]
        assert voice["languages"][0] == voice_language("kyutai", voice["id"])
    assert voice_languages("kyutai", "juergen") == POCKET_SEVEN
    assert "fr" not in voice_languages("kyutai", "alba")


def test_kokoro_voices_come_from_its_voices_file(tmp_path):
    np.savez(tmp_path / "voices.npz", af_bella=np.zeros(1), bm_lewis=np.zeros(1))
    (tmp_path / "voices.npz").rename(tmp_path / tts.KOKORO_VOICES_FILE)

    voices = list_voices("kokoro", tmp_path)
    assert [(v["id"], v["name"], v["language"]) for v in voices] == [
        ("af_bella", "Bella", "American English"),
        ("bm_lewis", "Lewis", "British English"),
    ]
    assert all(voice["languages"][0] == "en" for voice in voices)


def test_no_kokoro_voices_before_it_is_downloaded(tmp_path):
    assert list_voices("kokoro", tmp_path) == []


def test_a_voice_knows_its_language():
    assert voice_language("kokoro", "ff_siwis") == "fr"
    assert voice_language("kokoro", "bm_lewis") == "en"
    assert voice_language("kyutai", "giovanni") == "it"
    assert voice_language("kyutai", "alba") == "en"


# ---------------------------------------------------------------------------
# Previews through the speech pipeline
# ---------------------------------------------------------------------------


class RecordingGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, text, language="en", **voice):
        self.calls.append((text, language, voice))
        return "/tmp/none.wav"


class RecordingPlayer:
    def __init__(self):
        self.calls = []

    def play(self, audio_file, on_level=None, cancelled=None, **effect):
        self.calls.append(effect)

    def stop(self):
        pass


def test_a_preview_is_spoken_in_its_own_voice_and_effect():
    generator, player = RecordingGenerator(), RecordingPlayer()
    pipeline = SpeechPipeline(generator, player)
    sound = {"engine": "kokoro", "voice": "bm_lewis", "speed": 1.1, "audio_filter": "radio"}
    pipeline.queue_speech("Hello.", language="en", sound=sound)
    pipeline.queue_speech("A reply.")
    pipeline.wait_for_speech()

    assert generator.calls == [
        ("Hello.", "en", {"engine": "kokoro", "voice": "bm_lewis", "speed": 1.1}),
        ("A reply.", "en", {}),
    ]
    assert type(player.calls[0]["audio_filter"]).__name__ == "RadioFilter"
    assert player.calls[1] == {}


def test_a_preview_without_effect_plays_unfiltered():
    player = RecordingPlayer()
    pipeline = SpeechPipeline(RecordingGenerator(), player)
    pipeline.queue_speech("Hello.", sound={"engine": "kyutai", "voice": "alba", "audio_filter": ""})
    pipeline.wait_for_speech()

    assert player.calls == [{"audio_filter": None}]


def test_which_languages_a_voice_speaks():
    assert voice_languages("kyutai", "anna") == POCKET_SEVEN
    assert voice_languages("kyutai", "juergen") == POCKET_SEVEN
    # No other engine's voice lends a Kyutai voice Japanese, or a Kokoro one German
    assert "ja" not in voice_languages("kyutai", "anna")
    assert "de" not in voice_languages("kokoro", "af_sarah")
    assert "fr" in voice_languages("kokoro", "ff_siwis")
    assert voice_languages("nobody", "anna") == []


def test_a_conversation_may_be_in_any_language_iris_has_a_voice_for():
    """Also one the persona's voice does not speak: the reply is shown."""
    for engine, voice in [("kyutai", "anna"), ("kokoro", "bm_lewis")]:
        generator = SpeechGenerator(engine, voice, prepare=False)
        assert generator.languages() == {"en", "fr", "it", "de", "es", "pt", "nl", "hi", "ja", "zh"}
    assert not SpeechGenerator("kyutai", "anna", prepare=False).speaks("ja")
    assert not SpeechGenerator("kokoro", "bm_lewis", prepare=False).speaks("de")


# ---------------------------------------------------------------------------
# Text as Pocket TTS takes it best
# ---------------------------------------------------------------------------


def test_pocket_tts_is_given_one_sentence_at_a_time(mock_tts_model):
    gen = PocketGenerator(voice="juergen")
    os.unlink(gen.generate("Direkt heißt: kein Umweg. Keine Ausflüge, keine Umwege!", "de"))

    said = [c.args[1] for c in mock_tts_model.load_model.return_value.generate_audio.call_args_list]
    assert said == ["Direkt heißt: kein Umweg.", "Keine Ausflüge, keine Umwege!"]


def test_sentences_end_at_sentence_marks_not_abbreviations():
    assert pocket_sentences("Hallo. Wie geht's? Gut!\nUnd dir… Prima") == [
        "Hallo.",
        "Wie geht's?",
        "Gut!",
        "Und dir…",
        "Prima",
    ]
    assert pocket_sentences("Nimm z. B. das Rad. Es ist schnell.") == [
        "Nimm z. B. das Rad.",
        "Es ist schnell.",
    ]
    assert pocket_sentences("Ask Dr. Smith, e.g. Monday. Or not.") == [
        "Ask Dr. Smith, e.g. Monday.",
        "Or not.",
    ]


def test_dashes_become_pauses_and_hyphens_plain():
    assert pocket_sentences("Ich bin Felix—direkt und klar.") == ["Ich bin Felix, direkt und klar."]
    assert pocket_sentences("Kurz – und gut.") == ["Kurz, und gut."]
    assert pocket_sentences("graphics‑heavy tasks, 10–20 times") == [
        "graphics-heavy tasks, 10-20 times"
    ]
    assert pocket_sentences("— Ja.") == ["Ja."]


def test_nothing_to_say_says_nothing(mock_tts_model):
    assert PocketGenerator(voice="alba").generate(" — ", "en") is None

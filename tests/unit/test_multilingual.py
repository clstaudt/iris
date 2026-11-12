"""Unit tests for multilingual support."""

import os
import tempfile
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from kokoro_onnx.config import SUPPORTED_LANGUAGES
from pocket_tts.default_parameters import DEFAULT_VOICE_FOR_LANGUAGE

from iris.language import language_of
from iris.personas import BUILTIN_DIR, Persona
from iris.tts import (
    LANGUAGE_VOICE_MAP,
    POCKET_LANGUAGES,
    KokoroGenerator,
    SpeechGenerator,
    get_language_info,
    is_language_supported,
    voice_language,
)

# The languages Iris speaks its replies in, as Whisper codes
SPOKEN = ("en", "fr", "it", "de", "es", "pt", "nl", "ja", "zh")


# ---------------------------------------------------------------------------
# Language mapping helpers
# ---------------------------------------------------------------------------


class TestLanguageMap:
    def test_english_supported(self):
        assert is_language_supported("en") is True

    def test_french_supported(self):
        assert is_language_supported("fr") is True

    def test_italian_supported(self):
        assert is_language_supported("it") is True

    def test_japanese_supported(self):
        assert is_language_supported("ja") is True

    def test_chinese_supported(self):
        assert is_language_supported("zh") is True

    def test_german_not_supported(self):
        assert is_language_supported("de") is False

    def test_unknown_not_supported(self):
        assert is_language_supported("xx") is False

    def test_get_language_info_returns_code_and_voice(self):
        info = get_language_info("fr")
        assert info is not None
        assert info["code"] == "fr-fr"
        assert info["default_voice"] == "ff_siwis"

    def test_get_language_info_unsupported_returns_none(self):
        assert get_language_info("de") is None

    def test_all_map_entries_have_required_keys(self):
        for lang, info in LANGUAGE_VOICE_MAP.items():
            assert "code" in info, f"Missing 'code' for {lang}"
            assert "default_voice" in info, f"Missing 'default_voice' for {lang}"


# ---------------------------------------------------------------------------
# KokoroGenerator.generate with language parameter
# ---------------------------------------------------------------------------


class TestGenerateLanguage:
    """Test that generate() passes the correct lang/voice to kokoro."""

    @patch.object(KokoroGenerator, "_locate_models")
    @patch("iris.tts.Kokoro")
    def test_generate_french_uses_french_voice(self, MockKokoro, mock_locate):
        mock_locate.return_value = ("/fake/model.onnx", "/fake/voices.bin")
        mock_kokoro = MockKokoro.return_value
        mock_kokoro.create.return_value = (__import__("numpy").zeros(100, dtype="float32"), 24000)

        gen = KokoroGenerator(voice="af_bella", speed=1.0)
        result = gen.generate("Bonjour", language="fr")

        assert result is not None
        mock_kokoro.create.assert_called_once()
        call_kwargs = mock_kokoro.create.call_args
        assert call_kwargs.kwargs["lang"] == "fr-fr"
        assert call_kwargs.kwargs["voice"] == "ff_siwis"

        # Cleanup
        if result:
            os.unlink(result)

    @patch.object(KokoroGenerator, "_locate_models")
    @patch("iris.tts.Kokoro")
    def test_generate_japanese_uses_japanese_voice(self, MockKokoro, mock_locate):
        mock_locate.return_value = ("/fake/model.onnx", "/fake/voices.bin")
        mock_kokoro = MockKokoro.return_value
        mock_kokoro.create.return_value = (__import__("numpy").zeros(100, dtype="float32"), 24000)

        gen = KokoroGenerator(voice="af_bella", speed=1.0)
        result = gen.generate("こんにちは", language="ja")

        assert result is not None
        call_kwargs = mock_kokoro.create.call_args
        assert call_kwargs.kwargs["lang"] == "ja"
        assert call_kwargs.kwargs["voice"] == "jf_alpha"

        if result:
            os.unlink(result)

    @patch.object(KokoroGenerator, "_locate_models")
    @patch("iris.tts.Kokoro")
    def test_generate_unsupported_falls_back_to_english(self, MockKokoro, mock_locate):
        mock_locate.return_value = ("/fake/model.onnx", "/fake/voices.bin")
        mock_kokoro = MockKokoro.return_value
        mock_kokoro.create.return_value = (__import__("numpy").zeros(100, dtype="float32"), 24000)

        gen = KokoroGenerator(voice="af_bella", speed=1.0)
        result = gen.generate("Hallo", language="de")

        assert result is not None
        call_kwargs = mock_kokoro.create.call_args
        assert call_kwargs.kwargs["lang"] == "en-us"
        assert call_kwargs.kwargs["voice"] == "af_bella"  # falls back to instance voice

        if result:
            os.unlink(result)

    @patch.object(KokoroGenerator, "_locate_models")
    @patch("iris.tts.Kokoro")
    def test_generate_english_default(self, MockKokoro, mock_locate):
        mock_locate.return_value = ("/fake/model.onnx", "/fake/voices.bin")
        mock_kokoro = MockKokoro.return_value
        mock_kokoro.create.return_value = (__import__("numpy").zeros(100, dtype="float32"), 24000)

        gen = KokoroGenerator(voice="af_bella", speed=1.0)
        result = gen.generate("Hello")  # no language arg -> default "en"

        assert result is not None
        call_kwargs = mock_kokoro.create.call_args
        assert call_kwargs.kwargs["lang"] == "en-us"

        if result:
            os.unlink(result)


class TestPersonaVoice:
    """Personas must keep their own voice wherever it can speak the language."""

    @patch.object(KokoroGenerator, "_locate_models")
    @patch("iris.tts.Kokoro")
    def _generator(self, voice, MockKokoro, mock_locate):
        mock_locate.return_value = ("/fake/model.onnx", "/fake/voices.bin")
        return KokoroGenerator(voice=voice, speed=1.0)

    def test_english_keeps_the_persona_voice(self):
        assert self._generator("af_sarah").voice_for("en") == ("en-us", "af_sarah")

    def test_british_voice_gets_british_english(self):
        assert self._generator("bm_lewis").voice_for("en") == ("en-gb", "bm_lewis")

    def test_language_the_voice_cannot_speak_uses_its_default(self):
        assert self._generator("bm_lewis").voice_for("fr") == ("fr-fr", "ff_siwis")

    def test_voice_switch_applies_to_the_next_sentence(self):
        """Switching persona swaps generator.voice; the next reply must use it."""
        gen = self._generator("af_sarah")
        gen.voice = "bm_lewis"
        assert gen.voice_for("en") == ("en-gb", "bm_lewis")


# ---------------------------------------------------------------------------
# Who speaks each language, for each persona shipped with Iris
# ---------------------------------------------------------------------------

# (engine, voice, the engine's code for the language) a reply in each language
# is spoken with. Kokoro personas leave the languages kokoro-onnx cannot speak
# to Pocket TTS, Kyutai personas Japanese and Chinese to Kokoro.
KOKORO_ABROAD = {
    "fr": ("kokoro", "ff_siwis", "fr-fr"),
    "it": ("kokoro", "if_sara", "it"),
    "de": ("kyutai", "juergen", "german"),
    "es": ("kyutai", "lola", "spanish"),
    "pt": ("kyutai", "rafael", "portuguese"),
    "nl": ("kyutai", "daan", "dutch"),
    "ja": ("kokoro", "jf_alpha", "ja"),
    "zh": ("kokoro", "zf_xiaoni", "cmn"),
}
SPOKEN_WITH = {
    "computer": {"en": ("kokoro", "af_sarah", "en-us"), **KOKORO_ABROAD},
    "hal": {"en": ("kokoro", "bm_lewis", "en-gb"), **KOKORO_ABROAD},
    "iris": {
        "en": ("kyutai", "alba", "english"),
        "fr": ("kyutai", "estelle", "french"),
        "it": ("kyutai", "giovanni", "italian"),
        "de": ("kyutai", "juergen", "german"),
        "es": ("kyutai", "lola", "spanish"),
        "pt": ("kyutai", "rafael", "portuguese"),
        "nl": ("kyutai", "daan", "dutch"),
        "ja": ("kokoro", "jf_alpha", "ja"),
        "zh": ("kokoro", "zf_xiaoni", "cmn"),
    },
}
# The language each code of the engines stands for
CODE_LANGUAGE = {
    **{info["code"]: lang for lang, info in LANGUAGE_VOICE_MAP.items()},
    "en-gb": "en",
    **{model: lang for lang, model in POCKET_LANGUAGES.items()},
}


@pytest.fixture
def engines_used():
    """Kokoro and Pocket TTS stand-ins, which note the voice and the
    language code they are asked to speak with (no models, no sound)."""
    with patch("iris.tts.TTSModel") as pocket, patch("iris.tts.Kokoro") as kokoro, patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        model = pocket.load_model.return_value
        model.sample_rate = 24000
        model.generate_audio.return_value = torch.zeros(100)
        kokoro.return_value.create.return_value = (np.zeros(100, dtype="float32"), 24000)

        def spoken_with(persona_id: str, language: str) -> tuple[str, str, str]:
            pocket.reset_mock()
            kokoro.reset_mock()
            persona = Persona.from_file(BUILTIN_DIR / f"{persona_id}.yaml")
            generator = SpeechGenerator(persona.engine, persona.voice, persona.speed)
            os.unlink(generator.generate("Text.", language=language))
            if kokoro.return_value.create.called:
                spoken = kokoro.return_value.create.call_args.kwargs
                return "kokoro", spoken["voice"], spoken["lang"]
            # The model loaded last (English loads with a Kyutai persona)
            voice = model.get_state_for_audio_prompt.call_args.args[0]
            return "kyutai", voice, pocket.load_model.call_args.kwargs["language"]

        yield spoken_with


@pytest.mark.parametrize("persona_id", SPOKEN_WITH)
@pytest.mark.parametrize("language", SPOKEN)
def test_each_language_is_spoken_by_its_engine_voice_and_code(engines_used, persona_id, language):
    assert engines_used(persona_id, language) == SPOKEN_WITH[persona_id][language]


@pytest.mark.parametrize("persona_id", SPOKEN_WITH)
@pytest.mark.parametrize("language", SPOKEN)
def test_no_language_is_spoken_with_the_voice_or_code_of_another(
    engines_used, persona_id, language
):
    """Italian once came out of the Portuguese model, in its voice: it
    sounded Portuguese."""
    engine, voice, code = engines_used(persona_id, language)
    assert CODE_LANGUAGE[code] == language
    assert voice_language(engine, voice) == language


@pytest.mark.parametrize("persona_id", SPOKEN_WITH)
def test_a_language_no_voice_speaks_is_spoken_as_english(engines_used, persona_id):
    assert engines_used(persona_id, "ru") == SPOKEN_WITH[persona_id]["en"]


def test_kokoro_is_only_given_languages_kokoro_onnx_speaks():
    """kokoro-onnx refuses any other: a reply in it would not be heard."""
    codes = {info["code"] for info in LANGUAGE_VOICE_MAP.values()} | {"en-gb"}
    assert codes <= set(SUPPORTED_LANGUAGES)


def test_each_pocket_language_has_a_voice_of_its_own():
    for lang, model in POCKET_LANGUAGES.items():
        if lang != "en":
            assert voice_language("kyutai", DEFAULT_VOICE_FOR_LANGUAGE[model]) == lang


def test_borrowed_pocket_tts_loads_only_the_model_of_its_language():
    with patch("iris.tts.TTSModel") as pocket, patch("iris.tts.Kokoro"), patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        pocket.load_model.return_value.sample_rate = 24000
        pocket.load_model.return_value.generate_audio.return_value = torch.zeros(100)
        generator = SpeechGenerator("kokoro", "af_sarah")
        os.unlink(generator.generate("Guten Tag.", language="de"))

        assert pocket.load_model.call_args_list == [call(language="german")]


# ---------------------------------------------------------------------------
# The language a text is written in
# ---------------------------------------------------------------------------


class TestLanguageOf:
    @pytest.mark.parametrize(
        "language, text",
        [
            ("en", "What would you like to talk about today?"),
            ("fr", "De quoi voulez-vous parler aujourd'hui ?"),
            ("it", "Di cosa vorresti parlare oggi?"),
            ("de", "Worüber möchtest du heute sprechen?"),
            ("es", "¿De qué quieres hablar hoy?"),
            ("pt", "Sobre o que você quer falar hoje?"),
            ("nl", "Waar wil je vandaag over praten?"),
            ("ja", "今日は何について話したいですか？"),
            ("zh", "你今天想聊些什么？"),
        ],
    )
    def test_each_language_is_told(self, language, text):
        assumed = "de" if language == "en" else "en"
        assert language_of(text, assumed, SPOKEN) == language

    def test_italian_heard_as_portuguese_is_italian(self):
        assert language_of("Parliamo italiano.", "pt", SPOKEN) == "it"

    @pytest.mark.parametrize(
        "piece",
        [
            "Sì, certamente.",
            "Posso parlare italiano.",
            "Va bene, parliamo italiano.",
            "Di cosa vorresti parlare?",
        ],
    )
    def test_an_italian_reply_to_english_is_italian(self, piece):
        assert language_of(piece, "en", SPOKEN) == "it"

    @pytest.mark.parametrize(
        "text, assumed",
        [("Okay.", "de"), ("Ciao!", "en"), ("Sure.", "it"), ("Fitzcarraldo.", "en"),
         ("42", "fr"), ("...", "en"), ("", "en")],
    )  # fmt: skip
    def test_a_word_or_two_keep_the_language(self, text, assumed):
        assert language_of(text, assumed, SPOKEN) == assumed

    @pytest.mark.parametrize(
        "text",
        [
            "Werner Herzog is a German director, born in Munich in 1942.",
            "Who is Jürgen Klopp?",
            "Look up tickets for the Musée d'Orsay.",
            "Search for the opening hours of Schloss Neuschwanstein.",
        ],
    )
    def test_names_from_elsewhere_keep_a_sentence_english(self, text):
        assert language_of(text, "en", SPOKEN) == "en"

    @pytest.mark.parametrize("text", ["Опиши себя коротко.", "안녕하세요, 잘 지내세요?"])
    def test_a_language_none_speaks_is_not_taken_for_one_of_them(self, text):
        assert language_of(text, "en", SPOKEN) == "en"

    def test_only_the_languages_given_are_told(self):
        assert language_of("Worüber möchtest du heute sprechen?", "en", {"en", "fr"}) == "en"


# ---------------------------------------------------------------------------
# SpeechPipeline passes language through
# ---------------------------------------------------------------------------


class TestSpeechPipelineLanguage:
    def test_current_language_defaults_to_english(self):
        from iris.tts import SpeechPipeline

        mock_gen = MagicMock()
        mock_player = MagicMock()
        pipeline = SpeechPipeline(mock_gen, mock_player)
        assert pipeline.current_language == "en"
        pipeline.stop()

    def test_queue_speech_sends_language_tuple(self):
        from iris.tts import SpeechPipeline

        mock_gen = MagicMock()
        mock_player = MagicMock()
        pipeline = SpeechPipeline(mock_gen, mock_player)

        pipeline.current_language = "ja"
        pipeline.text_queue = MagicMock()
        pipeline.queue_speech("テスト")
        # (text, language, the speech epoch it was queued in)
        pipeline.text_queue.put.assert_called_once_with(("テスト", "ja", 0, None))
        pipeline.stop()


# ---------------------------------------------------------------------------
# Transcription returns language
# ---------------------------------------------------------------------------


class TestTranscriptionLanguage:
    @patch("iris.transcription.WhisperModel")
    def test_transcribe_returns_language(self, MockWhisper):
        from iris.transcription import WhisperTranscriber

        mock_info = MagicMock()
        mock_info.language = "fr"
        mock_segment = MagicMock()
        mock_segment.text = "Bonjour"
        mock_segment.no_speech_prob = 0.02
        MockWhisper.return_value.transcribe.return_value = ([mock_segment], mock_info)

        with patch("iris.transcription.torch", create=True) as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.cuda.is_available.return_value = False
            transcriber = WhisperTranscriber(
                model="tiny", device="cpu", engine="faster-whisper"
            )

        text, lang = transcriber.transcribe("/fake/audio.wav")
        assert text == "Bonjour"
        assert lang == "fr"

    @patch("iris.transcription.WhisperModel")
    def test_transcribe_drops_segments_rated_as_silence(self, MockWhisper):
        """Noise transcribed as "You" must not become a user turn."""
        from iris.transcription import WhisperTranscriber

        mock_info = MagicMock()
        mock_info.language = "en"
        phantom = MagicMock()
        phantom.text = "You"
        phantom.no_speech_prob = 0.9
        MockWhisper.return_value.transcribe.return_value = ([phantom], mock_info)

        transcriber = WhisperTranscriber(
            model="tiny", device="cpu", engine="faster-whisper"
        )
        text, _ = transcriber.transcribe("/fake/audio.wav")
        assert text == ""


# ---------------------------------------------------------------------------
# LLM language instruction
# ---------------------------------------------------------------------------


class TestLLMLanguageInstruction:
    @patch("iris.llm.ollama")
    def test_chat_stream_injects_language_for_non_english(self, mock_ollama):
        mock_ollama.show.return_value = MagicMock(
            modelinfo={}, details=MagicMock(family="test")
        )
        mock_ollama.list.return_value = []

        # Make chat return a simple stream
        mock_ollama.chat.return_value = iter(
            [{"message": {"content": "Bonjour!"}}]
        )

        from iris.llm import ConversationManager

        cm = ConversationManager(model="test", tools=None)
        # Consume the generator
        list(cm.chat_stream("Salut", response_language="fr"))

        # Check the messages passed to ollama.chat
        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        # Told with the user's message: a system message is only ever first
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "Respond in French" in messages[-1]["content"]
        assert messages[-1]["content"].endswith("Salut")

    @patch("iris.llm.ollama")
    def test_chat_stream_no_language_instruction_for_english(self, mock_ollama):
        mock_ollama.show.return_value = MagicMock(
            modelinfo={}, details=MagicMock(family="test")
        )
        mock_ollama.list.return_value = []
        mock_ollama.chat.return_value = iter(
            [{"message": {"content": "Hello!"}}]
        )

        from iris.llm import ConversationManager

        cm = ConversationManager(model="test", tools=None)
        list(cm.chat_stream("Hi", response_language="en"))

        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        # No language instruction should be injected for English
        lang_instructions = [
            m for m in messages
            if m["role"] == "system" and "speaking in" in m.get("content", "").lower()
        ]
        assert len(lang_instructions) == 0

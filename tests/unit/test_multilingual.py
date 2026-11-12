"""Unit tests for multilingual support."""

import os
import re
import tempfile
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from kokoro_onnx.config import SUPPORTED_LANGUAGES
from kokoro_onnx.tokenizer import Tokenizer
from pocket_tts.default_parameters import DEFAULT_VOICE_FOR_LANGUAGE

from iris.controller import PREVIEW_LINES, AssistantController
from iris.events import EventBus
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
SPOKEN = ("en", "fr", "it", "de", "es", "pt", "nl", "hi", "ja", "zh")


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
        assert self._generator("ef_dora").voice_for("it") == ("it", "if_sara")

    @pytest.mark.parametrize(
        "voice",
        ["ef_dora", "em_alex", "pf_dora", "pm_alex", "hf_alpha", "hm_omega", "ff_siwis",
         "if_sara", "jf_alpha", "zf_xiaoni"],
    )  # fmt: skip
    def test_a_voice_of_another_language_speaks_english_itself(self, voice):
        """Every Kokoro voice speaks English clearly; the persona keeps it."""
        assert self._generator(voice).voice_for("en") == ("en-us", voice)

    @pytest.mark.parametrize(
        "voice, language, code",
        [("ef_dora", "es", "es"), ("em_alex", "es", "es"), ("pf_dora", "pt", "pt-br"),
         ("pm_alex", "pt", "pt-br"), ("hf_alpha", "hi", "hi"), ("hm_omega", "hi", "hi")],
    )  # fmt: skip
    def test_a_spanish_portuguese_or_hindi_voice_speaks_its_language_itself(
        self, voice, language, code
    ):
        assert self._generator(voice).voice_for(language) == (code, voice)

    def test_voice_switch_applies_to_the_next_sentence(self):
        """Switching persona swaps generator.voice; the next reply must use it."""
        gen = self._generator("af_sarah")
        gen.voice = "bm_lewis"
        assert gen.voice_for("en") == ("en-gb", "bm_lewis")


# ---------------------------------------------------------------------------
# Who speaks each language, for each persona shipped with Iris
# ---------------------------------------------------------------------------

# (engine, voice, the engine's code for the language) a reply in each language
# is spoken with. Kokoro personas leave German and Dutch, which Kokoro has no
# voices for, to Pocket TTS; Kyutai personas Hindi, Japanese and Chinese to
# Kokoro.
KOKORO_ABROAD = {
    "fr": ("kokoro", "ff_siwis", "fr-fr"),
    "it": ("kokoro", "if_sara", "it"),
    "de": ("kyutai", "juergen", "german"),
    "es": ("kokoro", "ef_dora", "es"),
    "pt": ("kokoro", "pf_dora", "pt-br"),
    "nl": ("kyutai", "daan", "dutch"),
    "hi": ("kokoro", "hf_alpha", "hi"),
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
        "hi": ("kokoro", "hf_alpha", "hi"),
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
# Kokoro's voices for the languages kokoro-onnx refuses by name
KOKORO_ESPEAK_VOICES = ["ef_dora", "em_alex", "pf_dora", "pm_alex", "hf_alpha", "hm_omega"]


class PreviewPipeline:
    """Takes what a voice preview queues, instead of speaking it."""

    def __init__(self):
        self.queued = []

    def stop(self):
        pass

    def queue_speech(self, text, language=None, sound=None):
        self.queued.append((text, language, sound))


@pytest.fixture
def engines_used():
    """Kokoro and Pocket TTS stand-ins, which note the voice and the
    language code they are asked to speak with (no models, no sound).

    Yields spoken_with(persona_id, language) for a shipped persona, and
    spoken_with(engine, language, voice=...) for a persona on that voice;
    preview holds the engine, voice and speed SpeechGenerator.generate is
    given for a voice preview."""
    with patch("iris.tts.TTSModel") as pocket, patch("iris.tts.Kokoro") as kokoro, patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        model = pocket.load_model.return_value
        model.sample_rate = 24000
        model.generate_audio.return_value = torch.zeros(100)
        kokoro.return_value.create.return_value = (np.zeros(100, dtype="float32"), 24000)
        phonemize = kokoro.return_value.tokenizer.phonemize

        def spoken_with(who: str, language: str, voice: str = "", preview: dict | None = None):
            pocket.reset_mock()
            kokoro.reset_mock()
            if voice:
                generator = SpeechGenerator(who, voice)
            else:
                persona = Persona.from_file(BUILTIN_DIR / f"{who}.yaml")
                generator = SpeechGenerator(persona.engine, persona.voice, persona.speed)
            os.unlink(generator.generate("Text.", language=language, **(preview or {})))
            if kokoro.return_value.create.called:
                spoken = kokoro.return_value.create.call_args.kwargs
                if "phonemes" in spoken:
                    # Spoken from the phonemes espeak made in the language's code
                    assert spoken["phonemes"] is phonemize.return_value
                    return "kokoro", spoken["voice"], phonemize.call_args.args[1]
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


@pytest.mark.parametrize("voice", KOKORO_ESPEAK_VOICES)
def test_a_persona_on_a_spanish_portuguese_or_hindi_voice_is_heard(engines_used, voice):
    """These voices were never heard: kokoro-onnx refused their languages,
    and af_bella spoke in their place, in English as in their own language."""
    language = voice_language("kokoro", voice)
    code = LANGUAGE_VOICE_MAP[language]["code"]
    assert engines_used("kokoro", language, voice=voice) == ("kokoro", voice, code)
    assert engines_used("kokoro", "en", voice=voice) == ("kokoro", voice, "en-us")


@pytest.mark.parametrize("voice", KOKORO_ESPEAK_VOICES + ["ff_siwis", "bm_lewis"])
def test_a_preview_says_its_line_in_the_voice_itself_and_its_language(engines_used, voice):
    controller = AssistantController(EventBus())
    controller.assistant = MagicMock()
    controller.assistant.speech_pipeline = pipeline = PreviewPipeline()
    controller.persona_preview({"name": "Nova", "engine": "kokoro", "voice": voice})

    [(line, language, sound)] = pipeline.queued
    assert language == voice_language("kokoro", voice)
    assert line == PREVIEW_LINES[language].format(name="Nova")
    # Rendered as the speech pipeline renders it, in the voice and language given
    preview = {key: sound[key] for key in ("engine", "voice", "speed")}
    engine, spoken, code = engines_used("iris", language, preview=preview)
    assert (engine, spoken, CODE_LANGUAGE[code]) == ("kokoro", voice, language)


def test_kokoro_onnx_is_only_given_languages_it_takes_and_phonemes_for_the_rest():
    """kokoro-onnx refuses any other language by name: a reply in it would
    not be heard."""
    with patch("iris.tts.Kokoro") as kokoro, patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        kokoro.return_value.create.return_value = (np.zeros(100, dtype="float32"), 24000)
        phonemize = kokoro.return_value.tokenizer.phonemize
        generator = KokoroGenerator(voice="af_sarah")
        for language, info in LANGUAGE_VOICE_MAP.items():
            kokoro.reset_mock()
            os.unlink(generator.generate("Text.", language=language))
            spoken = kokoro.return_value.create.call_args.kwargs
            assert spoken["lang"] in SUPPORTED_LANGUAGES
            if info["code"] not in SUPPORTED_LANGUAGES:
                phonemize.assert_called_once_with("Text.", info["code"])
                assert spoken["phonemes"] is phonemize.return_value


@pytest.mark.parametrize(
    "code, text, word",
    [("fr-fr", "J'aime le football.", "fˈʊtbɔːl"), ("hi", "मैं Nova हूँ।", "nˈəʊvə"),
     ("cmn", "我喜欢Queen的歌。", "kwˈiːn")],
)  # fmt: skip
def test_a_word_espeak_reads_as_english_is_said_without_the_switch(code, text, word):
    """espeak marks where it reads a word as English, "(en)fˈʊtbɔːl(fr)";
    Kokoro said the marks: "en football fr". Real espeak, through Kokoro's
    tokenizer as Iris sets it up."""
    try:
        phonemes = Tokenizer().phonemize(text, code)
    except RuntimeError as error:
        pytest.skip(f"espeak cannot run here: {error}")
    assert word in phonemes
    assert "en" + word not in phonemes


@pytest.mark.xfail(
    strict=True,
    reason="espeak reads no kanji: Kokoro needs them in kana, which needs a kanji reader "
    "(a new dependency, such as pykakasi)",
)
def test_a_japanese_reply_is_read_not_called_chinese_letters():
    """Kokoro speaks Japanese from espeak's phonemes, and espeak calls every
    kanji a "Chinese letter", in English: a Japanese reply is gibberish
    ("Chinese letter, Chinese letter, no Chinese letter..."). The text Iris
    gives Kokoro, through real espeak as Kokoro's tokenizer runs it."""
    with patch("iris.tts.Kokoro") as kokoro, patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        kokoro.return_value.create.return_value = (np.zeros(100, dtype="float32"), 24000)
        os.unlink(KokoroGenerator(voice="jf_alpha").generate("東京の天気は晴れです。", language="ja"))
        given = kokoro.return_value.create.call_args.kwargs
    try:
        phonemes = Tokenizer().phonemize(given["text"], given["lang"])
    except RuntimeError as error:
        pytest.skip(f"espeak cannot run here: {error}")
    assert "tʃˈaɪniːz" not in phonemes and "dʒˈapəniːz" not in phonemes, phonemes


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


def test_the_voice_for_a_language_is_prepared_once_ahead_of_its_first_reply():
    """A Kyutai persona's first Japanese reply waited three seconds for
    Kokoro and espeak's Japanese to load; prepared while the model thinks,
    they are ready when the reply comes."""
    with patch("iris.tts.TTSModel") as pocket, patch("iris.tts.Kokoro") as kokoro, patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        pocket.load_model.return_value.sample_rate = 24000
        pocket.load_model.return_value.generate_audio.return_value = torch.zeros(100)
        kokoro.return_value.create.return_value = (np.zeros(100, dtype="float32"), 24000)
        generator = SpeechGenerator("kyutai", "alba")
        assert not kokoro.called

        generator.prepare_language("ja")
        generator.prepare_language("ja")
        [spoken] = kokoro.return_value.create.call_args_list
        assert (spoken.kwargs["voice"], spoken.kwargs["lang"]) == ("jf_alpha", "ja")

        generator.prepare_language("de")
        assert call(language="german") in pocket.load_model.call_args_list
        # A language no voice speaks has nothing to prepare
        generator.prepare_language("ru")
        assert kokoro.return_value.create.call_count == 1


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

    @patch("iris.llm.ollama")
    def test_back_in_english_the_model_is_told_to_answer_in_english(self, mock_ollama):
        """After German turns, the model went on answering in German: the
        conversation it is given is German, and nothing named English."""
        mock_ollama.show.return_value = MagicMock(modelinfo={}, details=MagicMock(family="test"))
        mock_ollama.list.return_value = []
        mock_ollama.chat.side_effect = lambda **kwargs: iter([{"message": {"content": "Gut."}}])

        from iris.llm import ConversationManager

        cm = ConversationManager(model="test", tools=None)

        def told(prompt, language):
            list(cm.chat_stream(prompt, response_language=language))
            messages = mock_ollama.chat.call_args.kwargs["messages"]
            return re.findall(r"Respond in ([\w ()]+)\.", messages[-1]["content"])

        assert told("Hi", "en") == []
        assert told("Wie spät ist es?", "de") == ["German"]
        assert told("Und morgen?", "de") == ["German"]
        assert told("Back to English, please. What time is it?", "en") == ["English"]
        # Once back, English goes on without a note, as at the start
        assert told("And tomorrow?", "en") == []

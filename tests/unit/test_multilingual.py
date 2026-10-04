"""Unit tests for multilingual support."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from iris.tts import (
    LANGUAGE_VOICE_MAP,
    KokoroGenerator,
    get_language_info,
    is_language_supported,
)


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

        gen = KokoroGenerator(voice="af_bella", speed=1.0, quiet=True)
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

        gen = KokoroGenerator(voice="af_bella", speed=1.0, quiet=True)
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

        gen = KokoroGenerator(voice="af_bella", speed=1.0, quiet=True)
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

        gen = KokoroGenerator(voice="af_bella", speed=1.0, quiet=True)
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
        return KokoroGenerator(voice=voice, speed=1.0, quiet=True)

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
        pipeline.text_queue.put.assert_called_once_with(("テスト", "ja", 0))
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
            transcriber = WhisperTranscriber(model="tiny", device="cpu", quiet=True)

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

        transcriber = WhisperTranscriber(model="tiny", device="cpu", quiet=True)
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

        cm = ConversationManager(model="test", tools=None, quiet=True)
        # Consume the generator
        list(cm.chat_stream("Salut", response_language="fr"))

        # Check the messages passed to ollama.chat
        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msgs = [m for m in messages if m["role"] == "system"]
        lang_instructions = [m for m in system_msgs if "French" in m.get("content", "")]
        assert len(lang_instructions) == 1

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

        cm = ConversationManager(model="test", tools=None, quiet=True)
        list(cm.chat_stream("Hi", response_language="en"))

        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        # No language instruction should be injected for English
        lang_instructions = [
            m for m in messages
            if m["role"] == "system" and "speaking in" in m.get("content", "").lower()
        ]
        assert len(lang_instructions) == 0

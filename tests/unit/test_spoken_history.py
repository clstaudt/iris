"""A reply cut short is kept in the conversation as far as it was heard:
the model then knows what the user knows, as in Pipecat."""

import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

from iris.assistant import VoiceAssistant
from iris.audio import AudioPlayer
from iris.llm import ConversationManager
from iris.tts import SpeechPipeline, heard_part


class TestHeardPart:
    TEXT = "The lighthouse keeper rose with the dawn."

    def test_all_of_it(self):
        assert heard_part(self.TEXT, 1.0) == self.TEXT

    def test_none_of_it(self):
        assert heard_part(self.TEXT, 0.0) == ""

    def test_a_word_cut_in_half_was_not_heard(self):
        inside_rose = (self.TEXT.index("rose") + 2) / len(self.TEXT)
        assert heard_part(self.TEXT, inside_rose) == "The lighthouse keeper"

    def test_a_cut_between_words(self):
        end = self.TEXT.index(" rose")
        assert heard_part(self.TEXT, end / len(self.TEXT)) == "The lighthouse keeper"

    def test_less_than_the_first_word(self):
        assert heard_part(self.TEXT, 0.05) == ""

    def test_japanese_and_chinese_are_heard_up_to_the_cut(self):
        """They put no spaces between words: a clip cut short kept nothing
        of what was heard, as if the user had heard none of it."""
        japanese = "東京の明日の天気は晴れで、最高気温は十八度です。"
        assert heard_part(japanese, 0.5) == japanese[: len(japanese) // 2]
        chinese = "北京明天是晴天，最高气温十八度左右。"
        assert heard_part(chinese, 0.5) == chinese[: len(chinese) // 2]
        assert heard_part(chinese, 0.0) == ""

    def test_a_latin_word_in_japanese_is_still_not_cut_in_half(self):
        text = "今日はFootballの試合です。"
        inside = (text.index("Football") + 3) / len(text)
        assert heard_part(text, inside) == "今日は"


class InstantGenerator:
    def generate(self, text, language="en"):
        return tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name


class HeldPlayer:
    """Plays a clip until released, reporting how far it has got."""

    def __init__(self):
        self.playing = threading.Event()
        self.release = threading.Event()
        self.heard = 0.0
        self.played = 0

    def play(self, audio_file, on_level=None, cancelled=None):
        self.playing.set()
        while not self.release.wait(0.01):
            if cancelled is not None and cancelled():
                break
        self.playing.clear()
        self.played += 1

    def progress(self):
        return self.heard

    def stop(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass


class AnyVoiceGenerator(InstantGenerator):
    """Also speaks in a voice given for a preview."""

    def generate(self, text, language="en", **voice):
        return super().generate(text, language)


class AnyFilterPlayer(HeldPlayer):
    """Also plays a clip with a filter of its own, as a preview has."""

    def play(self, audio_file, on_level=None, cancelled=None, audio_filter=None):
        super().play(audio_file, on_level, cancelled)


def _wait(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


class TestSaid:
    def test_pieces_played_to_their_end(self):
        player = HeldPlayer()
        player.release.set()
        pipeline = SpeechPipeline(InstantGenerator(), player)
        pipeline.begin_reply()
        pipeline.queue_speech("The keeper rose with the dawn.")
        pipeline.queue_speech("A storm was coming.")
        pipeline.wait_for_speech()
        assert pipeline.said() == "The keeper rose with the dawn. A storm was coming."

    def test_a_stop_keeps_what_was_heard_of_the_clip(self):
        player = HeldPlayer()
        pipeline = SpeechPipeline(InstantGenerator(), player)
        pipeline.begin_reply()
        pipeline.queue_speech("The keeper rose with the dawn.")
        pipeline.queue_speech("A storm was coming.")
        assert player.playing.wait(2)
        player.heard = 0.5
        pipeline.stop()
        player.release.set()
        assert _wait(lambda: not player.playing.is_set())
        time.sleep(0.05)
        assert pipeline.said() == "The keeper rose"

    def test_a_preview_in_another_voice_is_not_part_of_the_reply(self):
        player = AnyFilterPlayer()
        player.release.set()
        pipeline = SpeechPipeline(AnyVoiceGenerator(), player)
        pipeline.begin_reply()
        pipeline.queue_speech("The keeper rose with the dawn.")
        pipeline.queue_speech(
            "Hi, I'm Vera.", sound={"engine": "kyutai", "voice": "vera", "audio_filter": ""}
        )
        pipeline.wait_for_speech()
        assert player.played == 2
        assert pipeline.said() == "The keeper rose with the dawn."

    def test_a_new_reply_starts_afresh(self):
        player = HeldPlayer()
        player.release.set()
        pipeline = SpeechPipeline(InstantGenerator(), player)
        pipeline.begin_reply()
        pipeline.queue_speech("Old reply.")
        pipeline.wait_for_speech()
        pipeline.begin_reply()
        assert pipeline.said() == ""


def test_player_progress_counts_from_when_the_clip_is_heard():
    player = AudioPlayer()
    assert player.progress() == 0.0
    # A 2 s clip at 1 kHz, its second half playing for 0.5 s already
    player._place = (2000, 1000, 1000, time.monotonic() - 0.5)
    assert abs(player.progress() - 0.75) < 0.05
    # Paused at three quarters
    player._place = (2000, 1000, 1500, None)
    assert player.progress() == 0.75
    # Not yet heard: still in the output latency
    player._place = (2000, 1000, 0, time.monotonic() + 0.2)
    assert player.progress() == 0.0


def _conversation(reply):
    with patch("iris.llm.ollama") as mock_ollama:
        mock_ollama.show.return_value = MagicMock(modelinfo={}, details=MagicMock(family="test"))
        mock_ollama.list.return_value = []
        conversation = ConversationManager(model="test")
    conversation.client = MagicMock()
    conversation.client.chat.return_value = iter([{"message": {"content": reply}}])
    return conversation


class TestKeptByTheCaller:
    def test_a_reply_streamed_without_keeping_is_not_kept(self):
        conversation = _conversation("It's sunny.")
        assert "".join(conversation.chat_stream("Weather?", keep=False)) == "It's sunny."
        assert conversation.conversation_history == []
        assert conversation.reply == "It's sunny."

    def test_the_caller_keeps_it(self):
        conversation = _conversation("It's sunny.")
        list(conversation.chat_stream("Weather?", keep=False))
        conversation.keep("Weather?", "It's", "voice")
        assert conversation.conversation_history == [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "It's"},
        ]

    def test_kept_as_before_by_default(self):
        conversation = _conversation("It's sunny.")
        list(conversation.chat_stream("Weather?"))
        assert conversation.conversation_history[-1] == {
            "role": "assistant", "content": "It's sunny.",
        }


class FakePipeline:
    def __init__(self, said):
        self._said = said

    def said(self):
        return self._said


def _assistant(said="", speak=True):
    assistant = object.__new__(VoiceAssistant)
    assistant.conversation = _conversation("")
    assistant.conversation.reply = "It's sunny and warm, a perfect day for the beach."
    assistant.speech_pipeline = FakePipeline(said)
    assistant.speak_replies = speak
    return assistant


def _kept(assistant):
    return assistant.conversation.conversation_history


class TestKeepExchange:
    ANSWER = "It's sunny and **warm**, a perfect day"

    def test_a_reply_heard_to_the_end(self):
        assistant = _assistant()
        assistant._keep_exchange("Weather?", self.ANSWER, "voice", cut=False)
        assert _kept(assistant)[-1]["content"] == assistant.conversation.reply

    def test_a_reply_cut_short_is_kept_as_far_as_it_was_said(self):
        assistant = _assistant(said="It's sunny and warm,")
        assistant._keep_exchange("Weather?", self.ANSWER, "voice", cut=True)
        assert _kept(assistant) == [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "It's sunny and warm,"},
        ]

    def test_with_the_voice_off_as_far_as_it_was_shown(self):
        assistant = _assistant(speak=False)
        assistant._keep_exchange("Weather?", self.ANSWER, "text", cut=True)
        assert _kept(assistant)[-1]["content"] == "It's sunny and warm, a perfect day."

    def test_nothing_said_nothing_kept(self):
        assistant = _assistant(said="")
        assistant._keep_exchange("Weather?", self.ANSWER, "voice", cut=True)
        assert _kept(assistant) == []

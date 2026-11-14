"""Unit tests for how VoiceAssistant takes typed turns and gates speech.

The assistant is built with object.__new__ so no models are loaded; only the
attributes the code under test touches are filled in.
"""

import queue
import threading

import pytest

from iris.assistant import VoiceAssistant
from iris.events import EventBus
from iris.tts import SpeechGenerator


class RecordingPipeline:
    def __init__(self):
        self.spoken = []
        self.languages = []

    def queue_speech(self, text, language=None):
        self.spoken.append(text)
        self.languages.append(language)


class PlainConversation:
    def clean_markdown(self, text):
        return text.replace("**", "")


def _bare_assistant():
    assistant = object.__new__(VoiceAssistant)
    assistant.speech_pipeline = RecordingPipeline()
    assistant.conversation = PlainConversation()
    assistant.turn_queue = queue.Queue()
    assistant.hold_event = threading.Event()
    assistant.speak_replies = True
    assistant.wake_word = None
    assistant.conversation_language = "en"
    assistant._reply_language = "en"
    assistant._turn_active = threading.Event()
    assistant.event_bus = EventBus()
    assistant.tts_generator = SpeechGenerator.__new__(SpeechGenerator)
    assistant.tts_generator.engine = "kyutai"
    return assistant


def test_speak_respects_voice_output_toggle():
    assistant = _bare_assistant()
    assistant._speak("**Hello** there")
    assert assistant.speech_pipeline.spoken == ["Hello there"]

    assistant.speak_replies = False
    assistant._speak("Not spoken")
    assert assistant.speech_pipeline.spoken == ["Hello there"]


def test_take_turn_is_fifo_and_non_blocking():
    assistant = _bare_assistant()
    assert assistant._take_turn() is None
    assistant.turn_queue.put({"text": "a", "id": "t1"})
    assistant.turn_queue.put({"text": "b", "id": "t2"})
    assert assistant._take_turn()["id"] == "t1"
    assert assistant._take_turn()["id"] == "t2"
    assert assistant._take_turn() is None


def test_text_turn_is_announced_as_typed_and_answered():
    assistant = _bare_assistant()
    events = []
    assistant.event_bus.subscribe(lambda event_type, data: events.append((event_type, data)))
    answered = []
    assistant.chat = lambda text, response_language, source=None, attachments=None: (
        answered.append((text, response_language, source))
    )

    assistant.conversation_language = "fr"
    assistant._run_text_turn({"text": "bonjour", "id": "t7"})

    assert events == [
        ("chat.user", {"text": "bonjour", "language": "fr", "source": "text", "id": "t7"})
    ]
    assert answered == [("bonjour", "fr", "text")]


def test_a_turn_marks_the_conversation_busy():
    """Background memory filing must see a typed turn as conversation."""
    assistant = _bare_assistant()
    seen = []
    assistant.chat = lambda text, response_language, source=None, attachments=None: (
        seen.append(assistant._turn_active.is_set())
    )
    assistant._run_text_turn({"text": "hi", "id": "t1"})
    assert seen == [True]
    assert not assistant._turn_active.is_set()


class QuietStatus:
    status_lock = threading.Lock()
    status_understanding = False


class Transcript:
    def __init__(self, text, language):
        self.heard = (text, language)

    def transcribe(self, audio_path):
        return self.heard


def _answer_heard(engine, text, language):
    """Answer a spoken turn with a persona on this engine; the reply language."""
    assistant = _bare_assistant()
    assistant.status_manager = QuietStatus()
    assistant.interrupt_event = threading.Event()
    assistant.transcriber = Transcript(text, language)
    assistant.tts_generator.engine = engine
    answered = []
    assistant.chat = lambda text, response_language, source=None: answered.append(
        response_language
    )
    assistant._answer_spoken("/tmp/heard.wav")
    return answered[0]


def test_a_kyutai_persona_answers_german_in_german():
    assert _answer_heard("kyutai", "Beschreibe dich kurz.", "de") == "de"


def test_a_kokoro_persona_answers_german_through_kyutai():
    assert _answer_heard("kokoro", "Beschreibe dich kurz.", "de") == "de"
    assert _answer_heard("kokoro", "Bonjour.", "fr") == "fr"


def test_a_language_no_voice_speaks_keeps_to_english():
    assert _answer_heard("kokoro", "Опиши себя коротко.", "ru") == "en"
    assert _answer_heard("kyutai", "Опиши себя коротко.", "ru") == "en"


def test_a_kyutai_persona_answers_japanese_through_kokoro():
    assert _answer_heard("kyutai", "こんにちは", "ja") == "ja"


def test_italian_that_whisper_takes_for_portuguese_is_answered_in_italian():
    """Whisper heard a short "Parliamo italiano." as Portuguese; the words
    are plainly Italian."""
    assert _answer_heard("kyutai", "Parliamo italiano.", "pt") == "it"
    assert _answer_heard("kokoro", "Parliamo italiano.", "pt") == "it"


def test_a_few_words_keep_to_the_language_whisper_heard():
    assert _answer_heard("kyutai", "Okay.", "de") == "de"
    assert _answer_heard("kyutai", "Ja.", "de") == "de"


def test_each_piece_of_the_reply_is_spoken_in_the_language_it_is_in():
    """Asked in English whether it speaks Italian, the model answers in
    Italian: that is spoken in Italian, not in English."""
    assistant = _bare_assistant()
    for piece in ["Sì, certamente.", "Posso parlare italiano.", "Ciao!"]:
        assistant._speak(piece)
    assistant._speak("Sorry, back to English: what would you like to talk about?")

    assert assistant.speech_pipeline.languages == ["it", "it", "it", "en"]


def test_a_kokoro_persona_speaks_a_german_reply_in_german():
    assistant = _bare_assistant()
    assistant.tts_generator.engine = "kokoro"
    assistant._speak("Ja, natürlich, das mache ich gern.")
    assert assistant.speech_pipeline.languages == ["de"]


def test_a_reply_in_a_language_no_voice_speaks_keeps_the_conversations():
    assistant = _bare_assistant()
    assistant._speak("Конечно, я могу говорить по-русски.")
    assert assistant.speech_pipeline.languages == ["en"]


def test_a_typed_turn_goes_on_in_the_language_typed():
    assistant = _bare_assistant()
    answered = []
    assistant.chat = lambda text, response_language, source=None, attachments=None: (
        answered.append(response_language)
    )
    assistant._run_text_turn({"text": "Parliamo italiano, per favore.", "id": "t1"})
    assistant._run_text_turn({"text": "Ok", "id": "t2"})

    assert answered == ["it", "it"]
    assert assistant.conversation_language == "it"


def test_english_typed_after_german_speech_is_answered_in_english():
    """Typed turns took the language last spoken: English typed after German
    speech was answered "in German"."""
    assistant = _bare_assistant()
    assistant.status_manager = QuietStatus()
    assistant.interrupt_event = threading.Event()
    assistant.transcriber = Transcript("Was ist die Hauptstadt von Australien?", "de")
    answered = []
    assistant.chat = lambda text, response_language, source=None, attachments=None: (
        answered.append((source, response_language))
    )
    assistant._answer_spoken("/tmp/heard.wav")
    assistant._run_text_turn({"text": "What is the capital of Canada?", "id": "t1"})
    assistant._run_text_turn({"text": "Wie spät ist es jetzt in Tokio?", "id": "t2"})

    assert answered == [("voice", "de"), ("text", "en"), ("text", "de")]


@pytest.mark.parametrize(
    "text, language",
    [
        ("Wie wird das Wetter morgen in Hamburg?", "de"),
        ("明日の東京の天気を教えてください。", "ja"),
        ("北京明天的天气怎么样？", "zh"),
        ("Qual é a capital do Brasil?", "pt"),
        ("Hoe laat gaat het Rijksmuseum morgen open?", "nl"),
        ("कल दिल्ली में मौसम कैसा रहेगा?", "hi"),
    ],
)
def test_a_first_typed_message_is_answered_and_spoken_in_its_language(text, language):
    assistant = _bare_assistant()
    events = []
    assistant.event_bus.subscribe(lambda kind, data: events.append((kind, data)))
    answered = []
    assistant.chat = lambda text, response_language, source=None, attachments=None: (
        answered.append(response_language)
    )
    assistant._run_text_turn({"text": text, "id": "t1"})

    assert answered == [language]
    assert events[0][1]["language"] == language

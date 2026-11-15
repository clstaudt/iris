"""Unit tests for how VoiceAssistant takes typed turns and gates speech.

The assistant is built with object.__new__ so no models are loaded; only the
attributes the code under test touches are filled in.
"""

import queue
import threading
from types import SimpleNamespace

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
    assistant.sleep_word = None
    assistant.conversation_language = "en"
    assistant._reply_language = "en"
    assistant._turn_active = threading.Event()
    assistant.event_bus = EventBus()
    assistant.tts_generator = SpeechGenerator.__new__(SpeechGenerator)
    assistant.tts_generator.engine = "kyutai"
    assistant.tts_generator.voice = "anna"
    assistant.persona = SimpleNamespace(name="Iris")
    assistant._reply_voiced = assistant._reply_unspoken = False
    assistant._unspoken_told = set()
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
        # The conversation's language and the user's it was given, turn by turn
        self.conversations = []
        self.languages = []

    def transcribe(self, audio_path, conversation=None, languages=None):
        self.conversations.append(conversation)
        self.languages.append(languages)
        return self.heard


def _answer_heard(engine, text, language):
    """Answer a spoken turn with a persona on this engine; the reply language."""
    assistant = _bare_assistant()
    assistant.status_manager = QuietStatus()
    assistant.interrupt_event = threading.Event()
    assistant.transcriber = Transcript(text, language)
    assistant.tts_generator.engine = engine
    assistant.tts_generator.voice = {"kyutai": "anna", "kokoro": "bm_lewis"}[engine]
    answered = []
    assistant.chat = lambda text, response_language, source=None: answered.append(
        response_language
    )
    assistant._answer_spoken("/tmp/heard.wav")
    return answered[0]


def test_a_kyutai_persona_answers_german_in_german():
    assert _answer_heard("kyutai", "Beschreibe dich kurz.", "de") == "de"


def test_a_kokoro_persona_answers_german_in_german_though_its_voice_does_not_speak_it():
    """The reply is shown; no other voice says it."""
    assert _answer_heard("kokoro", "Beschreibe dich kurz.", "de") == "de"
    assert _answer_heard("kokoro", "Bonjour.", "fr") == "fr"


def test_a_language_no_voice_speaks_keeps_to_english():
    assert _answer_heard("kokoro", "Опиши себя коротко.", "ru") == "en"
    assert _answer_heard("kyutai", "Опиши себя коротко.", "ru") == "en"


def test_a_kyutai_persona_answers_japanese_in_japanese():
    assert _answer_heard("kyutai", "こんにちは", "ja") == "ja"


def test_italian_that_whisper_takes_for_portuguese_is_answered_in_italian():
    """Whisper heard a short "Parliamo italiano." as Portuguese; the words
    are plainly Italian."""
    assert _answer_heard("kyutai", "Parliamo italiano.", "pt") == "it"
    assert _answer_heard("kokoro", "Parliamo italiano.", "pt") == "it"


def test_a_few_words_keep_to_the_language_whisper_heard():
    assert _answer_heard("kyutai", "Okay.", "de") == "de"
    assert _answer_heard("kyutai", "Ja.", "de") == "de"


def _hearing(text, language):
    """A bare assistant that hears this said, and answers nothing."""
    assistant = _bare_assistant()
    assistant.status_manager = QuietStatus()
    assistant.interrupt_event = threading.Event()
    assistant.transcriber = Transcript(text, language)
    assistant.chat = lambda text, response_language, source=None, attachments=None: None
    return assistant


def test_the_first_thing_said_is_heard_as_whisper_hears_it():
    """The conversation has no language before the user has said or typed
    anything in it: Whisper is given none, as before."""
    assistant = _hearing("Il teatro è aperto tutte le sere.", "it")
    assistant._answer_spoken("/tmp/heard.wav")
    assert assistant.transcriber.conversations == [None]


def test_what_is_said_next_is_heard_in_the_conversations_language():
    """Italian said with an accent was heard as English, and written as an
    English translation: Whisper is told the conversation is in Italian,
    and which languages the user chose."""
    assistant = _hearing("Il teatro è aperto tutte le sere.", "it")
    assistant.chosen_languages = frozenset({"en", "it", "de"})
    assistant._answer_spoken("/tmp/heard.wav")
    assistant._answer_spoken("/tmp/heard.wav")
    assert assistant.transcriber.conversations == [None, "it"]
    assert assistant.transcriber.languages[-1] == {"en", "it", "de"}


def test_a_typed_message_gives_the_conversation_its_language():
    assistant = _hearing("Il teatro è aperto stasera?", "it")
    assistant._run_text_turn({"text": "Parliamo un po' in italiano, per favore.", "id": "t1"})
    assistant._answer_spoken("/tmp/heard.wav")
    assert assistant.transcriber.conversations == ["it"]


def test_each_piece_of_the_reply_is_spoken_in_the_language_it_is_in():
    """Asked in English whether it speaks Italian, the model answers in
    Italian: that is spoken in Italian, not in English."""
    assistant = _bare_assistant()
    for piece in ["Sì, certamente.", "Posso parlare italiano.", "Ciao!"]:
        assistant._speak(piece)
    assistant._speak("Sorry, back to English: what would you like to talk about?")

    assert assistant.speech_pipeline.languages == ["it", "it", "it", "en"]


def test_a_german_reply_of_a_persona_whose_voice_speaks_no_german_is_only_shown():
    """It was spoken by Pocket TTS's German voice, a man's: HAL changed his
    voice in the middle of the conversation. Now no other voice says it,
    and the user is told once why the voice is silent."""
    assistant = _bare_assistant()
    assistant.tts_generator.engine, assistant.tts_generator.voice = "kokoro", "bm_lewis"
    assistant.persona = SimpleNamespace(name="HAL")
    notices = []
    assistant.event_bus.subscribe(lambda kind, data: kind == "notice" and notices.append(data))
    assistant._speak("Ja, natürlich, das mache ich gern.")
    assistant._speak("Worüber möchtest du heute sprechen?")
    assistant._speak("Sorry, back to English: what would you like to talk about?")

    assert assistant.speech_pipeline.spoken == [
        "Sorry, back to English: what would you like to talk about?"
    ]
    assert notices == [
        {
            "message": "HAL's voice does not speak German: "
            "replies in German are shown, not spoken.",
            "kind": "unspoken",
            "language": "de",
        }
    ]
    assert assistant._reply_unspoken


def test_a_kyutai_voice_speaks_german_itself():
    assistant = _bare_assistant()
    assistant._speak("Ja, natürlich, das mache ich gern.")
    assert assistant.speech_pipeline.languages == ["de"]
    assert not assistant._reply_unspoken


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

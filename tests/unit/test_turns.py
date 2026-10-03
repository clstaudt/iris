"""Unit tests for how VoiceAssistant takes typed turns and gates speech.

The assistant is built with object.__new__ so no models are loaded; only the
attributes the code under test touches are filled in.
"""

import queue
import threading

from iris.assistant import VoiceAssistant
from iris.events import EventBus


class RecordingPipeline:
    def __init__(self):
        self.spoken = []

    def queue_speech(self, text):
        self.spoken.append(text)


class PlainConversation:
    def clean_markdown(self, text):
        return text.replace("**", "")


class SilentDisplay:
    def show_user_message(self, text):
        pass


def _bare_assistant():
    assistant = object.__new__(VoiceAssistant)
    assistant.speech_pipeline = RecordingPipeline()
    assistant.conversation = PlainConversation()
    assistant.display_manager = SilentDisplay()
    assistant.turn_queue = queue.Queue()
    assistant.hold_event = threading.Event()
    assistant.speak_replies = True
    assistant.wake_word = None
    assistant.conversation_language = "en"
    assistant._turn_active = threading.Event()
    assistant.event_bus = EventBus()
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
    assistant.chat = lambda text, response_language, source=None: answered.append(
        (text, response_language, source)
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
    assistant.chat = lambda text, response_language, source=None: seen.append(
        assistant._turn_active.is_set()
    )
    assistant._run_text_turn({"text": "hi", "id": "t1"})
    assert seen == [True]
    assert not assistant._turn_active.is_set()

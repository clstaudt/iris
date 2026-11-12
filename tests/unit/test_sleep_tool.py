"""Unit tests for going to sleep by Iris's own decision: the sleep tool,
offered while a wake word is set, ends the conversation the way the sleep
word does, once the goodbye has been heard to the end.

No model, no audio devices: the model is scripted and the speaker is a
fake that plays each clip until it is released.
"""

import queue
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from iris import assistant as assistant_module
from iris.assistant import DORMANT_SILENCE, VoiceAssistant
from iris.controller import AssistantController
from iris.events import EventBus
from iris.llm import ConversationManager
from iris.plugin import Plugin, Tool
from iris.plugins import BUILT_IN, sleep
from iris.tools import Tools
from iris.tts import SpeechPipeline


def say(text):
    return {"message": {"content": text}}


def call(*names):
    calls = [{"function": {"name": name, "arguments": {}}} for name in names]
    return {"message": {"content": "", "tool_calls": calls}}


class ScriptedClient:
    """Answers each request with the next scripted reply, chunk by chunk,
    noting the tools offered with it."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.offered = []

    def chat(self, model, messages, stream=False, tools=None):
        self.offered.append([tool["function"]["name"] for tool in tools or []])
        return iter(self.replies.pop(0))


class InstantGenerator:
    def generate(self, text, language="en"):
        return tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name


class HeldPlayer:
    """Plays each clip until released, as a speaker takes its time."""

    def __init__(self):
        self.playing = threading.Event()
        self.release = threading.Event()

    def play(self, audio_file, on_level=None, cancelled=None):
        self.playing.set()
        while not self.release.wait(0.01):
            if cancelled is not None and cancelled():
                break
        self.playing.clear()

    def progress(self):
        return 1.0

    def stop(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass


class Status:
    def __init__(self):
        self.status_lock = threading.Lock()
        self.status_thinking = False
        self.status_listening = False
        self.status_dormant = False


class Search(Plugin):
    id = "search"
    name = "Search"

    def tools(self, context):
        return [
            Tool(
                name="search",
                description="Looks it up.",
                run=lambda **_: {"success": True, "result": "Sunny."},
            )
        ]


def _wait(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def _assistant(*replies, wake_word="Iris", speak=True, plugins=(sleep.plugin,)):
    """An assistant awake in a conversation, with the real tools, reply
    stream and speech pipeline, and the sleep tool on."""
    a = object.__new__(VoiceAssistant)
    a.event_bus = EventBus()
    a.events = []
    a.event_bus.subscribe(lambda kind, data: a.events.append(kind))
    a.status_manager = Status()
    a.speak_replies = speak
    a.memory = None
    a.wake_word = wake_word
    a.wake_window = 30.0
    a.awake_until = 0.0
    a._was_dormant = True
    a._last_window_emit = 0.0
    a.interrupt_event = threading.Event()
    a.suspend_event = threading.Event()
    a.pause_event = threading.Event()
    a.conversation_language = "en"
    a.player = HeldPlayer()
    a.speech_pipeline = SpeechPipeline(InstantGenerator(), a.player)
    # The voice's languages, which a typed message's language is told among
    a.tts_generator = SimpleNamespace(languages=lambda: frozenset({"en", "de", "fr", "it"}))
    a.tools = Tools(list(plugins), a._plugin_context())
    with patch("iris.llm.ollama") as mock_ollama:
        mock_ollama.show.return_value = MagicMock(modelinfo={}, details=MagicMock(family="test"))
        mock_ollama.list.return_value = []
        a.conversation = ConversationManager(model="test", tools=a.tools)
    a.conversation.client = ScriptedClient(replies)
    return a


# -- going to sleep after the goodbye ----------------------------------------


def test_iris_falls_asleep_once_its_goodbye_has_been_spoken():
    a = _assistant([say("Good night, "), say("sleep well!"), call("end_conversation")])
    a._stay_awake()
    turn = threading.Thread(target=a.chat, args=("Thanks, that's all for today.",))
    turn.start()

    assert a.player.playing.wait(3)
    assert _wait(lambda: a._ending)  # the model called the tool...
    assert not a.dormant  # ...but the goodbye is still being spoken

    a.player.release.set()
    turn.join(3)
    assert not turn.is_alive()
    assert a.dormant
    assert a.speech_pipeline.said() == "Good night, sleep well!"
    # The goodbye came with the call: the model is not asked to say more
    assert a.conversation.client.offered == [["end_conversation"]]
    # The voice is given time to die away before the wake word is listened for
    assert a._quiet_until > time.monotonic()


def test_a_goodbye_cut_short_leaves_iris_awake():
    a = _assistant([say("Good night!"), call("end_conversation")])
    a._stay_awake()
    turn = threading.Thread(target=a.chat, args=("Thanks, bye.",))
    turn.start()
    assert a.player.playing.wait(3)
    assert _wait(lambda: a._ending)

    # The user speaks over the goodbye: they were not done after all
    a.interrupt_event.set()
    a.speech_pipeline.stop()
    turn.join(3)
    assert not turn.is_alive()
    assert not a.dormant


def test_a_call_without_words_is_followed_by_the_goodbye():
    a = _assistant([call("end_conversation")], [say("Sleep well!")], speak=False)
    a._stay_awake()
    assert a.chat("Go to sleep, Iris.") == "Sleep well!"
    # Asked once more for its goodbye, without tools
    assert a.conversation.client.offered == [["end_conversation"], []]
    assert a.dormant


def test_a_call_with_another_tool_lets_the_model_use_its_result():
    a = _assistant(
        [call("search", "end_conversation")],
        [say("Sunny all day. Goodbye!")],
        speak=False,
        plugins=(Search(), sleep.plugin),
    )
    a._stay_awake()
    assert a.chat("Weather tomorrow? Then I'm off.") == "Sunny all day. Goodbye!"
    assert a.conversation.client.offered == [["search", "end_conversation"]] * 2
    assert a.dormant


def test_an_ordinary_reply_keeps_iris_awake():
    a = _assistant([say("It's sunny.")], speak=False)
    a._stay_awake()
    a.chat("Weather?")
    assert not a.dormant


def test_a_typed_goodbye_ends_the_conversation_too():
    a = _assistant([say("Bye for now!"), call("end_conversation")], speak=False)
    a._answer_typed({"text": "Thanks, bye!"})
    assert a.dormant
    # With the voice off there is nothing to die away
    assert a._quiet_until == 0.0


def test_a_typed_message_still_wakes_iris():
    a = _assistant([say("It's sunny.")], speak=False)
    assert a.dormant
    a._answer_typed({"text": "Weather?"})
    assert not a.dormant


# -- the voice's echo cannot wake it -----------------------------------------


class Recorder:
    """Notes when and how the microphone opens, then shuts Iris down."""

    silence_threshold = 0.015
    _device_error = False

    def __init__(self, assistant):
        self.assistant = assistant
        self.opened = []

    def check_audio_devices(self):
        return True, ""

    def close_microphone(self):
        pass

    def listen(self, on_level=None, should_stop=None, silence_duration=None):
        a = self.assistant
        self.opened.append((time.monotonic(), a.dormant, silence_duration))
        a.stop_event.set()


def test_the_wake_word_is_listened_for_only_once_the_goodbye_has_died_away(monkeypatch):
    monkeypatch.setattr(assistant_module, "GOODBYE_SETTLE", 0.3)
    monkeypatch.setattr(
        assistant_module.AudioRecorder, "check_audio_devices", staticmethod(lambda: (True, ""))
    )
    chimes = []
    a = _assistant()
    a.player = SimpleNamespace(
        audio=SimpleNamespace(play=lambda audio, rate, **options: chimes.append(time.monotonic()))
    )
    a.stop_event = threading.Event()
    a.hold_event = threading.Event()
    a.turn_queue = queue.Queue()
    a.recorder = Recorder(a)
    a._stay_awake()
    a._note_dormancy(False)  # awake, as during the conversation

    spoken = time.monotonic()  # the goodbye has just been heard
    a._sleep_after_goodbye()
    a.run_loop()

    [(opened, dormant, silence)] = a.recorder.opened
    assert dormant and silence == DORMANT_SILENCE
    assert opened - spoken >= 0.3
    # The falling chime once the voice has died away, as for the sleep word
    [chimed] = chimes
    assert spoken + 0.3 <= chimed <= opened
    assert a.events.count("sleep") == 1


# -- offered only with a wake word -------------------------------------------


def test_offered_only_while_a_wake_word_is_set():
    a = _assistant(wake_word=None)
    assert a.tools.names() == []
    assert a.tools.get_schemas() == [] and a.tools.guidance() == ""
    assert a.tools.execute("end_conversation")["success"] is False

    a.wake_word = "Iris"
    assert a.tools.names() == ["end_conversation"]
    assert "'end_conversation' function" in a.tools.guidance()

    a.wake_word = None  # switched off again
    assert a.tools.names() == []


def test_without_a_wake_word_a_goodbye_leaves_iris_listening():
    a = _assistant([say("Bye!")], wake_word=None, speak=False)
    a.chat("Thanks, bye.")
    assert a.conversation.client.offered == [[]]  # asked without the tool
    assert not a.dormant


# -- the switch ----------------------------------------------------------------


def test_it_is_a_tool_switch_on_by_default():
    assert sleep.plugin in BUILT_IN
    a = _assistant()
    assert a.tools.states()["sleep"] == {
        "enabled": True,
        "available": True,
        "name": "Going to sleep",
    }
    off = Tools([sleep.plugin], a._plugin_context(), choices={"sleep": False})
    assert off.names() == []


def test_the_switch_turns_it_off_and_on():
    a = _assistant([say("Bye!")], speak=False)
    a.config_info = lambda: {"tools": a.tools.states()}
    controller = AssistantController(EventBus())
    controller.assistant = a
    configs = []
    controller.bus.subscribe(lambda kind, data: configs.append(data) if kind == "config" else None)

    assert controller.set_tool("sleep", False) == {"ok": True}
    assert configs[-1]["tools"]["sleep"]["enabled"] is False
    assert a.tools.names() == []
    a._stay_awake()
    a.chat("Thanks, bye.")
    assert a.conversation.client.offered == [[]]
    assert not a.dormant

    assert controller.set_tool("sleep", True) == {"ok": True}
    assert a.tools.names() == ["end_conversation"]

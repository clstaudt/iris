"""Unit tests for AssistantController lifecycle using a fake assistant."""

import threading
import time
from types import SimpleNamespace

from iris.audio import AudioPlayer
from iris.controller import AssistantController, silent_microphone_message
from iris.events import EventBus
from iris.memory import MemoryStore
from iris.personas import get_persona
from iris.tts import SpeechGenerator


class FakePipeline:
    @property
    def is_active(self):
        return False

    def stop(self):
        pass

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class FakeConversation:
    model = "fake-llm"
    amnesia = False

    def set_amnesia(self, on):
        self.amnesia = on

    def set_model(self, name):
        self.model = name


class FakeAssistant:
    """Stands in for VoiceAssistant without loading any models."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.conversation = FakeConversation()
        self.speech_pipeline = FakePipeline()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.interrupt_event = threading.Event()
        self.hold_event = threading.Event()
        self.suspend_event = threading.Event()
        self.tools = None
        self.turn_queue = kwargs.get("turn_queue")
        self.speak_replies = kwargs.get("speak_replies", True)
        self.conversation_language = "en"
        # Loaded: recognition and the voice
        self.transcriber = object()
        self.voice_error = None

    def run_loop(self):
        while not self.stop_event.is_set():
            self.stop_event.wait(0.05)

    def set_wake_word(self, word):
        self.wake_word = word or None

    def config_info(self):
        return {
            "persona": "Iris",
            "llm_model": "fake-llm",
            "whisper_model": "fake-whisper",
            "whisper_device": "cpu",
            "tts_voice": "af_fake",
            "input_device": "Fake Microphone",
            "mic_on": not self.pause_event.is_set(),
            "paused": self.suspend_event.is_set(),
        }


def _wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_tool_activity_outranks_thinking():
    """"thinking" stays set while a tool runs, so it must not mask the tool."""
    controller = AssistantController(EventBus())
    controller._flags = {"thinking": True, "looking": True}
    assert controller._flag_state() == "looking"


def test_start_pause_resume_stop(monkeypatch):
    monkeypatch.setattr("iris.controller.VoiceAssistant", FakeAssistant)

    bus = EventBus()
    states = []
    bus.subscribe(lambda event_type, data: states.append(data.get("state")) if event_type == "state" else None)

    controller = AssistantController(bus)
    try:
        assert controller.start({})["ok"] is True
        assert _wait_for(lambda: controller.assistant is not None)
        assert _wait_for(lambda: controller.info()["running"])

        # Pause holds the whole conversation, speech included
        assert controller.pause()["ok"] is True
        assert _wait_for(lambda: "paused" in states)
        assert controller.assistant.suspend_event.is_set()
        assert controller.assistant.speech_pipeline.paused is True

        # Resume → back to a non-paused state
        assert controller.resume()["ok"] is True
        assert not controller.assistant.suspend_event.is_set()
        assert controller.assistant.speech_pipeline.paused is False

        # The mic switch is separate from pausing
        assert controller.set_mic(False)["ok"] is True
        assert controller.assistant.pause_event.is_set()
        assert _wait_for(lambda: "muted" in states)
        controller.set_mic(True)
        assert not controller.assistant.pause_event.is_set()

        # Starting again while running is a no-op
        assert controller.start({})["already_running"] is True

        # Model switching goes through to the conversation manager
        assert controller.set_model("other-llm")["ok"] is True
        assert controller.assistant.conversation.model == "other-llm"
    finally:
        controller.stop()
    assert _wait_for(lambda: not controller.info()["running"])


def test_typed_turns_queue_before_and_after_start(monkeypatch):
    """Messages typed while models load are kept and handed to the assistant."""
    monkeypatch.setattr("iris.controller.VoiceAssistant", FakeAssistant)

    controller = AssistantController(EventBus())
    first = controller.send_text("hello")
    assert first["ok"] is True
    assert controller.send_text("   ")["ok"] is False
    try:
        controller.start({})
        assert _wait_for(lambda: controller.assistant is not None)
        second = controller.send_text("again")
        assert second["id"] != first["id"]

        queued = controller.assistant.turn_queue
        assert queued.get_nowait() == {"text": "hello", "id": first["id"], "attachments": []}
        assert queued.get_nowait() == {"text": "again", "id": second["id"], "attachments": []}
    finally:
        controller.stop()


def test_typing_hold_publishes_typing_state(monkeypatch):
    monkeypatch.setattr("iris.controller.VoiceAssistant", FakeAssistant)

    bus = EventBus()
    states = []

    def record(event_type, data):
        if event_type == "state":
            states.append(data.get("state"))

    bus.subscribe(record)

    controller = AssistantController(bus)
    try:
        controller.start({})
        assert _wait_for(lambda: controller.assistant is not None)
        assert controller.hold_listening(True)["ok"] is True
        assert controller.assistant.hold_event.is_set()
        assert _wait_for(lambda: states and states[-1] == "typing")

        # A switched-off mic outranks the typing hold, a paused conversation both
        controller.set_mic(False)
        assert _wait_for(lambda: states[-1] == "muted")
        controller.pause()
        assert _wait_for(lambda: states[-1] == "paused")

        controller.resume()
        controller.set_mic(True)
        controller.hold_listening(False)
        assert not controller.assistant.hold_event.is_set()
        assert _wait_for(lambda: states[-1] == "idle")
    finally:
        controller.stop()


def test_voice_output_toggle(monkeypatch):
    monkeypatch.setattr("iris.controller.VoiceAssistant", FakeAssistant)

    controller = AssistantController(EventBus())
    assert controller.set_voice_output(False)["ok"] is False  # not running yet
    try:
        controller.start({"speak_replies": False})
        assert _wait_for(lambda: controller.assistant is not None)
        assert controller.assistant.speak_replies is False
        controller.set_voice_output(True)
        assert controller.assistant.speak_replies is True
    finally:
        controller.stop()


def test_history_keeps_the_visible_conversation():
    """A reloaded UI rebuilds its transcript from this; chunks are not kept."""
    bus = EventBus()
    controller = AssistantController(bus)
    bus.emit("chat.user", {"text": "hi", "source": "text", "id": "t1"})
    bus.emit("chat.chunk", {"text": "Hel"})
    bus.emit("chat.done", {"text": "Hello", "interrupted": False})
    bus.emit("audio.level", {"rms": 0.1})

    events = controller.history()["events"]
    assert [e["type"] for e in events] == ["chat.user", "chat.done"]
    assert events[1]["data"]["text"] == "Hello"


def test_the_conversation_is_shown_again_after_a_restart(tmp_path):
    """With memory, one conversation goes on across launches: a new core
    shows its recent turns again, oldest first, as the app shows turns, and
    keeps them for a window that reloads; notes to the model stay out."""
    earlier = MemoryStore(tmp_path / "memory.db")
    earlier.add_turn("user", "My sister's cat is called Pepper.", "voice")
    earlier.add_turn("assistant", "Pepper is a good name for a cat.")
    earlier.add_turn("system", "You are now HAL.")
    earlier.add_turn("user", "明日の東京の天気は？", "text")
    earlier.add_turn("assistant", "晴れです。")

    bus = EventBus()
    shown = []
    bus.subscribe(lambda kind, data: shown.append((kind, data)))
    controller = AssistantController(bus)
    # The new core reads what the last one wrote
    controller.assistant = SimpleNamespace(memory=SimpleNamespace(store=MemoryStore(tmp_path / "memory.db")))
    controller._restore_thread()

    assert shown == [
        ("chat.user", {"text": "My sister's cat is called Pepper.", "source": "voice", "restored": True}),
        ("chat.done", {"text": "Pepper is a good name for a cat.", "interrupted": False, "restored": True}),
        ("chat.user", {"text": "明日の東京の天気は？", "source": "text", "restored": True}),
        ("chat.done", {"text": "晴れです。", "interrupted": False, "restored": True}),
    ]
    assert [(e["type"], e["data"]) for e in controller.history()["events"]] == shown

    # Without memory there is nothing to show again
    shown.clear()
    controller.assistant = SimpleNamespace(memory=None)
    controller._restore_thread()
    assert shown == []


def test_the_conversation_before_a_restart_comes_back_marked_as_such():
    """The app shows these turns above whatever it said meanwhile."""
    bus = EventBus()
    controller = AssistantController(bus)
    turns = [
        {"role": "user", "text": "Hi", "source": "text"},
        {"role": "system", "text": "a note", "source": None},
        {"role": "assistant", "text": "Hello", "source": None},
    ]
    store = type("Store", (), {"recent_turns": lambda self: turns})()
    memory = type("Memory", (), {"store": store})()
    controller.assistant = type("Assistant", (), {"memory": memory})()
    controller._restore_thread()

    events = controller.history()["events"]
    assert events == [
        {"type": "chat.user", "data": {"text": "Hi", "source": "text", "restored": True}},
        {"type": "chat.done", "data": {"text": "Hello", "interrupted": False, "restored": True}},
    ]


def test_a_silent_microphone_is_warned_about_once(monkeypatch):
    """A real microphone always hears some noise: exact zeros for more than
    four seconds mean the system withholds it. That is said once, not with
    every level, and said again after another such silence."""
    now = [100.0]
    monkeypatch.setattr("iris.controller.time.monotonic", lambda: now[0])
    bus = EventBus()
    warnings = []
    bus.subscribe(lambda event_type, data: warnings.append(data) if event_type == "error" else None)
    AssistantController(bus)

    def levels(rms, seconds):
        for _ in range(round(seconds * 15)):
            bus.emit("audio.level", {"rms": rms})
            now[0] += 1 / 15

    levels(0.0, 3.8)
    assert warnings == []
    levels(0.0, 6)
    assert len(warnings) == 1
    assert "silence" in warnings[0]["message"]
    # Sound, then a pause in it: no warning
    levels(0.02, 1)
    levels(0.0, 3.8)
    assert len(warnings) == 1
    levels(0.0, 1)
    assert len(warnings) == 2


def test_the_silence_warning_names_what_to_do_where_iris_runs(monkeypatch):
    """Only macOS withholds a microphone silently, and it asks for it under
    the app's name: Iris when packaged, the dev server's app in development."""
    monkeypatch.setattr("iris.controller.sys.platform", "linux")
    elsewhere = silent_microphone_message()
    assert "macOS" not in elsewhere and "System Settings" not in elsewhere
    assert "muted" in elsewhere

    monkeypatch.setattr("iris.controller.sys.platform", "darwin")
    development = silent_microphone_message()
    assert "Terminal" in development and "System Settings" in development
    monkeypatch.setattr("iris.controller.sys.frozen", True, raising=False)
    packaged = silent_microphone_message()
    assert "for Iris" in packaged and "Terminal" not in packaged


def test_work_shows_through_a_paused_mic(monkeypatch):
    """Typed turns are answered with the mic off; thinking must not read as paused."""
    monkeypatch.setattr("iris.controller.VoiceAssistant", FakeAssistant)

    bus = EventBus()
    states = []

    def record(event_type, data):
        if event_type == "state":
            states.append(data.get("state"))

    bus.subscribe(record)
    controller = AssistantController(bus)
    try:
        controller.start({})
        assert _wait_for(lambda: controller.assistant is not None)
        controller.set_mic(False)
        assert _wait_for(lambda: states[-1] == "muted")

        controller._flags = {"thinking": True}
        assert _wait_for(lambda: states[-1] == "thinking")

        controller._flags = {}
        assert _wait_for(lambda: states[-1] == "muted")
    finally:
        controller.stop()


def test_amnesia_is_switched_on_the_conversation(monkeypatch):
    monkeypatch.setattr("iris.controller.VoiceAssistant", FakeAssistant)

    controller = AssistantController(EventBus())
    try:
        controller.start({"amnesia": True})
        assert _wait_for(lambda: controller.assistant is not None)
        assert _wait_for(lambda: controller.assistant.conversation.amnesia is True)
        controller.set_amnesia(False)
        assert controller.assistant.conversation.amnesia is False
    finally:
        controller.stop()


def test_wake_word_is_set_on_the_assistant(monkeypatch):
    monkeypatch.setattr("iris.controller.VoiceAssistant", FakeAssistant)

    controller = AssistantController(EventBus())
    assert controller.set_wake_word("Iris")["ok"] is False  # not running yet
    try:
        controller.start({})
        assert _wait_for(lambda: controller.assistant is not None)
        assert controller.set_wake_word("Iris")["ok"] is True
        assert controller.assistant.wake_word == "Iris"
        controller.set_wake_word("")
        assert controller.assistant.wake_word is None
    finally:
        controller.stop()


class PersonaAssistant(FakeAssistant):
    """A running assistant whose persona can be switched."""

    def __init__(self):
        super().__init__()
        self.persona = get_persona("iris")
        self.conversation.add_note = lambda note: None
        self.tts_generator = SpeechGenerator.__new__(SpeechGenerator)
        self.tts_generator.prepare = lambda: None
        self.player = AudioPlayer()


def test_a_voice_that_cannot_speak_the_language_goes_back_to_english():
    controller = AssistantController(EventBus())
    controller.assistant = PersonaAssistant()
    # Spoken by a plugin's engine, say: neither Kokoro nor Kyutai speaks it
    controller.assistant.conversation_language = "ru"

    controller.set_persona("hal")

    assert controller.assistant.tts_generator.engine == "kokoro"
    assert controller.assistant.conversation_language == "en"


def test_a_kokoro_persona_keeps_german_through_kyutai():
    controller = AssistantController(EventBus())
    controller.assistant = PersonaAssistant()
    controller.assistant.conversation_language = "de"

    controller.set_persona("hal")

    assert controller.assistant.conversation_language == "de"


def test_switching_to_a_voice_that_speaks_the_language_keeps_it():
    controller = AssistantController(EventBus())
    controller.assistant = PersonaAssistant()
    controller.assistant.conversation_language = "fr"

    controller.set_persona("hal")

    assert controller.assistant.conversation_language == "fr"


def test_plugins_and_engines_are_listed_for_the_app():
    controller = AssistantController(EventBus())

    listed = controller.plugins_list()
    ids = [p["id"] for p in listed["plugins"]]
    assert ids[:4] == ["memory", "camera", "screen", "web"]
    assert listed["failures"] == []
    camera = listed["plugins"][1]
    assert camera["switchable"] is True and camera["source"] == "built-in"

    engines = controller.engines_list()
    assert [e["id"] for e in engines["tts"]][:2] == ["kyutai", "kokoro"]
    assert "whisper" in [e["id"] for e in engines["stt"]]


def test_the_persona_editor_learns_the_engines_from_the_core():
    voices = AssistantController(EventBus()).persona_voices()
    info = {e["id"]: e for e in voices["engine_info"]}
    assert info["kyutai"] == {"id": "kyutai", "name": "Kyutai", "takes_speed": False}
    assert info["kokoro"]["takes_speed"] is True
    assert set(voices["engines"]) >= {"kyutai", "kokoro"}

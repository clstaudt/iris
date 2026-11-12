"""The recognition model and the turn detection are chosen while Iris runs."""

import threading
import time

from iris import controller as controller_module
from iris import downloads
from iris.assistant import VoiceAssistant
from iris.controller import AssistantController
from iris.events import EventBus
from iris.transcription import WHISPER_SIZES


class Recorder:
    def __init__(self, turn_end=None):
        self.turn_end = turn_end
        self.silence_duration = 3.5


class Transcriber:
    def __init__(self, model="small", engine="mlx", model_path=None):
        self.model_name = model
        self.engine = engine
        self.model_path = model_path


class Assistant:
    """What the controller touches of a running assistant."""

    def __init__(self):
        self.transcriber = Transcriber()
        self.recognition_engine = "whisper"
        self.recorder = Recorder()
        self.modes = []

    def set_turn_detection(self, mode):
        self.modes.append(mode)
        self.recorder.turn_end = object() if mode == "smart_turn" else None
        return True

    def config_info(self):
        return {"whisper_model": self.transcriber.model_name}


def _controller(monkeypatch, assistant=None):
    monkeypatch.setattr(controller_module, "load_plugins", lambda: type("P", (), {"plugins": []})())
    bus = EventBus()
    events = []
    bus.subscribe(lambda kind, data: events.append((kind, data)))
    controller = AssistantController(bus)
    controller.assistant = assistant
    return controller, events


def _wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


class TestRecognitionModels:
    def test_every_size_with_whether_it_is_here(self, monkeypatch):
        monkeypatch.setattr(downloads, "is_cached", lambda repo, filename=None: "small" in repo)
        controller, _ = _controller(monkeypatch, Assistant())
        listed = controller.stt_models()
        assert [m["id"] for m in listed["models"]] == list(WHISPER_SIZES)
        small = next(m for m in listed["models"] if m["id"] == "small")
        assert small["cached"] and small["active"]
        assert not any(m["active"] for m in listed["models"] if m["id"] != "small")
        assert listed["engine"] == "mlx"
        assert listed["folder"] == ""

    def test_a_chosen_folder_is_in_use_instead_of_a_size(self, monkeypatch, tmp_path):
        monkeypatch.setattr(downloads, "is_cached", lambda repo, filename=None: True)
        assistant = Assistant()
        assistant.transcriber = Transcriber(model="my-model", model_path=tmp_path)
        controller, _ = _controller(monkeypatch, assistant)
        listed = controller.stt_models()
        assert listed["folder"] == str(tmp_path)
        assert not any(m["active"] for m in listed["models"])

    def test_switching_loads_the_size_and_tells_the_app(self, monkeypatch):
        fetched = []
        created = threading.Event()

        class Whisper:
            def create(self, options):
                created.set()
                return Transcriber(model=options["model"])

        monkeypatch.setattr(controller_module.engines, "recognition_engine", lambda id: Whisper())
        assistant = Assistant()
        controller, events = _controller(monkeypatch, assistant)
        monkeypatch.setattr(controller, "_fetch_stt", lambda model: fetched.append(model))

        assert controller.set_stt_model("large-v3-turbo") == {"ok": True}
        assert _wait_for(lambda: assistant.transcriber.model_name == "large-v3-turbo")
        assert fetched == ["large-v3-turbo"]
        assert _wait_for(lambda: ("config", {"whisper_model": "large-v3-turbo"}) in events)

    def test_an_unknown_size_is_refused(self, monkeypatch):
        controller, _ = _controller(monkeypatch, Assistant())
        assert not controller.set_stt_model("huge")["ok"]

    def test_another_recognition_engine_has_no_whisper_sizes(self, monkeypatch):
        assistant = Assistant()
        assistant.recognition_engine = "parakeet"
        controller, _ = _controller(monkeypatch, assistant)
        assert not controller.set_stt_model("small")["ok"]

    def test_nothing_to_switch_before_the_start(self, monkeypatch):
        controller, _ = _controller(monkeypatch)
        assert not controller.set_stt_model("small")["ok"]
        assert not controller.set_turn_detection("silence")["ok"]


class TestTurnDetection:
    def test_the_controller_switches_and_tells_the_app(self, monkeypatch):
        assistant = Assistant()
        controller, events = _controller(monkeypatch, assistant)
        monkeypatch.setattr(controller, "_fetch_turn", lambda: None)
        assert controller.set_turn_detection("silence") == {"ok": True}
        assert _wait_for(lambda: assistant.modes == ["silence"])
        assert _wait_for(lambda: any(kind == "config" for kind, _ in events))

    def test_an_unknown_mode_is_refused(self, monkeypatch):
        controller, _ = _controller(monkeypatch, Assistant())
        assert not controller.set_turn_detection("guess")["ok"]

    def test_the_assistant_drops_and_reloads_smart_turn(self, monkeypatch):
        smart_turn = object()
        monkeypatch.setattr("iris.assistant.SmartTurn.load", classmethod(lambda cls: smart_turn))
        assistant = object.__new__(VoiceAssistant)
        assistant.recorder = Recorder(turn_end=smart_turn)
        assert assistant.set_turn_detection("silence")
        assert assistant.recorder.turn_end is None
        assert assistant.set_turn_detection("smart_turn")
        assert assistant.recorder.turn_end is smart_turn

    def test_without_smart_turn_the_silence_ends_turns(self, monkeypatch):
        monkeypatch.setattr("iris.assistant.SmartTurn.load", classmethod(lambda cls: None))
        assistant = object.__new__(VoiceAssistant)
        assistant.recorder = Recorder()
        assert not assistant.set_turn_detection("smart_turn")
        assert assistant.recorder.turn_end is None

"""Starting without the network, or without parts: Iris starts with what it
has and says what is missing.

On a first start offline nothing can be downloaded: not the speech models
of the setup, and not Pocket TTS's English model, which the default voice
fetches while Iris starts. Each failure is reported as an error the app
shows, and Iris carries on as far as it can: without recognition it cannot
listen but answers typed messages, without a voice it answers in text. Once
the network is back, switching the microphone or the voice on loads what
was missing. Without a microphone, or when it is disconnected, typed
messages are answered too.

The language model is the scripted one; the speech models never load."""

import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from iris import assistant as assistant_module
from iris import downloads, tts
from iris.assistant import VoiceAssistant
from iris.controller import AssistantController
from iris.events import EventBus, brief
from iris.tts import SpeechGenerator
from iris.turns import SMART_TURN_FILE, SmartTurn
from irisdev.voicetest.assistant import wait_until
from irisdev.voicetest.stub_llm import MODEL, SHORT, StubModel

OFFLINE = ConnectionError(
    "An error happened while trying to locate the file on the Hub and we cannot find the\n"
    "requested files in the local cache. Please check your connection and try again."
)


class Microphone:
    """An audio device: a microphone giving silence, or none at all."""

    def __init__(self, present: bool = True, unplugged: bool = False):
        self.present = present
        # Listed, but it fails when opened, as a microphone unplugged does
        self.unplugged = unplugged
        self.opened = 0

    def query_devices(self, device=None, kind=None):
        info = {
            "name": "Microphone" if self.present else "Speakers",
            "max_input_channels": 1 if self.present else 0,
            "max_output_channels": 2,
            "default_samplerate": 16000,
            "index": 0,
        }
        return dict(info) if kind is not None or device is not None else [info]

    def input_stream(self, samplerate=16000, blocksize=512, callback=None, **options):
        self.opened += 1
        if self.unplugged or not self.present:
            raise RuntimeError("Error opening InputStream: Invalid device")
        device = self

        class Stream:
            def start(self):
                self._running = True
                threading.Thread(target=self._feed, daemon=True).start()

            def _feed(self):
                while self._running:
                    callback(np.zeros((blocksize, 1), dtype=np.float32), blocksize, None, None)
                    time.sleep(blocksize / samplerate)

            def stop(self):
                self._running = False

            def close(self):
                self._running = False

        return Stream()

    def play(self, audio, sample_rate, **options) -> float:
        return 0.0

    def stop(self):
        pass

    def playing(self) -> bool:
        return False


class Recorded:
    """The events the app would get."""

    def __init__(self, bus: EventBus):
        self.events: list[tuple[str, dict]] = []
        bus.subscribe(lambda kind, data: self.events.append((kind, dict(data))))

    def errors(self) -> list[str]:
        return [d["message"] for k, d in self.events if k == "error"]

    def last(self, kind: str):
        found = [d for k, d in self.events if k == kind]
        return found[-1] if found else None

    def done(self) -> list[str]:
        return [d["text"] for k, d in self.events if k == "chat.done"]


@pytest.fixture
def stub():
    model = StubModel(first_token=0.0, per_word=0.0)
    yield model
    model.close()


@pytest.fixture
def offline(monkeypatch):
    """Nothing can be downloaded, and nothing was before: recognition and
    the voice fail to load, Smart Turn is not there."""
    state = SimpleNamespace(online=False)

    def create(options):
        if not state.online:
            raise OFFLINE
        return SimpleNamespace(model_name="small", device="cpu", engine="faster-whisper",
                               transcribe=lambda path: ("", "en"))

    def load_model(language=None, **kwargs):
        if not state.online:
            raise OFFLINE
        return SimpleNamespace(get_state_for_audio_prompt=lambda voice: None, sample_rate=24000)

    monkeypatch.setattr(
        assistant_module.engines, "recognition_engine", lambda engine_id: SimpleNamespace(create=create)
    )
    monkeypatch.setattr(tts.TTSModel, "load_model", staticmethod(load_model))
    monkeypatch.setattr(SmartTurn, "load", classmethod(lambda cls, download=True: None))
    return state


def _assistant(stub, bus, audio, **options) -> VoiceAssistant:
    return VoiceAssistant(
        llm_model=MODEL,
        ollama_host=stub.url,
        memory=False,
        plugins=[],
        event_bus=bus,
        turn_queue=queue.Queue(),
        audio=audio,
        **options,
    )


def _run(assistant) -> threading.Thread:
    thread = threading.Thread(target=assistant.run_loop, daemon=True)
    thread.start()
    return thread


def _stop(assistant, thread):
    assistant.stop_event.set()
    thread.join(5)
    assert not thread.is_alive()


def test_errors_are_told_in_a_line_not_a_page():
    assert brief(OFFLINE) == (
        "An error happened while trying to locate the file on the Hub and we cannot find the"
    )
    assert brief(RuntimeError("x" * 500)).endswith("…") and len(brief(RuntimeError("x" * 500))) == 240
    assert brief(KeyError()) == "KeyError"


# Iris without its speech models ----------------------------------------------------


def test_offline_iris_starts_says_what_is_missing_and_answers_typed_messages(stub, offline):
    bus = EventBus()
    seen = Recorded(bus)
    microphone = Microphone()
    iris = _assistant(stub, bus, microphone, wake_word="Iris")

    errors = seen.errors()
    assert any("Speech recognition could not be loaded" in e and "Typed messages" in e for e in errors)
    assert any("The voice could not be loaded, so replies are shown as text" in e for e in errors)
    assert any("Smart Turn could not be loaded" in e for e in errors)
    # Each in a line, not a page of download advice
    assert all("\n" not in e for e in errors)
    # Nothing to listen with: no wake word to wait for either, and no
    # second error about it
    assert not any("wake word" in e for e in errors)
    config = iris.config_info()
    assert config["mic_on"] is False and config["speak_replies"] is False
    assert config["wake_word"] == ""

    thread = _run(iris)
    try:
        iris.turn_queue.put({"text": "What is the capital of France?", "id": "t1"})
        assert wait_until(lambda: seen.done() == [SHORT], 5)
        # The microphone was never opened
        assert microphone.opened == 0
    finally:
        _stop(iris, thread)


def test_once_online_the_microphone_and_the_voice_load_when_switched_on(stub, offline, monkeypatch):
    monkeypatch.setattr(downloads, "is_cached", lambda repo, filename=None: True)
    bus = EventBus()
    seen = Recorded(bus)
    controller = AssistantController(bus)
    controller.assistant = _assistant(stub, bus, Microphone(), wake_word="Iris")
    offline.online = True

    assert controller.set_mic(True)["ok"]
    assert wait_until(lambda: (seen.last("config") or {}).get("mic_on") is True, 5)
    assert controller.assistant.transcriber is not None
    # The wake word asked for at the start is listened for now
    assert controller.assistant.wake_word == "Iris"

    assert controller.set_voice_output(True)["ok"]
    assert wait_until(lambda: (seen.last("config") or {}).get("speak_replies") is True, 5)
    assert controller.assistant.voice_error is None


def test_still_offline_switching_on_says_so_again_and_stays_off(stub, offline, monkeypatch):
    monkeypatch.setattr(downloads, "is_cached", lambda repo, filename=None: True)
    bus = EventBus()
    seen = Recorded(bus)
    controller = AssistantController(bus)
    controller.assistant = _assistant(stub, bus, Microphone())
    before = len(seen.errors())

    controller.set_mic(True)
    controller.set_voice_output(True)
    assert wait_until(lambda: len(seen.errors()) >= before + 2, 5)
    assert controller.assistant.pause_event.is_set()
    assert controller.assistant.speak_replies is False


def test_switching_on_again_while_it_loads_loads_it_once(stub, offline, monkeypatch):
    fetches = []
    loading = threading.Event()

    def download(repo, on_progress, filename=None, interval=0.5):
        fetches.append(repo)
        loading.wait(5)

    monkeypatch.setattr(downloads, "is_cached", lambda repo, filename=None: False)
    monkeypatch.setattr(downloads, "download", download)
    bus = EventBus()
    seen = Recorded(bus)
    controller = AssistantController(bus)
    controller.assistant = _assistant(stub, bus, Microphone())
    offline.online = True
    controller.set_mic(True)
    assert wait_until(lambda: fetches, 5)
    controller.set_mic(True)
    controller.set_mic(True)
    loading.set()
    assert wait_until(lambda: (seen.last("config") or {}).get("mic_on") is True, 5)
    assert len(fetches) == 1


# The first start, as the app makes it ---------------------------------------------------


def test_an_offline_first_start_reports_each_download_and_starts(stub, offline, monkeypatch, tmp_path):
    """The setup's downloads fail one by one, each is reported, and Iris
    starts anyway, answering a typed message."""
    tried = []

    def download(repo, on_progress, filename=None, interval=0.5):
        tried.append(filename or repo)
        raise OFFLINE

    def download_files(files, on_progress, interval=0.5):
        tried.extend(target.name for _, target in files)
        raise OFFLINE

    monkeypatch.setattr(downloads, "is_cached", lambda repo, filename=None: False)
    monkeypatch.setattr(downloads, "download", download)
    monkeypatch.setattr(downloads, "download_files", download_files)
    monkeypatch.setattr("iris.controller.find_kokoro_models", lambda: None)
    monkeypatch.setattr("iris.controller.DEFAULT_KOKORO_DIR", tmp_path)

    bus = EventBus()
    seen = Recorded(bus)
    controller = AssistantController(bus)
    controller.plugins.plugins = []
    controller.start({
        "llm_model": MODEL,
        "ollama_host": stub.url,
        "memory": False,
        "audio": Microphone(),
        "whisper_model": "small",
        "whisper_engine": "faster-whisper",
        "whisper_device": "cpu",
    })
    try:
        assert wait_until(lambda: seen.last("config") is not None, 10)
        errors = seen.errors()
        # Every download was tried, whatever became of the one before
        assert tried == [
            "Systran/faster-whisper-small",
            SMART_TURN_FILE,
            tts.KOKORO_MODEL_FILE,
            tts.KOKORO_VOICES_FILE,
        ]
        assert any(e.startswith("Could not download the speech recognition model:") for e in errors)
        assert any(e.startswith("Could not download the Kokoro voice model:") for e in errors)
        assert not any(e.startswith("Failed to start assistant") for e in errors)
        # The setup screen's progress is done with
        assert seen.last("setup.progress")["label"] == ""
        assert controller.info()["running"]

        controller.send_text("What is the capital of France?")
        assert wait_until(lambda: seen.done() == [SHORT], 5)
    finally:
        controller.stop()


def test_a_failed_download_does_not_keep_the_others_from_being_fetched(monkeypatch, tmp_path):
    fetched = []

    def download(repo, on_progress, filename=None, interval=0.5):
        if filename is None:
            raise OSError("No space left on device")
        fetched.append(filename)

    monkeypatch.setattr(downloads, "is_cached", lambda repo, filename=None: False)
    monkeypatch.setattr(downloads, "download", download)
    monkeypatch.setattr(downloads, "download_files", lambda files, on_progress: fetched.extend(
        target.name for _, target in files))
    monkeypatch.setattr("iris.controller.find_kokoro_models", lambda: None)
    monkeypatch.setattr("iris.controller.DEFAULT_KOKORO_DIR", tmp_path)
    bus = EventBus()
    seen = Recorded(bus)
    AssistantController(bus)._fetch_models(
        {"whisper_model": "small", "whisper_engine": "faster-whisper", "whisper_device": "cpu"}
    )
    assert seen.errors() == ["Could not download the speech recognition model: No space left on device"]
    assert len(fetched) == 3  # Smart Turn and both of Kokoro's files


# The microphone ---------------------------------------------------------------------


def test_without_a_microphone_typed_messages_are_answered(stub, monkeypatch):
    bus = EventBus()
    seen = Recorded(bus)
    monkeypatch.setattr(SmartTurn, "load", classmethod(lambda cls, download=True: None))
    monkeypatch.setattr(
        assistant_module.engines, "recognition_engine",
        lambda engine_id: SimpleNamespace(create=lambda options: SimpleNamespace(
            model_name="small", device="cpu", engine="faster-whisper")),
    )
    monkeypatch.setattr(tts.TTSModel, "load_model", staticmethod(lambda **kw: SimpleNamespace(
        get_state_for_audio_prompt=lambda voice: None, sample_rate=24000)))
    iris = _assistant(stub, bus, Microphone(present=False), speak_replies=False)
    thread = _run(iris)
    try:
        assert wait_until(lambda: any(e.startswith("No microphone") for e in seen.errors()), 5)
        assert wait_until(lambda: (seen.last("config") or {}).get("mic_on") is False, 5)
        iris.turn_queue.put({"text": "Hello?", "id": "t1"})
        assert wait_until(lambda: seen.done() == [SHORT], 5)
        assert thread.is_alive()
    finally:
        _stop(iris, thread)


def test_a_microphone_unplugged_while_listening_leaves_typed_messages_answered(stub, monkeypatch):
    bus = EventBus()
    seen = Recorded(bus)
    monkeypatch.setattr(SmartTurn, "load", classmethod(lambda cls, download=True: None))
    monkeypatch.setattr(
        assistant_module.engines, "recognition_engine",
        lambda engine_id: SimpleNamespace(create=lambda options: SimpleNamespace(
            model_name="small", device="cpu", engine="faster-whisper")),
    )
    monkeypatch.setattr(tts.TTSModel, "load_model", staticmethod(lambda **kw: SimpleNamespace(
        get_state_for_audio_prompt=lambda voice: None, sample_rate=24000)))
    microphone = Microphone(unplugged=True)
    iris = _assistant(stub, bus, microphone, speak_replies=False)
    thread = _run(iris)
    try:
        assert wait_until(lambda: any("disconnected" in e for e in seen.errors()), 5)
        assert wait_until(lambda: (seen.last("config") or {}).get("mic_on") is False, 5)
        opened = microphone.opened
        iris.turn_queue.put({"text": "Hello?", "id": "t1"})
        assert wait_until(lambda: seen.done() == [SHORT], 5)
        # Off until switched on: it is not opened again and again
        time.sleep(0.3)
        assert microphone.opened == opened and thread.is_alive()
        # Plugged in again and switched on, it listens
        microphone.unplugged = False
        iris.pause_event.clear()
        assert wait_until(lambda: microphone.opened > opened, 5)
        assert len([e for e in seen.errors() if "disconnected" in e]) == 1
    finally:
        _stop(iris, thread)


# A language's voice, loaded at its first reply ----------------------------------------


def test_a_languages_voice_that_cannot_be_loaded_is_told_once(monkeypatch):
    """Pocket TTS loads its model for another language at the first reply
    in it. Offline that fails: the reply is not spoken, and the user is
    told why, once, until the voice speaks."""
    online = {"english"}

    def load_model(language=None, **kwargs):
        if language not in online:
            raise OFFLINE
        return SimpleNamespace(
            sample_rate=24000,
            get_state_for_audio_prompt=lambda voice: None,
            generate_audio=lambda state, sentence: tts.torch.zeros(240),
        )

    monkeypatch.setattr(tts.TTSModel, "load_model", staticmethod(load_model))
    told = []
    generator = SpeechGenerator("kyutai", "alba")
    generator.on_failure = told.append
    assert generator.generate("Hello there.", language="en")
    assert generator.generate("Bonjour.", language="fr") is None
    assert generator.generate("Merci.", language="fr") is None
    assert len(told) == 1
    assert told[0].startswith("The voice for French could not be loaded, so Iris does not speak French")
    assert "\n" not in told[0]
    # Back online, it speaks French, and a later failure is told again
    online.add("french")
    assert generator.generate("Bonjour.", language="fr")
    online.discard("french")
    generator._engines["kyutai"]._models.pop("french")
    assert generator.generate("Encore.", language="fr") is None
    assert len(told) == 2

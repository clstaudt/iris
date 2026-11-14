"""What the first-run walkthrough asks of the core before Iris runs: the
speech models downloading in the background with real progress, a sound
check of the microphone, voice previews, servers found on this computer,
and Iris's greeting at the end."""

import http.server
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from iris import downloads, speech_models
from iris.assistant import GREETING, VoiceAssistant
from iris.audio import HEARD_AFTER, SoundCheck
from iris.controller import AssistantController
from iris.downloads import HubFile
from iris.echo import STEP
from iris.events import EventBus
from iris.llm import OPENING, ConversationManager
from iris.servers import discover
from iris.speech_models import Fetcher, SpeechModel
from iris.tts import pocket_files
from iris.turns import SAMPLE_RATE


def _wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def _record(bus):
    events = []
    bus.subscribe(lambda event_type, data: events.append((event_type, data)))
    return events


# -- downloads -----------------------------------------------------------


def test_pocket_tts_names_its_files_with_a_repo_and_a_revision():
    file = HubFile.parse("hf://kyutai/pocket-tts/languages/english/model.safetensors@abc123")
    assert file == HubFile("kyutai/pocket-tts", "languages/english/model.safetensors", "abc123")
    assert HubFile.parse("hf://owner/repo/a.json").revision is None


class _Bar:
    """Feeds a download's progress bar as Hugging Face would: one per file."""

    @staticmethod
    def fill(tqdm_class, sizes):
        for size in sizes:
            bar = tqdm_class(total=size, unit="B")
            parts = [size // 4] * 3 + [size - 3 * (size // 4)]
            for part in parts:
                bar.update(part)
            bar.close()


@pytest.mark.downloads
def test_a_repo_download_knows_its_total_before_the_first_byte(monkeypatch):
    planned = [
        SimpleNamespace(file_size=100, will_download=True),
        SimpleNamespace(file_size=40, will_download=False),  # cached already
        SimpleNamespace(file_size=300, will_download=True),
    ]

    def snapshot_download(repo, allow_patterns=None, dry_run=False, tqdm_class=None):
        if dry_run:
            return planned
        _Bar.fill(tqdm_class, [100, 300])
        return "/cache/repo"

    monkeypatch.setattr(downloads, "snapshot_download", snapshot_download)
    seen = []
    path = downloads.download("org/repo", lambda d, t: seen.append((d, t)), interval=0)
    assert path == "/cache/repo"
    # The total holds from the first report on, so the bar never runs back
    assert {total for _, total in seen} == {400}
    assert all(b[0] >= a[0] for a, b in zip(seen, seen[1:]))
    assert seen[-1] == (400, 400)


def test_single_files_of_repos_download_with_their_bytes_summed(monkeypatch):
    sizes = {"a.safetensors": 80, "b.json": 20, "c.safetensors": 50}
    cached = {"b.json"}
    fetched = []

    def hf_hub_download(repo, filename, revision=None, dry_run=False, tqdm_class=None):
        if dry_run:
            return SimpleNamespace(file_size=sizes[filename], will_download=filename not in cached)
        fetched.append((repo, filename, revision))
        _Bar.fill(tqdm_class, [sizes[filename]])
        return f"/cache/{filename}"

    monkeypatch.setattr(downloads, "hf_hub_download", hf_hub_download)
    seen = []
    files = [HubFile("o/r", name, "rev") for name in sizes]
    downloads.download_hub_files(files, lambda d, t: seen.append((d, t)), interval=0)
    assert fetched == [("o/r", "a.safetensors", "rev"), ("o/r", "c.safetensors", "rev")]
    assert seen[-1] == (130, 130)
    assert all(b[0] >= a[0] for a, b in zip(seen, seen[1:]))


# -- the speech models ---------------------------------------------------


def test_the_voices_come_first_and_folders_leave_their_models_out(tmp_path):
    ids = [m.id for m in speech_models.needed()]
    assert ids == ["kyutai", "kokoro", "turn", "stt"]
    # A wake word needs the smallest Whisper as well
    assert [m.id for m in speech_models.needed(wake_word="Iris")][-1] == "wake"
    # Model folders are used as they are
    folder = tmp_path / "whisper"
    folder.mkdir()
    ids = [m.id for m in speech_models.needed(whisper_model=str(folder), tts_dir=str(tmp_path))]
    assert ids == ["kyutai", "turn"]
    # Another recognition engine fetches its own model
    assert "stt" not in [m.id for m in speech_models.needed(recognition_engine="other")]


def test_pocket_tts_is_fetched_with_the_voices_of_the_personas():
    model = speech_models.kyutai(["alba", "jean"])
    assert model.size_mb == speech_models.KYUTAI_MB + 2 * speech_models.KYUTAI_VOICE_MB
    assert "anna" in speech_models.kyutai_voices()


def test_a_failure_is_said_in_a_few_words():
    offline = ConnectionError("HTTPSConnectionPool: Max retries")
    assert speech_models.failure(offline) == "No connection"
    full = OSError("[Errno 28] No space left on device")
    assert speech_models.failure(full).startswith("[Errno 28]")
    assert len(speech_models.failure(ValueError("x" * 500))) == 120


def _model(id, log, fail=False, cached=False, size=100):
    def fetch(progress):
        log.append(id)
        if fail:
            raise ConnectionError("connection refused")
        progress(size // 2, size)
        progress(size, size)

    return SpeechModel(id, id.title(), "detail", 1, f"Downloading {id}", lambda: cached, fetch)


def _fetcher(lock=None):
    changes = []
    progress = []
    fetcher = Fetcher(
        lock or threading.RLock(),
        lambda states, busy: changes.append(([dict(s) for s in states], busy)),
        lambda model, done, total: progress.append((model and model.id, done, total)),
    )
    return fetcher, changes, progress


def test_the_fetcher_brings_the_models_one_after_the_other():
    log = []
    models = [_model("a", log), _model("b", log, cached=True), _model("c", log, size=10)]
    fetcher, changes, progress = _fetcher()
    states = fetcher.start(lambda: models)
    assert [s["state"] for s in states] == ["waiting", "done", "waiting"]
    assert _wait_for(lambda: not fetcher.busy)
    assert log == ["a", "c"]
    assert [s["state"] for s in fetcher.states()] == ["done", "done", "done"]
    assert fetcher.states()[2] | {} == {
        "id": "c", "label": "C", "detail": "detail", "size_mb": 1,
        "state": "done", "done": 10, "total": 10, "error": "",
    }  # fmt: skip
    # Each one is seen downloading with its bytes, and the end is said
    assert (["downloading", "done", "waiting"], True) in [
        ([s["state"] for s in states], busy) for states, busy in changes
    ]
    assert ("a", 50, 100) in progress and progress[-1] == (None, 0, 0)
    assert changes[-1][1] is False


def test_a_failed_model_is_kept_failed_until_asked_again():
    log = []
    broken = {"fail": True}
    a = _model("a", log)
    models = [lambda: _model("b", log, fail=broken["fail"]), lambda: a]
    fetcher, _, _ = _fetcher()
    fetcher.start(lambda: [m() for m in models])
    assert _wait_for(lambda: not fetcher.busy)
    states = {s["id"]: s for s in fetcher.states()}
    assert states["b"]["state"] == "failed" and states["b"]["error"] == "No connection"
    assert states["a"]["state"] == "done"
    # Retried: only the failed one is fetched again
    broken["fail"] = False
    fetcher.start(lambda: [m() for m in models])
    assert _wait_for(lambda: not fetcher.busy)
    assert log == ["b", "a", "b"]
    assert {s["state"] for s in fetcher.states()} == {"done"}


def test_a_model_added_while_fetching_is_taken_up():
    log = []
    gate = threading.Event()
    chosen = {"wake": False}

    def slow(progress):
        gate.wait(2)
        log.append("slow")

    slow_model = SpeechModel("slow", "Slow", "", 1, "", lambda: False, slow)

    def choose():
        return [slow_model] + ([_model("wake", log)] if chosen["wake"] else [])

    fetcher, _, _ = _fetcher()
    fetcher.start(choose)
    chosen["wake"] = True  # a wake word chosen on a later step
    fetcher.start(choose)
    gate.set()
    assert _wait_for(lambda: not fetcher.busy)
    assert log == ["slow", "wake"]


def test_nothing_else_downloads_while_the_fetcher_does():
    lock = threading.RLock()
    gate = threading.Event()
    model = SpeechModel("a", "A", "", 1, "", lambda: False, lambda progress: gate.wait(2))
    fetcher, _, _ = _fetcher(lock)
    fetcher.start(lambda: [model])
    assert _wait_for(lambda: fetcher.busy)
    time.sleep(0.05)
    assert not lock.acquire(blocking=False)
    gate.set()
    assert _wait_for(lambda: not fetcher.busy)
    assert lock.acquire(blocking=False)
    lock.release()


def test_describe_tells_what_is_here_without_fetching():
    log = []
    fetcher, changes, _ = _fetcher()
    states = fetcher.describe([_model("a", log, cached=True), _model("b", log)])
    assert [s["state"] for s in states] == ["done", "waiting"]
    assert log == [] and changes == []


# -- the controller ------------------------------------------------------


def test_the_walkthrough_downloads_in_the_background_with_events(monkeypatch):
    log = []
    models = [_model("kyutai", log), _model("stt", log)]
    chosen = []
    monkeypatch.setattr(speech_models, "needed", lambda **options: chosen.append(options) or models)
    bus = EventBus()
    events = _record(bus)
    controller = AssistantController(bus)
    result = controller.setup_download({"wake_word": "Iris", "tts_dir": "/models"})
    assert result["ok"] and [m["id"] for m in result["models"]] == ["kyutai", "stt"]
    assert _wait_for(lambda: not controller.fetcher.busy)
    assert chosen[0]["wake_word"] == "Iris" and chosen[0]["tts_dir"] == "/models"
    updates = [data for kind, data in events if kind == "setup.models"]
    assert updates[-1]["busy"] is False
    assert [m["state"] for m in updates[-1]["models"]] == ["done", "done"]
    # The orb's progress follows the same downloads
    progress = [data for kind, data in events if kind == "setup.progress"]
    assert progress[0]["model"] == "kyutai" and progress[0]["label"] == "Downloading kyutai"
    assert progress[-1]["label"] == ""


def test_starting_waits_for_the_walkthroughs_downloads(monkeypatch):
    gate = threading.Event()
    order = []

    def fetch(progress):
        gate.wait(2)
        order.append("walkthrough")

    first = SpeechModel("stt", "Recognition", "", 1, "", lambda: "walkthrough" in order, fetch)
    monkeypatch.setattr(speech_models, "needed", lambda **options: [first])
    controller = AssistantController(EventBus())
    controller.setup_download({})
    assert _wait_for(lambda: controller.fetcher.busy)
    started = threading.Thread(target=controller._fetch_models, args=({},))
    started.start()
    time.sleep(0.1)
    assert started.is_alive()  # waits instead of fetching the same files
    gate.set()
    started.join(2)
    assert order == ["walkthrough"]


def test_the_greeting_is_queued_as_iris_first_turn(monkeypatch):
    class Assistant:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.stop_event = threading.Event()
            self.suspend_event = threading.Event()
            self.pause_event = threading.Event()
            self.hold_event = threading.Event()
            self.speech_pipeline = SimpleNamespace(is_active=False)

        def config_info(self):
            return {}

        conversation = SimpleNamespace(set_amnesia=lambda on: None, reachable=True)

        def run_loop(self):
            self.stop_event.wait(5)

    monkeypatch.setattr("iris.controller.VoiceAssistant", Assistant)
    monkeypatch.setattr(AssistantController, "_fetch_models", lambda self, kwargs: None)
    monkeypatch.setattr(AssistantController, "_pick_vision_model", lambda self, *args: None)
    controller = AssistantController(EventBus())
    controller.start({"greet": True, "persona": "iris"})
    assert _wait_for(lambda: controller.assistant is not None)
    assert "greet" not in controller.assistant.kwargs
    assert controller._turns.get_nowait() == {"greet": True}
    controller.assistant.stop_event.set()
    # Asked for later, as at the end of a walkthrough run again
    assert controller.greet() == {"ok": True}
    assert controller._turns.get_nowait() == {"greet": True}


def test_a_voice_cannot_be_previewed_before_it_is_downloaded(monkeypatch):
    controller = AssistantController(EventBus())
    monkeypatch.setattr(controller, "_voice_downloaded", lambda engine: False)
    result = controller.persona_preview({"engine": "kyutai", "voice": "alba", "name": "Iris"})
    assert result == {"ok": False, "error": "The voice is still downloading"}


def test_a_voice_is_previewed_before_iris_runs(monkeypatch):
    spoken = []

    class Generator:
        def __init__(self, engine, voice, speed, models_dir=None):
            self.engine = engine

    class Pipeline:
        def __init__(self, generator, player):
            self.on_output_level = None
            self.text_queue = SimpleNamespace(put=lambda item: None)
            self._active = False

        @property
        def is_active(self):
            return self._active

        def stop(self):
            self._active = False

        def queue_speech(self, text, language=None, sound=None):
            spoken.append((text, language, sound["engine"]))
            self._active = True
            threading.Timer(0.2, self.stop).start()

    monkeypatch.setattr("iris.controller.SpeechGenerator", Generator)
    monkeypatch.setattr("iris.controller.SpeechPipeline", Pipeline)
    monkeypatch.setattr("iris.controller.AudioPlayer", lambda: None)
    bus = EventBus()
    events = _record(bus)
    controller = AssistantController(bus)
    monkeypatch.setattr(controller, "_voice_downloaded", lambda engine: True)
    result = controller.persona_preview({"engine": "kokoro", "voice": "bm_lewis", "name": "HAL"})
    assert result == {"ok": True}
    states = lambda: [d["state"] for k, d in events if k == "state"]  # noqa: E731
    assert _wait_for(lambda: states() == ["speaking", "idle"])
    assert spoken == [("Hi, I'm HAL. This is how I sound when we talk.", "en", "kokoro")]
    # Iris starting gives the speaker back
    controller._release_first_run()
    assert controller._preview is None


def test_the_sound_check_reports_as_mic_check_events(monkeypatch):
    made = []

    class Check:
        def __init__(self, on_update, device=None):
            self.on_update, self.device, self.stopped = on_update, device, False
            made.append(self)

        def start(self):
            self.on_update({"level": 0.4, "heard": True, "silent": False})

        def stop(self):
            self.stopped = True

        def device_name(self):
            return "Desk microphone"

    monkeypatch.setattr("iris.controller.SoundCheck", Check)
    bus = EventBus()
    events = _record(bus)
    controller = AssistantController(bus)
    assert controller.sound_check(True, 3) == {"ok": True, "device": "Desk microphone"}
    assert made[0].device == 3
    assert ("mic.check", {"level": 0.4, "heard": True, "silent": False}) in events
    # Another microphone closes the first; off closes it
    controller.sound_check(True, None)
    assert made[0].stopped and not made[1].stopped
    assert controller.sound_check(False) == {"ok": True}
    assert made[1].stopped


def test_a_microphone_that_cannot_be_opened_is_said(monkeypatch):
    class Check:
        def __init__(self, on_update, device=None):
            pass

        def start(self):
            raise RuntimeError("Error querying device -1")

    monkeypatch.setattr("iris.controller.SoundCheck", Check)
    result = AssistantController(EventBus()).sound_check(True)
    assert result["ok"] is False and "could not be opened" in result["error"]


def test_plugins_say_whether_they_are_on_by_default():
    plugins = {p["id"]: p for p in AssistantController(EventBus()).plugins_list()["plugins"]}
    assert plugins["camera"]["enabled_by_default"] is True
    assert plugins["camera"]["enabled"] is None  # Iris does not run


# -- the sound check -----------------------------------------------------


class _Stream:
    def __init__(self, callback, **options):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class _Microphone:
    """A microphone the test speaks into, block by block."""

    def __init__(self):
        self.stream = None

    def input_stream(self, **options):
        self.stream = _Stream(**options)
        return self.stream

    def query_devices(self, *args, **kwargs):
        return {"name": "Test microphone"}

    def feed(self, value: float, seconds: float):
        block = 3 * STEP
        for _ in range(int(seconds * SAMPLE_RATE / block)):
            time.sleep(block / SAMPLE_RATE / 4)
            now = time.monotonic()
            info = SimpleNamespace(currentTime=now, inputBufferAdcTime=now - 0.01)
            self.stream.callback(np.full((block, 1), value, dtype=np.float32), block, info, None)


def test_the_sound_check_hears_a_voice_and_shows_its_level():
    updates = []
    microphone = _Microphone()
    # Loud frames are speech, as far as this test's detector goes
    check = SoundCheck(
        updates.append, audio=microphone, detector=lambda frame: float(abs(frame).max() > 0.05)
    )
    check.start()
    try:
        assert check.device_name() == "Test microphone"
        microphone.feed(0.01, 0.5)  # the room
        assert _wait_for(lambda: updates)
        assert not any(u["heard"] for u in updates)
        microphone.feed(0.1, HEARD_AFTER + 0.2)  # the user speaks
        assert _wait_for(lambda: updates[-1]["heard"])
        assert max(u["level"] for u in updates) == pytest.approx(0.1 / 0.15, rel=0.01)
        assert not any(u["silent"] for u in updates)
    finally:
        check.stop()


def test_the_sound_check_says_when_the_microphone_gives_only_silence(monkeypatch):
    monkeypatch.setattr("iris.audio.SILENT_AFTER", 0.2)
    updates = []
    microphone = _Microphone()
    check = SoundCheck(updates.append, audio=microphone, detector=lambda frame: 0.0)
    check.start()
    try:
        microphone.feed(0.0, 1.2)
        assert _wait_for(lambda: updates and updates[-1]["silent"])
    finally:
        check.stop()


# -- servers on this computer --------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    body = b""

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_a_model_server_on_this_computer_is_found(local_server):
    _Handler.body = json.dumps({"object": "list", "data": [{"id": "a-model"}]}).encode()
    nobody = "http://127.0.0.1:9/v1"
    assert discover((f"{local_server}/v1", nobody), timeout=0.5) == [f"{local_server}/v1"]


def test_something_else_on_a_usual_port_is_not_taken_for_one(local_server):
    _Handler.body = b"<html>a web app</html>"
    assert discover((f"{local_server}/v1",), timeout=0.5) == []


# -- the greeting --------------------------------------------------------


def test_what_iris_was_asked_to_open_with_is_kept_as_a_note():
    conversation = ConversationManager.__new__(ConversationManager)
    conversation.memory = None
    conversation.amnesia = False
    conversation.conversation_history = []
    conversation._record("Greet the user.", "Hello! I'm Iris.", OPENING)
    assert conversation.conversation_history == [
        {"role": "system", "content": "Greet the user."},
        {"role": "assistant", "content": "Hello! I'm Iris."},
    ]


def test_the_greeting_opens_the_conversation_without_a_user_turn():
    asked = []
    assistant = VoiceAssistant.__new__(VoiceAssistant)
    assistant._turn_active = threading.Event()
    assistant.wake_word = "Iris"
    assistant.wake_window = 30.0
    assistant.awake_until = 0.0
    assistant.event_bus = None
    assistant.conversation_language = "en"

    def chat(prompt, response_language="en", source=None, attachments=None):
        asked.append((prompt, source))
        return "Hello"

    assistant.chat = chat
    assistant._greet()
    assert asked == [(GREETING, OPENING)]
    # Iris stays awake for the answer
    assert assistant.awake_until > time.monotonic()
    assert not assistant._turn_active.is_set()


def test_the_pocket_files_are_its_model_tokenizer_and_voices():
    files = [HubFile.parse(path) for path in pocket_files(["alba", "not-a-voice"])]
    names = [Path(f.filename).name for f in files]
    assert names == ["model.safetensors", "tokenizer.json", "alba.safetensors"]
    assert all(f.revision for f in files)

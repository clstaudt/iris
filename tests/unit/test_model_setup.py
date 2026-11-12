"""Unit tests for getting speech models: local folders, downloads, checks."""

import pytest

from iris import downloads, transcription, tts
from iris.controller import AssistantController
from iris.events import EventBus
from iris.transcription import WhisperTranscriber, folder_engine
from iris.turns import SMART_TURN_FILE


def test_a_model_folder_names_its_engine(tmp_path):
    ct2 = tmp_path / "faster-whisper-small"
    ct2.mkdir()
    (ct2 / "model.bin").write_bytes(b"")
    mlx = tmp_path / "whisper-turbo-mlx"
    mlx.mkdir()
    (mlx / "weights.safetensors").write_bytes(b"")
    assert folder_engine(ct2) == "faster-whisper"
    assert folder_engine(mlx) == "mlx"
    assert folder_engine(tmp_path) is None


def test_the_transcriber_loads_a_local_folder(tmp_path, monkeypatch):
    folder = tmp_path / "faster-whisper-small"
    folder.mkdir()
    (folder / "model.bin").write_bytes(b"")
    loaded = []
    monkeypatch.setattr(transcription, "WhisperModel", lambda path, **kw: loaded.append(path))
    t = WhisperTranscriber(str(folder))
    assert t.engine == "faster-whisper"
    assert t.model_path == folder and t.model_name == "faster-whisper-small"
    assert loaded == [str(folder)]


def test_a_folder_without_a_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="holds no Whisper model"):
        WhisperTranscriber(str(tmp_path))


def test_kokoro_files_are_found_where_they_usually_live(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(tts, "DEFAULT_KOKORO_DIR", tmp_path / "nowhere")
    monkeypatch.chdir(tmp_path)
    assert tts.find_kokoro_models() is None
    (tmp_path / tts.KOKORO_MODEL_FILE).write_bytes(b"")
    (tmp_path / tts.KOKORO_VOICES_FILE).write_bytes(b"")
    assert tts.find_kokoro_models() == tmp_path


def test_a_chosen_voice_folder_must_hold_the_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not hold"):
        tts.KokoroGenerator(models_dir=tmp_path)


class FakeResponse:
    def __init__(self, body):
        self.body = body
        self.headers = {"content-length": str(len(body))}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 4):
            yield self.body[i : i + 4]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_files_download_with_progress_and_land_whole(tmp_path, monkeypatch):
    bodies = {"https://x/model": b"m" * 10, "https://x/voices": b"v" * 6}
    monkeypatch.setattr(downloads.requests, "head", lambda url, **kw: FakeResponse(bodies[url]))
    monkeypatch.setattr(downloads.requests, "get", lambda url, **kw: FakeResponse(bodies[url]))
    seen = []
    downloads.download_files(
        [("https://x/model", tmp_path / "m.onnx"), ("https://x/voices", tmp_path / "v.bin")],
        lambda done, total: seen.append((done, total)),
        interval=0,
    )
    assert (tmp_path / "m.onnx").read_bytes() == b"m" * 10
    assert (tmp_path / "v.bin").read_bytes() == b"v" * 6
    assert not list(tmp_path.glob("*.part"))
    assert seen[-1] == (16, 16)
    assert all(b[0] >= a[0] for a, b in zip(seen, seen[1:]))


def test_bad_model_folders_are_refused_at_once(tmp_path):
    controller = AssistantController(EventBus())
    controller.assistant = object()  # running, as far as the check goes
    result = controller.set_model_paths(stt_path=str(tmp_path))
    assert result["ok"] is False and "holds no Whisper model" in result["error"]
    result = controller.set_model_paths(tts_dir=str(tmp_path))
    assert result["ok"] is False and "does not hold" in result["error"]


def _record(bus):
    events = []
    bus.subscribe(lambda event_type, data: events.append((event_type, data)))
    return events


def test_first_start_fetches_what_is_missing_with_progress(tmp_path, monkeypatch):
    fetched = []

    def download(repo, on_progress, filename=None):
        fetched.append(filename or repo)
        on_progress(50, 100)

    def download_files(files, on_progress):
        fetched.extend(target.name for _, target in files)
        on_progress(3, 4)

    monkeypatch.setattr(downloads, "is_cached", lambda repo, filename=None: False)
    monkeypatch.setattr(downloads, "download", download)
    monkeypatch.setattr(downloads, "download_files", download_files)
    monkeypatch.setattr("iris.controller.find_kokoro_models", lambda: None)
    monkeypatch.setattr("iris.controller.DEFAULT_KOKORO_DIR", tmp_path)

    bus = EventBus()
    events = _record(bus)
    AssistantController(bus)._fetch_models(
        {"whisper_model": "small", "whisper_engine": "faster-whisper", "whisper_device": "cpu"}
    )

    assert fetched == [
        "Systran/faster-whisper-small",
        SMART_TURN_FILE,
        tts.KOKORO_MODEL_FILE,
        tts.KOKORO_VOICES_FILE,
    ]
    progress = [data for kind, data in events if kind == "setup.progress"]
    steps = [(p["model"], p["done"], p["total"]) for p in progress[:3]]
    assert steps == [("stt", 50, 100), ("turn", 50, 100), ("tts", 3, 4)]
    # Done: the UI drops its progress and leaves the setup screen
    assert progress[-1]["label"] == ""


def test_chosen_folders_are_used_as_they_are(tmp_path, monkeypatch):
    stt = tmp_path / "whisper"
    stt.mkdir()
    (stt / "model.bin").write_bytes(b"")
    monkeypatch.setattr(downloads, "download", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(downloads, "download_files", lambda *a: pytest.fail("downloaded"))
    # Smart Turn has no folder to choose: it is fetched unless cached
    monkeypatch.setattr(downloads, "is_cached", lambda repo, filename=None: True)

    bus = EventBus()
    events = _record(bus)
    controller = AssistantController(bus)
    controller._fetch_models({"whisper_model": str(stt), "tts_models_dir": str(tmp_path)})
    assert [data["label"] for kind, data in events if kind == "setup.progress"] == [""]


def test_setup_info_says_what_is_already_here(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "is_cached", lambda repo, filename=None: True)
    monkeypatch.setattr("iris.controller.find_kokoro_models", lambda: tmp_path)
    info = AssistantController(EventBus()).setup_info()
    assert info["stt"]["cached"] is True and info["stt"]["model"]
    assert info["tts"] == {"found": str(tmp_path), "size_mb": 335}
    assert info["ollama_url"].startswith("http")


def test_good_model_folders_pass_the_check(tmp_path):
    stt = tmp_path / "whisper-mlx"
    stt.mkdir()
    (stt / "weights.npz").write_bytes(b"")
    (tmp_path / tts.KOKORO_MODEL_FILE).write_bytes(b"")
    (tmp_path / tts.KOKORO_VOICES_FILE).write_bytes(b"")
    assert AssistantController.check_model_paths(str(stt), str(tmp_path)) == {"ok": True}


class _Conversation:
    reachable = False


class _Assistant:
    conversation = _Conversation()


def test_a_missing_ollama_is_reported_plainly(monkeypatch):
    bus = EventBus()
    events = _record(bus)
    controller = AssistantController(bus)
    controller.assistant = _Assistant()
    assert controller._warn_if_ollama_unreachable() is False
    errors = [data for kind, data in events if kind == "error"]
    assert errors[0]["kind"] == "ollama_unreachable"
    assert "ollama.com" in errors[0]["message"]

    # Once a server answers, a reloaded UI no longer sees the warning
    class Client:
        def list(self):
            return {"models": []}

    monkeypatch.setattr("iris.controller.ollama_client", lambda host, timeout=None: Client())
    controller.assistant = None
    assert controller.set_ollama_host("http://other:11434")["ok"] is True
    assert not [e for e in controller.history()["events"] if e["type"] == "error"]

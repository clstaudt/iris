"""The walkthrough's Download: a model pulled onto an Ollama server, with its
progress told as llm.pull events."""

import threading
from types import SimpleNamespace

import pytest

from iris import controller as controller_module
from iris.controller import AssistantController
from iris.events import EventBus


def _part(status, digest=None, total=None, completed=None):
    return SimpleNamespace(status=status, digest=digest, total=total, completed=completed)


class FakeOllama:
    """Ollama's client, pulling what it is given: each layer in halves."""

    def __init__(self, fail=None):
        self.pulled = []
        self.fail = fail

    def pull(self, model, stream):
        assert stream
        self.pulled.append(model)
        yield _part("pulling manifest")
        for half in (1, 2):
            yield _part("pulling a", "sha256:a", 4000, 2000 * half)
            yield _part("pulling b", "sha256:b", 1000, 500 * half)
            if self.fail and half == 1:
                raise self.fail
        yield _part("success")


@pytest.fixture
def pulling(monkeypatch):
    """A controller whose Ollama is the fake, with its llm.pull events and a
    way to wait for the download's end."""
    bus = EventBus()
    events, ended = [], threading.Event()

    def seen(kind, data):
        if kind == "llm.pull":
            events.append(data)
            if data["state"] != "downloading":
                ended.set()

    bus.subscribe(seen)
    fake = FakeOllama()
    monkeypatch.setattr(controller_module, "resolve_host", lambda url: url or None)
    monkeypatch.setattr(controller_module, "ollama_client", lambda host, **kw: fake)
    monkeypatch.setattr(controller_module, "PULL_TELL_EVERY", 0)
    return AssistantController(bus), fake, events, ended


def test_a_model_downloads_with_its_progress(pulling):
    controller, fake, events, ended = pulling

    assert controller.pull_model("http://127.0.0.1:11434", "qwen3.5:9b") == {"ok": True}
    assert ended.wait(5)

    assert fake.pulled == ["qwen3.5:9b"]
    assert all(e["model"] == "qwen3.5:9b" for e in events)
    downloading = [(e["done"], e["total"]) for e in events if e["state"] == "downloading"]
    assert (2500, 5000) in downloading
    assert events[-1] == {"model": "qwen3.5:9b", "state": "done", "done": 5000, "total": 5000}


def test_a_failed_download_says_why_and_can_be_tried_again(pulling):
    controller, fake, events, ended = pulling
    fake.fail = ConnectionError("the server went away")

    controller.pull_model("", "qwen3.5:9b")
    assert ended.wait(5)
    assert events[-1]["state"] == "failed"
    assert "the server went away" in events[-1]["error"]

    fake.fail, events[:] = None, []
    ended.clear()
    assert controller.pull_model("", "qwen3.5:9b") == {"ok": True}
    assert ended.wait(5)
    assert events[-1]["state"] == "done"


def test_only_ollama_downloads_and_one_model_at_a_time(pulling, monkeypatch):
    controller, fake, events, ended = pulling

    refused = controller.pull_model("http://127.0.0.1:8000/v1", "qwen3.5:9b")
    assert refused["ok"] is False and "Ollama" in refused["error"]
    assert controller.pull_model("", "")["ok"] is False

    controller._pulling = "qwen3.5:9b"
    busy = controller.pull_model("", "another")
    assert busy == {"ok": False, "error": "qwen3.5:9b is downloading"}
    assert fake.pulled == []

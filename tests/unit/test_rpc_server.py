"""Roundtrip tests for the JSON-RPC stdio server used by the Electron shell."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from iris.rpc_server import encode_frame, serve

REPO_ROOT = Path(__file__).resolve().parents[2]


class RPCClient:
    """Minimal NDJSON JSON-RPC client over subprocess stdio."""

    def __init__(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "iris.rpc_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            cwd=REPO_ROOT,
        )
        self.responses = {}
        self.events = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            with self._lock:
                if frame.get("method") == "event":
                    self.events.append(frame["params"])
                elif "id" in frame:
                    self.responses[frame["id"]] = frame

    def call(self, method: str, params: dict | None = None, timeout: float = 10.0) -> dict:
        req_id = id(method) ^ int(time.monotonic() * 1e6)
        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if req_id in self.responses:
                    return self.responses.pop(req_id)
            time.sleep(0.02)
        raise TimeoutError(f"No response for {method}")

    def wait_for_event(self, event_type: str, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                for event in self.events:
                    if event["type"] == event_type:
                        return event
            time.sleep(0.02)
        raise TimeoutError(f"No event of type {event_type}")

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)


@pytest.fixture
def client():
    c = RPCClient()
    yield c
    c.close()


def test_app_info_roundtrip(client):
    response = client.call("app.info")
    assert "error" not in response
    result = response["result"]
    assert result["running"] is False
    assert result["state"] == "idle"
    assert "iris" in result["personas"]
    assert "version" in result


def test_the_walkthrough_is_answered_before_iris_runs(client):
    assert client.call("mic.check", {"on": False})["result"] == {"ok": True}
    assert client.call("assistant.greet")["result"] == {"ok": True}
    assert isinstance(client.call("llm.find_servers")["result"]["servers"], list)
    probe = client.call("llm.probe", {"url": "http://127.0.0.1:9"})["result"]
    assert probe["ok"] is False and "Cannot reach" in probe["error"]
    plugins = client.call("plugins.list")["result"]["plugins"]
    assert all("enabled_by_default" in plugin for plugin in plugins)


def test_unknown_method_returns_error(client):
    response = client.call("no.such.method")
    assert "error" in response
    assert "Unknown method" in response["error"]["message"]


def test_event_notification_framing(client):
    client.call("debug.emit", {"type": "state", "data": {"state": "listening"}})
    event = client.wait_for_event("state")
    assert event["data"] == {"state": "listening"}


def test_encode_frame_handles_numpy_scalars():
    """Mic levels are numpy floats; plain json.dumps would drop the event."""
    frame = {"params": {"data": {"rms": np.float32(0.25)}}}
    assert json.loads(encode_frame(frame))["params"]["data"]["rms"] == pytest.approx(0.25)


def test_persona_list(client):
    response = client.call("persona.list")
    assert "error" not in response
    assert isinstance(response["result"]["personas"], list)
    assert len(response["result"]["personas"]) >= 1


def test_persona_editor_gets_a_persona_and_the_voices(client):
    persona = client.call("persona.get", {"id": "iris"})["result"]
    assert persona["ok"] is True
    assert persona["engine"] in ("kyutai", "kokoro")
    assert persona["shipped"] is True

    voices = client.call("persona.voices")["result"]
    assert any(v["id"] == "alba" for v in voices["engines"]["kyutai"])
    assert "intercom" in voices["filters"]


def test_a_persona_without_a_name_is_not_saved(client):
    result = client.call("persona.save", {"name": " ", "engine": "kyutai", "voice": "alba"})
    assert result["result"]["ok"] is False


def test_a_persona_is_exported_and_imported_only_with_a_file_chosen(client):
    exported = client.call("persona.export", {"id": "iris", "path": ""})["result"]
    imported = client.call("persona.import", {"path": ""})["result"]
    assert exported == imported == {"ok": False, "error": "No file was chosen"}


def test_pause_resume_require_running_assistant(client):
    for method in ("assistant.pause", "assistant.resume"):
        response = client.call(method)
        assert "error" not in response
        assert response["result"]["ok"] is False


def test_audio_list_inputs(client):
    response = client.call("audio.list_inputs")
    assert "error" not in response
    devices = response["result"]["devices"]
    assert isinstance(devices, list)
    for device in devices:
        assert set(device.keys()) == {"index", "name"}


def test_audio_set_input_requires_running_assistant(client):
    response = client.call("audio.set_input", {"index": 0})
    assert "error" not in response
    assert response["result"]["ok"] is False


def test_llm_list(client):
    response = client.call("llm.list", timeout=30)
    assert "error" not in response
    models = response["result"]["models"]
    assert isinstance(models, list)
    for model in models:
        assert set(model.keys()) == {"name", "tools", "vision"}


def test_llm_set_requires_running_assistant(client):
    response = client.call("llm.set", {"name": "some-model"})
    assert "error" not in response
    assert response["result"]["ok"] is False


def test_stop_never_waits_behind_a_slow_request():
    """Stop and the other controls are answered as they arrive; the rest in
    the order they came."""
    answered = []
    lock = threading.Lock()
    release = threading.Event()

    def slow(params):
        release.wait(5)
        return {"ok": True}

    methods = {
        "llm.set_host": slow,
        "persona.list": lambda params: {"personas": []},
        "assistant.interrupt": lambda params: {"ok": True},
    }
    lines = iter(
        json.dumps({"jsonrpc": "2.0", "id": n, "method": m})
        for n, m in enumerate(["llm.set_host", "persona.list", "assistant.interrupt"], 1)
    )

    def read_line():
        line = next(lines, None)
        if line is None:
            # The interrupt was answered while the slow request still runs
            deadline = time.monotonic() + 2
            while 3 not in [f["id"] for f in answered] and time.monotonic() < deadline:
                time.sleep(0.01)
            release.set()
            return ""
        return line + "\n"

    def write(frame):
        with lock:
            answered.append(frame)

    serve(read_line, methods, write)
    assert [frame["id"] for frame in answered] == [3, 1, 2]
    assert all("result" in frame for frame in answered)


def test_a_check_of_the_servers_holds_up_no_other_request():
    """Checking the servers takes seconds when one is away; a message sent
    meanwhile is answered without waiting for it."""
    answered = []
    lock = threading.Lock()
    release = threading.Event()

    def check(params):
        release.wait(5)
        return {"servers": []}

    methods = {
        "llm.check_hosts": check,
        "assistant.send_text": lambda params: {"ok": True},
    }
    lines = iter(
        json.dumps({"jsonrpc": "2.0", "id": n, "method": m})
        for n, m in enumerate(["llm.check_hosts", "assistant.send_text"], 1)
    )

    def read_line():
        line = next(lines, None)
        if line is None:
            # The message was answered while the check still runs
            deadline = time.monotonic() + 2
            while 2 not in [f["id"] for f in answered] and time.monotonic() < deadline:
                time.sleep(0.01)
            release.set()
            deadline = time.monotonic() + 2
            while len(answered) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            return ""
        return line + "\n"

    def write(frame):
        with lock:
            answered.append(frame)

    serve(read_line, methods, write)
    assert [frame["id"] for frame in answered] == [2, 1]
    assert all("result" in frame for frame in answered)


def test_a_line_that_is_not_json_is_answered_with_an_error():
    lines = iter(["not json\n", ""])
    answered = []
    serve(lambda: next(lines), {}, answered.append)
    assert answered[0]["error"]["code"] == -32700

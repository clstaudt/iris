"""Servers with the OpenAI-compatible API that list their models at
/v1/models in Ollama's shape ({"models": [{"name": ..., "model": ...,
"capabilities": [...]}]}) rather than the OpenAI API's ({"data": [{"id":
...}]}), against a stand-in on this computer: Iris finds them, lists their
models, and knows which can see."""

import http.server
import json
import threading
from urllib.parse import urlsplit

import pytest

from iris import llm_client, servers
from iris.llm_client import OpenAIClient, model_entries, resolve_host


class OllamaShapedServer:
    """A stand-in that answers only the OpenAI-compatible API (no
    /api/version), listing its models in Ollama's shape."""

    def __init__(self, models):
        self.models = models
        stand_in = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                if urlsplit(self.path).path == "/v1/models":
                    stand_in.send(self, 200, {"models": stand_in.models})
                else:
                    stand_in.send(self, 404, {"error": {"message": "not found"}})

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    @staticmethod
    def send(handler, status, body):
        data = json.dumps(body).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


SEES = {"name": "robin-flash", "model": "robin-flash", "capabilities": ["completion", "multimodal"]}
BLIND = {"name": "felix-small", "model": "felix-small", "capabilities": ["completion"]}


@pytest.fixture
def server():
    stand_in = OllamaShapedServer([SEES, BLIND])
    yield stand_in
    stand_in.close()


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(llm_client, "_resolved", {})
    monkeypatch.delenv(llm_client.API_KEY_ENV, raising=False)


def test_a_server_listing_models_in_ollamas_shape_is_taken_as_openai_compatible(server):
    # Typed without /v1, as people copy it: not Ollama (no /api/version),
    # but its /v1/models answers with a list of models
    assert resolve_host(server.url) == f"{server.url}/v1"
    assert servers.model_server(f"{server.url}/v1")


def test_its_models_are_listed_by_name(server):
    listed = OpenAIClient(f"{server.url}/v1").list()
    assert [m.model for m in listed.models] == ["robin-flash", "felix-small"]


def test_a_model_said_to_be_multimodal_sees_images(server):
    client = OpenAIClient(f"{server.url}/v1")
    assert "vision" in client.show("robin-flash").capabilities
    assert "vision" not in client.show("felix-small").capabilities


def test_the_openai_shape_is_read_as_before():
    body = {"object": "list", "data": [{"id": "ada-chat", "object": "model"}]}
    assert [m["id"] for m in model_entries(body)] == ["ada-chat"]


def test_entries_without_a_name_and_answers_without_a_list_give_no_models():
    assert model_entries({"models": [{"capabilities": ["completion"]}, "text"]}) == []
    assert model_entries({"models": "none"}) == []
    assert model_entries(["robin-flash"]) == []

"""Model servers are checked for whether they answer."""

import http.server
import threading

import pytest

from iris import servers
from iris.controller import AssistantController


class Handler(http.server.BaseHTTPRequestHandler):
    answers: dict = {}

    def do_GET(self):
        self.send_response(self.answers.get(self.path, 404))
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    """A local server whose answers per path are set by the test."""
    Handler.answers = {}
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_ollama_answers_at_its_root(server):
    Handler.answers = {"/api/version": 200}
    assert servers.reachable(server)
    assert servers.reachable(f"{server}/v1")
    assert servers.reachable(f"{server}/")


def test_an_openai_compatible_server_answers_at_v1(server):
    Handler.answers = {"/v1/models": 200}
    assert servers.reachable(f"{server}/v1")
    assert servers.reachable(server)


def test_a_server_that_wants_a_key_is_there(server):
    Handler.answers = {"/v1/models": 401}
    assert servers.reachable(f"{server}/v1")


def test_something_else_at_the_address_is_no_model_server(server):
    assert not servers.reachable(server)


def test_nothing_at_the_address():
    assert not servers.reachable("http://127.0.0.1:9", timeout=0.5)


def test_a_url_without_scheme_is_tried_as_http(server):
    Handler.answers = {"/api/version": 200}
    assert servers.reachable(server.removeprefix("http://"))


def test_the_default_server_is_the_empty_url(server, monkeypatch):
    Handler.answers = {"/api/version": 200}
    monkeypatch.setattr(servers, "effective_host", lambda host: server)
    assert servers.reachable("")


def test_several_are_checked_in_their_order(server):
    Handler.answers = {"/api/version": 200}
    result = AssistantController.check_servers([server, "http://127.0.0.1:9"])
    assert result == {
        "servers": [
            {"url": server, "reachable": True},
            {"url": "http://127.0.0.1:9", "reachable": False},
        ]
    }
    assert servers.check([]) == []

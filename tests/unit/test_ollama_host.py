"""Unit tests for configuring the Ollama server URL."""

from unittest.mock import patch

from iris.controller import AssistantController
from iris.events import EventBus
from iris.llm import effective_host, normalize_host, ollama_client


def test_untidy_urls_are_normalized():
    base = "https://box.example:11434"
    # /v1 stays until the server is asked what it is (resolve_host)
    assert normalize_host(f"{base}/v1/") == f"{base}/v1"
    assert normalize_host(f"  {base}/api ") == base
    assert normalize_host("") is None
    assert normalize_host(None) is None


def test_default_client_is_the_module(monkeypatch):
    import ollama

    assert ollama_client(None) is ollama
    assert ollama_client("") is ollama


def test_effective_host_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    assert effective_host(None) == "http://gpu-box:11434"
    assert effective_host("https://other:1/v1") == "https://other:1/v1"
    monkeypatch.delenv("OLLAMA_HOST")
    assert effective_host(None) == "http://localhost:11434"


def test_unreachable_server_is_refused_and_not_adopted():
    controller = AssistantController(EventBus())
    with patch("iris.controller.resolve_host", lambda url: url), patch(
        "iris.controller.ollama_client"
    ) as make_client:
        make_client.return_value.list.side_effect = ConnectionError("refused")
        result = controller.set_ollama_host("https://nowhere:1/v1")
    assert result["ok"] is False
    assert "nowhere" in result["error"]
    assert controller._ollama_host is None


def test_reachable_server_is_adopted_before_start():
    controller = AssistantController(EventBus())
    # An Ollama server given with /v1 is kept as its root
    resolved = {"https://box:11434/v1": "https://box:11434"}
    with patch("iris.controller.resolve_host", resolved.get), patch(
        "iris.controller.ollama_client"
    ) as make_client:
        make_client.return_value.list.return_value = []
        result = controller.set_ollama_host("https://box:11434/v1")
    assert result == {"ok": True, "host": "https://box:11434"}
    assert controller._ollama_host == "https://box:11434"


def test_another_server_is_adopted_with_its_v1():
    controller = AssistantController(EventBus())
    with patch("iris.controller.resolve_host", lambda url: url), patch(
        "iris.controller.ollama_client"
    ) as make_client:
        make_client.return_value.list.return_value = []
        result = controller.set_ollama_host("https://mlx:8000/v1")
    assert result == {"ok": True, "host": "https://mlx:8000/v1"}

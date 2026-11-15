"""Unit tests for servers with the OpenAI-compatible API: telling them from
Ollama, and translating requests and streamed replies. HTTP is answered by
httpx.MockTransport, so no server is needed."""

import json

import httpx
import ollama
import pytest

from iris import llm, llm_client, memory
from iris.llm import ConversationManager, ollama_client
from iris.llm_client import OpenAIClient, is_openai, normalize_host, openai_messages, resolve_host
from iris.plugin import Plugin, Tool
from iris.tools import Tools

BASE = "https://box.example:8000/v1"
PNG = "iVBORw0KGgoAAAANSUhEUg"
JPEG = "/9j/4AAQSkZJRgABAQ"


def _sse(*events) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def _delta(**delta) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


class Server:
    """Answers chat requests with scripted replies and keeps what it got."""

    def __init__(self, *replies, models=None):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.headers: list[httpx.Headers] = []
        self.models = models or [{"id": "big-model", "object": "model"}]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.headers.append(request.headers)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": self.models})
        body = json.loads(request.content)
        self.requests.append(body)
        reply = self.replies.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, content=_sse(*reply), headers={"content-type": "text/event-stream"})

    def client(self) -> OpenAIClient:
        return OpenAIClient(BASE, transport=httpx.MockTransport(self))


def test_urls_keep_v1_and_drop_single_call_paths():
    assert normalize_host(f" {BASE}/ ") == BASE
    assert normalize_host(f"{BASE}/chat/completions") == BASE
    assert normalize_host(f"{BASE}/models") == BASE
    assert normalize_host("http://localhost:11434/api") == "http://localhost:11434"
    assert normalize_host("") is None
    assert is_openai(BASE) and not is_openai("http://localhost:11434") and not is_openai(None)


def test_the_client_follows_the_kept_url():
    assert isinstance(ollama_client(BASE), OpenAIClient)
    assert isinstance(ollama_client("http://localhost:11434"), ollama.Client)
    assert ollama_client(None) is ollama


class Probe:
    """Stands in for httpx.get: {url: (status, json)}, or an exception."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def __call__(self, url, **kwargs):
        self.asked.append(url)
        answer = self.answers.get(url, (404, {"message": "unknown route"}))
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        return httpx.Response(status, json=body, request=httpx.Request("GET", url))


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(llm_client, "_resolved", {})

    def install(answers):
        fake = Probe(answers)
        monkeypatch.setattr(llm_client.httpx, "get", fake)
        return fake

    return install


def test_ollama_is_kept_as_its_root_even_with_v1(probe):
    probe({"https://ol:11434/api/version": (200, {"version": "0.12.0"})})
    assert resolve_host("https://ol:11434/v1") == "https://ol:11434"
    assert resolve_host("https://ol:11434") == "https://ol:11434"


def test_another_server_keeps_its_v1(probe):
    probe({})  # every route unknown: not Ollama
    assert resolve_host(BASE) == BASE


def test_v1_is_found_when_left_out(probe):
    probe({"https://box.example:8000/v1/models": (200, {"object": "list", "data": []})})
    assert resolve_host("https://box.example:8000") == BASE


def test_a_server_that_does_not_answer_keeps_its_url_and_is_asked_again(probe):
    fake = probe({"https://off:1/api/version": httpx.ConnectError("refused")})
    assert resolve_host("https://off:1/v1") == "https://off:1/v1"
    assert resolve_host("https://off:1/v1") == "https://off:1/v1"
    assert fake.asked.count("https://off:1/api/version") == 2


def test_an_answer_is_remembered(probe):
    fake = probe({"https://ol:11434/api/version": (200, {"version": "0.12.0"})})
    resolve_host("https://ol:11434/v1")
    resolve_host("https://ol:11434/v1")
    assert len(fake.asked) == 1


def test_images_tool_calls_and_tool_results_are_translated():
    call = ollama.Message.ToolCall(
        function=ollama.Message.ToolCall.Function(name="look", arguments={"angle": 2})
    )
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is this?", "images": [PNG]},
        {"role": "assistant", "content": "Let me look.", "tool_calls": [call, call]},
        {"role": "tool", "content": "Image captured", "images": [JPEG]},
        {"role": "tool", "content": "Error: no camera"},
        {"role": "system", "content": "Answer now."},
    ]
    out = openai_messages(messages)

    assert out[0] == {"role": "system", "content": "Be brief."}
    assert out[1]["content"] == [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
    ]
    calls = out[2]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["look", "look"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"angle": 2}
    assert len({c["id"] for c in calls}) == 2
    # Each result answers the call in its position
    assert out[3] == {"role": "tool", "tool_call_id": calls[0]["id"], "content": "Image captured"}
    assert out[4]["tool_call_id"] == calls[1]["id"]
    # The image follows the results, before the next message
    assert out[5] == {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{JPEG}"}}],
    }
    assert out[6] == {"role": "system", "content": "Answer now."}


def test_a_stream_yields_text_then_the_pieced_tool_calls():
    server = Server(
        [
            _delta(role="assistant"),
            _delta(reasoning_content="The user wants the time."),
            _delta(content="Let me "),
            _delta(content="check."),
            _delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "get_time", "arguments": '{"ci'}}]),
            _delta(tool_calls=[{"index": 0, "function": {"arguments": 'ty": "Tokyo"}'}}]),
            _delta(tool_calls=[{"index": 1, "id": "c2", "function": {"name": "search", "arguments": "{}"}}]),
        ]
    )
    chunks = list(server.client().chat("big-model", [{"role": "user", "content": "Time?"}], stream=True))

    text = "".join(c["message"]["content"] or "" for c in chunks)
    assert text == "Let me check."  # reasoning is not part of the reply
    calls = chunks[-1]["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["get_time", "search"]
    assert calls[0]["function"]["arguments"] == {"city": "Tokyo"}
    assert calls[1]["function"].get("arguments") == {}


def test_options_map_and_ollama_only_settings_stay_out():
    server = Server([_delta(content="{}")])
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    list(
        server.client().chat(
            "big-model",
            [{"role": "user", "content": "x"}],
            tools=tools,
            stream=True,
            format={"type": "object"},
            options={"temperature": 0, "num_predict": 64, "num_ctx": 8192},
            think=False,
        )
    )
    body = server.requests[0]
    assert body["temperature"] == 0 and body["max_tokens"] == 64
    assert body["tools"] == tools and body["stream"] is True
    assert not {"format", "think", "options", "num_ctx"} & set(body)


def test_a_server_refusing_tools_is_asked_again_without_them():
    refused = httpx.Response(400, json={"error": {"message": "tools are not supported"}})
    server = Server(refused, [_delta(content="Hi.")], [_delta(content="Again.")])
    client = server.client()
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]

    text = "".join(c.message.content for c in client.chat("big-model", [], tools=tools, stream=True))
    assert text == "Hi."
    assert "tools" not in server.requests[1]
    assert client.tools_refused and "tools" not in client.show("big-model").capabilities
    # From then on tools are not offered at all
    list(client.chat("big-model", [], tools=tools, stream=True))
    assert "tools" not in server.requests[2]


def test_server_errors_raise_as_ollama_does():
    server = Server(httpx.Response(404, json={"error": {"message": "model 'x' not found"}}))
    with pytest.raises(ollama.ResponseError) as raised:
        list(server.client().chat("x", [], stream=True))
    assert raised.value.status_code == 404
    assert "model 'x' not found" in raised.value.error


def test_models_and_what_the_server_tells_of_them():
    server = Server(
        models=[
            {"id": "a", "max_model_len": 32768},
            {"id": "b", "context_length": 8192, "capabilities": ["vision"]},
            {"id": "c"},
        ]
    )
    client = server.client()
    assert [m.model for m in client.list().models] == ["a", "b", "c"]
    assert client.show("a").modelinfo == {"context_length": 32768}
    assert "vision" in client.show("b").capabilities
    shown = client.show("c")
    assert shown.modelinfo == {} and shown.capabilities == ["completion", "tools"]
    assert shown.details.family == ""
    with pytest.raises(ollama.ResponseError):
        client.show("missing")


def test_a_key_is_sent_when_set(monkeypatch):
    monkeypatch.setenv(llm_client.API_KEY_ENV, "sk-test")
    server = Server()
    server.client().list()
    assert server.headers[0]["authorization"] == "Bearer sk-test"


def test_no_key_is_sent_by_default(monkeypatch):
    monkeypatch.delenv(llm_client.API_KEY_ENV, raising=False)
    server = Server()
    server.client().list()
    assert "authorization" not in server.headers[0]


class Times(Plugin):
    id = "times"
    name = "Times"

    def tools(self, context):
        return [
            Tool(
                name="get_time",
                description="The time in a city.",
                run=lambda city="", **kw: {"success": True, "result": f"14:00 in {city}"},
                parameters={"city": {"type": "string"}},
            )
        ]


def test_a_conversation_takes_a_tool_round_on_such_a_server(monkeypatch):
    server = Server(
        [_delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "get_time", "arguments": '{"city": "Oslo"}'}}])],
        [_delta(content="It is two in Oslo.")],
        models=[{"id": "big-model", "context_length": 40000}],
    )
    monkeypatch.setattr(
        llm, "OpenAIClient", lambda host, **kw: OpenAIClient(host, transport=httpx.MockTransport(server))
    )
    tools = Tools([Times()])
    cm = ConversationManager("big-model", host=BASE, tools=tools)
    assert cm.max_context_tokens == int(40000 * 0.8)

    assert "".join(cm.chat_stream("Time in Oslo?")) == "It is two in Oslo."
    second = server.requests[1]["messages"]
    call_id = next(m for m in second if m.get("tool_calls"))["tool_calls"][0]["id"]
    assert {"role": "tool", "tool_call_id": call_id, "content": "14:00 in Oslo"} in second


def test_filing_works_on_such_a_server_and_skips_inline_reasoning(monkeypatch):
    reply = '<think>Maybe {"title": "wrong"}?</think>{"title": "Sourdough"}'
    server = Server([_delta(content=reply)])
    monkeypatch.setattr(
        memory, "OpenAIClient", lambda host, **kw: OpenAIClient(host, transport=httpx.MockTransport(server))
    )
    chat = memory.chat_json(lambda: BASE, lambda: "big-model")
    schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    assert chat([{"role": "user", "content": "Title?"}], schema) == {"title": "Sourdough"}


def test_filing_on_a_server_that_refuses_the_reasoning_switch_asks_again_without_it(monkeypatch):
    unknown = "Unrecognized request argument supplied: chat_template_kwargs"
    refused = httpx.Response(400, json={"error": {"message": unknown}})
    server = Server(refused, [_delta(content='{"title": "Sourdough"}')])
    transport = httpx.MockTransport(server)
    monkeypatch.setattr(
        memory, "OpenAIClient", lambda host, **kw: OpenAIClient(host, transport=transport)
    )
    chat = memory.chat_json(lambda: BASE, lambda: "big-model")
    schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    assert chat([{"role": "user", "content": "Title?"}], schema) == {"title": "Sourdough"}
    assert server.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in server.requests[1]

"""Iris on an MTPLX server, against a stand-in that answers as MTPLX's own
server does (mtplx/server/openai.py):

- Every route wants the key MTPLX was started with, if any, as a bearer
  token or X-API-Key, and answers 401 without it; a browser page from
  another origin gets 403 before that.
- /v1/models lists the one model it has loaded, as its own (owned_by
  "mtplx"), with its context window, supports_vision and
  modalities {"input": [...], "output": [...]}.
- A chat is answered by the loaded model, whatever model it names. Fields it
  does not know are let through. Reasoning comes as reasoning_content, unless
  switched off by enable_thinking or chat_template_kwargs.enable_thinking;
  reasoning_effort only sets how much ("none" and "off" mean low).
- Pictures come as image_url parts with base64 data URLs; a model without
  sight refuses them with 400.
- The stream says keep-alive before the first token and ends with [DONE].

The stand-in listens on this computer, so the requests Iris makes go over
HTTP as they would to MTPLX.
"""

import base64
import http.server
import io
import json
import os
import re
import stat
import threading
import time
from urllib.parse import urlsplit

import httpx
import pytest
from PIL import Image

from iris import llm_client, memory, servers
from iris.attachments import Attachment
from iris.controller import AssistantController
from iris.llm import IMAGES_UNSEEN, ConversationManager
from iris.llm_client import OpenAIClient, resolve_host, set_server_key

REASONING_LEVELS = ("low", "medium", "high", "xhigh")
# What MTPLX takes for no reasoning in reasoning_effort, and answers with low
OFF_ISH = ("minimal", "none", "off", "disable", "disabled")


class Mtplx:
    """A stand-in for an MTPLX server: what it was asked is kept."""

    def __init__(self, loaded="loaded-model", key=None, vision=False, context=32768):
        self.loaded = loaded
        self.key = key
        self.vision = vision
        self.context = context
        self.reply = "Hello from the loaded model."
        # (method, path, headers) of every request, and the chat bodies
        self.seen: list[tuple[str, str, dict]] = []
        self.chats: list[dict] = []
        stand_in = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                stand_in.answer(self, "GET")

            def do_POST(self):
                stand_in.answer(self, "POST")

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # ------------------------------------------------------------ answers

    @staticmethod
    def send(handler, status: int, body: dict, headers: dict | None = None):
        data = json.dumps(body).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            handler.send_header(name, value)
        handler.end_headers()
        handler.wfile.write(data)

    @classmethod
    def error(cls, handler, status: int, message: str, kind: str, headers=None):
        body = {"error": {"message": message, "type": kind, "code": str(status), "param": None}}
        cls.send(handler, status, body, headers)

    def answer(self, handler, method: str):
        path = urlsplit(handler.path).path
        headers = dict(handler.headers.items())
        self.seen.append((method, path, headers))
        length = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(length) if length else b""
        origin = handler.headers.get("Origin")
        if origin is not None and origin != f"http://{handler.headers.get('Host')}":
            return self.error(handler, 403, "cross-origin request refused", "origin_error")
        if self.key:
            scheme, _, token = (handler.headers.get("Authorization") or "").partition(" ")
            given = token if scheme.lower() == "bearer" and token else None
            if (given or handler.headers.get("X-API-Key")) != self.key:
                return self.error(
                    handler, 401, "missing or invalid API key", "authentication_error",
                    {"WWW-Authenticate": "Bearer"},
                )
        if method == "GET" and path == "/v1/models":
            return self.send(handler, 200, self.models())
        if method == "POST" and path == "/v1/chat/completions":
            return self.chat(handler, json.loads(body or b"{}"))
        return self.error(handler, 404, "Not Found", "invalid_request_error")

    def models(self) -> dict:
        entry = {
            "id": self.loaded,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "mtplx",
            "capability": "chat",
            "supports_vision": self.vision,
            "modalities": {
                "input": ["text", "image"] if self.vision else ["text"],
                "output": ["text"],
            },
            "context_length": self.context,
            "max_context_length": self.context,
            "max_model_len": self.context,
        }
        return {"object": "list", "data": [entry]}

    def chat(self, handler, request: dict):
        self.chats.append(request)
        images = 0
        for message in request.get("messages") or []:
            content = message.get("content")
            for part in content if isinstance(content, list) else []:
                if part.get("type") == "image_url" or "image_url" in part:
                    url = (part.get("image_url") or {}).get("url") or ""
                    header, _, payload = url.partition(",")
                    if not url.startswith("data:") or ";base64" not in header or not payload:
                        return self.error(
                            handler, 400, "image data URL must be base64 encoded",
                            "invalid_request_error",
                        )
                    base64.b64decode(payload)
                    images += 1
        if images and not self.vision:
            return self.error(
                handler, 400,
                "this model has no vision tower; image content is not supported for it",
                "invalid_request_error",
            )
        effort = request.get("reasoning_effort")
        if effort is not None and str(effort).lower() not in (*REASONING_LEVELS, *OFF_ISH, "auto"):
            return self.error(
                handler, 400, "reasoning_effort must be one of: auto, low, medium, high, xhigh",
                "invalid_request_error",
            )
        thinking = request.get("enable_thinking")
        if thinking is None:
            thinking = (request.get("chat_template_kwargs") or {}).get("enable_thinking")
        thinking = True if not isinstance(thinking, bool) else thinking
        reply = self.reply + (f" I see {images} picture(s)." if images else "")
        chunk = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": self.loaded}

        def delta(**fields):
            return {**chunk, "choices": [{"index": 0, "delta": fields, "finish_reason": None}]}

        events = [delta(role="assistant")]
        if thinking:
            events += [delta(reasoning_content="Let me think about it.")]
        events += [delta(content=piece) for piece in re.findall(r"\S+\s*", reply)]
        events += [{**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
        if not request.get("stream"):
            message = {"role": "assistant", "content": reply}
            return self.send(handler, 200, {**chunk, "choices": [{"index": 0, "message": message}]})
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(b": keep-alive\n\n")
        for event in events:
            handler.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
        handler.close_connection = True


@pytest.fixture
def mtplx():
    made = []

    def make(**kwargs) -> Mtplx:
        stand_in = Mtplx(**kwargs)
        made.append(stand_in)
        return stand_in

    yield make
    for stand_in in made:
        stand_in.close()


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """Nothing resolved before, and no key from the environment."""
    monkeypatch.setattr(llm_client, "_resolved", {})
    monkeypatch.delenv(llm_client.API_KEY_ENV, raising=False)


def _picture() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 120, 40)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _bearer(headers: dict) -> str | None:
    return {k.lower(): v for k, v in headers.items()}.get("authorization")


# ------------------------------------------------------------------- the key


def test_a_server_that_wants_a_key_is_used_with_the_one_kept_for_it(mtplx):
    server = mtplx(key="sk-ada")
    set_server_key(server.url, "sk-ada")

    # Typed without /v1, as people copy it
    host = resolve_host(server.url)
    assert host == f"{server.url}/v1"
    assert servers.reachable(server.url)
    assert AssistantController.probe_server(server.url)["models"][0]["name"] == "loaded-model"
    stream = OpenAIClient(host).chat(
        "loaded-model", [{"role": "user", "content": "Hi"}], stream=True
    )
    reply = "".join(chunk.message.content or "" for chunk in stream)
    assert reply == "Hello from the loaded model."
    assert all(_bearer(headers) == "Bearer sk-ada" for _, _, headers in server.seen)


def test_without_its_key_the_server_says_so_and_is_asked_again_once_it_has_one(mtplx):
    server = mtplx(key="sk-ada")

    # Taken for an OpenAI-compatible server that wants a key, not for Ollama
    host = resolve_host(server.url)
    assert host == f"{server.url}/v1"
    refused = AssistantController.probe_server(server.url)
    assert not refused["ok"] and "missing or invalid API key" in refused["error"]
    # A server that wants a key is there
    assert servers.reachable(server.url)

    set_server_key(f"{server.url}/v1", "sk-ada")
    assert AssistantController.probe_server(server.url)["ok"]


def test_a_wrong_key_is_refused(mtplx):
    server = mtplx(key="sk-ada")
    set_server_key(server.url, "sk-wrong")
    result = AssistantController.probe_server(f"{server.url}/v1")
    assert not result["ok"] and "missing or invalid API key" in result["error"]


def test_the_key_goes_to_its_server_alone(mtplx):
    keyed, other = mtplx(key="sk-ada"), mtplx()
    set_server_key(keyed.url, "sk-ada")
    OpenAIClient(f"{other.url}/v1").list()
    assert servers.reachable(other.url)
    assert not any(_bearer(headers) for _, _, headers in other.seen)


def test_the_environments_key_still_goes_to_servers_without_their_own(mtplx, monkeypatch):
    keyed, other = mtplx(key="sk-ada"), mtplx(key="sk-shared")
    monkeypatch.setenv(llm_client.API_KEY_ENV, "sk-shared")
    set_server_key(keyed.url, "sk-ada")
    OpenAIClient(f"{keyed.url}/v1").list()
    OpenAIClient(f"{other.url}/v1").list()
    assert _bearer(keyed.seen[-1][2]) == "Bearer sk-ada"
    assert _bearer(other.seen[-1][2]) == "Bearer sk-shared"


def test_a_key_is_forgotten_with_an_empty_one(mtplx):
    server = mtplx(key="sk-ada")
    set_server_key(server.url, "sk-ada")
    set_server_key(f"{server.url}/v1/", "")
    assert not AssistantController.probe_server(f"{server.url}/v1")["ok"]
    assert json.loads(llm_client.KEYS_PATH.read_text()) == {}


def test_keys_are_kept_for_this_computers_user_alone(mtplx):
    set_server_key("box.example:8000", "sk-ada")
    mode = stat.S_IMODE(os.stat(llm_client.KEYS_PATH).st_mode)
    assert mode == 0o600
    # The address as typed, without a scheme, and as kept name one server
    assert llm_client.auth_headers("http://box.example:8000/v1") == {
        "Authorization": "Bearer sk-ada"
    }


def test_the_key_is_not_told(mtplx):
    server = mtplx(key="sk-ada")
    assert AssistantController.set_key(server.url, "sk-ada") == {"ok": True}
    told = json.dumps(
        [
            AssistantController.probe_server(server.url),
            AssistantController.check_servers([server.url]),
            AssistantController.probe_server("http://127.0.0.1:9"),
        ]
    )
    assert "sk-ada" not in told


def test_a_server_that_wants_a_key_is_found_on_this_computer_once_it_has_one(mtplx):
    server = mtplx(key="sk-ada")
    url = f"{server.url}/v1"
    assert servers.discover((url,)) == []
    set_server_key(server.url, "sk-ada")
    assert servers.discover((url,)) == [url]


# --------------------------------------------------------- the loaded model


def test_the_loaded_model_holds_the_conversation(mtplx):
    server = mtplx(loaded="loaded-model", context=40000)
    cm = ConversationManager("model-of-another-server", host=f"{server.url}/v1")
    assert cm.model == "loaded-model"
    assert cm.max_context_tokens == int(40000 * 0.8)
    # Nothing else can be chosen: MTPLX would answer with it all the same
    cm.set_model("not-loaded")
    assert cm.model == "loaded-model"
    assert "".join(cm.chat_stream("Hi")) == "Hello from the loaded model."
    assert server.chats[-1]["model"] == "loaded-model"
    assert [m["name"] for m in ConversationManager.list_available_models(f"{server.url}/v1")] == [
        "loaded-model"
    ]


def test_a_model_loaded_meanwhile_is_followed_when_the_server_is_set_again(mtplx):
    server = mtplx(loaded="first-model")
    host = f"{server.url}/v1"
    cm = ConversationManager("first-model", host=host)
    server.loaded = "second-model"
    cm.set_host(host)
    assert cm.model == "second-model"


def test_other_servers_answer_with_the_model_asked_for(mtplx):
    listing = {"object": "list", "data": [{"id": "a", "owned_by": "vllm"}, {"id": "b"}]}

    def answer(request):
        return httpx.Response(200, json=listing)

    client = OpenAIClient("https://box.example:8000/v1", transport=httpx.MockTransport(answer))
    assert client.loaded() is None


# ------------------------------------------------------------------ pictures


def test_a_loaded_model_that_sees_gets_the_pictures_as_image_parts(mtplx):
    server = mtplx(vision=True)
    host = f"{server.url}/v1"
    listed = ConversationManager.list_available_models(host)
    assert listed == [{"name": "loaded-model", "tools": True, "vision": True}]
    cm = ConversationManager("loaded-model", host=host)
    picture = Attachment(name="photo.jpg", kind="image", mime="image/jpeg", image=_picture())

    reply = "".join(cm.chat_stream("What is this?", attachments=[picture]))
    assert reply == "Hello from the loaded model. I see 1 picture(s)."
    parts = server.chats[-1]["messages"][-1]["content"]
    image = next(p for p in parts if p["type"] == "image_url")
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_a_loaded_model_that_cannot_see_is_told_so_and_sent_no_pictures(mtplx):
    server = mtplx(vision=False)
    cm = ConversationManager("loaded-model", host=f"{server.url}/v1")
    picture = Attachment(name="photo.jpg", kind="image", mime="image/jpeg", image=_picture())

    assert "".join(cm.chat_stream("What is this?", attachments=[picture]))
    last = server.chats[-1]["messages"][-1]
    assert isinstance(last["content"], str) and IMAGES_UNSEEN in last["content"]


# ----------------------------------------------------------------- reasoning


def test_filing_asks_for_no_reasoning_as_mtplx_reads_it(mtplx, monkeypatch):
    server = mtplx()
    server.reply = '{"title": "Sourdough"}'
    chat = memory.chat_json(lambda: f"{server.url}/v1", lambda: "loaded-model")
    schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}

    assert chat([{"role": "user", "content": "Title?"}], schema) == {"title": "Sourdough"}
    asked = server.chats[-1]
    assert asked["chat_template_kwargs"] == {"enable_thinking": False}
    # reasoning_effort cannot switch MTPLX's reasoning off: it is not sent
    assert "reasoning_effort" not in asked
    assert len(server.chats) == 1


def test_a_conversations_reasoning_is_not_part_of_the_reply(mtplx):
    server = mtplx()
    cm = ConversationManager("loaded-model", host=f"{server.url}/v1")
    assert "".join(cm.chat_stream("Hi")) == "Hello from the loaded model."
    assert "chat_template_kwargs" not in server.chats[-1]


# ------------------------------------------------------- the browser's origin


def test_iris_sends_no_origin_so_mtplxs_browser_check_lets_it_in(mtplx):
    server = mtplx()
    cm = ConversationManager("loaded-model", host=f"{server.url}/v1")
    "".join(cm.chat_stream("Hi"))
    servers.reachable(server.url)
    assert server.seen and not any(
        "origin" in {k.lower() for k in headers} for _, _, headers in server.seen
    )
    # As a page in a browser from elsewhere would be
    refused = httpx.get(f"{server.url}/v1/models", headers={"Origin": "https://elsewhere.example"})
    assert refused.status_code == 403

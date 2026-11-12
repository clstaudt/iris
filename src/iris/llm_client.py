"""Clients for the servers that run Iris's language models.

Ollama is talked to through its own API, which also tells each model's
context length and capabilities. Any other server is talked to through the
OpenAI-compatible API most servers offer (vLLM, llama.cpp, LM Studio, MLX
servers, LiteLLM, OpenAI itself). OpenAIClient offers the part of
ollama.Client that Iris uses and answers with ollama's own types, so the rest
of Iris does not need to know which kind of server it talks to.

A server URL is resolved once, when it is set (resolve_host): an Ollama
server is kept as its root URL, any other as its OpenAI base URL, which ends
in /v1. The kept URL alone then says which client to use (is_openai).
"""

import json
import os
from typing import Iterator, Optional

import httpx
import ollama

# Fallback shown when neither a server URL nor OLLAMA_HOST is configured
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# A key for servers that need one, sent as a bearer token
API_KEY_ENV = "IRIS_LLM_API_KEY"
# How long resolve_host waits for a server to say what it is
PROBE_TIMEOUT = 5.0
# Fields OpenAI-compatible servers use for a model's context length in /models
CONTEXT_KEYS = ("context_length", "max_model_len", "max_context_length", "context_window")
# Paths of single calls that people copy along with a server URL
CALL_PATHS = ("/chat/completions", "/models", "/api/tags", "/api/chat", "/api")

# Server URLs already resolved: {URL as given (normalized): URL to keep}
_resolved: dict[str, str] = {}


def normalize_host(url: Optional[str]) -> Optional[str]:
    """Clean up a server URL as people copy it around: spaces, trailing
    slashes and the path of a single call go. Returns None for an empty
    value, meaning "use the default"."""
    url = (url or "").strip().rstrip("/")
    for path in CALL_PATHS:
        if url.endswith(path):
            url = url[: -len(path)].rstrip("/")
    return url or None


def is_openai(host: Optional[str]) -> bool:
    """Whether a kept server URL is an OpenAI-compatible server (not Ollama)."""
    return bool(host) and host.endswith("/v1")


def effective_host(host: Optional[str]) -> str:
    """The server URL actually in use, for display."""
    return normalize_host(host) or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL


def _auth_headers() -> dict:
    key = os.environ.get(API_KEY_ENV)
    return {"Authorization": f"Bearer {key}"} if key else {}


def _answers(url: str, key: str) -> bool:
    """Whether a GET of the URL returns JSON holding this key. Raises
    httpx.TransportError when nothing answers at all."""
    response = httpx.get(
        url, timeout=PROBE_TIMEOUT, headers=_auth_headers(), follow_redirects=True
    )
    if response.status_code != 200:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and key in body


def resolve_host(url: Optional[str]) -> Optional[str]:
    """The URL to keep for a server: its root for Ollama, its OpenAI base URL
    (ending in /v1) for any other.

    The server is asked what it is: Ollama answers /api/version. A URL given
    with /v1 is otherwise taken as OpenAI-compatible; one without is
    OpenAI-compatible if /v1/models answers. A server that does not answer
    keeps the URL as given, and is asked again next time.
    """
    url = normalize_host(url)
    if not url:
        return None
    if url in _resolved:
        return _resolved[url]
    root = url[: -len("/v1")] if url.endswith("/v1") else url
    try:
        if _answers(f"{root}/api/version", "version"):
            resolved = root
        elif url.endswith("/v1"):
            resolved = url
        elif _answers(f"{url}/v1/models", "data"):
            resolved = f"{url}/v1"
        else:
            resolved = url
    except httpx.TransportError:
        return url
    _resolved[url] = resolved
    return resolved


def _image_url(image) -> str:
    """An image as Ollama takes it (base64, or bytes) as a data URL."""
    if isinstance(image, bytes):
        image = image.decode("ascii")
    mime = "image/png" if image.startswith("iVBOR") else "image/jpeg"
    return f"data:{mime};base64,{image}"


def _field(obj, key: str):
    """A field of a dict or of one of ollama's response objects."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _with_images(text: str, images: list) -> list[dict]:
    parts = [{"type": "text", "text": text}] if text else []
    return parts + [{"type": "image_url", "image_url": {"url": _image_url(i)}} for i in images]


def openai_messages(messages: list) -> list[dict]:
    """Messages as Iris builds them for Ollama, in the OpenAI format.

    - Images become image_url parts with data URLs.
    - Tool calls get ids, and their arguments become JSON text. A tool
      result answers the call of the same position, as Iris adds the results
      in the order of the calls.
    - A tool result cannot carry images in that format: they follow the
      results as a user message.
    """
    out: list[dict] = []
    call_ids: list[str] = []
    pending_images: list = []

    def flush_images():
        if pending_images:
            out.append({"role": "user", "content": _with_images("", pending_images)})
            pending_images.clear()

    for message in messages:
        role = _field(message, "role")
        content = _field(message, "content") or ""
        images = list(_field(message, "images") or [])
        tool_calls = _field(message, "tool_calls") or []
        if role == "tool":
            call_id = call_ids.pop(0) if call_ids else f"call_{len(out)}"
            out.append({"role": "tool", "tool_call_id": call_id, "content": content})
            pending_images += images
            continue
        flush_images()
        if role == "assistant" and tool_calls:
            calls = []
            for n, call in enumerate(tool_calls):
                function = _field(call, "function") or {}
                calls.append(
                    {
                        "id": f"call_{len(out)}_{n}",
                        "type": "function",
                        "function": {
                            "name": _field(function, "name") or "",
                            "arguments": json.dumps(dict(_field(function, "arguments") or {})),
                        },
                    }
                )
            call_ids = [call["id"] for call in calls]
            out.append({"role": "assistant", "content": content, "tool_calls": calls})
            continue
        out.append({"role": role, "content": _with_images(content, images) if images else content})
    flush_images()
    return out


def _tool_calls(calls: list[dict]) -> list:
    """Tool calls from an OpenAI reply as ollama's ToolCall objects."""
    result = []
    for call in calls:
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                arguments = {}
        result.append(
            ollama.Message.ToolCall(
                function=ollama.Message.ToolCall.Function(
                    name=function.get("name") or "", arguments=arguments
                )
            )
        )
    return result


def _reply(model: str, content: str = "", tool_calls: Optional[list] = None, done=False):
    return ollama.ChatResponse(
        model=model,
        done=done,
        message=ollama.Message(role="assistant", content=content, tool_calls=tool_calls or None),
    )


class OpenAIClient:
    """The part of ollama.Client Iris uses (list, show, chat), on a server
    with the OpenAI-compatible API. Answers are ollama's own types."""

    def __init__(
        self,
        host: str,
        timeout: Optional[float] = None,
        transport: Optional[httpx.BaseTransport] = None,
        **kwargs,
    ):
        """
        Args:
            host: The server's OpenAI base URL, ending in /v1
            timeout: Seconds to wait for the server; None waits as long as
                a reply takes, as ollama.Client does
            transport: How requests are sent (tests answer them themselves)
        """
        self.base = normalize_host(host)
        self._client = httpx.Client(timeout=timeout, headers=_auth_headers(), transport=transport)
        self._models: Optional[dict[str, dict]] = None
        # Set once the server has refused a request because of its tools
        self.tools_refused = False

    @staticmethod
    def _raise(response: httpx.Response):
        """Raise the server's error as ollama does (ollama.ResponseError)."""
        if response.is_success:
            return
        response.read()
        try:
            body = response.json()
            error = body.get("error", body) if isinstance(body, dict) else body
            if isinstance(error, dict):
                error = error.get("message", error)
        except ValueError:
            error = response.text
        raise ollama.ResponseError(str(error), response.status_code)

    def list(self) -> ollama.ListResponse:
        response = self._client.get(f"{self.base}/models")
        self._raise(response)
        entries = response.json().get("data") or []
        self._models = {e["id"]: e for e in entries if isinstance(e, dict) and e.get("id")}
        return ollama.ListResponse(
            models=[ollama.ListResponse.Model(model=name) for name in self._models]
        )

    def show(self, model: str) -> ollama.ShowResponse:
        """What the server tells of a model: its context length when it says,
        that it holds conversations and, unless refused, calls tools, and
        that it sees images when it says so."""
        if self._models is None or model not in self._models:
            self.list()
        entry = self._models.get(model)
        if entry is None:
            raise ollama.ResponseError(f"model '{model}' not found", 404)
        modelinfo = {}
        for key in CONTEXT_KEYS:
            if isinstance(entry.get(key), int) and entry[key] > 0:
                modelinfo["context_length"] = entry[key]
                break
        capabilities = ["completion"] if self.tools_refused else ["completion", "tools"]
        declared = list(entry.get("capabilities") or []) + list(
            entry.get("input_modalities") or entry.get("modalities") or []
        )
        if "vision" in declared or "image" in declared:
            capabilities.append("vision")
        return ollama.ShowResponse(
            model_info=modelinfo, details={"family": ""}, capabilities=capabilities
        )

    def chat(
        self,
        model: str,
        messages: Optional[list] = None,
        tools: Optional[list] = None,
        stream: bool = False,
        format=None,
        options: Optional[dict] = None,
        think=None,
        keep_alive=None,
        **kwargs,
    ):
        """A chat completion, streamed as ollama streams one when asked.

        `format`, `think` and `keep_alive` are Ollama's: OpenAI-compatible
        servers differ too much in how they take them, so they are left out.
        Reasoning a server streams apart from the reply is left out too.
        """
        payload = {"model": model, "messages": openai_messages(messages or []), "stream": stream}
        if tools and not self.tools_refused:
            payload["tools"] = [dict(tool) for tool in tools]
        for ours, theirs in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("num_predict", "max_tokens"),
            ("seed", "seed"),
            ("stop", "stop"),
        ):
            if options and options.get(ours) is not None:
                payload[theirs] = options[ours]
        if stream:
            return self._stream(payload)
        response = self._send(payload, stream=False)
        message = response.json()["choices"][0]["message"]
        return _reply(
            model, message.get("content") or "", _tool_calls(message.get("tool_calls") or []), True
        )

    def _send(self, payload: dict, stream: bool) -> httpx.Response:
        """POST a chat request. A server that refuses it with tools is asked
        again without them, and from then on is not offered any."""
        url = f"{self.base}/chat/completions"
        response = self._client.send(self._client.build_request("POST", url, json=payload), stream=stream)
        if not response.is_success and "tools" in payload and 400 <= response.status_code < 500:
            response.close()
            retry = {k: v for k, v in payload.items() if k != "tools"}
            response = self._client.send(
                self._client.build_request("POST", url, json=retry), stream=stream
            )
            if response.is_success:
                self.tools_refused = True
        self._raise(response)
        return response

    def _stream(self, payload: dict) -> Iterator[ollama.ChatResponse]:
        response = self._send(payload, stream=True)
        try:
            yield from self._chunks(response, payload["model"])
        finally:
            # Ends the request, so the server stops generating
            response.close()

    @staticmethod
    def _chunks(response: httpx.Response, model: str) -> Iterator[ollama.ChatResponse]:
        """The reply's text as it streams in, then its tool calls (pieced
        together from the streamed fragments) in a last chunk."""
        calls: dict[int, dict] = {}
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if event.get("error"):
                error = event["error"]
                raise ollama.ResponseError(str(_field(error, "message") or error), 500)
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    yield _reply(model, delta["content"])
                for fragment in delta.get("tool_calls") or []:
                    call = calls.setdefault(fragment.get("index", len(calls)), {"arguments": ""})
                    function = fragment.get("function") or {}
                    if function.get("name") and not call.get("name"):
                        call["name"] = function["name"]
                    arguments = function.get("arguments")
                    if isinstance(arguments, dict):
                        call["arguments"] = json.dumps(arguments)
                    elif arguments:
                        call["arguments"] += arguments
        if calls:
            ordered = [calls[i] for i in sorted(calls)]
            yield _reply(
                model,
                tool_calls=_tool_calls([{"function": call} for call in ordered]),
                done=True,
            )

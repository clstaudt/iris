"""A scripted language model behind Ollama's API, for testing the voice.

It answers at once and always the same way, so a test measures Iris rather
than a model: a short reply to a question, and a long one when asked for a
story or a long answer. How long it takes to start and how fast it writes
are set, so a test can stand in for a fast or a slow model.

Rules script more than that. A rule matches the user's last message and
says what to answer: a reply, or a tool call (a web search, reading a page)
with what to say before it and what to make of the tool's result. So Iris's
tool loop is tested as a model that supports tools would drive it: the
search is called, its result comes back as a "tool" message, and the answer
is drawn from that result.

Every request is recorded (StubRequest): what it carried, which tools it
offered, when the tool call and the first byte went out, and whether the
client hung up before the reply was complete.

A rule can also stand for a model server in trouble: one that stalls part
way through a reply (stall_after), as a network that drops out does, or a
small model that writes its tool call into the reply as text (as_text),
streamed in pieces that split the markup (chunk). The model itself can be
one without tool support: its capabilities say so, or it refuses requests
with tools as Ollama does (refuse_tools).
"""

import json
import re
import select
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional, Union

MODEL = "voicetest:latest"

SHORT = "Paris is the capital of France. It lies on the Seine."
LONG = " ".join(
    f"This is sentence number {n} of a long answer, which goes on for a while."
    for n in range(1, 9)
)
# A request for a long reply, so the voice has time to be interrupted
LONG_REQUEST = re.compile(r"\b(story|long|everything|in detail)\b", re.IGNORECASE)
# What a tool's result is answered with when a rule says nothing else
AFTER_TOOL = "Here is what I found. {first}"
# Iris's notes ahead of the user's words: the language to answer in, a
# persona that changed, no more tools ("(Note: ...)" and a blank line each)
NOTES = re.compile(r"^(?:\(Note: .*?\)\n\n)+", re.DOTALL)
NOTE = re.compile(r"\(Note: (.*?)\)\n\n", re.DOTALL)
# A search result as iris.plugins.web formats it
RESULT_LINE = re.compile(r"^- (?P<title>.*?): (?P<body>.*) \((?P<href>[^()]*)\)\s*$", re.MULTILINE)
# A request to summarize the conversation (iris.llm.summarize_old_messages)
SUMMARY_PROMPT = "Summarize this conversation"
# How often a waiting request looks whether the client is still there
POLL = 0.02
# What the model can do, as /api/show reports it
CAPABILITIES = ("completion", "tools")


# Reading a conversation ------------------------------------------------------


def user_text(message: dict) -> str:
    """A user message's text without Iris's notes in front of it."""
    return NOTES.sub("", message.get("content") or "")


def notes_of(message: dict) -> list[str]:
    """The notes Iris put in front of a user message."""
    found = NOTES.match(message.get("content") or "")
    return NOTE.findall(found.group(0)) if found else []


def last_user(messages: list[dict]) -> str:
    """The user's last message, without notes; "" if there is none."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return user_text(message)
    return ""


def user_messages(messages: list[dict]) -> list[str]:
    """Every user message, without notes, oldest first."""
    return [user_text(m) for m in messages if m.get("role") == "user"]


def tool_rounds(messages: list[dict]) -> int:
    """How many tool results have come back since the user's last message."""
    rounds = 0
    for message in reversed(messages):
        if message.get("role") == "user":
            break
        rounds += message.get("role") == "tool"
    return rounds


def last_tool(messages: list[dict]) -> tuple[str, str]:
    """(name, result) of the last tool that ran; ("", "") if none did."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") != "tool":
            continue
        content = messages[index].get("content") or ""
        for before in reversed(messages[:index]):
            calls = before.get("tool_calls") or []
            if before.get("role") == "assistant" and calls:
                return (calls[-1].get("function") or {}).get("name", ""), content
        return "", content
    return "", ""


def results(content: str) -> list[dict]:
    """The search results in a tool's content: [{title, body, href}]."""
    return [m.groupdict() for m in RESULT_LINE.finditer(content or "")]


def reply_to(messages: list[dict]) -> str:
    """The reply to the conversation: long when the last user message asks
    for it, short otherwise."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return LONG if LONG_REQUEST.search(message.get("content") or "") else SHORT
    return SHORT


# Rules -------------------------------------------------------------------------


@dataclass
class Call:
    """A tool call the model makes, and what it says before it."""

    name: str
    arguments: dict
    lead_in: str = ""


def search(query: Union[str, Callable[[list], str], None] = None) -> tuple:
    """The tool of a rule that searches: for query(messages), a fixed query,
    or the user's last message."""

    def arguments(messages):
        if callable(query):
            return {"query": query(messages)}
        return {"query": query if query is not None else last_user(messages)}

    return ("search", arguments)


def browse(url: Union[str, Callable[[list], str]]) -> tuple:
    """The tool of a rule that reads a page."""
    return ("browse", lambda messages: {"url": url(messages) if callable(url) else url})


def fill(template: str, messages: list[dict]) -> str:
    """A reply template filled from the last tool's result: {result} (all of
    it), {first} and {title} (the first search result's body and title) and
    {href} (its address)."""
    _, content = last_tool(messages)
    found = results(content)
    first = found[0] if found else {"title": "", "body": content.strip(), "href": ""}
    return template.format(result=content, first=first["body"], title=first["title"], href=first["href"])


@dataclass
class Rule:
    """What the model answers to a user message that matches.

    match is a regular expression searched in the user's last message (case
    ignored, Iris's notes left out), or callable(messages) -> bool. A rule
    with a tool calls it when the request offers that tool: it streams
    lead_in first (word by word, with no space added after it, as a model
    streams it) and then the call. The request that brings the tool's
    result back is answered with after_tool.
    """

    match: Union[str, Callable[[list], bool]]
    # The reply: text, or callable(messages) -> text; None for SHORT or LONG
    reply: Union[str, Callable[[list], str], None] = None
    # Said before the tool call: text, or callable(messages) -> text
    lead_in: Union[str, Callable[[list], str]] = ""
    # (name, callable(messages) -> arguments), as search() and browse() make
    tool: Optional[tuple] = None
    # The answer to a tool's result: a template (see fill()), or
    # callable(messages) -> text or a Call (another tool)
    after_tool: Union[str, Callable[[list], Union[str, Call]], None] = None
    # The reply when the rule would call a tool but none is offered
    no_tools: Optional[str] = None
    # Seconds before the first word, instead of the model's
    first_token: Optional[float] = None
    # Answer this many tool results with another search, while tools are offered
    loop: int = 0
    # Words of the reply sent before it stalls: nothing more comes, and the
    # connection is held open until the client gives up
    stall_after: Optional[int] = None
    # The call written into the reply as text, instead of made:
    # callable(call) -> the reply
    as_text: Optional[Callable[[Call], str]] = None
    # Characters per streamed piece of the reply, instead of a word each
    chunk: Optional[int] = None

    def matches(self, messages: list[dict]) -> bool:
        if callable(self.match):
            return bool(self.match(messages))
        return re.search(self.match, last_user(messages), re.IGNORECASE) is not None

    def call(self, messages: list[dict]) -> Call:
        name, arguments = self.tool
        lead_in = self.lead_in(messages) if callable(self.lead_in) else self.lead_in
        return Call(name, dict(arguments(messages) if callable(arguments) else arguments), lead_in)


@dataclass
class Answer:
    """What the model sends back to one request."""

    text: str = ""
    call: Optional[Call] = None
    first_token: Optional[float] = None
    stall_after: Optional[int] = None
    chunk: Optional[int] = None


# The record of a request -------------------------------------------------------


@dataclass
class StubRequest:
    """A request the model was sent, and how it was answered."""

    received: float
    # "reply" (to the user), "tool_round" (a tool's result came back) or
    # "summary" (Iris summarizes the conversation)
    kind: str
    body: dict = field(repr=False)
    messages: list = field(repr=False)
    # The names of the tools offered
    tools: list
    stream: bool
    index: int = 0
    # When the tool call and the first byte of the reply went out
    tool_call_sent: Optional[float] = None
    first_byte: Optional[float] = None
    # When the client hung up before the reply was complete
    closed: Optional[float] = None
    # When the reply was complete
    finished: Optional[float] = None
    # When the reply stalled (Rule.stall_after)
    stalled: Optional[float] = None
    # Refused for its tools (StubModel.refuse_tools)
    refused: bool = False
    # What was sent: the text, and a tool call
    text: str = ""
    call: Optional[Call] = None

    @property
    def system(self) -> str:
        """The system prompt."""
        first = self.messages[0] if self.messages else {}
        return first.get("content") or "" if first.get("role") == "system" else ""

    def carries(self, text: str) -> bool:
        """Whether any message holds this text (case ignored)."""
        text = text.lower()
        return any(text in (m.get("content") or "").lower() for m in self.messages)

    def tool_messages(self) -> list[str]:
        return [m.get("content") or "" for m in self.messages if m.get("role") == "tool"]

    def users(self) -> list[str]:
        return user_messages(self.messages)


# The server --------------------------------------------------------------------


def _gone(connection) -> bool:
    """Whether the client closed the connection (a waiting request has
    nothing more to read, so a readable socket with nothing in it is closed)."""
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        if not readable:
            return False
        return connection.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


class StubModel:
    """Ollama's API on a local port, answering with scripted replies."""

    def __init__(
        self,
        rules: Optional[list[Rule]] = None,
        first_token: float = 0.3,
        per_word: float = 0.02,
        serial: bool = False,
        context_length: int = 32768,
        summary_delay: float = 0.0,
        on_request: Optional[Callable[[str, StubRequest], None]] = None,
        capabilities: tuple = CAPABILITIES,
        refuse_tools: bool = False,
    ):
        """
        Args:
            rules: Tried in order before the short and long replies
            first_token: Seconds before the reply's first word
            per_word: Seconds per further word
            serial: One request at a time, as one Ollama slot: the next
                waits until the reply before it is complete or cut off
            context_length: The context the model reports
            summary_delay: Seconds a request to summarize takes
            on_request: Called with (kind, record) as each request arrives
            capabilities: What /api/show says the model can do
            refuse_tools: Refuse a request that offers tools, as Ollama does
                for a model without tool support (400, "does not support
                tools")
        """
        self.first_token = first_token
        self.per_word = per_word
        self.context_length = context_length
        self.summary_delay = summary_delay
        self.on_request = on_request
        self.capabilities = tuple(capabilities)
        self.refuse_tools = refuse_tools
        self._default_rules = list(rules or [])
        self.rules = list(self._default_rules)
        self.serial = serial
        self._slot = threading.Semaphore(1) if serial else None
        self._lock = threading.Lock()
        # The requests the model was sent, oldest first
        self.requests: list[StubRequest] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _json(self, body: dict, status: int = 200):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self) -> dict:
                length = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(length) or b"{}")

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                if self.path == "/api/version":
                    self._json({"version": "0.0.0-voicetest"})
                elif self.path == "/api/tags":
                    self._json({"models": [stub._entry()]})
                elif self.path == "/":
                    self._json({"status": "ok"})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                body = self._body()
                if self.path == "/api/show":
                    self._json(stub._show())
                elif self.path == "/api/chat":
                    stub._chat(self, body)
                else:
                    self._json({"error": "not found"}, 404)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        # A short poll, so closing the server does not keep a test waiting
        threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self._server.shutdown()

    def set_rules(self, rules: list[Rule]):
        """Answer by these rules from now on (before SHORT and LONG)."""
        self.rules = list(rules)

    def reset_rules(self):
        """Back to the rules the model was made with."""
        self.rules = list(self._default_rules)

    # The settings a conversation may change
    SETTINGS = ("first_token", "per_word", "serial", "context_length", "summary_delay", "capabilities",
                "refuse_tools")

    def configure(self, **settings) -> dict:
        """Change settings (SETTINGS) from now on; returns what they were."""
        unknown = set(settings) - set(self.SETTINGS)
        if unknown:
            raise ValueError(f"The scripted model has no setting {', '.join(sorted(unknown))}")
        before = {name: getattr(self, name) for name in settings}
        for name, value in settings.items():
            setattr(self, name, value)
            if name == "serial":
                self._slot = threading.Semaphore(1) if value else None
        return before

    # The records ---------------------------------------------------------------

    def requests_between(self, start: float, end: float = float("inf")) -> list[StubRequest]:
        with self._lock:
            return [r for r in self.requests if start <= r.received <= end]

    def carrying(self, text: str, since: float = 0.0) -> list[StubRequest]:
        """The requests since a moment whose messages hold this text."""
        return [r for r in self.requests_between(since) if r.carries(text)]

    # What Ollama reports -------------------------------------------------------

    @staticmethod
    def _details() -> dict:
        return {
            "format": "gguf",
            "family": "voicetest",
            "families": ["voicetest"],
            "parameter_size": "1B",
            "quantization_level": "Q4_0",
        }

    def _entry(self) -> dict:
        return {
            "name": MODEL,
            "model": MODEL,
            "modified_at": "2026-01-01T00:00:00Z",
            "size": 1,
            "digest": "0" * 64,
            "details": self._details(),
        }

    def _show(self) -> dict:
        return {
            "modelfile": "",
            "parameters": "",
            "template": "",
            "details": self._details(),
            "model_info": {
                "general.architecture": "voicetest",
                "voicetest.context_length": self.context_length,
            },
            "capabilities": list(self.capabilities),
        }

    # Answering -----------------------------------------------------------------

    def answer(self, messages: list[dict], offered: list[str]) -> Answer:
        """What the model answers to these messages, with these tools offered."""
        last = messages[-1] if messages else {}
        rule = next((r for r in self.rules if r.matches(messages)), None)
        if rule is None:
            return Answer(reply_to(messages))
        if last.get("role") == "tool":
            rounds = tool_rounds(messages)
            if rule.loop and rule.tool and rule.tool[0] in offered and rounds < rule.loop:
                call = rule.call(messages)
                if "query" in call.arguments:
                    call.arguments["query"] = f"{call.arguments['query']} ({rounds + 1})"
                return Answer(call=Call(call.name, call.arguments), first_token=rule.first_token)
            after = rule.after_tool if rule.after_tool is not None else AFTER_TOOL
            out = after(messages) if callable(after) else fill(after, messages)
            if isinstance(out, Call):
                if out.name in offered:
                    return Answer(call=out, first_token=rule.first_token)
                return Answer(rule.no_tools or SHORT, first_token=rule.first_token)
            return Answer(out, first_token=rule.first_token)
        if rule.tool:
            if rule.tool[0] in offered and rule.as_text is not None:
                return Answer(rule.as_text(rule.call(messages)), first_token=rule.first_token,
                              chunk=rule.chunk)
            if rule.tool[0] in offered:
                return Answer(call=rule.call(messages), first_token=rule.first_token)
            return Answer(rule.no_tools if rule.no_tools is not None else SHORT, first_token=rule.first_token)
        if rule.reply is not None:
            text = rule.reply(messages) if callable(rule.reply) else rule.reply
        else:
            text = reply_to(messages)
        return Answer(text, first_token=rule.first_token, stall_after=rule.stall_after,
                      chunk=rule.chunk)

    def _record(self, body: dict) -> StubRequest:
        messages = body.get("messages") or []
        offered = [((t or {}).get("function") or {}).get("name", "") for t in body.get("tools") or []]
        first = messages[0] if messages else {}
        if first.get("role") == "system" and (first.get("content") or "").startswith(SUMMARY_PROMPT):
            kind = "summary"
        elif messages and messages[-1].get("role") == "tool":
            kind = "tool_round"
        else:
            kind = "reply"
        record = StubRequest(
            received=time.monotonic(),
            kind=kind,
            body=body,
            messages=messages,
            tools=offered,
            stream=body.get("stream", True) is not False,
        )
        with self._lock:
            record.index = len(self.requests)
            self.requests.append(record)
        if self.on_request is not None:
            try:
                self.on_request(kind, record)
            except Exception:
                pass
        return record

    def _pause(self, handler, seconds: float, record: StubRequest) -> bool:
        """Wait, watching the client; False if it hung up meanwhile."""
        deadline = time.monotonic() + seconds
        while True:
            if _gone(handler.connection):
                record.closed = time.monotonic()
                return False
            left = deadline - time.monotonic()
            if left <= 0:
                return True
            time.sleep(min(POLL, left))

    def _chat(self, handler: BaseHTTPRequestHandler, body: dict):
        record = self._record(body)
        if self.refuse_tools and body.get("tools"):
            record.refused = True
            handler._json({"error": f"registry.ollama.ai/library/{MODEL} does not support tools"}, 400)
            return
        if self._slot is not None:
            # One slot: wait for it, unless the client gives up first
            while not self._slot.acquire(timeout=POLL):
                if _gone(handler.connection):
                    record.closed = time.monotonic()
                    return
        try:
            if record.kind == "summary":
                answer = Answer(self._summary(record.messages), first_token=self.summary_delay)
            else:
                answer = self.answer(record.messages, record.tools)
            record.text, record.call = answer.text, answer.call
            first_token = self.first_token if answer.first_token is None else answer.first_token
            if not self._pause(handler, first_token, record):
                return
            if record.stream:
                self._stream(handler, answer, record)
            else:
                self._whole(handler, answer, record)
        finally:
            if self._slot is not None:
                self._slot.release()

    @staticmethod
    def _summary(messages: list[dict]) -> str:
        said = "\n".join(m.get("content") or "" for m in messages if m.get("role") == "user")
        count = len(re.findall(r"^(?:USER|ASSISTANT|SYSTEM|TOOL): ", said, re.MULTILINE))
        return f"SUMMARY of {count} messages."

    @staticmethod
    def _message(content: str, call: Optional[Call] = None) -> dict:
        message = {"role": "assistant", "content": content}
        if call is not None:
            message["tool_calls"] = [{"function": {"name": call.name, "arguments": call.arguments}}]
        return message

    def _stall(self, handler, record: StubRequest):
        """Send nothing more, until the client hangs up."""
        record.stalled = time.monotonic()
        self._pause(handler, float("inf"), record)

    def _whole(self, handler, answer: Answer, record: StubRequest):
        """The reply as one JSON object ("stream": false)."""
        if answer.stall_after is not None:
            self._stall(handler, record)
            return
        text = answer.call.lead_in if answer.call else answer.text
        payload = {
            "model": MODEL,
            "created_at": "2026-01-01T00:00:00Z",
            "message": self._message(text, answer.call),
            "done": True,
            "done_reason": "stop",
        }
        try:
            record.first_byte = time.monotonic()
            handler._json(payload)
            if answer.call:
                record.tool_call_sent = record.first_byte
            record.finished = time.monotonic()
        except (BrokenPipeError, ConnectionResetError):
            record.closed = time.monotonic()

    def _stream(self, handler, answer: Answer, record: StubRequest):
        """The reply streamed word by word, as Ollama streams it."""
        text = answer.call.lead_in if answer.call else answer.text
        if answer.chunk:
            pieces = [text[i : i + answer.chunk] for i in range(0, len(text), answer.chunk)]
        else:
            pieces = re.findall(r"\s*\S+|\s+$", text)
        handler.send_response(200)
        handler.send_header("Content-Type", "application/x-ndjson")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()

        def send(payload: dict):
            data = (json.dumps(payload) + "\n").encode()
            handler.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            handler.wfile.flush()
            if record.first_byte is None:
                record.first_byte = time.monotonic()

        def chunk(message: dict, done: bool) -> dict:
            payload = {"model": MODEL, "created_at": "2026-01-01T00:00:00Z", "message": message, "done": done}
            if done:
                payload["done_reason"] = "stop"
            return payload

        try:
            for n, piece in enumerate(pieces):
                if n == answer.stall_after:
                    self._stall(handler, record)
                    return
                if n and not self._pause(handler, self.per_word, record):
                    return
                send(chunk(self._message(piece), False))
            if answer.call:
                if pieces and not self._pause(handler, self.per_word, record):
                    return
                send(chunk(self._message("", answer.call), False))
                record.tool_call_sent = time.monotonic()
            send(chunk(self._message(""), True))
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
            record.finished = time.monotonic()
        except (BrokenPipeError, ConnectionResetError):
            # Iris cut the request off (Stop, or the user went on)
            record.closed = time.monotonic()

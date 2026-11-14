"""A model server in trouble: one that stops answering part way through a
reply, or cannot be reached, ends the turn with an error that says so, and
Iris answers the next turn as usual. The servers are the scripted model on
a local port and httpx.MockTransport, so no real server is needed."""

import json
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import ollama
import pytest
from ollama._client import CONNECTION_ERROR_MESSAGE

from iris import llm, memory
from iris.assistant import VoiceAssistant
from iris.llm import ConversationManager, failure_message, ollama_client
from iris.llm_client import MODEL_TIMEOUT, OpenAIClient
from iris.plugin import Plugin, Tool
from iris.tools import Tools
from irisdev.voicetest.assistant import wait_until
from irisdev.voicetest.stub_llm import MODEL, SHORT, Rule, StubModel

BASE = "https://box.example:8000/v1"
# A wait short enough for a test, in place of MODEL_TIMEOUT
SHORT_WAIT = httpx.Timeout(0.5, connect=0.5)
# What a reply that stalls sends: a sentence for the voice, and a word more
BEGUN = "The first part of the answer comes here. And"
STALLS = Rule("stall", f"{BEGUN} then it goes on.", stall_after=len(BEGUN.split()))


@pytest.fixture
def stub():
    model = StubModel(rules=[STALLS], first_token=0.0, per_word=0.0)
    yield model
    model.close()


class Pipeline:
    current_language = "en"

    def __init__(self):
        self.spoken = []

    def begin_reply(self):
        pass

    def queue_speech(self, text, language=None, sound=None):
        self.spoken.append(text)

    def stop(self):
        pass

    def wait_for_speech(self):
        pass

    def said(self):
        return " ".join(self.spoken)


def _assistant(conversation) -> VoiceAssistant:
    assistant = object.__new__(VoiceAssistant)
    assistant.event_bus = MagicMock()
    assistant.speak_replies = True
    assistant.conversation = conversation
    assistant.speech_pipeline = Pipeline()
    assistant.tts_generator = SimpleNamespace(
        languages=lambda: frozenset({"en"}), speaks=lambda language: True
    )
    assistant.interrupt_event = threading.Event()
    assistant.suspend_event = threading.Event()
    assistant.tools = MagicMock()

    class Status:
        status_lock = threading.Lock()
        status_thinking = False

    assistant.status_manager = Status()
    return assistant


def _events(assistant, kind: str) -> list[dict]:
    return [c.args[1] for c in assistant.event_bus.emit.call_args_list if c.args[0] == kind]


# Every request to the model has a timeout -------------------------------------------


def test_clients_for_either_kind_of_server_give_up_on_a_silent_one():
    assert MODEL_TIMEOUT.connect is not None and MODEL_TIMEOUT.read is not None
    assert MODEL_TIMEOUT.connect < MODEL_TIMEOUT.read
    assert ollama_client("http://ollama.example:11434")._client.timeout == MODEL_TIMEOUT
    assert OpenAIClient(BASE)._client.timeout == MODEL_TIMEOUT
    # A timeout asked for (the controller's quick check) is kept
    quick = ollama_client("http://ollama.example:11434", timeout=8)
    assert quick._client.timeout == httpx.Timeout(8)


def test_replies_without_a_configured_server_have_a_timeout_too(monkeypatch):
    """With no server set, the ollama module answers list and show, and
    replies stream through a client of their own."""
    monkeypatch.setattr(ollama, "list", lambda: [])
    monkeypatch.setattr(ollama, "show", MagicMock(side_effect=ConnectionError("no server")))
    conversation = ConversationManager(model=MODEL)
    assert conversation.client is ollama
    assert conversation._reply_client()._client.timeout == MODEL_TIMEOUT


def test_filing_into_memory_has_a_timeout(monkeypatch):
    made = []

    class Client:
        def __init__(self, **kwargs):
            made.append(kwargs)
            self._client = MagicMock()

        def chat(self, **kwargs):
            message = ollama.Message(role="assistant", content="{}")
            return iter([ollama.ChatResponse(message=message)])

    monkeypatch.setattr(memory.ollama, "Client", Client)
    memory._abortable_chat(
        "http://ollama.example:11434", MODEL, [], {}, {}, abort=lambda: False, think=False
    )
    assert made[0]["timeout"] == MODEL_TIMEOUT


# A server that stops answering --------------------------------------------------------


def test_a_reply_that_stalls_ends_in_a_timeout(stub, monkeypatch):
    monkeypatch.setattr(llm, "MODEL_TIMEOUT", SHORT_WAIT)
    conversation = ConversationManager(model=MODEL, host=stub.url)
    said = []
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        for chunk in conversation.chat_stream("Please stall"):
            said.append(chunk)
    assert time.monotonic() - started < 5
    assert "".join(said) == BEGUN
    # Iris hung up, so the server is free again
    request = stub.requests[-1]
    assert request.stalled is not None
    assert wait_until(lambda: request.closed is not None, 2)


def test_the_turn_ends_with_an_error_that_says_so_and_the_next_is_answered(stub, monkeypatch):
    monkeypatch.setattr(llm, "MODEL_TIMEOUT", SHORT_WAIT)
    assistant = _assistant(ConversationManager(model=MODEL, host=stub.url))

    message = assistant.chat("Please stall")
    assert message == (
        f"The model server at {stub.url} stopped answering: nothing came for 60 seconds."
    )
    assert _events(assistant, "error") == [{"message": message}]
    # What came of the reply stays shown, ended as a reply cut short
    assert _events(assistant, "chat.done") == [
        {"text": BEGUN, "interrupted": True}
    ]
    assert assistant.speech_pipeline.spoken == ["The first part of the answer comes here."]
    assert not assistant.status_manager.status_thinking

    assistant.event_bus.reset_mock()
    assert assistant.chat("What is the capital of France?") == SHORT
    assert _events(assistant, "error") == []


class Stalls(httpx.SyncByteStream):
    """A streamed reply of one chunk, then nothing: the read times out."""

    def __iter__(self):
        delta = {"choices": [{"index": 0, "delta": {"content": "It begins. "}}]}
        yield f"data: {json.dumps(delta)}\n\n".encode()
        raise httpx.ReadTimeout("The read operation timed out")


def test_an_openai_compatible_server_that_stalls_is_not_called_ollama(monkeypatch):
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "big-model"}]})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stalls())

    transport = httpx.MockTransport(answer)
    monkeypatch.setattr(
        llm, "OpenAIClient", lambda host, **kw: OpenAIClient(host, transport=transport)
    )
    assistant = _assistant(ConversationManager("big-model", host=BASE))
    message = assistant.chat("Hello?")
    assert message == f"The model server at {BASE} stopped answering: nothing came for 60 seconds."
    assert "ollama" not in message.lower()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_a_server_out_of_reach_is_said_to_be(monkeypatch):
    host = f"http://127.0.0.1:{_free_port()}"  # nothing listens there
    conversation = ConversationManager(model=MODEL, host=host)
    assert not conversation.reachable
    message = _assistant(conversation).chat("Hello?")
    assert message == f"The model server at {host} cannot be reached."


def test_failures_are_told_in_words_that_fit_any_server():
    for error in (httpx.ConnectError("refused"), ConnectionError(CONNECTION_ERROR_MESSAGE)):
        assert failure_message(error, BASE) == f"The model server at {BASE} cannot be reached."
    assert "did not answer in time" in failure_message(httpx.WriteTimeout("slow"), BASE)
    lost = failure_message(httpx.RemoteProtocolError("peer closed connection"), BASE)
    assert lost == f"The connection to the model server at {BASE} was lost."
    assert failure_message(RuntimeError("no such model"), BASE) == (
        "The language model could not answer: no such model"
    )


# An Ollama model without tool support ------------------------------------------------

GUIDANCE = "Call get_time for the time anywhere."


class Clock(Plugin):
    id = "clock"
    name = "Clock"

    def tools(self, context):
        return [
            Tool(
                name="get_time",
                description="The time in a city.",
                run=lambda city="", **kw: {"success": True, "result": f"14:00 in {city}"},
                parameters={"city": {"type": "string"}},
                guidance=GUIDANCE,
            )
        ]


def _with_tools(stub) -> ConversationManager:
    return ConversationManager(model=MODEL, host=stub.url, tools=Tools([Clock()]))


def _offered(request) -> bool:
    return bool(request.tools) or GUIDANCE in request.system


def test_a_model_that_refuses_tools_is_asked_again_without_them_and_not_offered_them_again():
    stub = StubModel(first_token=0.0, per_word=0.0, refuse_tools=True)
    try:
        conversation = _with_tools(stub)
        assert "".join(conversation.chat_stream("What is the capital of France?")) == SHORT
        refused, again = stub.requests
        assert refused.refused and refused.tools == ["get_time"]
        # ...without the tools or the guidance on them
        assert not again.refused and not _offered(again)

        assert "".join(conversation.chat_stream("And of Italy?")) == SHORT
        assert len(stub.requests) == 3 and not _offered(stub.requests[-1])

        # Remembered for that model only, and for that server
        conversation.set_model("other:latest")
        list(conversation.chat_stream("Hello?"))
        assert stub.requests[-2].refused and stub.requests[-2].tools == ["get_time"]
        conversation.set_host(stub.url)
        conversation.set_model(MODEL)
        list(conversation.chat_stream("Hello?"))
        assert stub.requests[-2].refused
    finally:
        stub.close()


def test_a_model_ollama_says_cannot_take_tools_is_never_offered_them():
    stub = StubModel(first_token=0.0, per_word=0.0, refuse_tools=True, capabilities=["completion"])
    try:
        conversation = _with_tools(stub)
        assert "".join(conversation.chat_stream("What is the capital of France?")) == SHORT
        [request] = stub.requests
        assert not request.refused and not _offered(request)
    finally:
        stub.close()


def test_a_model_that_takes_tools_is_offered_them():
    stub = StubModel(first_token=0.0, per_word=0.0)
    try:
        list(_with_tools(stub).chat_stream("What is the capital of France?"))
        [request] = stub.requests
        assert request.tools == ["get_time"] and GUIDANCE in request.system
    finally:
        stub.close()


def test_other_errors_are_not_taken_for_a_refusal():
    stub = StubModel(first_token=0.0, per_word=0.0)
    try:
        conversation = _with_tools(stub)
        failing = MagicMock()
        failing.chat.side_effect = ollama.ResponseError("model 'x' not found", 404)
        conversation.client = failing
        with pytest.raises(ollama.ResponseError):
            list(conversation.chat_stream("Hello?"))
        assert failing.chat.call_count == 1 and not conversation._no_tools
    finally:
        stub.close()

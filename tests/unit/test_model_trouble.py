"""A model server that misbehaves beyond going silent: it fails part way
through a reply or before it (HTTP 500), the model chosen disappears from
it (404), or the user switches servers while a reply is coming. The user
is told what happened in plain words, and Iris answers the next turn. The
servers are the scripted model on a local port and httpx.MockTransport."""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from iris import llm
from iris.assistant import VoiceAssistant
from iris.controller import AssistantController
from iris.events import EventBus
from iris.llm import ConversationManager
from iris.llm_client import OpenAIClient
from iris.plugin import Plugin, Tool
from iris.tools import Tools
from irisdev.voicetest.assistant import wait_until
from irisdev.voicetest.stub_llm import LONG, MODEL, SHORT, Rule, StubModel

BEGUN = "The first part of the answer comes here. And"
FAILS = Rule("fail", f"{BEGUN} then it goes on.", error_after=len(BEGUN.split()))
REFUSES = Rule("refuse", status=500, error="model runner has unexpectedly stopped")
BASE = "https://box.example:8000/v1"


@pytest.fixture
def stub():
    model = StubModel(rules=[FAILS, REFUSES], first_token=0.0, per_word=0.0)
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
    assistant.tts_generator = SimpleNamespace(languages=lambda: frozenset({"en"}))
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


# HTTP 500 -----------------------------------------------------------------------------


def test_a_server_that_fails_part_way_ends_the_turn_and_the_next_is_answered(stub):
    """Ollama tells of a failure after the reply has begun in the stream
    itself: what came stays shown as a reply cut short, and the error says
    who failed, without the status code Ollama's client adds (-1 here)."""
    assistant = _assistant(ConversationManager(model=MODEL, host=stub.url))
    message = assistant.chat("Please fail")
    assert message == (
        f"The model server at {stub.url} ran into a problem: model runner has unexpectedly stopped"
    )
    assert _events(assistant, "error") == [{"message": message}]
    assert _events(assistant, "chat.done") == [{"text": BEGUN, "interrupted": True}]
    assert assistant.speech_pipeline.spoken == ["The first part of the answer comes here."]
    assert not assistant.status_manager.status_thinking

    assistant.event_bus.reset_mock()
    assert assistant.chat("What is the capital of France?") == SHORT
    assert _events(assistant, "error") == []


def test_a_server_that_answers_500_says_so(stub):
    assistant = _assistant(ConversationManager(model=MODEL, host=stub.url))
    message = assistant.chat("Please refuse")
    assert message == (
        f"The model server at {stub.url} ran into a problem: model runner has unexpectedly stopped"
    )
    assert stub.requests[-1].status == 500
    # Nothing came, so there is no reply to show
    assert _events(assistant, "chat.done") == []
    assert assistant.chat("And now?") == SHORT


class FailsPartWay(httpx.SyncByteStream):
    """An OpenAI-compatible stream that begins, then reports an error."""

    def __iter__(self):
        delta = {"choices": [{"index": 0, "delta": {"content": "It begins. "}}]}
        yield f"data: {json.dumps(delta)}\n\n".encode()
        yield f"data: {json.dumps({'error': {'message': 'out of memory'}})}\n\n".encode()


def _openai(monkeypatch, answer):
    transport = httpx.MockTransport(answer)
    monkeypatch.setattr(llm, "OpenAIClient", lambda host, **kw: OpenAIClient(host, transport=transport))
    return _assistant(ConversationManager("big-model", host=BASE))


def _models(request: httpx.Request):
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json={"data": [{"id": "big-model"}]})
    return None


def test_an_openai_compatible_server_that_fails_part_way_says_so(monkeypatch):
    calls = []

    def answer(request: httpx.Request) -> httpx.Response:
        if listed := _models(request):
            return listed
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=FailsPartWay())
        delta = {"choices": [{"index": 0, "delta": {"content": "Fine again."}}]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=f"data: {json.dumps(delta)}\n\ndata: [DONE]\n\n".encode())

    assistant = _openai(monkeypatch, answer)
    message = assistant.chat("Hello?")
    assert message == f"The model server at {BASE} ran into a problem: out of memory"
    assert _events(assistant, "chat.done") == [{"text": "It begins. ", "interrupted": True}]
    assert assistant.chat("Hello again?") == "Fine again."


def test_an_openai_compatible_server_that_answers_500_says_so(monkeypatch):
    def answer(request: httpx.Request) -> httpx.Response:
        return _models(request) or httpx.Response(500, json={"error": {"message": "engine crashed"}})

    message = _openai(monkeypatch, answer).chat("Hello?")
    assert message == f"The model server at {BASE} ran into a problem: engine crashed"


# A model that disappears ------------------------------------------------------------


def test_a_model_gone_from_the_server_is_named_and_another_one_answers(stub):
    """The model was chosen, then removed from the server: the user is told
    which model is gone, and once another is chosen the next turn is
    answered. The warning is retracted then (kind model_missing)."""
    conversation = ConversationManager(model=MODEL, host=stub.url)
    assistant = _assistant(conversation)
    assert assistant.chat("Hello?") == SHORT
    stub.configure(missing=True)

    message = assistant.chat("Are you there?")
    assert MODEL in message and "not on the server" in message and "Settings" in message
    assert "status code" not in message
    assert _events(assistant, "error")[-1] == {"message": message, "kind": "model_missing"}

    conversation.set_model("other:latest")
    assistant.event_bus.reset_mock()
    assert assistant.chat("And now?") == SHORT
    assert _events(assistant, "error") == []


def test_choosing_another_model_retracts_the_warning(stub):
    bus = EventBus()
    controller = AssistantController(bus)
    conversation = ConversationManager(model=MODEL, host=stub.url)
    controller.assistant = SimpleNamespace(conversation=conversation, config_info=lambda: {})
    bus.emit("error", {"message": "gone", "kind": "model_missing"})
    assert controller.set_model("other:latest")["ok"]
    assert not [e for e in controller.history()["events"] if e["type"] == "error"]


# Switching servers while a reply comes ------------------------------------------------


@pytest.fixture
def two():
    """Two servers: the reply comes slowly from the first."""
    first = StubModel(first_token=0.0, per_word=0.03)
    second = StubModel(first_token=0.0, per_word=0.0)
    yield first, second
    first.close()
    second.close()


def test_a_reply_finishes_on_the_server_it_began_on(two):
    first, second = two
    conversation = ConversationManager(model=MODEL, host=first.url)
    stream = conversation.chat_stream("Tell me a long story.")
    said = [next(stream)]
    conversation.set_host(second.url)
    said += list(stream)
    assert "".join(said) == LONG
    assert len(first.requests) == 1 and first.requests[0].finished is not None
    # The next turn goes to the server switched to
    assert "".join(conversation.chat_stream("What is the capital of France?")) == SHORT
    assert len(second.requests) == 1


class Clock(Plugin):
    """A tool that takes a moment; the user switches servers meanwhile."""

    id = "clock"
    name = "Clock"

    def __init__(self, during):
        self.during = during

    def tools(self, context):
        def run(**kwargs):
            self.during()
            return {"success": True, "result": "It is 14:00."}

        return [Tool(name="get_time", description="The time.", run=run)]


def test_the_rounds_of_a_reply_stay_on_its_server(two):
    """A tool's result is answered by the server that called the tool: the
    other may not have the model, and knows nothing of the call."""
    first, second = two
    first.set_rules([Rule("time", tool=("get_time", lambda m: {}), after_tool="It is two o'clock.")])
    conversation = ConversationManager(model=MODEL, host=first.url)
    conversation.tools = Tools([Clock(lambda: conversation.set_host(second.url))])
    assert "".join(conversation.chat_stream("What time is it?")) == "It is two o'clock."
    assert [r.kind for r in first.requests] == ["reply", "tool_round"]
    assert second.requests == []


def test_stop_after_a_switch_cuts_off_the_reply_being_streamed(two):
    first, second = two
    first.configure(per_word=0.2)
    conversation = ConversationManager(model=MODEL, host=first.url)
    stream = conversation.chat_stream("Tell me a long story.")
    next(stream)
    conversation.set_host(second.url)
    conversation.cancel()
    # The first server is told at once: the request is cut off, and it
    # stops generating
    assert wait_until(lambda: first.requests[0].closed is not None, 2.0)
    assert first.requests[0].finished is None
    # The client of the server switched to is still there for the next turn
    assert "".join(conversation.chat_stream("What is the capital of France?")) == SHORT


def test_the_app_shows_the_switch_while_the_reply_goes_on(two):
    """Through the controller, as the app switches: the reply in progress
    comes whole, and the config shows the new server."""
    first, second = two
    bus = EventBus()
    events = []
    bus.subscribe(lambda kind, data: events.append((kind, data)))
    controller = AssistantController(bus)
    controller._ollama_host = first.url
    conversation = ConversationManager(model=MODEL, host=first.url)
    controller.assistant = SimpleNamespace(
        conversation=conversation, config_info=lambda: {"ollama_host": conversation.host}
    )
    said = []

    def reply():
        said.extend(conversation.chat_stream("Tell me a long story."))

    thread = threading.Thread(target=reply)
    thread.start()
    assert wait_until(lambda: first.requests and first.requests[0].first_byte, 2.0)
    assert controller.set_ollama_host(second.url)["ok"]
    thread.join(10)
    assert "".join(said) == LONG
    assert ("config", {"ollama_host": second.url}) in events
    started = time.monotonic()
    assert "".join(conversation.chat_stream("Hello?")) == SHORT
    assert time.monotonic() - started < 2 and len(second.requests) == 1

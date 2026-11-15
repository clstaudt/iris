"""Memory written: the core tells the app when filing wrote to memory
(memory.updated, written), which the orb shows, and not when memory changes
otherwise: a topic or a thread forgotten, another thread opened, everything
forgotten. While memory is off or the thread is in amnesia nothing is
written, and nothing is told.

Iris is the real assistant, started offline as in test_offline_start, with
the scripted language model and a silent stand-in for the audio devices.
Filing answers from a script, and runs when the test asks, not in a quiet
spell on a thread of its own."""

import queue
import threading

import pytest

from iris import memory as memory_module
from iris.assistant import VoiceAssistant
from iris.controller import AssistantController
from iris.events import EventBus
from irisdev.voicetest.assistant import wait_until
from irisdev.voicetest.stub_llm import MODEL
from tests.unit.test_memory import ScriptedChat
from tests.unit.test_offline_start import Microphone, Recorded, offline, stub  # noqa: F401  (fixtures)

QUESTION = "What is the capital of France?"


@pytest.fixture
def iris(stub, offline, tmp_path, monkeypatch):  # noqa: F811  (fixtures)
    """Iris running with memory, and the controller the app talks to."""
    monkeypatch.setenv("IRIS_MEMORY_DB", str(tmp_path / "memory.db"))
    # Filed as soon as the conversation rests, when the test files
    monkeypatch.setattr(memory_module, "QUIET_BEFORE_FILING", 0.0)
    bus = EventBus()
    seen = Recorded(bus)
    assistant = VoiceAssistant(
        llm_model=MODEL,
        ollama_host=stub.url,
        plugins=[],
        event_bus=bus,
        turn_queue=queue.Queue(),
        audio=Microphone(),
    )
    assistant.memory.stop()
    assistant.memory.filer.chat = ScriptedChat(
        decisions=[{"remember": True, "topic_id": 0, "title": "Capitals"}],
        summaries=[{"summary": "Paris is the capital of France.", "gist": "Paris"}],
    )
    controller = AssistantController(bus)
    controller.assistant = assistant
    loop = threading.Thread(target=assistant.run_loop, daemon=True)
    loop.start()
    yield controller, seen
    assistant.stop_event.set()
    loop.join(5)
    assert not loop.is_alive()


def _answered(controller, seen, text: str):
    """Type a message, and wait until its reply is done and the
    conversation rests."""
    assistant = controller.assistant
    replies = len(seen.done())
    assistant.turn_queue.put({"text": text, "id": f"t{replies}"})
    assert wait_until(lambda: len(seen.done()) == replies + 1, 10)
    assert wait_until(lambda: not assistant._conversation_busy(), 10)


def _updates(seen) -> list[dict]:
    return [data for kind, data in seen.events if kind == "memory.updated"]


def test_filing_a_turn_says_memory_was_written_and_nothing_else_does(iris):
    controller, seen = iris
    assistant = controller.assistant
    store = assistant.memory.store
    _answered(controller, seen, QUESTION)
    assert wait_until(lambda: store.turn_count() == 2, 10)
    first = assistant.conversation.thread
    assert _updates(seen) == []  # a turn kept is not filed yet

    # Filed into a topic, as in a quiet spell: once, and written
    assistant.memory._file()
    topics = store.topics(first)
    assert [t.title for t in topics] == ["Capitals"]
    assert _updates(seen) == [{"written": True}]
    # Nothing left to file: nothing told
    assistant.memory._file()
    assert _updates(seen) == [{"written": True}]

    # The topic forgotten, a new thread and back, the new one forgotten:
    # memory is updated, and nothing is written
    assert controller.memory_forget(topics[0].id)["ok"]
    assert controller.new_thread("iris")["ok"]
    _answered(controller, seen, "And of Italy?")
    second = assistant.conversation.thread
    assert second not in (None, first)
    assert controller.open_thread(first)["ok"]
    assert controller.forget_thread(second)["ok"]
    assert controller.memory_forget_all()["ok"]
    assert any(kind == "thread.opened" for kind, _ in seen.events)
    assert any(kind == "memory.forgotten" for kind, _ in seen.events)
    assert _updates(seen) == [{"written": True}, {}, {}]


def test_with_memory_off_or_in_amnesia_nothing_is_written_or_told(iris):
    controller, seen = iris
    assistant = controller.assistant
    assert controller.set_remember(False)["ok"]
    _answered(controller, seen, QUESTION)
    assistant.memory._file()
    assert controller.set_remember(True)["ok"]
    assert controller.set_amnesia(True)["ok"]
    _answered(controller, seen, "And of Italy?")
    assistant.memory._file()
    assert assistant.memory.store.turn_count() == 0
    assert _updates(seen) == []

"""Unit tests for long-term memory: storage, filing, recall and context."""

import threading
import time
from unittest.mock import patch

import pytest

from iris import memory as memory_module
from iris.llm import ConversationManager
from iris.memory import PIN_TURNS, Filer, MemoryKeeper, MemoryStore
from iris.tools import Tools


@pytest.fixture
def store():
    return MemoryStore(":memory:")


def _exchange(store, user, assistant, source="voice"):
    store.add_turn("user", user, source)
    store.add_turn("assistant", assistant)


# -- store -------------------------------------------------------------------


def test_recent_turns_fit_the_budget_oldest_first(store):
    for n in range(10):
        _exchange(store, f"question {n} " + "x" * 80, f"answer {n} " + "y" * 80)
    turns = store.recent_turns(budget=100)
    assert turns[-1]["text"].startswith("answer 9")
    assert [t["id"] for t in turns] == sorted(t["id"] for t in turns)
    assert sum(len(t["text"]) // 4 for t in turns) <= 100


def test_unfiled_turns_group_into_exchanges_without_notes(store):
    _exchange(store, "Plan a Lisbon trip", "Three days: Alfama, Belém, Sintra.")
    store.add_turn("system", "You are now HAL.")
    _exchange(store, "Thanks!", "You're welcome.")
    exchanges = store.unfiled_exchanges()
    assert len(exchanges) == 2
    assert (
        exchanges[0].text
        == "User: Plan a Lisbon trip\nAssistant: Three days: Alfama, Belém, Sintra."
    )
    assert len(exchanges[0].turn_ids) == 2


def test_search_finds_words_anywhere_and_names_the_topic(store):
    topic = store.create_topic("Lisbon trip", "3 days, Sintra", "Trip in late October.")
    _exchange(store, "Which neighborhood in Lisbon?", "Stay in Chiado, near Rossio station.")
    store.mark_filed([t["id"] for t in store.recent_turns()], topic)
    _exchange(store, "Tomatoes before the frost?", "No, after the last frost.")

    hits = store.search("what was that station called? (Rossio)")
    assert hits and hits[0]["title"] == "Lisbon trip"
    assert "[Rossio]" in hits[0]["snippet"]
    # Unfiled turns are searchable too
    assert store.search("frost")[0]["topic_id"] is None
    assert store.search("???") == []


def test_forget_removes_the_topic_its_turns_and_their_search_entries(store):
    topic = store.create_topic("Secret", "surprise party", "A surprise party for Ana.")
    _exchange(store, "Plan a surprise party for Ana", "Saturday at eight.")
    store.mark_filed([t["id"] for t in store.recent_turns()], topic)
    store.forget(topic)
    assert store.topic(topic) is None
    assert store.recent_turns() == []
    assert store.search("surprise") == []


def test_index_lists_recent_topics_first_within_budget(store, monkeypatch):
    clock = iter(range(1000, 2000))
    monkeypatch.setattr(memory_module.time, "time", lambda: next(clock))
    for n in range(30):
        store.create_topic(f"Topic {n}", "a gist " * 5, "summary")
    lines = store.index_lines(budget=60)
    assert lines[0].startswith("#30 Topic 29:")
    assert 1 < len(lines) < 30


# -- filing ------------------------------------------------------------------


class ScriptedChat:
    """Answers the filer's two kinds of question from a script: one decision
    per exchange, one summary per topic touched."""

    def __init__(self, decisions, summaries=()):
        self.decisions = list(decisions)
        self.summaries = list(summaries)
        self.prompts = []

    def __call__(self, messages, schema, abort=lambda: False):
        self.prompts.append(messages[0]["content"])
        if "remember" in schema["properties"]:
            return self.decisions.pop(0)
        return self.summaries.pop(0)


def test_new_subjects_become_topics_and_small_talk_does_not(store):
    _exchange(store, "Plan a Lisbon trip for late October", "Alfama, Belém, Sintra.")
    _exchange(store, "How do I get to Sintra?", "Train from Rossio.")
    _exchange(store, "My sister's birthday is November 3rd", "Noted.")
    _exchange(store, "Thanks!", "You're welcome.")
    chat = ScriptedChat(
        decisions=[
            {"remember": True, "topic_id": 0, "title": "Lisbon trip"},
            # Linked to the topic the first exchange just created
            {"remember": True, "topic_id": 1, "title": "Getting to Sintra"},
            {"remember": True, "topic_id": 0, "title": "Sister's birthday"},
            {"remember": False, "topic_id": 0, "title": "Thanks"},
        ],
        summaries=[
            {"summary": "Trip in late October, Sintra by train.", "gist": "late Oct, Sintra"},
            {"summary": "Birthday on November 3rd.", "gist": "Nov 3"},
        ],
    )
    assert Filer(store, chat).file() is True
    topics = {t.title: t.id for t in store.topics()}
    assert set(topics) == {"Lisbon trip", "Sister's birthday"}
    assert len(store.topic(topics["Lisbon trip"]).turns) == 4
    assert store.unfiled_exchanges() == []
    # One summary per topic, not per exchange
    assert chat.summaries == []
    # Small talk is kept and searchable, just not a topic
    assert store.search("welcome")[0]["topic_id"] is None


def test_each_decision_sees_the_exchange_before_it(store):
    _exchange(store, "My Golf's brakes squeak", "Probably worn pads.")
    _exchange(store, "I booked the garage for the 12th", "Noted.")
    chat = ScriptedChat(
        decisions=[
            {"remember": True, "topic_id": 0, "title": "Golf brakes"},
            {"remember": True, "topic_id": 1, "title": "Garage booking"},
        ],
        summaries=[{"summary": "Brake service on the 12th.", "gist": "brakes, 12th"}],
    )
    Filer(store, chat).file()
    assert "Said just before:\nUser: My Golf's brakes squeak" in chat.prompts[1]
    assert "#1 Golf brakes" in chat.prompts[1]


def test_continuing_a_subject_updates_its_topic(store):
    topic = store.create_topic("Lisbon trip", "late Oct", "Trip in late October.")
    _exchange(store, "For Lisbon, where should I stay?", "Chiado.")
    chat = ScriptedChat(
        decisions=[{"remember": True, "topic_id": topic, "title": "Lisbon stay"}],
        summaries=[
            {"summary": "Trip in late October, staying in Chiado.", "gist": "late Oct, Chiado"}
        ],
    )
    assert Filer(store, chat).file() is True
    assert store.topic(topic).summary == "Trip in late October, staying in Chiado."
    assert len(store.topics()) == 1
    assert "What is known so far:\nTrip in late October." in chat.prompts[-1]


def test_a_topic_number_that_does_not_exist_starts_a_new_topic(store):
    _exchange(store, "Plan a Lisbon trip", "Sure.")
    chat = ScriptedChat(
        decisions=[{"remember": True, "topic_id": 42, "title": "Lisbon"}],
        summaries=[{"summary": "Lisbon trip.", "gist": "Lisbon"}],
    )
    Filer(store, chat).file()
    assert [t.title for t in store.topics()] == ["Lisbon"]
    assert store.unfiled_exchanges() == []


def test_failed_filing_leaves_turns_for_next_time(store):
    _exchange(store, "Plan a Lisbon trip", "Sure.")

    def broken(messages, schema):
        raise ConnectionError("server away")

    keeper = MemoryKeeper(store, broken)
    keeper._file()
    assert len(store.unfiled_exchanges()) == 1


# -- tools -------------------------------------------------------------------


def test_recall_returns_the_summary_and_latest_exchanges(store):
    topic = store.create_topic("Lisbon trip", "late Oct", "Trip in late October with Sintra.")
    _exchange(store, "Where to stay in Lisbon?", "Chiado.")
    store.mark_filed([t["id"] for t in store.recent_turns()], topic)
    tools = Tools()
    tools.enable_memory(store)

    result = tools.execute("recall", topic_id=topic)
    assert result["success"] and result["topic_id"] == topic
    assert "Summary: Trip in late October with Sintra." in result["result"]
    assert "Chiado." in result["result"]
    assert tools.execute("recall", topic_id=99)["success"] is False
    assert "Chiado" in tools.execute("search_memory", query="Chiado")["result"]


# -- context -----------------------------------------------------------------


@patch("iris.llm.ollama")
def _conversation(store, mock_ollama, tools=None):
    mock_ollama.show.return_value.modelinfo = {}
    keeper = MemoryKeeper(store, lambda messages, schema: {})
    return ConversationManager("fake-llm", tools=tools, memory=keeper)


def test_context_holds_the_index_and_the_live_thread(store):
    store.create_topic("Lisbon trip", "late Oct, Sintra", "Trip in late October.")
    _exchange(store, "Hi", "Hello!")
    messages = _conversation(store).manage_context()
    assert messages[0]["role"] == "system"
    assert "#1 Lisbon trip: late Oct, Sintra (today)" in messages[0]["content"]
    assert [m["content"] for m in messages[1:]] == ["Hi", "Hello!"]


def test_recalled_topics_stay_in_context_for_a_few_turns(store):
    topic = store.create_topic("Lisbon trip", "late Oct", "Staying in Chiado.")
    conversation = _conversation(store)
    conversation.pinned[topic] = PIN_TURNS
    assert (
        "Recalled #1 Lisbon trip: Staying in Chiado." in conversation.manage_context()[0]["content"]
    )

    # Each user turn counts down; afterwards the summary leaves the prompt
    for _ in range(PIN_TURNS):
        conversation.pinned = {t: n - 1 for t, n in conversation.pinned.items() if n > 1}
    assert "Recalled" not in conversation.manage_context()[0]["content"]


def test_turns_are_recorded_with_their_source(store):
    conversation = _conversation(store)
    conversation._record("What's on today?", "Nothing planned.", "text")
    conversation.add_note("You are now HAL.")
    turns = store.recent_turns()
    assert [(t["role"], t["source"]) for t in turns] == [
        ("user", "text"),
        ("assistant", None),
        ("system", None),
    ]


# -- talking to models -------------------------------------------------------


@pytest.mark.parametrize(
    "reply, parsed",
    [
        (
            '{"topic_id": 1, "remember": true, "title": "Lisbon"}',
            {"topic_id": 1, "remember": True, "title": "Lisbon"},
        ),
        (
            '```json\n{"topic_id": 0, "remember": false, "title": "Hi"}\n```',
            {"topic_id": 0, "remember": False, "title": "Hi"},
        ),
        (
            'Sure! {"topic_id": 1, "remember": true, "title": "x"}\n{"other": 1}',
            {"topic_id": 1, "remember": True, "title": "x"},
        ),
        ("Yes, topic #1. Title: Lisbon stay", None),
        ('{broken {"summary": "s", "gist": "g"}', {"summary": "s", "gist": "g"}),
    ],
)
def test_json_is_found_in_untidy_replies(reply, parsed):
    assert memory_module.parse_json(reply) == parsed


class Chunk:
    def __init__(self, content):
        self.message = type("Message", (), {"content": content})()


class FakeHttp:
    def __init__(self):
        self.closed = threading.Event()

    def close(self):
        self.closed.set()


class FakeOllama:
    """Stands in for ollama.Client: scripted replies, streamed in chunks,
    slowly if asked, and cut off like a real connection when closed."""

    replies: list = []
    calls: list = []
    delay = 0.0

    def __init__(self, host=None):
        self._client = FakeHttp()

    def chat(self, model, messages, format, options, stream, think=None):
        FakeOllama.calls.append(messages)
        reply = FakeOllama.replies.pop(0)

        def chunks():
            for part in (reply[: len(reply) // 2], reply[len(reply) // 2 :]):
                if self._client.closed.wait(FakeOllama.delay):
                    raise ConnectionError("connection closed")
                yield Chunk(part)

        return chunks()


@pytest.fixture
def fake_ollama(monkeypatch):
    monkeypatch.setattr(memory_module.ollama, "Client", FakeOllama)
    FakeOllama.replies, FakeOllama.calls, FakeOllama.delay = [], [], 0.0
    return FakeOllama


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "remember": {"type": "boolean"},
        "topic_id": {"enum": [1, 0]},
        "title": {"type": "string"},
    },
    "required": ["remember", "topic_id", "title"],
}


def test_a_model_that_ignores_the_schema_is_asked_again(fake_ollama):
    fake_ollama.replies = [
        "Yes, worth remembering. Topic #1.",
        '{"remember": true, "topic_id": 1, "title": "Lisbon stay"}',
    ]
    chat = memory_module.chat_json(lambda: None, lambda: "cloud-model")
    assert chat([{"role": "user", "content": "Decide."}], DECISION_SCHEMA)["topic_id"] == 1
    first_prompt = fake_ollama.calls[0][-1]["content"]
    shape = '{"remember": <true or false>, "topic_id": <one of 1, 0>, "title": "<text>"}'
    assert f"in this shape:\n{shape}" in first_prompt
    assert fake_ollama.calls[1][-1]["content"] == "Reply with only the JSON object, nothing else."


def test_a_call_in_flight_is_cut_off_when_the_conversation_resumes(fake_ollama):
    fake_ollama.replies = ['{"remember": true, "topic_id": 1, "title": "Lisbon"}']
    fake_ollama.delay = 5.0  # a long generation
    chat = memory_module.chat_json(lambda: None, lambda: "model")
    started = time.monotonic()

    def user_speaks():
        return time.monotonic() - started > 0.1

    with pytest.raises(memory_module.FilingInterruptedError):
        chat([{"role": "user", "content": "Decide."}], DECISION_SCHEMA, abort=user_speaks)
    assert time.monotonic() - started < 0.5


def test_an_interrupted_filing_saves_nothing_half_done(store):
    _exchange(store, "Plan a Lisbon trip", "Sure.")

    def interrupted(messages, schema, abort=lambda: False):
        raise memory_module.FilingInterruptedError()

    assert Filer(store, interrupted).file() is False
    assert len(store.unfiled_exchanges()) == 1
    assert store.topics() == []


# -- the conversation comes first --------------------------------------------


def test_filing_stops_before_the_next_call_and_resumes_later(store):
    _exchange(store, "Plan a Lisbon trip", "Alfama, Belém, Sintra.")
    _exchange(store, "My sister's birthday is November 3rd", "Noted.")
    chat = ScriptedChat(
        decisions=[
            {"remember": True, "topic_id": 0, "title": "Lisbon trip"},
            {"remember": True, "topic_id": 0, "title": "Sister's birthday"},
        ],
        summaries=[
            {"summary": "Lisbon in late October.", "gist": "Lisbon"},
            {"summary": "Birthday Nov 3.", "gist": "Nov 3"},
        ],
    )
    asked = []

    def user_speaks_after_one_call():
        asked.append(len(chat.prompts))
        return len(chat.prompts) >= 1

    filer = Filer(store, chat)
    assert filer.file(user_speaks_after_one_call) is True
    # One decision made and saved, then filing gave way: no further call
    assert len(chat.prompts) == 1
    assert len(store.unfiled_exchanges()) == 1
    assert list(store.pending_summaries()) == [1]

    # The next quiet spell picks up exactly there
    assert filer.file() is True
    assert len(chat.prompts) == 4  # one decision, two summaries
    assert store.unfiled_exchanges() == [] and store.pending_summaries() == {}
    assert store.topic(1).summary == "Lisbon in late October."


def test_a_busy_conversation_gets_the_llm_first(store, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(memory_module.time, "monotonic", lambda: clock[0])
    busy = [False]
    keeper = MemoryKeeper(store, lambda m, s: {}, busy=lambda: busy[0])

    assert keeper.should_yield()  # just started: not yet quiet
    clock[0] += memory_module.QUIET_BEFORE_FILING + 1
    assert not keeper.should_yield()  # quiet long enough

    busy[0] = True  # the user starts talking
    assert keeper.should_yield()
    busy[0] = False
    assert keeper.should_yield()  # and the quiet period starts over
    clock[0] += memory_module.QUIET_BEFORE_FILING + 1
    assert not keeper.should_yield()

    keeper.note_turn()
    assert keeper.should_yield()


def test_old_databases_get_the_summarized_column(tmp_path):
    import sqlite3

    path = tmp_path / "memory.db"
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE turns (id INTEGER PRIMARY KEY, ts REAL NOT NULL, role TEXT NOT NULL, "
        "text TEXT NOT NULL, source TEXT, topic_id INTEGER, filed INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO turns (ts, role, text, filed, topic_id) VALUES (1, 'user', 'hello', 1, 1);"
    )
    db.commit()
    db.close()

    store = MemoryStore(path)
    assert store.pending_summaries() == {}  # existing filed turns count as summarized
    assert store.recent_turns()[0]["text"] == "hello"


# -- amnesia mode ------------------------------------------------------------


def test_amnesia_keeps_the_conversation_going_but_writes_nothing(store):
    _exchange(store, "Plan a Lisbon trip", "Alfama, Belém, Sintra.")
    conversation = _conversation(store)
    conversation.set_amnesia(True)
    conversation._record("Something private", "Understood.", "voice")
    conversation.add_note("You are now HAL.")

    # Nothing reached the store, but the model still follows the conversation
    assert [t["text"] for t in store.recent_turns()] == [
        "Plan a Lisbon trip",
        "Alfama, Belém, Sintra.",
    ]
    contents = [m["content"] for m in conversation.manage_context()[1:]]
    assert contents[-3:] == ["Something private", "Understood.", "You are now HAL."]
    assert store.search("private") == []

    # Ending amnesia forgets those turns entirely
    conversation.set_amnesia(False)
    assert "Something private" not in [m["content"] for m in conversation.manage_context()]
    conversation._record("Back on the record", "Noted.", "text")
    assert store.recent_turns()[-1]["text"] == "Noted."


def test_amnesia_still_reads_memory(store):
    store.create_topic("Lisbon trip", "late Oct, Sintra", "Trip in late October.")
    conversation = _conversation(store)
    conversation.set_amnesia(True)
    assert "#1 Lisbon trip" in conversation.manage_context()[0]["content"]

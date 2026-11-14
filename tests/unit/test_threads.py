"""Unit tests for threads: each a conversation with one persona and a memory
of its own. A thread is stored with its first turn; its index, search and
recall cover it alone, and filing files into its own topics. Opening one
takes on its persona (voice and words too), forgetting one leaves nothing
of it, and a memory from before threads becomes the first, with Iris."""

import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

from iris import personas
from iris.audio import AudioPlayer
from iris.controller import LAST_THREAD, AssistantController
from iris.events import EventBus
from iris.llm import ConversationManager
from iris.memory import Filer, MemoryKeeper, MemoryStore
from iris.plugin import PluginContext
from iris.plugins import memory as memory_plugin
from iris.tools import Tools
from iris.tts import SpeechGenerator
from tests.unit.test_controller import FakeAssistant
from tests.unit.test_memory import ScriptedChat


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Iris, and HAL with words of his own."""
    builtin, user = tmp_path / "builtin", tmp_path / "user"
    builtin.mkdir()
    (builtin / "iris.yaml").write_text(
        'name: "Iris"\nengine: "kyutai"\nvoice: "anna"\ndescription: |\n  Warm.\n'
    )
    (builtin / "hal.yaml").write_text(
        'name: "HAL"\nengine: "kokoro"\nvoice: "bm_lewis"\nwake_word: "Hey HAL"\n'
        'sleep_word: "Goodbye, HAL"\ndescription: |\n  Calm.\n'
    )
    monkeypatch.setattr(personas, "BUILTIN_DIR", builtin)
    monkeypatch.setattr(personas, "USER_DIR", user)
    monkeypatch.delenv("IRIS_PERSONAS_DIR", raising=False)
    saved = dict(personas.PERSONAS)
    personas.reload_personas()
    yield builtin, user
    personas.PERSONAS.clear()
    personas.PERSONAS.update(saved)


@pytest.fixture
def store():
    return MemoryStore(":memory:")


@patch("iris.llm.ollama")
def _conversation(store, mock_ollama, tools=None):
    mock_ollama.show.return_value.modelinfo = {}
    keeper = MemoryKeeper(store, lambda messages, schema: {})
    return ConversationManager("fake-llm", tools=tools, memory=keeper)


def _exchange(store, thread, user, assistant):
    store.add_turn(thread, "user", user, "voice")
    store.add_turn(thread, "assistant", assistant)


# -- the store ---------------------------------------------------------------


def test_a_thread_is_stored_with_its_first_turn(store):
    conversation = _conversation(store)
    stored = []
    conversation.on_stored = lambda: stored.append(conversation.thread)
    # A new thread: nothing said yet for a note to be about
    conversation.add_note("You are now HAL.")
    assert store.threads() == [] and conversation.thread is None

    conversation._record("Plan a Lisbon trip", "Alfama, Belém, Sintra.", "text")
    [thread] = store.threads()
    assert thread["persona"] == "iris" and conversation.thread == thread["id"]
    assert stored == [thread["id"]]
    assert [t["text"] for t in store.recent_turns(thread["id"])] == [
        "Plan a Lisbon trip",
        "Alfama, Belém, Sintra.",
    ]
    # The next exchange goes on in it
    conversation._record("And Sintra?", "Take the train.", "voice")
    assert len(store.threads()) == 1


def test_amnesia_in_a_new_thread_stores_no_thread(store):
    conversation = _conversation(store)
    conversation.set_amnesia(True)
    conversation._record("Something private", "Understood.", "voice")
    assert store.threads() == [] and conversation.thread is None


def test_each_thread_has_its_own_index_live_turns_search_and_recall(store):
    lisbon, garden = store.create_thread("iris"), store.create_thread("hal")
    trip = store.create_topic(lisbon, "Lisbon trip", "late Oct, Sintra", "Trip in late October.")
    tomatoes = store.create_topic(garden, "Tomatoes", "after the frost", "Plant after the frost.")
    _exchange(store, lisbon, "Where to stay in Lisbon?", "Chiado, near Rossio.")
    _exchange(store, garden, "When do tomatoes go out?", "After the last frost.")
    store.mark_filed(garden, [t["id"] for t in store.recent_turns(garden)], tomatoes)

    conversation = _conversation(store)
    conversation.open_thread(lisbon)
    messages = conversation.manage_context()
    assert "Lisbon trip" in messages[0]["content"]
    assert "Tomatoes" not in messages[0]["content"]
    assert [m["content"] for m in messages[1:]] == [
        "Where to stay in Lisbon?",
        "Chiado, near Rossio.",
    ]

    tools = Tools(
        [memory_plugin.plugin], PluginContext(memory=store, thread=lambda: conversation.thread)
    )
    assert tools.execute("recall", topic_id=trip)["success"] is True
    assert tools.execute("recall", topic_id=tomatoes)["success"] is False
    assert "Nothing in memory" in tools.execute("search_memory", query="frost")["result"]
    assert "Rossio" in tools.execute("search_memory", query="Rossio")["result"]

    # A new thread remembers nothing yet
    conversation.open_thread(None)
    assert tools.execute("recall", topic_id=trip)["success"] is False
    assert "Nothing in memory" in tools.execute("search_memory", query="Rossio")["result"]
    assert "(nothing yet)" in conversation.manage_context()[0]["content"]


def test_filing_files_each_exchange_into_topics_of_its_own_thread(store):
    lisbon, garden = store.create_thread("iris"), store.create_thread("hal")
    store.create_topic(garden, "Tomatoes", "after the frost", "Plant after the frost.")
    # Said in turn, in two threads
    _exchange(store, lisbon, "Plan a Lisbon trip", "Alfama, Belém, Sintra.")
    _exchange(store, garden, "And the basil?", "Also after the frost.")
    _exchange(store, lisbon, "How do I get to Sintra?", "Train from Rossio.")
    chat = ScriptedChat(
        decisions=[
            {"remember": True, "topic_id": 0, "title": "Lisbon trip"},
            {"remember": True, "topic_id": 2, "title": "Sintra"},
            {"remember": True, "topic_id": 1, "title": "Basil"},
        ],
        summaries=[
            {"summary": "Basil after the frost too.", "gist": "frost"},
            {"summary": "Trip with Sintra by train.", "gist": "Sintra"},
        ],
    )
    assert Filer(store, chat).file() is True

    # Lisbon's decisions saw Lisbon's topics, and what was said in Lisbon
    # just before; the garden's only the garden's
    assert "(none yet)" in chat.prompts[0]
    assert "#2 Lisbon trip" in chat.prompts[1] and "Tomatoes" not in chat.prompts[1]
    assert "Said just before:\nUser: Plan a Lisbon trip" in chat.prompts[1]
    assert "#1 Tomatoes" in chat.prompts[2] and "Lisbon" not in chat.prompts[2]
    assert "Said just before" not in chat.prompts[2]
    assert [t.title for t in store.topics(lisbon)] == ["Lisbon trip"]
    assert len(store.topic(2).turns) == 4
    assert [t.title for t in store.topics(garden)] == ["Tomatoes"]
    assert store.unfiled_exchanges() == []


def test_a_thread_is_titled_by_its_first_topic_else_the_user_s_first_words(store):
    thread = store.create_thread("iris")
    _exchange(
        store,
        thread,
        "What would be a good plan for three days in Lisbon at the end of October?",
        "Alfama, Belém and Sintra.",
    )
    assert store.thread(thread)["title"] == "What would be a good plan for three days in…"
    store.create_topic(thread, "Lisbon trip", "late Oct", "Trip in late October.")
    store.create_topic(thread, "Flights", "Friday", "Arriving Friday.")
    assert store.thread(thread)["title"] == "Lisbon trip"

    # Nothing of Iris's own titles a thread: until the user has said
    # anything, it has no title (the app calls it a new thread)
    greeted = store.create_thread("iris")
    store.add_turn(greeted, "system", "Greet the user.")
    store.add_turn(greeted, "assistant", "Hello! I'm here whenever you want to talk.")
    assert store.thread(greeted)["title"] == ""
    store.add_turn(greeted, "user", "Thanks, what's the weather?", "voice")
    assert store.thread(greeted)["title"] == "Thanks, what's the weather?"


def test_threads_are_listed_the_one_talked_in_last_first(store, monkeypatch):
    clock = iter(range(1000, 2000))
    monkeypatch.setattr("iris.memory.time.time", lambda: next(clock))
    first, second = store.create_thread("iris"), store.create_thread("hal")
    _exchange(store, second, "Hi HAL", "Good evening.")
    _exchange(store, first, "Hi Iris", "Hello!")
    assert [t["id"] for t in store.threads()] == [first, second]
    assert store.last_thread()["id"] == first


def test_a_forgotten_thread_leaves_nothing_and_filing_that_read_it_writes_nothing(store):
    kept, forgotten = store.create_thread("iris"), store.create_thread("hal")
    _exchange(store, kept, "Plan a Lisbon trip", "Sure.")
    _exchange(store, forgotten, "A secret for the party", "My lips are sealed.")
    topic = store.create_topic(forgotten, "Party", "a secret", "Surprise party.")
    exchanges = store.unfiled_exchanges()

    store.forget_thread(forgotten)
    assert store.thread(forgotten) is None and store.topic(topic) is None
    assert store.recent_turns(forgotten) == [] and store.search(forgotten, "secret") == []
    assert [t["id"] for t in store.threads()] == [kept]

    # Filing that read the forgotten thread before writes nothing into it
    stale = next(e for e in exchanges if e.thread_id == forgotten)
    assert store.create_topic(forgotten, "Party", "a secret", "") is None
    store.mark_filed(forgotten, stale.turn_ids, None)
    assert len(store.unfiled_exchanges()) == 1
    # Nor can a new thread ever take its number
    assert store.create_thread("iris") > forgotten


def test_a_database_from_before_threads_becomes_the_first_thread_with_iris(tmp_path):
    path = tmp_path / "memory.db"
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE topics (id INTEGER PRIMARY KEY, title TEXT NOT NULL, gist TEXT NOT NULL, "
        "summary TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);"
        "CREATE TABLE turns (id INTEGER PRIMARY KEY, ts REAL NOT NULL, role TEXT NOT NULL, "
        "text TEXT NOT NULL, source TEXT, topic_id INTEGER, filed INTEGER NOT NULL DEFAULT 0, "
        "summarized INTEGER NOT NULL DEFAULT 1);"
        "CREATE VIRTUAL TABLE turns_fts USING fts5(text, content='turns', content_rowid='id');"
        "CREATE TRIGGER turns_ai AFTER INSERT ON turns BEGIN "
        "INSERT INTO turns_fts(rowid, text) VALUES (new.id, new.text); END;"
        "INSERT INTO topics VALUES (1, 'Lisbon trip', 'late Oct', 'Trip in late October.', 5, 6);"
        "INSERT INTO turns (ts, role, text, source, topic_id, filed) "
        "VALUES (5, 'user', 'Plan a Lisbon trip', 'voice', 1, 1);"
        "INSERT INTO turns (ts, role, text, topic_id, filed) "
        "VALUES (6, 'assistant', 'Sure.', 1, 1);"
        "INSERT INTO turns (ts, role, text) VALUES (9, 'user', 'And the weather?');"
    )
    db.commit()
    db.close()

    store = MemoryStore(path)
    [thread] = store.threads()
    assert (thread["persona"], thread["created"], thread["updated"]) == ("iris", 5, 9)
    assert thread["title"] == "Lisbon trip"
    assert len(store.recent_turns(thread["id"])) == 3
    assert [t.title for t in store.topics(thread["id"])] == ["Lisbon trip"]
    assert store.search(thread["id"], "weather")
    assert len(store.unfiled_exchanges()) == 1

    # Opened again, it stays one thread
    assert len(MemoryStore(path).threads()) == 1


def test_an_empty_database_from_before_threads_gets_no_thread(tmp_path):
    path = tmp_path / "memory.db"
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE turns (id INTEGER PRIMARY KEY, ts REAL NOT NULL, role TEXT NOT NULL, "
        "text TEXT NOT NULL, source TEXT, topic_id INTEGER, filed INTEGER NOT NULL DEFAULT 0);"
    )
    db.close()
    assert MemoryStore(path).threads() == []


# -- opening threads ---------------------------------------------------------


class ThreadAssistant(FakeAssistant):
    """A running assistant with a real conversation and memory, whose
    persona can be switched."""

    def __init__(self, store, wake_word=None):
        persona = personas.get_persona("iris")
        super().__init__(wake_word=wake_word, sleep_word=persona.sleep_word or None)
        self.persona = persona
        self.memory = MemoryKeeper(store, lambda messages, schema: {})
        with patch("iris.llm.ollama") as mock_ollama:
            mock_ollama.show.return_value.modelinfo = {}
            self.conversation = ConversationManager("fake-llm", persona=persona, memory=self.memory)
        self.tts_generator = SpeechGenerator.__new__(SpeechGenerator)
        self.tts_generator.engine, self.tts_generator.voice = persona.engine, persona.voice
        self.tts_generator.speed = persona.speed
        self.tts_generator.prepare = lambda: None
        self.tts_generator.speaks = lambda language: True
        self.player = AudioPlayer()

    def config_info(self):
        return {"persona": self.persona.name, "thread": self.conversation.thread}


class Shown:
    def __init__(self, bus):
        self.events = []
        bus.subscribe(lambda kind, data: self.events.append((kind, dict(data))))

    def of(self, kind):
        return [data for k, data in self.events if k == kind]


def _controller(store, wake_word=None):
    bus = EventBus()
    controller = AssistantController(bus)
    controller.assistant = ThreadAssistant(store, wake_word)
    return controller, Shown(bus)


def _two_threads(store):
    iris = store.create_thread("iris")
    _exchange(store, iris, "Plan a Lisbon trip", "Alfama, Belém, Sintra.")
    hal = store.create_thread("hal")
    _exchange(store, hal, "Open the pod bay doors", "I'm sorry, Dave.")
    return iris, hal


def test_opening_a_thread_takes_on_its_persona_its_voice_and_its_words(dirs, store):
    iris, hal = _two_threads(store)
    controller, shown = _controller(store, wake_word="Iris")
    assistant = controller.assistant
    assistant.conversation.open_thread(iris)

    assert controller.open_thread(hal)["ok"] is True
    assert assistant.persona.id == "hal" and assistant.conversation.persona.id == "hal"
    assert (assistant.tts_generator.engine, assistant.tts_generator.voice) == ("kokoro", "bm_lewis")
    assert assistant.wake_word == "Hey HAL" and assistant.sleep_word == "Goodbye, HAL"
    assert assistant.conversation.thread == hal
    # No note: what the thread holds was said by HAL himself
    assert [t["role"] for t in store.recent_turns(hal)] == ["user", "assistant"]
    # The app is shown the thread's turns, and they are what a reload shows
    assert shown.of("thread.opened") == [{"id": hal}]
    restored = [d["text"] for d in shown.of("chat.user") + shown.of("chat.done")]
    assert restored == ["Open the pod bay doors", "I'm sorry, Dave."]
    assert [e["data"]["text"] for e in controller.history()["events"]] == [
        "Open the pod bay doors",
        "I'm sorry, Dave.",
    ]

    controller.open_thread(iris)
    assert assistant.persona.id == "iris" and assistant.wake_word == "Iris"
    assert assistant.sleep_word is None
    assert "Lisbon" in str(assistant.conversation.manage_context())
    assert "pod bay" not in str(assistant.conversation.manage_context())


def test_a_new_thread_is_stored_with_the_persona_chosen_for_it(dirs, store):
    iris, _ = _two_threads(store)
    controller, shown = _controller(store)
    conversation = controller.assistant.conversation
    conversation.open_thread(iris)

    assert controller.new_thread("hal")["ok"] is True
    assert conversation.thread is None and controller.assistant.persona.id == "hal"
    assert shown.of("thread.opened") == [{"id": None}]
    assert controller.new_thread("nobody")["ok"] is False

    conversation._record("Hello HAL", "Good afternoon.", "text")
    assert store.thread(conversation.thread)["persona"] == "hal"
    assert controller.threads_list()["open"] == conversation.thread


def test_amnesia_s_held_back_turns_are_dropped_when_the_thread_changes(dirs, store):
    iris, hal = _two_threads(store)
    controller, _ = _controller(store)
    conversation = controller.assistant.conversation
    conversation.open_thread(iris)
    conversation.set_amnesia(True)
    conversation._record("Something private", "Understood.", "voice")
    assert "Something private" in str(conversation.manage_context())

    controller.open_thread(hal)
    controller.open_thread(iris)
    assert "Something private" not in str(conversation.manage_context())
    assert conversation.amnesia is True  # one switch for every thread
    assert store.search(iris, "private") == []


def test_opening_a_thread_stops_a_reply_and_keeps_it_where_it_was_said(dirs, store):
    iris, hal = _two_threads(store)
    controller, _ = _controller(store)
    assistant = controller.assistant
    assistant.conversation.open_thread(iris)
    assistant._turn_active.set()

    def reply():
        # Stopped, the reply keeps what was said of it, then ends
        assistant.interrupt_event.wait(5)
        time.sleep(0.1)
        assistant.conversation._record("Tell me about Sintra", "Sintra has palaces", "text")
        assistant._turn_active.clear()

    underway = threading.Thread(target=reply)
    underway.start()
    controller._turns.put({"text": "And Porto?", "id": "t1"})
    controller.open_thread(hal)
    underway.join()

    assert store.recent_turns(iris)[-1]["text"] == "Sintra has palaces"
    assert "Sintra" not in str(store.recent_turns(hal))
    # The message waiting for the reply was meant for the thread left
    assert controller._turns.empty()


def test_forgetting_the_open_thread_leaves_a_new_one_with_the_same_persona(dirs, store):
    iris, hal = _two_threads(store)
    controller, shown = _controller(store)
    assistant = controller.assistant
    controller.open_thread(hal)

    assert controller.forget_thread(iris)["ok"] is True
    assert assistant.conversation.thread == hal

    assert controller.forget_thread(hal)["ok"] is True
    assert store.threads() == []
    assert assistant.conversation.thread is None and assistant.persona.id == "hal"
    assert assistant.conversation.manage_context()[1:] == []
    assert shown.of("thread.opened")[-1] == {"id": None}
    assert {"memory.updated", "threads.changed"} <= {kind for kind, _ in shown.events}


def test_a_thread_whose_persona_is_gone_opens_with_iris(dirs, store):
    _, hal = _two_threads(store)
    gone = store.create_thread("klaus")
    _exchange(store, gone, "Hi Klaus", "Moin.")
    controller, _ = _controller(store)
    controller.open_thread(hal)

    controller.open_thread(gone)
    assert controller.assistant.persona.id == "iris"
    listed = {t["id"]: t for t in controller.threads_list()["threads"]}
    assert (listed[gone]["persona"], listed[gone]["persona_name"]) == ("iris", "Iris")
    assert (listed[hal]["persona"], listed[hal]["persona_name"]) == ("hal", "HAL")
    assert controller.open_thread(999)["ok"] is False


def test_iris_starts_in_the_thread_asked_for_else_the_one_talked_in_last(dirs, store):
    iris, hal = _two_threads(store)
    controller, _ = _controller(store)
    conversation = controller.assistant.conversation

    controller._enter_first_thread(LAST_THREAD)
    assert conversation.thread == hal and controller.assistant.persona.id == "hal"

    controller._enter_first_thread(iris)
    assert conversation.thread == iris and controller.assistant.persona.id == "iris"

    # One forgotten since is the one talked in last
    controller._enter_first_thread(999)
    assert conversation.thread == hal

    # None: a new thread, with the persona Iris started with
    controller, _ = _controller(store)
    controller._enter_first_thread(None)
    assert controller.assistant.conversation.thread is None
    assert controller.assistant.persona.id == "iris"


def test_a_first_start_has_no_thread_until_something_is_said(dirs, store):
    controller, _ = _controller(store)
    controller._enter_first_thread(LAST_THREAD)
    assert controller.assistant.conversation.thread is None
    assert controller.threads_list() == {"threads": [], "open": None}


def test_the_memory_panel_shows_the_open_thread_s_topics(dirs, store):
    iris, hal = _two_threads(store)
    trip = store.create_topic(iris, "Lisbon trip", "late Oct", "Trip in late October.")
    store.create_topic(hal, "Pod bay doors", "closed", "They stay closed.")
    controller, _ = _controller(store)
    controller.open_thread(iris)

    assert [t["title"] for t in controller.memory_topics()["topics"]] == ["Lisbon trip"]
    assert controller.memory_topic(trip)["ok"] is True
    controller.open_thread(hal)
    assert [t["title"] for t in controller.memory_topics()["topics"]] == ["Pod bay doors"]
    assert controller.memory_topic(trip)["ok"] is False


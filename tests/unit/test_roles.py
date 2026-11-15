"""Unit tests for roles: a focus a thread can be given. The shipped roles
load, the user's are saved, a shipped one edited is the user's copy until it
is reset, and the user's own can be deleted. The focus follows the persona in
the prompt, a thread keeps the role it was created with, an edited focus
applies from the next turn, and a deleted role leaves its threads without
one."""

import re
import sqlite3
from unittest.mock import patch

import yaml

from iris import roles
from iris.controller import AssistantController
from iris.events import EventBus
from iris.llm import ConversationManager
from iris.memory import MemoryKeeper, MemoryStore
from iris.personas import get_persona
from iris.plugin import PluginContext
from iris.plugins import BUILT_IN
from iris.roles import find_role, list_roles, save_role
from iris.tools import Tools
from tests.unit.test_threads import _controller, _exchange, dirs  # noqa: F401  (a fixture)


@patch("iris.llm.ollama")
def _conversation(store, mock_ollama, tools=None):
    mock_ollama.show.return_value.modelinfo = {}
    keeper = MemoryKeeper(store, lambda messages, schema: {})
    return ConversationManager("fake-llm", persona=get_persona("iris"), tools=tools, memory=keeper)


# -- the roles -----------------------------------------------------------------


def test_iris_ships_two_roles_whose_focus_describes_a_manner():
    shipped = {role.id: role for role in list_roles()}
    assert set(shipped) == {"research", "language-partner"}
    assert [role.name for role in list_roles()] == ["Language partner", "Research"]
    for role in shipped.values():
        assert role.builtin and role.focus.startswith("You")
        # A manner, never a count of sentences or words
        limit = r"\b(\d+|one|two|three|few) (short )?(sentences?|words?)\b"
        assert not re.search(limit, role.focus)


def test_a_new_role_is_saved_to_the_user_s_folder():
    role_id = save_role({"name": "Gardener", "focus": "You help with the garden."})
    assert role_id == "gardener"
    saved = yaml.safe_load((roles.USER_DIR / "gardener.yaml").read_text())
    assert saved == {"name": "Gardener", "focus": "You help with the garden.\n"}
    assert find_role("gardener").focus == "You help with the garden."
    # Another of the same name gets a file of its own
    assert save_role({"name": "Gardener", "focus": "Roses."}) == "gardener-2"


def test_a_shipped_role_edited_is_the_user_s_copy_until_it_is_reset():
    shipped_focus = find_role("research").focus
    save_role({"name": "Research", "focus": "You read papers."}, "research")
    research = find_role("research")
    assert research.focus == "You read papers." and not research.builtin
    assert research.to_dict()["edited"] is True and research.to_dict()["shipped"] is True

    assert roles.delete_role("research") is True  # still there: reset
    assert find_role("research").focus == shipped_focus and find_role("research").builtin
    assert find_role("research").to_dict()["edited"] is False


def test_the_user_s_own_role_can_be_deleted_but_a_shipped_one_cannot():
    save_role({"name": "Gardener", "focus": "You help with the garden."})
    assert roles.delete_role("gardener") is False
    assert find_role("gardener") is None
    assert roles.delete_role("research") is True and find_role("research") is not None


# -- in the prompt -------------------------------------------------------------


def test_the_focus_follows_the_persona_and_comes_before_the_tools_and_memory():
    store = MemoryStore(":memory:")
    tools = Tools(list(BUILT_IN), PluginContext(memory=store))
    conversation = _conversation(store, tools=tools)
    conversation.open_thread(None, "research")
    prompt = conversation.manage_context()[0]["content"]

    persona = prompt.index(get_persona("iris").personality.strip()[:40])
    focus = prompt.index(find_role("research").focus[:40])
    assert "Your role in this conversation: Research." in prompt
    assert persona < focus < prompt.index("Memory index:")
    guidance = tools.guidance()
    if guidance:
        assert focus < prompt.index(guidance.strip()[:40])

    # Without a role there is no focus
    conversation.open_thread(None)
    assert "Your role in this conversation" not in conversation.manage_context()[0]["content"]


def test_an_edited_focus_applies_from_the_next_turn():
    conversation = _conversation(MemoryStore(":memory:"))
    conversation.open_thread(None, "research")
    save_role({"name": "Research", "focus": "You only read papers."}, "research")
    assert "You only read papers." in conversation.manage_context()[0]["content"]


# -- threads and their roles ---------------------------------------------------


def test_a_thread_created_with_a_role_keeps_it(dirs):  # noqa: F811
    store = MemoryStore(":memory:")
    controller, _ = _controller(store)
    conversation = controller.assistant.conversation

    assert controller.new_thread("hal", "language-partner")["ok"] is True
    assert conversation.role == "language-partner"
    conversation._record("Parliamo un po'", "Volentieri!", "text")
    talking = conversation.thread
    assert store.thread(talking)["role"] == "language-partner"

    # Another thread, without one, and back: the role comes with the thread
    controller.new_thread("iris")
    assert conversation.role is None
    conversation._record("Hello", "Hi!", "text")
    controller.open_thread(talking)
    assert conversation.role == "language-partner" and controller.assistant.persona.id == "hal"
    listed = {t["id"]: t for t in controller.threads_list()["threads"]}
    assert (listed[talking]["role"], listed[talking]["role_name"]) == ("language-partner", "Language partner")
    assert listed[conversation.thread]["role"] == "language-partner"
    assert controller.new_thread("iris", "nobody")["ok"] is False


def test_forgetting_the_open_thread_keeps_its_persona_and_role_for_the_new_one(dirs):  # noqa: F811
    store = MemoryStore(":memory:")
    controller, _ = _controller(store)
    conversation = controller.assistant.conversation
    controller.new_thread("hal", "research")
    conversation._record("What is a pod bay?", "A dock for pods.", "text")

    controller.forget_thread(conversation.thread)
    assert conversation.thread is None and conversation.role == "research"


def test_a_deleted_role_leaves_its_threads_without_one(dirs):  # noqa: F811
    store = MemoryStore(":memory:")
    save_role({"name": "Gardener", "focus": "You help with the garden."})
    controller, _ = _controller(store)
    conversation = controller.assistant.conversation
    controller.new_thread("iris", "gardener")
    conversation._record("When do tomatoes go out?", "After the frost.", "text")
    garden = conversation.thread

    assert controller.role_delete("gardener")["ok"] is True
    assert store.thread(garden)["role"] is None
    assert conversation.role is None
    assert "help with the garden" not in conversation.manage_context()[0]["content"]
    assert [t for t in controller.roles_list()["roles"] if t["id"] == "gardener"] == []


def test_resetting_a_shipped_role_keeps_it_on_its_threads(dirs):  # noqa: F811
    store = MemoryStore(":memory:")
    controller, _ = _controller(store)
    conversation = controller.assistant.conversation
    controller.new_thread("iris", "research")
    controller.role_save({"id": "research", "name": "Research", "focus": "Papers only."})
    conversation._record("Sources?", "Papers.", "text")
    controller.role_delete("research")
    assert store.thread(conversation.thread)["role"] == "research"
    assert conversation.role == "research"


def test_a_role_iris_no_longer_ships_leaves_its_threads_without_one(dirs):  # noqa: F811
    store = MemoryStore(":memory:")
    thread = store.create_thread("hal", "cook")  # shipped once, gone since
    _exchange(store, thread, "Dinner?", "Soup.")
    controller, _ = _controller(store)
    conversation = controller.assistant.conversation

    controller.open_thread(thread)
    assert "Your role in this conversation" not in conversation.manage_context()[0]["content"]
    listed = {t["id"]: t for t in controller.threads_list()["threads"]}
    assert (listed[thread]["role"], listed[thread]["role_name"]) == (None, "")


def test_the_user_s_copy_of_a_role_iris_no_longer_ships_stays_theirs():
    (roles.USER_DIR / "cook.yaml").write_text("name: Cook\nfocus: You cook Italian.\n")
    roles.reload_roles()
    cook = find_role("cook")
    assert cook.focus == "You cook Italian."
    assert cook.to_dict()["shipped"] is False and cook.to_dict()["edited"] is False
    assert (roles.USER_DIR / "cook.yaml").exists()


def test_a_role_needs_a_name():
    assert AssistantController(EventBus()).role_save({"name": " ", "focus": "x"})["ok"] is False


def test_threads_from_before_roles_get_none(tmp_path):
    path = tmp_path / "memory.db"
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE threads (id INTEGER PRIMARY KEY AUTOINCREMENT, persona TEXT NOT NULL, "
        "created REAL NOT NULL, updated REAL NOT NULL);"
        "INSERT INTO threads (persona, created, updated) VALUES ('hal', 1, 2);"
    )
    db.close()
    store = MemoryStore(path)
    assert store.thread(1)["role"] is None and store.thread(1)["persona"] == "hal"
    _exchange(store, 1, "Hello", "Good evening.")
    assert store.create_thread("iris", "research") == 2

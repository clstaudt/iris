"""Unit tests for the tools of the plugins switched on: switching, what the
model is told, and what the app is shown."""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from iris.assistant import VoiceAssistant
from iris.events import EventBus
from iris.llm import ConversationManager
from iris.personas import get_persona
from iris.plugin import Plugin, Tool
from iris.plugins import BUILT_IN, camera, screen, web
from iris.tools import Tools

# The tool part of the system prompt as Iris wrote it before tools came
# from plugins: the built-ins must still produce it to the letter
PROMPTS = json.loads((Path(__file__).parent / "fixtures" / "tool_prompts.json").read_text())


class Echo(Plugin):
    id = "echo"
    name = "Echo"

    def __init__(self, available=True, default=True):
        self._available = available
        self.enabled_by_default = default

    def available(self):
        return self._available

    def tools(self, context):
        return [
            Tool(
                name="echo",
                description="Says it back.",
                run=lambda text="": {"success": True, "result": text},
                parameters={"text": {"type": "string"}},
                required=("text",),
                guidance="- 'echo' function: says it back\n",
                activity="searching",
                label="Echoed",
            )
        ]


class Status:
    def __init__(self):
        self.status_lock = threading.Lock()
        self.seen = []
        self._searching = False

    @property
    def status_searching(self):
        return self._searching

    @status_searching.setter
    def status_searching(self, value):
        self._searching = value
        self.seen.append(value)


def test_plugins_switch_their_tools(monkeypatch):
    monkeypatch.setattr(camera, "HAS_CAMERA", True)
    monkeypatch.setattr(screen, "HAS_SCREEN", True)
    tools = Tools([camera.plugin, screen.plugin])

    assert tools.states()["camera"] == {"enabled": True, "available": True, "name": "Camera"}
    assert tools.set_enabled("camera", False) is True
    assert tools.names() == ["capture_screen"]
    assert tools.states()["camera"]["enabled"] is False

    assert tools.set_enabled("camera", True) is True
    # Offered in plugin order, whatever order they were switched in
    assert tools.names() == ["look", "capture_screen"]


def test_web_covers_search_and_browse():
    if not web.HAS_SEARCH:
        return
    tools = Tools([web.plugin])
    assert tools.names() == ["search", "browse"]
    tools.set_enabled("web", False)
    assert not tools.has_tools()


def test_unavailable_or_unknown_plugins_refuse(monkeypatch):
    monkeypatch.setattr(screen, "HAS_SCREEN", False)
    tools = Tools([screen.plugin])
    assert tools.set_enabled("screen", True) is False
    assert tools.states()["screen"] == {"enabled": False, "available": False, "name": "Screen"}
    assert tools.set_enabled("teleport", True) is False


def test_choices_override_the_defaults():
    tools = Tools([Echo(default=False)])
    assert not tools.has_tools()
    assert Tools([Echo(default=False)], choices={"echo": True}).names() == ["echo"]
    assert Tools([Echo()], choices={"echo": False}).names() == []


def test_memory_is_not_a_switch():
    memory = next(p for p in BUILT_IN if p.id == "memory")
    tools = Tools([memory])
    assert "memory" not in tools.states()
    assert tools.set_enabled("memory", False) is False
    # Without a memory store it brings no tools
    assert not tools.has_tools()


def test_a_tool_shows_its_activity_while_it_runs():
    status = Status()
    tools = Tools([Echo()], status_manager=status)
    assert tools.execute("echo", text="hi") == {"success": True, "result": "hi"}
    assert status.seen == [True, False]


def test_a_failing_tool_is_a_failed_result_and_clears_its_activity():
    class Broken(Echo):
        def tools(self, context):
            [tool] = super().tools(context)
            tool.run = lambda **_: 1 / 0
            return [tool]

    status = Status()
    result = Tools([Broken()], status_manager=status).execute("echo", text="hi")
    assert result["success"] is False and "division" in result["message"]
    assert status.seen == [True, False]


def test_a_plugin_that_cannot_make_its_tools_stays_off():
    class Failing(Echo):
        def tools(self, context):
            raise RuntimeError("no")

    tools = Tools([Failing()])
    assert tools.states()["echo"]["enabled"] is False


def test_the_transcript_gets_the_label_and_what_it_was_called_on():
    tools = Tools([Echo()])
    assert tools.describe("echo", {"text": "hello"}) == ("Echoed", "hello")
    assert tools.describe("missing", {}) == ("missing", "")


def test_tools_without_guidance_still_get_the_frame():
    tools = Tools([Echo()])
    assert tools.guidance().startswith("\n\nIMPORTANT - You have access to tools")
    assert "- 'echo' function: says it back\n" in tools.guidance()
    assert Tools([]).guidance() == ""


@patch("iris.llm.ollama")
def _prompt(tools, mock_ollama):
    mock_ollama.show.return_value = MagicMock(modelinfo={}, details=MagicMock(family="t"))
    mock_ollama.list.return_value = []
    conversation = ConversationManager(model="t", tools=tools)
    return conversation.manage_context()[0]["content"][len(get_persona("iris").description) :]


def test_the_built_in_tools_tell_the_model_what_they_always_did(monkeypatch):
    monkeypatch.setattr(camera, "HAS_CAMERA", True)
    monkeypatch.setattr(screen, "HAS_SCREEN", True)
    monkeypatch.setattr(web, "HAS_SEARCH", True)
    monkeypatch.setattr(web, "Pages", MagicMock())
    tools = Tools([camera.plugin, screen.plugin, web.plugin])
    expected_prompt, expected_names = PROMPTS["all"]
    assert tools.names() == expected_names
    assert _prompt(tools) == expected_prompt

    only_web = Tools([web.plugin])
    assert _prompt(only_web) == PROMPTS["web"][0]


def test_the_app_is_told_what_a_tool_did():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda kind, data: seen.append((kind, data)))

    def chat_stream(prompt, on_tool_use=None, **kwargs):
        on_tool_use("echo", {"text": "hello"})
        yield "Done."

    assistant = object.__new__(VoiceAssistant)
    assistant.tools = Tools([Echo()])
    assistant.conversation = MagicMock(chat_stream=chat_stream, reply="Done.")
    assistant.status_manager = MagicMock()
    assistant.speech_pipeline = MagicMock()
    assistant.event_bus = bus
    assistant.memory = None
    assistant.speak_replies = False
    assistant.suspend_event = threading.Event()
    assistant.interrupt_event = threading.Event()

    assistant.chat("Say hello", source="text")

    assert ("tool.use", {"tool": "echo", "label": "Echoed", "detail": "hello"}) in seen

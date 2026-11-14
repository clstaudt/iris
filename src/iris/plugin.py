"""What a plugin is: tools for the language model, and engines for the
model slots (see iris.engines).

A plugin is a module that exposes `plugin`, an instance of a Plugin
subclass. Iris loads the plugins it ships (iris.plugins), those of
installed packages (entry point group "iris.plugins") and the modules in
~/.iris/plugins (or IRIS_PLUGINS_DIR). A minimal one, in
~/.iris/plugins/clock.py:

    import time

    from iris.plugin import Plugin, Tool


    class Clock(Plugin):
        id = "clock"
        name = "Clock"

        def tools(self, context):
            return [
                Tool(
                    name="current_time",
                    description="The current local time.",
                    run=lambda **_: {"success": True, "result": time.strftime("%H:%M")},
                    label="Looked at the clock",
                )
            ]


    plugin = Clock()

Its tools then show as a switch under Settings > Tools, and the model can
call them.
"""

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from iris.engines import RecognitionEngine, SpeechEngine
from iris.memory import MemoryStore


@dataclass
class Tool:
    """A function the language model can call.

    run(**arguments) returns a dict with "success" (bool) and one of
    "result" (text for the model), "image" (a base64 JPEG the model is
    shown; a model that can see answers) or "message". A failed call's
    "message" is passed on to the model as the error. "notice" is told the
    user as well, shown as an error line: what only they can fix, such as a
    permission to grant, should not depend on the model passing it on.
    "pin_topic" (a memory topic id) keeps that topic in context for the
    next few turns.
    "ends_reply" (True) makes the call the end of the reply: the model is
    not asked to go on, unless the reply has said nothing yet.
    """

    name: str
    description: str
    run: Callable[..., dict]
    #: JSON schema of each parameter, by name
    parameters: dict[str, dict] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    #: How to use it, added to the system prompt while the tool is on
    guidance: str = ""
    #: What the app shows while it runs: "looking", "capturing_screen",
    #: "searching" or "browsing" (see iris.status); None keeps "thinking"
    activity: Optional[str] = None
    #: What the transcript says it did, e.g. "Searched the web"
    label: str = ""
    #: What it was called on, for the transcript; by default the first
    #: text argument (a query, a URL)
    detail: Optional[Callable[[dict], str]] = None
    #: Whether it is offered to the model now, asked on every turn; None
    #: for always while its plugin is on
    offered: Optional[Callable[[], bool]] = None

    def schema(self) -> dict:
        """The tool as offered to the model (Ollama's function format)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.required),
                },
            },
        }

    def describe(self, arguments: dict) -> str:
        """What the tool was called on, for the transcript."""
        if self.detail is not None:
            return self.detail(arguments)
        return next((v for v in arguments.values() if isinstance(v, str) and v), "")


@dataclass
class PluginContext:
    """What Iris shares with plugins when they make their tools."""

    #: Long-term memory, when it is on
    memory: Optional[MemoryStore] = None
    #: Ends the conversation: once the reply being given has been spoken,
    #: Iris goes to sleep as its sleep word puts it. None where there is no
    #: conversation to end
    sleep: Optional[Callable[[], None]] = None
    #: Whether the model may put Iris to sleep now: only while the wake
    #: word is on
    can_sleep: Callable[[], bool] = lambda: False


class Plugin:
    """A capability Iris can have: tools for the model, engines, or both.

    Subclass it, set id and name, and override what it brings.
    """

    #: Unique key, also the key of its switch in the app's settings
    id = ""
    #: Shown in the app
    name = ""
    #: Modules it needs; it is "not installed" while one is missing
    requires: tuple[str, ...] = ()
    #: Its tools can be switched on and off in the app
    switchable = True
    #: Its tools are on until the user switches them off
    enabled_by_default = True
    #: Where it was found: "built-in", "package <name>" or its file; set
    #: when it is loaded
    source = "built-in"

    def available(self) -> bool:
        """Whether what it needs is installed."""
        return all(importlib.util.find_spec(module) is not None for module in self.requires)

    def tools(self, context: PluginContext) -> list[Tool]:
        """Its tools, made when it is switched on."""
        return []

    def speech_engines(self) -> list[type[SpeechEngine]]:
        """Speech engines it adds, registered when it is loaded."""
        return []

    def recognition_engines(self) -> list[RecognitionEngine]:
        """Speech recognition engines it adds, registered when it is loaded."""
        return []

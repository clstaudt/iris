"""The tools the language model can call: those of the plugins switched on."""

import logging
from typing import Optional

from iris.plugin import Plugin, PluginContext, Tool
from iris.status import STATUS_FLAG_NAMES

logger = logging.getLogger(__name__)

# The tool section of the system prompt opens and closes with these; each
# tool's guidance goes in between
GUIDANCE_START = "\n\nIMPORTANT - You have access to tools that you must use appropriately:\n"
GUIDANCE_END = (
    "\nIMPORTANT: You MUST use these functions when appropriate. "
    "NEVER make up or guess what you see."
)


class Tools:
    """The plugins' tools, switched on and off by plugin.

    The schemas and the system prompt are rebuilt from the tools on every
    turn, so a switch applies from the next message.
    """

    def __init__(
        self,
        plugins: list[Plugin],
        context: Optional[PluginContext] = None,
        status_manager=None,
        choices: Optional[dict[str, bool]] = None,
    ):
        """
        Args:
            plugins: The plugins whose tools are offered, in offering order
            context: What the plugins get to make their tools
            status_manager: Shows each tool's activity while it runs
            choices: {plugin id: on} chosen by the user; a plugin not in it
                follows its enabled_by_default
        """
        self.status_manager = status_manager
        self._context = context or PluginContext()
        self._plugins = {plugin.id: plugin for plugin in plugins}
        # The tools of each plugin switched on
        self._tools: dict[str, list[Tool]] = {}
        choices = choices or {}
        for plugin in plugins:
            if not plugin.switchable or choices.get(plugin.id, plugin.enabled_by_default):
                self._switch_on(plugin)

    def _switch_on(self, plugin: Plugin) -> bool:
        if not plugin.available():
            return False
        try:
            self._tools[plugin.id] = list(plugin.tools(self._context))
        except Exception:
            logger.exception("Plugin %s could not make its tools", plugin.id)
            return False
        return True

    def set_enabled(self, plugin_id: str, enabled: bool) -> bool:
        """Switch a plugin's tools on or off.

        Returns:
            False if the plugin is unknown, cannot be switched, or what it
            needs is not installed
        """
        plugin = self._plugins.get(plugin_id)
        if plugin is None or not plugin.switchable:
            return False
        if not enabled:
            self._tools.pop(plugin_id, None)
            return True
        return plugin_id in self._tools or self._switch_on(plugin)

    def states(self) -> dict[str, dict]:
        """The switchable plugins as {id: {"enabled", "available", "name"}}."""
        return {
            plugin.id: {
                "enabled": plugin.id in self._tools,
                "available": plugin.available(),
                "name": plugin.name,
            }
            for plugin in self._plugins.values()
            if plugin.switchable
        }

    def enabled(self, plugin_id: str) -> bool:
        """Whether the plugin is on and brings tools."""
        return bool(self._tools.get(plugin_id))

    def _enabled_tools(self) -> list[Tool]:
        return [tool for pid in self._plugins if pid in self._tools for tool in self._tools[pid]]

    def tool(self, name: str) -> Optional[Tool]:
        """The tool of this name, if it is on."""
        return next((tool for tool in self._enabled_tools() if tool.name == name), None)

    def names(self) -> list[str]:
        return [tool.name for tool in self._enabled_tools()]

    def names_of(self, plugin_id: str) -> list[str]:
        """The names of a plugin's tools, while they are on."""
        return [tool.name for tool in self._tools.get(plugin_id, [])]

    def get_schemas(self) -> list[dict]:
        """The tools on, as offered to the model."""
        return [tool.schema() for tool in self._enabled_tools()]

    def has_tools(self) -> bool:
        return bool(self._enabled_tools())

    def guidance(self) -> str:
        """The tool section of the system prompt, or "" without tools."""
        tools = self._enabled_tools()
        if not tools:
            return ""
        return GUIDANCE_START + "".join(tool.guidance for tool in tools) + GUIDANCE_END

    def execute(self, tool_name: str, **kwargs) -> dict:
        """Run a tool, showing its activity meanwhile.

        Returns:
            The tool's result dict; a failure if the tool is not on or raised
        """
        tool = self.tool(tool_name)
        if tool is None:
            return {"success": False, "message": f"Unknown tool: {tool_name}"}
        activity = tool.activity if tool.activity in STATUS_FLAG_NAMES else None
        self._show(activity, True)
        try:
            return tool.run(**kwargs)
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return {"success": False, "message": f"Error: {e}"}
        finally:
            self._show(activity, False)

    def describe(self, tool_name: str, arguments: dict) -> tuple[str, str]:
        """(label, detail) for the transcript: what the tool did, and on what."""
        tool = self.tool(tool_name)
        if tool is None:
            return tool_name, ""
        return tool.label or tool.name, tool.describe(arguments)

    def _show(self, activity: Optional[str], on: bool):
        if activity is None or self.status_manager is None:
            return
        with self.status_manager.status_lock:
            setattr(self.status_manager, f"status_{activity}", on)

    def cleanup(self):
        """Release resources (the built-in tools hold none between calls)."""

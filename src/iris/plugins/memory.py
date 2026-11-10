"""Memory: Iris recalls earlier conversations by topic and searches them.

The topic index in the system prompt tells the model what it remembers;
these tools are how it fetches the details. They come with long-term memory
and cannot be switched off on their own.
"""

from iris.memory import RECALL_TOKENS, MemoryStore, estimate_tokens, relative_day
from iris.plugin import Plugin, PluginContext, Tool


def _topic_id(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


class Recall:
    """The memory tools over one store."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def recall(self, topic_id=None, **kwargs) -> dict:
        topic = self.store.topic(_topic_id(topic_id))
        if topic is None:
            return {"success": False, "message": f"No topic #{topic_id} in memory"}
        # The summary, then as many of the latest exchanges as fit
        excerpts, used = [], 0
        for turn in reversed(topic.turns):
            line = f"{turn['role'].capitalize()} ({relative_day(turn['ts'])}): {turn['text']}"
            if used + estimate_tokens(line) > RECALL_TOKENS:
                break
            excerpts.append(line)
            used += estimate_tokens(line)
        text = (
            f"Memory #{topic.id} {topic.title}\nSummary: {topic.summary}\n\n"
            "Latest exchanges:\n" + "\n".join(reversed(excerpts))
        )
        return {
            "success": True,
            "result": text,
            "topic_id": topic.id,
            "title": topic.title,
            # Recalled, the topic stays in context for the next few turns
            "pin_topic": topic.id,
        }

    def search_memory(self, query: str = "", **kwargs) -> dict:
        hits = self.store.search(query)
        if not hits:
            return {"success": True, "result": f"Nothing in memory matches '{query}'."}
        lines = [
            f"- {relative_day(h['ts'])}, "
            f"{'#' + str(h['topic_id']) + ' ' + h['title'] if h['topic_id'] else 'no topic'}: "
            f"{h['snippet']}"
            for h in hits
        ]
        return {"success": True, "result": f"Memory matches for '{query}':\n" + "\n".join(lines)}

    def topic_title(self, arguments: dict) -> str:
        """The recalled topic's title, for the transcript."""
        topic = self.store.topic(_topic_id(arguments.get("topic_id")))
        return topic.title if topic else ""


class Memory(Plugin):
    id = "memory"
    name = "Memory"
    switchable = False

    def tools(self, context: PluginContext) -> list[Tool]:
        if context.memory is None:
            return []
        memory = Recall(context.memory)
        return [
            Tool(
                name="recall",
                description=(
                    "Load an earlier conversation from memory by its topic number"
                    " (#id in the memory index). Use it whenever the user refers to"
                    " something discussed before that is not in the recent messages."
                ),
                run=memory.recall,
                parameters={"topic_id": {"type": "integer", "description": "Topic number"}},
                required=("topic_id",),
                label="Recalled",
                detail=memory.topic_title,
            ),
            Tool(
                name="search_memory",
                description=(
                    "Search everything ever said in earlier conversations for words,"
                    " names or details that the memory index does not show."
                ),
                run=memory.search_memory,
                parameters={"query": {"type": "string", "description": "Words to look for"}},
                required=("query",),
                label="Searched memory",
            ),
        ]


plugin = Memory()

"""Unit tests for streamed replies: tool rounds, and nothing unspoken."""

from unittest.mock import MagicMock, patch

from iris.llm import MAX_TOOL_ROUNDS, ConversationManager, UnspokenFilter


class FakeTools:
    status_manager = None

    def __init__(self):
        self._enabled_tools = {"search": {}, "browse": {}}
        self.executed = []

    def has_tools(self):
        return True

    def get_schemas(self):
        return [{"type": "function", "function": {"name": "search"}}]

    def execute(self, name, **kwargs):
        self.executed.append(name)
        return {"success": True, "result": f"{name} result"}


def call(name):
    tool_call = {"function": {"name": name, "arguments": {}}}
    return {"message": {"content": "", "tool_calls": [tool_call]}}


def say(text):
    return {"message": {"content": text}}


class FakeClient:
    """Answers each chat request with the next scripted reply."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.offered_tools = []

    def chat(self, model, messages, stream=False, tools=None):
        self.offered_tools.append(tools is not None)
        self.last_messages = messages
        return iter([self.replies.pop(0)])


def _conversation(replies):
    tools = FakeTools()
    with patch("iris.llm.ollama") as mock_ollama:
        mock_ollama.show.return_value = MagicMock(modelinfo={}, details=MagicMock(family="test"))
        mock_ollama.list.return_value = []
        conversation = ConversationManager(model="test", tools=tools, quiet=True)
    conversation.client = FakeClient(replies)
    return conversation, tools


def test_a_search_can_be_followed_by_reading_a_page():
    """Without tools on the follow-up, a model writes its next call out as
    text, and that text would be spoken."""
    conversation, tools = _conversation(
        [call("search"), call("browse"), say("It's 16 degrees and cloudy.")]
    )
    reply = "".join(conversation.chat_stream("Weather in Berlin?"))
    assert reply == "It's 16 degrees and cloudy."
    assert tools.executed == ["search", "browse"]
    assert conversation.client.offered_tools == [True, True, True]


def test_the_last_round_has_to_answer():
    replies = [call("search")] * MAX_TOOL_ROUNDS + [say("Here is what I found.")]
    conversation, tools = _conversation(replies)
    reply = "".join(conversation.chat_stream("Weather in Berlin?"))
    assert reply == "Here is what I found."
    assert len(tools.executed) == MAX_TOOL_ROUNDS
    # The first request and every follow-up but the last offer tools
    assert conversation.client.offered_tools == [True] * MAX_TOOL_ROUNDS + [False]
    # ...and the last one is told so
    assert "No more tools" in conversation.client.last_messages[-1]["content"]


def _filtered(chunks):
    unspoken = UnspokenFilter()
    return "".join(unspoken.feed(c) for c in chunks) + unspoken.flush()


def test_reasoning_and_typed_tool_calls_are_not_spoken():
    chunks = ["<think>the user wants", " weather</think>It's ", "sunny. <tool", "_call>\n",
              "<function=browse>x</function>\n</tool_", "call> Enjoy!"]
    assert _filtered(chunks) == "It's sunny.  Enjoy!"


def test_other_angle_brackets_are_spoken():
    assert _filtered(["If x <", " 3 and y", " <thin", "g> holds"]) == "If x < 3 and y <thing> holds"


def test_an_unclosed_tag_is_dropped():
    assert _filtered(["Let me check. <tool_call>\n<function=browse>"]) == "Let me check. "


def test_a_typed_tool_call_stays_out_of_reply_and_history():
    conversation, _ = _conversation(
        [say("Checking another source. <tool_call><function=browse></function></tool_call>")]
    )
    reply = "".join(conversation.chat_stream("Weather?"))
    assert reply == "Checking another source. "
    assert conversation.conversation_history[-1] == {
        "role": "assistant", "content": "Checking another source.",
    }

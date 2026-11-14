"""Unit tests for streamed replies: tool rounds, and nothing unspoken."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from iris.assistant import VoiceAssistant
from iris.llm import MAX_TOOL_ROUNDS, NO_TOOLS_NOTE, ConversationManager
from iris.markup import MarkupFilter, TypedCall


class FakeTools:
    def __init__(self):
        self.executed = []

    def has_tools(self):
        return True

    def guidance(self):
        return ""

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
        conversation = ConversationManager(model="test", tools=tools)
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
    unspoken = MarkupFilter()
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


class ChunkedClient(FakeClient):
    """Streams each scripted reply chunk by chunk, noting how far it got."""

    def chat(self, model, messages, stream=False, tools=None):
        self.offered_tools.append(tools is not None)
        self.sent = 0

        def chunks(reply):
            for chunk in reply:
                self.sent += 1
                yield chunk

        return chunks(self.replies.pop(0))


def test_a_reply_streams_while_tools_are_on_offer():
    """The voice starts before the model has finished its reply."""
    conversation, _ = _conversation([])
    conversation.client = ChunkedClient([[say("It's "), say("sunny. "), say("Enjoy!")]])
    stream = conversation.chat_stream("Weather?")
    assert next(stream) == "It's "
    assert conversation.client.sent == 1
    assert "".join(stream) == "sunny. Enjoy!"


def test_text_before_a_tool_call_is_said_and_kept():
    conversation, tools = _conversation([])
    conversation.client = ChunkedClient(
        [[say("Let me look that up. "), call("search")], [say("It's sunny.")]]
    )
    reply = "".join(conversation.chat_stream("Weather?"))
    assert reply == "Let me look that up. It's sunny."
    assert tools.executed == ["search"]
    assert conversation.conversation_history[-1]["content"] == "Let me look that up. It's sunny."


# Tool calls written out as text ----------------------------------------------------------

GOOD_NIGHT = "Good night!\n\n<function=end_conversation>\n</function>\n</tool_call>"
SEARCH_CALL = '{"name": "search", "arguments": {"query": "Mars news"}}'
MARS = TypedCall("search", {"query": "Mars news"})

# (reply, what is let through, the calls found)
TYPED_CALLS = [
    (GOOD_NIGHT, "Good night!\n\n\n", [TypedCall("end_conversation", {})]),
    (f"Let me look. <tool_call>{SEARCH_CALL}</tool_call>", "Let me look. ", [MARS]),
    (SEARCH_CALL, "", [MARS]),
    (f"<|tool_call|>[{SEARCH_CALL}]", "", [MARS]),
    (f"<|python_tag|>{SEARCH_CALL}", "", [MARS]),
    (f"[TOOL_CALLS][{SEARCH_CALL}]", "", [MARS]),
    ('[TOOL_CALLS]search[ARGS]{"query": "x"}', "", [TypedCall("search", {"query": "x"})]),
    (f"Sure. ```json\n{SEARCH_CALL}\n``` One moment.", "Sure.  One moment.", [MARS]),
    (
        '<function=browse>{"url": "https://a.example"}</function>',
        "",
        [TypedCall("browse", {"url": "https://a.example"})],
    ),
    (
        "<tool_call>\n<function=search>\n<parameter=query>\nweather\n</parameter>\n</function>\n"
        "</tool_call>",
        "",
        [TypedCall("search", {"query": "weather"})],
    ),
    (
        '{"type": "function", "function": '
        '{"name": "search", "arguments": "{\\"query\\": \\"q\\"}"}}',
        "",
        [TypedCall("search", {"query": "q"})],
    ),
    # Arguments that do not parse, a call that cannot be read
    ('<function=search>{"query": </function>', "", [TypedCall("search", None)]),
    ("<tool_call>search for it</tool_call> Hm.", " Hm.", [TypedCall("", None)]),
    # Left open at the end of the reply
    ('One moment. {"name": "search", "arguments": {"query"', "One moment. ", [TypedCall("", None)]),
    # What is left of markup: a special token, a closing tag
    ("Done.<|im_end|>", "Done.", []),
    ("Done.</tool_call>", "Done.", []),
]

# Text that only looks a little like markup, said as it is
NOT_MARKUP = [
    'The JSON is {"city": "Oslo"} here.',
    '{"name": "Bob", "age": 3}',
    "A list [1, 2], {braces}, `code`, a < b and <thing>.",
    "```python\nprint(1)\n```",
    "[TOOL]",
]


def _streamed(text: str, size: int) -> tuple[str, list]:
    markup = MarkupFilter()
    out = "".join(markup.feed(text[i : i + size]) for i in range(0, len(text), size))
    return out + markup.flush(), markup.calls


def test_typed_tool_calls_are_found_however_the_reply_is_split():
    for text, said, calls in TYPED_CALLS:
        for size in (1, 2, 3, 5, 8, len(text)):
            assert _streamed(text, size) == (said, calls), (text, size)


def test_text_that_only_looks_like_markup_is_said():
    for text in NOT_MARKUP:
        for size in (1, 2, 3, len(text)):
            assert _streamed(text, size) == (text, []), (text, size)


def test_text_before_markup_is_let_through_at_once_and_none_of_the_markup():
    markup = MarkupFilter()
    let_through = [markup.feed(char) for char in GOOD_NIGHT]
    before = len("Good night!\n\n")
    assert "".join(let_through[:before]) == "Good night!\n\n"
    assert "".join(let_through[before:]) + markup.flush() == "\n"


class ToolsOn(FakeTools):
    """Tools on offer by name; the calls run are kept with their arguments."""

    def __init__(self, names=("search", "end_conversation")):
        super().__init__()
        self.names = names
        self.runs = []

    def get_schemas(self):
        return [{"type": "function", "function": {"name": name}} for name in self.names]

    def execute(self, name, **kwargs):
        self.runs.append((name, kwargs))
        return super().execute(name, **kwargs)


class Recording(ChunkedClient):
    """Keeps the messages of every request."""

    def chat(self, model, messages, stream=False, tools=None):
        self.requests = getattr(self, "requests", []) + [messages]
        return super().chat(model, messages, stream=stream, tools=tools)


def _chunks(text: str, size: int = 3) -> list[dict]:
    return [say(text[i : i + size]) for i in range(0, len(text), size)]


def _typed(replies, tools=None):
    conversation, _ = _conversation([])
    conversation.tools = tools or ToolsOn()
    conversation.client = Recording(replies)
    return conversation, conversation.tools


def test_an_unambiguous_typed_call_is_run_as_if_made():
    conversation, tools = _typed(
        [_chunks(f"Let me look. <tool_call>{SEARCH_CALL}</tool_call>"), [say("Mars is red.")]]
    )
    assert "".join(conversation.chat_stream("News about Mars?")) == "Let me look. Mars is red."
    assert tools.runs == [("search", {"query": "Mars news"})]
    # The model sees its call as one it made, with the text it said
    made = next(m for m in conversation.client.requests[-1] if m.get("tool_calls"))
    assert made["content"] == "Let me look. "
    assert made["tool_calls"] == [
        {"function": {"name": "search", "arguments": {"query": "Mars news"}}}
    ]
    assert conversation.conversation_history[-1]["content"] == "Let me look. Mars is red."


def test_the_good_night_with_a_call_of_a_tool_not_offered_is_said_without_it():
    conversation, tools = _typed([_chunks(GOOD_NIGHT, 2)], ToolsOn(names=("search",)))
    assert "".join(conversation.chat_stream("Good night")) == "Good night!\n\n\n"
    assert tools.runs == []
    assert conversation.conversation_history[-1]["content"] == "Good night!"
    assert conversation.client.offered_tools == [True]


def test_a_typed_call_of_an_offered_tool_runs_after_its_text_is_said():
    conversation, tools = _typed([_chunks(GOOD_NIGHT), [say("Sleep well.")]])
    assert "".join(conversation.chat_stream("Good night")) == "Good night!\n\n\nSleep well."
    assert tools.runs == [("end_conversation", {})]


def test_calls_that_are_not_unambiguous_are_left_out():
    for reply in (
        'Checking. <tool_call>{"name": "weather", "arguments": {}}</tool_call>',  # not offered
        'Checking. <function=search>{"query": </function>',  # arguments that do not parse
        "Checking. <tool_call>search for Mars</tool_call>",  # no call to be read
    ):
        conversation, tools = _typed([_chunks(reply)])
        assert "".join(conversation.chat_stream("Mars?")) == "Checking. ", reply
        assert tools.runs == []


def test_a_reply_that_was_only_a_call_it_cannot_make_is_asked_again_in_words():
    conversation, tools = _typed(
        [
            _chunks('<tool_call>{"name": "weather", "arguments": {}}</tool_call>'),
            [say("I cannot check the weather.")],
        ]
    )
    assert "".join(conversation.chat_stream("Weather?")) == "I cannot check the weather."
    assert tools.runs == []
    assert conversation.client.offered_tools == [True, False]
    assert conversation.client.requests[-1][-1] == {
        "role": "user", "content": f"(Note: {NO_TOOLS_NOTE})",
    }


class Voice:
    current_language = "en"

    def __init__(self):
        self.spoken = []

    def begin_reply(self):
        pass

    def queue_speech(self, text, language=None, sound=None):
        self.spoken.append(text)

    def wait_for_speech(self):
        pass


def test_neither_the_voice_nor_the_transcript_gets_any_markup():
    """The voice speaks sentence by sentence as the reply streams in, so
    markup split across chunks must be known as such before any of it
    could make up a sentence."""
    conversation, _ = _typed(
        [_chunks(f"Let me look that up for you. {SEARCH_CALL} Hm.", 1), [say("Mars is red.")]]
    )
    assistant = object.__new__(VoiceAssistant)
    assistant.event_bus = MagicMock()
    assistant.speak_replies = True
    assistant.conversation = conversation
    assistant.speech_pipeline = Voice()
    assistant.tts_generator = SimpleNamespace(languages=lambda: frozenset({"en"}))
    assistant.interrupt_event = threading.Event()
    assistant.suspend_event = threading.Event()
    assistant.tools = MagicMock()
    assistant.tools.describe.return_value = ("Searched the web", "Mars news")
    assistant.status_manager = MagicMock(status_lock=threading.Lock())

    assistant.chat("News about Mars?")
    events = [c.args for c in assistant.event_bus.emit.call_args_list]
    shown = "".join(data["text"] for kind, data in events if kind == "chat.chunk")
    assert shown == "Let me look that up for you.  Hm.Mars is red."
    used = {"tool": "search", "label": "Searched the web", "detail": "Mars news"}
    assert ("tool.use", used) in events
    assert assistant.speech_pipeline.spoken == ["Let me look that up for you.", "Hm.Mars is red."]


def test_without_tools_a_typed_call_is_never_run():
    conversation, tools = _typed([_chunks(SEARCH_CALL), [say("I cannot search.")]])
    tools.has_tools = lambda: False
    assert "".join(conversation.chat_stream("Search Mars")) == "I cannot search."
    assert tools.runs == []
    assert conversation.client.offered_tools == [False, False]

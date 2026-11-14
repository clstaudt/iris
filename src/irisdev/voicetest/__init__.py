"""Testing a voice assistant by talking to it.

The assistant runs as it would on a computer, but its microphone and speaker
belong to a virtual acoustic space (acoustics.Space): what the microphone
hears is scripted, the speaker records what it plays, and each speaker
echoes into its own microphone and is heard by the others. No sound reaches
the computer's speakers, and nothing in the room reaches the test.

- scripted.ScriptedConversations: a synthesized user holds conversations
  (conversations.CONVERSATIONS) with the assistant: questions and
  follow-ups, pauses in a sentence, interrupting it, Stop, a wake word. Each
  turn is checked (heard as said, one turn, a reply that fits) and timed
  from the end of the user's speech to the first word heard.
- selftalk.SelfTalk: two assistants talk to each other, unscripted.
- assistant.IrisUnderTest runs Iris for either; another assistant is tested
  by implementing assistant.AssistantUnderTest.
- stub_llm.StubModel is a scripted language model behind Ollama's API, for
  testing the voice without a model's time and whims; its rules make it
  call tools (a web search, reading a page) as a model with tools would.
- fakeweb.FakeWeb is a scripted web for Iris's searches and page reads.
- fakesight.FakeCamera and FakeScreen are a scripted camera and screen.
- script.py is what conversations are written in (turns, the moments things
  happen at, asides, actions, what is expected); conversations.GROUPS holds
  them by what they test.

    python -m irisdev.voicetest conversations [--group search] [--llm-host URL --llm-model NAME]
    python -m irisdev.voicetest selftalk --llm-host URL --llm-model NAME
"""

from irisdev.voicetest.acoustics import Endpoint, Space
from irisdev.voicetest.assistant import AssistantUnderTest, IrisUnderTest, finished, wait_until
from irisdev.voicetest.conversations import CONVERSATIONS, GROUPS, group
from irisdev.voicetest.fakesight import FakeCamera, FakeScreen
from irisdev.voicetest.fakeweb import FakeWeb
from irisdev.voicetest.script import At, Conversation, Say, Silence, Sound, Turn
from irisdev.voicetest.scripted import BUDGETS, ScriptedConversations, TurnResult, summarize
from irisdev.voicetest.selftalk import SelfTalk, SelfTalkResult
from irisdev.voicetest.stub_llm import Rule, StubModel
from irisdev.voicetest.voices import VOICES, UserVoice

__all__ = [
    "AssistantUnderTest",
    "At",
    "BUDGETS",
    "CONVERSATIONS",
    "Conversation",
    "Endpoint",
    "FakeCamera",
    "FakeScreen",
    "FakeWeb",
    "GROUPS",
    "IrisUnderTest",
    "Rule",
    "Say",
    "ScriptedConversations",
    "SelfTalk",
    "SelfTalkResult",
    "Silence",
    "Sound",
    "Space",
    "StubModel",
    "Turn",
    "TurnResult",
    "UserVoice",
    "VOICES",
    "finished",
    "group",
    "summarize",
    "wait_until",
]

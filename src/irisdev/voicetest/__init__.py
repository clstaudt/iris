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
  testing the voice without a model's time and whims.

    python -m irisdev.voicetest conversations [--llm-host URL --llm-model NAME]
    python -m irisdev.voicetest selftalk --llm-host URL --llm-model NAME
"""

from irisdev.voicetest.acoustics import Endpoint, Space
from irisdev.voicetest.assistant import AssistantUnderTest, IrisUnderTest, finished, wait_until
from irisdev.voicetest.conversations import CONVERSATIONS, Conversation, Turn
from irisdev.voicetest.scripted import BUDGETS, ScriptedConversations, TurnResult, summarize
from irisdev.voicetest.selftalk import SelfTalk, SelfTalkResult
from irisdev.voicetest.stub_llm import StubModel
from irisdev.voicetest.voices import UserVoice

__all__ = [
    "AssistantUnderTest",
    "BUDGETS",
    "CONVERSATIONS",
    "Conversation",
    "Endpoint",
    "IrisUnderTest",
    "ScriptedConversations",
    "SelfTalk",
    "SelfTalkResult",
    "Space",
    "StubModel",
    "Turn",
    "TurnResult",
    "UserVoice",
    "finished",
    "summarize",
    "wait_until",
]

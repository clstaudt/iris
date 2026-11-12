"""The conversations the voice test holds with Iris.

Each is a sequence of turns, said by the user into the virtual room. A turn
says when the user speaks (after Iris has finished, while it talks, or after
pressing Stop), what it says (pauses included) and what is expected: that
Iris heard it as one turn, what its reply mentions, and that it answered or
stayed silent.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

Part = Union[str, float]


@dataclass
class Turn:
    """One thing the user says, and what should follow."""

    parts: list[Part]
    # Words of which the reply should mention at least one (a real model only)
    reply_mentions: list[str] = field(default_factory=list)
    # Iris should not answer (a word that is not addressed to it)
    no_reply: bool = False
    # Said while Iris is talking, after it has spoken this many seconds
    interrupt_after: Optional[float] = None
    # Stop is pressed after Iris has spoken this many seconds of the reply
    stop_after: Optional[float] = None
    # The reply is listened to in silence to its end (Iris must not take its
    # own voice for the user)
    listen_through: bool = False
    # Said as soon as Iris has spoken its reply to the end (its speaker's
    # stream played out), and at least this many seconds after the last of
    # it was heard, rather than once it has been quiet for a while
    answer_after: Optional[float] = None
    # How fast the user speaks (1 is the voice's own pace)
    speed: float = 1.0

    @property
    def text(self) -> str:
        """What the user says, without its pauses."""
        return " ".join(part for part in self.parts if isinstance(part, str))


@dataclass
class Conversation:
    name: str
    turns: list[Turn]
    # A wake word for this conversation; None leaves it off
    wake_word: Optional[str] = None


CONVERSATIONS = [
    Conversation(
        "a question and a follow-up",
        [
            Turn(["What is the capital of France?"], reply_mentions=["Paris"]),
            Turn(["And roughly how many people live there?"], reply_mentions=["million"]),
        ],
    ),
    Conversation(
        "answering at once",
        [
            Turn(["What is the capital of France?"], reply_mentions=["Paris"]),
            # A brisk "Yes", a third of a second
            Turn(["Yes."], answer_after=0.3, speed=1.8),
            Turn(["And how many people live there?"], reply_mentions=["million"], answer_after=0.4),
        ],
    ),
    Conversation(
        "a pause in the middle of a sentence",
        [
            Turn(
                ["I'd like to know", 0.8, "how tall the Eiffel Tower is."],
                reply_mentions=["meter", "metre", "feet", "330", "324", "300"],
            ),
        ],
    ),
    Conversation(
        "thinking aloud",
        [
            Turn(
                ["Can you tell me about", 1.2, "the first landing on the moon?"],
                reply_mentions=["1969", "Apollo", "Armstrong"],
            ),
        ],
    ),
    Conversation(
        "a sentence, then a question",
        [
            Turn(
                ["I'm cooking pasta tonight.", 0.6, "How long should I boil spaghetti?"],
                reply_mentions=["minute"],
            ),
        ],
    ),
    Conversation(
        "interrupting a long answer",
        [
            Turn(["Tell me a long story about a lighthouse keeper."]),
            Turn(["Sorry, stop. What is two plus two?"], interrupt_after=3.0, reply_mentions=["four", "4"]),
        ],
    ),
    Conversation(
        "listening to a long answer to its end",
        [
            Turn(["Explain in detail how a rainbow forms."], listen_through=True),
        ],
    ),
    Conversation(
        "the stop button",
        [
            Turn(["Tell me a long story about the sea."], stop_after=2.0),
            Turn(["What day comes after Monday?"], reply_mentions=["Tuesday"]),
        ],
    ),
    Conversation(
        "the wake word",
        [
            Turn(["What's the weather like today?"], no_reply=True),
            Turn(["Iris, what is the capital of Italy?"], reply_mentions=["Rome"]),
            Turn(["And of Spain?"], reply_mentions=["Madrid"]),
        ],
        wake_word="Iris",
    ),
    Conversation(
        "quick questions back to back",
        [
            Turn(["What color is the sky on a clear day?"], reply_mentions=["blue"]),
            Turn(["How many legs does a spider have?"], reply_mentions=["eight", "8"]),
            Turn(["What is the opposite of hot?"], reply_mentions=["cold"]),
        ],
    ),
]

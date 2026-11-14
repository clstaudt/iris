"""What a scripted conversation is made of.

A Conversation is a list of turns and the settings it is held with (tools,
a scripted web, the model's rules, a persona, noise in the room). A Turn is
one thing the user says or types, and says when (At), what else happens
meanwhile (actions such as Stop, asides and sounds in the room), and what is
expected of Iris.

At says when something happens, relative to the turn before (for a turn)
or to the turn itself (for its actions and asides):

- At.finished(gap): once Iris has answered and fallen quiet, after a gap;
- At.reply_audible(s): after s seconds of the reply being heard;
- At.reply_end(gap): at the end of the reply's last sound, plus a gap (it
  may be negative: on Iris's last syllable);
- At.answered(gap): as soon as the reply has been spoken to its end (the
  assistant done speaking, its speaker's stream played out), and at least
  gap after its last sound;
- At.in_gap(n): in the n-th pause between sentences of the reply;
- At.before_end(s): s seconds before the reply's natural end;
- At.state(name, s): once a state (or flag) has held for s seconds;
- At.event(kind, s): s seconds after an event in the log;
- At.speech_end(s), At.speech_start(s): from the end or start of the
  utterance (the turn before's for a turn, the turn's own for its actions);
- At.start(s): from the start of the conversation.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

Part = Union[str, float]

# The user waits this long after the assistant has finished before speaking
GAP = 0.6
# What a turn may expect of Iris
EXPECTATIONS = ("answer", "silent", "continued", "either", "silenced", "wake", "sleep")


@dataclass(frozen=True)
class At:
    """When something happens (see the module's description)."""

    kind: str
    seconds: float = 0.0
    # The state or event kind waited for
    name: str = ""
    # A further condition on the event's data
    where: Optional[Callable[[dict], bool]] = None
    # For in_gap: which pause, how long it must be, and from when in the reply
    n: int = 1
    min_silence: float = 0.15
    after: float = 1.0

    @classmethod
    def finished(cls, gap: float = GAP) -> "At":
        return cls("finished", gap)

    @classmethod
    def reply_audible(cls, seconds: float) -> "At":
        return cls("reply_audible", seconds)

    @classmethod
    def reply_end(cls, gap: float = 0.0) -> "At":
        return cls("reply_end", gap)

    @classmethod
    def answered(cls, gap: float = 0.0) -> "At":
        return cls("answered", gap)

    @classmethod
    def in_gap(cls, n: int = 1, min_silence: float = 0.15, after: float = 1.0) -> "At":
        return cls("in_gap", 0.0, n=n, min_silence=min_silence, after=after)

    @classmethod
    def before_end(cls, seconds: float) -> "At":
        return cls("before_end", seconds)

    @classmethod
    def state(cls, name: str, seconds: float = 0.0) -> "At":
        return cls("state", seconds, name=name)

    @classmethod
    def event(cls, kind: str, seconds: float = 0.0, where: Optional[Callable[[dict], bool]] = None,
              n: int = 1) -> "At":
        """s seconds after the n-th such event (and data where holds)."""
        return cls("event", seconds, name=kind, where=where, n=n)

    @classmethod
    def speech_end(cls, seconds: float = 0.0) -> "At":
        return cls("speech_end", seconds)

    @classmethod
    def speech_start(cls, seconds: float = 0.0) -> "At":
        return cls("speech_start", seconds)

    @classmethod
    def start(cls, seconds: float = 0.0) -> "At":
        return cls("start", seconds)

    @property
    def before_turn(self) -> bool:
        """An action at this moment happens before its turn is said (it
        waits for the turn before to finish, or for the clock)."""
        return self.kind in ("finished", "start")

    def describe(self) -> str:
        detail = self.name or (f"{self.n}" if self.kind == "in_gap" else "")
        args = ", ".join(a for a in (detail, f"{self.seconds:g}" if self.seconds else "") if a)
        return f"{self.kind}({args})"


@dataclass
class Say:
    """Something said aside, during a turn, not waited for."""

    parts: list[Part]
    # A voice of voices.VOICES, or the user's
    voice: Optional[str] = None
    gain: float = 1.0
    # Part of the turn (it goes on with it), rather than to be ignored
    continues: bool = False
    splice: bool = False

    @property
    def text(self) -> str:
        return " ".join(p for p in self.parts if isinstance(p, str))


@dataclass
class Sound:
    """A sound of the room (sounds.NAMES), played during a turn."""

    name: str
    gain: float = 1.0
    seconds: Optional[float] = None


@dataclass
class Noise:
    """A sound in the room for the whole conversation, from lead seconds
    before the first turn: at snr dB below the user's speech, or at a gain."""

    sound: str
    snr: Optional[float] = None
    gain: Optional[float] = None
    lead: float = 8.0
    # A fixed level of the sound (RMS) instead of snr or gain
    rms: Optional[float] = None


@dataclass
class Echo:
    """How loud Iris's speaker comes back into its microphone, in what room,
    and how hard the speaker is driven."""

    gain: Optional[float] = None
    room: Optional[float] = None
    drive: Optional[float] = None


@dataclass
class Check:
    """A check of a turn beyond the usual ones: fn(result, view) returns a
    failure, or None when the check passes."""

    name: str
    fn: Callable


# An action at a moment: "stop", "pause", "resume", ("mic", on), ("persona",
# name), ("tool", id, on), ("send", text) or ("turn_detection", mode)
Action = Union[str, tuple]


@dataclass
class Turn:
    """One thing the user says or types, and what should follow."""

    parts: list[Part]
    # Words of which the reply should mention at least one (a real model,
    # or the scripted one when check_with_stub is set)
    reply_mentions: list[str] = field(default_factory=list)
    # Iris should not answer (a word that is not addressed to it); the same
    # as expect="silent"
    no_reply: bool = False
    # Said while Iris is talking, after it has spoken this many seconds;
    # the same as at=At.reply_audible(seconds)
    interrupt_after: Optional[float] = None
    # Stop is pressed after Iris has spoken this many seconds of the reply;
    # the same as an action ("stop") at At.reply_audible(seconds)
    stop_after: Optional[float] = None
    # The reply is listened to in silence to its end (Iris must not take its
    # own voice for the user)
    listen_through: bool = False
    # Said as soon as Iris has spoken its reply to the end, and at least this
    # many seconds after the last of it was heard, rather than once it has
    # been quiet for a while; the same as at=At.answered(seconds)
    answer_after: Optional[float] = None

    # When, and how ------------------------------------------------------------
    at: Optional[At] = None
    # Typed in the app instead of said
    typed: bool = False
    # [(At, action)]: done from a thread at that moment, logged as "action.<name>"
    actions: list = field(default_factory=list)
    # [(At, Say or Sound)]: said or played meanwhile, not waited for
    during: list = field(default_factory=list)
    voice: Optional[str] = None
    speed: float = 1.0
    # The peak level the user arrives at
    level: Optional[float] = None
    # (rt60, drr): said across a room
    room: Optional[tuple] = None
    # The parts synthesized as one sentence, the pauses cut in
    splice: bool = False
    # Words synthesized with the parts as one sentence, and cut off: the
    # user trails off, sounding unfinished
    unsaid: list = field(default_factory=list)
    # What is said instead to a real model
    real_parts: Optional[list] = None
    # Said only when this holds for the reply before it (its text)
    when: Optional[Callable[[str], bool]] = None

    # What is expected -----------------------------------------------------------
    expect: str = "answer"
    # Times the user may be heard going on within the turn (a hesitation
    # Smart Turn took for its end) without the turn counting as taken early
    allow_continued: int = 0
    must_hear: list[str] = field(default_factory=list)
    must_hear_in_order: list[str] = field(default_factory=list)
    must_not_hear: list[str] = field(default_factory=list)
    # What was heard, normalized, is exactly what was said
    exact: bool = False
    # Regular expressions found in what was heard, normalized
    heard_terms: list[str] = field(default_factory=list)
    # (lo, hi): seconds Iris took to hear that the turn was over
    turn_end: Optional[tuple] = None
    # Check reply_mentions with the scripted model too (facts from the
    # scripted web or a rule)
    check_with_stub: bool = False
    # No complete reply in the turn may contain these
    reply_excludes: list[str] = field(default_factory=list)
    # The tools used, in order; [] allows none; None does not check
    tools_used: Optional[list] = None
    # The words a tool was called on (a search's query) holds, normalized
    search_mentions: list[str] = field(default_factory=list)
    # How many requests the scripted model gets in the turn
    model_requests: Optional[int] = None
    reply_language: Optional[str] = None
    # "rising" (awake) or "falling" (asleep)
    chime: Optional[str] = None
    # A predicate on the turn's first request to the model (its messages),
    # or on the assistant's history without a scripted model: True passes,
    # False or a text fails
    history: Optional[Callable] = None
    # Seconds from the end of speech to the first word, at most
    first_word: Optional[float] = None
    # Seconds of the user's speech no microphone listened to, at most
    deaf_limit: Optional[float] = None
    # Phrases the reply's voice says exactly once, and never, as transcribed
    spoken_once: list[str] = field(default_factory=list)
    spoken_excludes: list[str] = field(default_factory=list)
    # Words of which an error the turn reports should mention at least one
    # (case ignored): the turn must report such an error, and it does not
    # fail the turn
    error_mentions: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    name: str = ""

    def __post_init__(self):
        if self.expect not in EXPECTATIONS:
            raise ValueError(f"A turn expects one of {', '.join(EXPECTATIONS)}, not {self.expect!r}")

    @property
    def text(self) -> str:
        """What the user says, without its pauses."""
        return " ".join(part for part in self.parts if isinstance(part, str))

    @property
    def expectation(self) -> str:
        return "silent" if self.no_reply else self.expect

    @property
    def trigger(self) -> At:
        """When the turn is said."""
        if self.at is not None:
            return self.at
        if self.interrupt_after is not None:
            return At.reply_audible(self.interrupt_after)
        if self.answer_after is not None:
            return At.answered(self.answer_after)
        return At.finished()

    def all_actions(self) -> list:
        """The turn's actions, stop_after among them."""
        actions = list(self.actions)
        if self.stop_after is not None:
            actions.append((At.reply_audible(self.stop_after), "stop"))
        return actions

    @property
    def presses_stop(self) -> bool:
        return any(action == "stop" for _, action in self.all_actions())


@dataclass
class Silence:
    """A stretch in which nobody speaks: nothing is to happen in it."""

    seconds: float
    # Events that must not come
    kinds: tuple = ("chat.user", "chat.done", "chat.continued", "error")
    name: str = ""

    @property
    def text(self) -> str:
        return f"(silence, {self.seconds:g} s)"


@dataclass
class Conversation:
    name: str
    turns: list
    # A wake word for this conversation; None leaves it off
    wake_word: Optional[str] = None
    # {plugin: on}, e.g. {"web": True}
    tools: dict = field(default_factory=dict)
    # FakeWeb's arguments (results, pages, delay, browse_delay)
    web: Optional[dict] = None
    # The scripted model's rules
    stub_rules: Optional[list] = None
    # The scripted model's settings (first_token, per_word, serial,
    # context_length, summary_delay, capabilities, refuse_tools)
    stub_options: dict = field(default_factory=dict)
    persona: Optional[str] = None
    turn_detection: Optional[str] = None
    stt_model: Optional[str] = None
    # {"window": seconds, "sleep_word": text}
    wake_options: Optional[dict] = None
    # A voice of voices.VOICES for every turn
    voice: Optional[str] = None
    noise: list = field(default_factory=list)
    echo: Optional[Echo] = None
    # Held with an Iris started for it
    fresh: bool = False
    # Messages the conversation starts with
    seed_history: list = field(default_factory=list)
    # Checks known to fail today: "check" for any turn, "T2:check" for one,
    # "T2:*" for all of a turn's
    expected_failures: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    # What it needs: "stub" (the scripted model), "real_model", "real_web"
    needs: tuple = ()
    # Seconds a reply may take, instead of the runner's
    reply_timeout: Optional[float] = None

    def expected(self, number: int, check: str) -> bool:
        """Whether a check of a turn is known to fail."""
        for entry in self.expected_failures:
            turn, colon, name = entry.partition(":")
            if not colon:
                if entry == check:
                    return True
            elif turn == f"T{number}" and name in (check, "*"):
                return True
        return False

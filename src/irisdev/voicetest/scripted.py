"""Scripted conversations: a user's voice talks to an assistant, turn by turn.

Every turn is measured from the space's ground truth: when the user's last
syllable ended, when the assistant's first word was heard, when its voice
fell silent after the user spoke over it or Stop was pressed. What the
assistant reported (what it heard, its reply, its own timings, the tools it
used) is checked against what was said, and against what the scripted model
and the scripted web saw.

A turn is said when its moment comes (script.At): once Iris has finished,
over its reply, at the reply's last syllable, while it searches. When the
next turn's moment comes before Iris has finished, the turn hands over
then, and is judged on what happened until that moment.
"""

import os
import re
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from iris.assistant import DEFAULT_WAKE_WINDOW, _sleep_chime, _thinking_ping, _wake_chime
from irisdev.voicetest.acoustics import (
    AUDIBLE,
    RATE,
    Clip,
    Space,
    merge,
    sound_span,
    subtract,
    to_rate,
)
from irisdev.voicetest.assistant import (
    BUSY,
    DEFAULT_TOOLS,
    AssistantUnderTest,
    finished,
    wait_until,
)
from irisdev.voicetest.fakesight import FakeCamera, FakeScreen
from irisdev.voicetest.fakeweb import FakeWeb
from irisdev.voicetest.script import GAP, At, Conversation, Echo, Noise, Restart, Say, Silence, Sound, Turn
from irisdev.voicetest.sounds import gain_for_snr, sound
from irisdev.voicetest.voices import UserVoice
from irisdev.voicetest.wording import (
    contains,
    heard_precision,
    heard_share,
    in_order,
    normalized,
    same_words,
    term_found,
    words,
)

__all__ = ["BUDGETS", "GAP", "ScriptedConversations", "TurnResult", "heard_share", "summarize", "words"]

# What a turn may take, in seconds
BUDGETS = {
    # From the end of the user's speech to the first word, without the time
    # the language model takes for its first sentence (and without the long
    # silence, when that ends turns)
    "voice_pipeline": 2.5,
    # From the user's first word, said while the assistant talks, to silence
    "barge_in": 1.0,
    # From pressing Stop to silence
    "stop": 0.5,
    # From the end of speech until Iris knew the turn was over
    "turn_end": 1.0,
    # What a web search may add to the first word
    "search_added": 6.0,
    # The first reply in a language not spoken before in the session
    "new_language": 3.5,
}


# The scripted model's settings that change what the assistant knows of it
MODEL_SETTINGS = {"context_length", "capabilities", "refuse_tools", "vision", "missing"}


def budgets_from(text: str) -> dict:
    """Budgets given as "name=seconds,name=seconds" (as in the environment
    variable IRIS_VOICETEST_BUDGETS), checked against the known names."""
    budgets = {}
    for item in filter(None, (part.strip() for part in (text or "").split(","))):
        name, _, seconds = item.partition("=")
        if name.strip() not in BUDGETS:
            raise ValueError(f"No budget called {name.strip()!r}; there are {', '.join(BUDGETS)}")
        budgets[name.strip()] = float(seconds)
    return budgets


# How much of what the user said the assistant must have heard (its words)
HEARD = 0.75
# How long a turn not addressed to the assistant is watched for a reply
NO_REPLY_WATCH = 8.0
# How long after an aside a reaction to it is looked for
ASIDE_WATCH = 4.0
# A clip cut and followed this soon by its remainder was held, not stopped
HOLD_FOLLOW = 1.2
# The part of a resumed clip looked for in the clip it resumes
HOLD_PROBE = 0.3
# How far back in what a clip was heard playing a later clip's start is
# looked for, to find sound played twice
REPLAY_LOOKBACK = 3.0
# A resumed clip that goes on this much before where its clip was cut, or
# more, repeats what was heard (seconds)
REPEAT_SLACK = 0.05
# What a merged turn may have said of an answer to its first part
EITHER_ANSWER = 1.0
# Sentences run together in a reply: "up.In"
GLUED = re.compile(r"[.!?][A-Za-z]")
# The chimes Iris plays when it wakes and falls asleep, and the ping it sounds
# while a reply is long in coming, as a speaker records them
CHIMES = {
    "rising": to_rate(_wake_chime(), 24000),
    "falling": to_rate(_sleep_chime(), 24000),
    "thinking": to_rate(_thinking_ping(), 24000),
}


def chime_of(clip: Clip) -> Optional[str]:
    """"rising", "falling" or "thinking" if the clip is one of Iris's chimes."""
    for name, chime in CHIMES.items():
        if abs(len(clip.audio) - len(chime)) <= 2:
            n = min(len(clip.audio), len(chime))
            if n and float(np.max(np.abs(clip.audio[:n] - chime[:n]))) < 1e-3:
                return name
    return None


def inner_silences(audio: np.ndarray, min_silence: float) -> list[tuple[float, float]]:
    """Pauses of at least min_silence inside audio (not before its first
    sound or after its last), in seconds from its start."""
    step = RATE // 50
    windows = len(audio) // step
    if windows == 0:
        return []
    loud = np.abs(audio[: windows * step]).reshape(windows, step).max(axis=1) > AUDIBLE
    found, quiet_from = [], None
    for i, is_loud in enumerate(loud):
        if is_loud:
            if quiet_from is not None and (i - quiet_from) * step / RATE >= min_silence:
                found.append((quiet_from * step / RATE, i * step / RATE))
            quiet_from = None
        elif quiet_from is None and i > 0 and loud[: i].any():
            quiet_from = i
    return found


@dataclass
class Hold:
    """The voice paused (held) and went on: when, for how long it was silent,
    and how much it repeated of what had been heard."""

    at: float
    silence: float
    rewind: Optional[float]


def holds(clips: list[Clip]) -> list[Hold]:
    """Clips cut and followed within HOLD_FOLLOW by their remainder. Where
    the remainder starts is found by looking for its first HOLD_PROBE
    seconds in the clip it resumes."""
    found = []
    for a, b in zip(clips, clips[1:]):
        if a.stopped is None or b.started - a.stopped > HOLD_FOLLOW:
            continue
        probe = b.audio[: int(HOLD_PROBE * RATE)]
        if len(probe) < RATE // 10 or float(np.abs(probe).max()) < AUDIBLE:
            continue
        cut = int((a.stopped - a.started) * RATE)
        lo, hi = max(0, cut - 2 * RATE), min(len(a.audio), cut + RATE // 2 + len(probe))
        segment = a.audio[lo:hi]
        if len(segment) < len(probe):
            continue
        offset = _locate(segment, probe)
        if offset is None:
            continue
        found.append(Hold(a.stopped, round(b.started - a.stopped, 3), round((cut - (lo + offset)) / RATE, 3)))
    return found


def replayed(earlier: Clip, clip: Clip) -> Optional[float]:
    """Seconds of sound a clip plays again that an earlier clip was heard
    playing, when it goes on with the earlier one's audio from before where
    that was cut (its first HOLD_PROBE found in the earlier one's audio,
    within REPLAY_LOOKBACK of where it was last heard). None if it plays
    nothing again: a new clip, or one going on where its clip was cut."""
    probe = clip.audio[: int(HOLD_PROBE * RATE)]
    if len(probe) < RATE // 10 or float(np.abs(probe).max()) < AUDIBLE:
        return None
    heard = len(earlier.heard())
    lo = max(0, heard - int(REPLAY_LOOKBACK * RATE))
    segment = earlier.audio[lo : heard + RATE // 2 + len(probe)]
    if len(segment) < len(probe):
        return None
    offset = _locate(segment, probe)
    if offset is None:
        return None
    again = (heard - (lo + offset)) / RATE
    return round(again, 3) if again > REPEAT_SLACK else None


def voice_holds(lines: list) -> dict:
    """What Iris's voice log (iris.voice: (at, message) lines) says of the
    holds: how many, how many were judged its own echo, too little for a
    turn (an aside) and the user's, how often the voice went on after being
    cut, and how often speech over it was weighed and did not hold it."""
    messages = [message for _, message in lines]
    return {
        "held": sum(m.startswith("held") for m in messages),
        "echo": sum(m.startswith("echo") for m in messages),
        "aside": sum(m.startswith("aside") for m in messages),
        "user": sum(m.startswith("user") for m in messages),
        "resumed": sum(m.startswith("resumes") for m in messages),
        "not_held": sum(m.startswith("not held") for m in messages),
    }


def _locate(segment: np.ndarray, probe: np.ndarray, threshold: float = 0.9) -> Optional[int]:
    """Where in the segment the probe is found, by normalized correlation."""
    n = len(probe)
    corr = np.correlate(segment, probe, mode="valid")
    energy = np.convolve(segment**2, np.ones(n), mode="valid")
    score = corr / (np.sqrt(np.maximum(energy, 1e-12)) * float(np.linalg.norm(probe)) + 1e-12)
    best = int(np.argmax(score))
    return best if score[best] >= threshold else None


def flag_windows(events: list[tuple], start: float, end: float) -> dict:
    """{flag: [(on, off)]} from "flags" events, clipped to [start, end]."""
    windows: dict = {}
    on_since: dict = {}
    for at, _, data in events:
        for flag, value in (data.get("flags") or {}).items():
            if value and flag not in on_since:
                on_since[flag] = at
            elif not value and flag in on_since:
                windows.setdefault(flag, []).append((on_since.pop(flag), at))
    for flag, since in on_since.items():
        windows.setdefault(flag, []).append((since, float("inf")))
    return {
        flag: [(max(a, start), min(b, end)) for a, b in spans if b > start and a < end]
        for flag, spans in windows.items()
        if any(b > start and a < end for a, b in spans)
    }


def state_timeline(events: list[tuple], start: float, end: float) -> list[tuple[float, str]]:
    """[(from, state)] from "state" events: the state at start, then each change."""
    timeline = []
    for at, _, data in events:
        state = data.get("state", "")
        if at <= start:
            timeline = [(start, state)]
        elif at <= end:
            timeline.append((at, state))
    return timeline


def longest_busy_silence(timeline: list[tuple[float, str]], sounds: list[tuple[float, float]],
                         start: float, end: float) -> float:
    """The longest stretch in which the assistant was working on the turn
    (a busy state) and its speaker was silent."""
    busy = []
    for (at, state), following in zip(timeline, timeline[1:] + [(end, "")]):
        if state in BUSY and state != "speaking":
            busy.append((max(at, start), min(following[0], end)))
    quiet = subtract(merge(busy), sounds)
    return round(max((b - a for a, b in quiet), default=0.0), 3)


@dataclass
class TurnResult:
    conversation: str
    turn: int
    said: str
    heard: str = ""
    heard_share: float = 0.0
    # How much of what was heard was said
    heard_precision: Optional[float] = None
    # The assistant took the turn before the user had finished
    taken_early: bool = False
    reply: str = ""
    spoke: bool = False
    # Seconds from the end of the user's speech to the first word heard
    first_word: Optional[float] = None
    # The same without the language model's time to its first sentence
    voice_pipeline: Optional[float] = None
    # The assistant's own measurement, in ms, and what ran each stage
    stages: dict = field(default_factory=dict)
    components: dict = field(default_factory=dict)
    barge_in: Optional[float] = None
    stop: Optional[float] = None
    # Times the voice was cut off while nobody spoke, apart from Stop
    self_cuts: int = 0
    errors: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    # The checks failed, by name (one per failure)
    failed: list = field(default_factory=list)
    # Failures known today (Conversation.expected_failures), and known
    # failures that passed this time
    expected: list = field(default_factory=list)
    now_passes: list = field(default_factory=list)
    # Why the turn or conversation was not held
    skipped: str = ""
    tags: list = field(default_factory=list)
    # What kind of turn it is (Turn.name), for grouping
    label: str = ""
    # When it was said (the trigger), and how late it was said
    at: str = ""
    late: Optional[float] = None
    language: str = ""
    # Seconds below are from the end of the user's speech (or the send)
    users: list = field(default_factory=list)  # (at, source, text)
    replies: list = field(default_factory=list)  # (at, text, interrupted)
    continued: int = 0
    tool_uses: list = field(default_factory=list)  # (at, tool, detail)
    flags: dict = field(default_factory=dict)  # {flag: [(on, off)]}
    states: list = field(default_factory=list)  # (from, state)
    requests: list = field(default_factory=list)  # the scripted model's requests
    web_calls: list = field(default_factory=list)  # the scripted web's calls
    model_rounds: int = 0
    deaf: float = 0.0
    deaf_spans: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)  # (at, probability)
    holds: list = field(default_factory=list)  # (at, silence, rewind)
    # What Iris's voice log said in the turn (each hold, what it was judged
    # to be, where the voice went on)
    voice: list = field(default_factory=list)  # (at, message)
    chimes: list = field(default_factory=list)  # (at, rising or falling)
    busy_silence: Optional[float] = None
    glued: bool = False
    # Seconds of the reply heard
    audible: Optional[float] = None
    asides: list = field(default_factory=list)  # (at, what)
    threads: list = field(default_factory=list)
    resources: dict = field(default_factory=dict)
    # What the reply's voice said, as Iris's recognizer heard it afterwards
    spoken: str = ""
    spoken_language: str = ""

    @property
    def passed(self) -> bool:
        return not self.failures

    def fail(self, check: str, message: str):
        self.failures.append(message)
        self.failed.append(check)

    def line(self) -> str:
        """One line of report, and a line per failure."""
        if self.skipped:
            turn = f" · turn {self.turn}" if self.turn else ""
            return f"  skip {self.conversation}{turn} · {self.skipped}"
        mark = "ok  " if self.passed else "FAIL"
        if self.passed and self.expected:
            mark = "xfail"
        parts = [f"  {mark} {self.conversation} · turn {self.turn}"]
        if self.first_word is not None:
            names = (("turn_end", "turn end"), ("recognition", "recognition"),
                     ("reply", "model"), ("voice", "voice"))
            detail = ", ".join(
                f"{name} {self.stages[key] / 1000:.2f}" for key, name in names
                if self.stages.get(key) is not None
            )
            parts.append(f"first word {self.first_word:.2f} s" + (f" ({detail})" if detail else ""))
        if self.barge_in is not None:
            parts.append(f"silent {self.barge_in:.2f} s after the user began")
        if self.stop is not None:
            parts.append(f"silent {self.stop:.2f} s after Stop")
        if self.tool_uses:
            parts.append("used " + ", ".join(f"{tool} ({detail})" for _, tool, detail in self.tool_uses))
        counts = voice_holds(self.voice)
        if counts["held"] or counts["resumed"]:
            repeated = [rewind for _, _, rewind in self.holds if rewind is not None and rewind > REPEAT_SLACK]
            parts.append(f"held {counts['held']} ({counts['echo']} echo, {counts['aside']} aside, "
                         f"{counts['user']} user), resumed {counts['resumed']}"
                         + (f", repeating {', '.join(f'{r:.2f}' for r in repeated)} s" if repeated else ""))
        lines = [f"\n         {f}" for f in self.failures]
        lines += [f"\n         expected failure: {f}" for f in self.expected]
        lines += [f"\n         now passes: {f}" for f in self.now_passes]
        return " · ".join(parts) + "".join(lines)


@dataclass
class Aside:
    """Something said or played during a turn."""

    item: object
    start: float
    end: float


@dataclass
class Running:
    """A turn being held: what was said or sent, and when."""

    turn: Turn
    number: int
    result: TurnResult
    # Events of the turn count from here (when it was said or sent)
    since: float = 0.0
    said: object = None  # the Utterance
    sent: Optional[float] = None
    # The turn before, which this one may go on with
    previous: Optional["Running"] = None
    # The next turn goes on with this one, which expects no answer of its own
    merged_into_next: bool = False
    # Stop presses, actions done, asides said or played
    pressed: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    asides: list = field(default_factory=list)
    threads: list = field(default_factory=list)
    completed: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def ref_end(self) -> float:
        """The end of the user's speech, or the moment the turn was sent."""
        return self.said.speech_end if self.said is not None else (self.sent or self.since)

    @property
    def ref_start(self) -> float:
        return self.said.speech_start if self.said is not None else (self.sent or self.since)

    # The turn was heard as one with the turn before (chat.continued)
    merged: bool = False

    @property
    def judged_text(self) -> str:
        """What the turn is judged against: with the turn it goes on with,
        and with what was said aside to go on with it."""
        text = self.result.said
        goes_on = self.turn.expectation == "continued" or (self.turn.expectation == "either" and self.merged)
        if goes_on and self.previous is not None:
            text = f"{self.previous.result.said} {text}"
        more = [a.item.text for a in self.asides if isinstance(a.item, Say) and a.item.continues]
        return " ".join([text] + more)


@dataclass
class View:
    """What a turn's own checks (script.Check) may look at."""

    runner: "ScriptedConversations"
    running: Running
    end: float

    @property
    def log(self):
        return self.runner.assistant.log

    @property
    def speaker(self):
        return self.runner.assistant.endpoint.speaker

    @property
    def stub(self):
        return self.runner.stub

    @property
    def web(self) -> Optional[FakeWeb]:
        return self.runner.web

    @property
    def camera(self) -> Optional[FakeCamera]:
        return self.runner.camera

    @property
    def screen(self) -> Optional[FakeScreen]:
        return self.runner.screen

    @property
    def previous(self) -> Optional[Running]:
        return self.running.previous

    def events(self, kind: Optional[str] = None, start: Optional[float] = None, end: Optional[float] = None):
        start = self.running.since if start is None else start
        return self.log.since(start, kind, self.end if end is None else end)

    def requests(self) -> list:
        if self.stub is None:
            return []
        return self.stub.requests_between(self.running.since, self.end)

    def history(self) -> list[dict]:
        return self.runner.assistant.history()


class ScriptedConversations:
    """Holds scripted conversations with one assistant in a space."""

    def __init__(
        self,
        assistant: AssistantUnderTest,
        space: Space,
        voice: UserVoice,
        check_replies: bool = True,
        reply_timeout: float = 300.0,
        budgets: Optional[dict] = None,
        report: Callable[[str], None] = print,
        stub=None,
        transcribe_replies: bool = False,
        resources=None,
        real_web: Optional[bool] = None,
    ):
        """
        Args:
            check_replies: Check what replies mention (off for a scripted model)
            reply_timeout: Seconds a reply may take to be complete
            budgets: Budgets to use instead of BUDGETS, by name
            stub: The scripted model the assistant talks to, if it does
            transcribe_replies: Transcribe every reply's voice afterwards
            resources: A resources.Resources, sampled after every turn
            real_web: Search the real web (IRIS_VOICETEST_WEB=real by default)
        """
        self.assistant = assistant
        self.space = space
        self.voice = voice
        self.check_replies = check_replies
        self.reply_timeout = reply_timeout
        self.budgets = {**BUDGETS, **(budgets or {})}
        self.report = report
        self.stub = stub
        self.transcribe_replies = transcribe_replies
        self.resources = resources
        self.real_web = os.environ.get("IRIS_VOICETEST_WEB", "") == "real" if real_web is None else real_web
        self.web: Optional[FakeWeb] = None
        self.camera: Optional[FakeCamera] = None
        self.screen: Optional[FakeScreen] = None
        self._voices: dict = {}
        self._touched: dict = {}
        self._saved: dict = {}
        self._conversation: Optional[Conversation] = None
        self._timeout = reply_timeout
        self._start = 0.0
        # Languages replied in so far, for the budget of the first reply in each
        self._languages: set = set()
        # Each turn's start and the end of its speech (or its send), by number
        self._moments: dict = {}
        # The threads running before each turn was said, by number
        self._threads_before: dict = {}
        # The results of the conversation being held, as each turn is judged
        self.results_so_far: list[TurnResult] = []

    @property
    def real_model(self) -> bool:
        return self.stub is None

    # Voices ----------------------------------------------------------------------

    def voice_for(self, name: Optional[str], speed: float = 1.0) -> UserVoice:
        """The user's voice (or one of voices.VOICES), at a speed."""
        if name is None and speed == 1.0:
            return self.voice
        key = (name, speed)
        if key not in self._voices:
            if name is None:
                base = self.voice
                self._voices[key] = UserVoice(base.cache, base.engine, base.voice, base.language, speed)
            else:
                self._voices[key] = UserVoice.named(self.voice.cache, name, speed)
        return self._voices[key]

    def _sound_voices(self, kokoro_voice: str) -> UserVoice:
        """Voices for the sounds that speak (babble, TV, music)."""
        return UserVoice(self.voice.cache, "kokoro", kokoro_voice, "en")

    def _utterance(self, turn: Turn, conversation: Conversation) -> np.ndarray:
        parts = turn.real_parts if (turn.real_parts and self.real_model) else (turn.reading or turn.parts)
        voice = self.voice_for(turn.voice or conversation.voice, turn.speed)
        return voice.utterance(parts, level=turn.level, room_response=turn.room, splice=turn.splice,
                               unsaid=turn.unsaid or None)

    def _synthesize_ahead(self, conversation: Conversation):
        """Synthesize every utterance first, so none is late for its moment."""
        for turn in conversation.turns:
            if isinstance(turn, Turn) and not turn.typed:
                self._utterance(turn, conversation)
                for _, item in turn.during:
                    if isinstance(item, Say):
                        self.voice_for(item.voice or conversation.voice).utterance(item.parts, splice=item.splice)

    # Holding a conversation ------------------------------------------------------

    def missing(self, conversation: Conversation) -> str:
        """Why the conversation cannot be held here, or ""."""
        needs = set(conversation.needs)
        restarts = conversation.memory or any(isinstance(t, Restart) for t in conversation.turns)
        if restarts and not hasattr(self.assistant, "restart"):
            return "the assistant cannot be restarted"
        if conversation.memory_topics and not hasattr(self.assistant, "seed_memory"):
            return "the assistant's memory cannot be seeded"
        if "stub" in needs and self.stub is None:
            return "needs the scripted model"
        if "real_model" in needs and self.stub is not None:
            return "needs a real model (IRIS_VOICETEST_HOST)"
        if "real_web" in needs and not self.real_web:
            return "needs the real web (IRIS_VOICETEST_WEB=real)"
        downloaded = getattr(self.assistant, "stt_downloaded", None)
        if conversation.stt_model and downloaded is not None and conversation.stt_model not in downloaded():
            return f"Whisper {conversation.stt_model} is not on this computer"
        return ""

    def hold(self, conversation: Conversation) -> list[TurnResult]:
        skipped = self.missing(conversation)
        if skipped:
            result = TurnResult(conversation.name, 0, "", skipped=skipped, tags=list(conversation.tags))
            self.report(result.line())
            return [result]
        self._conversation = conversation
        self._timeout = conversation.reply_timeout or self.reply_timeout
        self._moments = {}
        self._threads_before = {}
        results: list[TurnResult] = []
        try:
            self._prepare(conversation)
            results = self._hold(conversation)
            self._transcribe(results, conversation)
            self._expected_globally(results, conversation)
        except Exception as error:
            self.report(traceback.format_exc())
            failed = TurnResult(conversation.name, len(results) + 1, "", tags=list(conversation.tags))
            failed.fail("held", f"the conversation could not be held: {error!r}")
            results.append(failed)
        finally:
            wait_until(lambda: finished(self.assistant), self._timeout)
            self._restore()
        for result in results:
            self.report(result.line())
        return results

    def _prepare(self, conversation: Conversation):
        assistant = self.assistant
        self._synthesize_ahead(conversation)
        if conversation.fresh and hasattr(assistant, "restart"):
            assistant.restart()
        if conversation.memory:
            # First: it starts Iris again, with the other settings as they were
            self._set("memory", True, "memory", False)
            if conversation.memory_topics:
                assistant.seed_memory(conversation.memory_topics)
        assistant.clear_conversation()
        assistant.set_wake_word(conversation.wake_word or "")
        wait_until(lambda: finished(assistant), self._timeout)
        if self.stub is not None:
            if conversation.stub_rules is not None:
                self.stub.set_rules(conversation.stub_rules)
            if conversation.stub_options:
                self._saved["stub"] = self._configure_stub(conversation.stub_options)
        if conversation.web is not None or conversation.tools.get("web"):
            self.web = FakeWeb(**(conversation.web or {}), real=self.real_web or None)
            self.web.__enter__()
        # The camera and the screen are always scripted: a test never looks
        # at the room or the screen it runs in
        if conversation.camera is not None or conversation.tools.get("camera"):
            self.camera = FakeCamera(**(conversation.camera or {})).__enter__()
        if conversation.screen is not None or conversation.tools.get("screen"):
            self.screen = FakeScreen(**(conversation.screen or {})).__enter__()
        for plugin, on in conversation.tools.items():
            self._set("tool", (plugin, on), ("tool", plugin), (plugin, DEFAULT_TOOLS.get(plugin, False)))
        if conversation.persona_voice:
            persona_id = assistant.persona_speaking_with(*conversation.persona_voice)
            self._set("persona", persona_id, "persona", getattr(assistant, "persona", "iris"))
        if conversation.persona:
            self._set("persona", conversation.persona, "persona", getattr(assistant, "persona", "iris"))
        if conversation.turn_detection:
            self._set("turn_detection", conversation.turn_detection, "turn_detection", "smart_turn")
        if conversation.stt_model:
            self._set("stt_model", conversation.stt_model, "stt_model", getattr(assistant, "default_stt", ""))
            self._warm_up()
        if conversation.wake_options:
            self._set("wake_options", conversation.wake_options, "wake_options",
                      {"window": DEFAULT_WAKE_WINDOW, "sleep_word": ""})
        if conversation.echo is not None:
            space = self.space
            self._saved["echo"] = (space.echo_gain, space.echo_room, space.echo_drive)
            echo: Echo = conversation.echo
            space.set_echo(echo.gain, echo.room, echo.drive)
        if conversation.seed_history:
            assistant.seed_history(conversation.seed_history)

    def _configure_stub(self, options: dict) -> dict:
        """Change the scripted model's settings; what they were."""
        before = self.stub.configure(**options)
        if MODEL_SETTINGS & set(options) and hasattr(self.assistant, "refresh_model"):
            self.assistant.refresh_model()
        return before

    def _warm_up(self):
        """Recognize something once, so a newly chosen recognizer is loaded."""
        if not hasattr(self.assistant, "warm_up"):
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
            path = file.name
        try:
            sf.write(path, self.voice.utterance(["Hello there."]), RATE)
            self.assistant.warm_up(path)
        finally:
            os.unlink(path)

    def _set(self, name: str, value, key, default):
        """Change a setting for the conversation; it is put back afterwards."""
        if not self.assistant.configure(name, value):
            raise RuntimeError(f"could not set {name} to {value!r}")
        self._touched.setdefault(key, (name, default))

    def _restore(self):
        """Put back what the conversation changed, so the next one finds
        the defaults: web off, persona iris, Smart Turn, the default
        recognizer, the microphone on, the default wake window and sleep
        word, not paused, no scripted web, camera or screen, no rules, no
        sounds in the room, the echo as it was."""
        assistant = self.assistant
        touched, self._touched = self._touched, {}
        if "paused" in touched:
            assistant.resume()
            touched.pop("paused")
        for key, (name, default) in touched.items():
            try:
                if not assistant.configure(name, default):
                    self.report(f"  could not put {name} back to {default!r}")
                elif name == "stt_model":
                    self._warm_up()
            except Exception as error:
                self.report(f"  could not put {name} back to {default!r}: {error!r}")
        if self.web is not None:
            self.web.__exit__(None, None, None)
            self.web = None
        for name in ("camera", "screen"):
            if getattr(self, name) is not None:
                getattr(self, name).__exit__(None, None, None)
                setattr(self, name, None)
        if self.stub is not None:
            self.stub.reset_rules()
            if "stub" in self._saved:
                self._configure_stub(self._saved.pop("stub"))
        self.space.remove_sources()
        if "echo" in self._saved:
            self.space.set_echo(*self._saved.pop("echo"))
        self._saved.clear()

    def _place_noise(self, conversation: Conversation, start: float):
        """The conversation's sounds in the room, from before its first turn."""
        first = next((t for t in conversation.turns if isinstance(t, Turn) and not t.typed), None)
        speech = self._utterance(first, conversation) if first else self.voice.utterance(["Hello there."])
        for noise in conversation.noise:
            noise: Noise
            audio = sound(noise.sound, 30.0 if noise.sound in ("fan", "babble", "music", "tv") else None,
                          voices=self._sound_voices)
            if noise.rms is not None:
                gain = noise.rms / max(1e-9, float(np.sqrt(np.mean(audio**2))))
            elif noise.snr is not None:
                gain = gain_for_snr(audio, speech, noise.snr)
            else:
                gain = noise.gain if noise.gain is not None else 1.0
            self.space.add_source(audio, at=start, gain=gain, loop=True, name=noise.sound)
        if conversation.noise:
            time.sleep(max(n.lead for n in conversation.noise))

    def _hold(self, conversation: Conversation) -> list[TurnResult]:
        assistant = self.assistant
        start = self._start = time.monotonic()
        self._place_noise(conversation, start)
        results: list[TurnResult] = []
        self.results_so_far = results
        running: Optional[Running] = None
        for number, turn in enumerate(conversation.turns, 1):
            if isinstance(turn, (Silence, Restart)):
                if running is not None:
                    end = self._complete(running)
                    results.append(self._finish(running, end))
                    running = None
                step = self._silence if isinstance(turn, Silence) else self._restart
                results.append(step(turn, number, conversation))
                continue
            text = " ".join(p for p in (turn.real_parts if (turn.real_parts and self.real_model) else turn.parts)
                            if isinstance(p, str))
            new = Running(turn, number, TurnResult(conversation.name, number, text, tags=list(conversation.tags),
                                                   at=turn.trigger.describe(), label=turn.name),
                          previous=running)
            for at, action in self._actions(turn):
                if at.before_turn:
                    self._thread(new, self._act, new, running, at, action)
            for at, item in turn.during:
                if at.before_turn:
                    self._thread(new, self._aside, new, at, item, conversation)
            moment = self._moment(turn.trigger, running, running.since if running else start)
            if moment is None:
                new.result.fail("moment", f"its moment never came ({turn.trigger.describe()})")
                moment = time.monotonic()
            if turn.when is not None:
                last = assistant.log.last("chat.done") or {}
                if not turn.when(last.get("text", "")):
                    new.result.skipped = "the reply before did not call for it"
                    results.append(new.result)
                    continue
            self._say(new, moment, conversation)
            for at, action in self._actions(turn):
                if not at.before_turn:
                    self._thread(new, self._act, new, new, at, action)
            for at, item in turn.during:
                if not at.before_turn:
                    self._thread(new, self._aside, new, at, item, conversation)
            if running is not None:
                running.merged_into_next = turn.expectation in ("continued", "either")
                results.append(self._finish(running, moment))
            running = new
        if running is not None:
            end = self._complete(running)
            for thread in running.threads:
                thread.join(timeout=max(0.0, end + self._timeout - time.monotonic()))
            results.append(self._finish(running, max(end, time.monotonic())))
        # A skipped turn is noted when it would have been said, before the
        # turn ahead of it is judged
        results.sort(key=lambda r: r.turn)
        return results

    @staticmethod
    def _actions(turn: Turn) -> list:
        """(At, action) for each of the turn's actions; an action given
        without a moment happens once the turn before has finished."""
        out = []
        for entry in turn.all_actions():
            if isinstance(entry, tuple) and len(entry) == 2 and (entry[0] is None or isinstance(entry[0], At)):
                at, action = entry
            else:
                at, action = None, entry
            out.append((at or At.finished(0.0), action))
        return out

    @staticmethod
    def _thread(owner: Running, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        owner.threads.append(thread)
        thread.start()

    # When things happen ----------------------------------------------------------

    def _reply_clips(self, after: float, end: float = float("inf")) -> list[Clip]:
        """The clips of the reply after a moment, chimes left out."""
        return [c for c in self.assistant.endpoint.speaker.clips_between(after, end) if not chime_of(c)]

    def _reply_sounds(self, after: float, end: float = float("inf")) -> list[tuple[float, float]]:
        spans = []
        for clip in self._reply_clips(after, end):
            spans += [(max(a, after), min(b, end)) for a, b in clip.sounds() if b > after and a < end]
        return merge(spans, 0.25)

    def _first_reply_sound(self, after: float, end: float = float("inf")) -> Optional[float]:
        sounds = self._reply_sounds(after, end)
        return sounds[0][0] if sounds else None

    def _expects_reply(self, running: Running) -> bool:
        expectation = running.turn.expectation
        if running.merged_into_next or expectation in ("silent", "silenced", "sleep", "unanswered"):
            return False
        if expectation == "wake":
            return len(words(running.result.said)) > 2
        return True

    def _last_clip(self, running: Running) -> Optional[Clip]:
        """The reply's last clip, once Iris has nothing more to play: its
        reply is complete (when one is expected) and nothing is waiting to
        be synthesized or played. It must hold a moment, so a piece between
        the queues is not missed."""
        log = self.assistant.log
        queued = getattr(self.assistant, "voice_queued", lambda: False)
        after = running.ref_end
        expects = self._expects_reply(running)
        steady = [None]

        def ready() -> bool:
            done = not expects or bool(log.since(after, "chat.done"))
            clips = self.assistant.endpoint.speaker.clips_between(after)
            if not (done and clips and not queued()):
                steady[0] = None
                return False
            if steady[0] is None or steady[0][0] is not clips[-1]:
                steady[0] = (clips[-1], time.monotonic())
            return time.monotonic() - steady[0][1] >= 0.25

        if not wait_until(ready, self._timeout, 0.02):
            return None
        return self.assistant.endpoint.speaker.clips_between(after)[-1]

    def _answered_moment(self, running: Running, gap: float) -> Optional[float]:
        """Once the reply has been spoken to its end (its last clip given,
        the assistant done speaking, the speaker quiet), and at least gap
        after its last sound; None if it never was."""
        last = self._last_clip(running)
        if last is None:
            return None
        speaker = self.assistant.endpoint.speaker
        speaking = getattr(self.assistant, "speaking", lambda: False)
        if not wait_until(lambda: not speaking() and not speaker.audible(), self._timeout, 0.02):
            return None
        heard = speaker.last_sound(running.ref_end)
        end = heard if heard is not None else last.started + sound_span(last.audio)[1]
        return max(time.monotonic(), end + gap)

    def _gap_moment(self, running: Running, at: At) -> Optional[float]:
        """The n-th pause of at least min_silence inside the reply's clips,
        from after seconds into the reply on, found in the clips' audio
        before it is played."""
        log = self.assistant.log
        after = running.ref_end
        if not wait_until(lambda: self._first_reply_sound(after) is not None, self._timeout):
            return None
        first = self._first_reply_sound(after)
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            silences = []
            for clip in self._reply_clips(after):
                silences += [(clip.started + a, clip.started + b) for a, b in inner_silences(clip.audio, at.min_silence)
                             if clip.started + a < clip.end]
            silences = [s for s in sorted(silences) if s[0] >= first + at.after]
            if len(silences) >= at.n:
                return silences[at.n - 1][0] + 0.03
            done = log.since(after, "chat.done")
            if done and finished(self.assistant):
                return None
            time.sleep(0.05)
        return None

    def _after_event(self, at: At, since: float) -> Optional[float]:
        log = self.assistant.log

        def found():
            return [e for e in log.since(since, at.name) if at.where is None or at.where(e[2])]

        if not wait_until(lambda: len(found()) >= at.n, self._timeout):
            return None
        if at.name != "wake.window":
            return found()[at.n - 1][0] + at.seconds

        # The awake window's end, followed as it is started over
        def window_end() -> Optional[float]:
            open_ = [e for e in log.since(since, "wake.window") if e[2].get("remaining", 0) > 0]
            return open_[-1][0] + open_[-1][2]["remaining"] + at.seconds if open_ else None

        target = window_end()
        while target is not None and time.monotonic() < target - 0.5:
            time.sleep(0.05)
            target = window_end() or target
        return target

    def _after_state(self, at: At, since: float) -> Optional[float]:
        log = self.assistant.log

        def holding_since() -> Optional[float]:
            state = log.since(since, "state")
            if state and state[-1][2].get("state") == at.name:
                return state[-1][0]
            flags = log.since(since, "flags")
            began = None
            for moment, _, data in flags:
                on = bool((data.get("flags") or {}).get(at.name))
                began = (began or moment) if on else None
            return began

        def held() -> bool:
            began = holding_since()
            return began is not None and time.monotonic() - began >= at.seconds

        if not wait_until(held, self._timeout, 0.02):
            return None
        return holding_since() + at.seconds

    def _moment(self, at: At, ref: Optional[Running], since: float) -> Optional[float]:
        """When something happens (see script.At), relative to a turn: the
        moment, which may be ahead (a reply's end is known before it is
        heard), or None if it never came."""
        kind, seconds = at.kind, at.seconds
        now = time.monotonic
        if kind == "finished":
            if ref is not None:
                self._complete(ref)
            else:
                wait_until(lambda: finished(self.assistant), self._timeout)
            return now() + seconds
        if kind == "start":
            return self._start + seconds
        if kind == "event":
            return self._after_event(at, since)
        if kind == "state":
            return self._after_state(at, since)
        if ref is None:
            return now() + seconds
        if kind == "speech_end":
            return ref.ref_end + seconds
        if kind == "speech_start":
            return ref.ref_start + seconds
        after = ref.ref_end
        if kind == "reply_audible":
            if not wait_until(lambda: (self._first_reply_sound(after) or float("inf")) < now(), self._timeout):
                return None
            first = self._first_reply_sound(after)
            speaker = self.assistant.endpoint.speaker
            if not wait_until(lambda: speaker.audible() and now() - first >= seconds, self._timeout, 0.02):
                return None
            return now()
        if kind == "answered":
            return self._answered_moment(ref, seconds)
        if kind in ("reply_end", "before_end"):
            last = self._last_clip(ref)
            if last is None:
                return None
            end = last.started + sound_span(last.audio)[1]
            return end + seconds if kind == "reply_end" else end - seconds
        if kind == "in_gap":
            return self._gap_moment(ref, at)
        raise ValueError(f"No moment called {kind!r}")

    def _complete(self, running: Running) -> float:
        """Wait until the turn is over as far as it is concerned: answered
        and quiet, watched in silence, or stopped. When it was."""
        with running.lock:
            if running.completed is not None:
                return running.completed
            log, assistant = self.assistant.log, self.assistant
            turn = running.turn
            expectation = turn.expectation
            if expectation in ("silent", "sleep") or (expectation == "wake" and not self._expects_reply(running)):
                time.sleep(max(0.0, running.ref_end + NO_REPLY_WATCH - time.monotonic()))
                wait_until(lambda: finished(assistant), self._timeout)
            elif expectation == "unanswered":
                # Over once the turn was heard and what went wrong was shown
                def told() -> bool:
                    users = log.since(running.ref_end, "chat.user")
                    return bool(users) and bool(log.since(users[-1][0], "error")) and finished(assistant)

                if not wait_until(told, self._timeout):
                    running.result.fail("complete", f"no error within {self._timeout:.0f} s")
            elif expectation == "silenced" or running.merged_into_next:
                wait_until(lambda: finished(assistant), self._timeout)
            elif turn.presses_stop:
                for thread in list(running.threads):
                    thread.join(timeout=self._timeout)
                wait_until(lambda: finished(assistant), 30)
            else:
                source = "text" if turn.typed else "voice"

                def answered() -> bool:
                    users = [e for e in log.since(running.ref_end, "chat.user")
                             if e[2].get("source", "voice") == source]
                    return bool(users) and bool(log.since(users[-1][0], "chat.done")) and finished(assistant)

                if not wait_until(answered, self._timeout):
                    running.result.fail("complete", f"no complete reply within {self._timeout:.0f} s")
            running.completed = time.monotonic()
            return running.completed

    # Saying, doing ---------------------------------------------------------------

    def _say(self, running: Running, moment: float, conversation: Conversation):
        turn = running.turn
        self._threads_before[running.number] = threading.active_count()
        if turn.typed:
            time.sleep(max(0.0, moment - time.monotonic()))
            running.sent = running.since = time.monotonic()
            self.assistant.send_text(running.result.said)
            return
        audio = self._utterance(turn, conversation)
        lead = sound_span(audio)[0]
        running.said = self.space.say(running.result.said, audio, delay=max(0.0, moment - time.monotonic() - lead))
        running.since = running.said.start
        late = running.said.speech_start - moment
        if late > 0.05:
            running.result.late = round(late, 3)

    def _act(self, owner: Running, ref: Optional[Running], at: At, action):
        """Do an action at its moment (see script.Turn.actions)."""
        since = ref.since if ref is not None else self._start
        if ref is owner:
            wait_until(lambda: owner.since > 0, self._timeout)
            since = owner.since
        moment = self._moment(at, ref, since)
        name = action if isinstance(action, str) else action[0]
        if moment is None:
            owner.result.fail("action", f"{name} never came ({at.describe()})")
            return
        time.sleep(max(0.0, moment - time.monotonic()))
        args = [] if isinstance(action, str) else list(action[1:])
        self.assistant.log(f"action.{name}", {"args": args})
        owner.actions.append((time.monotonic(), name))
        assistant = self.assistant
        if name == "stop":
            owner.pressed.append(time.monotonic())
            assistant.interrupt()
        elif name == "pause":
            self._touched["paused"] = ("paused", False)
            assistant.pause()
        elif name == "resume":
            assistant.resume()
        elif name == "send":
            assistant.send_text(args[0])
        elif name == "mic":
            self._touched.setdefault("mic", ("mic", True))
            assistant.configure("mic", bool(args[0]))
        elif name == "persona":
            self._touched.setdefault("persona", ("persona", getattr(assistant, "persona", "iris")))
            assistant.configure("persona", args[0])
        elif name == "tool":
            plugin, on = args
            self._touched.setdefault(("tool", plugin), ("tool", (plugin, DEFAULT_TOOLS.get(plugin, False))))
            if plugin == "web" and on and self.web is None:
                self.web = FakeWeb(real=self.real_web or None).__enter__()
            if plugin == "camera" and on and self.camera is None:
                self.camera = FakeCamera().__enter__()
            if plugin == "screen" and on and self.screen is None:
                self.screen = FakeScreen().__enter__()
            assistant.configure("tool", (plugin, bool(on)))
        elif name == "turn_detection":
            self._touched.setdefault("turn_detection", ("turn_detection", "smart_turn"))
            assistant.configure("turn_detection", args[0])
        elif name == "model":
            self._touched.setdefault("model", ("model", getattr(assistant, "options", {}).get("llm_model")))
            assistant.configure("model", args[0])
        elif name == "stub" and self.stub is not None:
            before = self._configure_stub(args[0])
            self._saved["stub"] = {**before, **self._saved.get("stub", {})}
        else:
            owner.result.fail("action", f"no action called {name!r}")

    def _aside(self, owner: Running, at: At, item, conversation: Conversation):
        """Say or play something during a turn, without waiting for Iris.
        At the start of the conversation or once the turn before has
        finished, it comes before the turn is said."""
        if at.before_turn:
            moment = self._moment(at, owner.previous, owner.previous.since if owner.previous else self._start)
        else:
            wait_until(lambda: owner.since > 0, self._timeout)
            moment = self._moment(at, owner, owner.since)
        if moment is None:
            owner.result.fail("aside", f"an aside's moment never came ({at.describe()})")
            return
        if isinstance(item, Say):
            audio = self.voice_for(item.voice or conversation.voice).utterance(item.parts, splice=item.splice)
            audio = audio * item.gain
            lead = sound_span(audio)[0]
            said = self.space.say(item.text, audio, delay=max(0.0, moment - time.monotonic() - lead), queued=False)
            owner.asides.append(Aside(item, said.speech_start, said.speech_end))
            self.assistant.log("aside", {"say": item.text})
        elif isinstance(item, Sound):
            audio = sound(item.name, item.seconds, voices=self._sound_voices)
            self.space.add_source(audio, at=moment, gain=item.gain, name=item.name)
            owner.asides.append(Aside(item, moment, moment + len(audio) / RATE))
            self.assistant.log("aside", {"sound": item.name})

    def _silence(self, silence: Silence, number: int, conversation: Conversation) -> TurnResult:
        """Nobody speaks for a while, and nothing is to happen."""
        log, speaker = self.assistant.log, self.assistant.endpoint.speaker
        wait_until(lambda: finished(self.assistant), self._timeout)
        result = TurnResult(conversation.name, number, silence.text, tags=list(conversation.tags), at="silence")
        start = time.monotonic()
        time.sleep(silence.seconds)
        end = time.monotonic()
        for at, kind, data in log.since(start, None, end):
            if kind in silence.kinds:
                result.fail("silence", f"{kind} in the silence, {at - start:.1f} s in: {data}")
        heard = speaker.sound_heard(start, end)
        if heard > 0:
            result.fail("silence", f"the speaker played {heard:.1f} s in the silence")
        if any(d.get("state") == "understanding" for _, _, d in log.since(start, "state", end)):
            result.fail("silence", "it set about understanding something in the silence")
        if self.resources is not None:
            result.resources = self.resources.sample(silence.text)
        return result

    def _restart(self, restart: Restart, number: int, conversation: Conversation) -> TurnResult:
        """Quit the assistant and start it again, once it has finished; the
        conversation it shows again must hold what restart.restores names,
        and the replies as well as the user's turns."""
        log = self.assistant.log
        wait_until(lambda: finished(self.assistant), self._timeout)
        result = TurnResult(conversation.name, number, restart.text, tags=list(conversation.tags), at="restart",
                            label=restart.name)
        start = time.monotonic()
        log("action.restart", {"args": []})
        self.assistant.restart()
        shown = log.since(start, "chat.user") + log.since(start, "chat.done")
        result.users = [(round(at - start, 3), d.get("source", "voice"), d.get("text", ""))
                        for at, kind, d in shown if kind == "chat.user"]
        result.replies = [(round(at - start, 3), d.get("text", ""), bool(d.get("interrupted")))
                          for at, kind, d in shown if kind == "chat.done"]
        texts = [d.get("text", "").lower() for _, _, d in shown]
        missing = [w for w in restart.restores if not any(w.lower() in text for text in texts)]
        if missing:
            result.fail("restored", f"the conversation shown again lacks {missing}: {texts}")
        if restart.restores and not (result.users and result.replies):
            result.fail("restored", f"the conversation shown again has {len(result.users)} turns of the user's "
                                    f"and {len(result.replies)} replies")
        result.errors = [d.get("message", "") for _, _, d in log.since(start, "error")]
        for error in result.errors:
            result.fail("error", f"error: {error}")
        if self.resources is not None:
            result.resources = self.resources.sample(restart.text)
        return result

    # Measuring and judging -------------------------------------------------------

    def _finish(self, running: Running, end: float) -> TurnResult:
        if running.asides:
            last = max(a.start for a in running.asides)
            time.sleep(max(0.0, last + ASIDE_WATCH - time.monotonic()))
            end = max(end, last + ASIDE_WATCH)
        self._moments[running.number] = (running.since, running.ref_end)
        self._measure(running, end)
        self._judge(running, View(self, running, end))
        self._expected_in_turn(running)
        if self.resources is not None:
            running.result.resources = self.resources.sample(f"turn {running.number}")
        return running.result

    def _measure(self, running: Running, end: float):
        result, turn = running.result, running.turn
        log, speaker = self.assistant.log, self.assistant.endpoint.speaker
        start, ref = running.since, running.ref_end

        def rel(moment: float) -> float:
            return round(moment - ref, 3)

        users = log.since(start, "chat.user", end)
        result.users = [(rel(at), d.get("source", "voice"), d.get("text", "")) for at, _, d in users]
        if running.previous is not None and turn.expectation == "either":
            running.merged = bool(log.since(running.previous.since, "chat.continued", end))
        source = "text" if turn.typed else "voice"
        own = [(at, d) for at, _, d in users if d.get("source", "voice") == source]
        result.continued = len(log.since(start, "chat.continued", end))
        result.errors = [d.get("message", "") for _, _, d in log.since(start, "error", end)]
        if not turn.typed:
            early = any(at < running.said.speech_end for at, _ in own)
            went_on = result.continued > turn.allow_continued and turn.expectation not in ("continued", "either")
            result.taken_early = early or went_on
        dones = log.since(start, "chat.done", end)
        result.replies = [(rel(at), d.get("text", ""), bool(d.get("interrupted"))) for at, _, d in dones]
        if own:
            heard_at, data = own[-1]
            result.heard = data.get("text", "")
            result.language = data.get("language", "")
            language = turn.reply_language or result.language or "en"
            result.heard_share = round(heard_share(result.heard, running.judged_text, language), 2)
            result.heard_precision = round(heard_precision(result.heard, running.judged_text), 2)
            after = [d for at, _, d in dones if at >= heard_at]
            if after:
                result.reply = after[-1].get("text", "")
        first = self._first_reply_sound(ref, end)
        result.spoke = first is not None and first <= min(end, time.monotonic())
        if result.spoke:
            result.first_word = rel(first)
            result.audible = round(sum(b - a for a, b in self._reply_sounds(first, end)), 3)
        timings = log.since(ref, "turn.timing", end)
        if timings:
            result.stages = timings[-1][2].get("stages", {})
            result.components = timings[-1][2].get("components", {})
            if result.first_word is not None:
                pipeline = result.first_word - (result.stages.get("reply") or 0) / 1000
                if result.components.get("turn_end") == "silence":
                    pipeline -= (result.stages.get("turn_end") or 0) / 1000
                result.voice_pipeline = round(pipeline, 3)
        if not turn.typed and self._voice_playing(running.said.speech_start):
            silent = speaker.silent_after(running.said.speech_start)
            result.barge_in = None if silent is None else round(max(0.0, silent - running.said.speech_start), 3)
        for pressed in running.pressed:
            # Audible just before it was pressed: the press cut the voice
            if speaker.audible(pressed - 0.05):
                silent = speaker.silent_after(pressed)
                result.stop = None if silent is None else round(max(0.0, silent - pressed), 3)
                break
        acted = [moment for moment, _ in running.actions]
        if result.spoke:
            for clip in self._reply_clips(first, end):
                cut = clip.stopped
                if cut is None or cut >= end or any(abs(cut - moment) < 1.0 for moment in acted):
                    continue
                if not any(self.space.voice_speaking(cut + d) for d in (-0.3, 0.0, 0.3)):
                    result.self_cuts += 1
        result.tool_uses = [(rel(at), d.get("tool", ""), d.get("detail", ""))
                            for at, _, d in log.since(start, "tool.use", end)]
        result.flags = {f: [(rel(a), rel(b) if b != float("inf") else None) for a, b in spans]
                        for f, spans in flag_windows(log.since(0, "flags", end), start, end).items()}
        timeline = state_timeline(log.since(0, "state", end), start, end)
        result.states = [(rel(at), state) for at, state in timeline]
        if self.stub is not None:
            result.requests = [
                {"at": rel(r.received), "kind": r.kind, "tools": r.tools, "users": len(r.users()),
                 "tool_messages": len(r.tool_messages()), "closed": rel(r.closed) if r.closed else None,
                 "tool_call_sent": rel(r.tool_call_sent) if r.tool_call_sent else None}
                for r in self.stub.requests_between(start, end)
            ]
        if self.web is not None:
            result.web_calls = [
                {"kind": c.kind, "query": c.query, "start": rel(c.start), "end": rel(c.end) if c.end else None,
                 "outcome": c.outcome, "tries": c.tries}
                for c in self.web.between(start, end)
            ]
        result.model_rounds = len(log.since(start, "model.ask", end))
        if running.said is not None:
            endpoint = self.assistant.endpoint
            spans = subtract(running.said.spans(), endpoint.open_spans(running.said.start, end))
            result.deaf_spans = [(rel(a), rel(b)) for a, b in spans]
            result.deaf = round(sum(b - a for a, b in spans), 3)
        result.verdicts = [(rel(at), d.get("probability")) for at, _, d in log.since(start, "turn.verdict", end)]
        clips = speaker.clips_between(start, end)
        result.holds = [(rel(h.at), h.silence, h.rewind) for h in holds([c for c in clips if not chime_of(c)])]
        result.voice = [(rel(at), d.get("message", "")) for at, _, d in log.since(start, "voice", end)]
        result.chimes = [(rel(c.started), chime_of(c)) for c in clips if chime_of(c)]
        reply_sounds = self._reply_sounds(ref, end)
        result.busy_silence = longest_busy_silence(timeline, reply_sounds, ref, end)
        result.glued = any(GLUED.search(text) for _, text, _ in result.replies)
        result.asides = [(rel(a.start), getattr(a.item, "text", None) or getattr(a.item, "name", ""))
                         for a in running.asides]
        result.threads = sorted(t.name for t in threading.enumerate())

    def _voice_playing(self, moment: float) -> bool:
        """Whether a clip of the reply was playing at a moment (in one of its
        pauses too)."""
        return any(c.started <= moment < c.end for c in self._reply_clips(moment - 0.01, moment + 0.01))

    def _judge(self, running: Running, view: View):
        result, turn = running.result, running.turn
        fail = result.fail
        budget = self.budgets
        expectation = turn.expectation
        log = self.assistant.log
        start, end = running.since, view.end
        if running.merged_into_next:
            pass
        elif expectation == "silent":
            if result.heard or result.spoke:
                fail("silent", f"answered something not addressed to it (heard: {result.heard!r})")
        elif expectation == "unanswered":
            # Heard, but no reply could be had: what went wrong is shown
            # (error_mentions) and nothing is said
            if not result.heard:
                fail("heard", "did not hear the turn")
            if result.spoke:
                fail("spoke", "spoke a reply it could not have")
        elif expectation == "sleep":
            if not log.since(start, "sleep", end):
                fail("sleep", "did not fall asleep")
            if result.heard:
                fail("sleep", f"took it for a turn: {result.heard!r}")
            if result.spoke:
                fail("sleep", "answered it")
        elif expectation == "silenced":
            self._judge_barge_in(result, budget, fail)
        elif expectation == "shown":
            self._judge_shown(running, view)
        else:
            if expectation == "wake" and not log.since(start, "wake", end):
                fail("wake", "did not wake")
            if expectation == "wake" and not self._expects_reply(running):
                if result.heard:
                    fail("wake", f"took the wake word for a turn: {result.heard!r}")
                if result.spoke:
                    fail("wake", "answered the bare wake word")
            else:
                self._judge_answer(running, view)
        self._judge_words(running)
        self._judge_tools(running, view)
        if turn.turn_end is not None:
            measured = result.stages.get("turn_end")
            lo, hi = turn.turn_end
            if measured is None:
                fail("turn_end", "no turn end was measured")
            elif not lo <= measured / 1000 <= hi:
                fail("turn_end", f"the turn's end took {measured / 1000:.2f} s to hear (expected {lo} to {hi} s)")
        for word in turn.reply_excludes:
            for _, text, interrupted in result.replies:
                if not interrupted and word.lower() in text.lower():
                    fail("excludes", f"a reply mentions {word!r}: {text[:120]!r}")
        if turn.reply_language and result.language and result.language != turn.reply_language:
            fail("language", f"heard in {result.language}, not {turn.reply_language}")
        if turn.chime and turn.chime not in [name for _, name in result.chimes]:
            fail("chime", f"no {turn.chime} chime")
        if turn.history is not None:
            requests = view.requests()
            messages = requests[0].messages if requests else view.history()
            verdict = turn.history(messages)
            if verdict is not True:
                fail("history", verdict if isinstance(verdict, str)
                     else "the conversation the model was given is not as expected")
        if turn.deaf_limit is not None and result.deaf > turn.deaf_limit:
            fail("deaf", f"{result.deaf:.2f} s of the user's speech went unheard (no microphone open), "
                         f"at {result.deaf_spans}")
        for aside in running.asides:
            if getattr(aside.item, "continues", False):
                continue
            reacted = [k for _, k, _ in log.since(aside.start, None, aside.start + ASIDE_WATCH)
                       if k in ("chat.user", "chat.continued")]
            if reacted:
                what = getattr(aside.item, "text", None) or getattr(aside.item, "name", "")
                fail("aside", f"took {what!r} for a turn ({', '.join(reacted)})")
        for check in turn.checks:
            try:
                problem = check.fn(result, view)
            except Exception as error:
                problem = f"the check failed to run: {error!r}"
            if problem:
                fail(check.name, problem)
        expected = [e for e in result.errors if any(w.lower() in e.lower() for w in turn.error_mentions)]
        if turn.error_mentions and not expected:
            fail("error", f"no error mentions any of {turn.error_mentions}")
        for error in result.errors:
            if error not in expected:
                fail("error", f"error: {error}")

    def _judge_barge_in(self, result: TurnResult, budget: dict, fail):
        if result.barge_in is None:
            fail("barge_in", "the voice did not stop when the user spoke")
        elif result.barge_in > budget["barge_in"]:
            fail("barge_in", f"the voice stopped {result.barge_in:.2f} s after the user began "
                             f"(budget {budget['barge_in']} s)")

    def _judge_answer(self, running: Running, view: View):
        result, turn = running.result, running.turn
        fail, budget = result.fail, self.budgets
        log = self.assistant.log
        if not result.heard:
            fail("heard", "did not hear the turn")
        elif result.heard_share < HEARD:
            fail("heard", f"heard {result.heard!r}")
        if turn.expectation == "continued":
            since = running.previous.since if running.previous else running.since
            if not log.since(since, "chat.continued", view.end):
                fail("continued", "the turn was not taken as going on with the one before (no chat.continued)")
        if turn.expectation == "either" and running.previous is not None:
            merged = bool(log.since(running.previous.since, "chat.continued", view.end))
            if not merged:
                answered_first = sum(b - a for a, b in self._reply_sounds(running.previous.ref_end, running.ref_end))
                if answered_first > EITHER_ANSWER:
                    fail("either", f"{answered_first:.1f} s of an answer to the first part alone")
        if result.taken_early:
            fail("taken_early", "took the turn before the user had finished")
        if not result.spoke:
            fail("spoke", "no spoken reply")
        cut = view.end < (running.completed or float("inf"))
        if turn.reply_mentions and result.reply and not cut and (self.check_replies or turn.check_with_stub):
            if not any(w.lower() in result.reply.lower() for w in turn.reply_mentions):
                fail("mentions", f"the reply mentions none of {turn.reply_mentions}: {result.reply[:120]!r}")
        pipeline = result.voice_pipeline
        if pipeline is not None and pipeline > budget["voice_pipeline"]:
            fail("voice_pipeline", f"voice pipeline {pipeline:.2f} s (budget {budget['voice_pipeline']} s)")
        language = result.language
        if language and language not in self._languages:
            if self._languages and result.first_word is not None and result.first_word > budget["new_language"]:
                fail("new_language", f"the first reply in {language} took {result.first_word:.2f} s "
                                     f"(budget {budget['new_language']} s)")
            self._languages.add(language)
        if turn.first_word is not None and result.first_word is not None and result.first_word > turn.first_word:
            fail("first_word", f"the first word came {result.first_word:.2f} s after the user stopped "
                               f"(at most {turn.first_word} s)")
        if turn.interrupt_after is not None or turn.trigger.kind in ("reply_audible", "in_gap", "before_end"):
            self._judge_barge_in(result, budget, fail)
        if turn.presses_stop and running.pressed:
            if result.stop is None and any(self.assistant.endpoint.speaker.audible(p - 0.05) for p in running.pressed):
                fail("stop", "the voice did not stop on Stop")
            elif result.stop is not None and result.stop > budget["stop"]:
                fail("stop", f"Stop took {result.stop:.2f} s (budget {budget['stop']} s)")
        if turn.listen_through and result.self_cuts:
            fail("self_cuts", f"the voice was cut {result.self_cuts} times with nobody speaking")

    def _judge_shown(self, running: Running, view: View):
        """Answered in writing only: the reply is in a language the persona's
        voice does not speak, no other voice says it, and the user was told
        so (once for the voice and language, so in this turn or before)."""
        result, turn = running.result, running.turn
        fail = result.fail
        if not result.heard:
            fail("heard", "did not hear the turn")
        elif result.heard_share < HEARD:
            fail("heard", f"heard {result.heard!r}")
        if not result.reply:
            fail("shown", "no reply was shown")
        if result.spoke:
            fail("spoke", "spoke a reply its voice does not speak")
        language = turn.reply_language or result.language
        told = [d for _, _, d in self.assistant.log.since(0, "notice", view.end)
                if d.get("kind") == "unspoken" and d.get("language") == language]
        if not told:
            fail("notice", f"not told that replies in {language} are not spoken")

    def _judge_words(self, running: Running):
        result, turn = running.result, running.turn
        fail, heard = result.fail, result.heard
        missing = [w for w in turn.must_hear if not contains(heard, w)]
        if missing:
            fail("must_hear", f"did not hear {missing} in {heard!r}")
        if turn.must_hear_in_order and not in_order(heard, turn.must_hear_in_order):
            fail("order", f"did not hear {turn.must_hear_in_order} in order in {heard!r}")
        extra = [w for w in turn.must_not_hear if contains(heard, w)]
        if extra:
            fail("must_not_hear", f"heard {extra} in {heard!r}")
        if turn.exact and not same_words(heard, running.judged_text):
            fail("exact", f"heard {normalized(heard)!r}, not {normalized(running.judged_text)!r}")
        missing_terms = [t for t in turn.heard_terms if not term_found(heard, t)]
        if missing_terms:
            fail("terms", f"{missing_terms} not found in {normalized(heard)!r}")

    def _judge_tools(self, running: Running, view: View):
        result, turn = running.result, running.turn
        fail = result.fail
        used = [tool for _, tool, _ in result.tool_uses]
        if turn.tools_used is not None and used != list(turn.tools_used):
            fail("tools", f"used {used or 'no tools'}, expected {turn.tools_used or 'none'}")
        if turn.search_mentions:
            details = [detail for _, _, detail in result.tool_uses]
            if not any(all(term_found(detail, m) for m in turn.search_mentions) for detail in details):
                fail("search_mentions", f"no tool was called on {turn.search_mentions}: {details}")
        if turn.model_requests is not None and self.stub is not None:
            requests = [r for r in view.requests() if r.kind != "summary"]
            if len(requests) != turn.model_requests:
                fail("model_requests", f"{len(requests)} requests to the model, expected {turn.model_requests}")

    def _expected_in_turn(self, running: Running):
        """Move failures known today to the expected ones."""
        result = running.result
        conversation = self._conversation
        if conversation is None or not conversation.expected_failures:
            return
        kept, kept_checks = [], []
        for check, message in zip(result.failed, result.failures):
            if conversation.expected(running.number, check):
                result.expected.append(f"{check}: {message}")
            else:
                kept.append(message)
                kept_checks.append(check)
        result.failures, result.failed = kept, kept_checks
        for entry in conversation.expected_failures:
            turn, colon, name = entry.partition(":")
            if colon and turn == f"T{running.number}" and name != "*":
                if not any(e.startswith(f"{name}: ") for e in result.expected):
                    result.now_passes.append(entry)

    def _expected_globally(self, results: list[TurnResult], conversation: Conversation):
        """A check known to fail in any turn that failed in none now passes."""
        for entry in conversation.expected_failures:
            if ":" in entry:
                continue
            if results and not any(any(e.startswith(f"{entry}: ") for e in r.expected) for r in results):
                results[-1].now_passes.append(entry)

    # Afterwards ----------------------------------------------------------------

    def _transcribe(self, results: list[TurnResult], conversation: Conversation):
        """What each reply's voice said, as Iris's own recognizer hears it,
        once the conversation is over (doing it meanwhile would disturb the
        timing)."""
        turns = {n: t for n, t in enumerate(conversation.turns, 1) if isinstance(t, Turn)}
        wanted = self.transcribe_replies or any(
            t.reply_language or t.spoken_once or t.spoken_excludes or t.spoken_share is not None
            or t.spoken_numbers for t in turns.values()
        )
        if not wanted or not hasattr(self.assistant, "transcribe"):
            return
        speaker = self.assistant.endpoint.speaker
        by_number = {r.turn: r for r in results if r.turn in turns and not r.skipped and r.turn in self._moments}
        numbers = sorted(by_number)
        for index, number in enumerate(numbers):
            result, turn = by_number[number], turns[number]
            if result.first_word is None:
                continue
            first = self._moments[number][1] + result.first_word
            following = [self._moments[n][0] for n in numbers[index + 1 :]]
            stop = following[0] if following else (speaker.last_sound(first, time.monotonic()) or first)
            audio = speaker.recording(first, max(first, stop))
            if not len(audio):
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
                path = file.name
            try:
                sf.write(path, audio, RATE)
                result.spoken, result.spoken_language = self.assistant.transcribe(path)
            finally:
                os.unlink(path)
            spoken = normalized(result.spoken)
            for phrase in turn.spoken_once:
                count = len(re.findall(r"\b" + re.escape(normalized(phrase)) + r"\b", spoken))
                if count != 1:
                    result.fail("spoken_once", f"the voice said {phrase!r} {count} times: {result.spoken!r}")
            for phrase in turn.spoken_excludes:
                if phrase.lower() in result.spoken.lower():
                    result.fail("spoken_excludes", f"the voice said {phrase!r}: {result.spoken!r}")
            if turn.reply_language and result.spoken_language and result.spoken_language != turn.reply_language:
                result.fail("language", f"the reply was spoken in {result.spoken_language}, "
                                        f"not {turn.reply_language}")
            missing = [n for n in turn.spoken_numbers if not re.search(rf"(?<!\d){re.escape(n)}(?!\d)", result.spoken)]
            if missing:
                result.fail("spoken_numbers", f"the voice was not heard saying {missing}: {result.spoken!r}")
            if turn.spoken_share is not None and result.reply:
                share = heard_share(result.spoken, result.reply, turn.reply_language or result.language or "en")
                if share < turn.spoken_share:
                    result.fail("spoken_share", f"the voice was heard saying {share:.0%} of the reply: "
                                                f"{result.spoken!r} for {result.reply!r}")
            self._expected_after(result, number, conversation)

    def _expected_after(self, result: TurnResult, number: int, conversation: Conversation):
        kept, kept_checks = [], []
        for check, message in zip(result.failed, result.failures):
            if conversation.expected(number, check):
                result.expected.append(f"{check}: {message}")
                result.now_passes = [e for e in result.now_passes if e != f"T{number}:{check}"]
            else:
                kept.append(message)
                kept_checks.append(check)
        result.failures, result.failed = kept, kept_checks


def _percentile(values: list[float], share: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round(share * (len(values) - 1)))))
    return values[index]


STAGES = ("turn_end", "recognition", "reply", "voice")
MEASURES = ("first_word", "voice_pipeline", "barge_in", "stop")


def stage_values(result: TurnResult) -> dict:
    """A turn's measurements in seconds: its stages (Iris's own), the first
    word, the voice pipeline, barge-in and Stop, and the time the stages do
    not account for."""
    values = {name: result.stages[name] / 1000 for name in STAGES if result.stages.get(name) is not None}
    for name in MEASURES:
        if getattr(result, name) is not None:
            values[name] = getattr(result, name)
    if result.first_word is not None and values.keys() & set(STAGES):
        values["unaccounted"] = round(result.first_word - sum(values.get(s, 0.0) for s in STAGES), 3)
    return values


def stats(values) -> Optional[dict]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {
        "median": round(median(values), 3),
        "p90": round(_percentile(values, 0.9), 3),
        "worst": round(values[-1], 3),
        "turns": len(values),
    }


def summarize(results: list[TurnResult]) -> dict:
    """Medians, 90th percentiles and worst cases over the turns, overall and
    by tag."""
    held = [r for r in results if not r.skipped]

    def block(turns: list[TurnResult]) -> dict:
        names = STAGES + MEASURES + ("unaccounted",)
        per = [stage_values(r) for r in turns]
        return {name: stats(v.get(name) for v in per) for name in names}

    summary = {
        "turns": len(held),
        "passed": sum(r.passed for r in held),
        "expected_failures": sum(bool(r.expected) for r in held),
        "now_passes": sorted({f"{r.conversation}: {e}" for r in held for e in r.now_passes}),
        "skipped": sorted({f"{r.conversation}: {r.skipped}" for r in results if r.skipped}),
        "taken_early": sum(r.taken_early for r in held),
        # The voice held, and what the holds were judged to be (iris.voice)
        "holds": {key: sum(voice_holds(r.voice)[key] for r in held)
                  for key in ("held", "echo", "aside", "user", "resumed", "not_held")},
        "repeated": sum(1 for r in held for _, _, rewind in r.holds if rewind is not None and rewind > REPEAT_SLACK),
        **block(held),
    }
    tags = sorted({tag for r in held for tag in r.tags})
    if tags:
        summary["by_tag"] = {tag: block([r for r in held if tag in r.tags]) for tag in tags}
    return summary

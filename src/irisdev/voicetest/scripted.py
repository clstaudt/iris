"""Scripted conversations: a user's voice talks to an assistant, turn by turn.

Every turn is measured from the space's ground truth: when the user's last
syllable ended, when the assistant's first word was heard, when its voice
fell silent after the user spoke over it or Stop was pressed. What the
assistant reported (what it heard, its reply, its own timings) is checked
against what was said.
"""

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from irisdev.voicetest.acoustics import Space
from irisdev.voicetest.assistant import AssistantUnderTest, finished, wait_until
from irisdev.voicetest.conversations import Conversation, Turn
from irisdev.voicetest.voices import UserVoice

# What a turn may take, in seconds
BUDGETS = {
    # From the end of the user's speech to the first word, without the time
    # the language model takes for its first sentence
    "voice_pipeline": 2.5,
    # From the user's first word, said while the assistant talks, to silence
    "barge_in": 1.0,
    # From pressing Stop to silence
    "stop": 0.5,
}


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
# The user waits this long after the assistant has finished before speaking
GAP = 0.6
# How long a turn not addressed to the assistant is watched for a reply
NO_REPLY_WATCH = 8.0


NUMBERS = {
    word: str(n)
    for n, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}


def words(text: str) -> list[str]:
    """The words of a text, lower case, numbers as digits ("two" and "2" are
    the same word heard)."""
    return [NUMBERS.get(w, w) for w in re.findall(r"[a-z0-9']+", (text or "").lower())]


def heard_share(heard: str, said: str) -> float:
    """How much of what was said is in what was heard, word by word."""
    expected = words(said)
    if not expected:
        return 1.0
    got = set(words(heard))
    return sum(1 for word in expected if word in got) / len(expected)


@dataclass
class TurnResult:
    conversation: str
    turn: int
    said: str
    heard: str = ""
    heard_share: float = 0.0
    # The assistant took the turn before the user had finished
    taken_early: bool = False
    reply: str = ""
    spoke: bool = False
    # Seconds from the end of the user's speech to the first word heard
    first_word: Optional[float] = None
    # The same without the language model's time to its first sentence
    voice_pipeline: Optional[float] = None
    # The assistant's own measurement, in ms
    stages: dict = field(default_factory=dict)
    barge_in: Optional[float] = None
    stop: Optional[float] = None
    # Times the voice was cut off while nobody spoke, apart from Stop
    self_cuts: int = 0
    errors: list = field(default_factory=list)
    failures: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def line(self) -> str:
        """One line of report, and a line per failure."""
        mark = "ok  " if self.passed else "FAIL"
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
        return " · ".join(parts) + "".join(f"\n         {f}" for f in self.failures)


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
    ):
        """
        Args:
            check_replies: Check what replies mention (off for a scripted model)
            reply_timeout: Seconds a reply may take to be complete
            budgets: Budgets to use instead of BUDGETS, by name
        """
        self.assistant = assistant
        self.space = space
        self.voice = voice
        self.check_replies = check_replies
        self.reply_timeout = reply_timeout
        self.budgets = {**BUDGETS, **(budgets or {})}
        self.report = report
        self._reply_sound: Optional[float] = None
        # When the user's last turn ended
        self._said_end = 0.0

    def hold(self, conversation: Conversation) -> list[TurnResult]:
        self.assistant.clear_conversation()
        self.assistant.set_wake_word(conversation.wake_word or "")
        wait_until(lambda: finished(self.assistant), self.reply_timeout)
        results = []
        for index, turn in enumerate(conversation.turns):
            following = conversation.turns[index + 1] if index + 1 < len(conversation.turns) else None
            hand_over = following is not None and following.interrupt_after is not None
            result = self.turn(conversation.name, index + 1, turn, hand_over)
            self.report(result.line())
            results.append(result)
        wait_until(lambda: finished(self.assistant), self.reply_timeout)
        return results

    def turn(self, conversation: str, number: int, turn: Turn, hand_over: bool) -> TurnResult:
        result = TurnResult(conversation, number, turn.text)
        log, speaker = self.assistant.log, self.assistant.endpoint.speaker
        audio = self.voice.utterance(turn.parts, turn.speed)

        # When the user speaks: over the reply, or once the assistant is done
        if turn.interrupt_after is not None:
            began = self._reply_sound or time.monotonic()
            wait_until(lambda: speaker.audible() and time.monotonic() - began >= turn.interrupt_after,
                       self.reply_timeout)
        elif turn.answer_after is not None:
            since = self._said_end

            def heard_to_its_end() -> bool:
                last = speaker.last_sound(since)
                return (
                    bool(log.since(since, "chat.done"))
                    and last is not None
                    and not speaker.audible()
                    and time.monotonic() - last >= turn.answer_after
                )

            wait_until(heard_to_its_end, self.reply_timeout)
        else:
            wait_until(lambda: finished(self.assistant), self.reply_timeout)
            time.sleep(GAP)
        said = self.space.say(turn.text, audio)
        self._said_end = said.speech_end

        pressed: list[Optional[float]] = [None]
        presser = None
        if turn.stop_after is not None:
            def press():
                if wait_until(lambda: speaker.first_sound_after(said.speech_end) is not None,
                              self.reply_timeout):
                    began = speaker.first_sound_after(said.speech_end)
                    wait_until(lambda: time.monotonic() - began >= turn.stop_after, turn.stop_after + 5)
                    pressed[0] = time.monotonic()
                    self.assistant.interrupt()

            presser = threading.Thread(target=press, daemon=True)
            presser.start()

        def answered() -> bool:
            users = log.since(said.speech_end, "chat.user")
            return bool(users) and bool(log.since(users[-1][0], "chat.done")) and finished(self.assistant)

        if turn.no_reply:
            time.sleep(max(0.0, said.speech_end + NO_REPLY_WATCH - time.monotonic()))
        elif hand_over:
            # Until the reply is heard: a clip starts being heard after the
            # output's latency
            if not wait_until(lambda: (speaker.first_sound_after(said.speech_end) or float("inf"))
                              < time.monotonic(), self.reply_timeout):
                result.failures.append(f"no reply within {self.reply_timeout:.0f} s")
        elif presser is not None:
            presser.join(self.reply_timeout + 10)
            wait_until(lambda: pressed[0] is not None and finished(self.assistant), 30)
        elif not wait_until(answered, self.reply_timeout):
            result.failures.append(f"no complete reply within {self.reply_timeout:.0f} s")
        end = time.monotonic()

        self._measure(result, turn, said, pressed[0], end)
        self._judge(result, turn, hand_over)
        return result

    def _measure(self, result: TurnResult, turn: Turn, said, pressed: Optional[float], end: float):
        log, speaker = self.assistant.log, self.assistant.endpoint.speaker
        users = log.since(said.start, "chat.user", end)
        result.errors = [d.get("message", "") for _, _, d in log.since(said.start, "error", end)]
        result.taken_early = bool(log.since(said.start, "chat.continued", end)) or any(
            at < said.speech_end for at, _, _ in users
        )
        if users:
            result.heard = users[-1][2].get("text", "")
            result.heard_share = round(heard_share(result.heard, turn.text), 2)
            dones = log.since(users[-1][0], "chat.done", end)
            if dones:
                result.reply = dones[-1][2].get("text", "")
        first = speaker.first_sound_after(said.speech_end)
        result.spoke = bool(first is not None and first <= end)
        if result.spoke:
            result.first_word = round(float(first - said.speech_end), 3)
            self._reply_sound = first
        timings = log.since(said.speech_end, "turn.timing", end)
        if timings:
            result.stages = timings[-1][2].get("stages", {})
            if result.first_word is not None:
                model_time = (result.stages.get("reply") or 0) / 1000
                result.voice_pipeline = round(result.first_word - model_time, 3)
        if turn.interrupt_after is not None:
            silent = speaker.silent_after(said.speech_start)
            result.barge_in = None if silent is None else round(max(0.0, silent - said.speech_start), 3)
        if pressed is not None:
            silent = speaker.silent_after(pressed)
            result.stop = None if silent is None else round(max(0.0, silent - pressed), 3)
        if result.spoke:
            for clip in speaker.clips_between(first, end):
                cut = clip.stopped
                if cut is None or cut >= end or (pressed is not None and abs(cut - pressed) < 1.0):
                    continue
                if not any(self.space.voice_speaking(cut + d) for d in (-0.3, 0.0, 0.3)):
                    result.self_cuts += 1

    def _judge(self, result: TurnResult, turn: Turn, hand_over: bool):
        fail = result.failures.append
        if turn.no_reply:
            if result.heard or result.spoke:
                fail(f"answered something not addressed to it (heard: {result.heard!r})")
            return
        if not result.heard:
            fail("did not hear the turn")
        elif result.heard_share < HEARD:
            fail(f"heard {result.heard!r}")
        if result.taken_early:
            fail("took the turn before the user had finished")
        if not result.spoke:
            fail("no spoken reply")
        if self.check_replies and turn.reply_mentions and result.reply and not hand_over:
            if not any(w.lower() in result.reply.lower() for w in turn.reply_mentions):
                fail(f"the reply mentions none of {turn.reply_mentions}: {result.reply[:120]!r}")
        budget = self.budgets
        if result.voice_pipeline is not None and result.voice_pipeline > budget["voice_pipeline"]:
            fail(f"voice pipeline {result.voice_pipeline:.2f} s (budget {budget['voice_pipeline']} s)")
        if turn.interrupt_after is not None:
            if result.barge_in is None:
                fail("the voice did not stop when the user spoke")
            elif result.barge_in > budget["barge_in"]:
                fail(f"the voice stopped {result.barge_in:.2f} s after the user began "
                     f"(budget {budget['barge_in']} s)")
        if turn.stop_after is not None:
            if result.stop is None:
                fail("the voice did not stop on Stop")
            elif result.stop > budget["stop"]:
                fail(f"Stop took {result.stop:.2f} s (budget {budget['stop']} s)")
        if turn.listen_through and result.self_cuts:
            fail(f"the voice was cut {result.self_cuts} times with nobody speaking")
        for error in result.errors:
            fail(f"error: {error}")


def summarize(results: list[TurnResult]) -> dict:
    """Medians and worst cases over the turns."""

    def stats(values):
        values = sorted(v for v in values if v is not None)
        if not values:
            return None
        return {"median": round(values[len(values) // 2], 3), "worst": round(values[-1], 3), "turns": len(values)}

    return {
        "turns": len(results),
        "passed": sum(r.passed for r in results),
        "first_word": stats(r.first_word for r in results),
        "voice_pipeline": stats(r.voice_pipeline for r in results),
        "barge_in": stats(r.barge_in for r in results),
        "stop": stats(r.stop for r in results),
        "taken_early": sum(r.taken_early for r in results),
    }

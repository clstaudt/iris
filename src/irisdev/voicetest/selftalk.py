"""Self-talk: two assistants in one space, talking to each other.

One is asked, in writing, to start a conversation with the other; from then
on each hears the other's voice through its microphone (and its own as an
echo) and answers it. Nothing is scripted, so this tests what scripts miss:
long replies with pauses between their sentences, a synthetic voice to
recognize, two voices that may talk over each other, and the time each
takes to answer, turn after turn.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from irisdev.voicetest.assistant import AssistantUnderTest, finished
from irisdev.voicetest.scripted import heard_share

# A speaker's sounds less than this apart are one reply
REPLY_GAP = 1.2
# The conversation has stalled when both have been idle this long
STALL = 45.0


@dataclass
class Exchange:
    """One reply, and how the other assistant heard it."""

    speaker: str
    listener: str
    # When the listener had heard it
    at: float
    said: str
    heard: str
    heard_share: float
    # Turns the listener made of the reply (more than one: it took the turn
    # in a pause between sentences)
    turns_heard: int
    # Seconds from the end of the reply to the listener's first word
    gap: Optional[float] = None


@dataclass
class SelfTalkResult:
    duration: float
    exchanges: list[Exchange] = field(default_factory=list)
    # Seconds both voices were heard at once
    overlap: float = 0.0
    # Times a voice was cut off while the other was silent (it took its own
    # echo for the other) and while the other spoke (interrupted)
    self_cuts: dict = field(default_factory=dict)
    interrupted: dict = field(default_factory=dict)
    stalled: bool = False
    errors: list = field(default_factory=list)


class SelfTalk:
    def __init__(self, first: AssistantUnderTest, second: AssistantUnderTest,
                 report: Callable[[str], None] = print):
        self.first = first
        self.second = second
        self.report = report

    def run(self, opening: str, seconds: float = 300.0, replies: int = 12) -> SelfTalkResult:
        """Let them talk, after the first is given the opening in writing,
        for so long or so many replies."""
        both = (self.first, self.second)
        for assistant in both:
            assistant.clear_conversation()
            assistant.set_wake_word("")
        start = time.monotonic()
        self.first.send_text(opening)
        idle_since = None
        stalled = False
        while time.monotonic() - start < seconds:
            time.sleep(0.5)
            done = sum(len(a.log.since(start, "chat.done")) for a in both)
            if done >= replies:
                break
            if all(finished(a) for a in both):
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since > STALL:
                    stalled = True
                    break
            else:
                idle_since = None
        for assistant in both:
            assistant.interrupt()
        end = time.monotonic()
        result = self._analyse(start, end)
        result.stalled = stalled
        for exchange in result.exchanges:
            gap = "" if exchange.gap is None else f" · answered after {exchange.gap:.2f} s"
            split = f" · in {exchange.turns_heard} turns" if exchange.turns_heard > 1 else ""
            self.report(f"  {exchange.speaker} → {exchange.listener}: heard {exchange.heard_share:.0%} "
                        f"of the words{split}{gap}")
        return result

    def _analyse(self, start: float, end: float) -> SelfTalkResult:
        result = SelfTalkResult(duration=round(end - start, 1))
        pairs = ((self.first, self.second), (self.second, self.first))
        for speaker, listener in pairs:
            result.errors += [f"{speaker.name}: {d.get('message')}" for _, _, d in speaker.log.since(start, "error", end)]
            # What the listener heard, turn by turn, against what was said before it
            heard_turns = listener.log.since(start, "chat.user", end)
            replies = speaker.log.since(start, "chat.done", end)
            previous = start
            for at, _, data in heard_turns:
                said = " ".join(d.get("text", "") for t, _, d in replies if previous <= t <= at)
                if said.strip():
                    result.exchanges.append(
                        Exchange(
                            speaker=speaker.name,
                            listener=listener.name,
                            at=at,
                            said=said,
                            heard=data.get("text", ""),
                            heard_share=round(heard_share(data.get("text", ""), said), 2),
                            turns_heard=1 + len(listener.log.since(previous, "chat.continued", at)),
                            gap=self._gap(speaker, listener, previous, at),
                        )
                    )
                previous = at
        result.exchanges.sort(key=lambda e: e.at)

        a_sounds = self.first.endpoint.speaker.sounds(start, end, REPLY_GAP)
        b_sounds = self.second.endpoint.speaker.sounds(start, end, REPLY_GAP)
        result.overlap = round(
            sum(max(0.0, min(a1, b1) - max(a0, b0)) for a0, a1 in a_sounds for b0, b1 in b_sounds), 2
        )
        for speaker, other in pairs:
            cuts = [c.stopped for c in speaker.endpoint.speaker.clips_between(start, end)
                    if c.stopped is not None and c.stopped < end - 1]
            spoke = [other.endpoint.speaker.audible(t) for t in cuts]
            result.interrupted[speaker.name] = sum(spoke)
            result.self_cuts[speaker.name] = len(cuts) - sum(spoke)
        return result

    @staticmethod
    def _gap(speaker, listener, since: float, heard_at: float) -> Optional[float]:
        """From the end of the speaker's reply heard at this moment to the
        listener's first word after it."""
        sounds = speaker.endpoint.speaker.sounds(since, heard_at, REPLY_GAP)
        if not sounds:
            return None
        reply_end = sounds[-1][1]
        answer = listener.endpoint.speaker.first_sound_after(reply_end)
        return None if answer is None else round(answer - reply_end, 2)

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
from irisdev.voicetest.scripted import chime_of, heard_share

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
    # Every event each assistant reported, seconds from the start
    logs: dict = field(default_factory=dict)


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
            result.exchanges += self._exchanges(speaker, listener, start, end)
        result.exchanges.sort(key=lambda e: e.at)

        a_sounds = self.first.endpoint.speaker.sounds(start, end, REPLY_GAP)
        b_sounds = self.second.endpoint.speaker.sounds(start, end, REPLY_GAP)
        result.overlap = round(
            sum(max(0.0, min(a1, b1) - max(a0, b0)) for a0, a1 in a_sounds for b0, b1 in b_sounds), 2
        )
        for speaker, other in pairs:
            cuts = [c.stopped for c in speaker.endpoint.speaker.clips_between(start, end)
                    if c.stopped is not None and c.stopped < end - 1 and not chime_of(c)]
            spoke = [other.endpoint.speaker.audible(t) for t in cuts]
            result.interrupted[speaker.name] = sum(spoke)
            result.self_cuts[speaker.name] = len(cuts) - sum(spoke)
        result.logs = {a.name: [{"at": round(at - start, 3), "event": k, **d} for at, k, d in a.log.since(start, end=end)]
                       for a in (self.first, self.second)}
        return result

    def _exchanges(self, speaker, listener, start: float, end: float) -> list[Exchange]:
        """Each of the speaker's replies and how the listener heard it.

        A reply is what the speaker said between two turns it heard; the
        listener heard it in the turns it took meanwhile, of which the last
        is what it answered (a turn taken in a pause and given up when the
        speaker went on is followed by the whole one).
        """
        exchanges = []
        heard_by_speaker = [at for at, _, _ in speaker.log.since(start, "chat.user", end)]
        bounds = [start] + heard_by_speaker + [end]
        for begin, until in zip(bounds, bounds[1:]):
            replies = [d for _, _, d in speaker.log.since(begin, "chat.done", until) if d.get("text", "").strip()]
            sounds = speaker.endpoint.speaker.sounds(begin, until, REPLY_GAP)
            if not replies or not sounds:
                continue
            said = replies[-1]["text"]
            turns = listener.log.since(sounds[0][0], "chat.user", until)
            if not turns:
                continue
            at, _, heard = turns[-1]
            reply_end = sounds[-1][1]
            answer = listener.endpoint.speaker.first_sound_after(reply_end)
            exchanges.append(
                Exchange(
                    speaker=speaker.name,
                    listener=listener.name,
                    at=at,
                    said=said,
                    heard=heard.get("text", ""),
                    heard_share=round(heard_share(heard.get("text", ""), said), 2),
                    turns_heard=len(turns),
                    gap=None if answer is None else round(answer - reply_end, 2),
                )
            )
        return exchanges

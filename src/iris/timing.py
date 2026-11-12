"""Where a turn's time goes, from the end of the user's speech to the first
word of the reply.

A spoken turn passes four stages: the turn end (until Iris is sure the user
has finished), recognition (transcribing what was said), the reply (until
the model has written its first sentence) and the voice (until that sentence
is heard). A typed turn has only the last two. Each turn's timing is sent to
the app as a ``turn.timing`` event, together with the components that took
part, so the app can tell whether a measurement still describes the setup.
"""

import time
from typing import Optional

STAGES = ("turn_end", "recognition", "reply", "voice")


def _ms(seconds: Optional[float]) -> Optional[int]:
    return None if seconds is None else max(0, round(seconds * 1000))


class TurnTiming:
    """The stages of one turn, measured as the turn passes them."""

    def __init__(
        self,
        source: str,
        turn_end: Optional[float] = None,
        recognition: Optional[float] = None,
        components: Optional[dict] = None,
        clock=time.monotonic,
    ):
        """
        Args:
            source: "voice" or "text"
            turn_end: Seconds from the end of speech until the turn was over
            recognition: Seconds the transcription took
            components: {stage: the engine or model that ran it}
            clock: Monotonic clock in seconds (replaceable in tests)
        """
        self.source = source
        self.components = dict(components or {})
        self.stages: dict[str, Optional[int]] = dict.fromkeys(STAGES)
        self.stages["turn_end"] = _ms(turn_end)
        self.stages["recognition"] = _ms(recognition)
        self.reported = False
        self._clock = clock
        self._reply_started: Optional[float] = None
        self._first_piece: Optional[float] = None

    def reply_started(self):
        """The model was asked for the reply."""
        self._reply_started = self._clock()

    def first_piece(self):
        """The first part of the reply that can be spoken is complete."""
        if self._first_piece is not None or self._reply_started is None:
            return
        self._first_piece = self._clock()
        self.stages["reply"] = _ms(self._first_piece - self._reply_started)

    def first_audio(self):
        """The voice started saying the reply."""
        if self._first_piece is None or self.stages["voice"] is not None:
            return
        self.stages["voice"] = _ms(self._clock() - self._first_piece)

    def event(self) -> dict:
        """The measurement as sent to the app."""
        measured = [ms for ms in self.stages.values() if ms is not None]
        return {
            "source": self.source,
            "stages": dict(self.stages),
            "total": sum(measured),
            "components": dict(self.components),
        }

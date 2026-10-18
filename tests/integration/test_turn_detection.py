"""Integration tests for turn-taking on synthesized speech: Silero VAD and
Smart Turn v3 (downloaded on first use, 9 MB).

A synthesized voice ends every sentence on a falling tone, which Smart Turn
hears as finished; the unfinished examples break off where the words alone
make that plain."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from iris.tts import KokoroGenerator
from iris.turns import FRAME, SAMPLE_RATE, SmartTurn, SpeechDetector, TurnTracker

FINISHED = [
    "What's the weather like in Berlin today?",
    "Can you tell me how to make a good risotto?",
    "Thanks, that's all I needed.",
]
UNFINISHED = [
    "I was wondering if you could tell me about the",
    "So what I'd like to do is, um",
]


@pytest.fixture(scope="module")
def speak():
    generator = KokoroGenerator(quiet=True)

    def speak(text: str) -> np.ndarray:
        path = generator.generate(text)
        samples, rate = sf.read(path, dtype="float32")
        Path(path).unlink(missing_ok=True)
        return resample_poly(samples, SAMPLE_RATE, rate).astype(np.float32)

    return speak


@pytest.fixture(scope="module")
def smart_turn():
    model = SmartTurn.load()
    assert model is not None, "Smart Turn could not be downloaded"
    return model


def _turn_end(speech: np.ndarray, smart_turn, max_silence=3.5) -> float | None:
    """Seconds after the end of the speech at which the turn ended."""
    lead = np.zeros(SAMPLE_RATE, dtype=np.float32)
    trail = np.zeros(int(SAMPLE_RATE * (max_silence + 1)), dtype=np.float32)
    signal = np.concatenate([lead, speech, trail])
    tracker = TurnTracker(SpeechDetector(), smart_turn, max_silence)
    for i in range(0, len(signal) - FRAME + 1, FRAME):
        if tracker.feed(signal[i : i + FRAME]):
            return (i + FRAME - len(lead) - len(speech)) / SAMPLE_RATE
    return None


@pytest.mark.integration
class TestSmartTurn:
    @pytest.mark.parametrize("text", FINISHED)
    def test_a_finished_question_ends_the_turn(self, speak, smart_turn, text):
        assert smart_turn(speak(text)) > 0.5

    @pytest.mark.parametrize("text", UNFINISHED)
    def test_a_sentence_broken_off_does_not(self, speak, smart_turn, text):
        assert smart_turn(speak(text)) < 0.5


@pytest.mark.integration
class TestTurnTaking:
    @pytest.mark.parametrize("text", FINISHED)
    def test_answered_a_moment_after_a_finished_question(self, speak, smart_turn, text):
        # Kokoro ends its clips with a little silence of its own
        assert _turn_end(speak(text), smart_turn) < 0.7

    @pytest.mark.parametrize("text", UNFINISHED)
    def test_given_time_to_go_on(self, speak, smart_turn, text):
        assert _turn_end(speak(text), smart_turn) > 3.0

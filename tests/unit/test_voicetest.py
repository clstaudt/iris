"""The voice test's own parts: the acoustic space, the scripted model, and
how turns and self-talk are measured."""

import threading
import time

import numpy as np
import ollama
import pytest

from irisdev.voicetest.acoustics import OUTPUT_LATENCY, RATE, Space, merge, sound_span
from irisdev.voicetest.assistant import Log
from irisdev.voicetest.scripted import heard_share
from irisdev.voicetest.selftalk import SelfTalk
from irisdev.voicetest.stub_llm import LONG, MODEL, SHORT, StubModel, reply_to


def tone(seconds: float, level: float = 0.5, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 220 * t) * level).astype(np.float32)


class TestSpeaker:
    def test_a_clip_is_heard_after_the_latency_until_it_ends(self):
        speaker = Space().endpoint("a").speaker
        before = time.monotonic()
        latency = speaker.play(tone(0.5), RATE)
        clip = speaker.clips[-1]
        assert latency == OUTPUT_LATENCY
        assert before + OUTPUT_LATENCY <= clip.started <= time.monotonic() + OUTPUT_LATENCY
        assert abs(clip.end - clip.started - 0.5) < 0.01
        assert speaker.playing()

    def test_stop_cuts_the_clip_and_a_new_one_ends_the_last(self):
        speaker = Space().endpoint("a").speaker
        speaker.play(tone(2.0), RATE)
        time.sleep(0.2)
        speaker.stop()
        first = speaker.clips[-1]
        assert first.stopped is not None and first.end < first.started + 0.3
        assert not speaker.playing()
        speaker.play(tone(2.0), RATE)
        speaker.play(tone(1.0), RATE)
        assert speaker.clips[-2].stopped is not None

    def test_another_rate_is_resampled(self):
        speaker = Space().endpoint("a").speaker
        speaker.play(tone(1.0, rate=24000), 24000)
        assert abs(len(speaker.clips[-1].audio) - RATE) <= 1

    def test_sounds_first_and_last(self):
        speaker = Space().endpoint("a").speaker
        audio = np.concatenate([np.zeros(RATE // 2, np.float32), tone(0.5), np.zeros(RATE // 2, np.float32)])
        speaker.play(audio, RATE)
        started = speaker.clips[-1].started
        [(a, b)] = speaker.sounds(0, started + 2)
        assert abs(a - (started + 0.5)) < 0.03 and abs(b - (started + 1.0)) < 0.03
        assert abs(speaker.first_sound_after(started) - (started + 0.5)) < 0.03
        assert abs(speaker.last_sound(started, started + 2) - (started + 1.0)) < 0.03
        assert abs(speaker.sound_heard(started, started + 2) - 0.5) < 0.05

    def test_silence_after_a_stop(self):
        speaker = Space().endpoint("a").speaker
        speaker.play(tone(3.0), RATE)
        time.sleep(0.3)
        stopped = time.monotonic()
        speaker.stop()
        silent = speaker.silent_after(stopped - 0.2, limit=2)
        assert silent is not None and stopped - 0.25 < silent < stopped + 0.1


class TestSpace:
    def _hear(self, endpoint, start, seconds):
        return endpoint.space._block(endpoint, start, int(seconds * RATE))

    def test_a_voice_is_heard_by_every_microphone(self):
        space = Space(noise=0)
        a, b = space.endpoint("a"), space.endpoint("b")
        said = space.say("hello", tone(0.5))
        assert sound_span(said.audio) == (pytest.approx(0, abs=0.01), pytest.approx(0.5, abs=0.01))
        for endpoint in (a, b):
            assert np.abs(self._hear(endpoint, said.start, 0.5)).max() == pytest.approx(0.5, abs=0.01)

    def test_a_voice_heard_by_one_only(self):
        space = Space(noise=0)
        a, b = space.endpoint("a"), space.endpoint("b")
        said = space.say("hello", tone(0.5), heard_by={a: 0.5})
        assert np.abs(self._hear(a, said.start, 0.5)).max() == pytest.approx(0.25, abs=0.01)
        assert np.abs(self._hear(b, said.start, 0.5)).max() == 0

    def test_own_speaker_as_echo_and_the_other_across_the_table(self):
        space = Space(noise=0, echo_gain=0.3, echo_delay=0.02, across_gain=0.6, across_delay=0.01)
        a, b = space.endpoint("a"), space.endpoint("b")
        a.play(tone(0.5), RATE)
        started = a.speaker.clips[-1].started
        own = self._hear(a, started, 0.6)
        other = self._hear(b, started, 0.6)
        assert np.abs(own).max() == pytest.approx(0.15, abs=0.01)
        assert np.abs(other).max() == pytest.approx(0.3, abs=0.01)
        # The echo arrives after its delay
        assert np.abs(own[: int(0.015 * RATE)]).max() == 0
        assert np.abs(own[int(0.025 * RATE) : int(0.05 * RATE)]).max() > 0

    def test_a_microphone_is_never_silent(self):
        space = Space()
        block = self._hear(space.endpoint("a"), time.monotonic(), 0.1)
        assert 0 < np.abs(block).max() < 0.01

    def test_utterances_follow_one_another(self):
        space = Space()
        first = space.say("one", tone(1.0))
        second = space.say("two", tone(1.0))
        assert second.start == pytest.approx(first.start + 1.0, abs=0.001)

    def test_the_microphone_delivers_blocks_in_real_time(self):
        space = Space()
        endpoint = space.endpoint("a")
        blocks = []
        done = threading.Event()

        def callback(indata, frames, info, status):
            blocks.append((time.monotonic(), indata.shape, info.currentTime - info.inputBufferAdcTime))
            if len(blocks) == 10:
                done.set()

        started = time.monotonic()
        with endpoint.input_stream(samplerate=RATE, blocksize=512, callback=callback):
            assert done.wait(2)
        assert all(shape == (512, 1) for _, shape, _ in blocks)
        # Ten blocks of 32 ms take about 0.32 s, and arrive just after they were heard
        assert blocks[-1][0] - started == pytest.approx(0.32, abs=0.08)
        assert all(0 < lag < 1.0 for _, _, lag in blocks)
        assert len(endpoint.recording(started, time.monotonic())) > 0

    def test_it_is_an_audio_device(self):
        endpoint = Space().endpoint("a")
        assert endpoint.query_devices()[0]["max_input_channels"] == 1
        assert endpoint.query_devices(kind="input")["name"].startswith("Virtual")
        endpoint.play(tone(0.2), RATE)
        assert endpoint.playing()
        endpoint.stop()
        assert not endpoint.playing()


def test_merge():
    assert merge([(0, 1), (1.1, 2), (5, 6)], gap=0.2) == [(0, 2), (5, 6)]


class TestStubModel:
    @pytest.fixture
    def stub(self):
        model = StubModel(first_token=0.0, per_word=0.0)
        yield model
        model.close()

    def test_it_is_listed_and_shown_as_ollama_does(self, stub):
        client = ollama.Client(host=stub.url)
        assert [m.model for m in client.list().models] == [MODEL]
        shown = client.show(MODEL)
        assert "completion" in shown.capabilities
        assert shown.modelinfo["voicetest.context_length"] == 32768

    def test_a_question_gets_the_short_reply_streamed(self, stub):
        client = ollama.Client(host=stub.url)
        stream = client.chat(model=MODEL, messages=[{"role": "user", "content": "Capital of France?"}], stream=True)
        chunks = [c.message.content for c in stream]
        assert "".join(chunks) == SHORT
        assert len(chunks) > 3
        assert stub.requests[-1]["messages"][-1]["content"] == "Capital of France?"

    def test_a_story_gets_the_long_reply(self):
        assert reply_to([{"role": "user", "content": "Tell me a long story"}]) == LONG
        assert reply_to([{"role": "user", "content": "Explain it in detail"}]) == LONG
        assert reply_to([{"role": "system", "content": "story"}, {"role": "user", "content": "Hi"}]) == SHORT

    def test_a_reply_cut_off_does_not_break_the_server(self):
        stub = StubModel(first_token=0.0, per_word=0.05)
        try:
            client = ollama.Client(host=stub.url)
            stream = client.chat(model=MODEL, messages=[{"role": "user", "content": "story"}], stream=True)
            next(iter(stream))
            client._client.close()
            again = ollama.Client(host=stub.url)
            assert again.list().models
        finally:
            stub.close()


def test_heard_share_counts_the_words_said():
    assert heard_share("What is the capital of France?", "what is the capital of france") == 1.0
    assert heard_share("How many people live there?", "And roughly how many people live there?") == pytest.approx(5 / 7)
    assert heard_share("Sorry, stop. What is 2 plus 2?", "Sorry, stop. What is two plus two?") == 1.0
    assert heard_share("", "Hello") == 0.0


class FakeAssistant:
    """What self-talk reads of an assistant: its name, log and endpoint."""

    def __init__(self, name, endpoint):
        self.name = name
        self.endpoint = endpoint
        self.log = Log()


def test_self_talk_is_measured_turn_by_turn():
    space = Space(noise=0)
    a = FakeAssistant("iris", space.endpoint("iris"))
    b = FakeAssistant("hal", space.endpoint("hal"))
    start = time.monotonic()
    # Iris says a reply; HAL hears it and answers 0.3 s after it ended
    a.endpoint.play(tone(0.4), RATE)
    a.log("chat.done", {"text": "Hello HAL, what is your favourite planet?"})
    time.sleep(0.75)
    b.log("chat.user", {"text": "Hello Hal, what is your favorite planet?"})
    b.endpoint.play(tone(0.4), RATE)
    b.log("chat.done", {"text": "Saturn, for its rings."})
    time.sleep(0.5)
    a.log("chat.user", {"text": "Saturn for its rings"})
    result = SelfTalk(a, b, report=lambda line: None)._analyse(start, time.monotonic())
    first, second = result.exchanges
    assert (first.speaker, first.listener) == ("iris", "hal")
    assert first.heard_share == pytest.approx(6 / 7, abs=0.01)
    assert first.turns_heard == 1
    assert first.gap == pytest.approx(0.35, abs=0.1)
    assert (second.speaker, second.heard_share) == ("hal", 1.0)
    assert result.overlap == 0
    assert result.self_cuts == {"iris": 0, "hal": 0}


def test_a_reply_heard_in_two_turns_counts_the_last():
    """The listener took the speaker's first sentence as a turn, gave it up
    when the speaker went on, and heard the whole reply as its next turn."""
    space = Space(noise=0)
    a = FakeAssistant("iris", space.endpoint("iris"))
    b = FakeAssistant("hal", space.endpoint("hal"))
    start = time.monotonic()
    a.endpoint.play(tone(0.6), RATE)
    a.log("chat.done", {"text": "I was thinking. Where do people find meaning?"})
    time.sleep(0.3)
    b.log("chat.user", {"text": "I was thinking."})
    b.log("chat.continued", {})
    time.sleep(0.5)
    b.log("chat.user", {"text": "I was thinking. Where do people find meaning?"})
    b.endpoint.play(tone(0.3), RATE)
    time.sleep(0.4)
    a.log("chat.user", {"text": "In each other."})
    result = SelfTalk(a, b, report=lambda line: None)._analyse(start, time.monotonic())
    [exchange] = [e for e in result.exchanges if e.speaker == "iris"]
    assert exchange.heard_share == 1.0
    assert exchange.turns_heard == 2
    assert "hal" in result.logs and any(e["event"] == "chat.continued" for e in result.logs["hal"])

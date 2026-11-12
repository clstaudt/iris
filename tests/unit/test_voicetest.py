"""The voice test's own parts: the acoustic space, the scripted model and
web, the sounds of a room, how words are compared, and how turns and
self-talk are held and measured. Nothing here loads a speech model."""

import http.client
import json
import os
import threading
import time
from types import SimpleNamespace

import httpx
import numpy as np
import ollama
import pytest
import torch
from ddgs.exceptions import DDGSException
from kokoro_onnx.config import SUPPORTED_LANGUAGES as KOKORO_ONNX_LANGUAGES

from iris.events import EventBus
from iris.llm import ConversationManager
from iris.memory import MemoryStore
from iris.plugins import web
from iris.status import StatusManager
from iris.tools import Tools
from iris.turns import SmartTurn
from irisdev.voicetest import assistant as under_test
from irisdev.voicetest import scripted, sounds, voices, wording
from irisdev.voicetest.acoustics import (
    HEARD_KEPT,
    LOOKBACK,
    OUTPUT_LATENCY,
    RATE,
    Room,
    Space,
    in_room,
    merge,
    room,
    sound_span,
    subtract,
)
from irisdev.voicetest.assistant import IrisUnderTest, Log, unwatch_verdicts, watch_verdicts
from irisdev.voicetest.conversations import DEFAULT_GROUPS, GROUPS, all_conversations, group
from irisdev.voicetest.fakeweb import Each, FakeWeb, as_results, latest_match, match_query
from irisdev.voicetest.script import EXPECTATIONS, At, Check, Conversation, Restart, Say, Silence, Turn
from irisdev.voicetest.scripted import (
    BUDGETS,
    CHIMES,
    ScriptedConversations,
    TurnResult,
    budgets_from,
    chime_of,
    flag_windows,
    heard_share,
    holds,
    inner_silences,
    longest_busy_silence,
    state_timeline,
    summarize,
)
from irisdev.voicetest.selftalk import SelfTalk
from irisdev.voicetest.stub_llm import (
    LONG,
    MODEL,
    SHORT,
    Call,
    Rule,
    StubModel,
    last_user,
    reply_to,
    results,
    search,
)
from irisdev.voicetest.voices import VOICES, UserVoice, kokoro_language


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

    def test_the_stream_plays_on_to_the_end_of_its_last_block(self):
        speaker = Space().endpoint("a").speaker
        speaker.play(tone(0.1), RATE, blocksize=RATE // 2)
        clip = speaker.clips[-1]
        time.sleep(clip.end - time.monotonic() + 0.1)
        # Heard no more, but still playing the silence after it
        assert not speaker.audible() and speaker.playing()
        time.sleep(0.45)
        assert not speaker.playing()

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

    def test_after_a_cut_only_the_room_rings_on(self):
        space = Space(noise=0, echo_gain=1.0, echo_delay=0.0, room=Room(extra_delay=0.0))
        a = space.endpoint("a")
        a.play(tone(2.0), RATE)
        time.sleep(0.5)
        a.stop()
        stopped = a.speaker.clips[-1].stopped

        def level(audio):
            return float(np.sqrt(np.mean(audio**2)))

        before = self._hear(a, stopped - 0.2, 0.2)
        after = self._hear(a, stopped + 0.1, 0.2)
        assert level(after) < 0.1 * level(before)

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
        assert "completion" in shown.capabilities and "tools" in shown.capabilities
        assert shown.modelinfo["voicetest.context_length"] == 32768

    def test_a_question_gets_the_short_reply_streamed(self, stub):
        client = ollama.Client(host=stub.url)
        stream = client.chat(model=MODEL, messages=[{"role": "user", "content": "Capital of France?"}], stream=True)
        chunks = [c.message.content for c in stream]
        assert "".join(chunks) == SHORT
        assert len(chunks) > 3
        assert stub.requests[-1].body["messages"][-1]["content"] == "Capital of France?"
        assert stub.requests[-1].kind == "reply" and stub.requests[-1].finished is not None

    def test_a_story_gets_the_long_reply(self):
        assert reply_to([{"role": "user", "content": "Tell me a long story"}]) == LONG
        assert reply_to([{"role": "user", "content": "Explain it in detail"}]) == LONG
        assert reply_to([{"role": "system", "content": "story"}, {"role": "user", "content": "Hi"}]) == SHORT

    def test_a_model_without_tool_support_refuses_them_as_ollama_does(self):
        stub = StubModel(first_token=0.0, per_word=0.0, refuse_tools=True, capabilities=["completion"])
        try:
            client = ollama.Client(host=stub.url)
            assert client.show(MODEL).capabilities == ["completion"]
            schema = {"type": "function", "function": {"name": "search"}}
            with pytest.raises(ollama.ResponseError) as refused:
                list(client.chat(model=MODEL, messages=[{"role": "user", "content": "Hi"}],
                                 tools=[schema], stream=True))
            assert refused.value.status_code == 400 and "does not support tools" in refused.value.error
            assert stub.requests[-1].refused
            reply = client.chat(model=MODEL, messages=[{"role": "user", "content": "Hi"}], stream=True)
            assert "".join(c.message.content for c in reply) == SHORT
        finally:
            stub.close()

    def test_a_reply_can_stall_until_the_client_gives_up(self):
        stub = StubModel(rules=[Rule("stall", "One two three four.", stall_after=2)],
                         first_token=0.0, per_word=0.0)
        try:
            client = ollama.Client(host=stub.url, timeout=0.3)
            said = []
            with pytest.raises(httpx.ReadTimeout):
                for chunk in client.chat(model=MODEL, messages=[{"role": "user", "content": "stall"}],
                                         stream=True):
                    said.append(chunk.message.content)
            assert "".join(said) == "One two"
            record = stub.requests[-1]
            assert record.stalled is not None and record.finished is None
            assert under_test.wait_until(lambda: record.closed is not None, 2)
        finally:
            stub.close()

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


class NamedLanguagesOnly:
    """Kokoro as kokoro-onnx is: it takes only some languages by name, and
    any other spoken from phonemes given with it. Notes what it was asked."""

    def __init__(self):
        self.asked = []
        self.tokenizer = SimpleNamespace(phonemize=lambda text, code: f"<{code}:{text}>")

    def create(self, text, voice, speed=1.0, lang="en-us", phonemes=None):
        assert lang in KOKORO_ONNX_LANGUAGES, f"Language must be one of them. Got {lang}"
        self.asked.append((voice, lang, phonemes))
        return np.zeros(2400, dtype=np.float32), 24000


@pytest.mark.parametrize("name", [n for n, v in VOICES.items() if v["engine"] == "kokoro"])
def test_every_kokoro_voice_of_the_user_can_speak(name, monkeypatch, tmp_path):
    """The Spanish user could not say a word: kokoro-onnx refuses Spanish,
    Portuguese and Hindi by name, so "across languages" could not be held."""
    kokoro = NamedLanguagesOnly()
    monkeypatch.setattr(UserVoice, "_kokoro", kokoro)
    voice = UserVoice.named(tmp_path, name)
    assert len(voice.utterance(["Hello there."])) > 0
    [(spoken, lang, phonemes)] = kokoro.asked
    code = kokoro_language(voice.voice, voice.language)
    assert spoken == voice.voice
    assert (lang, phonemes) == ((code, None) if code in KOKORO_ONNX_LANGUAGES else
                                ("en-us", f"<{code}:Hello there.>"))


class PocketWithoutCloning:
    """Pocket TTS as it loads without the gated weights for voice cloning:
    it speaks only with its own voices."""

    def __init__(self, cloning: bool):
        self.has_voice_cloning = cloning
        self.sample_rate = 24000
        self.prompts = []

    def get_state_for_audio_prompt(self, prompt):
        if not self.has_voice_cloning and "/" in str(prompt):
            raise ValueError("We could not download the weights for the model with voice cloning")
        self.prompts.append(prompt)
        return prompt

    def generate_audio(self, state, text):
        return torch.zeros(2400)


@pytest.mark.parametrize("name, own", [("de", "juergen"), ("nl", "daan")])
def test_without_voice_cloning_the_user_speaks_with_pocket_tts_own_voice(name, own, monkeypatch, tmp_path):
    """The German and Dutch users are cloned voices: where cloning cannot be
    had, no German or Dutch turn could be held at all."""
    model = PocketWithoutCloning(cloning=False)
    monkeypatch.setattr(voices.TTSModel, "load_model", lambda language: model)
    monkeypatch.setattr(UserVoice, "_pocket", {})
    voice = UserVoice.named(tmp_path, name)
    assert len(voice.utterance(["Hallo."])) > 0
    assert model.prompts == [own]
    # Kept apart from what a cloned voice said
    clone = UserVoice.named(tmp_path, name)
    clone._pocket = {}
    monkeypatch.setattr(voices.TTSModel, "load_model", lambda language: PocketWithoutCloning(cloning=True))
    monkeypatch.setattr(UserVoice, "_kokoro", NamedLanguagesOnly())
    assert clone._key("Hallo.") != voice._key("Hallo.")


def test_the_user_speaks_every_language_iris_speaks():
    spoken = {v["language"] for v in VOICES.values()}
    assert {"en", "fr", "it", "es", "ja", "de", "nl", "pt", "zh", "hi"} <= spoken


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


# The scripted model's rules, tool calls and records ---------------------------------------

SEARCH_TOOL = {"type": "function", "function": {"name": "search", "description": "Search the web",
                                                "parameters": {"type": "object", "properties": {}}}}
BROWSE_TOOL = {"type": "function", "function": {"name": "browse", "description": "Read a page",
                                                "parameters": {"type": "object", "properties": {}}}}
LISBON = "- Lisbon weather today: Sunny, 19 degrees. (https://weather.example/lisbon)"


def tool_round(question: str, content: str, tool: str = "search") -> list[dict]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": tool, "arguments": {}}}]},
        {"role": "tool", "content": content},
    ]


class TestStubRules:
    @pytest.fixture
    def stub(self):
        model = StubModel(first_token=0.0, per_word=0.0)
        yield model
        model.close()

    def test_a_tool_call_streams_after_its_lead_in(self, stub):
        stub.set_rules([Rule("weather", lead_in="Let me look that up.\n\n", tool=search())])
        client = ollama.Client(host=stub.url)
        stream = client.chat(model=MODEL, messages=[{"role": "user", "content": "The weather in Lisbon?"}],
                             tools=[SEARCH_TOOL, BROWSE_TOOL], stream=True)
        chunks = list(stream)
        text = "".join(c.message.content or "" for c in chunks)
        calls = [call for c in chunks for call in (c.message.tool_calls or [])]
        assert text == "Let me look that up.\n\n"
        assert [(call.function.name, call.function.arguments) for call in calls] == [
            ("search", {"query": "The weather in Lisbon?"})]
        record = stub.requests[-1]
        assert record.kind == "reply" and record.tools == ["search", "browse"]
        assert record.tool_call_sent is not None and record.first_byte <= record.tool_call_sent

    def test_a_tool_message_is_answered_from_its_result(self, stub):
        stub.set_rules([Rule("weather", tool=search())])
        answer = stub.answer(tool_round("The weather in Lisbon?", LISBON), ["search"])
        assert answer.text == "Here is what I found. Sunny, 19 degrees."
        templated = Rule("weather", tool=search(), after_tool="{title}: {result}")
        stub.set_rules([templated])
        assert stub.answer(tool_round("The weather?", LISBON), ["search"]).text == (
            f"Lisbon weather today: {LISBON}")
        client = ollama.Client(host=stub.url)
        reply = client.chat(model=MODEL, messages=tool_round("The weather?", LISBON), stream=True)
        assert "".join(c.message.content for c in reply).startswith("Lisbon weather today")
        assert stub.requests[-1].kind == "tool_round"
        assert stub.requests[-1].tool_messages() == [LISBON]

    def test_without_tools_the_no_tools_text_comes(self, stub):
        stub.set_rules([Rule("search", tool=search(), no_tools="<tool_call>{}</tool_call>")])
        assert stub.answer([{"role": "user", "content": "Search the web"}], []).text == "<tool_call>{}</tool_call>"
        stub.set_rules([Rule("search", tool=search())])
        assert stub.answer([{"role": "user", "content": "Search the web"}], []).text == SHORT

    def test_a_call_can_be_written_as_text_in_small_pieces(self, stub):
        stub.set_rules([Rule("search", tool=search("comet"), lead_in="Wait.", chunk=4,
                             as_text=lambda call: f"{call.lead_in} <{call.name}>{call.arguments['query']}")])
        client = ollama.Client(host=stub.url)
        stream = client.chat(model=MODEL, messages=[{"role": "user", "content": "Search it"}],
                             tools=[{"type": "function", "function": {"name": "search"}}], stream=True)
        chunks = list(stream)
        assert [c.message.content for c in chunks if c.message.content] == ["Wait", ". <s", "earc", "h>co", "met"]
        assert not any(c.message.tool_calls for c in chunks)

    def test_notes_are_left_out_of_what_rules_match(self, stub):
        message = {"role": "user", "content": "(Note: The user is speaking in Chinese (Mandarin). Respond in "
                                              "Chinese (Mandarin).)\n\nyes please"}
        assert last_user([message]) == "yes please"
        stub.set_rules([Rule(r"^yes", "Great.")])
        assert stub.answer([message], []).text == "Great."

    def test_after_a_tool_a_rule_can_call_another(self, stub):
        def after(messages):
            found = results(messages[-1]["content"])
            return Call("browse", {"url": found[0]["href"]}) if found else "Done."

        stub.set_rules([Rule("read", tool=search(), after_tool=after)])
        answer = stub.answer(tool_round("Read the report", LISBON), ["search", "browse"])
        assert answer.call == Call("browse", {"url": "https://weather.example/lisbon"})
        assert stub.answer(tool_round("Read the report", LISBON), []).text == SHORT

    def test_a_looping_rule_searches_again_while_tools_are_offered(self, stub):
        stub.set_rules([Rule("who won", tool=search(), loop=2, after_tool="Harbor City won.")])
        first = stub.answer(tool_round("Who won?", LISBON), ["search"])
        assert first.call.name == "search" and first.call.arguments["query"] == "Who won? (2)"
        twice = tool_round("Who won?", LISBON) + tool_round("Who won?", LISBON)[1:]
        assert stub.answer(twice, ["search"]).text == "Harbor City won."
        assert stub.answer(tool_round("Who won?", LISBON), []).text == "Harbor City won."

    def test_rules_are_tried_in_order_then_the_short_and_long_replies(self, stub):
        stub.set_rules([Rule("dinner", "Should it be vegetarian?"), Rule("dinner", "never")])
        assert stub.answer([{"role": "user", "content": "plan dinner"}], []).text == "Should it be vegetarian?"
        assert stub.answer([{"role": "user", "content": "a long story"}], []).text == LONG
        stub.reset_rules()
        assert stub.answer([{"role": "user", "content": "plan dinner"}], []).text == SHORT

    def test_a_whole_reply_and_a_summary(self, stub):
        client = ollama.Client(host=stub.url)
        summary = client.chat(model=MODEL, messages=[
            {"role": "system", "content": "Summarize this conversation in 2-3 sentences."},
            {"role": "user", "content": "Summarize this conversation:\n\nUSER: hi\nASSISTANT: hello\nUSER: bye"}])
        assert summary.message.content == "SUMMARY of 3 messages."
        assert stub.requests[-1].kind == "summary" and not stub.requests[-1].stream

    def test_requests_are_recorded_and_announced(self):
        heard = []
        stub = StubModel(first_token=0.0, per_word=0.0, on_request=lambda kind, record: heard.append(kind))
        try:
            client = ollama.Client(host=stub.url)
            before = time.monotonic()
            list(client.chat(model=MODEL, messages=[{"role": "user", "content": "Lisbon?"}], stream=True))
            assert heard == ["reply"]
            assert [r.index for r in stub.requests_between(before)] == [0]
            assert stub.carrying("lisbon") and not stub.carrying("porto")
            assert stub.requests[0].users() == ["Lisbon?"]
        finally:
            stub.close()

    def test_one_slot_a_second_request_waits_for_the_first_to_close(self):
        stub = StubModel(first_token=5.0, per_word=0.0, serial=True,
                         rules=[Rule("quick", "Quick.", first_token=0.0)])
        try:
            port = int(stub.url.rsplit(":", 1)[1])
            body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "slow"}]})
            first = http.client.HTTPConnection("127.0.0.1", port)
            first.request("POST", "/api/chat", body, {"Content-Type": "application/json"})
            assert under_test.wait_until(lambda: len(stub.requests) == 1, 2)
            answered = []

            def second():
                client = ollama.Client(host=stub.url)
                answered.append("".join(c.message.content for c in client.chat(
                    model=MODEL, messages=[{"role": "user", "content": "quick"}], stream=True)))

            thread = threading.Thread(target=second, daemon=True)
            thread.start()
            time.sleep(0.4)
            assert not answered and stub.requests[1].first_byte is None
            first.close()
            thread.join(3)
            assert answered == ["Quick."]
            assert stub.requests[0].closed is not None and stub.requests[0].finished is None
            assert stub.requests[1].first_byte >= stub.requests[0].closed
        finally:
            stub.close()


class TestToolLoop:
    """Iris's tool loop (iris.llm) driven by the scripted model, searching
    the scripted web: no speech, a second or two."""

    def test_a_search_is_called_and_answered_from_its_result(self):
        stub = StubModel(first_token=0.0, per_word=0.0, rules=[
            Rule("weather", lead_in="Let me look that up.\n\n", tool=search(lambda m: "weather in Lisbon"))])
        events = []
        bus = EventBus()
        bus.subscribe(lambda kind, data: events.append((time.monotonic(), kind, data)))
        tools = Tools([web.plugin], status_manager=StatusManager(bus))
        try:
            with FakeWeb({"lisbon": "Lisbon today: 19 degrees and sunny."}, delay=0.2) as fake:
                conversation = ConversationManager(model=MODEL, tools=tools, host=stub.url)
                text = "".join(conversation.chat_stream("What's the weather in Lisbon?", on_tool_use=lambda n, a: None))
            assert text == "Let me look that up.\n\nHere is what I found. 19 degrees and sunny."
            assert [c.query for c in fake.searches()] == ["weather in Lisbon"]
            assert [r.kind for r in stub.requests] == ["reply", "tool_round"]
            assert stub.requests[0].tools == ["search", "browse"]
            assert "Lisbon today: 19 degrees" in stub.requests[1].tool_messages()[0]
            on = flag_windows([e for e in events if e[1] == "flags"], 0, float("inf")).get("searching")
            assert on and 0.15 < on[0][1] - on[0][0] < 1.0
            # Only what was said is kept, not the tool's result
            assert [m["role"] for m in conversation.conversation_history] == ["user", "assistant"]
        finally:
            stub.close()


# The scripted web --------------------------------------------------------------------------


class TestFakeWeb:
    def test_searches_and_pages_are_scripted_recorded_and_restored(self):
        ddgs, reader = getattr(web, "DDGS", None), web.trafilatura
        pages = {"https://sport.example/report": "The winning goal came late.", "https://sport.example/gone": None}
        with FakeWeb({"lisbon": "Lisbon today: 19 degrees.", "porto": [{"title": "Porto", "body": "17 degrees",
                                                                         "href": "https://p.example"}]},
                     pages=pages, delay=0.05, browse_delay=0.05) as fake:
            assert web.DDGS is not ddgs and web.trafilatura is not reader
            found = web.search("Lisbon, no, I meant Porto")
            assert found["success"] and "Porto: 17 degrees (https://p.example)" in found["result"]
            assert web.search("Madrid")["result"] == "No results found."
            reading = web.Pages()
            assert reading.browse("sport.example/report")["result"] == "The winning goal came late."
            assert not reading.browse("https://sport.example/gone")["success"]
            calls = fake.calls
            assert [(c.kind, c.query, c.outcome) for c in calls] == [
                ("search", "Lisbon, no, I meant Porto", "results"), ("search", "Madrid", "nothing"),
                ("browse", "https://sport.example/report", "page"), ("browse", "https://sport.example/gone", "failed")]
            # Iris tried the search that found nothing again, as one call
            assert [c.tries for c in calls] == [1, len(web.SEARCH_TRIES), 1, 1]
            assert all(c.end - c.start >= 0.04 for c in calls)
        assert getattr(web, "DDGS", None) is ddgs and web.trafilatura is reader

    def test_failures_delays_and_answers_per_call(self):
        with FakeWeb({"museum": ConnectionError("timed out"), "lisbon": Each(["22 degrees.", "23 degrees."])},
                     delay=[0.05, 0.15, 0.0]) as fake:
            failed = web.search("museum hours")
            assert not failed["success"] and "timed out" in failed["message"]
            assert "22 degrees" in web.search("Lisbon")["result"]
            assert "23 degrees" in web.search("Lisbon")["result"]
            first, second, third = fake.searches()
            assert first.outcome == "error" and first.tries == len(web.SEARCH_TRIES)
            assert first.end - first.start >= 0.04 * len(web.SEARCH_TRIES)
            assert second.end - second.start >= 0.14 and third.end - third.start < 0.1

    def test_a_delay_by_query(self):
        with FakeWeb({}, delay={"slow": 0.2, "quick": 0.0}) as fake:
            web.search("a quick one")
            web.search("a slow one")
            quick, slow = fake.searches()
            assert quick.end - quick.start < 0.1 <= slow.end - slow.start

    def test_the_real_web_is_recorded_in_place(self, monkeypatch):
        class Network:
            def __init__(self, *args, **kwargs):
                pass

            def text(self, query, max_results=5, **kwargs):
                return [{"title": "Real", "body": query, "href": "https://real.example"}]

        monkeypatch.setattr(web, "DDGS", Network)
        with FakeWeb(real=True) as fake:
            assert "Real: weather (https://real.example)" in web.search("weather")["result"]
            assert fake.searches()[0].outcome == "results"
        assert web.DDGS is Network

    @pytest.mark.parametrize("message, outcome", [("No results found.", "nothing"),
                                                  ("ConnectError: Connection refused", "error")])
    def test_the_real_web_finding_nothing_is_told_from_a_failure(self, monkeypatch, message, outcome):
        class Network:
            def __init__(self, *args, **kwargs):
                pass

            def text(self, query, max_results=5, **kwargs):
                raise DDGSException(message)

        monkeypatch.setattr(web, "DDGS", Network)
        with FakeWeb(real=True) as fake:
            web.search("weather")
            (call,) = fake.searches()
            assert call.outcome == outcome and call.tries == len(web.SEARCH_TRIES)

    def test_results_from_text(self):
        assert as_results("- Paris weather: tomorrow 22 degrees (weather.example/paris)") == [
            {"title": "Paris weather", "body": "tomorrow 22 degrees", "href": "weather.example/paris"}]
        assert as_results("Sunny, 22 degrees.") == [{"title": "", "body": "Sunny, 22 degrees.", "href": ""}]
        assert latest_match(["lisbon", "porto"], "Porto first, then Lisbon") == "lisbon"
        assert latest_match(["lisbon"], "Madrid") is None
        assert match_query(["lh ?402"], "Is flight L H four oh two on time?") == "lh ?402"


# Comparing words ---------------------------------------------------------------------------


class TestWording:
    @pytest.mark.parametrize("text, expected", [
        ("übermorgen in the Musée", ["ubermorgen", "in", "the", "musee"]),
        ("twenty six", ["26"]),
        ("two thousand twenty seven", ["2027"]),
        ("three hundred eighty four thousand", ["384000"]),
        ("the 3rd and the third", ["the", "3", "and", "the", "3"]),
        ("L H four oh two", ["lh402"]),
        ("L.H. 402", ["lh402"]),
        ("four, nine, two", ["4", "9", "2"]),
        ("Oh, that is nice", ["oh", "that", "is", "nice"]),
        ("What is two plus two?", ["what", "is", "2", "plus", "2"]),
        ("V A R G A", ["varga"]),
        ("I need 4 eggs", ["i", "need", "4", "eggs"]),
    ])
    def test_words(self, text, expected):
        assert wording.words(text) == expected

    def test_codes_count_alike_however_they_are_written(self):
        assert heard_share("My locker code is 4-9-2.", "My locker code is four, nine, two.") == 1.0
        assert heard_share("Is flight LH402 on time?", "Is flight L H four oh two on time?") == 1.0
        assert wording.digits("Call me on oh one seven six, two four five") == "0176245"

    def test_fillers_are_not_expected(self):
        assert heard_share("So, what's the tallest mountain in Europe?",
                           "So, um, what's the, uh, tallest mountain in Europe?") == 1.0

    def test_exact_precision_order(self):
        assert wording.same_words("Yes.", "yes") and wording.same_words("4.", "Four.")
        assert not wording.same_words("Yes, yes.", "Yes.") and not wording.same_words("Thank you.", "Yes.")
        precision = wording.heard_precision("what is the weather breaking news", "what is the weather")
        assert precision == pytest.approx(4 / 6)
        assert wording.in_order("4 9 2", ["4", "9", "2"]) and not wording.in_order("2 9 4", ["4", "9", "2"])
        assert wording.contains("Is flight LH402 on time", "lh402")
        assert wording.term_found("the 26th of December", "26|twenty.?six")

    def test_japanese_by_characters(self):
        said = "明日の東京の天気を調べてください。"
        assert heard_share(said, said, "ja") == 1.0
        assert heard_share("明日の東京の天気", said, "ja") == 0.5
        assert wording.character_error_rate("", said) == 1.0


# The space's sources, rooms and microphones --------------------------------------------------


class TestSpaceMore:
    def _hear(self, endpoint, start, seconds):
        return endpoint.space._block(endpoint, start, int(seconds * RATE))

    def test_a_source_is_heard_but_is_no_voice(self):
        space = Space(noise=0)
        a = space.endpoint("a")
        at = time.monotonic()
        source = space.add_source(tone(0.2), at=at, gain=0.5, loop=True)
        assert np.abs(self._hear(a, at + 0.5, 0.1)).max() == pytest.approx(0.25, abs=0.01)
        assert not space.voice_speaking(at + 0.1)
        source.removed = at + 0.6
        assert np.abs(self._hear(a, at + 0.6, 0.1)).max() == 0
        space.remove_sources()
        assert not space.sources
        once = space.add_source(tone(0.2), at=at, name="door")
        assert np.abs(self._hear(a, at + 0.1, 0.05)).max() > 0
        assert np.abs(self._hear(a, at + 0.3, 0.1)).max() == 0
        assert once.end == pytest.approx(at + 0.2)

    def test_another_voice_speaks_over_one_not_after_it(self):
        space = Space()
        first = space.say("one", tone(1.0))
        over = space.say("two", tone(0.5), queued=False)
        assert over.start < first.start + 0.1

    def test_a_room_dies_away(self):
        response = room(0.5, drr=0.0)
        assert response[0] == 1.0
        tail = response[1:]
        assert np.sum(tail**2) == pytest.approx(1.0, rel=0.01)
        first, last = tail[: RATE // 10], tail[int(0.45 * RATE) : int(0.55 * RATE)]
        assert 20 * np.log10(np.sqrt(np.mean(first**2)) / np.sqrt(np.mean(last**2))) > 40
        assert len(in_room(tone(0.2), response)) == len(tone(0.2)) + len(response) - 1

    def test_the_echo_rings_on_past_a_cut_and_distorts_when_driven(self):
        space = Space(noise=0, echo_gain=1.0, echo_room=0.5)
        a = space.endpoint("a")
        a.play(tone(2.0), RATE)
        clip = a.speaker.clips[-1]
        clip.stopped = clip.started + 0.5
        after = self._hear(a, clip.stopped + 0.1, 0.1)
        assert 0 < np.abs(after).max() < 0.5
        space.set_echo(1.0, None, 5.0)
        assert np.abs(space.echo_of(clip)).max() == pytest.approx(np.tanh(2.5) / np.tanh(5.0), abs=0.001)
        space.set_echo(1.0, None, None)
        assert space.echo_of(clip) is not None and np.abs(self._hear(a, clip.stopped + 0.1, 0.1)).max() == 0

    def test_what_nobody_listened_to(self):
        space = Space()
        endpoint = space.endpoint("a")
        said = space.say("hello", tone(1.0))
        with endpoint.input_stream(callback=lambda *args: None):
            time.sleep(0.3)
        assert endpoint.open_spans(said.start)[0][0] <= said.start + 0.05
        deaf = endpoint.deaf_spans(said.start, said.start + 1.0)
        assert sum(b - a for a, b in deaf) == pytest.approx(0.7, abs=0.1)
        assert subtract([(0, 10)], [(2, 3), (5, 6)]) == [(0, 2), (3, 5), (6, 10)]

    def test_long_runs_keep_little(self):
        endpoint = Space().endpoint("a")
        now = time.monotonic()
        endpoint._heard_block(now - HEARD_KEPT - 10, np.zeros(10, np.float32))
        endpoint._heard_block(now, np.zeros(10, np.float32))
        assert len(endpoint._heard) == 1
        speaker = endpoint.speaker
        speaker.play(tone(0.2), RATE)
        clip = speaker.clips[-1]
        assert clip.sounds() is clip.sounds()
        clip.stopped = clip.started + 0.1
        assert clip.sounds()[-1][1] <= clip.started + 0.12
        assert speaker.last_sound(end=clip.started + LOOKBACK + 5) is None
        log = Log()
        log("state", {"state": "a"})
        log("chat.user", {"text": "x"})
        log("state", {"state": "b"})
        assert log.last("state") == {"state": "b"} and log.last_before(time.monotonic(), "chat.user")


# Sounds of a room ----------------------------------------------------------------------------


class TestSounds:
    def test_the_sounds_without_voices(self):
        door = sounds.door()
        assert 0.3 < len(door) / RATE < 0.5 and 0.6 < np.abs(door).max() <= 0.71
        assert len(sounds.keyboard(2.0)) == 2 * RATE
        assert len(sounds.mug()) == int(0.3 * RATE) and len(sounds.cough()) < RATE
        fan = sounds.fan(5.0)
        assert len(fan) == 5 * RATE
        spectrum = np.abs(np.fft.rfft(fan))
        assert np.argmax(spectrum[50 * 5 : 150 * 5]) + 50 * 5 == pytest.approx(100 * 5, abs=5)
        assert len(sounds.music(None, 4.0)) == 4 * RATE
        assert sounds.sound("keyboard", 1.0).shape == (RATE,)
        with pytest.raises(ValueError):
            sounds.sound("babble")
        with pytest.raises(ValueError):
            sounds.sound("thunder")

    def test_a_noise_at_a_signal_to_noise_ratio(self):
        speech = np.concatenate([tone(1.0, 0.5), np.zeros(RATE, np.float32)])
        noise = sounds.fan(2.0)
        gain = sounds.gain_for_snr(noise, speech, 10)
        speech_rms = sounds.active_rms(speech)
        noise_rms = np.sqrt(np.mean((noise * gain) ** 2))
        assert 20 * np.log10(speech_rms / noise_rms) == pytest.approx(10, abs=0.1)

    def test_what_the_voice_detector_makes_of_a_sound(self):
        profile = sounds.vad_profile(np.zeros(RATE, np.float32))
        assert profile["max"] < 0.5 and profile["longest"] == 0


# The script ---------------------------------------------------------------------------------


class TestScript:
    def test_moments_and_shorthands(self):
        assert Turn(["hi"]).trigger == At.finished()
        assert Turn(["hi"], interrupt_after=3.0).trigger == At.reply_audible(3.0)
        assert Turn(["hi"], answer_after=0.3).trigger == At.answered(0.3)
        assert not At.answered().before_turn
        assert Turn(["hi"], stop_after=2.0).all_actions() == [(At.reply_audible(2.0), "stop")]
        assert Turn(["hi"], no_reply=True).expectation == "silent"
        assert At.finished().before_turn and At.start(1).before_turn and not At.reply_end().before_turn
        assert At.event("tool.use", 0.5).describe() == "event(tool.use, 0.5)"
        with pytest.raises(ValueError):
            Turn(["hi"], expect="maybe")

    def test_known_failures(self):
        conversation = Conversation("c", [], expected_failures=["deaf", "T2:heard", "T3:*"])
        assert conversation.expected(1, "deaf") and conversation.expected(2, "heard")
        assert conversation.expected(3, "anything") and not conversation.expected(1, "heard")

    def test_every_scenario_is_well_formed(self):
        conversations = all_conversations()
        assert len({c.name for c in conversations}) == len(conversations)
        assert set(DEFAULT_GROUPS) <= set(GROUPS) and group("basic")[0].name == "a question and a follow-up"
        for conversation in conversations:
            turns = [t for t in conversation.turns if isinstance(t, (Turn, Silence, Restart))]
            assert turns == conversation.turns, conversation.name
            for topic in conversation.memory_topics:
                assert conversation.memory and {"title", "gist", "summary"} <= set(topic), conversation.name
            for entry in conversation.expected_failures:
                number, colon, name = entry.partition(":")
                if colon:
                    assert 1 <= int(number[1:]) <= len(turns) and name, (conversation.name, entry)
            for turn in turns:
                if isinstance(turn, Turn):
                    assert turn.expectation in EXPECTATIONS
                    assert all(isinstance(check, Check) and check.name for check in turn.checks)
                    for at, item in turn.during:
                        assert isinstance(at, At)
            assert set(conversation.needs) <= {"stub", "real_model", "real_web"}
        with pytest.raises(ValueError):
            group("nothing")


# Measuring turns -------------------------------------------------------------------------------


class TestMeasures:
    def test_chimes_are_known(self):
        speaker = Space().endpoint("a").speaker
        speaker.play(CHIMES["rising"], RATE)
        speaker.play(tone(0.18), RATE)
        rising, other = speaker.clips
        assert chime_of(rising) == "rising" and chime_of(other) is None

    def test_pauses_inside_audio(self):
        audio = np.concatenate([tone(0.3), np.zeros(int(0.2 * RATE), np.float32), tone(0.3),
                                np.zeros(int(0.1 * RATE), np.float32), tone(0.3), np.zeros(RATE, np.float32)])
        [(start, end)] = inner_silences(audio, 0.15)
        assert start == pytest.approx(0.3, abs=0.03) and end == pytest.approx(0.5, abs=0.03)

    def test_a_hold_is_found_with_what_it_repeated(self):
        speaker = Space().endpoint("a").speaker
        audio = (np.random.default_rng(1).standard_normal(3 * RATE) * 0.2).astype(np.float32)
        speaker.play(audio, RATE)
        first = speaker.clips[-1]
        first.stopped = first.started + 1.0
        speaker.clips.append(type(first)(audio[int(0.6 * RATE):].copy(), first.stopped + 0.5))
        [hold] = holds(speaker.clips)
        assert hold.silence == pytest.approx(0.5) and hold.rewind == pytest.approx(0.4, abs=0.01)

    def test_flags_states_and_busy_silences(self):
        events = [(1.0, "flags", {"flags": {"searching": True}}), (4.0, "flags", {"flags": {"searching": False}}),
                  (5.0, "flags", {"flags": {"browsing": True}})]
        assert flag_windows(events, 0, 10) == {"searching": [(1.0, 4.0)], "browsing": [(5.0, 10)]}
        states = [(0.5, "state", {"state": "listening"}), (1.0, "state", {"state": "searching"}),
                  (6.0, "state", {"state": "speaking"})]
        timeline = state_timeline(states, 0.8, 10)
        assert timeline == [(0.8, "listening"), (1.0, "searching"), (6.0, "speaking")]
        assert longest_busy_silence(timeline, [(2.0, 3.0)], 0.8, 10) == pytest.approx(3.0)

    def test_budgets_and_the_summary(self):
        assert {"turn_end", "search_added", "new_language"} <= set(BUDGETS)
        assert budgets_from("turn_end=0.8,search_added=4") == {"turn_end": 0.8, "search_added": 4.0}
        turns = [TurnResult("c", n, "x", first_word=f, stages={"turn_end": 200, "recognition": 300, "reply": 400,
                                                               "voice": 200}, tags=[tag])
                 for n, (f, tag) in enumerate([(1.0, "a"), (2.0, "a"), (3.0, "b")], 1)]
        summary = summarize(turns + [TurnResult("c", 0, "", skipped="needs the real web")])
        assert summary["turns"] == 3 and summary["first_word"]["median"] == 2.0 and summary["first_word"]["p90"] == 3.0
        assert summary["unaccounted"]["median"] == pytest.approx(0.9)
        assert summary["by_tag"]["a"]["first_word"]["worst"] == 2.0 and summary["skipped"]


# The Smart Turn verdicts and settings of IrisUnderTest, without Iris ----------------------------


def test_smart_turn_verdicts_are_logged_to_whoever_judges(monkeypatch):
    monkeypatch.setattr(under_test, "_ORIGINAL_JUDGE", lambda self, audio: 0.7)
    judge = object.__new__(SmartTurn)
    mine = SimpleNamespace(log=Log(), judges_with=lambda smart_turn: smart_turn is judge)
    other = SimpleNamespace(log=Log(), judges_with=lambda smart_turn: False)
    watch_verdicts(mine)
    watch_verdicts(other)
    try:
        assert judge(np.zeros(RATE, np.float32)) == 0.7
        assert mine.log.last("turn.verdict")["probability"] == 0.7 and other.log.last("turn.verdict") is None
    finally:
        unwatch_verdicts(mine)
        unwatch_verdicts(other)
    assert SmartTurn.__call__ is under_test._ORIGINAL_JUDGE or SmartTurn.__call__.__name__ != "_judge_and_log"


def test_settings_wait_for_the_config_that_shows_them():
    iris = object.__new__(IrisUnderTest)
    iris.log = Log()
    calls = []

    class Controller:
        def set_tool(self, plugin, on):
            calls.append(("tool", plugin, on))
            iris.log("config", {"tools": {plugin: {"enabled": on}}})
            return {"ok": True}

        def set_persona(self, name):
            calls.append(("persona", name))
            return {"ok": True}

        def set_mic(self, on):
            return {"ok": False, "error": "Assistant not running"}

    iris.controller = Controller()
    assert iris.configure("tool", ("web", True))
    assert not iris.configure("persona", "hal", wait=0.2)
    assert not iris.configure("mic", False) and "Could not set mic" in iris.log.last("error")["message"]
    assert calls == [("tool", "web", True), ("persona", "hal")]
    with pytest.raises(ValueError):
        iris.configure("volume", 3)


def test_memory_starts_iris_again_with_a_memory_of_its_own(monkeypatch, tmp_path):
    """The app runs Iris with memory; the voice test, without it unless a
    conversation asks for it, and then in a new memory each time."""
    monkeypatch.setenv("IRIS_MEMORY_DB", str(tmp_path / "memory.db"))
    iris = object.__new__(IrisUnderTest)
    iris.log, iris.home, iris.options, iris._memories = Log(), tmp_path, {"memory": False}, 0
    iris.controller = None
    started = []

    def restart():
        started.append((iris.options["memory"], os.environ["IRIS_MEMORY_DB"]))
        iris.log("config", {"memory": iris.options["memory"]})

    iris.restart = restart
    assert iris.configure("memory", True) and iris.configure("memory", False) and iris.configure("memory", True)
    assert started == [(True, str(tmp_path / "memory-1.db")), (False, str(tmp_path / "memory-1.db")),
                       (True, str(tmp_path / "memory-2.db"))]

    store = MemoryStore(":memory:")
    iris.controller = SimpleNamespace(assistant=SimpleNamespace(memory=SimpleNamespace(store=store)))
    iris.seed_memory([{"title": "Nazaré", "gist": "a lighthouse", "summary": "A red lighthouse."}])
    assert [t.title for t in store.topics()] == ["Nazaré"] and store.recent_turns() == []


# Holding conversations, with a stand-in assistant ---------------------------------------------


class FakeVoice:
    """The user, as tones: 0.15 s a word, pauses as given."""

    cache = None

    def utterance(self, parts, level=None, room_response=None, splice=False, unsaid=None):
        pieces = []
        for part in parts:
            if isinstance(part, (int, float)):
                pieces.append(np.zeros(int(part * RATE), np.float32))
            else:
                pieces.append(tone(0.15 * len(part.split()), 0.5))
        return np.concatenate(pieces)


class StandIn:
    """An assistant in a space, standing in for Iris: it takes each utterance
    for a turn once it has been said, answers it with a tone, falls silent
    when someone speaks over it, and stops on Stop."""

    name = "fake"

    def __init__(self, space, think=0.3, reply=0.6):
        self.space = space
        self.endpoint = space.endpoint("fake")
        self.log = Log()
        self.think, self.reply = think, reply
        # An error to report with each reply
        self.error = None
        # What a recognizer hears the voice say
        self.heard_back = ""
        self.persona = "iris"
        self.settings = []
        self._history = []
        self._seen = 0
        self._stop = threading.Event()
        self.log("state", {"state": "listening"})
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while not self._stop.is_set():
            time.sleep(0.02)
            if self.endpoint.speaker.playing() and self.space.voice_speaking():
                time.sleep(0.2)
                self.endpoint.stop()
            utterances = list(self.space.utterances)
            while self._seen < len(utterances) and time.monotonic() >= utterances[self._seen].speech_end + 0.2:
                text = utterances[self._seen].text
                self._seen += 1
                threading.Thread(target=self._answer, args=(text, "voice"), daemon=True).start()

    def _answer(self, text, source):
        self.log("chat.user", {"text": text, "source": source, "language": "en"})
        self.log("state", {"state": "thinking"})
        time.sleep(self.think)
        self.log("turn.timing", {"stages": {"turn_end": 200, "recognition": 100, "reply": 100, "voice": 50},
                                 "components": {"turn_end": "smart_turn"}})
        self.endpoint.play(tone(self.reply), RATE)
        self.log("state", {"state": "speaking"})
        # As Iris does, the reply is complete before it has been spoken
        reply = f"Reply to {text}"
        self.log("chat.done", {"text": reply, "interrupted": False})
        if self.error:
            self.log("error", {"message": self.error})
        while self.endpoint.speaker.playing():
            time.sleep(0.02)
        self._history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        self.log("state", {"state": "listening"})

    def close(self):
        self._stop.set()

    def interrupt(self):
        self.endpoint.stop()

    def pause(self):
        self.log("state", {"state": "paused"})

    def resume(self):
        self.log("state", {"state": "listening"})

    def send_text(self, text):
        threading.Thread(target=self._answer, args=(text, "text"), daemon=True).start()

    def set_wake_word(self, word):
        pass

    def clear_conversation(self):
        self._history = []

    def configure(self, name, value):
        self.settings.append((name, value))
        self.log("config", {name: value})
        return True

    def history(self):
        return list(self._history)

    def seed_history(self, messages):
        self._history += messages

    def voice_queued(self):
        return False

    def restart(self):
        """Quit and start again; the conversation is shown again, as Iris
        shows it from its memory."""
        self.log("state", {"state": "idle"})
        for message in self._history:
            if message["role"] == "user":
                self.log("chat.user", {"text": message["content"], "source": "voice"})
            else:
                self.log("chat.done", {"text": message["content"], "interrupted": False})
        self.log("state", {"state": "listening"})

    def transcribe(self, path):
        """What a recognizer hears its voice say: heard_back, when set."""
        return self.heard_back, "en"


@pytest.fixture
def held(monkeypatch):
    """A runner and the stand-in it talks to, quicker to finish than Iris."""
    real_finished = under_test.finished
    monkeypatch.setattr(scripted, "finished", lambda assistant, quiet=0.3: real_finished(assistant, 0.3))
    monkeypatch.setattr(scripted, "NO_REPLY_WATCH", 1.0)
    monkeypatch.setattr(scripted, "ASIDE_WATCH", 1.0)
    space = Space(noise=0)
    fake = StandIn(space)
    lines = []
    runner = ScriptedConversations(fake, space, FakeVoice(), check_replies=False, reply_timeout=10,
                                   report=lines.append)
    yield runner, fake, space
    fake.close()


class TestHolding:
    def test_two_turns_are_answered_and_measured(self, held):
        runner, fake, space = held
        results = runner.hold(Conversation("two", [Turn(["hello there"], at=At.finished(0.2)),
                                                   Turn(["and again"], at=At.finished(0.2))]))
        assert [r.passed for r in results] == [True, True], [r.failures for r in results]
        first = results[0]
        assert first.heard == "hello there" and first.heard_share == 1.0 and not first.taken_early
        assert 0.5 < first.first_word < 0.9 and first.voice_pipeline == pytest.approx(first.first_word - 0.1)
        assert first.states[0][1] == "listening" and ("thinking" in [s for _, s in first.states])
        assert first.audible == pytest.approx(0.6, abs=0.1) and first.replies[0][1] == "Reply to hello there"

    def test_speaking_over_the_reply_measures_the_barge_in(self, held):
        runner, fake, space = held
        fake.reply = 2.0
        results = runner.hold(Conversation("over", [Turn(["tell me"], at=At.finished(0.2)),
                                                    Turn(["sorry stop"], at=At.reply_audible(0.5))]))
        second = results[1]
        assert second.passed, second.failures
        assert second.barge_in is not None and second.barge_in < 1.0
        assert fake.endpoint.speaker.clips[0].stopped is not None

    def test_a_typed_turn_and_stop(self, held):
        runner, fake, space = held
        fake.reply = 2.0
        [result] = runner.hold(Conversation("typed", [Turn(["hi"], typed=True, at=At.finished(0.2),
                                                           actions=[(At.reply_audible(0.3), "stop")])]))
        assert result.passed, result.failures
        assert result.users[0][1] == "text" and result.stop is not None and result.stop < 0.5
        assert fake.log.last("action.stop") == {"args": []}

    def test_the_reply_end_is_met_to_the_sample(self, held):
        runner, fake, space = held
        results = runner.hold(Conversation("at once", [Turn(["one"], at=At.finished(0.2)),
                                                       Turn(["two"], at=At.reply_end(0.1))]))
        assert results[1].passed, results[1].failures
        clip = fake.endpoint.speaker.clips[0]
        end = clip.started + sound_span(clip.audio)[1]
        assert space.utterances[1].speech_start - end == pytest.approx(0.1, abs=0.01)

    def test_known_failures_silences_and_checks(self, held):
        runner, fake, space = held
        results = runner.hold(Conversation("known", [
            Turn(["hello"], at=At.finished(0.2), must_hear=["banana"]),
            Silence(0.5),
            Turn(["again"], at=At.finished(0.2), checks=[Check("custom", lambda r, v: None),
                                                         Check("strict", lambda r, v: f"saw {len(v.requests())}")]),
        ], expected_failures=["T1:must_hear", "T3:custom"]))
        first, silence, third = results
        assert first.passed and first.expected and first.expected[0].startswith("must_hear")
        assert silence.passed and silence.said == "(silence, 0.5 s)"
        assert third.now_passes == ["T3:custom"] and third.failures == ["saw 0"]

    def test_settings_are_put_back_and_the_web_is_scripted_meanwhile(self, held):
        runner, fake, space = held
        original = getattr(web, "DDGS", None)
        seen = []
        conversation = Conversation(
            "settings", [Turn(["hi"], at=At.finished(0.2), checks=[
                Check("web", lambda r, v: seen.append(web.DDGS is not original and v.web is not None))])],
            tools={"web": True}, persona="hal", web={"results": {"x": "y"}})
        [result] = runner.hold(conversation)
        assert result.passed, result.failures
        assert seen == [True] and getattr(web, "DDGS", None) is original and runner.web is None
        assert fake.settings == [("tool", ("web", True)), ("persona", "hal"), ("tool", ("web", False)),
                                 ("persona", "iris")]

    def test_an_aside_taken_for_a_turn_fails(self, held):
        runner, fake, space = held
        fake.reply = 1.5
        [result] = runner.hold(Conversation("aside", [Turn(["tell me"], at=At.finished(0.2),
                                                           during=[(At.reply_audible(0.3), Say(["hmm"]))])]))
        assert "aside" in result.failed
        assert result.asides and result.asides[0][1] == "hmm"

    def test_an_error_fails_a_turn_unless_it_is_expected(self, held):
        runner, fake, space = held
        fake.error = "The model server stopped answering."
        results = runner.hold(Conversation("errors", [
            Turn(["one"], at=At.finished(0.2), error_mentions=["stopped answering"]),
            Turn(["two"], at=At.finished(0.2)),
            Turn(["three"], at=At.finished(0.2), error_mentions=["cannot be reached"]),
        ]))
        assert [r.failed for r in results] == [[], ["error"], ["error", "error"]]
        assert results[2].failures == ["no error mentions any of ['cannot be reached']",
                                       "error: The model server stopped answering."]
        fake.error = None
        [quiet] = runner.hold(Conversation("none", [Turn(["four"], at=At.finished(0.2),
                                                         error_mentions=["stopped"])]))
        assert quiet.failed == ["error"]

    def test_a_restart_between_turns_shows_the_conversation_again(self, held):
        runner, fake, space = held
        results = runner.hold(Conversation("restart", [
            Turn(["hello there"], at=At.finished(0.2)),
            Restart(restores=["hello there", "Reply to hello there"]),
            Turn(["and again"], at=At.finished(0.2)),
            Restart(restores=["banana"]),
        ]))
        first, restart, third, missing = results
        assert first.passed and third.passed, (first.failures, third.failures)
        assert restart.passed and restart.said == "(restart)"
        assert restart.users == [(restart.users[0][0], "voice", "hello there")]
        assert [text for _, text, _ in restart.replies] == ["Reply to hello there"]
        # What was shown again is no turn of the user's
        assert third.heard == "and again" and third.users == [(third.users[0][0], "voice", "and again")]
        assert missing.failed == ["restored"]
        assert len(fake.log.since(0, "action.restart")) == 2
        stand_in = SimpleNamespace(**{k: getattr(fake, k) for k in ("log", "endpoint")})
        elsewhere = ScriptedConversations(stand_in, space, FakeVoice(), report=lambda line: None)
        [skipped] = elsewhere.hold(Conversation("no restart", [Restart()]))
        assert skipped.skipped == "the assistant cannot be restarted"

    def test_the_voice_reads_what_it_is_given_and_the_turn_is_judged_on_its_words(self, held):
        runner, fake, space = held
        [result] = runner.hold(Conversation("reading", [
            Turn(["一二三"], reading=["ichi ni san shi go roku"], at=At.finished(0.2))]))
        assert result.passed, result.failures
        said = space.utterances[0]
        assert said.text == "一二三" and said.speech_end - said.speech_start == pytest.approx(0.9, abs=0.05)

    def test_what_the_voice_is_heard_saying_is_compared_with_the_reply(self, held):
        runner, fake, space = held
        fake.heard_back = "Reply to hello"
        [heard] = runner.hold(Conversation("heard", [Turn(["hello"], at=At.finished(0.2), spoken_share=0.9)]))
        assert heard.passed, heard.failures
        assert heard.spoken == "Reply to hello"
        fake.heard_back = "Chinese letter, Chinese letter"
        [garbled] = runner.hold(Conversation("garbled", [Turn(["hello"], at=At.finished(0.2), spoken_share=0.9)]))
        assert garbled.failed == ["spoken_share"]

    def test_a_turn_not_called_for_is_skipped(self, held):
        runner, fake, space = held
        results = runner.hold(Conversation("skip", [Turn(["question"], at=At.finished(0.2)),
                                                    Turn(["yes"], at=At.finished(0.2), when=lambda reply: False)]))
        assert results[0].passed and results[1].skipped
        held_elsewhere = ScriptedConversations(fake, space, FakeVoice(), report=lambda line: None)
        [skipped] = held_elsewhere.hold(Conversation("needs", [Turn(["x"])], needs=("stub",)))
        assert skipped.skipped == "needs the scripted model"
        fake.stt_downloaded = lambda: ["small"]
        [skipped] = held_elsewhere.hold(Conversation("big", [Turn(["x"])], stt_model="large-v3"))
        assert skipped.skipped == "Whisper large-v3 is not on this computer"


def test_the_scripted_models_settings_change_and_come_back():
    stub = StubModel(first_token=0.0)
    try:
        before = stub.configure(serial=True, context_length=2048)
        assert before == {"serial": False, "context_length": 32768} and stub._slot is not None
        assert ollama.Client(host=stub.url).show(MODEL).modelinfo["voicetest.context_length"] == 2048
        stub.configure(**before)
        assert stub._slot is None and stub.context_length == 32768
        with pytest.raises(ValueError):
            stub.configure(temperature=1)
    finally:
        stub.close()


def scripted_search_scenarios():
    """The scenarios whose answers come from the scripted model and web,
    with turns that do not depend on timing."""
    found = []
    for conversation in all_conversations():
        if not conversation.stub_rules or conversation.web is None or "real_web" in conversation.needs:
            continue
        turns = [t for t in conversation.turns if isinstance(t, Turn)]
        if any(t.expectation not in ("answer", "wake") or t.at not in (None, At.finished()) or t.actions
               or t.during for t in turns[1:]):
            turns = turns[:1]
        found.append(pytest.param(conversation, turns, id=conversation.name))
    return found


@pytest.mark.parametrize("conversation, turns", scripted_search_scenarios())
def test_the_scenarios_scripted_answers_come_through_the_tool_loop(conversation, turns):
    """Each turn, typed into Iris's tool loop, uses the tools the scenario
    expects, searches for what it expects, and its reply has the facts."""
    stub = StubModel(first_token=0.0, per_word=0.0, rules=conversation.stub_rules)
    tools = Tools([web.plugin], choices={"web": bool(conversation.tools.get("web"))})
    uses = []
    try:
        with FakeWeb(**{**conversation.web, "delay": 0.0, "browse_delay": 0.0}):
            manager = ConversationManager(model=MODEL, tools=tools, host=stub.url)
            for turn in turns:
                uses.clear()
                reply = "".join(manager.chat_stream(turn.text, on_tool_use=lambda name, args: uses.append(
                    (name, tools.describe(name, args)[1]))))
                if turn.tools_used is not None:
                    assert [name for name, _ in uses] == list(turn.tools_used), (turn.text, uses)
                for mention in turn.search_mentions:
                    assert any(wording.term_found(detail, mention) for _, detail in uses), (turn.text, mention, uses)
                if turn.check_with_stub and turn.reply_mentions:
                    assert any(m.lower() in reply.lower() for m in turn.reply_mentions), (turn.text, reply)
    finally:
        stub.close()

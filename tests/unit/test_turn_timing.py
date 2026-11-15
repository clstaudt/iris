"""Where a turn's time goes: each stage measured, reported once per turn."""

import tempfile
import threading
import time

import numpy as np

from iris.assistant import VoiceAssistant
from iris.timing import TurnTiming
from iris.tts import SpeechPipeline
from iris.turns import FRAME, SAMPLE_RATE, SPEECH_PAUSE, TurnTracker

FRAME_SECONDS = FRAME / SAMPLE_RATE


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class TestTurnTiming:
    def test_a_spoken_turn_passes_all_four_stages(self):
        clock = Clock()
        timing = TurnTiming("voice", turn_end=0.14, recognition=0.24, clock=clock)
        timing.reply_started()
        clock.now += 0.26
        timing.first_piece()
        clock.now += 0.16
        timing.first_audio()
        event = timing.event()
        assert event["stages"] == {"turn_end": 140, "recognition": 240, "reply": 260, "voice": 160}
        assert event["total"] == 800
        assert event["source"] == "voice"

    def test_a_typed_turn_has_no_turn_end_or_recognition(self):
        clock = Clock()
        timing = TurnTiming("text", clock=clock)
        timing.reply_started()
        clock.now += 0.3
        timing.first_piece()
        event = timing.event()
        assert event["stages"]["turn_end"] is None
        assert event["stages"]["recognition"] is None
        assert event["stages"]["voice"] is None
        assert event["total"] == 300

    def test_only_the_first_piece_and_first_audio_count(self):
        clock = Clock()
        timing = TurnTiming("text", clock=clock)
        timing.reply_started()
        clock.now += 0.2
        timing.first_piece()
        clock.now += 0.1
        timing.first_audio()
        clock.now += 5
        timing.first_piece()
        timing.first_audio()
        assert timing.stages["reply"] == 200
        assert timing.stages["voice"] == 100

    def test_audio_before_any_piece_is_not_the_reply(self):
        timing = TurnTiming("text", clock=Clock())
        timing.first_audio()
        assert timing.stages["voice"] is None

    def test_the_components_travel_with_the_measurement(self):
        timing = TurnTiming("voice", components={"reply": "some-model"})
        assert timing.event()["components"] == {"reply": "some-model"}


def _speech(probabilities):
    """A VAD that reports the given probabilities, one per frame."""
    values = iter(probabilities)
    return lambda frame: next(values)


def _feed(tracker, frames):
    frame = np.zeros(FRAME, dtype=np.float32)
    for _ in range(frames):
        if tracker.feed(frame):
            return True
    return False


class TestTurnEndWait:
    def test_silence_that_ends_the_turn_is_the_wait(self):
        speech_frames = 20
        silent_frames = 40
        tracker = TurnTracker(
            _speech([0.9] * speech_frames + [0.0] * silent_frames),
            finished=None,
            max_silence=silent_frames * FRAME_SECONDS,
        )
        assert _feed(tracker, speech_frames + silent_frames)
        assert abs(tracker.end_wait - silent_frames * FRAME_SECONDS) < 1e-6

    def test_with_smart_turn_the_pause_and_its_verdict(self):
        def finished(audio):
            time.sleep(0.05)
            return 0.9

        tracker = TurnTracker(_speech([0.9] * 20 + [0.0] * 100), finished=finished, max_silence=3.0)
        assert _feed(tracker, 120)
        pause = round(SPEECH_PAUSE / FRAME_SECONDS) * FRAME_SECONDS
        assert pause + 0.05 <= tracker.end_wait < pause + 0.5

    def test_a_turn_not_finished_waits_on(self):
        tracker = TurnTracker(
            _speech([0.9] * 20 + [0.0] * 40), finished=lambda audio: 0.1, max_silence=3.0
        )
        assert not _feed(tracker, 60)
        assert tracker.end_wait == 0.0


class Generator:
    def generate(self, text, language="en", **voice):
        return tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name


class Player:
    def __init__(self):
        self.played = threading.Event()

    def play(self, audio_file, on_level=None, cancelled=None, **effect):
        self.played.set()

    def stop(self):
        pass


class TestFirstAudio:
    def _pipeline(self):
        pipeline = SpeechPipeline(Generator(), Player())
        heard = []
        pipeline.on_reply_audio = lambda: heard.append(time.monotonic())
        return pipeline, heard

    def test_once_per_reply(self):
        pipeline, heard = self._pipeline()
        pipeline.begin_reply()
        pipeline.queue_speech("One.")
        pipeline.queue_speech("Two.")
        pipeline.wait_for_speech()
        assert len(heard) == 1
        pipeline.begin_reply()
        pipeline.queue_speech("Three.")
        pipeline.wait_for_speech()
        assert len(heard) == 2

    def test_a_preview_is_not_a_reply(self):
        pipeline, heard = self._pipeline()
        pipeline.begin_reply()
        pipeline.queue_speech("Hi, I'm Iris.", sound={"voice": "alba"})
        pipeline.wait_for_speech()
        assert heard == []


class Bus:
    def __init__(self):
        self.events = []

    def emit(self, event_type, data):
        self.events.append((event_type, data))

    def timings(self):
        return [data for kind, data in self.events if kind == "turn.timing"]


class SpeechNow:
    """A speech pipeline whose voice starts the moment it is given text."""

    def __init__(self, assistant):
        self.assistant = assistant
        self.current_language = "en"

    def begin_reply(self):
        pass

    def queue_speech(self, text, language=None):
        self.assistant._reply_heard()

    def wait_for_speech(self):
        pass

    def said(self):
        return ""


class Conversation:
    model = "some-model"
    reply = "Sunny all day."

    def chat_stream(self, prompt, **kwargs):
        yield "Sunny "
        yield "all day."

    def clean_markdown(self, text):
        return text

    def keep(self, *args):
        pass


class Recorder:
    turn_end = object()
    last_turn_wait = 0.3


class Transcriber:
    model_name = "large-v3-turbo"

    def transcribe(self, path, conversation=None, languages=None):
        return "Will it rain?", "en"


class Voice:
    engine = "pocket"
    voice = "alba"

    def speaks(self, language):
        return True

    def ready(self, language):
        return True

    def languages(self):
        return frozenset({"en", "de", "fr"})


def _assistant(speak=True):
    assistant = object.__new__(VoiceAssistant)
    assistant.event_bus = Bus()
    assistant.speak_replies = speak
    assistant.conversation = Conversation()
    assistant.recorder = Recorder()
    assistant.transcriber = Transcriber()
    assistant.recognition_engine = "whisper"
    assistant.tts_generator = Voice()
    assistant.speech_pipeline = SpeechNow(assistant)
    assistant.interrupt_event = threading.Event()
    assistant.suspend_event = threading.Event()
    assistant.wake_word = None
    assistant.sleep_word = None
    assistant.conversation_language = "en"
    assistant._timing = None
    assistant._announced = False

    class Status:
        status_lock = threading.Lock()

    assistant.status_manager = Status()
    return assistant


class TestReportedByTheAssistant:
    def test_a_spoken_turn_reports_every_stage_once(self):
        assistant = _assistant()
        assistant._answer_spoken("turn.wav")
        [timing] = assistant.event_bus.timings()
        assert timing["source"] == "voice"
        assert timing["stages"]["turn_end"] == 300
        assert all(timing["stages"][stage] is not None for stage in timing["stages"])
        assert timing["components"] == {
            "turn_end": "smart_turn",
            "recognition": "whisper:large-v3-turbo",
            "reply": "some-model",
            "voice": "pocket:alba",
        }

    def test_without_the_voice_once_the_reply_is_written(self):
        assistant = _assistant(speak=False)
        assistant._answer_typed({"text": "Will it rain?"})
        [timing] = assistant.event_bus.timings()
        assert timing["source"] == "text"
        assert timing["stages"]["reply"] is not None
        assert timing["stages"]["voice"] is None

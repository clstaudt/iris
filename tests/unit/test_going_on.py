"""While a turn is answered the microphone listens on. A user who goes on
with a turn Smart Turn took for finished takes it back: the answer begun is
dropped and the whole turn answered. A user who speaks over the voice (with
echo cancellation) interrupts it: what they say is the next turn."""

import threading
import time

import numpy as np
import soundfile as sf

from iris.assistant import VoiceAssistant, echo_of
from iris.events import EventBus


class Flags:
    status_listening = False

    def __init__(self):
        self.status_lock = threading.Lock()


class Pipeline:
    def __init__(self, said=""):
        self.is_speaking_flag = threading.Event()
        self.stops = 0
        self._said = said
        self.resumes = 0
        self.rewound = []

    def stop(self):
        self.stops += 1

    def pause(self):
        pass

    def resume(self, rewind=True):
        self.resumes += 1
        self.rewound.append(rewind)

    @property
    def is_active(self):
        return self.is_speaking_flag.is_set()

    def said(self):
        return self._said

    def begin_reply(self):
        self._said = ""


class Recorder:
    """Listens on: the user speaks after the first turn only."""

    def __init__(self, whole_turn, new_turn, answering):
        self.whole_turn, self.new_turn = whole_turn, new_turn
        self.answering = answering
        self.calls = []

    def listen_on(self, turn, on_speech, until, should_stop=None, on_level=None, **kw):
        self.calls.append(kw.get("speaking") is not None)
        self.options = kw
        if len(self.calls) == 1:
            # The user goes on while the turn is being answered
            self.answering.wait(2)
            return self.whole_turn if on_speech() else self.new_turn
        while not until():
            time.sleep(0.001)
        return None


def _assistant(tmp_path, announce=True, said="", barge_in=False, said_before=""):
    """An assistant answering a turn the user goes on with. said is what the
    voice says of the answer before the user goes on; said_before what it
    said of the previous turn's answer."""
    assistant = object.__new__(VoiceAssistant)
    assistant.interrupt_event = threading.Event()
    for name in ("pause_event", "suspend_event", "hold_event", "stop_event"):
        setattr(assistant, name, threading.Event())
    assistant.status_manager = Flags()
    assistant.speech_pipeline = Pipeline(said_before)
    assistant.event_bus = EventBus()
    assistant._last_voice = 0.0
    assistant.barge_in = barge_in
    assistant.echo = object() if barge_in else None
    paths = {}
    for name in ("first", "whole", "new"):
        paths[name] = tmp_path / f"{name}.wav"
        paths[name].write_bytes(b"")
    answering = threading.Event()
    assistant.recorder = Recorder(str(paths["whole"]), str(paths["new"]), answering)
    answered = []

    def handle_heard(path, dormant):
        assistant._announced = announce
        if not answered:
            # The answer begins, and the voice says some of it
            if said:
                assistant.speech_pipeline._said = said
            answering.set()
            deadline = time.monotonic() + 2
            while not assistant.interrupt_event.is_set() and time.monotonic() < deadline:
                time.sleep(0.001)
        answered.append((path, assistant.interrupt_event.is_set()))

    assistant._handle_heard = handle_heard
    return assistant, {k: str(v) for k, v in paths.items()}, answered


def _events(assistant):
    events = []
    assistant.event_bus.subscribe(lambda kind, data: events.append(kind))
    return events


def test_the_whole_turn_is_answered(tmp_path):
    assistant, paths, answered = _assistant(tmp_path)
    events = _events(assistant)

    assistant._answer_listening(paths["first"], dormant=False)

    # The first answer was cancelled as soon as the user went on, the whole
    # turn answered afresh
    assert answered == [(paths["first"], True), (paths["whole"], False)]
    assert assistant.speech_pipeline.stops == 1
    assert events.count("chat.continued") == 1
    assert not assistant.interrupt_event.is_set()


def test_a_turn_not_shown_yet_needs_no_taking_back(tmp_path):
    assistant, paths, _ = _assistant(tmp_path, announce=False)
    events = _events(assistant)
    assistant._answer_listening(paths["first"], dormant=False)
    assert "chat.continued" not in events


def test_without_echo_cancellation_listening_ends_with_the_voice(tmp_path):
    assistant, paths, _ = _assistant(tmp_path)
    assistant._answer_listening(paths["first"], dormant=False)
    assert assistant.recorder.calls == [False, False]


def test_speaking_over_the_voice_is_the_next_turn(tmp_path):
    assistant, paths, answered = _assistant(tmp_path, said="Lisbon in May is", barge_in=True)
    events = _events(assistant)

    assistant._answer_listening(paths["first"], dormant=False)

    # The voice was stopped; what the user said is a turn of its own, and the
    # answer it interrupted stays in the conversation as far as it was heard
    assert assistant.recorder.calls == [True, True]
    assert answered == [(paths["first"], True), (paths["new"], False)]
    assert assistant.speech_pipeline.stops == 1
    assert "chat.continued" not in events


def test_the_voice_counts_as_speaking_until_its_stream_has_played_out(tmp_path):
    assistant, paths, _ = _assistant(tmp_path, barge_in=True)
    assistant._answer_listening(paths["first"], dormant=False)
    speaking, pipeline = assistant.recorder.options["speaking"], assistant.speech_pipeline
    pipeline.is_speaking_flag.set()
    assert speaking()
    # Paused by the user
    assistant.suspend_event.set()
    assert not speaking()
    assistant.suspend_event.clear()
    pipeline.is_speaking_flag.clear()
    assert not speaking()


def test_a_hold_ended_by_the_users_pause_keeps_the_voice_paused(tmp_path):
    assistant, paths, _ = _assistant(tmp_path, barge_in=True)
    assistant._answer_listening(paths["first"], dormant=False)
    release, pipeline = assistant.recorder.options["release"], assistant.speech_pipeline
    before = pipeline.resumes
    release()
    assert pipeline.resumes == before + 1
    # A hold that heard nothing of the user goes on exactly where it was cut
    release(False)
    assert pipeline.resumes == before + 2 and pipeline.rewound[-2:] == [True, False]
    assistant.suspend_event.set()
    release()
    assert pipeline.resumes == before + 2


def test_barge_in_switched_off(tmp_path):
    assistant, paths, _ = _assistant(tmp_path, barge_in=True)
    assistant.barge_in = False
    assistant._answer_listening(paths["first"], dormant=False)
    assert assistant.recorder.calls == [False, False]



def test_what_was_said_of_the_previous_answer_does_not_count(tmp_path):
    """The user pauses and goes on before anything of this answer was said:
    the whole turn is answered, its beginning included, however much of the
    previous answer the voice had said."""
    assistant, paths, answered = _assistant(tmp_path, said_before="Paris is the capital.")
    assistant._answer_listening(paths["first"], dormant=False)
    assert answered == [(paths["first"], True), (paths["whole"], False)]


class TestOwnVoice:
    """A turn made of the voice's own words is its echo, not the user's."""

    def test_the_voice_heard_back(self):
        spoken = "I'll run the search for Ali G Indahouse quotes."
        assert echo_of("I'll run the", spoken)
        assert echo_of("the search.", spoken)
        # A longer echo comes back garbled
        assert echo_of("I’ll run the search for Ali G Indahouse quote", spoken)
        # Numbers come back as words
        assert echo_of("Number two.", "This is sentence number 2 of a long answer.")

    def test_the_users_words(self):
        spoken = "Would you like coffee or tea?"
        assert not echo_of("Tea", spoken)
        assert not echo_of("Tea please, with milk", spoken)
        assert not echo_of("What is the capital of France?", "Paris is the capital of France.")
        assert not echo_of("I'll run the search", "")


def test_a_turn_of_the_voices_own_words_is_not_answered(tmp_path):
    """Heard just after the voice said them: no turn, nothing answered."""
    assistant = object.__new__(VoiceAssistant)
    assistant.interrupt_event = threading.Event()
    assistant.status_manager = Flags()
    assistant.wake_word = None
    answered = []
    assistant.chat = lambda *args, **kwargs: answered.append(args)
    assistant.event_bus = EventBus()

    class Spoken:
        def spoken_since(self, moment):
            return "I'll run the search for Ali G Indahouse quotes."

    class Transcriber:
        def __init__(self, text):
            self.text = text

        def transcribe(self, path):
            return self.text, "en"

    assistant.speech_pipeline = Spoken()
    turn = tmp_path / "turn.wav"
    sf.write(turn, np.zeros(16000, dtype=np.float32), 16000)
    events = _events(assistant)

    assistant.transcriber = Transcriber("I'll run the")
    assistant._answer_spoken(str(turn))
    assert answered == [] and "chat.user" not in events


def test_an_answer_begun_after_the_voice_fell_quiet_is_the_users(tmp_path):
    """"The blue one", said back to "the red one or the blue one?" 0.4 s
    after the voice fell quiet, is answered; the same words begun as the
    voice fell quiet are its echo."""
    assistant = object.__new__(VoiceAssistant)
    assistant.interrupt_event = threading.Event()
    assistant.status_manager = Flags()
    assistant.wake_word = None
    answered = []
    assistant.chat = lambda *args, **kwargs: answered.append(args)
    assistant.event_bus = EventBus()

    class Spoken:
        """The voice's question, heard until a moment before the turn."""

        ended = 0.0

        def spoken_since(self, moment):
            return "Would you like the red one or the blue one?" if self.ended >= moment else ""

    class Transcriber:
        def transcribe(self, path):
            return "The blue one.", "en"

    class Voice:
        def speaks(self, language):
            return True

        def languages(self):
            return frozenset({"en", "de", "fr"})

    assistant.speech_pipeline = Spoken()
    assistant.transcriber = Transcriber()
    assistant.tts_generator = Voice()
    assistant.conversation_language = "en"
    # Two seconds, from PRE_SPEECH before its first syllable: that came 1.5 s
    # before it is recognized
    turn = tmp_path / "turn.wav"
    sf.write(turn, np.zeros(32000, dtype=np.float32), 16000)

    assistant.speech_pipeline.ended = time.monotonic() - 1.5 - 0.4
    assistant._answer_spoken(str(turn))
    assert len(answered) == 1

    assistant.speech_pipeline.ended = time.monotonic() - 1.5 - 0.1
    assistant._answer_spoken(str(turn))
    assert len(answered) == 1

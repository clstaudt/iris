"""Voice assistant coordinator that orchestrates all components."""

import logging
import math
import os
import queue
import re
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from faster_whisper.audio import decode_audio

from iris import cloning, engines, language
from iris.attachments import Attachment, recorded_text
from iris.audio import AudioPlayer, AudioRecorder, SoundDevice, get_filter
from iris.echo import HAS_ECHO_CANCELLATION, EchoCanceller
from iris.events import EventBus, brief
from iris.llm import (
    OPENING,
    ConversationManager,
    effective_host,
    failure_message,
    model_missing,
    split_speech,
)
from iris.memory import DEFAULT_PATH, MemoryKeeper, MemoryStore, chat_json
from iris.personas import Persona, get_persona
from iris.plugin import Plugin, PluginContext
from iris.plugins import BUILT_IN
from iris.roles import find_role
from iris.status import StatusManager
from iris.timing import TurnTiming
from iris.tools import Tools
from iris.transcription import WhisperTranscriber
from iris.tts import LANGUAGE_NAMES, SpeechGenerator, SpeechPipeline
from iris.turns import PRE_SPEECH, SmartTurn
from iris.wakeword import WakeWordDetector, match_sleep_word

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# With a wake word set: how long the assistant keeps listening after a turn
# (or after the user last spoke) before it goes back to sleep. A window of
# 0 is none: it then sleeps only when told, by the sleep word or the
# Going to sleep tool.
DEFAULT_WAKE_WINDOW = 30.0
# The shortest window, which still leaves time to say a follow-up
SHORTEST_WAKE_WINDOW = 5.0
# While dormant an utterance ends after this much silence, so the wake word
# is recognized promptly instead of after the conversational pause
DORMANT_SILENCE = 0.8
# Seconds the voice is given to die away in the room after Iris's parting
# words, before it falls asleep and listens for the wake word. When the
# voice leaves the speaker is not known exactly (Bluetooth adds its own
# delay), so this is generous rather than measured.
GOODBYE_SETTLE = 1.0


# Seconds the microphone stays closed after a voice preview, for the room
# to fall quiet
PREVIEW_SETTLE = 0.4

# What Iris is asked at the end of the walkthrough: to open the conversation
# with a greeting of its own, in the persona's manner
GREETING = (
    "(The user has just finished setting you up and is with you now. Greet them "
    "the way you would greet someone in person, in your own manner, and let them "
    "know they can simply talk to you.)"
)

# Seconds the loop waits after a round of it failed, before the next
LOOP_RETRY = 0.2

# Seconds forgetting everything waits for the turn underway to end once it
# is stopped; it takes a moment, unless the user is still speaking
FORGET_WAIT = 10.0

# Told the user when an image (attached, or from the camera or the screen)
# has no model to see it
NO_VISION = (
    "None of the installed models can see images. Install one with vision, "
    "such as qwen3-vl, to talk about pictures."
)


def _chime_tones(freqs: tuple[float, ...]) -> np.ndarray:
    """Soft tones in sequence, each swelling in and out."""
    rate = 24000
    tones = []
    for freq in freqs:
        t = np.arange(int(rate * 0.09)) / rate
        envelope = np.sin(np.pi * t / t[-1]) ** 2
        tones.append(0.12 * envelope * np.sin(2 * np.pi * freq * t))
    return np.concatenate(tones).astype(np.float32)


def _wake_chime() -> np.ndarray:
    """Two soft rising tones that say "I'm listening" across the room."""
    return _chime_tones((660.0, 990.0))


def _window(seconds) -> float:
    """An awake window as Iris keeps it: 0 (or none) for none, otherwise
    at least the shortest one."""
    seconds = float(seconds or 0)
    return max(SHORTEST_WAKE_WINDOW, seconds) if seconds > 0 else 0.0


def _sleep_chime() -> np.ndarray:
    """The same tones falling: "I'm going to sleep"."""
    return _chime_tones((990.0, 660.0))


def _thread_chime() -> np.ndarray:
    """One soft tone twice, neither rising nor falling: "another thread"."""
    return _chime_tones((784.0, 784.0))


def _echo_canceller() -> Optional[EchoCanceller]:
    """The echo canceller, or None where it cannot run."""
    if not HAS_ECHO_CANCELLATION:
        return None
    try:
        return EchoCanceller()
    except Exception:
        return None


# A turn that begins while the voice speaks, or this long after its stream
# ended (its last syllables may still arrive, through the room), may be its
# echo
ECHO_AFTER = 0.3
# A turn of four words or more is the voice's echo when this share of its
# words come, in order, from what the voice said
ECHO_SHARE = 0.85


# Numbers as speech recognition writes them, and as a reply may
_NUMBERS = {
    word: str(n)
    for n, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}


def _words(text: str) -> list[str]:
    """The words of a text, lower case, numbers as digits."""
    found = re.findall(r"[a-z0-9']+", (text or "").lower().replace("\u2019", "'"))
    return [_NUMBERS.get(word, word) for word in found]


def echo_of(heard: str, spoken: str) -> bool:
    """Whether a turn heard is the voice's own words coming back: two or
    three words the voice said just so, one after the other, or for a
    longer turn (which comes back garbled), most of its words in the order
    the voice said them. A single word ("yes", "tea") could be the user's."""
    heard_words, spoken_words = _words(heard), _words(spoken)
    if len(heard_words) < 2 or not spoken_words:
        return False
    if len(heard_words) < 4:
        n = len(heard_words)
        return any(spoken_words[k : k + n] == heard_words for k in range(len(spoken_words) - n + 1))
    # How many of the heard words the voice said in the same order
    previous = [0] * (len(spoken_words) + 1)
    for word in heard_words:
        current = [0] * (len(spoken_words) + 1)
        for k, said in enumerate(spoken_words, 1):
            current[k] = previous[k - 1] + 1 if word == said else max(previous[k], current[k - 1])
        previous = current
    return previous[-1] >= ECHO_SHARE * len(heard_words)


def _seconds_of(path: str) -> float:
    """How long a recording is (0 if it cannot be read)."""
    try:
        return sf.info(path).duration
    except Exception:
        return 0.0


class VoiceAssistant:
    """Voice assistant with speech-to-text, LLM, and text-to-speech."""

    # Where the current turn's time goes, reported once the reply is heard
    _timing: Optional[TurnTiming] = None
    # Asleep by the sleep word while the wake word is off: until called by
    # the persona's wake word, then listening to everything again
    _asleep_until_called = False
    # The reply being given ends the conversation: the model called the
    # sleep tool, and Iris goes to sleep once the reply has been heard
    _ending = False
    # Until then the microphone stays closed after parting words
    _quiet_until = 0.0
    # Listening paused, as on opening another thread: shown as paused, the
    # microphone closed until resumed, typed turns still answered
    listening_paused = False
    # Told (engine, voice, language) when a reply comes to a language the
    # persona's voice speaks but is not ready for: the app makes it ready,
    # where the user sees it (iris.readiness)
    on_voice_not_ready: Optional[Callable[[str, str, str], None]] = None
    # Whether the user has spoken or typed in the conversation's language
    # yet: until then Whisper goes by what it hears alone
    _language_known = False
    # Iris's languages, as the user chose them in the app; None for any
    chosen_languages: Optional[frozenset[str]] = None

    def __init__(
        self,
        whisper_model: str = "auto",
        whisper_device: str = "auto",
        whisper_engine: str = "auto",
        recognition_engine: str = "whisper",
        llm_model: str = "qwen3:1.7b",
        sample_rate: int = 16000,
        silence_threshold: float = 0.015,
        silence_duration: float = 3.5,
        persona: Optional[Persona] = None,
        vision_model: Optional[str] = "qwen3-vl:8b",
        plugins: Optional[list[Plugin]] = None,
        tools: Optional[dict[str, bool]] = None,
        event_bus: Optional[EventBus] = None,
        turn_queue: Optional[queue.Queue] = None,
        speak_replies: bool = True,
        ollama_host: Optional[str] = None,
        memory: bool = True,
        wake_word: Optional[str] = None,
        wake_window: float = DEFAULT_WAKE_WINDOW,
        sleep_word: Optional[str] = None,
        tts_models_dir: Optional[str] = None,
        barge_in: bool = True,
        turn_detection: str = "smart_turn",
        audio: Optional[SoundDevice] = None,
    ):
        """
        Initialize the voice assistant.

        Args:
            whisper_model: Whisper model size, or "auto" for the best one the
                hardware decodes quickly (large-v3-turbo on a GPU, small on CPU)
            whisper_device: Device for faster-whisper (cpu, cuda, or auto)
            whisper_engine: "auto" (MLX on Apple Silicon, else faster-whisper),
                "mlx" or "faster-whisper"
            recognition_engine: Id of the speech recognition engine (see
                iris.engines); the whisper_* settings go to it as options
            llm_model: Ollama model name (primary model, should be fast)
            sample_rate: Audio sample rate for recording
            silence_threshold: RMS threshold for silence detection
            silence_duration: Silence that ends a turn Smart Turn has not ended
            persona: Persona to use for the assistant
            vision_model: Vision model to use when vision is needed (default: qwen3-vl:8b)
            plugins: The plugins whose tools the model may call (see
                iris.plugins); None for those shipped with Iris
            tools: {plugin id: on} chosen in the app; a plugin not in it
                follows its own default
            event_bus: Optional EventBus for streaming activity to an external UI
            turn_queue: Optional queue of typed turns from an external UI. They
                are taken in run_loop, between listens, so typed and spoken
                turns share one conversation and never overlap.
            speak_replies: If False, replies are streamed as text only
            ollama_host: Ollama server URL; None follows OLLAMA_HOST
            memory: Keep the conversation in long-term memory (~/.iris/memory.db,
                or IRIS_MEMORY_DB), in threads remembered by topic; it starts
                in a new thread (see ConversationManager.open_thread)
            wake_word: If set, listen only for this word until addressed, and
                fall dormant again after wake_window seconds without a turn
                (the persona's, which the controller passes when it is on)
            wake_window: Seconds the assistant stays awake after a turn; 0
                to fall asleep only when told to
            sleep_word: Said while awake, puts the assistant back to sleep;
                None for none (the persona's, which is optional)
            tts_models_dir: Folder holding the Kokoro voice model; None looks
                in the usual places
            barge_in: Let the user interrupt the voice by speaking (needs
                echo cancellation, which takes the voice out of the microphone)
            turn_detection: "smart_turn" to hear when the user has finished,
                or "silence" to end a turn after silence_duration of silence
            audio: The microphone and speaker; the computer's if None (a
                test gives it a virtual room)
        """
        # Where activity is pushed to the app (None in tests)
        self.event_bus = event_bus

        # Cooperative control signals for external UIs
        self.interrupt_event = threading.Event()  # barge-in: cancel current turn
        # Whether the turn being answered was shown (chat.user) yet
        self._announced = False
        self.stop_event = threading.Event()  # request run_loop shutdown
        self.pause_event = threading.Event()  # mic off (models stay loaded)
        self.hold_event = threading.Event()  # user is typing: keep the mic closed
        # Conversation paused: no listening, no new turns, the reply and its
        # speech hold where they are until resumed
        self.suspend_event = threading.Event()
        # Conversation activity, so background work can give way to it
        self._turn_active = threading.Event()
        self._last_voice = 0.0
        # Held while a turn is taken and answered, and while everything is
        # forgotten, so that happens between turns
        self._turn_lock = threading.Lock()

        # Turns submitted by an external UI: {"text": str, "id": str} for typed
        # messages, {"audio": path} for recorded speech
        self.turn_queue = turn_queue if turn_queue is not None else queue.Queue()

        # Voice output toggle; False turns the assistant into a text chat
        self.speak_replies = speak_replies

        # Wake word mode: off when None; awake_until is when it falls dormant
        self.wake_word: Optional[str] = None
        self.wake_detector: Optional[WakeWordDetector] = None
        self.awake_until = 0.0
        self.wake_window = _window(wake_window)
        self.sleep_word = (sleep_word or "").strip() or None
        self._was_dormant = True  # to notice the moment it falls asleep
        self._last_window_emit = 0.0

        # Track current conversation language for continuity
        self.conversation_language = "en"
        # The language the reply being spoken is in, piece by piece
        self._reply_language = "en"
        # Of the reply: whether a piece of it was queued for the voice, and
        # whether one was only shown, in a language the voice does not speak
        self._reply_voiced = False
        self._reply_unspoken = False
        # (engine, voice, language) the user was told are shown, not spoken
        self._unspoken_told: set[tuple[str, str, str]] = set()

        # The persona speaks with its own engine and voice
        self.persona = persona if persona else get_persona("iris")

        # Initialize components
        # Smart Turn hears when the user has finished speaking; without it
        # (offline on a first run) a turn ends after the silence. The echo
        # canceller, told what the player plays, lets the microphone hear the
        # user over the voice.
        self.barge_in = barge_in
        self.echo = _echo_canceller()
        turn_end = SmartTurn.load() if turn_detection == "smart_turn" else None
        if turn_detection == "smart_turn" and turn_end is None:
            self._error("Smart Turn could not be loaded, so a turn ends after a silence.")
        self.recorder = AudioRecorder(
            sample_rate,
            silence_threshold,
            silence_duration,
            turn_end=turn_end,
            echo=self.echo,
            audio=audio,
        )

        # Initialize audio player with filter from persona
        audio_filter = get_filter(self.persona.audio_filter) if self.persona.audio_filter else None
        self.player = AudioPlayer(audio_filter=audio_filter, audio=audio)
        self.player.echo = self.echo

        # Speech recognition and the voice may fail to load, offline before
        # their models were ever downloaded: Iris then starts without them,
        # says so, and answers typed messages, in text when it has no voice
        self.recognition_engine = recognition_engine
        self.recognition_options = {
            "model": whisper_model,
            "device": whisper_device,
            "engine": whisper_engine,
        }
        self.transcriber = None
        self.load_recognition()

        # Initialize TTS with persona's engine, voice and speed
        self.tts_generator = SpeechGenerator(
            self.persona.engine,
            self.persona.voice,
            self.persona.speed,
            models_dir=tts_models_dir,
            prepare=False,
        )
        # Why the voice could not be loaded; None while it speaks
        self.voice_error: Optional[str] = None
        self.load_voice()
        # A language's voice that cannot be loaded later on (Pocket TTS
        # loads one per language at its first reply) is told too
        self.tts_generator.on_failure = self._error
        self.speech_pipeline = SpeechPipeline(self.tts_generator, self.player)
        self.speech_pipeline.on_reply_audio = self._reply_heard
        # A reply the speaker cannot play is shown, and the user told why
        self.speech_pipeline.on_speaker_error = lambda error: self._error(
            f"The speaker could not be opened: {brief(error)}"
        )
        # Told what language each piece of a reply is in, before the first
        language.prepare()

        if self.event_bus:
            self.speech_pipeline.on_output_level = lambda level: self.event_bus.emit(
                "audio.output", {"rms": level}
            )

        # What the assistant is doing, pushed to the app as it changes
        self.status_manager = StatusManager(self.event_bus)

        # Long-term memory: filed in the background by the conversation's own
        # model, recalled by it through the memory tools
        self.memory: Optional[MemoryKeeper] = None
        if memory:
            store = MemoryStore(os.environ.get("IRIS_MEMORY_DB") or DEFAULT_PATH)
            self.memory = MemoryKeeper(
                store,
                chat_json(lambda: self.conversation.host, lambda: self.conversation.model),
                on_change=lambda: self.event_bus and self.event_bus.emit("memory.updated", {}),
                # Nor while memory is switched off: nothing is written then
                busy=lambda: self._conversation_busy() or not self.conversation.remember,
            )

        # The plugins' tools, on as chosen; a plugin whose optional
        # dependencies are missing stays off and shows as not installed
        self.plugins = list(plugins) if plugins is not None else list(BUILT_IN)
        self.tools = Tools(
            self.plugins,
            self._plugin_context(),
            status_manager=self.status_manager,
            choices=tools,
        )

        # Asked again once recognition loads, should it not have
        self._wanted_wake_word = wake_word
        self._set_wanted_wake_word(engine=whisper_engine)

        # Initialize conversation manager with tools
        # The vision model answers when the camera, the screen or an attached
        # image is involved and the primary model cannot see
        self.conversation = ConversationManager(
            llm_model,
            persona=self.persona,
            tools=self.tools,
            vision_model=vision_model,
            host=ollama_host,
            memory=self.memory,
        )
        # The app lists the threads anew once one was talked in
        self.conversation.on_stored = lambda: self.event_bus and self.event_bus.emit(
            "threads.changed", {}
        )
        if self.memory:
            self.memory.start()

    def load_recognition(self) -> bool:
        """Load speech recognition. Without it Iris cannot listen: the
        microphone stays off, typed messages are still answered, and the
        user is told. Returns whether it loaded."""
        try:
            self.transcriber = engines.recognition_engine(self.recognition_engine).create(
                dict(self.recognition_options)
            )
        except Exception as e:
            logger.exception("Speech recognition could not be loaded")
            self.transcriber = None
            self.pause_event.set()
            self._error(
                "Speech recognition could not be loaded, so Iris cannot listen. "
                f"Typed messages are answered. {brief(e)}"
            )
            return False
        if getattr(self, "conversation", None) is not None:
            self._set_wanted_wake_word()
        return True

    def load_voice(self) -> bool:
        """Load the persona's voice. Without it replies are shown as text,
        and the user is told; switching the voice on tries again. Returns
        whether it loaded."""
        try:
            self.tts_generator.prepare()
        except Exception as e:
            logger.exception("The voice could not be loaded")
            self.voice_error = brief(e)
            self.speak_replies = False
            self._error(f"The voice could not be loaded, so replies are shown as text. {brief(e)}")
            return False
        self.voice_error = None
        return True

    def _set_wanted_wake_word(self, engine: str = "auto"):
        """Listen for the wake word asked for at the start. Without
        recognition there is nothing to listen with yet; a wake word that
        cannot be set up leaves Iris listening to everything."""
        if not self._wanted_wake_word or self.transcriber is None:
            return
        try:
            self.set_wake_word(self._wanted_wake_word, engine=engine)
        except Exception as e:
            logger.exception("The wake word could not be set up")
            self._error(
                f"The wake word could not be set up, so Iris listens without it. {brief(e)}"
            )

    def _plugin_context(self) -> PluginContext:
        """What the plugins get to make their tools: long-term memory, and
        going to sleep, which is offered only while the wake word is on: the
        model ends a conversation only where waiting for the wake word is
        what the user chose."""
        return PluginContext(
            memory=self.memory.store if self.memory else None,
            thread=lambda: self.conversation.thread,
            remember=lambda: self.conversation.remember,
            sleep=self._end_conversation,
            can_sleep=lambda: bool(self.wake_word),
        )

    def config_info(self) -> dict:
        """Describe the loaded models and audio input for external UIs."""
        return {
            "persona": self.persona.name,
            "persona_id": self.persona.id,
            "llm_model": self.conversation.model,
            "recognition_engine": self.recognition_engine,
            "whisper_model": getattr(self.transcriber, "model_name", ""),
            "whisper_device": getattr(self.transcriber, "device", ""),
            "whisper_engine": getattr(self.transcriber, "engine", ""),
            "stt_model_path": str(getattr(self.transcriber, "model_path", None) or ""),
            "tts_models_dir": str(self.tts_generator.models_dir or ""),
            "tts_engine": self.tts_generator.engine,
            "tts_voice": self.tts_generator.voice,
            "input_device": self.recorder.current_device_name(),
            "speak_replies": self.speak_replies,
            "barge_in": self.barge_in,
            "barge_in_available": self.echo is not None,
            "turn_detection": "smart_turn" if self.recorder.turn_end else "silence",
            "silence_duration": self.recorder.silence_duration,
            "ollama_host": self.conversation.host or "",
            "ollama_url": effective_host(self.conversation.host),
            "memory": self.memory is not None,
            # Memory switched on, for every thread, and the open thread in
            # amnesia mode
            "remember": self.conversation.remember,
            "amnesia": self.conversation.amnesia,
            # The thread the conversation is in; None for a new one
            "thread": self.conversation.thread,
            # Its role, if it has one that is still there
            "role": role.id if (role := find_role(self.conversation.role)) else None,
            "role_name": role.name if role else "",
            "wake_word": self.wake_word or "",
            "wake_window": self.wake_window,
            "sleep_word": self.sleep_word or "",
            # The persona's wake word, shown while the wake word is off
            "persona_wake_word": self.persona.wake_word,
            "mic_on": not self.pause_event.is_set(),
            "paused": self.suspend_event.is_set() or self.listening_paused,
            "tools": self.tools.states(),
            "engines": engines.describe(
                speech=self.tts_generator.engine,
                recognition=self.recognition_engine,
                models_dir=self.tts_generator.models_dir,
            ),
            "plugins": self.plugin_states(),
        }

    def _timing_components(self) -> dict:
        """What ran each stage of a turn, so a measurement can be matched to
        the setup it was taken with."""
        recorder = getattr(self, "recorder", None)
        transcriber = getattr(self, "transcriber", None)
        speech = getattr(self, "tts_generator", None)
        return {
            "turn_end": "smart_turn" if getattr(recorder, "turn_end", None) else "silence",
            "recognition": f"{getattr(self, 'recognition_engine', '')}:"
            f"{getattr(transcriber, 'model_name', '')}",
            "reply": getattr(getattr(self, "conversation", None), "model", ""),
            "voice": f"{getattr(speech, 'engine', '')}:{getattr(speech, 'voice', '')}",
        }

    def _report_timing(self):
        """Send the current turn's timing to the app, once."""
        timing = self._timing
        if timing is None or timing.reported:
            return
        timing.reported = True
        if self.event_bus:
            self.event_bus.emit("turn.timing", timing.event())

    def _reply_heard(self):
        """The voice started saying the reply: the turn's timing is complete."""
        timing = self._timing
        if timing is not None and not timing.reported:
            timing.first_audio()
            self._report_timing()

    def _reply_piece(self):
        """A part of the reply is ready to be spoken."""
        timing = self._timing
        if timing is None:
            return
        timing.first_piece()
        # Without the voice, the reply is complete once it is written
        if not self.speak_replies:
            self._report_timing()

    def plugin_states(self) -> list[dict]:
        """The plugins for the app: {id, label, active, available, source};
        active when its tools are on or one of its engines is in use."""
        in_use = {self.tts_generator.engine, self.recognition_engine}
        states = []
        for plugin in self.plugins:
            provides = {e.id for e in plugin.speech_engines()}
            provides |= {e.id for e in plugin.recognition_engines()}
            states.append(
                {
                    "id": plugin.id,
                    "label": plugin.name,
                    "active": self.tools.enabled(plugin.id) or bool(provides & in_use),
                    "available": plugin.available(),
                    "source": plugin.source,
                }
            )
        return states

    def set_turn_detection(self, mode: str) -> bool:
        """Hear the end of a turn with Smart Turn ("smart_turn"), or after a
        silence ("silence"). False when Smart Turn could not be loaded, which
        leaves the silence to end turns."""
        self.recorder.turn_end = SmartTurn.load() if mode == "smart_turn" else None
        return mode != "smart_turn" or self.recorder.turn_end is not None

    def set_wake_word(self, word: Optional[str], engine: str = "auto"):
        """Listen only for this word from now on (None or "" listens to all).

        The tiny detector model is loaded the first time it is needed.
        """
        word = (word or "").strip() or None
        if word:
            self._load_wake_detector(engine)
        self.wake_word = word
        self.awake_until = 0.0  # a new word starts dormant, without a chime
        self._was_dormant = True
        self._asleep_until_called = False

    def _load_wake_detector(self, engine: str = "auto"):
        """Load the tiny Whisper that hears the wake word, unless it is."""
        if self.wake_detector is not None:
            return
        # On the backend recognition runs on, when that is Whisper too
        if engine == "auto" and isinstance(self.transcriber, WhisperTranscriber):
            engine = self.transcriber.engine
        self.wake_detector = WakeWordDetector(engine)

    @property
    def word_to_wake(self) -> str:
        """The word that wakes Iris: the one listened for, or with the wake
        word off, the persona's (its name unless it has another)."""
        return self.wake_word or self.persona.wake_word

    def set_persona_words(self, wake_word: str, sleep_word: str):
        """Take on a persona's words, as it is switched to or edited. While
        a wake word is listened for, the new one is, and Iris stays awake
        or asleep as it was; with the wake word off it stays off."""
        if self.wake_word:
            self.wake_word = (wake_word or "").strip() or self.wake_word
        self.sleep_word = (sleep_word or "").strip() or None

    def set_wake_options(self, window: Optional[float] = None, sleep_word: Optional[str] = None):
        """Change how long it stays awake (0: until told to sleep), and its
        sleep word ("" for none). Awake now, it stays awake for the new
        window from now: one switched off must not still run out."""
        if window is not None:
            self.wake_window = _window(window)
            if self.wake_word and self.awake_until > time.monotonic():
                self._stay_awake()
        if sleep_word is not None:
            self.sleep_word = sleep_word.strip() or None

    @property
    def replying(self) -> bool:
        """A turn is being answered, or its reply is still heard."""
        return self._turn_active.is_set() or self.speech_pipeline.is_active

    @property
    def dormant(self) -> bool:
        """Waiting to be addressed by the wake word."""
        if self._asleep_until_called:
            return True
        return bool(self.wake_word) and time.monotonic() >= self.awake_until

    def _go_to_sleep(self):
        """End the conversation at once, as the sleep word does: the run
        loop then marks it (the sleep event, the falling chime) and listens
        only for the wake word. The turn that did it does not wake Iris
        again (awake_until is zero)."""
        self.awake_until = 0.0

    def _is_sleep_word(self, text: str) -> bool:
        """Whether this is the sleep word, said on its own or with the wake
        word or the persona's name ("Goodbye, Iris" for a "Hey Iris")."""
        if not self.sleep_word:
            return False
        return any(
            match_sleep_word(text, self.sleep_word, word)
            for word in dict.fromkeys([self.word_to_wake, self.persona.name])
        )

    def _sleep_word_said(self):
        """The sleep word always puts Iris to sleep, whatever the model would
        make of it. With the wake word off there is no window to wake into:
        Iris sleeps until called by the persona's wake word, then listens to
        everything again. Without the model that hears the wake word it
        could not be called back, so it stays awake and says so."""
        if not self.wake_word:
            try:
                self._load_wake_detector()
            except Exception as e:
                logger.exception("The wake word could not be set up")
                self._error(f"Iris cannot sleep, as nothing could wake it again. {brief(e)}")
                return
            self._asleep_until_called = True
        self._go_to_sleep()

    def _wake(self):
        """Woken by name, or by a typed message: awake for the window with
        the wake word on, and listening to everything again with it off."""
        self._asleep_until_called = False
        if self.wake_word:
            self._stay_awake()

    def _end_conversation(self):
        """The sleep tool: Iris goes to sleep once the reply being given has
        been heard to the end."""
        self._ending = True

    def _sleep_after_goodbye(self):
        """Go to sleep after the parting words were heard. The voice is left
        a moment to die away in the room first (see run_loop): the falling
        chime must not cut its last words, and the microphone, listening for
        the wake word, must not hear them, as they may well hold the wake
        word itself."""
        if self.speak_replies:
            self._quiet_until = time.monotonic() + GOODBYE_SETTLE
        self._go_to_sleep()

    def _stay_awake(self):
        """Start the awake window over and tell the UI how long it lasts.
        Without a window it stays awake until told to sleep, and there is
        nothing to count down."""
        if not self.wake_window:
            self.awake_until = math.inf
            return
        self.awake_until = time.monotonic() + self.wake_window
        self._last_window_emit = time.monotonic()
        if self.event_bus:
            self.event_bus.emit("wake.window", {"remaining": self.wake_window})

    def _keep_awake_while_speaking(self):
        """The user is talking: the window must not run out mid-sentence.
        The UI is told at most once a second."""
        if not self.wake_window:
            return
        now = time.monotonic()
        self.awake_until = max(self.awake_until, now + self.wake_window)
        if now - self._last_window_emit >= 1.0:
            self._last_window_emit = now
            if self.event_bus:
                self.event_bus.emit("wake.window", {"remaining": self.wake_window})

    def _note_dormancy(self, dormant: bool):
        """Mark the moment the assistant falls asleep: a falling chime, and
        the UI learns the awake window is over."""
        if dormant and not self._was_dormant:
            if self.event_bus:
                self.event_bus.emit("wake.window", {"remaining": 0})
                self.event_bus.emit("sleep", {})
            # Only audible when it was actually listening for follow-ups
            listening = not (self.pause_event.is_set() or self.suspend_event.is_set())
            if self.speak_replies and listening:
                self._play_chime(_sleep_chime())
        self._was_dormant = dormant

    def _chime(self):
        """Acknowledge the wake word when no request came with it."""
        if self.speak_replies:
            self._play_chime(_wake_chime())

    def thread_chime(self):
        """Mark going on in another thread, or a new one."""
        if self.speak_replies:
            self._play_chime(_thread_chime())

    def _play_chime(self, sound: np.ndarray):
        """Play a chime. A speaker that cannot be opened leaves it unheard:
        the next reply tries it again, and says so if it fails too."""
        try:
            self.player.audio.play(sound, 24000)
        except Exception:
            logger.warning("The chime could not be played", exc_info=True)

    def _handle_heard(self, audio_path: str, dormant: bool):
        """Answer an utterance, or while dormant, check it for the wake word.

        "Iris, what time is it?" wakes and is answered in one go, transcribed
        by the conversation model; a bare "Iris" chimes and waits.
        """
        if dormant:
            audio = decode_audio(audio_path, sampling_rate=16000)
            word = self.word_to_wake
            woken, remainder = self.wake_detector.heard(audio, word)
            if not woken:
                return
            self._wake()
            if self.event_bus:
                self.event_bus.emit("wake", {"word": word})
            if len(remainder.split()) < 2:
                self._chime()
                return
        self._run_voice_turn(audio_path)
        # Unless the turn just put it to sleep: the sleep word, or the reply
        # ending the conversation (awake_until is zero)
        if self.wake_word and self.awake_until > 0:
            self._stay_awake()

    def _mic_closed(self) -> bool:
        """Paused, mic off, the user typing, or shutting down."""
        return (
            self.pause_event.is_set()
            or self.suspend_event.is_set()
            or self.listening_paused
            or self.hold_event.is_set()
            or self.stop_event.is_set()
        )

    def _answer_listening(self, audio_path: str, dormant: bool):
        """Answer a turn the microphone heard, listening on until the answer
        has been spoken.

        Smart Turn can take a pause for the end of a turn that was not over.
        A user who goes on takes the turn back: the answer begun is dropped,
        and the whole turn is answered once it is over. With echo
        cancellation the user can also interrupt the voice, as in Pipecat:
        it stops, what was said of the answer is kept, and what the user
        says is the next turn. Without it the microphone closes when the
        voice starts, which it would hear.
        """
        while audio_path:
            # Nothing of this turn's answer has been said yet: what the voice
            # said of the previous one must not count as heard, or a user who
            # pauses and goes on loses what they said before the pause
            self.speech_pipeline.begin_reply()
            answered = threading.Event()
            went_on = threading.Event()
            whole_turn: list[Optional[str]] = [None]
            going_on = [True]
            through_voice = self.barge_in and self.echo is not None

            def on_speech() -> bool:
                went_on.set()
                self.interrupt_event.set()
                self.speech_pipeline.stop()
                self.speech_pipeline.resume()  # if it was held: stopped now
                with self.status_manager.status_lock:
                    self.status_manager.status_listening = True
                # Once some of the answer was heard, the user answers it
                going_on[0] = not self.speech_pipeline.said()
                return going_on[0]

            def on_level(level: float):
                self._last_voice = time.monotonic()
                if self.event_bus:
                    self.event_bus.emit("audio.level", {"rms": level})

            def voice_or_answer() -> bool:
                if through_voice:
                    return answered.is_set()
                return answered.is_set() or self.speech_pipeline.is_speaking_flag.is_set()

            def speaking() -> bool:
                """Whether the voice is speaking the answer (between two of
                its clips too, and until its stream has played out: when
                the voice comes out of the speaker cannot be told more
                closely), and not paused by the user."""
                return self.speech_pipeline.is_active and not self.suspend_event.is_set()

            def release(rewind: bool = True):
                """Let the voice go on after a hold, unless the user paused
                the conversation meanwhile: a little before where it was cut
                (rewind), or exactly there, when the hold heard nothing of
                the user."""
                if not self.suspend_event.is_set():
                    self.speech_pipeline.resume(rewind)

            def listen_on(turn=audio_path):
                whole_turn[0] = self.recorder.listen_on(
                    turn,
                    on_speech=on_speech,
                    until=voice_or_answer,
                    should_stop=self._mic_closed,
                    on_level=on_level,
                    speaking=speaking if through_voice else None,
                    hold=self.speech_pipeline.pause,
                    release=release,
                )

            listener = threading.Thread(target=listen_on, daemon=True)
            listener.start()
            self._announced = False
            try:
                self._handle_heard(audio_path, dormant)
            finally:
                # Also on an error, or the microphone would stay open
                answered.set()
                listener.join()
                with self.status_manager.status_lock:
                    self.status_manager.status_listening = False
                Path(audio_path).unlink(missing_ok=True)
            audio_path = whole_turn[0] if went_on.is_set() else None
            if audio_path:
                self.interrupt_event.clear()
                # The turn shown and the reply begun to it give way to the whole turn
                if going_on[0] and self._announced and self.event_bus:
                    self.event_bus.emit("chat.continued", {})
                dormant = False

    def _speak(self, text: str):
        """Queue reply text for speech, unless voice output is switched off.
        A piece in a language the persona's voice does not speak properly is
        only shown: no other voice says it in its place. So is one in a
        language the voice is not ready for yet: what it needs is never
        fetched or made here, unseen, but by the app, with its progress in
        the conversation, and the voice speaks the language once it is. A
        cloned voice that is not here is ready for none."""
        if not self.speak_replies:
            return
        clean_text = self.conversation.clean_markdown(text)
        if not clean_text.strip():
            return
        spoken_in = self._speech_language(clean_text)
        if not self.tts_generator.speaks(spoken_in):
            self._shown_only(spoken_in)
            return
        if not self.tts_generator.ready(spoken_in):
            self._not_ready(spoken_in)
            return
        self._reply_voiced = True
        self.speech_pipeline.queue_speech(clean_text, language=spoken_in)

    def _not_ready(self, language_code: str):
        """A piece of the reply is shown, not spoken: the persona's voice
        speaks its language, but is not ready for it yet. The app is told,
        to make it ready with its progress shown; without one to tell, the
        user is told once for the voice and language. A cloned voice that is
        not here cannot be made ready: the user is told to choose another."""
        self._reply_unspoken = True
        if not self._reply_voiced:
            self._report_timing()
        generator = self.tts_generator
        if cloning.voice_missing(generator.voice):
            self._voice_missing()
            return
        if self.on_voice_not_ready is not None:
            self.on_voice_not_ready(generator.engine, generator.voice, language_code)
            return
        told = (generator.engine, generator.voice, language_code)
        if told in self._unspoken_told:
            return
        self._unspoken_told.add(told)
        name = LANGUAGE_NAMES.get(language_code, language_code)
        if self.event_bus:
            self.event_bus.emit(
                "notice",
                {
                    "message": f"{self.persona.name}'s voice is not ready for {name} yet: "
                    f"replies in {name} are shown until it is.",
                    "kind": "not_ready",
                    "language": language_code,
                },
            )

    def _voice_missing(self):
        """The persona's voice is a cloned one that is not here, which
        nothing can make ready: the user is told once for the voice which
        persona speaks with it, and where to give it another."""
        generator = self.tts_generator
        told = (generator.engine, generator.voice, "")
        if told in self._unspoken_told:
            return
        self._unspoken_told.add(told)
        persona = self.persona.name
        name = cloning.voice_name(cloning.voice_folder(generator.voice))
        logger.info("%s's voice %s cannot be found: shown, not spoken", persona, generator.voice)
        if self.event_bus:
            self.event_bus.emit(
                "notice",
                {
                    "message": f"{persona}'s voice, {name}, cannot be found. "
                    f"Choose another voice for {persona} in the persona editor.",
                    "kind": "voice_missing",
                },
            )

    def _shown_only(self, language_code: str):
        """A piece of the reply is shown and not spoken, its language one the
        persona's voice does not speak. The user is told once for the voice
        and language. A reply nothing of which is spoken is complete once it
        is written, as with the voice off."""
        self._reply_unspoken = True
        if not self._reply_voiced:
            self._report_timing()
        generator = self.tts_generator
        told = (generator.engine, generator.voice, language_code)
        if told in self._unspoken_told:
            return
        self._unspoken_told.add(told)
        name = LANGUAGE_NAMES.get(language_code, language_code)
        logger.info("%s's voice does not speak %s: shown, not spoken", self.persona.name, name)
        if self.event_bus:
            self.event_bus.emit(
                "notice",
                {
                    "message": f"{self.persona.name}'s voice does not speak {name}: "
                    f"replies in {name} are shown, not spoken.",
                    "kind": "unspoken",
                    "language": language_code,
                },
            )

    def _prepare_voice(self, language: str):
        """Load what a reply in this language needs, ahead of it (see
        SpeechGenerator.prepare_language)."""
        try:
            self.tts_generator.prepare_language(language)
        except Exception:
            logger.exception("Could not prepare the voice for %s", language)

    def _speech_language(self, text: str) -> str:
        """The language a piece of the reply is spoken in: the one it is
        written in. The model may answer in another language than the
        user's (asked in English whether it speaks Italian, it answers in
        Italian), and spoken in the user's, the reply has a foreign accent.
        A piece too short to tell is spoken like the one before it."""
        self._reply_language = language.language_of(
            text, self._reply_language, self.tts_generator.languages()
        )
        return self._reply_language

    def _take_turn(self) -> dict | None:
        """Return the next turn submitted by an external UI, if any."""
        try:
            return self.turn_queue.get_nowait()
        except queue.Empty:
            return None

    def _conversation_busy(self) -> bool:
        """The conversation is using, or about to use, the LLM: a turn is
        underway, speech plays, a message waits, a draft is being typed, or
        the user was just heard speaking."""
        return (
            self._turn_active.is_set()
            or self.speech_pipeline.is_active
            or self.hold_event.is_set()
            or not self.turn_queue.empty()
            or time.monotonic() - self._last_voice < 2.0
        )

    def forget_everything(self):
        """Forget the conversation and all that is remembered, as if Iris
        had never been spoken to: every turn and topic in memory, what the
        next reply would be given of the conversation, and the messages
        waiting to be answered. A reply underway is stopped first, as Stop
        stops it, and no turn starts until all is forgotten. The app is told
        (memory.forgotten) so its conversation goes too: one shown that the
        model no longer knows would mislead. Amnesia mode stays as it is."""
        self.interrupt_event.set()
        self.speech_pipeline.stop()
        # The turn underway ends once stopped. One that does not end in
        # time cannot keep its exchange either (ConversationManager.forget)
        took = self._turn_lock.acquire(timeout=FORGET_WAIT)
        try:
            if took:
                # The stop was for the turn underway, not for the next one
                self.interrupt_event.clear()
            while True:
                try:
                    self.turn_queue.get_nowait()
                except queue.Empty:
                    break
            self.conversation.forget()
            if self.memory:
                self.memory.store.forget_all()
            self.conversation_language = "en"
            self._language_known = False
            if self.event_bus:
                self.event_bus.emit("memory.forgotten", {})
        finally:
            if took:
                self._turn_lock.release()

    def _run_text_turn(self, turn: dict):
        """Answer a typed message, exactly as if it had been spoken."""
        self._turn_active.set()
        try:
            self._answer_typed(turn)
        finally:
            self._turn_active.clear()

    def _answer_typed(self, turn: dict):
        text = turn["text"]
        attachments = turn.get("attachments") or []
        # A typed message wakes Iris, as its name does
        self._wake()
        # The conversation goes on in the language typed, as if spoken
        self.conversation_language = language.language_of(
            text, self.conversation_language, self.tts_generator.languages()
        )
        self._language_known = True
        self._timing = TurnTiming("text", components=self._timing_components())
        if self.event_bus:
            event = {
                "text": text,
                "language": self.conversation_language,
                "source": "text",
                "id": turn.get("id"),
            }
            if attachments:
                event["attachments"] = [a.describe() for a in attachments]
            self.event_bus.emit("chat.user", event)
            if any(a.kind == "image" for a in attachments) and not self.conversation.image_model():
                self.event_bus.emit("error", {"message": NO_VISION, "kind": "no_vision"})
        self.chat(
            text,
            response_language=self.conversation_language,
            source="text",
            attachments=attachments,
        )
        # Unless the reply ended the conversation (awake_until is zero)
        if self.wake_word and self.awake_until > 0:
            self._stay_awake()

    def _greet(self):
        """Say hello at the end of the walkthrough: the model words it in the
        persona's manner, and it opens the conversation with no turn of the
        user's before it. Iris stays awake for the answer, as after a turn."""
        self._turn_active.set()
        try:
            self._wake()
            self._timing = None
            self.chat(GREETING, response_language=self.conversation_language, source=OPENING)
            if self.wake_word and self.awake_until > 0:
                self._stay_awake()
        finally:
            self._turn_active.clear()

    def _run_voice_turn(self, audio_path: str):
        """Transcribe recorded speech and answer it."""
        self._turn_active.set()
        try:
            self._answer_spoken(audio_path)
        finally:
            self._turn_active.clear()

    def _answer_spoken(self, audio_path: str):
        with self.status_manager.status_lock:
            self.status_manager.status_understanding = True

        turn_wait = getattr(getattr(self, "recorder", None), "last_turn_wait", None)
        recognizing = time.monotonic()
        # Its first syllable: the recording ends as the turn is taken (with
        # the silence waited for), and begins PRE_SPEECH before it
        turn_began = recognizing - _seconds_of(audio_path) + PRE_SPEECH
        # What the user says is written in the conversation's language unless
        # Whisper clearly hears another: an accent can fool it
        text, detected_language = self.transcriber.transcribe(
            audio_path,
            self.conversation_language if self._language_known else None,
            self.chosen_languages,
        )
        self._timing = TurnTiming(
            "voice",
            turn_end=turn_wait,
            recognition=time.monotonic() - recognizing,
            components=self._timing_components(),
        )

        with self.status_manager.status_lock:
            self.status_manager.status_understanding = False

        # The user went on meanwhile: the whole turn is answered instead
        if not text or self.interrupt_event.is_set():
            return

        # The microphone heard Iris's own voice (an echo the canceller let
        # through, for instance while the computer was too busy to play on
        # time): answering it would have Iris talk to itself, turn after turn
        pipeline = getattr(self, "speech_pipeline", None)
        spoken_since = getattr(pipeline, "spoken_since", None)
        if spoken_since and echo_of(text, spoken_since(turn_began - ECHO_AFTER)):
            logger.info("Took no turn of the voice's own words: %r", text)
            return

        # "That's all, Iris": back to sleep, nothing to answer, also with the
        # wake word off (the persona's then wakes it)
        if self._is_sleep_word(text):
            self._sleep_word_said()
            return

        # Determine response language: the one the user spoke, if Iris
        # talks in it (whether the persona's voice speaks it or not),
        # otherwise keep the current conversation language. Whisper guesses
        # it from the sound, and a few words in an accent can fool it
        # ("Parliamo italiano." heard as Portuguese); the words heard tell it
        # better when they are clear.
        spoken = language.language_of(text, detected_language, self.tts_generator.languages())
        if spoken in self.tts_generator.languages():
            self.conversation_language = spoken
        self._language_known = True
        response_language = self.conversation_language

        self._announced = True
        if self.event_bus:
            self.event_bus.emit(
                "chat.user",
                {"text": text, "language": response_language, "source": "voice"},
            )

        # Send to LLM (streams response and speaks it)
        self.chat(text, response_language=response_language, source="voice")

    def _reply_chunks(self, stream):
        """The reply's chunks as they come, ending at once when the turn is
        interrupted.

        The model is read on a thread of its own: while it reasons, runs a
        tool or is slow to start, nothing arrives, and Stop would otherwise
        wait for the next chunk.
        """
        chunks: queue.Queue = queue.Queue()
        done = object()

        def read():
            try:
                for chunk in stream:
                    chunks.put(chunk)
                chunks.put(done)
            except Exception as error:
                chunks.put(error)

        threading.Thread(target=read, daemon=True).start()
        while True:
            try:
                chunk = chunks.get(timeout=0.05)
            except queue.Empty:
                if self.interrupt_event.is_set():
                    return
                continue
            if chunk is done:
                return
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def _keep_exchange(self, prompt: str, answer: str, source: Optional[str], cut: bool):
        """Keep the exchange in the conversation as the user took it in: a
        reply cut short only as far as it was said, or shown, with the voice
        off or in a language it does not speak. Of a reply stopped before
        anything was said, nothing is kept, not even the turn it answered."""
        if not cut:
            self.conversation.keep(prompt, self.conversation.reply or "", source)
            return
        if self.speak_replies and not self._reply_unspoken:
            kept = self.speech_pipeline.said()
        else:
            kept = self.conversation.clean_markdown(answer)
        if kept.strip():
            self.conversation.keep(prompt, kept, source)
        else:
            self.conversation.forgo()

    def chat(
        self,
        prompt: str,
        response_language: str = "en",
        source: Optional[str] = None,
        attachments: Optional[list[Attachment]] = None,
    ) -> str:
        """
        Send prompt to LLM, stream response, and speak it.

        Args:
            prompt: User prompt text
            response_language: Language to respond in (Whisper code)
            attachments: Files sent with the prompt

        Returns:
            LLM response text
        """
        try:
            # Set by the sleep tool, should the model end the conversation
            self._ending = False

            # Set thinking status
            with self.status_manager.status_lock:
                self.status_manager.status_thinking = True

            # Set the speech pipeline language for this turn; each piece of
            # the reply is spoken in the language it is written in
            self.speech_pipeline.current_language = response_language
            self._reply_language = response_language
            self._reply_voiced = self._reply_unspoken = False
            # What speaking that language needs loads while the model thinks
            if self.speak_replies:
                threading.Thread(
                    target=self._prepare_voice, args=(response_language,), daemon=True
                ).start()
            self.speech_pipeline.begin_reply()
            if self._timing is not None:
                self._timing.reply_started()

            # Stream the response
            full_response = []
            text_buffer = ""  # Buffer for speech chunks
            last_piece = 0  # length of the piece spoken last

            def on_tool_use(tool_name, tool_args):
                label, detail = self.tools.describe(tool_name, tool_args)
                if self.event_bus:
                    self.event_bus.emit(
                        "tool.use", {"tool": tool_name, "label": label, "detail": detail}
                    )

            def on_tool_result(tool_name, result):
                # What only the user can fix is told them, not just the model
                if not self.event_bus:
                    return
                if result.get("notice"):
                    self.event_bus.emit("error", {"message": result["notice"]})
                if result.get("image") and not self.conversation.image_model():
                    self.event_bus.emit("error", {"message": NO_VISION, "kind": "no_vision"})

            interrupted = False
            try:
                stream = self.conversation.chat_stream(
                    prompt,
                    response_language=response_language,
                    on_tool_use=on_tool_use,
                    source=source,
                    attachments=attachments,
                    keep=False,  # kept below, once it is known what was heard
                    on_tool_result=on_tool_result,
                )
                for chunk_text in self._reply_chunks(stream):
                    # Paused: stop consuming the reply until resumed (or stopped)
                    while self.suspend_event.is_set() and not self.interrupt_event.is_set():
                        time.sleep(0.05)

                    # Barge-in: cancel the rest of this turn
                    if self.interrupt_event.is_set():
                        interrupted = True
                        self.interrupt_event.clear()
                        break

                    full_response.append(chunk_text)
                    text_buffer += chunk_text

                    if self.event_bus:
                        self.event_bus.emit("chat.chunk", {"text": chunk_text})

                    # Speak each piece as it is complete (queued for
                    # sequential playback); the first one as soon as its
                    # first sentence is there
                    while True:
                        piece, text_buffer = split_speech(text_buffer, last_piece)
                        if not piece:
                            break
                        last_piece = len(piece)
                        self._reply_piece()
                        self._speak(piece)

                # Stopped while nothing was arriving
                if not interrupted and self.interrupt_event.is_set():
                    interrupted = True
                    self.interrupt_event.clear()

                # Clear thinking status now that LLM is done
                with self.status_manager.status_lock:
                    self.status_manager.status_thinking = False

                # Collect final answer
                answer = "".join(full_response)

                if interrupted:
                    # Stop generating (cutting the request off frees the
                    # model at once) and drop all pending speech
                    self.conversation.cancel()
                    self.speech_pipeline.stop()
                    text_buffer = ""

                if self.event_bus:
                    self.event_bus.emit("chat.done", {"text": answer, "interrupted": interrupted})

            except Exception:
                # The reply broke off (the server stopped answering, say):
                # what came of it stays shown, as a reply cut short
                if full_response and self.event_bus:
                    self.event_bus.emit(
                        "chat.done", {"text": "".join(full_response), "interrupted": True}
                    )
                raise
            finally:
                # Clear thinking status if still set
                with self.status_manager.status_lock:
                    self.status_manager.status_thinking = False

            # Speak any remaining text that wasn't spoken yet, unless the
            # reply was stopped in the meantime
            if text_buffer.strip() and not self.interrupt_event.is_set():
                self._reply_piece()
                self._speak(text_buffer)

            # Wait for all speech to complete
            self.speech_pipeline.wait_for_speech()

            # Stopped while streaming, or while speaking after the stream
            cut = interrupted or self.interrupt_event.is_set()
            self._keep_exchange(recorded_text(prompt, attachments or []), answer, source, cut)

            # The model ended the conversation: Iris goes to sleep now that
            # its parting words were heard to the end. Cut short, the user
            # was not done after all.
            if self._ending and not cut:
                self._sleep_after_goodbye()
            return answer

        except Exception as e:
            # Clear all statuses on error
            with self.status_manager.status_lock:
                self.status_manager.status_thinking = False
            # Nothing of a reply that failed is kept, nor the turn it answered
            self.conversation.forgo()

            conversation = self.conversation
            error_msg = failure_message(e, conversation.host, getattr(conversation, "asked", None))
            if self.event_bus:
                event = {"message": error_msg}
                if model_missing(e):
                    # Retracted by the app once another model is chosen
                    event["kind"] = "model_missing"
                self.event_bus.emit("error", event)
            logger.exception("Reply failed")
            return error_msg

    def _mic_lost(self):
        """Switch the microphone off after it failed, and show it off."""
        self.pause_event.set()
        self.recorder._device_error = False
        self.recorder.close_microphone()
        if self.event_bus:
            self.event_bus.emit("config", self.config_info())

    def _error(self, message: str):
        """Tell the app about a problem it should show."""
        if self.event_bus:
            self.event_bus.emit("error", {"message": message})
        logger.error(message)

    def _notice(self, message: str):
        """Tell the app something it shows in passing, not a problem."""
        if self.event_bus:
            self.event_bus.emit("notice", {"message": message})
        logger.info(message)

    def _devices_in_use(self) -> dict:
        """The speaker and the microphone Iris plays and listens through."""
        try:
            output = self.player.audio.query_devices(kind="output")["name"]
        except Exception:
            output = None
        return {"output": output, "input": self.recorder.current_device_name()}

    def _devices_changed(self):
        """The devices were read anew (one came or went, or the system's
        defaults changed): tell the user what Iris speaks or listens through
        now, if that is another device. The microphone is told of only while
        it is on."""
        now = self._devices_in_use()
        before, self._in_use = self._in_use, now
        changes = []
        if now["output"] and now["output"] != before.get("output"):
            changes.append(f"speaking through {now['output']}")
        if now["input"] != before.get("input") and not self.pause_event.is_set():
            changes.append(f"listening through {now['input']}")
        if changes:
            self._notice("Now " + " and ".join(changes) + ".")
        if self.event_bus:
            self.event_bus.emit("config", self.config_info())

    def run_loop(self):
        """Listen and answer until stop_event is set.

        Nothing that goes wrong in one round ends the loop: the error is
        logged and the loop goes on, listening and answering. Should it end
        all the same (the controller starts it again), Iris is shut down only
        once stop_event is set.
        """
        devices_ok, device_msg = self.recorder.check_audio_devices()
        if not devices_ok:
            # Typed messages are still answered; the microphone is off until
            # the user switches it on again
            self._error(f"No microphone: {device_msg}")
            self._mic_lost()

        # Told when the devices are read anew, to say where Iris speaks and
        # listens now (a device of a test's own is never read anew)
        audio = getattr(self.player, "audio", None)
        watch = getattr(audio, "watch", None)
        if watch is not None:
            self._in_use = self._devices_in_use()
            watch(self._devices_changed)
        try:
            while not self.stop_event.is_set():
                try:
                    self._loop_round()
                except Exception:
                    logger.exception("A round of the assistant's loop failed")
                    time.sleep(LOOP_RETRY)
        finally:
            if watch is not None:
                audio.unwatch(self._devices_changed)
            for close in (self.speech_pipeline.stop, self.recorder.close_microphone):
                try:
                    close()
                except Exception:
                    logger.exception("Could not close the audio")
            if self.stop_event.is_set():
                self.shut_down()

    def shut_down(self):
        """Release what Iris holds besides the audio: the tools' resources,
        and memory's filing in the background."""
        if self.tools:
            self.tools.cleanup()
        if self.memory:
            self.memory.stop()

    def _loop_round(self):
        """One round of the loop: a turn taken, a listen, or a moment's wait
        while there is nothing to listen for."""
        # After parting words Iris falls asleep once the voice has
        # died away (see _sleep_after_goodbye): until then the
        # microphone stays closed, and the orb is already asleep
        if time.monotonic() < self._quiet_until:
            with self.status_manager.status_lock:
                self.status_manager.status_listening = False
                self.status_manager.status_dormant = self.dormant
            time.sleep(0.05)
            return

        # Notice falling asleep whatever the loop is doing, so the UI
        # learns it even with the mic off
        self._note_dormancy(self.dormant)

        # Stale barge-in from a turn that already ended
        if self.interrupt_event.is_set():
            self.interrupt_event.clear()
            self.speech_pipeline.stop()

        # Paused conversation: nothing starts until it is resumed
        if self.suspend_event.is_set():
            self.recorder.close_microphone()
            with self.status_manager.status_lock:
                self.status_manager.status_listening = False
            time.sleep(0.05)
            return

        # Turns from an external UI come first, whether or not the mic
        # is live, so a typed message never waits for the user to speak
        with self._turn_lock:
            turn = self._take_turn()
            if turn is not None:
                try:
                    if "audio" in turn:
                        self._handle_heard(turn["audio"], self.dormant)
                    elif turn.get("greet"):
                        self._greet()
                    else:
                        self._run_text_turn(turn)
                except Exception:
                    logger.exception("Turn failed")
        if turn is not None:
            return

        # A voice is being previewed: the microphone would hear it as
        # the user, and answer it. It stays closed until the room is
        # quiet again.
        if self.speech_pipeline.is_active:
            with self.status_manager.status_lock:
                self.status_manager.status_listening = False
            while self.speech_pipeline.is_active and not self.stop_event.is_set():
                time.sleep(0.05)
            time.sleep(PREVIEW_SETTLE)
            return

        # Paused, the user is typing, or nothing to recognize speech
        # with: keep the microphone closed (models stay loaded) and
        # keep watching for typed turns
        closed = (
            self.pause_event.is_set()
            or self.hold_event.is_set()
            or self.listening_paused
        )
        if closed or self.transcriber is None:
            self.recorder.close_microphone()
            with self.status_manager.status_lock:
                self.status_manager.status_listening = False
            time.sleep(0.05)
            return

        try:
            # 1. Listen for voice input with level visualization;
            # with a wake word set and no recent turn, only for that
            dormant = self.dormant
            with self.status_manager.status_lock:
                self.status_manager.status_listening = not dormant
                self.status_manager.status_dormant = dormant

            last_level_emit = [0.0]
            heard_speech = [False]

            def on_audio_level(level: float):
                """Update audio level display."""
                if level * 0.15 > self.recorder.silence_threshold:
                    heard_speech[0] = True
                    # Talk Iris is not part of (while dormant) is not
                    # conversation: memory filing may use the time
                    if not dormant:
                        self._last_voice = time.monotonic()
                        if self.wake_word:
                            self._keep_awake_while_speaking()
                # Throttle level events to ~15 Hz for external UIs
                if self.event_bus:
                    now = time.monotonic()
                    if now - last_level_emit[0] >= 1 / 15:
                        last_level_emit[0] = now
                        self.event_bus.emit("audio.level", {"rms": level})

            audio_path = self.recorder.listen(
                on_level=on_audio_level,
                should_stop=lambda: self.pause_event.is_set()
                or self.suspend_event.is_set()
                or self.listening_paused
                or self.hold_event.is_set()
                or self.stop_event.is_set()
                or not self.turn_queue.empty()
                # A voice preview started
                or self.speech_pipeline.is_active
                # The awake window ran out with nobody speaking
                or (not dormant and not heard_speech[0] and self.dormant)
                # The wake word was switched off: listen to everything
                or (dormant and not self.dormant),
                silence_duration=DORMANT_SILENCE if dormant else None,
            )

            with self.status_manager.status_lock:
                self.status_manager.status_listening = False
                self.status_manager.status_dormant = False

            # The microphone went away: typed messages are still
            # answered, and the microphone is off until switched on
            if self.recorder._device_error:
                self.speech_pipeline.stop()
                self._error(
                    "The microphone was disconnected. Connect one and switch "
                    "the microphone on again."
                )
                self._mic_lost()
                return

            if not audio_path:
                return

            # 2. Transcribe, then 3. answer (streams and speaks the
            # reply), listening on in case the user was not finished
            with self._turn_lock:
                self._answer_listening(audio_path, dormant)

        except Exception:
            logger.exception("Listening failed")

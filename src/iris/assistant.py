"""Voice assistant coordinator that orchestrates all components."""

import logging
import os
import queue
import re
import threading
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from faster_whisper.audio import decode_audio

from iris import engines
from iris.attachments import Attachment, recorded_text
from iris.audio import AudioPlayer, AudioRecorder, SoundDevice, get_filter
from iris.echo import HAS_ECHO_CANCELLATION, EchoCanceller
from iris.events import EventBus
from iris.llm import ConversationManager, effective_host, split_speech
from iris.memory import DEFAULT_PATH, MemoryKeeper, MemoryStore, chat_json
from iris.personas import Persona, get_persona
from iris.plugin import Plugin, PluginContext
from iris.plugins import BUILT_IN
from iris.status import StatusManager
from iris.timing import TurnTiming
from iris.tools import Tools
from iris.transcription import WhisperTranscriber
from iris.tts import SpeechGenerator, SpeechPipeline
from iris.turns import HOLD_SETTLE, PRE_SPEECH, SPEECH_START, SmartTurn
from iris.wakeword import WakeWordDetector, match_sleep_word

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# With a wake word set: how long the assistant keeps listening after a turn
# (or after the user last spoke) before it goes back to sleep
DEFAULT_WAKE_WINDOW = 30.0
# Said while awake, this puts the assistant straight back to sleep
DEFAULT_SLEEP_WORD = "That's all"
# While dormant an utterance ends after this much silence, so the wake word
# is recognized promptly instead of after the conversational pause
DORMANT_SILENCE = 0.8


# Seconds the microphone stays closed after a voice preview, for the room
# to fall quiet
PREVIEW_SETTLE = 0.4

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


def _sleep_chime() -> np.ndarray:
    """The same tones falling: "I'm going to sleep"."""
    return _chime_tones((990.0, 660.0))


def _echo_canceller() -> Optional[EchoCanceller]:
    """The echo canceller, or None where it cannot run."""
    if not HAS_ECHO_CANCELLATION:
        return None
    try:
        return EchoCanceller()
    except Exception:
        return None


# A turn that begins while the voice is heard, or this long after it fell
# quiet (its last syllables still arrive, through the room and the device),
# may be its echo. A user answering at once begins later than that.
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
        sleep_word: Optional[str] = DEFAULT_SLEEP_WORD,
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
            memory: Keep one continuous conversation in long-term memory
                (~/.iris/memory.db, or IRIS_MEMORY_DB), remembered by topic
            wake_word: If set, listen only for this word until addressed, and
                fall dormant again after wake_window seconds without a turn
            wake_window: Seconds the assistant stays awake after a turn
            sleep_word: Said while awake, puts the assistant back to sleep
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

        # Turns submitted by an external UI: {"text": str, "id": str} for typed
        # messages, {"audio": path} for recorded speech
        self.turn_queue = turn_queue if turn_queue is not None else queue.Queue()

        # Voice output toggle; False turns the assistant into a text chat
        self.speak_replies = speak_replies

        # Wake word mode: off when None; awake_until is when it falls dormant
        self.wake_word: Optional[str] = None
        self.wake_detector: Optional[WakeWordDetector] = None
        self.awake_until = 0.0
        self.wake_window = float(wake_window)
        self.sleep_word = (sleep_word or "").strip() or None
        self._was_dormant = True  # to notice the moment it falls asleep
        self._last_window_emit = 0.0

        # Track current conversation language for continuity
        self.conversation_language = "en"

        # The persona speaks with its own engine and voice
        self.persona = persona if persona else get_persona("iris")

        # Initialize components
        # Smart Turn hears when the user has finished speaking; without it
        # (offline on a first run) a turn ends after the silence. The echo
        # canceller, told what the player plays, lets the microphone hear the
        # user over the voice.
        self.barge_in = barge_in
        self.echo = _echo_canceller()
        self.recorder = AudioRecorder(
            sample_rate,
            silence_threshold,
            silence_duration,
            turn_end=SmartTurn.load() if turn_detection == "smart_turn" else None,
            echo=self.echo,
            audio=audio,
        )

        # Initialize audio player with filter from persona
        audio_filter = get_filter(self.persona.audio_filter) if self.persona.audio_filter else None
        self.player = AudioPlayer(audio_filter=audio_filter, audio=audio)
        self.player.echo = self.echo

        self.recognition_engine = recognition_engine
        self.transcriber = engines.recognition_engine(recognition_engine).create(
            {"model": whisper_model, "device": whisper_device, "engine": whisper_engine}
        )

        # Initialize TTS with persona's engine, voice and speed
        self.tts_generator = SpeechGenerator(
            self.persona.engine,
            self.persona.voice,
            self.persona.speed,
            models_dir=tts_models_dir,
        )
        self.speech_pipeline = SpeechPipeline(self.tts_generator, self.player)
        self.speech_pipeline.on_reply_audio = self._reply_heard

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
                busy=self._conversation_busy,
            )

        # The plugins' tools, on as chosen; a plugin whose optional
        # dependencies are missing stays off and shows as not installed
        self.plugins = list(plugins) if plugins is not None else list(BUILT_IN)
        self.tools = Tools(
            self.plugins,
            PluginContext(memory=self.memory.store if self.memory else None),
            status_manager=self.status_manager,
            choices=tools,
        )

        self.set_wake_word(wake_word, engine=whisper_engine)

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
        if self.memory:
            self.memory.start()

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
            "amnesia": self.conversation.amnesia,
            "wake_word": self.wake_word or "",
            "wake_window": self.wake_window,
            "sleep_word": self.sleep_word or "",
            "mic_on": not self.pause_event.is_set(),
            "paused": self.suspend_event.is_set(),
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
        if word and self.wake_detector is None:
            # The wake word is heard by a tiny Whisper, on the backend
            # recognition runs on when that is Whisper too
            if engine == "auto" and isinstance(self.transcriber, WhisperTranscriber):
                engine = self.transcriber.engine
            self.wake_detector = WakeWordDetector(engine)
        self.wake_word = word
        self.awake_until = 0.0  # a new word starts dormant, without a chime
        self._was_dormant = True

    def set_wake_options(self, window: Optional[float] = None, sleep_word: Optional[str] = None):
        """Change how long it stays awake, and its sleep word ("" for none)."""
        if window is not None:
            self.wake_window = max(5.0, float(window))
        if sleep_word is not None:
            self.sleep_word = sleep_word.strip() or None

    @property
    def dormant(self) -> bool:
        """Waiting to be addressed by the wake word."""
        return bool(self.wake_word) and time.monotonic() >= self.awake_until

    def _stay_awake(self):
        """Start the awake window over and tell the UI how long it lasts."""
        self.awake_until = time.monotonic() + self.wake_window
        self._last_window_emit = time.monotonic()
        if self.event_bus:
            self.event_bus.emit("wake.window", {"remaining": self.wake_window})

    def _keep_awake_while_speaking(self):
        """The user is talking: the window must not run out mid-sentence.
        The UI is told at most once a second."""
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
                self.player.audio.play(_sleep_chime(), 24000)
        self._was_dormant = dormant

    def _chime(self):
        """Acknowledge the wake word when no request came with it."""
        if self.speak_replies:
            self.player.audio.play(_wake_chime(), 24000)

    def _handle_heard(self, audio_path: str, dormant: bool):
        """Answer an utterance, or while dormant, check it for the wake word.

        "Iris, what time is it?" wakes and is answered in one go, transcribed
        by the conversation model; a bare "Iris" chimes and waits.
        """
        if dormant:
            audio = decode_audio(audio_path, sampling_rate=16000)
            woken, remainder = self.wake_detector.heard(audio, self.wake_word)
            if not woken:
                return
            self._stay_awake()
            if self.event_bus:
                self.event_bus.emit("wake", {"word": self.wake_word})
            if len(remainder.split()) < 2:
                self._chime()
                return
        self._run_voice_turn(audio_path)
        # Unless the sleep word just put it to sleep (awake_until is zero)
        if self.wake_word and self.awake_until > 0:
            self._stay_awake()

    def _mic_closed(self) -> bool:
        """Paused, mic off, the user typing, or shutting down."""
        return (
            self.pause_event.is_set()
            or self.suspend_event.is_set()
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
                """Whether the voice is speaking the answer: heard now, or
                a moment ago (its echo dying away), or with more of it to
                come; not once the last of it has died away, though the
                speaker's stream plays silence for a while longer. Not
                while paused by the user."""
                if self.suspend_event.is_set():
                    return False
                pipeline = self.speech_pipeline
                return pipeline.more_to_say() or pipeline.player.audible(within=HOLD_SETTLE)

            def listen_on(turn=audio_path):
                whole_turn[0] = self.recorder.listen_on(
                    turn,
                    on_speech=on_speech,
                    until=voice_or_answer,
                    should_stop=self._mic_closed,
                    on_level=on_level,
                    speaking=speaking if through_voice else None,
                    hold=self.speech_pipeline.pause,
                    release=self.speech_pipeline.resume,
                    heard_until=lambda: self.speech_pipeline.player.heard_until,
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
        """Queue reply text for speech, unless voice output is switched off."""
        if not self.speak_replies:
            return
        clean_text = self.conversation.clean_markdown(text)
        if clean_text.strip():
            self.speech_pipeline.queue_speech(clean_text)

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
                self.event_bus.emit(
                    "error",
                    {
                        "message": "None of the installed models can see images. Install "
                        "one with vision, such as qwen3-vl, to talk about pictures.",
                        "kind": "no_vision",
                    },
                )
        self.chat(
            text,
            response_language=self.conversation_language,
            source="text",
            attachments=attachments,
        )
        if self.wake_word:
            self._stay_awake()

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
        # the silence waited for), and begins PRE_SPEECH before speech was
        # recognized, SPEECH_START into it
        turn_began = recognizing - _seconds_of(audio_path) + PRE_SPEECH - SPEECH_START
        text, detected_language = self.transcriber.transcribe(audio_path)
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

        # "That's all, Iris": back to sleep, nothing to answer
        if (
            self.wake_word
            and self.sleep_word
            and match_sleep_word(text, self.sleep_word, self.wake_word)
        ):
            self.awake_until = 0.0
            return

        # Determine response language: use detected if the persona's voice
        # can speak it, otherwise keep the current conversation language.
        if self.tts_generator.speaks(detected_language):
            self.conversation_language = detected_language
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
        reply cut short only as far as it was said, or shown with the voice
        off. Of a reply stopped before anything was said, nothing is kept."""
        if not cut:
            self.conversation.keep(prompt, self.conversation.reply or "", source)
            return
        if self.speak_replies:
            kept = self.speech_pipeline.said()
        else:
            kept = self.conversation.clean_markdown(answer)
        if kept.strip():
            self.conversation.keep(prompt, kept, source)

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
            # Set thinking status
            with self.status_manager.status_lock:
                self.status_manager.status_thinking = True

            # Set the speech pipeline language for this turn
            self.speech_pipeline.current_language = response_language
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

            interrupted = False
            try:
                stream = self.conversation.chat_stream(
                    prompt,
                    response_language=response_language,
                    on_tool_use=on_tool_use,
                    source=source,
                    attachments=attachments,
                    keep=False,  # kept below, once it is known what was heard
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
            return answer

        except Exception as e:
            # Clear all statuses on error
            with self.status_manager.status_lock:
                self.status_manager.status_thinking = False

            error_msg = f"The language model could not answer: {e}"
            if self.event_bus:
                self.event_bus.emit("error", {"message": error_msg})
            logger.exception("Reply failed")
            return error_msg

    def _error(self, message: str):
        """Tell the app about a problem it should show."""
        if self.event_bus:
            self.event_bus.emit("error", {"message": message})
        logger.error(message)

    def run_loop(self):
        """Listen and answer until stop_event is set."""
        devices_ok, device_msg = self.recorder.check_audio_devices()
        if not devices_ok:
            self._error(f"No microphone: {device_msg}")
            return

        try:
            while True:
                # External UI requested shutdown
                if self.stop_event.is_set():
                    break

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
                    continue

                # Turns from an external UI come first, whether or not the mic
                # is live, so a typed message never waits for the user to speak
                turn = self._take_turn()
                if turn is not None:
                    try:
                        if "audio" in turn:
                            self._handle_heard(turn["audio"], self.dormant)
                        else:
                            self._run_text_turn(turn)
                    except Exception:
                        logger.exception("Turn failed")
                    continue

                # A voice is being previewed: the microphone would hear it as
                # the user, and answer it. It stays closed until the room is
                # quiet again.
                if self.speech_pipeline.is_active:
                    with self.status_manager.status_lock:
                        self.status_manager.status_listening = False
                    while self.speech_pipeline.is_active and not self.stop_event.is_set():
                        time.sleep(0.05)
                    time.sleep(PREVIEW_SETTLE)
                    continue

                # Paused, or the user is typing: keep the microphone closed
                # (models stay loaded) and keep watching for typed turns
                if self.pause_event.is_set() or self.hold_event.is_set():
                    self.recorder.close_microphone()
                    with self.status_manager.status_lock:
                        self.status_manager.status_listening = False
                    time.sleep(0.05)
                    continue

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

                    # Check if we had a device error
                    if self.recorder._device_error:
                        self.speech_pipeline.stop()
                        self._error(
                            "The microphone was disconnected. Connect one and start Iris again."
                        )
                        break

                    if not audio_path:
                        continue

                    # 2. Transcribe, then 3. answer (streams and speaks the
                    # reply), listening on in case the user was not finished
                    self._answer_listening(audio_path, dormant)

                except Exception:
                    logger.exception("Listening failed")
        finally:
            self.speech_pipeline.stop()
            self.recorder.close_microphone()
            if self.tools:
                self.tools.cleanup()
            if self.memory:
                self.memory.stop()

"""Voice assistant coordinator that orchestrates all components."""

import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from faster_whisper.audio import decode_audio
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from iris.audio import AudioPlayer, AudioRecorder, get_filter
from iris.events import EventBus
from iris.llm import ConversationManager, effective_host
from iris.memory import DEFAULT_PATH, MemoryKeeper, MemoryStore, chat_json
from iris.personas import Persona, get_persona
from iris.tools import Tools
from iris.transcription import WhisperTranscriber
from iris.tts import KokoroGenerator, SpeechPipeline, is_language_supported
from iris.ui import DisplayManager, StatusManager
from iris.wakeword import WakeWordDetector, match_sleep_word

warnings.filterwarnings("ignore")

# With a wake word set: how long the assistant keeps listening after a turn
# (or after the user last spoke) before it goes back to sleep
DEFAULT_WAKE_WINDOW = 30.0
# Said while awake, this puts the assistant straight back to sleep
DEFAULT_SLEEP_WORD = "That's all"
# While dormant an utterance ends after this much silence, so the wake word
# is recognized promptly instead of after the conversational pause
DORMANT_SILENCE = 0.8


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


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


class VoiceAssistant:
    """Voice assistant with speech-to-text, LLM, and text-to-speech."""

    def __init__(
        self,
        whisper_model: str = "auto",
        whisper_device: str = "auto",
        whisper_engine: str = "auto",
        llm_model: str = "qwen3:1.7b",
        sample_rate: int = 16000,
        silence_threshold: float = 0.015,
        silence_duration: float = 3.5,
        tts_voice: str = "af_bella",
        tts_speed: float = 1.25,
        quiet: bool = True,
        persona: Optional[Persona] = None,
        enable_vision: bool = False,
        vision_model: Optional[str] = "qwen3-vl:8b",
        enable_screen_capture: bool = False,
        enable_web_search: bool = False,
        event_bus: Optional[EventBus] = None,
        turn_queue: Optional[queue.Queue] = None,
        speak_replies: bool = True,
        ollama_host: Optional[str] = None,
        memory: bool = True,
        wake_word: Optional[str] = None,
        wake_window: float = DEFAULT_WAKE_WINDOW,
        sleep_word: Optional[str] = DEFAULT_SLEEP_WORD,
        tts_models_dir: Optional[str] = None,
    ):
        """
        Initialize the voice assistant.

        Args:
            whisper_model: Whisper model size, or "auto" for the best one the
                hardware decodes quickly (large-v3-turbo on a GPU, small on CPU)
            whisper_device: Device for faster-whisper (cpu, cuda, or auto)
            whisper_engine: "auto" (MLX on Apple Silicon, else faster-whisper),
                "mlx" or "faster-whisper"
            llm_model: Ollama model name (primary model, should be fast)
            sample_rate: Audio sample rate for recording
            silence_threshold: RMS threshold for silence detection
            silence_duration: Duration of silence to stop recording
            tts_voice: Voice for TTS (overridden by persona.voice if persona is provided)
            tts_speed: Speed for TTS
            quiet: If True, suppress verbose output during initialization
            persona: Persona to use for the assistant
            enable_vision: Enable camera/vision tool
            vision_model: Vision model to use when vision is needed (default: qwen3-vl:8b)
            enable_screen_capture: Enable screen capture tool
            enable_web_search: Enable web search tool
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
        """
        # Initialize console
        self.console = Console()

        # Event bus for external UIs (None in pure CLI mode)
        self.event_bus = event_bus

        # Cooperative control signals for external UIs
        self.interrupt_event = threading.Event()  # barge-in: cancel current turn
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

        # Store persona and use its voice
        self.persona = persona if persona else get_persona("iris")
        if persona:
            tts_voice = persona.voice

        if not quiet:
            self.console.print("[dim]Initializing components...[/dim]")

        # Initialize components
        self.recorder = AudioRecorder(sample_rate, silence_threshold, silence_duration)

        # Initialize audio player with filter from persona
        audio_filter = get_filter(self.persona.audio_filter) if self.persona.audio_filter else None
        self.player = AudioPlayer(audio_filter=audio_filter)

        self.transcriber = WhisperTranscriber(
            whisper_model, whisper_device, quiet=quiet, engine=whisper_engine
        )

        # Initialize TTS with persona's voice and speed
        self.tts_generator = KokoroGenerator(
            tts_voice, self.persona.speed, quiet=quiet, models_dir=tts_models_dir
        )
        self.speech_pipeline = SpeechPipeline(self.tts_generator, self.player)

        if self.event_bus:
            self.speech_pipeline.on_output_level = lambda level: self.event_bus.emit(
                "audio.output", {"rms": level}
            )

        # Initialize UI
        self.status_manager = StatusManager(self.console)
        self.status_manager.event_bus = self.event_bus
        self.display_manager = DisplayManager(self.console, self.status_manager)

        # Setup tools (after status_manager is created)
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

        self.tools = None
        if enable_vision or enable_screen_capture or enable_web_search or self.memory:
            self.tools = Tools(console=self.console, status_manager=self.status_manager)
            if self.memory:
                self.tools.enable_memory(self.memory.store)

            if enable_vision:
                if self.tools.enable_camera():
                    if not quiet:
                        self.console.print("[dim]✓ Vision enabled[/dim]")
                else:
                    self.console.print(
                        "[yellow]⚠ Vision dependencies not installed (opencv-python, pillow)[/yellow]"
                    )

            if enable_screen_capture:
                if self.tools.enable_screen_capture():
                    if not quiet:
                        self.console.print("[dim]✓ Screen capture enabled[/dim]")
                        self.console.print(
                            f"[dim]  Available tools: {list(self.tools._enabled_tools.keys())}[/dim]"
                        )
                else:
                    self.console.print(
                        "[yellow]⚠ Screen capture dependencies not installed (mss)[/yellow]"
                    )

            if enable_web_search:
                if self.tools.enable_web_search():
                    if not quiet:
                        self.console.print("[dim]✓ Web search enabled[/dim]")
                else:
                    self.console.print(
                        "[yellow]⚠ Web search dependencies not installed (requests)[/yellow]"
                    )

        self.set_wake_word(wake_word, engine=whisper_engine)

        # Initialize conversation manager with tools
        # If vision OR screen capture is enabled, pass the vision model for automatic switching
        self.conversation = ConversationManager(
            llm_model,
            persona=self.persona,
            tools=self.tools,
            vision_model=vision_model if (enable_vision or enable_screen_capture) else None,
            quiet=quiet,
            host=ollama_host,
            memory=self.memory,
        )
        if self.memory:
            self.memory.start()

    def config_info(self) -> dict:
        """Describe the loaded models and audio input for external UIs."""
        return {
            "persona": self.persona.name,
            "llm_model": self.conversation.model,
            "whisper_model": self.transcriber.model_name,
            "whisper_device": self.transcriber.device,
            "whisper_engine": self.transcriber.engine,
            "stt_model_path": str(self.transcriber.model_path or ""),
            "tts_models_dir": str(self.tts_generator.models_dir or ""),
            "tts_voice": self.tts_generator.voice,
            "input_device": self.recorder.current_device_name(),
            "speak_replies": self.speak_replies,
            "ollama_host": self.conversation.host or "",
            "ollama_url": effective_host(self.conversation.host),
            "memory": self.memory is not None,
            "amnesia": self.conversation.amnesia,
            "wake_word": self.wake_word or "",
            "wake_window": self.wake_window,
            "sleep_word": self.sleep_word or "",
            "mic_on": not self.pause_event.is_set(),
            "paused": self.suspend_event.is_set(),
            "tools": self.tools.group_states() if self.tools else {},
        }

    def set_wake_word(self, word: Optional[str], engine: str = "auto"):
        """Listen only for this word from now on (None or "" listens to all).

        The tiny detector model is loaded the first time it is needed.
        """
        word = (word or "").strip() or None
        if word and self.wake_detector is None:
            engine = self.transcriber.engine if engine == "auto" else engine
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
                sd.play(_sleep_chime(), 24000)
        self._was_dormant = dormant

    def _chime(self):
        """Acknowledge the wake word when no request came with it."""
        if self.speak_replies:
            sd.play(_wake_chime(), 24000)

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
        self.display_manager.show_user_message(text)
        if self.event_bus:
            self.event_bus.emit(
                "chat.user",
                {
                    "text": text,
                    "language": self.conversation_language,
                    "source": "text",
                    "id": turn.get("id"),
                },
            )
        self.chat(text, response_language=self.conversation_language, source="text")
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
        self.status_manager.update_status_bar(self.speech_pipeline)

        text, detected_language = self.transcriber.transcribe(audio_path)

        with self.status_manager.status_lock:
            self.status_manager.status_understanding = False
        self.status_manager.update_status_bar(self.speech_pipeline)

        if not text:
            return

        # "That's all, Iris": back to sleep, nothing to answer
        if (
            self.wake_word
            and self.sleep_word
            and match_sleep_word(text, self.sleep_word, self.wake_word)
        ):
            self.awake_until = 0.0
            return

        # Determine response language: use detected if TTS supports it,
        # otherwise keep the current conversation language.
        if is_language_supported(detected_language):
            self.conversation_language = detected_language
        response_language = self.conversation_language

        # Display user message
        self.display_manager.show_user_message(text)
        if self.event_bus:
            self.event_bus.emit(
                "chat.user",
                {"text": text, "language": response_language, "source": "voice"},
            )

        # Send to LLM (streams response and speaks it)
        self.chat(text, response_language=response_language, source="voice")

    def chat(self, prompt: str, response_language: str = "en", source: Optional[str] = None) -> str:
        """
        Send prompt to LLM, stream response, and speak it.

        Args:
            prompt: User prompt text
            response_language: Language to respond in (Whisper code)

        Returns:
            LLM response text
        """
        try:
            # Set thinking status
            with self.status_manager.status_lock:
                self.status_manager.status_thinking = True
            self.status_manager.update_status_bar(self.speech_pipeline)

            # Set the speech pipeline language for this turn
            self.speech_pipeline.current_language = response_language

            # Stream the response and display it live
            full_response = []
            text_buffer = ""  # Buffer for speech chunks

            def on_tool_use(tool_name, tool_args):
                detail = tool_args.get("query") or tool_args.get("url", "")
                if tool_name == "recall" and self.memory:
                    topic = self.memory.store.topic(_as_int(tool_args.get("topic_id")))
                    detail = topic.title if topic else ""
                self.display_manager.show_tool_activity(tool_name, detail)
                if self.event_bus:
                    self.event_bus.emit("tool.use", {"tool": tool_name, "detail": detail})

            interrupted = False
            try:
                stream = self.conversation.chat_stream(
                    prompt,
                    response_language=response_language,
                    on_tool_use=on_tool_use,
                    source=source,
                )
                for chunk_text in stream:
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

                    # Build current response text
                    current_text = "".join(full_response)

                    # Update the display to show streaming message
                    if current_text.strip():
                        self.display_manager.update_streaming_display(
                            current_text, self.speech_pipeline
                        )

                    # Check if we should speak accumulated text
                    should_speak = False
                    text_to_speak = ""

                    # Use consistent chunk sizes for smooth speech flow
                    min_length = 180
                    search_length = 280
                    min_sentence_pos = 90

                    # First check for paragraph break
                    if "\n\n" in text_buffer:
                        parts = text_buffer.split("\n\n", 1)
                        text_to_speak = parts[0].strip()
                        text_buffer = parts[1] if len(parts) > 1 else ""
                        should_speak = True

                    # Otherwise check if we have enough text with sentence ending
                    elif len(text_buffer) > min_length:
                        search_text = text_buffer[:search_length]
                        last_period = max(
                            search_text.rfind(". "),
                            search_text.rfind("! "),
                            search_text.rfind("? "),
                        )

                        if last_period > min_sentence_pos:
                            text_to_speak = text_buffer[: last_period + 1].strip()
                            text_buffer = text_buffer[last_period + 2 :]
                            should_speak = True

                    # Speak the chunk (queue it for sequential playback)
                    if should_speak and text_to_speak:
                        self._speak(text_to_speak)

                # Clear thinking status now that LLM is done
                with self.status_manager.status_lock:
                    self.status_manager.status_thinking = False

                # Collect final answer
                answer = "".join(full_response)

                if interrupted:
                    # Stop generating (closing the stream ends the request, so
                    # the model is free at once) and drop all pending speech
                    stream.close()
                    self.speech_pipeline.stop()
                    text_buffer = ""

                if self.event_bus:
                    self.event_bus.emit("chat.done", {"text": answer, "interrupted": interrupted})

                # Final update: show ONLY the message (without status bar)
                self.display_manager.finalize_message(answer, self.speech_pipeline)

            finally:
                # Clear thinking status if still set
                with self.status_manager.status_lock:
                    self.status_manager.status_thinking = False

                # Make sure status bar is running
                if not self.status_manager.status_bar_active:
                    self.status_manager.start(self.speech_pipeline)

            # Speak any remaining text that wasn't spoken yet, unless the
            # reply was stopped in the meantime
            if text_buffer.strip() and not self.interrupt_event.is_set():
                self._speak(text_buffer)

            # Wait for all speech to complete
            self.speech_pipeline.wait_for_speech()

            return answer

        except Exception as e:
            # Clear all statuses on error
            with self.status_manager.status_lock:
                self.status_manager.status_thinking = False

            # Make sure status bar is running
            if not self.status_manager.status_bar_active:
                self.status_manager.start(self.speech_pipeline)

            error_msg = f"Error communicating with Ollama: {e}"
            if self.event_bus:
                self.event_bus.emit("error", {"message": error_msg})
            self.console.print(f"[red]Error:[/red] {error_msg}")
            import traceback

            traceback.print_exc()
            return error_msg

    def run_loop(self):
        """Run the main voice assistant loop."""
        self.console.clear()

        # Check if audio devices are available before starting
        from iris.audio import AudioRecorder

        devices_ok, device_msg = AudioRecorder.check_audio_devices()
        if not devices_ok:
            self.console.print()
            self.console.print("[bold red]✗ Microphone Error[/bold red]")
            self.console.print(f"\n{device_msg}")
            self.console.print()
            self.console.print("[dim]Possible solutions:[/dim]")
            self.console.print("  • Connect a microphone or headset")
            self.console.print("  • Check system audio input settings")
            self.console.print("  • Restart the application after connecting a microphone")
            self.console.print()
            return

        # App name display with persona
        name_text = Text("\niris\n", style="bold bright_white", justify="center")
        info = Text()
        info.append(f"◈ {self.persona.name}", style="dim magenta")
        info.append("  •  ", style="dim")
        info.append(f"{self.conversation.model}", style="dim")
        info.append("  •  ", style="dim")
        info.append("Ctrl+C to exit", style="dim cyan")

        display = Group(
            Align.center(name_text),
            Align.center(info),
            Align.center(Text("")),
        )

        self.console.print(display)
        self.console.print()

        # Initialize status bar at the bottom
        self.status_manager.start(self.speech_pipeline)

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
                    except Exception as e:
                        self.console.print(f"[red]Error:[/red] Error in turn: {e}")
                    continue

                # Paused, or the user is typing: keep the microphone closed
                # (models stay loaded) and keep watching for typed turns
                if self.pause_event.is_set() or self.hold_event.is_set():
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
                    self.status_manager.update_status_bar(self.speech_pipeline)

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
                        self.display_manager.update_audio_level_display(level, self.speech_pipeline)
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
                        # The awake window ran out with nobody speaking
                        or (not dormant and not heard_speech[0] and self.dormant),
                        silence_duration=DORMANT_SILENCE if dormant else None,
                    )

                    with self.status_manager.status_lock:
                        self.status_manager.status_listening = False
                        self.status_manager.status_dormant = False
                    self.status_manager.update_status_bar(self.speech_pipeline)

                    # Check if we had a device error
                    if self.recorder._device_error:
                        self.speech_pipeline.stop()
                        self.status_manager.stop()
                        self.console.print()
                        self.console.print(
                            "[bold red]✗ Microphone disconnected or unavailable[/bold red]"
                        )
                        self.console.print()
                        self.console.print(
                            "[dim]Please connect a microphone and restart the application.[/dim]"
                        )
                        self.console.print()
                        break

                    if not audio_path:
                        continue

                    # 2. Transcribe, then 3. answer (streams and speaks the reply)
                    self._handle_heard(audio_path, dormant)

                    # Cleanup
                    Path(audio_path).unlink(missing_ok=True)

                except KeyboardInterrupt:
                    self.speech_pipeline.stop()
                    self.status_manager.stop()
                    self.console.print("\n[bold cyan]Goodbye![/bold cyan]")
                    break
                except Exception as e:
                    self.console.print(f"[red]Error:[/red] Error in loop: {e}")
                    import traceback

                    traceback.print_exc()
        finally:
            self.speech_pipeline.stop()
            self.status_manager.stop()
            if self.tools:
                self.tools.cleanup()
            if self.memory:
                self.memory.stop()

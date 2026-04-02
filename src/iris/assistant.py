"""Voice assistant coordinator that orchestrates all components."""

import sys
import threading
import warnings
from pathlib import Path
from typing import Optional

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from iris.audio import AudioPlayer, AudioRecorder, get_filter
from iris.llm import ConversationManager
from iris.personas import Persona, get_persona
from iris.transcription import WhisperTranscriber
from iris.tts import KokoroGenerator, SpeechPipeline, is_language_supported
from iris.ui import DisplayManager, StatusManager

warnings.filterwarnings("ignore")


class VoiceAssistant:
    """Voice assistant with speech-to-text, LLM, and text-to-speech."""

    def __init__(
        self,
        whisper_model: str = "small",
        whisper_device: str = "auto",
        llm_model: str = "qwen3:1.7b",
        sample_rate: int = 16000,
        silence_threshold: float = 0.015,
        silence_duration: float = 3.5,
        tts_voice: str = "af_bella",
        tts_speed: float = 1.25,
        quiet: bool = True,
        persona: Optional[Persona] = None,
        enable_vision: bool = False,
        vision_model: str = "qwen3-vl:8b",
        enable_screen_capture: bool = False,
        enable_web_search: bool = False,
    ):
        """
        Initialize the voice assistant.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            whisper_device: Device for Whisper (cpu, cuda, or auto)
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
        """
        # Initialize console
        self.console = Console()

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

        self.transcriber = WhisperTranscriber(whisper_model, whisper_device, quiet=quiet)

        # Initialize TTS with persona's voice and speed
        self.tts_generator = KokoroGenerator(tts_voice, self.persona.speed, quiet=quiet)
        self.speech_pipeline = SpeechPipeline(self.tts_generator, self.player)

        # Initialize UI
        self.status_manager = StatusManager(self.console)
        self.display_manager = DisplayManager(self.console, self.status_manager)

        # Setup tools (after status_manager is created)
        self.tools = None
        if enable_vision or enable_screen_capture or enable_web_search:
            from iris.tools import Tools

            self.tools = Tools(console=self.console, status_manager=self.status_manager)

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

        # Initialize conversation manager with tools
        # If vision OR screen capture is enabled, pass the vision model for automatic switching
        self.conversation = ConversationManager(
            llm_model,
            persona=self.persona,
            tools=self.tools,
            vision_model=vision_model if (enable_vision or enable_screen_capture) else None,
            quiet=quiet,
        )

    def chat(self, prompt: str, response_language: str = "en") -> str:
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
                self.display_manager.show_tool_activity(tool_name, detail)

            try:
                for chunk_text in self.conversation.chat_stream(prompt, response_language=response_language, on_tool_use=on_tool_use):
                    full_response.append(chunk_text)
                    text_buffer += chunk_text

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
                        clean_text = self.conversation.clean_markdown(text_to_speak)
                        if clean_text.strip():
                            self.speech_pipeline.queue_speech(clean_text)

                # Clear thinking status now that LLM is done
                with self.status_manager.status_lock:
                    self.status_manager.status_thinking = False

                # Collect final answer
                answer = "".join(full_response)

                # Final update: show ONLY the message (without status bar)
                self.display_manager.finalize_message(answer, self.speech_pipeline)

            finally:
                # Clear thinking status if still set
                with self.status_manager.status_lock:
                    self.status_manager.status_thinking = False

                # Make sure status bar is running
                if not self.status_manager.status_bar_active:
                    self.status_manager.start(self.speech_pipeline)

            # Speak any remaining text that wasn't spoken yet
            if text_buffer.strip():
                clean_text = self.conversation.clean_markdown(text_buffer)
                if clean_text.strip():
                    self.speech_pipeline.queue_speech(clean_text)

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
                try:
                    # 1. Listen for voice input with level visualization
                    with self.status_manager.status_lock:
                        self.status_manager.status_listening = True
                    self.status_manager.update_status_bar(self.speech_pipeline)

                    def on_audio_level(level: float):
                        """Update audio level display."""
                        self.display_manager.update_audio_level_display(level, self.speech_pipeline)

                    audio_path = self.recorder.listen(on_level=on_audio_level)

                    with self.status_manager.status_lock:
                        self.status_manager.status_listening = False
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

                    # 2. Transcribe with Whisper
                    with self.status_manager.status_lock:
                        self.status_manager.status_understanding = True
                    self.status_manager.update_status_bar(self.speech_pipeline)

                    text, detected_language = self.transcriber.transcribe(audio_path)

                    with self.status_manager.status_lock:
                        self.status_manager.status_understanding = False
                    self.status_manager.update_status_bar(self.speech_pipeline)

                    if not text:
                        continue

                    # Determine response language: use detected if TTS supports it,
                    # otherwise keep the current conversation language.
                    if is_language_supported(detected_language):
                        self.conversation_language = detected_language
                    response_language = self.conversation_language

                    # Display user message
                    self.display_manager.show_user_message(text)

                    # 3. Send to LLM (streams response and speaks it)
                    self.chat(text, response_language=response_language)

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

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

from iris.audio import AudioPlayer, AudioRecorder
from iris.llm import ConversationManager
from iris.personas import Persona, get_persona
from iris.transcription import WhisperTranscriber
from iris.tts import KokoroGenerator, SpeechPipeline
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
        silence_threshold: float = 0.02,
        silence_duration: float = 2.5,
        tts_voice: str = "af_bella",
        tts_speed: float = 1.25,
        quiet: bool = True,
        persona: Optional[Persona] = None,
    ):
        """
        Initialize the voice assistant.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            whisper_device: Device for Whisper (cpu, cuda, or auto)
            llm_model: Ollama model name
            sample_rate: Audio sample rate for recording
            silence_threshold: RMS threshold for silence detection
            silence_duration: Duration of silence to stop recording
            tts_voice: Voice for TTS (overridden by persona.voice if persona is provided)
            tts_speed: Speed for TTS
            quiet: If True, suppress verbose output during initialization
            persona: Persona to use for the assistant
        """
        # Initialize console
        self.console = Console()
        
        # Store persona and use its voice
        self.persona = persona if persona else get_persona("iris")
        if persona:
            tts_voice = persona.voice

        if not quiet:
            self.console.print("[dim]Initializing components...[/dim]")

        # Initialize components
        self.recorder = AudioRecorder(sample_rate, silence_threshold, silence_duration)
        self.player = AudioPlayer()
        self.transcriber = WhisperTranscriber(whisper_model, whisper_device, quiet=quiet)
        self.conversation = ConversationManager(llm_model, persona=self.persona, quiet=quiet)

        # Initialize TTS
        self.tts_generator = KokoroGenerator(tts_voice, tts_speed, quiet=quiet)
        self.speech_pipeline = SpeechPipeline(self.tts_generator, self.player)

        # Initialize UI
        self.status_manager = StatusManager(self.console)
        self.display_manager = DisplayManager(self.console, self.status_manager)

    def chat(self, prompt: str) -> str:
        """
        Send prompt to LLM, stream response, and speak it.

        Args:
            prompt: User prompt text

        Returns:
            LLM response text
        """
        try:
            # Set thinking status
            with self.status_manager.status_lock:
                self.status_manager.status_thinking = True
            self.status_manager.update_status_bar(self.speech_pipeline)

            # Stream the response and display it live
            full_response = []
            text_buffer = ""  # Buffer for speech chunks

            try:
                for chunk_text in self.conversation.chat_stream(prompt):
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

        # App name display with persona
        name_text = Text(f"\niris - {self.persona.name}\n", style="bold bright_white", justify="center")
        info = Text()
        info.append(f"◈ {self.conversation.model}", style="dim")
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
                        self.display_manager.update_audio_level_display(
                            level, self.speech_pipeline
                        )

                    audio_path = self.recorder.listen(on_level=on_audio_level)

                    with self.status_manager.status_lock:
                        self.status_manager.status_listening = False
                    self.status_manager.update_status_bar(self.speech_pipeline)

                    if not audio_path:
                        continue

                    # 2. Transcribe with Whisper
                    with self.status_manager.status_lock:
                        self.status_manager.status_understanding = True
                    self.status_manager.update_status_bar(self.speech_pipeline)

                    text = self.transcriber.transcribe(audio_path)

                    with self.status_manager.status_lock:
                        self.status_manager.status_understanding = False
                    self.status_manager.update_status_bar(self.speech_pipeline)

                    if not text:
                        continue

                    # Display user message
                    self.display_manager.show_user_message(text)

                    # 3. Send to LLM (streams response and speaks it)
                    self.chat(text)

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


#!/usr/bin/env python3
"""
Iris Voice Assistant CLI - Listen, transcribe, chat with LLM, and speak response.
Uses: faster-whisper (GPU), Ollama (qwen3:1.7b), and kokoro-tts

Supports subcommands:
  iris                  Start the voice assistant (default)
  iris voice clone      Clone a voice from a reference audio
  iris voice list       List cloned voices
  iris voice delete     Delete a cloned voice
  iris voice test       Test a cloned voice
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

from rich.console import Console
from rich.table import Table

from iris.personas import list_personas, get_persona

warnings.filterwarnings("ignore")

console = Console()

LOG_DIR = Path.home() / ".iris"
LOG_FILE = LOG_DIR / "iris.log"


def setup_logging():
    """Configure file logging to ~/.iris/iris.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_assistant_parser(parser: argparse.ArgumentParser):
    """Add voice assistant arguments to a parser."""
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen3:1.7b",
        help="Ollama model name (default: qwen3:1.7b, must support function calling for tools)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for Whisper (default: auto)",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip the setup wizard (assume all dependencies are ready)",
    )
    parser.add_argument(
        "--persona",
        default="iris",
        choices=list_personas(),
        help="Persona/character for the assistant (default: iris)",
    )
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        help="Enable camera/vision tool (automatically switches to vision model when needed)",
    )
    parser.add_argument(
        "--vision-model",
        default="qwen3-vl:8b",
        help="Vision model to use when looking (default: qwen3-vl:8b)",
    )
    parser.add_argument(
        "--enable-screen-capture",
        action="store_true",
        help="Enable screen capture tool to see your computer screen",
    )
    parser.add_argument(
        "--enable-web-search",
        action="store_true",
        help="Enable web search tool to look up current information",
    )


def _run_assistant(args):
    """Run the voice assistant."""
    try:
        # Run setup wizard to check dependencies and download models
        if not args.skip_setup:
            from iris.setup import run_setup

            if not run_setup(
                whisper_model=args.whisper_model,
                llm_model=args.llm_model,
            ):
                sys.exit(1)

        # Get the selected persona
        persona = get_persona(args.persona)

        # Import assistant after setup to avoid loading models before they're ready
        from iris.assistant import VoiceAssistant

        # Create and run assistant
        assistant = VoiceAssistant(
            whisper_model=args.whisper_model,
            whisper_device=args.device,
            llm_model=args.llm_model,
            persona=persona,
            enable_vision=args.enable_vision,
            vision_model=args.vision_model,
            enable_screen_capture=args.enable_screen_capture,
            enable_web_search=args.enable_web_search,
        )

        assistant.run_loop()
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        console.print("\n[bold cyan]Goodbye![/bold cyan]")
        sys.exit(0)


def _run_voice_clone(args):
    """Clone a voice from a reference audio file."""
    from iris.voices import VoiceStore

    store = VoiceStore()
    try:
        info = store.save_voice(
            name=args.name,
            reference_audio_path=args.reference,
            base_voice=args.base_voice,
            description=args.description or "",
        )
        console.print(f"[bold green]✓ Voice '{info.name}' cloned successfully![/bold green]")
        console.print(f"  Base voice: [cyan]{info.base_voice}[/cyan]")
        console.print(f"  Reference duration: [cyan]{info.reference_duration:.1f}s[/cyan]")
        console.print()
        console.print("Use it in a persona YAML:")
        console.print(f'  [cyan]voice: "{info.name}"[/cyan]')
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def _run_voice_list(_args):
    """List all cloned voices."""
    from iris.voices import VoiceStore

    store = VoiceStore()
    voices = store.list_voices()

    if not voices:
        console.print("[dim]No cloned voices found.[/dim]")
        console.print("Clone a voice with: [cyan]iris voice clone --name <name> --reference <audio.wav>[/cyan]")
        return

    table = Table(title="Cloned Voices")
    table.add_column("Name", style="bold cyan")
    table.add_column("Base Voice")
    table.add_column("Duration")
    table.add_column("Created")
    table.add_column("Description", style="dim")

    for voice in voices:
        table.add_row(
            voice.name,
            voice.base_voice,
            f"{voice.reference_duration:.1f}s",
            voice.created[:10] if voice.created else "",
            voice.description or "",
        )

    console.print(table)


def _run_voice_delete(args):
    """Delete a cloned voice."""
    from iris.voices import VoiceStore

    store = VoiceStore()
    if store.delete_voice(args.name):
        console.print(f"[bold green]✓ Voice '{args.name}' deleted.[/bold green]")
    else:
        console.print(f"[bold red]Voice '{args.name}' not found.[/bold red]")
        sys.exit(1)


def _run_voice_test(args):
    """Test a cloned voice by generating and playing a sample."""
    from iris.voices import VoiceStore

    store = VoiceStore()
    base_voice, reference_audio = store.resolve_voice(args.name)

    if reference_audio is None:
        console.print(f"[bold red]Voice '{args.name}' not found in voice store.[/bold red]")
        sys.exit(1)

    console.print(f"Testing voice [bold cyan]{args.name}[/bold cyan]...")
    console.print(f"  Base voice: {base_voice}")
    console.print(f"  Text: \"{args.text}\"")
    console.print()

    try:
        from iris.tts import KokoroGenerator
        from iris.audio import AudioPlayer

        generator = KokoroGenerator(
            voice=base_voice,
            reference_audio=reference_audio,
        )
        audio_file = generator.generate(args.text)

        if audio_file:
            console.print("[dim]Playing...[/dim]")
            player = AudioPlayer()
            player.play(audio_file)
            from pathlib import Path

            Path(audio_file).unlink(missing_ok=True)
            console.print("[bold green]✓ Done![/bold green]")
        else:
            console.print("[bold red]Failed to generate audio.[/bold red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Iris Voice Assistant - Listen, transcribe, chat, speak",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Voice subcommand group
    voice_parser = subparsers.add_parser("voice", help="Manage cloned voices")
    voice_subparsers = voice_parser.add_subparsers(dest="voice_command")

    # iris voice clone
    clone_parser = voice_subparsers.add_parser("clone", help="Clone a voice from reference audio")
    clone_parser.add_argument("--name", required=True, help="Name for the cloned voice")
    clone_parser.add_argument(
        "--reference", required=True, help="Path to reference audio file (WAV, 3-30 seconds)"
    )
    clone_parser.add_argument(
        "--base-voice", required=True, help="Kokoro base voice for synthesis (e.g. af_bella, am_michael, bm_lewis)"
    )
    clone_parser.add_argument("--description", default="", help="Optional description of the voice")

    # iris voice list
    voice_subparsers.add_parser("list", help="List all cloned voices")

    # iris voice delete
    delete_parser = voice_subparsers.add_parser("delete", help="Delete a cloned voice")
    delete_parser.add_argument("name", help="Name of the voice to delete")

    # iris voice test
    test_parser = voice_subparsers.add_parser("test", help="Test a cloned voice")
    test_parser.add_argument("name", help="Name of the voice to test")
    test_parser.add_argument(
        "--text", default="Hello, this is a test of my cloned voice.", help="Text to speak"
    )

    # Add assistant arguments to the top-level parser (for backward compatibility)
    _build_assistant_parser(parser)

    args = parser.parse_args()

    setup_logging()

    if args.command == "voice":
        if args.voice_command == "clone":
            _run_voice_clone(args)
        elif args.voice_command == "list":
            _run_voice_list(args)
        elif args.voice_command == "delete":
            _run_voice_delete(args)
        elif args.voice_command == "test":
            _run_voice_test(args)
        else:
            voice_parser.print_help()
    else:
        # No subcommand or unknown — run the assistant (backward compatible)
        _run_assistant(args)


if __name__ == "__main__":
    main()

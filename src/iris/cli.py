#!/usr/bin/env python3
"""
Iris Voice Assistant CLI - Listen, transcribe, chat with LLM, and speak response.
Uses: faster-whisper (GPU), Ollama (qwen3:1.7b), and kokoro-tts
"""

import argparse
import sys
import warnings

from rich.console import Console

from iris.personas import list_personas, get_persona

warnings.filterwarnings("ignore")

console = Console()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Iris Voice Assistant - Listen, transcribe, chat, speak"
    )
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
        help="Enable web search tool for current information",
    )

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()


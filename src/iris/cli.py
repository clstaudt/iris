#!/usr/bin/env python3
"""
Iris Voice Assistant CLI - Listen, transcribe, chat with LLM, and speak response.
Uses: faster-whisper (GPU), Ollama (qwen3:8b), and kokoro-tts
"""

import argparse
import sys
import warnings

from rich.console import Console

from iris.assistant import VoiceAssistant

warnings.filterwarnings("ignore")

console = Console()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Iris Voice Assistant - Listen, transcribe, chat, speak"
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--llm-model", default="qwen3:8b", help="Ollama model name (default: qwen3:8b)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for Whisper (default: auto)",
    )

    args = parser.parse_args()

    try:
        # Create and run assistant
        assistant = VoiceAssistant(
            whisper_model=args.whisper_model,
            whisper_device=args.device,
            llm_model=args.llm_model,
        )

        assistant.run_loop()
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        console.print("\n[bold cyan]Goodbye![/bold cyan]")
        sys.exit(0)


if __name__ == "__main__":
    main()


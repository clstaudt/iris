"""Setup wizard for first-run experience."""

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text

# Kokoro model download URLs
KOKORO_MODEL_URL = (
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"
)

# Default paths
WHISPER_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
KOKORO_MODELS_DIR = Path.home() / ".iris" / "models"


class SetupWizard:
    """Handles first-run setup and dependency checking."""

    def __init__(self, whisper_model: str = "small", llm_model: str = "qwen3:1.7b"):
        """
        Initialize setup wizard.

        Args:
            whisper_model: Whisper model size to use
            llm_model: Ollama model name to use
        """
        self.console = Console()
        self.whisper_model = whisper_model
        self.llm_model = llm_model

    def run(self) -> bool:
        """
        Run the setup wizard.

        Checks all dependencies and prompts user to download missing components.

        Returns:
            True if setup completed successfully, False if user cancelled
        """
        self.console.print()
        self.console.print(
            Panel(
                "[bold]Iris Setup[/bold]\n\n"
                "Checking dependencies and downloading required models...",
                border_style="cyan",
            )
        )
        self.console.print()

        # Check all dependencies
        checks = self._run_checks()

        # Display status table
        self._display_status(checks)

        # If everything is ready, we're done
        if all(status == "ready" for status in checks.values()):
            self.console.print()
            self.console.print("[bold green]✓ All dependencies ready![/bold green]")
            self.console.print()
            return True

        # Collect what needs to be downloaded
        downloads_needed = []
        if checks["ollama"] == "not_running":
            self.console.print()
            self.console.print(
                "[bold red]✗ Ollama is not running[/bold red]\n"
                "  Please start Ollama first:\n"
                "  [cyan]brew services start ollama[/cyan]  (macOS)\n"
                "  [cyan]ollama serve[/cyan]               (Linux)\n"
            )
            return False

        if checks["ollama_model"] == "missing":
            downloads_needed.append(("ollama_model", f"Ollama model ({self.llm_model})"))

        if checks["whisper"] == "missing":
            downloads_needed.append(("whisper", f"Whisper model ({self.whisper_model})"))

        if checks["kokoro_cli"] == "missing":
            self.console.print()
            self.console.print(
                "[bold red]✗ kokoro-tts CLI not found[/bold red]\n"
                "  Please install it first:\n"
                "  [cyan]pip install kokoro-tts[/cyan]\n"
            )
            return False

        if checks["kokoro_models"] == "missing":
            downloads_needed.append(("kokoro_models", "Kokoro TTS models (~360 MB)"))

        # If nothing to download, we're ready
        if not downloads_needed:
            self.console.print()
            self.console.print("[bold green]✓ All dependencies ready![/bold green]")
            self.console.print()
            return True

        # Ask user to confirm downloads
        self.console.print()
        self.console.print("[bold yellow]The following components need to be downloaded:[/bold yellow]")
        for _, desc in downloads_needed:
            self.console.print(f"  • {desc}")

        # Show where files will be stored
        self.console.print()
        self.console.print("[dim]Download locations:[/dim]")
        if checks["ollama_model"] == "missing":
            self.console.print(f"  [dim]• Ollama models: ~/.ollama/models/[/dim]")
        if checks["whisper"] == "missing":
            self.console.print(f"  [dim]• Whisper model: {WHISPER_CACHE_DIR}/[/dim]")
        if checks["kokoro_models"] == "missing":
            self.console.print(f"  [dim]• Kokoro models: {KOKORO_MODELS_DIR}/[/dim]")

        self.console.print()
        try:
            response = self.console.input(
                "[bold]Download now? [Y/n]: [/bold]"
            ).strip().lower()
            if response not in ("", "y", "yes"):
                self.console.print("[dim]Setup cancelled.[/dim]")
                return False
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[dim]Setup cancelled.[/dim]")
            return False

        # Perform downloads
        self.console.print()
        for download_id, desc in downloads_needed:
            if download_id == "ollama_model":
                if not self._download_ollama_model():
                    return False
            elif download_id == "whisper":
                if not self._download_whisper_model():
                    return False
            elif download_id == "kokoro_models":
                if not self._download_kokoro_models():
                    return False

        self.console.print()
        self.console.print("[bold green]✓ Setup complete![/bold green]")
        self.console.print()
        return True

    def _run_checks(self) -> dict[str, str]:
        """
        Run all dependency checks.

        Returns:
            Dict mapping component name to status (ready/missing/not_running)
        """
        return {
            "ollama": self._check_ollama_running(),
            "ollama_model": self._check_ollama_model(),
            "whisper": self._check_whisper_model(),
            "kokoro_cli": self._check_kokoro_cli(),
            "kokoro_models": self._check_kokoro_models(),
        }

    def _display_status(self, checks: dict[str, str]):
        """Display a status table of all checks."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Status", width=3)
        table.add_column("Component")
        table.add_column("Details", style="dim")

        # Ollama service
        if checks["ollama"] == "ready":
            table.add_row("[green]✓[/green]", "Ollama", "running")
        else:
            table.add_row("[red]✗[/red]", "Ollama", "not running")

        # Ollama model
        if checks["ollama_model"] == "ready":
            table.add_row("[green]✓[/green]", f"LLM ({self.llm_model})", "available")
        elif checks["ollama"] != "ready":
            table.add_row("[dim]?[/dim]", f"LLM ({self.llm_model})", "unknown")
        else:
            table.add_row("[yellow]↓[/yellow]", f"LLM ({self.llm_model})", "needs download")

        # Whisper model
        if checks["whisper"] == "ready":
            table.add_row("[green]✓[/green]", f"Whisper ({self.whisper_model})", "cached")
        else:
            table.add_row("[yellow]↓[/yellow]", f"Whisper ({self.whisper_model})", "needs download")

        # Kokoro CLI
        if checks["kokoro_cli"] == "ready":
            table.add_row("[green]✓[/green]", "kokoro-tts CLI", "installed")
        else:
            table.add_row("[red]✗[/red]", "kokoro-tts CLI", "not found")

        # Kokoro models
        if checks["kokoro_models"] == "ready":
            table.add_row("[green]✓[/green]", "Kokoro TTS models", "cached")
        else:
            table.add_row("[yellow]↓[/yellow]", "Kokoro TTS models", "needs download")

        self.console.print(table)

    def _check_ollama_running(self) -> str:
        """Check if Ollama service is running."""
        try:
            import ollama
            ollama.list()
            return "ready"
        except Exception:
            return "not_running"

    def _check_ollama_model(self) -> str:
        """Check if the specified Ollama model is available."""
        try:
            import ollama
            models = ollama.list()
            model_names = [m.model for m in models.models]
            
            # Check for exact match or match without tag
            for name in model_names:
                if name == self.llm_model or name.startswith(f"{self.llm_model}:"):
                    return "ready"
                # Also check if our model matches without the tag
                if self.llm_model.split(":")[0] == name.split(":")[0]:
                    # Check if versions match
                    our_tag = self.llm_model.split(":")[-1] if ":" in self.llm_model else "latest"
                    their_tag = name.split(":")[-1] if ":" in name else "latest"
                    if our_tag == their_tag:
                        return "ready"
            
            return "missing"
        except Exception:
            return "missing"

    def _check_whisper_model(self) -> str:
        """Check if Whisper model is cached."""
        # faster-whisper uses huggingface_hub for downloads
        # Models are stored in ~/.cache/huggingface/hub/
        # The model name pattern is: models--Systran--faster-whisper-{size}
        model_dir_name = f"models--Systran--faster-whisper-{self.whisper_model}"
        model_path = WHISPER_CACHE_DIR / model_dir_name

        if model_path.exists() and any(model_path.iterdir()):
            return "ready"
        return "missing"

    def _check_kokoro_cli(self) -> str:
        """Check if kokoro-tts CLI is installed."""
        if shutil.which("kokoro-tts"):
            return "ready"

        # Check common locations
        candidates = [
            Path.home() / ".local" / "bin" / "kokoro-tts",
            Path.home() / ".cargo" / "bin" / "kokoro-tts",
        ]
        for candidate in candidates:
            if candidate.exists():
                return "ready"

        return "missing"

    def _check_kokoro_models(self) -> str:
        """Check if Kokoro TTS models are downloaded."""
        model_file = "kokoro-v1.0.onnx"
        voices_file = "voices-v1.0.bin"

        # Check environment variable first
        if env_dir := os.environ.get("IRIS_MODELS_DIR"):
            if (Path(env_dir) / model_file).exists() and (Path(env_dir) / voices_file).exists():
                return "ready"

        # Check standard locations
        search_paths = [
            Path.home() / ".iris" / "models",
            Path.home() / ".local" / "share" / "iris" / "models",
            Path.cwd() / "models",
        ]

        for search_dir in search_paths:
            if (search_dir / model_file).exists() and (search_dir / voices_file).exists():
                return "ready"

        return "missing"

    def _download_ollama_model(self) -> bool:
        """Download Ollama model with progress display."""
        self.console.print(f"[bold]Pulling Ollama model: {self.llm_model}[/bold]")
        self.console.print(f"[dim]Location: ~/.ollama/models/[/dim]")
        self.console.print()

        try:
            import ollama

            # Use ollama.pull with progress callback
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Downloading...", total=100)
                current_status = ""

                for response in ollama.pull(self.llm_model, stream=True):
                    status = response.get("status", "")
                    
                    if status != current_status:
                        current_status = status
                        progress.update(task, description=status[:40])

                    # Update progress if available
                    completed = response.get("completed", 0)
                    total = response.get("total", 0)
                    if total > 0:
                        percentage = (completed / total) * 100
                        progress.update(task, completed=percentage)

                progress.update(task, completed=100, description="Complete")

            self.console.print(f"[green]✓ Model {self.llm_model} downloaded[/green]")
            self.console.print()
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to download model: {e}[/red]")
            return False

    def _download_whisper_model(self) -> bool:
        """Download Whisper model with progress display."""
        self.console.print(f"[bold]Downloading Whisper model: {self.whisper_model}[/bold]")
        self.console.print(f"[dim]Location: {WHISPER_CACHE_DIR}/[/dim]")
        self.console.print()

        try:
            # Import faster_whisper which will trigger the download
            # We'll use huggingface_hub directly for better progress control
            from huggingface_hub import snapshot_download

            model_name = f"Systran/faster-whisper-{self.whisper_model}"

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"Downloading {self.whisper_model}...", total=None)

                # Download the model
                snapshot_download(
                    repo_id=model_name,
                    local_dir=None,  # Use default cache
                )

                progress.update(task, description="Complete", total=1, completed=1)

            self.console.print(f"[green]✓ Whisper model {self.whisper_model} downloaded[/green]")
            self.console.print()
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to download Whisper model: {e}[/red]")
            return False

    def _download_kokoro_models(self) -> bool:
        """Download Kokoro TTS models with progress display."""
        self.console.print("[bold]Downloading Kokoro TTS models[/bold]")
        self.console.print(f"[dim]Location: {KOKORO_MODELS_DIR}/[/dim]")
        self.console.print()

        # Create target directory
        KOKORO_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        downloads = [
            (KOKORO_MODEL_URL, KOKORO_MODELS_DIR / "kokoro-v1.0.onnx", "Kokoro model"),
            (KOKORO_VOICES_URL, KOKORO_MODELS_DIR / "voices-v1.0.bin", "Voice data"),
        ]

        try:
            for url, path, description in downloads:
                self._download_file(url, path, description)

            self.console.print("[green]✓ Kokoro TTS models downloaded[/green]")
            self.console.print()
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to download Kokoro models: {e}[/red]")
            self.console.print(f"[dim]You can download manually from:[/dim]")
            self.console.print(f"[dim]  {KOKORO_MODEL_URL}[/dim]")
            self.console.print(f"[dim]  {KOKORO_VOICES_URL}[/dim]")
            return False

    def _download_file(self, url: str, path: Path, description: str):
        """Download a single file with progress bar."""
        with urllib.request.urlopen(url, timeout=30) as response:
            total_size = int(response.headers.get("Content-Length", 0))

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(description, total=total_size)

                with open(path, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))


def run_setup(whisper_model: str = "small", llm_model: str = "qwen3:1.7b") -> bool:
    """
    Run setup wizard.

    Args:
        whisper_model: Whisper model size
        llm_model: Ollama model name

    Returns:
        True if setup succeeded, False otherwise
    """
    wizard = SetupWizard(whisper_model=whisper_model, llm_model=llm_model)
    return wizard.run()




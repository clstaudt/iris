"""Setup wizard for first-run experience."""

import os
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
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

from iris.transcription import model_repo, resolve_device, resolve_engine, resolve_model
from iris.turns import SMART_TURN_FILE, SMART_TURN_REPO

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

    def __init__(
        self,
        whisper_model: str = "auto",
        llm_model: str = "qwen3:1.7b",
        whisper_engine: str = "auto",
    ):
        """
        Initialize setup wizard.

        Args:
            whisper_model: Whisper model size to use ("auto" picks per engine)
            llm_model: Ollama model name to use
            whisper_engine: "auto", "mlx" or "faster-whisper"
        """
        self.console = Console()
        engine = resolve_engine(whisper_engine)
        self.whisper_model = resolve_model(whisper_model, engine, resolve_device(engine))
        # MLX and faster-whisper keep their weights in different repos
        self.whisper_repo = model_repo(self.whisper_model, engine)
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

        if checks["kokoro_models"] == "missing":
            downloads_needed.append(("kokoro_models", "Kokoro TTS models (~360 MB)"))

        if checks["turn_model"] == "missing":
            downloads_needed.append(("turn_model", "Smart Turn model (9 MB)"))

        # If nothing to download, we're ready
        if not downloads_needed:
            self.console.print()
            self.console.print("[bold green]✓ All dependencies ready![/bold green]")
            self.console.print()
            return True

        # Ask user to confirm downloads
        self.console.print()
        self.console.print(
            "[bold yellow]The following components need to be downloaded:[/bold yellow]"
        )
        for _, desc in downloads_needed:
            self.console.print(f"  • {desc}")

        # Show where files will be stored
        self.console.print()
        self.console.print("[dim]Download locations:[/dim]")
        if checks["ollama_model"] == "missing":
            self.console.print("  [dim]• Ollama models: ~/.ollama/models/[/dim]")
        if checks["whisper"] == "missing" or checks["turn_model"] == "missing":
            self.console.print(f"  [dim]• Whisper and Smart Turn: {WHISPER_CACHE_DIR}/[/dim]")
        if checks["kokoro_models"] == "missing":
            self.console.print(f"  [dim]• Kokoro models: {KOKORO_MODELS_DIR}/[/dim]")

        self.console.print()
        try:
            response = self.console.input("[bold]Download now? [Y/n]: [/bold]").strip().lower()
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
            elif download_id == "turn_model":
                if not self._download_turn_model():
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
            "kokoro_models": self._check_kokoro_models(),
            "turn_model": self._check_turn_model(),
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

        # Kokoro models
        if checks["kokoro_models"] == "ready":
            table.add_row("[green]✓[/green]", "Kokoro TTS models", "cached")
        else:
            table.add_row("[yellow]↓[/yellow]", "Kokoro TTS models", "needs download")

        # Smart Turn, which hears when the user has finished speaking
        if checks["turn_model"] == "ready":
            table.add_row("[green]✓[/green]", "Smart Turn model", "cached")
        else:
            table.add_row("[yellow]↓[/yellow]", "Smart Turn model", "needs download")

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
            model_names = [m.model for m in models.models if m.model]

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
        # Both engines download through huggingface_hub into
        # ~/.cache/huggingface/hub/models--{org}--{name}
        model_dir_name = "models--" + self.whisper_repo.replace("/", "--")
        model_path = WHISPER_CACHE_DIR / model_dir_name

        try:
            if model_path.exists() and model_path.is_dir() and any(model_path.iterdir()):
                return "ready"
        except (PermissionError, OSError):
            # Can't read directory - treat as missing
            pass
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
        self.console.print("[dim]Location: ~/.ollama/models/[/dim]")
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
                    completed = response.get("completed") or 0
                    total = response.get("total") or 0
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
            model_name = self.whisper_repo

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

    def _check_turn_model(self) -> str:
        """Check if the Smart Turn model is cached."""
        try:
            hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE, local_files_only=True)
            return "ready"
        except Exception:
            return "missing"

    def _download_turn_model(self) -> bool:
        """Download the Smart Turn model (small: no progress bar)."""
        try:
            with self.console.status("Downloading Smart Turn model..."):
                hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE)
            self.console.print("[green]✓ Smart Turn model downloaded[/green]")
            self.console.print()
            return True
        except Exception as e:
            # Turns then end after a silence, as without it
            self.console.print(f"[yellow]! Smart Turn model not downloaded: {e}[/yellow]")
            return True

    def _download_kokoro_models(self) -> bool:
        """
        Download Kokoro TTS models with progress display.
        Uses atomic download - both files must succeed or neither is kept.
        """
        self.console.print("[bold]Downloading Kokoro TTS models[/bold]")
        self.console.print(f"[dim]Location: {KOKORO_MODELS_DIR}/[/dim]")
        self.console.print()

        # Create target directory
        KOKORO_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = KOKORO_MODELS_DIR / "kokoro-v1.0.onnx"
        voices_path = KOKORO_MODELS_DIR / "voices-v1.0.bin"

        # Download to temp files first for atomicity
        downloads = [
            (KOKORO_MODEL_URL, model_path, "Kokoro model"),
            (KOKORO_VOICES_URL, voices_path, "Voice data"),
        ]

        try:
            # Download all files (they will be downloaded to .tmp first)
            for url, path, description in downloads:
                # Skip if file already exists and has content
                if path.exists() and path.stat().st_size > 0:
                    self.console.print(f"[dim]{description} already exists, skipping[/dim]")
                    continue
                self._download_file(url, path, description)

            self.console.print("[green]✓ Kokoro TTS models downloaded[/green]")
            self.console.print()
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to download Kokoro models: {e}[/red]")
            self.console.print("[dim]You can download manually from:[/dim]")
            self.console.print(f"[dim]  {KOKORO_MODEL_URL}[/dim]")
            self.console.print(f"[dim]  {KOKORO_VOICES_URL}[/dim]")

            # Clean up any partial downloads
            for _, path, _ in downloads:
                if path.exists() and path.stat().st_size == 0:
                    try:
                        path.unlink()
                    except Exception:
                        pass

            return False

    def _download_file(self, url: str, path: Path, description: str):
        """
        Download a single file with progress bar.

        Args:
            url: URL to download from
            path: Path to save to
            description: Description for progress display

        Raises:
            Exception: If download fails or file validation fails
        """
        temp_path = path.with_suffix(path.suffix + ".tmp")

        try:
            # Use longer timeout for large files (5 minutes)
            with urllib.request.urlopen(url, timeout=300) as response:
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

                    with open(temp_path, "wb") as f:
                        downloaded = 0
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress.update(task, advance=len(chunk))

                # Validate download
                if total_size > 0 and downloaded != total_size:
                    raise RuntimeError(
                        f"Download incomplete: expected {total_size} bytes, got {downloaded} bytes"
                    )

                # Validate file exists and has content
                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    raise RuntimeError("Downloaded file is empty or missing")

                # Move temp file to final location
                temp_path.replace(path)

        except Exception as e:
            # Clean up partial download
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise e


def run_setup(
    whisper_model: str = "auto", llm_model: str = "qwen3:1.7b", whisper_engine: str = "auto"
) -> bool:
    """
    Run setup wizard.

    Args:
        whisper_model: Whisper model size
        llm_model: Ollama model name
        whisper_engine: "auto", "mlx" or "faster-whisper"

    Returns:
        True if setup succeeded, False otherwise
    """
    wizard = SetupWizard(
        whisper_model=whisper_model, llm_model=llm_model, whisper_engine=whisper_engine
    )
    return wizard.run()

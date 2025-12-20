"""Text-to-speech generation and pipeline using kokoro-tts."""

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import urllib.request
import warnings
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn

warnings.filterwarnings("ignore")

# Kokoro model download URLs
KOKORO_MODEL_URL = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"


class KokoroGenerator:
    """Generates speech audio using kokoro-tts."""

    def __init__(self, voice: str = "af_bella", speed: float = 1.25, quiet: bool = True):
        """
        Initialize kokoro TTS generator.

        Args:
            voice: Voice to use for generation
            speed: Speech speed multiplier
            quiet: If True, skip interactive prompts (setup wizard handles downloads)
        """
        self.voice = voice
        self.speed = speed
        self.quiet = quiet
        self.kokoro_cmd = self._find_kokoro_cmd()
        
        # Find model files - check multiple locations
        self.model_path, self.voices_path = self._locate_models()

    def _find_kokoro_cmd(self) -> str:
        """
        Find kokoro-tts command.
        
        Returns:
            Path to kokoro-tts executable
            
        Raises:
            FileNotFoundError: If kokoro-tts is not found
        """
        # Check PATH first
        if cmd := shutil.which("kokoro-tts"):
            return cmd
        
        # Check common locations
        candidates = [
            Path.home() / ".local" / "bin" / "kokoro-tts",  # Unix pip --user
            Path.home() / ".cargo" / "bin" / "kokoro-tts",  # Rust install
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        
        raise FileNotFoundError(
            "kokoro-tts not found. Install with: pip install kokoro-tts"
        )

    def _locate_models(self) -> tuple[Path, Path]:
        """
        Locate kokoro model files.
        
        Searches in order:
        1. IRIS_MODELS_DIR environment variable
        2. ~/.iris/models/
        3. ~/.local/share/iris/models/
        4. ./models/ (relative to current directory)
        
        If models are not found and not in quiet mode, prompts the user to download them.
        
        Returns:
            Tuple of (model_path, voices_path)
        
        Raises:
            FileNotFoundError: If models cannot be found and user declines download
        """
        model_file = "kokoro-v1.0.onnx"
        voices_file = "voices-v1.0.bin"
        
        # Check environment variable first
        if env_dir := os.environ.get("IRIS_MODELS_DIR"):
            model_path = Path(env_dir) / model_file
            voices_path = Path(env_dir) / voices_file
            if model_path.exists() and voices_path.exists():
                return model_path, voices_path
        
        # Check standard locations
        search_paths = [
            Path.home() / ".iris" / "models",
            Path.home() / ".local" / "share" / "iris" / "models",
            Path.cwd() / "models",
        ]
        
        for search_dir in search_paths:
            model_path = search_dir / model_file
            voices_path = search_dir / voices_file
            if model_path.exists() and voices_path.exists():
                return model_path, voices_path
        
        # Models not found - prompt user to download (unless in quiet mode)
        default_dir = search_paths[0]  # ~/.iris/models
        
        if not self.quiet and self._prompt_download(default_dir, model_file, voices_file):
            return default_dir / model_file, default_dir / voices_file
        
        # Models not found
        raise FileNotFoundError(
            f"Could not find kokoro models ({model_file}, {voices_file}).\n"
            f"Please run 'iris' without --skip-setup to download models,\n"
            f"or download them manually and place in one of:\n"
            + "\n".join(f"  - {p}" for p in search_paths)
            + f"\nOr set IRIS_MODELS_DIR environment variable."
        )

    def _prompt_download(self, target_dir: Path, model_file: str, voices_file: str) -> bool:
        """
        Prompt the user to download kokoro models.
        
        Args:
            target_dir: Directory to download models to
            model_file: Name of the model file
            voices_file: Name of the voices file
            
        Returns:
            True if download succeeded, False if user declined or download failed
        """
        console = Console()
        
        console.print()
        console.print("[bold yellow]Kokoro TTS models not found![/bold yellow]")
        console.print()
        console.print("The following files are required for text-to-speech:")
        console.print(f"  • {model_file} (~350 MB)")
        console.print(f"  • {voices_file} (~8 MB)")
        console.print()
        console.print(f"They will be downloaded to: [cyan]{target_dir}[/cyan]")
        console.print()
        
        try:
            response = console.input("[bold]Download models now? [Y/n]: [/bold]").strip().lower()
            if response in ("", "y", "yes"):
                return self._download_models(target_dir, model_file, voices_file)
            else:
                console.print("[dim]Download cancelled.[/dim]")
                return False
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Download cancelled.[/dim]")
            return False

    def _download_models(self, target_dir: Path, model_file: str, voices_file: str) -> bool:
        """
        Download kokoro models from Hugging Face.
        
        Args:
            target_dir: Directory to save models to
            model_file: Name of the model file
            voices_file: Name of the voices file
            
        Returns:
            True if download succeeded, False otherwise
        """
        console = Console()
        
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        downloads = [
            (KOKORO_MODEL_URL, target_dir / model_file, "Kokoro model"),
            (KOKORO_VOICES_URL, target_dir / voices_file, "Voice data"),
        ]
        
        console.print()
        
        try:
            for url, path, description in downloads:
                self._download_file(console, url, path, description)
            
            console.print()
            console.print("[bold green]✓ Models downloaded successfully![/bold green]")
            console.print()
            return True
            
        except Exception as e:
            console.print(f"\n[bold red]Download failed: {e}[/bold red]")
            console.print("Please download manually from:")
            console.print(f"  {KOKORO_MODEL_URL}")
            console.print(f"  {KOKORO_VOICES_URL}")
            return False

    def _download_file(self, console: Console, url: str, path: Path, description: str):
        """
        Download a single file with progress bar.
        
        Args:
            console: Rich Console instance
            url: URL to download from
            path: Path to save to
            description: Description for progress display
        """
        # Get file size from headers
        with urllib.request.urlopen(url, timeout=30) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(description, total=total_size)
                
                with open(path, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

    def generate(self, text: str) -> str | None:
        """
        Generate audio file from text using kokoro-tts CLI.

        Args:
            text: Text to convert to speech

        Returns:
            Path to generated WAV file, or None if generation failed
        """
        temp_audio = None
        try:
            # Create temporary file for output
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio.close()

            result = subprocess.run(
                [
                    self.kokoro_cmd,
                    "/dev/stdin",
                    temp_audio.name,
                    "--voice",
                    self.voice,
                    "--speed",
                    str(self.speed),
                    "--lang",
                    "en-us",
                    "--format",
                    "wav",
                    "--model",
                    str(self.model_path),
                    "--voices",
                    str(self.voices_path),
                ],
                input=text,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"kokoro-tts CLI failed (exit {result.returncode}): "
                    f"stdout={result.stdout}, stderr={result.stderr}"
                )

            # Check if file was created and has content
            if not Path(temp_audio.name).exists():
                raise RuntimeError(
                    f"kokoro-tts did not create output file. "
                    f"stdout: {result.stdout}, stderr: {result.stderr}"
                )

            file_size = Path(temp_audio.name).stat().st_size
            if file_size == 0:
                raise RuntimeError(
                    f"kokoro-tts created empty file. "
                    f"stdout: {result.stdout}, stderr: {result.stderr}"
                )

            return temp_audio.name

        except FileNotFoundError:
            print("Error: kokoro-tts CLI not found. Install with: pip install kokoro")
            return None
        except subprocess.TimeoutExpired:
            print("Error: Speech generation timed out")
            return None
        except Exception as e:
            print(f"Error generating speech: {e}")
            import traceback

            traceback.print_exc()
            return None


class SpeechPipeline:
    """
    Parallel speech generation and playback pipeline.
    
    Generation and playback happen in parallel for maximum responsiveness.
    Text is queued for generation, which produces audio files that are
    then queued for playback.
    """

    def __init__(self, generator, player):
        """
        Initialize speech pipeline.

        Args:
            generator: KokoroGenerator instance for TTS
            player: AudioPlayer instance for playback
        """
        self.generator = generator
        self.player = player
        
        # Queues for pipeline
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Flags for status tracking
        self.is_speaking_flag = threading.Event()
        self.is_generating_flag = threading.Event()
        self._status_lock = threading.Lock()
        
        # Worker threads
        self.generation_worker_thread = None
        self.playback_worker_thread = None
        
        # Start the pipeline
        self._start_pipeline()

    def _start_pipeline(self):
        """
        Start pipeline: generation thread → audio queue → playback thread.
        """

        def generation_worker():
            """Generate TTS audio files from text queue."""
            while True:
                text = self.text_queue.get()
                if text is None:  # Shutdown signal
                    self.audio_queue.put(None)
                    break

                # Set flag to indicate TTS generation in progress
                self.is_generating_flag.set()

                # Generate audio file
                audio_file = self.generator.generate(text)

                # Clear flag when done generating
                self.is_generating_flag.clear()

                # Put generated audio in playback queue
                if audio_file:
                    self.audio_queue.put(audio_file)
                self.text_queue.task_done()

        def playback_worker():
            """Play audio files from audio queue sequentially."""
            while True:
                audio_file = self.audio_queue.get()
                if audio_file is None:  # Shutdown signal
                    break

                # Play the audio
                self.is_speaking_flag.set()
                self.player.play(audio_file)
                self.is_speaking_flag.clear()

                # Cleanup
                try:
                    Path(audio_file).unlink(missing_ok=True)
                except Exception:
                    pass

                self.audio_queue.task_done()

        # Start both threads
        self.generation_worker_thread = threading.Thread(
            target=generation_worker, daemon=True
        )
        self.playback_worker_thread = threading.Thread(
            target=playback_worker, daemon=True
        )
        self.generation_worker_thread.start()
        self.playback_worker_thread.start()

    @property
    def is_active(self) -> bool:
        """Check if pipeline has pending work or is currently processing."""
        with self._status_lock:
            return (
                not self.text_queue.empty()
                or self.is_generating_flag.is_set()
                or not self.audio_queue.empty()
                or self.is_speaking_flag.is_set()
            )

    def queue_speech(self, text: str):
        """
        Add text to speech pipeline (non-blocking).

        Args:
            text: Text to speak
        """
        self.text_queue.put(text)

    def wait_for_speech(self):
        """Wait for all speech generation and playback to finish."""
        self.text_queue.join()  # Wait for generation
        self.audio_queue.join()  # Wait for playback

    def stop(self):
        """Stop the speech pipeline immediately."""
        # Stop any currently playing audio
        self.player.stop()
        
        # Clear queues
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
                self.text_queue.task_done()
            except Exception:
                pass
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except Exception:
                pass


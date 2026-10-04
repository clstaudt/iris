"""Text-to-speech generation and pipeline using kokoro-onnx."""

import os
import queue
import tempfile
import threading
import urllib.request
import warnings
from pathlib import Path

import soundfile as sf
from kokoro_onnx import Kokoro
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn

warnings.filterwarnings("ignore")

# Kokoro model download URLs
KOKORO_MODEL_URL = (
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"
)

# Mapping from Whisper language codes to Kokoro TTS language/voice settings.
# Kokoro supports: en-us, en-gb, fr-fr, it, ja, cmn
# A Kokoro voice name starts with its language ("af_bella": American English,
# "bm_lewis": British English, "ff_siwis": French...). A persona keeps its own
# voice for languages that voice speaks, and borrows the default otherwise.
LANGUAGE_VOICE_MAP = {
    "en": {"code": "en-us", "default_voice": "af_bella", "voice_prefixes": "ab"},
    "fr": {"code": "fr-fr", "default_voice": "ff_siwis", "voice_prefixes": "f"},
    "it": {"code": "it", "default_voice": "if_sara", "voice_prefixes": "i"},
    "ja": {"code": "ja", "default_voice": "jf_alpha", "voice_prefixes": "j"},
    "zh": {"code": "cmn", "default_voice": "zf_xiaoni", "voice_prefixes": "z"},
}


def get_language_info(whisper_lang: str) -> dict | None:
    """Get Kokoro language info for a Whisper language code.

    Args:
        whisper_lang: Whisper language code (e.g. 'en', 'fr', 'ja', 'zh')

    Returns:
        Dict with 'code' and 'default_voice', or None if unsupported.
    """
    return LANGUAGE_VOICE_MAP.get(whisper_lang)


def is_language_supported(whisper_lang: str) -> bool:
    """Check if a Whisper language code is supported by Kokoro TTS."""
    return whisper_lang in LANGUAGE_VOICE_MAP


KOKORO_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"
# Where Iris keeps the voice model when it downloads it
DEFAULT_KOKORO_DIR = Path.home() / ".iris" / "models"


def kokoro_files_in(folder: Path) -> bool:
    """Whether a folder holds both Kokoro files."""
    return (folder / KOKORO_MODEL_FILE).exists() and (folder / KOKORO_VOICES_FILE).exists()


def find_kokoro_models() -> Path | None:
    """The first of the usual places that holds the Kokoro files, if any."""
    candidates = []
    if env_dir := os.environ.get("IRIS_MODELS_DIR"):
        candidates.append(Path(env_dir))
    candidates += [
        DEFAULT_KOKORO_DIR,
        Path.home() / ".local" / "share" / "iris" / "models",
        Path.cwd() / "models",
    ]
    return next((folder for folder in candidates if kokoro_files_in(folder)), None)


class KokoroGenerator:
    """Generates speech audio using kokoro-onnx."""

    def __init__(
        self,
        voice: str = "af_bella",
        speed: float = 1.25,
        quiet: bool = True,
        models_dir: str | Path | None = None,
    ):
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

        # A folder chosen by the user, or the usual locations
        self.models_dir = Path(models_dir).expanduser() if models_dir else None
        if self.models_dir is not None:
            if not kokoro_files_in(self.models_dir):
                raise FileNotFoundError(
                    f"{self.models_dir} does not hold {KOKORO_MODEL_FILE} and {KOKORO_VOICES_FILE}"
                )
            self.model_path = self.models_dir / KOKORO_MODEL_FILE
            self.voices_path = self.models_dir / KOKORO_VOICES_FILE
        else:
            self.model_path, self.voices_path = self._locate_models()

        # Initialize Kokoro
        self.kokoro = Kokoro(model_path=str(self.model_path), voices_path=str(self.voices_path))

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

    def voice_for(self, language: str) -> tuple[str, str]:
        """Pick (Kokoro language code, voice) for a reply in this language.

        The persona's voice is used whenever it speaks the language, so
        personas sound different from each other; for a language it cannot
        speak, that language's default voice takes over.
        """
        lang_info = get_language_info(language)
        if lang_info is None:
            # Unsupported language – fall back to English with the own voice
            kokoro_lang, voice = "en-us", self.voice
        elif self.voice[:1] in lang_info["voice_prefixes"]:
            kokoro_lang, voice = lang_info["code"], self.voice
        else:
            kokoro_lang, voice = lang_info["code"], lang_info["default_voice"]
        # British voices are given British pronunciation
        if kokoro_lang == "en-us" and voice.startswith("b"):
            kokoro_lang = "en-gb"
        return kokoro_lang, voice

    def generate(self, text: str, language: str = "en") -> str | None:
        """
        Generate audio file from text using kokoro-onnx Python API.

        Args:
            text: Text to convert to speech
            language: Whisper language code (e.g. 'en', 'fr', 'ja').
                Selects the Kokoro language and an appropriate voice.
                Falls back to English if unsupported.

        Returns:
            Path to generated WAV file, or None if generation failed
        """
        kokoro_lang, voice = self.voice_for(language)

        temp_audio = None
        try:
            # Create temporary file for output
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio.close()

            # Generate audio using kokoro-onnx Python API
            audio_data, sample_rate = self.kokoro.create(
                text=text, voice=voice, speed=self.speed, lang=kokoro_lang
            )

            # Save audio to file
            sf.write(temp_audio.name, audio_data, sample_rate)

            # Verify file was created and has content
            if not Path(temp_audio.name).exists():
                raise RuntimeError("Failed to create output file")

            file_size = Path(temp_audio.name).stat().st_size
            if file_size == 0:
                raise RuntimeError("Created empty audio file")

            return temp_audio.name

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

        # Current language for TTS (set by assistant on each turn)
        self.current_language = "en"

        # Optional callback receiving playback level (0.0-1.0) for output meters
        self.on_output_level = None

        # Queues for pipeline
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        # Flags for status tracking
        self.is_speaking_flag = threading.Event()
        self.is_generating_flag = threading.Event()
        self._status_lock = threading.Lock()

        # stop() starts a new epoch; work queued in an older one is dropped
        # wherever it is, even audio still being generated when stop() came
        self._epoch = 0
        self._generating_epoch: int | None = None

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
            """Generate TTS audio files from text queue with lower priority."""
            # Lower priority of generation thread to prevent audio underruns
            try:
                import os

                # Increase niceness significantly (lower priority) on Unix systems
                os.nice(10)
            except (AttributeError, OSError):
                pass  # Not available on all systems

            while True:
                item = self.text_queue.get()
                if item is None:  # Shutdown signal
                    self.audio_queue.put(None)
                    break

                # Unpack text, language and the epoch it was queued in
                text, language, epoch = item
                if epoch != self._epoch:  # stopped before its turn came
                    self.text_queue.task_done()
                    continue

                # Set flag to indicate TTS generation in progress
                self._generating_epoch = epoch
                self.is_generating_flag.set()

                # Generate audio file in the requested language
                audio_file = self.generator.generate(text, language=language)

                # Clear flag when done generating
                self.is_generating_flag.clear()
                self._generating_epoch = None

                # Put generated audio in playback queue, unless stop() came
                # while it was being generated
                if audio_file and epoch == self._epoch:
                    self.audio_queue.put((audio_file, epoch))
                elif audio_file:
                    Path(audio_file).unlink(missing_ok=True)
                self.text_queue.task_done()

        def playback_worker():
            """Play audio files from audio queue sequentially."""
            while True:
                item = self.audio_queue.get()
                if item is None:  # Shutdown signal
                    break
                audio_file, epoch = item

                # Play the audio, unless it belongs to a stopped epoch; the
                # player checks again while playing, so a stop() racing the
                # start of the clip still silences it
                if epoch == self._epoch:
                    self.is_speaking_flag.set()
                    self.player.play(
                        audio_file,
                        on_level=self.on_output_level,
                        cancelled=lambda: epoch != self._epoch,
                    )
                    self.is_speaking_flag.clear()

                # Cleanup
                try:
                    Path(audio_file).unlink(missing_ok=True)
                except Exception:
                    pass

                self.audio_queue.task_done()

        # Start both threads
        self.generation_worker_thread = threading.Thread(target=generation_worker, daemon=True)
        self.playback_worker_thread = threading.Thread(target=playback_worker, daemon=True)
        self.generation_worker_thread.start()
        self.playback_worker_thread.start()

    @property
    def is_active(self) -> bool:
        """Check if pipeline has pending work or is currently processing."""
        with self._status_lock:
            # Audio still being generated for a stopped epoch will be dropped:
            # it does not count as speaking
            generating = (
                self.is_generating_flag.is_set() and self._generating_epoch == self._epoch
            )
            return (
                not self.text_queue.empty()
                or generating
                or not self.audio_queue.empty()
                or self.is_speaking_flag.is_set()
            )

    def queue_speech(self, text: str):
        """
        Add text to speech pipeline (non-blocking).

        Uses self.current_language for the TTS language code.

        Args:
            text: Text to speak
        """
        self.text_queue.put((text, self.current_language, self._epoch))

    def wait_for_speech(self):
        """Wait for all speech generation and playback to finish."""
        self.text_queue.join()  # Wait for generation
        self.audio_queue.join()  # Wait for playback

    def pause(self):
        """Hold speech mid-sentence; queued text keeps being synthesized."""
        self.player.pause()

    def resume(self):
        """Continue speech from where pause() held it."""
        self.player.resume()

    def stop(self):
        """Stop the speech pipeline immediately: what plays, what waits, and
        what is still being generated."""
        with self._status_lock:
            self._epoch += 1
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
                item = self.audio_queue.get_nowait()
                if item is not None:
                    Path(item[0]).unlink(missing_ok=True)
                self.audio_queue.task_done()
            except Exception:
                pass

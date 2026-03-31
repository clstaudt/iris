"""Persistent voice store for user-cloned voices."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import soundfile as sf
import yaml


# Minimum and maximum reference audio duration in seconds
MIN_REFERENCE_DURATION = 3.0
MAX_REFERENCE_DURATION = 30.0

VOICES_DIR = Path.home() / ".iris" / "voices"


class VoiceInfo:
    """Metadata for a stored cloned voice."""

    def __init__(
        self,
        name: str,
        base_voice: str,
        created: str,
        description: str = "",
        reference_duration: float = 0.0,
    ):
        self.name = name
        self.base_voice = base_voice
        self.created = created
        self.description = description
        self.reference_duration = reference_duration

    @property
    def reference_path(self) -> Path:
        return VOICES_DIR / self.name / "reference.wav"


class VoiceStore:
    """Manages persistent storage of user-cloned voices in ~/.iris/voices/."""

    def __init__(self, voices_dir: Optional[Path] = None):
        self.voices_dir = voices_dir or VOICES_DIR

    def save_voice(
        self,
        name: str,
        reference_audio_path: str | Path,
        base_voice: str,
        description: str = "",
    ) -> VoiceInfo:
        """
        Save a new cloned voice from a reference audio file.

        Args:
            name: Name for the voice (used in persona YAML)
            reference_audio_path: Path to the reference audio file
            base_voice: Kokoro base voice used for initial synthesis
            description: Optional description of the voice

        Returns:
            VoiceInfo for the saved voice

        Raises:
            FileNotFoundError: If reference audio doesn't exist
            ValueError: If name is invalid or audio doesn't meet requirements
            FileExistsError: If a voice with this name already exists
        """
        reference_audio_path = Path(reference_audio_path)

        # Validate name
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Invalid voice name '{name}'. Use only letters, numbers, hyphens, and underscores."
            )

        # Check for conflicts with built-in Kokoro voices
        if self._is_builtin_voice(name):
            raise ValueError(
                f"'{name}' conflicts with a built-in Kokoro voice name. Choose a different name."
            )

        # Check if already exists
        voice_dir = self.voices_dir / name
        if voice_dir.exists():
            raise FileExistsError(
                f"Voice '{name}' already exists. Delete it first or choose a different name."
            )

        # Validate reference audio
        if not reference_audio_path.exists():
            raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path}")

        duration = self._get_audio_duration(reference_audio_path)
        if duration < MIN_REFERENCE_DURATION:
            raise ValueError(
                f"Reference audio too short ({duration:.1f}s). "
                f"Minimum {MIN_REFERENCE_DURATION:.0f} seconds required."
            )
        if duration > MAX_REFERENCE_DURATION:
            raise ValueError(
                f"Reference audio too long ({duration:.1f}s). "
                f"Maximum {MAX_REFERENCE_DURATION:.0f} seconds recommended."
            )

        # Create voice directory and copy reference audio
        voice_dir.mkdir(parents=True, exist_ok=True)
        dest_path = voice_dir / "reference.wav"

        # Read and re-save as WAV to normalize format
        audio_data, sample_rate = sf.read(str(reference_audio_path))
        sf.write(str(dest_path), audio_data, sample_rate)

        # Save metadata
        metadata = {
            "name": name,
            "base_voice": base_voice,
            "created": datetime.now().isoformat(),
            "description": description,
            "reference_duration": round(duration, 2),
        }
        with open(voice_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        return VoiceInfo(**metadata)

    def get_voice(self, name: str) -> Optional[VoiceInfo]:
        """Get voice info by name, or None if not found."""
        voice_dir = self.voices_dir / name
        metadata_path = voice_dir / "metadata.yaml"

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            data = yaml.safe_load(f)

        return VoiceInfo(
            name=data["name"],
            base_voice=data["base_voice"],
            created=data["created"],
            description=data.get("description", ""),
            reference_duration=data.get("reference_duration", 0.0),
        )

    def list_voices(self) -> list[VoiceInfo]:
        """List all stored cloned voices."""
        voices = []
        if not self.voices_dir.exists():
            return voices

        for voice_dir in sorted(self.voices_dir.iterdir()):
            if not voice_dir.is_dir():
                continue
            info = self.get_voice(voice_dir.name)
            if info:
                voices.append(info)

        return voices

    def delete_voice(self, name: str) -> bool:
        """Delete a stored voice. Returns True if deleted, False if not found."""
        voice_dir = self.voices_dir / name
        if not voice_dir.exists():
            return False
        shutil.rmtree(voice_dir)
        return True

    def resolve_voice(self, voice_name: str) -> tuple[str, Optional[Path]]:
        """
        Resolve a voice name to a (base_voice, reference_audio_path) tuple.

        If the name refers to a cloned voice, returns the stored base voice
        and reference audio path. Otherwise returns (voice_name, None) to
        indicate it's a built-in Kokoro voice.

        Args:
            voice_name: Voice name from persona config

        Returns:
            (base_voice, reference_audio_path) — reference_audio_path is None for built-in voices
        """
        info = self.get_voice(voice_name)
        if info:
            ref_path = self.voices_dir / voice_name / "reference.wav"
            if ref_path.exists():
                return info.base_voice, ref_path
        return voice_name, None

    def _get_audio_duration(self, path: Path) -> float:
        """Get duration of an audio file in seconds."""
        info = sf.info(str(path))
        return info.duration

    def _is_builtin_voice(self, name: str) -> bool:
        """Check if a name conflicts with built-in Kokoro voice names."""
        # Kokoro voice naming convention: {prefix}_{name}
        # e.g. af_bella, am_michael, bm_lewis, af_sarah
        builtin_prefixes = ("af_", "am_", "bf_", "bm_", "ef_", "em_", "ff_", "hf_", "jf_")
        return name.startswith(builtin_prefixes)

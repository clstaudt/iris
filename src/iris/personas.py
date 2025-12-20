"""Persona definitions for the Iris voice assistant."""

import os
from pathlib import Path
from typing import Optional

import yaml


# Common base prompt for all voice assistant personas
VOICE_ASSISTANT_BASE = (
    "Your responses are spoken aloud through text-to-speech. "
    "Keep responses brief and conversational - suitable for voice interaction. "
    "Avoid markdown formatting, emojis, lists, and abbreviations since they don't work well when spoken. "
    "Be concise by default, but elaborate when asked or when the topic requires depth. "
)


class Persona:
    """A persona with name, description, voice, and optional audio filter."""
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        voice: str, 
        audio_filter: Optional[str] = None,
        speed: float = 1.25,
    ):
        """
        Initialize a persona.
        
        Args:
            name: Display name of the persona
            description: Unique personality traits (base voice traits are added automatically)
            voice: Kokoro TTS voice to use (e.g., 'af_bella', 'am_adam', 'bf_emma')
            audio_filter: Optional audio filter name ('phone', 'radio', 'intercom', 'robotic')
            speed: Speech speed multiplier (default: 1.25)
        """
        self.name = name
        # Combine base voice traits with unique personality
        self.description = VOICE_ASSISTANT_BASE + description
        self.voice = voice
        self.audio_filter = audio_filter
        self.speed = speed
    
    @classmethod
    def from_file(cls, filepath: Path) -> "Persona":
        """Load a persona from a YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            name=data["name"],
            description=data["description"],
            voice=data["voice"],
            audio_filter=data.get("audio_filter"),  # Optional field
            speed=data.get("speed", 1.25),  # Default to 1.25
        )


def get_persona_dirs() -> list[Path]:
    """Get directories to search for persona files."""
    dirs = []
    
    # Built-in personas (shipped with iris package)
    package_dir = Path(__file__).parent / "personas"
    if package_dir.exists():
        dirs.append(package_dir)
    
    # User personas directory
    user_dir = Path.home() / ".iris" / "personas"
    if user_dir.exists():
        dirs.append(user_dir)
    
    # Environment variable override
    if env_dir := os.environ.get("IRIS_PERSONAS_DIR"):
        dirs.append(Path(env_dir))
    
    return dirs


def load_all_personas() -> dict[str, Persona]:
    """Load all available personas from configuration files."""
    personas = {}
    
    for persona_dir in get_persona_dirs():
        for yaml_file in persona_dir.glob("*.yaml"):
            try:
                persona = Persona.from_file(yaml_file)
                # Use filename (without extension) as the key
                persona_id = yaml_file.stem.lower()
                personas[persona_id] = persona
            except Exception as e:
                print(f"Warning: Could not load persona from {yaml_file}: {e}")
    
    # Fallback: if no personas loaded, create a basic default
    if not personas:
        personas["iris"] = Persona(
            name="Iris",
            description="You are Iris, a helpful and friendly assistant. Be conversational, warm, and natural.",
            voice="af_bella",
        )
    
    return personas


# Load personas on module import
PERSONAS = load_all_personas()


def get_persona(persona_name: str = "iris") -> Persona:
    """
    Get a persona by name.
    
    Args:
        persona_name: Name of the persona to retrieve
        
    Returns:
        The requested persona, or default 'iris' persona if not found
    """
    persona_name = persona_name.lower()
    if persona_name not in PERSONAS:
        available = ", ".join(list_personas())
        print(f"Warning: Persona '{persona_name}' not found. Available: {available}")
        # Return first available persona
        return list(PERSONAS.values())[0]
    return PERSONAS[persona_name]


def list_personas() -> list[str]:
    """Get list of available persona names."""
    return list(PERSONAS.keys())

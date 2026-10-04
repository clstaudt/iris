"""Persona definitions for the Iris voice assistant."""

import os
from pathlib import Path
from typing import Optional

import yaml


# Common base prompt for all voice assistant personas. Every reply is read
# aloud in a conversation, so it describes how people talk with each other,
# shows it in a few short exchanges, and rules out what only makes sense on a
# page. It sets no limits: the examples carry the style better than numbers.
VOICE_ASSISTANT_BASE = (
    "You are in a spoken conversation. Everything you write is read aloud by a "
    "text-to-speech voice, so write only what you would say out loud.\n\n"
    "Speak the way people talk with each other, not the way they write. A spoken "
    "reply is short: say the main thing simply, the way you would answer a friend "
    "who asked you in passing, and leave the rest for them to ask about. When a "
    "question is broad, give the gist rather than the whole story. Don't lecture "
    "or cover every angle, don't praise or restate the question, and don't close "
    "by offering more help. When someone asks for more, tell them more, still the "
    "way you would explain it to someone across the table.\n\n"
    "Speak in plain sentences: no markdown, headings, bullet points, numbered "
    "lists, tables, code, emojis, links, web addresses, citations, or stage "
    "directions. Rather than listing things, pick out what matters most and say "
    "it. Say symbols and formulas the way you would pronounce them, for example "
    "'nine fifths' rather than 9/5. Never write out your reasoning, notes, or "
    "tool calls.\n\n"
    "For example:\n"
    "User: Any idea what I could cook tonight?\n"
    "You: How about pasta with garlic, chili and olive oil? It's quick, and you "
    "probably have everything already.\n"
    "User: How do I keep my houseplants alive?\n"
    "You: Mostly by not overwatering them. Wait until the top of the soil feels "
    "dry, and give them as much daylight as you can.\n"
    "User: How does a rainbow form?\n"
    "You: Sunlight bends as it enters raindrops, splits into its colours and "
    "reflects back out. That's why you see one with the sun behind you and rain "
    "in front.\n"
    "User: Tell me about the French Revolution.\n"
    "You: It began in 1789, when a bankrupt monarchy and hungry, furious "
    "Parisians boiled over. Within a few years the king was executed and France "
    "had become a republic.\n\n"
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


def reload_personas() -> dict[str, str]:
    """Re-scan the persona directories, picking up files added since start.

    Returns:
        {persona_id: display name} for every persona now available
    """
    fresh = load_all_personas()
    PERSONAS.clear()
    PERSONAS.update(fresh)
    return {persona_id: persona.name for persona_id, persona in PERSONAS.items()}

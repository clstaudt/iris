"""Persona definitions for the Iris voice assistant."""

import os
import re
from pathlib import Path
from typing import Optional

import yaml

from iris.tts import KOKORO_VOICE_NAME


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
    """A persona with name, description, voice, optional audio filter, and
    the words that wake it and send it to sleep."""

    def __init__(
        self,
        name: str,
        description: str,
        voice: str,
        audio_filter: Optional[str] = None,
        speed: float = 1.25,
        engine: Optional[str] = None,
        persona_id: Optional[str] = None,
        path: Optional[Path] = None,
        wake_word: Optional[str] = None,
        sleep_word: Optional[str] = None,
    ):
        """
        Initialize a persona.

        Args:
            name: Display name of the persona
            description: Unique personality traits (base voice traits are added automatically)
            voice: Voice of the speech engine (e.g. 'alba' for Kyutai, 'af_bella' for Kokoro)
            audio_filter: Optional audio filter name ('phone', 'radio', 'intercom', 'robotic')
            speed: Speech speed multiplier, for Kokoro voices (default: 1.25)
            engine: Speech engine, 'kyutai' or 'kokoro'; without one, a Kokoro
                voice name means Kokoro
            persona_id: Key the persona is known by (its file name)
            path: File the persona was loaded from
            wake_word: Said to wake the persona, when the wake word is on;
                its name when none is set
            sleep_word: Said while awake, sends it back to sleep; none when
                none is set
        """
        self.name = name
        # The personality as written, and combined with the base voice traits
        self.personality = description
        self.description = VOICE_ASSISTANT_BASE + description
        self.voice = voice
        self.audio_filter = audio_filter
        self.speed = speed
        if engine is None:
            engine = "kokoro" if KOKORO_VOICE_NAME.fullmatch(voice) else "kyutai"
        self.engine = engine
        self.id = persona_id or slug(name)
        self.path = path
        # Iris is woken by calling it by its name, unless told otherwise
        self.wake_word = str(wake_word or "").strip() or name
        self.sleep_word = str(sleep_word or "").strip()

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
            engine=data.get("engine"),
            persona_id=filepath.stem.lower(),
            path=filepath,
            wake_word=data.get("wake_word"),
            sleep_word=data.get("sleep_word"),
        )

    @property
    def builtin(self) -> bool:
        """Whether the persona is one shipped with Iris, as shipped."""
        return self.path is not None and self.path.parent == BUILTIN_DIR

    def to_dict(self) -> dict:
        """The persona's settings, as edited in the UI and saved to its file."""
        return {
            "name": self.name,
            "engine": self.engine,
            "voice": self.voice,
            "speed": self.speed,
            "audio_filter": self.audio_filter or "",
            "description": self.personality,
            "wake_word": self.wake_word,
            "sleep_word": self.sleep_word,
        }


def slug(name: str) -> str:
    """A file name for a persona called this."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "persona"


class _PersonaDumper(yaml.SafeDumper):
    """Writes multi-line text as a block, the way the persona files are written."""


_PersonaDumper.add_representer(
    str,
    lambda dumper, text: dumper.represent_scalar(
        "tag:yaml.org,2002:str", text, style="|" if "\n" in text else None
    ),
)


# Personas shipped with Iris, and those created or edited by the user
BUILTIN_DIR = Path(__file__).parent / "personas"
USER_DIR = Path.home() / ".iris" / "personas"


def get_persona_dirs() -> list[Path]:
    """Get directories to search for persona files."""
    dirs = []

    # Built-in personas (shipped with iris package)
    if BUILTIN_DIR.exists():
        dirs.append(BUILTIN_DIR)

    # User personas directory; a persona there replaces a built-in one of
    # the same file name
    if USER_DIR.exists():
        dirs.append(USER_DIR)

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
            voice="anna",
            persona_id="iris",
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
        # Iris, rather than whichever persona file was found first
        return PERSONAS.get("iris") or list(PERSONAS.values())[0]
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


def save_persona(settings: dict, persona_id: Optional[str] = None) -> str:
    """Write a persona's settings to its file and load it.

    A built-in persona is saved as the user's own copy, which replaces it; a
    new one (no id) gets a file name from its name.

    Returns:
        The persona's id
    """
    existing = PERSONAS.get(persona_id) if persona_id else None
    if existing is not None and existing.path is not None and not existing.builtin:
        path = existing.path
    else:
        persona_id = persona_id or slug(settings["name"])
        if existing is None:
            base, n = persona_id, 2
            while persona_id in PERSONAS:
                persona_id, n = f"{base}-{n}", n + 1
        path = USER_DIR / f"{persona_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "name": settings["name"].strip(),
        "engine": settings["engine"],
        "voice": settings["voice"],
        "speed": float(settings.get("speed") or 1.0),
    }
    if settings.get("audio_filter"):
        data["audio_filter"] = settings["audio_filter"]
    # Written only where they are the persona's own: a wake word that is its
    # name follows the name when it is renamed
    wake_word = (settings.get("wake_word") or "").strip()
    if wake_word and wake_word.casefold() != data["name"].casefold():
        data["wake_word"] = wake_word
    if sleep_word := (settings.get("sleep_word") or "").strip():
        data["sleep_word"] = sleep_word
    description = settings.get("description", "").strip()
    data["description"] = description + "\n" if description else ""
    with open(path, "w") as f:
        yaml.dump(data, f, Dumper=_PersonaDumper, sort_keys=False, allow_unicode=True)

    reload_personas()
    return path.stem.lower()


def delete_persona(persona_id: str) -> bool:
    """Remove the user's persona file. An edited built-in persona goes back
    to how it was shipped.

    Returns:
        Whether a persona of that id is still there (the built-in one)
    """
    persona = PERSONAS.get(persona_id)
    if persona is None or persona.builtin or persona.path is None:
        return persona is not None
    persona.path.unlink(missing_ok=True)
    reload_personas()
    return persona_id in PERSONAS


def persona_stamp(persona_id: str) -> Optional[tuple]:
    """Which file the persona of this id is read from, and when it last
    changed, to notice it changing on disk; None when no file holds it."""
    found = None
    for persona_dir in get_persona_dirs():
        for path in persona_dir.glob("*.yaml"):
            if path.stem.lower() == persona_id:
                found = path
    if found is None:
        return None
    try:
        stat = found.stat()
    except OSError:
        return None
    return (str(found), stat.st_mtime_ns, stat.st_size)


def find_persona(persona_id: str) -> Optional[Persona]:
    """The persona of this id, or None."""
    return PERSONAS.get(persona_id)


def speaking_with(voice: str) -> list[Persona]:
    """The personas that speak with this voice, as their files say now, by
    name."""
    found = [persona for persona in load_all_personas().values() if persona.voice == voice]
    return sorted(found, key=lambda persona: persona.name.casefold())


def voice_in_use(voice: str) -> bool:
    """Whether a persona speaks with this voice, as its files say now (a
    cloned voice is kept while one does)."""
    return bool(speaking_with(voice))


def voice_users() -> dict[tuple[str, str], list[Persona]]:
    """The personas that speak with each voice, by (engine, voice), as their
    files say now, by name."""
    users: dict[tuple[str, str], list[Persona]] = {}
    for persona in sorted(load_all_personas().values(), key=lambda p: p.name.casefold()):
        users.setdefault((persona.engine, persona.voice), []).append(persona)
    return users


def shipped(persona_id: str) -> bool:
    """Whether Iris ships a persona of this id (which the user may have edited)."""
    return (BUILTIN_DIR / f"{persona_id}.yaml").exists()

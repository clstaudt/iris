"""Iris under test keeps to a folder of its own: the user's personas, memory
and plugins are left alone and stay out of what is tested. The speech models
already on this computer are used."""

import os
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from iris import personas

# Personas only the tests speak with besides Iris, who alone ships: voices
# of Kokoro, through the intercom effect
TEST_PERSONAS = {
    "test-kokoro": {
        "name": "Test Kokoro",
        "engine": "kokoro",
        "voice": "bm_lewis",
        "audio_filter": "intercom",
        "speed": 1.1,
        "description": "You are calm, precise and polite, and speak in a measured manner.\n",
    },
    "test-sarah": {
        "name": "Test Sarah",
        "engine": "kokoro",
        "voice": "af_sarah",
        "audio_filter": "intercom",
        "speed": 1.1,
        "description": "You are matter-of-fact and friendly.\n",
    },
}


def isolate(home: Optional[Path] = None) -> Path:
    """Point Iris's own folders (personas, memory, plugins) into home, a new
    temporary folder if none is given, for this process, with the test
    personas among the personas; returns it."""
    home = Path(home) if home is not None else Path(tempfile.mkdtemp(prefix="iris-voicetest-"))
    home.mkdir(parents=True, exist_ok=True)
    os.environ.update(IRIS_MEMORY_DB=str(home / "memory.db"), IRIS_PLUGINS_DIR=str(home / "plugins"))
    os.environ.pop("IRIS_PERSONAS_DIR", None)
    # Worked out when Iris was imported, from the user's home folder
    personas.USER_DIR = home / "personas"
    personas.USER_DIR.mkdir(exist_ok=True)
    for persona_id, fields in TEST_PERSONAS.items():
        path = personas.USER_DIR / f"{persona_id}.yaml"
        if not path.exists():
            path.write_text(yaml.safe_dump(fields, sort_keys=False, allow_unicode=True))
    personas.reload_personas()
    return home

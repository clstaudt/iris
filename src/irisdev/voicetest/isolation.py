"""Iris under test keeps to a folder of its own: the user's personas, memory
and plugins are left alone and stay out of what is tested. The speech models
already on this computer are used."""

import os
import tempfile
from pathlib import Path
from typing import Optional

from iris import personas


def isolate(home: Optional[Path] = None) -> Path:
    """Point Iris's own folders (personas, memory, plugins) into home, a new
    temporary folder if none is given, for this process; returns it."""
    home = Path(home) if home is not None else Path(tempfile.mkdtemp(prefix="iris-voicetest-"))
    home.mkdir(parents=True, exist_ok=True)
    os.environ.update(IRIS_MEMORY_DB=str(home / "memory.db"), IRIS_PLUGINS_DIR=str(home / "plugins"))
    os.environ.pop("IRIS_PERSONAS_DIR", None)
    # Worked out when Iris was imported, from the user's home folder
    personas.USER_DIR = home / "personas"
    personas.reload_personas()
    return home

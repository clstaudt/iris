"""The plugins Iris ships, and finding the others.

Plugins come from three places, in this order: the ones shipped here, the
entry point group "iris.plugins" of installed packages, and the modules in
the user's plugin folder (~/.iris/plugins, or IRIS_PLUGINS_DIR). A plugin
that fails to load is skipped and reported; Iris starts without it.
"""

import importlib.util
import logging
import os
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path

from iris import engines
from iris.plugin import Plugin
from iris.plugins import camera, memory, screen, web

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "iris.plugins"
USER_DIR = Path.home() / ".iris" / "plugins"

# Shipped with Iris, in the order their tools are offered
BUILT_IN: tuple[Plugin, ...] = (memory.plugin, camera.plugin, screen.plugin, web.plugin)


@dataclass
class Loaded:
    """The plugins found, and what could not be loaded."""

    plugins: list[Plugin]
    #: {"source": where it was found, "error": why it failed}
    failures: list[dict] = field(default_factory=list)


def plugin_dir() -> Path:
    """The user's plugin folder."""
    return Path(os.environ.get("IRIS_PLUGINS_DIR") or USER_DIR).expanduser()


def load(folder: Path | None = None, group: str = ENTRY_POINT_GROUP) -> Loaded:
    """Find the plugins and register the engines they bring.

    Args:
        folder: The user's plugin folder; None for plugin_dir()
        group: Entry point group installed packages declare plugins in
    """
    found: list[tuple[str, Callable[[], object]]] = []
    for entry in entry_points(group=group):
        package = entry.dist.name if entry.dist is not None else entry.name
        found.append((f"package {package}", entry.load))
    folder = folder or plugin_dir()
    if folder.is_dir():
        for path in sorted(folder.glob("*.py")):
            if not path.name.startswith("_"):
                found.append((str(path), lambda path=path: _from_file(path)))

    plugins = list(BUILT_IN)
    failures = []
    for source, get in found:
        try:
            plugin = _as_plugin(get())
            if not plugin.id:
                raise ValueError("it has no id")
            if any(p.id == plugin.id for p in plugins):
                raise ValueError(f"a plugin '{plugin.id}' is already loaded")
            plugin.source = source
            _register_engines(plugin)
        except Exception as e:
            logger.warning("Plugin from %s not loaded: %s", source, e)
            failures.append({"source": source, "error": str(e)})
            continue
        plugins.append(plugin)
    return Loaded(plugins, failures)


def _from_file(path: Path) -> types.ModuleType:
    """Run a plugin module from the user's folder."""
    name = f"iris_user_plugins.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


def _as_plugin(found: object) -> Plugin:
    """The plugin a module, class or object stands for."""
    if isinstance(found, types.ModuleType):
        if not hasattr(found, "plugin"):
            raise TypeError("the module has no 'plugin'")
        found = found.plugin
    if isinstance(found, type) and issubclass(found, Plugin):
        found = found()
    if not isinstance(found, Plugin):
        raise TypeError(f"{found!r} is not an iris.plugin.Plugin")
    return found


def _register_engines(plugin: Plugin):
    """Register the plugin's engines: all of them, or none if one is taken."""
    speech = list(plugin.speech_engines())
    recognition = list(plugin.recognition_engines())
    for engine in speech:
        if engine.id in engines.speech_engines():
            raise ValueError(f"a speech engine '{engine.id}' is already registered")
    for engine in recognition:
        if engine.id in engines.recognition_engines():
            raise ValueError(f"a recognition engine '{engine.id}' is already registered")
    for engine in speech:
        engines.register_speech_engine(engine, source=plugin.id)
    for engine in recognition:
        engines.register_recognition_engine(
            engines.RecognitionEngine(engine.id, engine.name, engine.create, source=plugin.id)
        )

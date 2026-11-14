# Plugins and engines

## Model slots and engines

Speech recognition and speech synthesis are slots, kept in `iris.engines`. An engine registers under an id, and the rest of Iris names it by that id: a persona's `engine`, the Voice and Recognition stages in Settings.

- **Speech synthesis:** `kyutai` (Pocket TTS) and `kokoro`. An engine is a `SpeechEngine` subclass with an `id`, a `name`, the `languages` it speaks, whether it `takes_speed`, and `voices()`, `voice_language()` (the language a voice was made for), `voice_languages()` (those a voice speaks properly; all of the engine's unless it says otherwise) and `generate(text, language)`. No engine speaks for another: a persona's replies in a language its voice does not speak are shown only.
- **Speech recognition:** `whisper`. An engine is a `RecognitionEngine(id, name, create)`, where `create(options)` returns an object whose `transcribe(audio_path)` gives the text and its language. A recognition engine other than Whisper is chosen with the `recognition_engine` option of `assistant.start`; the app has no menu for it yet.

Plugins register engines of their own (see below). Turn detection (Silero and Smart Turn) and echo cancellation are fixed parts of the audio pipeline for now.

## Plugins

A plugin gives Iris tools the language model can call, engines for the model slots, or both. The camera, screen, web search and memory are plugins themselves (`src/iris/plugins/`). Iris loads, in this order, the built-in plugins, those of installed packages (entry point group `iris.plugins`), and the modules in `~/.iris/plugins` (or `IRIS_PLUGINS_DIR`). A plugin that fails to load is logged and reported by `plugins.list`, and Iris starts without it; a plugin whose id is taken is refused.

A plugin module exposes `plugin`, an instance of a `Plugin` subclass. This one, from [`examples/plugins/clock.py`](../examples/plugins/clock.py), lets Iris tell the time; copied into `~/.iris/plugins`, it appears as **Clock** under Settings > Tools:

```python
import time

from iris.plugin import Plugin, Tool


def current_time(**_) -> dict:
    return {"success": True, "result": time.strftime("%A %d %B %Y, %H:%M")}


class Clock(Plugin):
    id = "clock"
    name = "Clock"

    def tools(self, context):
        return [
            Tool(
                name="current_time",
                description="The current local date and time.",
                run=current_time,
                guidance="- 'current_time' function: call it when the user asks for the time or date\n",
                label="Looked at the clock",
            )
        ]


plugin = Clock()
```

- A `Tool` has a `name`, a `description`, `run(**arguments)` returning `{"success": bool, "result" | "image" | "message": ...}`, JSON-schema `parameters` and `required`, `guidance` for the system prompt, a `label` for the conversation view, and optionally an `activity` the orb shows while it runs (`looking`, `capturing_screen`, `searching` or `browsing`) and `detail(arguments)` for the text after the label. When a tool returns an image, the vision model answers, if the server has one.
- A `Plugin` has an `id` (also the key of its switch), a `name`, optionally the modules it `requires` (it shows as not installed while one is missing), `switchable` and `enabled_by_default`. Besides `tools(context)` it can return `speech_engines()` and `recognition_engines()`, which are registered when it loads. `context.memory` is the memory store when memory is on.
- A package declares its plugin as an entry point; the value may name a module with `plugin`, a `Plugin` subclass or an instance:

  ```toml
  [project.entry-points."iris.plugins"]
  clock = "iris_clock:plugin"
  ```

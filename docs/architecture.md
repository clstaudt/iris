# Architecture

Iris is two programs. The Python core (`src/iris`) does everything an assistant does: it listens, recognizes speech, talks to the language model, speaks, runs tools and keeps memory. The desktop app (`ui/`, Electron with Svelte 5, Vite and Tailwind) shows it and passes on what you do. They talk over newline-delimited JSON-RPC 2.0 on stdio: the app starts the core as `python -m iris.rpc_server` (also installed as `iris-gui`), sends requests, and receives the core's activity as notifications. The core's stdout carries only RPC frames; prints, and native libraries writing to it, go to stderr.

| Module | What it does |
| --- | --- |
| `rpc_server.py` | JSON-RPC transport on stdio, and the table of methods |
| `controller.py` | Runs the assistant off the main thread for the app; settings, setup, model downloads, plugin loading |
| `assistant.py` | The conversation loop: listening, turns, replies, wake word, interruptions |
| `audio.py` | Recording, playback and voice effects |
| `turns.py` | Voice activity (Silero) and end of turn (Smart Turn) |
| `echo.py` | Echo cancellation (WebRTC AEC3 through LiveKit) |
| `transcription.py` | Speech recognition with Whisper, through MLX or faster-whisper |
| `wakeword.py` | Whether an utterance is addressed to Iris |
| `llm.py` | The conversation with the language model: context, tool rounds, text made for speaking |
| `tts.py` | Speech synthesis (Pocket TTS, Kokoro) and the pipeline that speaks a reply while it streams |
| `engines.py` | The model slots and the registries of engines that fill them |
| `plugin.py` | The plugin API: `Plugin`, `Tool`, `PluginContext` |
| `plugins/` | The built-in plugins (camera, screen, web, memory) and plugin discovery |
| `tools.py` | The tools of the plugins switched on, as the model sees and calls them |
| `memory.py` | Long-term memory by topic, filed in quiet spells |
| `personas.py` | Personas: loading, saving, deleting |
| `attachments.py` | Files attached to typed messages |
| `downloads.py` | Model downloads from Hugging Face, with progress |
| `events.py` | The event bus that streams activity to the app |
| `status.py` | What the assistant is doing, as flags |

## The RPC protocol

Requests are JSON-RPC 2.0; the core's activity arrives as notifications `{"jsonrpc": "2.0", "method": "event", "params": {"type": ..., "data": ...}}`.

| Area | Methods |
| --- | --- |
| App and setup | `app.info`, `setup.info`, `setup.download`, `models.check`, `models.set`, `mic.check` |
| Assistant | `assistant.start`, `assistant.stop`, `assistant.interrupt`, `assistant.pause`, `assistant.resume`, `assistant.hold`, `assistant.greet` |
| Listening and voice | `mic.set`, `audio.list_inputs`, `audio.set_input`, `wake.set`, `wake.options`, `wake.adopt`, `voice.set_output`, `voice.set_barge_in` |
| Conversation | `chat.send_text`, `chat.history` |
| Language model | `llm.list`, `llm.set`, `llm.set_host`, `llm.check_hosts`, `llm.probe`, `llm.find_servers` |
| Recognition and turn end | `stt.models`, `stt.set`, `turn.set` |
| Memory | `memory.topics`, `memory.topic`, `memory.forget`, `memory.set_amnesia` |
| Personas | `persona.list`, `persona.set`, `persona.get`, `persona.voices`, `persona.save`, `persona.words`, `persona.delete`, `persona.preview` |
| Plugins and engines | `tools.set`, `plugins.list`, `engines.list` |
| Debugging | `debug.emit`, `debug.inject_audio` |

Events: `config` (the loaded models and settings, with `tools`, `engines` and `plugins`), `state`, `flags`, `audio.level`, `audio.output`, `chat.user`, `chat.chunk`, `chat.done`, `chat.continued`, `tool.use`, `wake`, `wake.window`, `sleep`, `setup.progress`, `setup.models`, `mic.check`, `memory.updated`, `turn.timing`, `notice` and `error`. Their payloads are described in `src/iris/events.py`.

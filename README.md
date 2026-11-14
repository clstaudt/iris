<div id="top"></div>

<!-- PROJECT LOGO -->

<br />

<div align="center">

  <a href="#">
    <img width="80%" src="img/iris-title.png" alt="iris - a local AI voice assistant">
  </a>

  <p align="center">
    A modular voice and vision assistant that runs on your own computer
    <br />
    <br />
    <a href="#getting-iris"><strong>Get Iris »</strong></a>
    <br />
    <br />
    <a href="#using-iris">Using Iris</a>
    ·
    <a href="#for-developers">For developers</a>
  </p>

</div>

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-iris">About Iris</a></li>
    <li>
      <a href="#getting-iris">Getting Iris</a>
      <ul>
        <li><a href="#the-macos-app">The macOS app</a></li>
        <li><a href="#from-source">From source</a></li>
      </ul>
    </li>
    <li>
      <a href="#using-iris">Using Iris</a>
      <ul>
        <li><a href="#talking-with-iris">Talking with Iris</a></li>
        <li><a href="#presence-mode">Presence mode</a></li>
        <li><a href="#tools-and-files">Tools and files</a></li>
        <li><a href="#memory">Memory</a></li>
        <li><a href="#personas">Personas</a></li>
        <li><a href="#settings">Settings</a></li>
      </ul>
    </li>
    <li>
      <a href="#for-developers">For developers</a>
      <ul>
        <li><a href="#architecture">Architecture</a></li>
        <li><a href="#model-slots-and-engines">Model slots and engines</a></li>
        <li><a href="#plugins">Plugins</a></li>
        <li><a href="#the-rpc-protocol">The RPC protocol</a></li>
        <li><a href="#development">Development</a></li>
        <li><a href="#building-the-macos-app">Building the macOS app</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT -->

## About Iris

Iris is a voice and vision assistant you talk to as you would to a person. It answers out loud as soon as you have finished speaking, lets you interrupt it, looks through the camera or at your screen when a question needs eyes, reads the files you hand it, and remembers earlier conversations by topic. Speech recognition and speech synthesis run on your computer; the language model runs on Ollama or any server with the OpenAI-compatible API, on this computer or another one you choose.

Iris is used through its desktop app. Underneath, its Python core is built as a kit: each model fills a slot that other engines can fill as well, and capabilities such as the camera or web search are plugins. The app opens on good defaults and shows more of the machinery only when asked, so the same Iris serves someone who just wants to talk and someone who wants to try a new speech model or give the assistant a tool of their own.

What Iris runs by default:

| Part | Default |
| --- | --- |
| Speech recognition | Whisper large-v3-turbo, on the GPU through MLX on Apple Silicon; faster-whisper elsewhere (large-v3-turbo on an NVIDIA GPU, small on a CPU) |
| Turn-taking | Silero voice activity detection and Smart Turn v3, which hears when you have finished |
| Language model | Ollama (e.g. `qwen3:1.7b`), or any OpenAI-compatible server |
| Speech synthesis | Kyutai's Pocket TTS or Kokoro, chosen per persona |
| Echo cancellation | WebRTC's AEC3 through LiveKit, so you can talk over Iris |
| Tools | Camera, screen, web search, memory and going to sleep, as plugins |

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING IRIS -->

## Getting Iris

Iris needs a language model server in any case. The simplest is [Ollama](https://ollama.com) on the same computer, with a small model:

```bash
ollama pull qwen3:1.7b
```

Install Ollama from [ollama.com/download](https://ollama.com/download) (on macOS also `brew install ollama && brew services start ollama`, on Linux `curl -fsSL https://ollama.com/install.sh | sh`). A server on another computer works too, Ollama or OpenAI-compatible: the setup screen asks for its address.

### The macOS app

Iris ships as a self-contained app for Apple Silicon Macs (macOS 14 or later), packed into a disk image ([building it](#building-the-macos-app)). It brings its own Python; the speech models are downloaded at first launch.

1. Have a language model server, as above.
2. Open the disk image and drag **Iris** into **Applications**.
3. Open Iris. Released disk images are signed with a Developer ID and notarized by Apple, so the app opens like any other. A disk image built without a Developer ID is signed ad hoc only: macOS then refuses it the first time ("Apple could not verify..."). Click **Done**, then go to **System Settings > Privacy & Security**, scroll down to the message about Iris, click **Open Anyway**, and confirm. This is needed only once. If macOS instead says the app "is damaged", run `xattr -dr com.apple.quarantine /Applications/Iris.app` in Terminal and open it again.
4. Allow the microphone when asked. The camera and screen recording are asked for the first time Iris uses them.
5. Follow the setup screen. The speech models download once (about 2 GB) into `~/.cache/huggingface` and `~/.iris/models`.

### From source

On any platform, with Python 3.11 to 3.13, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Node.js with npm:

```bash
git clone https://github.com/clstaudt/iris.git
cd iris
uv sync                 # the Python core
cd ui
npm install
npm run dev             # starts Vite, Electron and the Python core
```

The first time it opens, Iris shows a setup screen before anything starts or downloads. Conversations need a large language model: setup looks for Ollama on this computer, or on any server you type, and offers its models, preferring `qwen3:1.7b`. It then asks where the speech models come from: downloaded (Whisper for recognition, Kokoro for the voice, about 2 GB together; models already downloaded are reused), or folders you already have. Kyutai's Pocket TTS, the default voice, fetches its model from Hugging Face the first time it loads. An iris bud at the top opens as setup comes along, a little when the language model is ready and further with each speech model in place; downloads show a thin progress bar. When the models are in, the setup screen gives way to the conversation and the flower blooms into the orb.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USING IRIS -->

## Using Iris

The window is a conversation: your messages and Iris's replies appear as bubbles, replies render markdown, and each of your messages is marked as spoken or typed. Below the message box sits the orb, which shows what Iris is doing through color and motion. It swells softly with your voice while listening and with its own voice while speaking, turns into a slowly rotating spinner while transcribing or thinking, and opens a camera aperture while looking through the camera or at the screen.

### Talking with Iris

Iris starts listening as soon as the models are loaded. Speaking and typing share one conversation.

- Iris answers as soon as you have finished, not after a fixed silence. A voice activity detector (Silero) hears when you pause, and [Smart Turn](https://github.com/pipecat-ai/smart-turn), a small model from the Pipecat project, judges from your words and intonation whether you have finished or are only thinking. Should you go on after it took a pause for the end, the answer begun is dropped and your whole turn is answered. Without Smart Turn (offline on the first run) a turn ends after 3.5 seconds of silence.
- Iris starts speaking as soon as the first sentence of its reply is there.
- Speak while Iris is talking to interrupt it: the voice pauses as soon as you start, and stops if you go on. Iris keeps what it got to say, as far as you heard it, and answers what you said. An echo canceller takes Iris's own voice out of what the microphone hears; it first learns the room from a few seconds of the voice, and a single short word said over loud speakers may not get through. Should the voice pause too often with your speakers, switch **Interrupt by voice** off under Settings > Pipeline > Output; the microphone then closes while Iris speaks.
- Typed and spoken messages are answered one at a time, in order. A message sent while Iris is still answering waits as a queued bubble. While you type, the microphone stays closed, and typing anywhere in the window goes to the message box.
- **Pause** (or a click on the orb) holds the whole conversation: speech stops mid-sentence, the reply stops streaming, the microphone closes, and queued messages wait. **Resume** picks up where it stopped. **Mic** switches listening off while still answering typed messages; **Voice** switches spoken replies off for a text-only chat; **Stop** cancels the current reply.
- **Wake word** (Settings): Iris sleeps until addressed by name. "Iris, what's the weather?" wakes it and is answered at once; a bare "Iris" plays a soft chime and waits for the request. Follow-ups need no name. Iris stays awake while you talk and for a while after each reply (**Stay awake**, 30 seconds by default); over the last 10 seconds the orb drifts toward its sleeping tone, then Iris goes back to sleep with a soft falling chime. Saying the **sleep word** ("That's all, Iris" by default) sends it to sleep right away. Iris also goes to sleep by itself once you are clearly done, when you say goodbye or ask it to stop listening: it says its goodbye, and falls asleep once the voice has died away, so its own words cannot wake it (the **Going to sleep** tool, offered only while a wake word is set). While asleep, only speech reaches the smallest Whisper model, and the wake word only counts at the start of an utterance, so "tell Iris I said hi" doesn't wake it.

### Presence mode

For using Iris by voice and vision alone, the shrink button beside Memory and Settings turns the window into presence mode: a small window with only the orb, floating above other windows. Drag it anywhere; a click on the orb still pauses and resumes. Escape, or the button that appears at the top right when you hover, brings the conversation back. When something needs your attention, such as a microphone that delivers only silence, the message appears under the orb, and a click on it opens the conversation. Iris remembers the mode and where the small window was.

### Tools and files

The camera, screen capture, web search and going to sleep are on by default and can be switched off under Settings > Pipeline > Tools; a tool whose optional dependencies are missing shows as not installed. Iris decides itself when to use them:

- **Camera:** "What am I holding?", "Read the text on this paper".
- **Screen:** "What's on my screen right now?", "Can you read the code I have open?" On macOS, screen recording must be allowed for Iris (in development, for the app that started it) in **System Settings > Privacy & Security > Screen Recording**.
- **Web search:** current facts, news, and pages it should read.
- **Going to sleep:** "Thanks, that's all for now", "Good night". Iris says goodbye and sleeps until called by its wake word; it is offered only while a wake word is set.

Images, from the camera, the screen or an attachment, go to a model that can see, such as `qwen3-vl:8b`, when the server has one.

Typed messages can carry files: images, PDFs and text files, added with the paperclip beside Send, by pasting, or by dropping them anywhere in the window. A message holds up to 10 files of up to 25 MB each. They stay in the model's context for six turns, so follow-up questions work, and only their names are kept in memory.

### Memory

Iris keeps one continuous conversation that survives restarts and is remembered by topic in `~/.iris/memory.db`. The model always sees the recent turns plus a one-line index of every topic it remembers. When you come back to something, it decides itself to `recall` that topic or to `search_memory`, a full-text search over everything said; there are no embeddings and no similarity thresholds. Conversations are sorted into topics by the same model, but only during quiet spells: filing gives way the moment you speak or type, so replies never wait for it. The **Memory** panel (top right) lists the topics with their summaries and conversations, and can forget a topic. **Amnesia** in the message box keeps the current conversation out of memory: Iris follows it as usual, but writes nothing, and forgets it entirely when amnesia is switched off.

### Personas

A persona is who Iris is: a name, a personality, and a voice. Iris ships three, chosen under Settings > Personas:

- **Iris** (default): friendly and helpful.
- **HAL**: inspired by HAL 9000, calm and mission-focused.
- **Computer**: matter-of-fact, like the USS Enterprise computer.

There you edit a persona or create one: name, personality and voice, with every voice playable, the speech engine, the effect, and the speed where the engine takes one. Edits are saved to `~/.iris/personas`; an edited built-in persona can be reset. Persona files can also be written by hand, in `~/.iris/personas` (or `IRIS_PERSONAS_DIR`):

```yaml
name: "HAL"
engine: "kokoro"      # or "kyutai"
voice: "bm_lewis"     # a voice of that engine
speed: 1.1            # for engines whose voices take a speed (Kokoro)
audio_filter: "intercom"
description: |
  You have the personality and speaking style of HAL 9000.
```

Kyutai's Pocket TTS sounds the most natural and speaks English, French, German, Italian, Spanish, Portuguese and Dutch; replies in Hindi, Japanese or Chinese fall back to Kokoro. Kokoro speaks English, French, Italian, Spanish, Portuguese, Hindi, Japanese and Chinese; a Kokoro persona's replies in German or Dutch fall back to Pocket TTS. A persona speaks English in its own voice, and another language in its own voice when that voice was made for it, in a native voice otherwise. Each part of a reply is spoken in the language it is written in, told from its words, so an Italian answer to an English question is not read with an English accent. A Kyutai voice can also be the path to a short audio file of a voice to clone.

### Settings

Settings (top right) open as a window of their own, in two pages. All choices are remembered between launches.

- **Pipeline** shows how a turn travels through Iris, as one line from top to bottom: what it hears (microphone, wake word, turn end, recognition), how it thinks (the language model, with the persona, tools and memory it draws on) and how it speaks (the voice and the output). Click a stage to change it in the inspector beside the line: the recognition model, Smart Turn or a silence for the turn end, the model server and model, the tool switches, and so on.
  Beside each timed stage is how long it took in the latest turn (turn end, recognition, the model's first sentence, the voice starting), on one scale, and below the line the time from the end of your speech to Iris's first word. A stage changed since shows no time until the next turn measures it.
- **Personas** holds who Iris is: each persona's name, personality and voice, with every voice playable before you pick it.

A **setup** (top of the window) is a named set of choices for how Iris works: the model server and model, the recognition model, the turn end and the tools. Changes go into the setup in use; make a new one to keep the current choices under another name, and switch between setups to compare them.

Two stages in more detail:

- **Model server** (Language model): Ollama, or any server with the OpenAI-compatible API (vLLM, llama.cpp, LM Studio, MLX servers, LiteLLM, OpenAI). Empty uses `OLLAMA_HOST` (or `http://localhost:11434`). Iris asks the server what it is: Ollama, with or without `/v1` in the URL, is used through its own API, which tells each model's context length and what it can do; any other server through its OpenAI-compatible API at `.../v1`, where each model is taken to support tools unless the server refuses them. A server that needs a key gets it from the `IRIS_LLM_API_KEY` environment variable. The server is checked before Iris switches to it.
- **Model folders** (Recognition and Voice): point Iris at speech models you already have instead of downloading them. Recognition takes a Whisper folder, either an MLX conversion (`weights.safetensors` or `weights.npz`) or a faster-whisper model (`model.bin`); Voice takes a folder holding `kokoro-v1.0.onnx` and `voices-v1.0.bin`. A folder without them is refused at once. **Use a download** goes back to the downloaded models.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- FOR DEVELOPERS -->

## For developers

### Architecture

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

### Model slots and engines

Speech recognition and speech synthesis are slots, kept in `iris.engines`. An engine registers under an id, and the rest of Iris names it by that id: a persona's `engine`, the Voice and Recognition stages in Settings.

- **Speech synthesis:** `kyutai` (Pocket TTS) and `kokoro`. An engine is a `SpeechEngine` subclass with an `id`, a `name`, the `languages` it speaks, whether it `takes_speed`, the engines it `borrows` voices from for languages it lacks (Kyutai borrows Kokoro), and `voices()`, `voice_language()`, `default_voice()` and `generate(text, language)`.
- **Speech recognition:** `whisper`. An engine is a `RecognitionEngine(id, name, create)`, where `create(options)` returns an object whose `transcribe(audio_path)` gives the text and its language. A recognition engine other than Whisper is chosen with the `recognition_engine` option of `assistant.start`; the app has no menu for it yet.

Plugins register engines of their own (see below). Turn detection (Silero and Smart Turn) and echo cancellation are fixed parts of the audio pipeline for now.

### Plugins

A plugin gives Iris tools the language model can call, engines for the model slots, or both. The camera, screen, web search and memory are plugins themselves (`src/iris/plugins/`). Iris loads, in this order, the built-in plugins, those of installed packages (entry point group `iris.plugins`), and the modules in `~/.iris/plugins` (or `IRIS_PLUGINS_DIR`). A plugin that fails to load is logged and reported by `plugins.list`, and Iris starts without it; a plugin whose id is taken is refused.

A plugin module exposes `plugin`, an instance of a `Plugin` subclass. This one, from [`examples/plugins/clock.py`](examples/plugins/clock.py), lets Iris tell the time; copied into `~/.iris/plugins`, it appears as **Clock** under Settings > Tools:

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

### The RPC protocol

Requests are JSON-RPC 2.0; the core's activity arrives as notifications `{"jsonrpc": "2.0", "method": "event", "params": {"type": ..., "data": ...}}`.

| Area | Methods |
| --- | --- |
| App and setup | `app.info`, `setup.info`, `models.check`, `models.set` |
| Assistant | `assistant.start`, `assistant.stop`, `assistant.interrupt`, `assistant.pause`, `assistant.resume`, `assistant.hold` |
| Listening and voice | `mic.set`, `audio.list_inputs`, `audio.set_input`, `wake.set`, `wake.options`, `voice.set_output`, `voice.set_barge_in` |
| Conversation | `chat.send_text`, `chat.history` |
| Language model | `llm.list`, `llm.set`, `llm.set_host` |
| Recognition and turn end | `stt.models`, `stt.set`, `turn.set` |
| Memory | `memory.topics`, `memory.topic`, `memory.forget`, `memory.set_amnesia` |
| Personas | `persona.list`, `persona.set`, `persona.get`, `persona.voices`, `persona.save`, `persona.delete`, `persona.preview` |
| Plugins and engines | `tools.set`, `plugins.list`, `engines.list` |
| Debugging | `debug.emit`, `debug.inject_audio` |

Events: `config` (the loaded models and settings, with `tools`, `engines` and `plugins`), `state`, `flags`, `audio.level`, `audio.output`, `chat.user`, `chat.chunk`, `chat.done`, `chat.continued`, `tool.use`, `wake`, `wake.window`, `sleep`, `setup.progress`, `memory.updated`, `turn.timing` and `error`. Their payloads are described in `src/iris/events.py`.

### Development

```bash
uv run pytest                  # unit tests
uv run pytest -m integration   # integration tests, which need Ollama, Kokoro and Whisper
```

The setup screen shows only once per settings folder. To go through it again as a new user:

```bash
cd ui
IRIS_SETUP=1 npm run dev          # fresh settings and memory; the speech models already on this computer
IRIS_SETUP=download npm run dev   # also hides the models, so they download (about 2 GB)
```

Everything a test run writes, downloaded models included, goes into one temporary folder (`$TMPDIR/iris-setup-test`), which is removed when Iris quits. Your own settings, memory and models are left alone. The built app takes the same switch: `IRIS_SETUP=download /Applications/Iris.app/Contents/MacOS/Iris`.

The app's interface is tested end to end against the real Python core, in a headless Chrome instead of Electron:

```bash
cd ui
npm run build && npm run e2e      # E2E_MODEL=<model> and E2E_HOST=<server URL> pick the language model
```

The test opens every stage of the pipeline, switches models, turn detection and tools, plays voices, edits personas, switches setups, has a typed conversation and stops replies. Its core has a home folder of its own, so your personas, memory and settings are left alone, and it sleeps until a wake word nobody says. It does speak: the voices it plays are heard. Screenshots of failed steps and the core's log go to `ui/e2e/output`.

Other environment variables: `OLLAMA_HOST` (the model server when none is set in the app), `IRIS_LLM_API_KEY` (a key for servers that need one), `IRIS_MEMORY_DB`, `IRIS_PERSONAS_DIR`, `IRIS_PLUGINS_DIR`, and `IRIS_MODELS_DIR` (where Kokoro's files are looked for).

The app icon is the fully open iris of the startup animation, drawn by the same code (`ui/src/lib/iris.ts`); `npm run icon` re-renders `ui/build/icon.png` and, on macOS, `icon.icns`. `iris-benchmark` compares kokoro-onnx with the kokoro-tts command line.

> **Cursor and VS Code terminals:** if Electron exits immediately with `Cannot set properties of undefined`, your terminal exports `ELECTRON_RUN_AS_NODE=1`. The `dev` script already clears it; when launching `electron .` manually, prefix it with `ELECTRON_RUN_AS_NODE=`.

**The microphone in development on macOS.** If the orb never reacts to your voice, macOS is returning pure silence instead of audio, the symptom of a missing microphone grant. In development the grant does not belong to Electron: macOS assigns it to the app that started the dev server (Cursor, VS Code, Terminal or iTerm), which is also why "Electron" may not appear in the Privacy list at all. Check what your shell captures:

```bash
.venv/bin/python -c "import sounddevice as sd, numpy as np; r = sd.rec(int(3*16000), samplerate=16000, channels=1); sd.wait(); print('max amplitude:', float(np.abs(r).max()))"
```

A working microphone reports a non-zero value (a quiet room still has a noise floor); exactly `0.0` means capture is being denied. Enable the microphone for the launching app under **System Settings > Privacy & Security > Microphone**, then quit and reopen that app (a reload is not enough). The app says so itself: a red line appears in the conversation when the microphone delivers only silence for a few seconds.

### Testing the voice

`irisdev.voicetest` tests Iris by talking to it. Iris runs as in the app, with its real recognition, Smart Turn, echo canceller, language model and voice, but its microphone and speaker belong to a virtual acoustic space: what the microphone hears is scripted, the speaker records what it plays, and each speaker comes back into its own microphone as an echo and is heard by the others. Nothing is played on the computer's speakers, and nothing in the room reaches the test. Iris keeps its personas, memory and plugins in a folder of its own, so yours are left alone.

```bash
# Scripted conversations: a synthesized user talks to Iris, turn by turn
uv run python -m irisdev.voicetest conversations
uv run python -m irisdev.voicetest conversations --group search
uv run python -m irisdev.voicetest conversations --group hearing --only babble --noise fan:15
uv run python -m irisdev.voicetest conversations --list
uv run python -m irisdev.voicetest --llm-host <server URL> --llm-model <model> conversations --only "follow-up"

# Self-talk: two Irises (Iris and HAL) talk to each other, unscripted
uv run python -m irisdev.voicetest --llm-host <server URL> --llm-model <model> selftalk --minutes 5

# The same conversations as pytest tests
uv run pytest -m voice            # the default groups
uv run pytest -m voice_slow       # Whisper's sizes, a session of 60 turns
IRIS_VOICETEST_WEB=real uv run pytest -m voice_network   # web search over the real network
```

- **Conversations** are held in turns and grouped by what they test: `basic` (a question and a follow-up, a pause in the middle of a sentence, interrupting a long answer, Stop, the wake word...), `search` (web search: the lead-in heard while the search runs, Stop and barge-in at each phase of a search, talking while Iris searches, corrections and follow-ups, searches that fail or hang, web search switched on and off, reading a page), `turn-taking`, `interruptions`, `hearing` (noise, distance, loud speakers, other voices, other languages), `conversation` (typed and spoken turns, waking and sleeping, switching persona), `performance` (a slow model, a context that fills up) and `model` (a model server in trouble: one that stops answering mid-reply, a model that writes its tool calls into the reply as text, a model without tool support). Each turn is checked (heard as said, as one turn, answered aloud, the tools it should use, a reply that fits) and timed from the end of the user's speech: to the first word heard, without the model's time (the voice pipeline), to silence when interrupted or stopped.
- A turn is said at a moment (`script.At`): once Iris has finished, over its reply, at the reply's last syllable, in a pause between its sentences, while it searches, after an event. Actions (Stop, pause, the microphone, a persona or a tool switched) and asides (a back-channel, a cough, a door) happen at moments of their own.
- Without a server a scripted model answers (`StubModel`, Ollama's API on a local port), which tests the voice without a model's time. Its rules make it call tools as a model with tools would: a search, reading a page, with something said first. Searches go to a scripted web (`FakeWeb`) unless `--web real` (or `IRIS_VOICETEST_WEB=real`) is given. Facts that come from the scripted model or web are checked in replies; other replies are checked only with a real model.
- Checks known to fail today are listed with each conversation and reported as expected failures (pytest xfail); one that passes is reported as such.
- **Self-talk** reports how much of each reply the other heard, whether it took a turn in the pauses between sentences, how long each took to answer, how long both spoke at once, and whether a voice cut itself off.
- Each run writes `report.json` (with medians, 90th percentiles and worst cases per stage, by tag) and, per conversation, what each microphone heard and each speaker played, as WAV files to listen to.
- Budgets (`voice_pipeline`, `barge_in`, `stop`, `turn_end`, `search_added`, `new_language`) are set with `--budgets voice_pipeline=3.5,barge_in=1` or `IRIS_VOICETEST_BUDGETS`.
- The parts are reusable: `Space` and its endpoints are audio devices as Iris takes them (`iris.audio.SoundDevice`), `ScriptedConversations` and `SelfTalk` talk to any `AssistantUnderTest`, and `IrisUnderTest` is Iris's.

### Building the macOS app

```bash
cd ui
npm run dist:mac        # -> ui/release/Iris-<version>-macOS-arm64.dmg
```

`ui/scripts/package-mac.sh` does three things:

1. It freezes the Python core with PyInstaller (`packaging/iris-rpc.spec`) into a standalone `iris-rpc` folder. The core's own Python, from a separate build environment in `packaging/.venv`, comes along, so the recipient needs no Python. `iris-rpc --check` then imports every engine inside the bundle (speech recognition on MLX and on the CPU, voice activity detection, turn detection, echo cancellation, Pocket TTS and Kokoro, the voice itself, OpenAI-compatible model servers, web search, web pages, camera, screen capture, audio devices) and stops the build if one is missing.
2. It builds the UI.
3. It runs electron-builder, which puts the core into the app's `Resources`, signs the app under the hardened runtime (`ui/build/entitlements.mac.plist`) and writes the `.dmg`.

What is not in the app: the speech models, which are downloaded at first launch (or taken from folders the recipient points to), and the language model server.

**Signing and notarizing.** With a Developer ID Application certificate the app is signed with it: one in the login keychain is found by itself, or give it as `CSC_LINK` (a base64-encoded `.p12`, or its path) and `CSC_KEY_PASSWORD`. It is then notarized by Apple when `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD` (an app-specific password of that Apple ID) and `APPLE_TEAM_ID` are set; electron-builder also takes a stored `notarytool` profile as `APPLE_KEYCHAIN_PROFILE`. Without a certificate, or with `CSC_IDENTITY_AUTO_DISCOVERY=false`, the app is signed ad hoc and not notarized.

**In CI.** `.github/workflows/pack-electron.yml` builds the disk image on a macOS runner for every push to `main` (signed ad hoc; its disk image replaces the one in the rolling "latest" pre-release, so the newest build can always be downloaded) and for every tag `v*` (signed with the Developer ID and notarized, using the repository secrets `CSC_LINK`, `CSC_KEY_PASSWORD`, `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD` and `APPLE_TEAM_ID`). A tag's disk image is uploaded to its GitHub release; a draft release is made if there is none. To release, set the version in `ui/package.json` and push a matching tag, e.g. `v0.2.0`.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- [Whisper](https://github.com/openai/whisper): OpenAI's speech recognition model
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper): Whisper on Apple Silicon GPUs
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): CTranslate2-based Whisper implementation, which also ships the Silero voice activity detector
- [Silero VAD](https://github.com/snakers4/silero-vad): voice activity detection
- [Smart Turn](https://github.com/pipecat-ai/smart-turn): Pipecat's end-of-turn model
- [LiveKit](https://github.com/livekit/python-sdks): WebRTC's echo canceller (AEC3) for Python
- [Ollama](https://ollama.com): local language model server
- [Pocket TTS](https://github.com/kyutai-labs/pocket-tts): Kyutai's natural-sounding text-to-speech that runs on the CPU
- [Kokoro TTS](https://github.com/nazdridoy/kokoro-tts) and [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx): text-to-speech with many voices
- [ddgs](https://github.com/deedy5/ddgs) and [trafilatura](https://github.com/adbar/trafilatura): web search and reading pages
- [pypdf](https://github.com/py-pdf/pypdf): text from PDFs

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->

## License

Copyright (C) 2025–2026 Christian Staudt

Iris is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Iris is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License in [LICENSE](LICENSE) for more details.

<p align="right">(<a href="#top">back to top</a>)</p>

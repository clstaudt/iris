<div id="top"></div>

<!-- PROJECT LOGO -->

<br />

<div align="center">

  <a href="#">
    <img width="80%" src="img/iris-title.png" alt="iris - a local AI voice assistant">
  </a>

  <p align="center">
    A fully local voice assistant powered by Whisper, Ollama, and Kyutai and Kokoro TTS
    <br />
    <br />
    <a href="#getting-started"><strong>Get Started »</strong></a>
    <br />
    <br />
    <a href="#usage">Usage</a>
  </p>

</div>

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#example-queries">Example Queries</a></li>
        <li><a href="#personas">Personas</a></li>
        <li><a href="#vision-capabilities">Vision Capabilities</a></li>
        <li><a href="#screen-capture">Screen Capture</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

Iris is a voice-controlled AI assistant that runs entirely on your local machine. Talk naturally and get spoken responses—all without sending data to external servers.

Built with:
- **Speech-to-Text**: OpenAI's Whisper
- **Language Model**: Ollama (any compatible model)
- **Text-to-Speech**: Kyutai's Pocket TTS or Kokoro TTS, chosen per persona

<p align="right">(<a href="#top">back to top</a>)</p>

### Key Features

- 🎤 **Voice Input** — Just speak naturally
- 🧠 **Local LLM** — Powered by Ollama, keeping all conversations private
- 🗣️ **Natural Speech** — High-quality text-to-speech
- ⚡ **Fast Response** — Natural conversation flow
- 🔄 **Conversation Memory** — Maintains context throughout the conversation
- 👁️ **Vision Support** — Can see through your webcam
- 🖥️ **Screen Capture** — Can see what's on your screen
- 🔒 **Fully Offline** — No internet required after setup

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

1. **Python 3.11+** and **uv** — [Install uv](https://docs.astral.sh/uv/getting-started/installation/), and **Node.js** with npm for the desktop app

2. **Ollama** — For local LLM inference

   **macOS:**
   ```bash
   brew install ollama
   brew services start ollama
   ollama pull qwen3:1.7b
   ```

   **Linux:**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve &
   ollama pull qwen3:1.7b
   ```

   **Windows:**
   
   Download and install from [ollama.com/download](https://ollama.com/download). Then open a terminal and run:
   ```bash
   ollama pull qwen3:1.7b
   ```

Feel free to experiment with different Ollama models! Try `llama3:8b`, `mistral`, `phi3`, or any model that fits your hardware.

### Installation

```bash
git clone https://github.com/clstaudt/iris.git
cd iris
uv sync
cd ui
npm install
npm run dev
```

On first launch the app shows a setup screen: it finds Ollama and its models, and downloads the speech models (see [Desktop UI](#desktop-ui-electron-shell-development-mode)).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE -->

## Usage

Iris is used through the desktop app: speak to it or type, and switch tools, personas and models in its settings.

### Example Queries

Try asking Iris:
- "Tell me a joke about programming"
- "Explain what recursion is in simple terms"
- "What are the key differences between Python and JavaScript?"
- "Help me brainstorm names for a new project"
- "Walk me through how to solve a merge conflict in Git"

### Personas

Choose a persona in the settings:

- **iris** (default) — Friendly and helpful assistant
- **hal** — Inspired by HAL 9000, calm and mission-focused. Pursues objectives you give it.
- **computer** — Matter-of-fact, like the USS Enterprise computer

Personas can be customized by editing the config files in `src/iris/personas/`, or in the desktop app. Each persona picks a speech engine and one of its voices:

```yaml
name: "HAL"
engine: "kokoro"      # or "kyutai"
voice: "bm_lewis"     # a voice of that engine
speed: 1.1            # Kokoro voices only
audio_filter: "intercom"
description: |
  You have the personality and speaking style of HAL 9000.
```

Kyutai's Pocket TTS sounds the most natural and speaks English, French, German, Italian, Spanish, Portuguese and Dutch; replies in Japanese or Chinese fall back to Kokoro. A Kyutai voice can also be a path to a short audio file of a voice to clone.

### Vision Capabilities

Iris can see through your webcam. The camera tool is on by default; images go to a vision model the Ollama server has, such as `qwen3-vl:8b` (`ollama pull qwen3-vl:8b`).

**Try asking:**
- "What am I holding?"
- "What color is this?"
- "Can you see what's on my desk?"
- "Read the text on this paper"

### Screen Capture

Iris can see what's displayed on your monitor. This uses the same vision model as the camera and requires screen recording permission on macOS.

**Try asking:**
- "What's on my screen right now?"
- "Can you read the code I have open?"
- "What application am I using?"
- "Describe what you see on my display"

> **macOS note:** Screen recording permission must be granted to Iris (in development, to the app that started it) in **System Settings > Privacy & Security > Screen Recording**. Restart it after granting permission.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- DESKTOP UI -->

## Desktop UI (Electron shell, development mode)

An experimental desktop UI lives in `ui/`: an Electron shell (Svelte 5 + Vite + Tailwind) that wraps the Python core. The assistant logic stays in Python; Electron is a thin presentation layer talking to it over newline-delimited JSON-RPC on stdio (`python -m iris.rpc_server`, also installed as the `iris-gui` entry point).

The window is a chat: your messages and Iris's replies appear as bubbles, replies render markdown, and each of your messages is marked as spoken or typed. Below the message box sits the orb, which shows what Iris is doing through color and motion: it swells softly with your voice while listening and with its own voice while speaking, turns into a slowly rotating spinner while transcribing or thinking, and opens a camera aperture while looking through the camera or at the screen. While the models load at startup, an iris flower unfolds and then turns into the orb.

**Run it:**

```bash
uv sync                 # Python core (once)
cd ui
npm install             # once
npm run dev             # starts Vite + Electron + the Python core
```

The app icon is the fully open iris of that startup animation, drawn by the same code (`ui/src/lib/iris.ts`). `npm run icon` re-renders `ui/build/icon.png` and, on macOS, `icon.icns`.

**First launch.** The first time it opens, Iris shows a setup screen before anything starts or downloads. Conversations need a large language model, served by Ollama: setup looks for Ollama on this computer, or on any server you type, and offers its models, preferring `qwen3:1.7b`. It then asks where the speech models come from: downloaded (Whisper for recognition, Kokoro for the voice, about 2 GB together; models already downloaded are reused), or folders you already have. An iris bud at the top opens as setup comes along: a little when the language model is ready, further with each speech model in place, and with the downloads, which show a thin progress bar. When the models are in, the setup screen gives way to the conversation and the flower carries on blooming into the orb.

**Testing the setup.** The setup screen shows only once per settings folder. To go through it again as a new user:

```bash
IRIS_SETUP=1 npm run dev          # fresh settings and memory; the speech models already on this Mac
IRIS_SETUP=download npm run dev   # also hides the models, so they download (about 2 GB)
```

Everything a test run writes, downloaded models included, goes into one temporary folder (`$TMPDIR/iris-setup-test`), which is removed when Iris quits. Your own settings, memory and models are left alone. The built app takes the same switch: `IRIS_SETUP=download /Applications/Iris.app/Contents/MacOS/Iris`.

Iris starts listening as soon as the models are loaded. Speaking and typing share one conversation:

- Iris answers as soon as you have finished, not after a fixed silence. A voice-activity detector (Silero) hears when you pause, and [Smart Turn](https://github.com/pipecat-ai/smart-turn), a small model from the Pipecat project, judges from your words and intonation whether you have finished or are only thinking. Should you go on after it took a pause for the end, the answer begun is dropped and your whole turn is answered. Without Smart Turn (offline on the first run) a turn ends after 3.5 seconds of silence.
- Speak while Iris is talking to interrupt it, as you would a person: the voice pauses as soon as you start, and stops if you go on. Iris remembers what it got to say and answers what you said. An echo canceller (WebRTC's, through LiveKit) takes Iris's own voice out of what the microphone hears; what little of it gets through can pause the voice too, but fades with it, and the voice goes on. The canceller first learns the room from a few seconds of the voice, and a single short word said over loud speakers may not get through; turning them down helps. Should pauses happen too often with your speakers, switch **Interrupt by voice** off in Settings; the microphone then closes while Iris speaks.
- Typed and spoken messages are answered one at a time, in order. A message sent while Iris is still answering waits as a queued bubble.
- While you type, the microphone stays closed, so keyboard noise never becomes a spoken turn. Typing anywhere in the window goes to the message box.
- **Pause** (or a click on the orb) holds the whole conversation: speech stops mid-sentence, the reply stops streaming, the microphone closes, and queued messages wait. **Resume** picks up where it stopped.
- **Mic** switches listening off while still answering typed messages; **Voice** switches spoken replies off for a text-only chat; **Stop** cancels the current reply.
- **Wake word** (Settings): Iris sleeps until addressed by name. "Iris, what's the weather?" wakes it and is answered at once; a bare "Iris" plays a soft chime and waits for the request. Follow-ups need no name. Iris stays awake while you talk and for a while after each reply (**Stay awake**, 30 seconds by default). Over the last 10 seconds the orb visibly drifts toward its sleeping tone, then it goes back to sleep with a soft falling chime. Saying the **sleep word** ("That's all, Iris" by default) sends it to sleep right away. While asleep, a voice-activity detector screens out everything that isn't speech, and only speech reaches the smallest Whisper model: about 30–40 ms per utterance on the GPU of an Apple Silicon Mac. The wake word only counts at the start of an utterance, so "tell Iris I said hi" doesn't wake it.

**Memory.** Iris keeps one continuous conversation that survives restarts and is remembered by topic in `~/.iris/memory.db`. The model always sees the recent turns plus a one-line index of every topic it remembers. When you come back to something, it decides itself to `recall` that topic or to `search_memory`, a full-text search over everything said; there are no embeddings and no similarity thresholds. Conversations are sorted into topics by the same model, but only during quiet spells. Filing gives way the moment you speak or type, and a filing call already in flight is cut off, so replies never wait for it. The memory panel (clock icon, top right) lists the topics with their summaries and conversations, and can forget a topic. **Amnesia** in the message box keeps the current conversation out of memory: Iris follows it as usual, but writes nothing, and forgets it entirely when amnesia is switched off.

The settings panel (top right) holds the persona, the Ollama server, the language model, switches for the camera, screen capture and web search tools, and the microphone. All choices are remembered between launches.

- **Personas:** besides the built-in ones, YAML files in `~/.iris/personas` appear in the menu. Each persona speaks with its own voice, except in languages that voice cannot speak. The pencil beside the menu edits a persona (name, speech engine, voice, speed, effect and personality, with a button to hear the voice), the plus creates one. Edits are saved to `~/.iris/personas`; an edited built-in persona can be reset.
- **Ollama server:** empty uses `OLLAMA_HOST` (or `http://localhost:11434`). An OpenAI-style URL ending in `/v1` works too. The server is checked before Iris switches to it.
- **Models:** the menu lists what the server has installed; switching keeps the conversation history.
- **Speech recognition:** on Apple Silicon Macs, Whisper runs on the GPU through MLX with large-v3-turbo, which is several times faster than CPU decoding and leaves the CPU idle. Elsewhere faster-whisper runs, with large-v3-turbo on an NVIDIA GPU and small on a CPU.
- **Tools:** the camera, screen capture and web search tools are on by default. A tool whose optional dependencies are missing is shown as not installed. Image prompts are routed to a vision model when the server has one.
- **Model folders** (under Advanced): point Iris at speech models you already have instead of downloading them. Recognition takes a Whisper folder, either an MLX conversion (`weights.safetensors` or `weights.npz`) or a faster-whisper model (`model.bin`); Voice takes a folder holding `kokoro-v1.0.onnx` and `voices-v1.0.bin`. A folder without them is refused at once. **Download** goes back to the downloaded models.

The RPC protocol: requests are JSON-RPC 2.0 (`app.info`, `setup.info`, `assistant.start/stop/interrupt/pause/resume/hold`, `mic.set`, `wake.set/options`, `voice.set_output`, `chat.send_text/history`, `memory.topics/topic/forget/set_amnesia`, `persona.list/set/get/voices/save/delete/preview`, `llm.list/set/set_host`, `models.check/set`, `tools.set`, `audio.list_inputs/set_input`); the Python core pushes events as notifications (`config`, `state`, `audio.level`, `audio.output`, `chat.user/chunk/done`, `tool.use`, `wake`, `wake.window`, `sleep`, `setup.progress`, `memory.updated`, `error`).

> **Note for Cursor/VSCode terminals:** if Electron exits immediately with `Cannot set properties of undefined`, your terminal exports `ELECTRON_RUN_AS_NODE=1`. The `dev` script already clears it; when launching `electron .` manually, prefix it with `ELECTRON_RUN_AS_NODE=`.

### macOS microphone permission in dev mode

If the waveform stays flat and the orb never reacts, macOS is returning pure silence instead of audio — the symptom of a missing microphone grant. In development the grant does **not** belong to Electron: macOS assigns it to the *responsible process*, which is the app that started the dev server (Cursor, VS Code, Terminal or iTerm). That is also why "Electron" may not appear in the Privacy list at all.

Check what your shell actually captures:

```bash
.venv/bin/python -c "import sounddevice as sd, numpy as np; r = sd.rec(int(3*16000), samplerate=16000, channels=1); sd.wait(); print('max amplitude:', float(np.abs(r).max()))"
```

A working microphone reports a non-zero value (a quiet room still has a noise floor). Exactly `0.0` means capture is being denied. Fix it by enabling the microphone for your launching app under **System Settings > Privacy & Security > Microphone**, then quitting and reopening that app (a reload is not enough). Running `npm run dev` from a terminal that already has the grant also resolves it.

The app surfaces this itself: a red line appears in the transcript when the microphone delivers only silence for a few seconds.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MAC APP -->

## macOS app

The desktop UI also ships as a self-contained app for Apple Silicon Macs (macOS 14 or later), packed into a disk image you can send to someone.

### Building the disk image

```bash
cd ui
npm run dist:mac        # -> ui/release/Iris-<version>-macOS-arm64.dmg
```

`ui/scripts/package-mac.sh` does three things:

1. It freezes the Python core with PyInstaller (`packaging/iris-rpc.spec`) into a standalone `iris-rpc` folder. The core's own Python, from a separate build environment in `packaging/.venv`, comes along, so the recipient needs no Python. `iris-rpc --check` then imports every engine inside the bundle (MLX and CPU speech recognition, voice activity detection, speech synthesis, web search, camera, screen capture, audio) and stops the build if one is missing.
2. It builds the UI.
3. It runs electron-builder, which puts the core into the app's `Resources` and writes the `.dmg`.

What is not in the app: the speech models, which are downloaded at first launch (or taken from folders the recipient points to), and Ollama, which the recipient installs.

The app is signed ad hoc, not with a Developer ID, and not notarized. `ui/build/entitlements.mac.plist` holds the entitlements a hardened, notarized build will need.

### Installing it (for the recipient)

1. Install [Ollama](https://ollama.com), start it, and download a model in Terminal: `ollama pull qwen3:1.7b`. Iris can also use Ollama on another computer; the setup screen asks for its address.
2. Open the disk image and drag **Iris** into **Applications**.
3. Open Iris. Because the app is not notarized, macOS refuses the first time ("Apple could not verify..."). Click **Done**, then go to **System Settings > Privacy & Security**, scroll down to the message about Iris, click **Open Anyway**, and confirm. This is needed only once. If macOS instead says the app "is damaged", run `xattr -dr com.apple.quarantine /Applications/Iris.app` in Terminal and open it again.
4. Allow the microphone when asked. The camera and screen recording are asked for the first time Iris uses them.
5. Follow the setup screen. The speech models download once (about 2 GB) into `~/.cache/huggingface` and `~/.iris/models`.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- [Whisper](https://github.com/openai/whisper) — OpenAI's speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper implementation
- [Ollama](https://ollama.ai/) — Local LLM inference
- [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) — Kyutai's natural-sounding text-to-speech that runs on the CPU
- [Kokoro TTS](https://github.com/nazdridoy/kokoro-tts) — High-quality text-to-speech engine

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->

## License

This software is provided as-is without any warranty or guarantee. All rights are currently reserved by the author. 

<p align="right">(<a href="#top">back to top</a>)</p>

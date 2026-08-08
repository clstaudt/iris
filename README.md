<div id="top"></div>

<!-- PROJECT LOGO -->

<br />

<div align="center">

  <a href="#">
    <img width="80%" src="img/iris-title.png" alt="iris - a local AI voice assistant">
  </a>

  <p align="center">
    A fully local voice assistant powered by Whisper, Ollama, and Kokoro TTS
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
- **Text-to-Speech**: Kokoro TTS

<p align="right">(<a href="#top">back to top</a>)</p>

### Key Features

- 🎤 **Voice Input** — Just speak naturally
- 🧠 **Local LLM** — Powered by Ollama, keeping all conversations private
- 🗣️ **Natural Speech** — High-quality text-to-speech
- ⚡ **Fast Response** — Natural conversation flow
- 📊 **Rich Terminal UI** — Live status and conversation display
- 🔄 **Conversation Memory** — Maintains context throughout the conversation
- 👁️ **Vision Support** — Can see through your webcam
- 🖥️ **Screen Capture** — Can see what's on your screen
- 🔒 **Fully Offline** — No internet required after setup

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

1. **Python 3.11+** and **uv** — [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

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
uv run iris
```

### First Run Setup

On first run, Iris will check dependencies and offer to download any missing models (LLM, Whisper, and Kokoro TTS ~360 MB). Use `--skip-setup` to bypass this wizard.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE -->

## Usage

Start the voice assistant:

```bash
iris
```

Press `Ctrl+C` to exit.

### Command-line Options

```
iris --help

Options:
  --llm-model LLM_MODEL
                        Ollama model name (default: qwen3:1.7b)
  --persona {iris,hal,computer}
                        Persona/character for the assistant (default: iris)
  --enable-vision       Enable camera/vision tool (requires vision model)
  --enable-screen-capture
                        Enable screen capture tool to see your computer screen
  --vision-model MODEL  Vision model to use (default: qwen3-vl:8b)
  --skip-setup          Skip the setup wizard
```

### Examples

```bash
# Try different LLMs — experiment to find what works best for you!
iris --llm-model llama3:8b
iris --llm-model mistral
iris --llm-model phi3

# Use a different persona
iris --persona hal

# Enable vision
ollama pull qwen3-vl:8b
iris --enable-vision

# Enable screen capture
iris --enable-screen-capture

# Enable both vision and screen capture
iris --enable-vision --enable-screen-capture
```

### Example Queries

Try asking Iris:
- "Tell me a joke about programming"
- "Explain what recursion is in simple terms"
- "What are the key differences between Python and JavaScript?"
- "Help me brainstorm names for a new project"
- "Walk me through how to solve a merge conflict in Git"

### Personas

Choose from different personality options:

- **iris** (default) — Friendly and helpful assistant
- **hal** — Inspired by HAL 9000, calm and mission-focused. Pursues objectives you give it.
- **computer** — Matter-of-fact, like the USS Enterprise computer

```bash
iris --persona hal
```

Personas can be customized by editing the config files in `src/iris/personas/`.

### Vision Capabilities

With vision enabled, Iris can see through your webcam.

```bash
# Pull the vision model
ollama pull qwen3-vl:8b

# Enable vision
iris --enable-vision
```

**Try asking:**
- "What am I holding?"
- "What color is this?"
- "Can you see what's on my desk?"
- "Read the text on this paper"

**Use a different vision model:**

```bash
# For a larger model
ollama pull qwen3-vl:14b
iris --enable-vision --vision-model qwen3-vl:14b
```

### Screen Capture

With screen capture enabled, Iris can see what's displayed on your monitor. This uses the same vision model as the camera and requires screen recording permission on macOS.

```bash
# Pull the vision model (if not already done)
ollama pull qwen3-vl:8b

# Enable screen capture
iris --enable-screen-capture
```

**Try asking:**
- "What's on my screen right now?"
- "Can you read the code I have open?"
- "What application am I using?"
- "Describe what you see on my display"

> **macOS note:** Screen recording permission must be granted to the **terminal app running Iris** in **System Settings > Privacy & Security > Screen Recording**. Restart the terminal app after granting permission.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- DESKTOP UI -->

## Desktop UI (Electron shell, development mode)

An experimental desktop UI lives in `ui/`: an Electron shell (Svelte 5 + Vite + Tailwind) that wraps the Python core. The assistant logic stays in Python; Electron is a thin presentation layer talking to it over newline-delimited JSON-RPC on stdio (`python -m iris.rpc_server`, also installed as the `iris-gui` entry point).

The window shows a terminal-style chat transcript that grows upward, anchored at the bottom by a large graphical status zone. At rest the status zone is a soft white circle. It turns red and pulses with your voice as it listens, becomes a slowly rotating purple spinner while transcribing and a cyan one while thinking, opens a camera aperture while looking through the camera or capturing the screen, and pulses yellow with the speech it is playing back — state is conveyed purely by color, glow, and motion, readable from across the room.

**Run it:**

```bash
uv sync                 # Python core (once)
cd ui
npm install             # once
npm run dev             # starts Vite + Electron + the Python core
```

The assistant starts listening as soon as the window opens (model loading takes a moment on first launch) — just speak, like in the terminal app. Use ⏸ to pause listening (models stay loaded, so resume is instant), ⏭ to interrupt speech (barge-in), or type a message in the input field.

Unlike the CLI, which gates them behind flags, the desktop UI enables the camera, screen capture and web search tools by default; each one degrades to a warning if its optional dependencies are missing. Image prompts are routed to an installed vision model when there is one, otherwise they stay on the primary model.

The control bar also lets you switch the persona, the Ollama model, and the microphone while the app runs. The model menu lists what Ollama has installed (embedding-only models are hidden, and models without tool-calling support are marked); switching keeps the conversation history. If the configured model is missing, Iris says so at startup instead of failing on the first reply.

The RPC protocol: requests are JSON-RPC 2.0 (`app.info`, `assistant.start/stop/interrupt/pause/resume`, `chat.send_text`, `persona.list/set`, `llm.list/set`, `audio.list_inputs/set_input`); the Python core pushes events as notifications (`config`, `state`, `audio.level`, `chat.user/chunk/done`, `tool.use`, `error`).

> **Note for Cursor/VSCode terminals:** if Electron exits immediately with `Cannot set properties of undefined`, your terminal exports `ELECTRON_RUN_AS_NODE=1`. The `dev` script already clears it; when launching `electron .` manually, prefix it with `ELECTRON_RUN_AS_NODE=`.

### macOS microphone permission in dev mode

If the waveform stays flat and the orb never reacts, macOS is returning pure silence instead of audio — the symptom of a missing microphone grant. In development the grant does **not** belong to Electron: macOS assigns it to the *responsible process*, which is the app that started the dev server (Cursor, VS Code, Terminal or iTerm). That is also why "Electron" may not appear in the Privacy list at all.

Check what your shell actually captures:

```bash
.venv/bin/python -c "import sounddevice as sd, numpy as np; r = sd.rec(int(3*16000), samplerate=16000, channels=1); sd.wait(); print('max amplitude:', float(np.abs(r).max()))"
```

A working microphone reports a non-zero value (a quiet room still has a noise floor). Exactly `0.0` means capture is being denied. Fix it by enabling the microphone for your launching app under **System Settings > Privacy & Security > Microphone**, then quitting and reopening that app (a reload is not enough). Running `npm run dev` from a terminal that already has the grant — the same one where `iris` works on the CLI — also resolves it.

The app surfaces this itself: a red line appears in the transcript when the microphone delivers only silence for a few seconds.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- [Whisper](https://github.com/openai/whisper) — OpenAI's speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper implementation
- [Ollama](https://ollama.ai/) — Local LLM inference
- [Kokoro TTS](https://github.com/nazdridoy/kokoro-tts) — High-quality text-to-speech engine
- [Rich](https://github.com/Textualize/rich) — Terminal UI framework

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->

## License

This software is provided as-is without any warranty or guarantee. All rights are currently reserved by the author. 

<p align="right">(<a href="#top">back to top</a>)</p>

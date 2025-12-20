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
    ·
    <a href="#troubleshooting">Troubleshooting</a>
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
    <li><a href="#usage">Usage</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

Iris is a voice-controlled AI assistant that runs entirely on your local machine. It listens to your voice, transcribes your speech, thinks through a response using a local LLM, and speaks back to you—all without sending data to external servers.

The assistant combines three powerful technologies into a seamless conversational interface:
- **Speech-to-Text**: OpenAI's Whisper via faster-whisper with GPU acceleration
- **Language Model**: Any Ollama-compatible model for intelligent responses
- **Text-to-Speech**: Kokoro TTS for natural-sounding voice output

<p align="right">(<a href="#top">back to top</a>)</p>

### Key Features

- 🎤 **Voice Input** — Real-time microphone recording with automatic silence detection
- 🧠 **Local LLM** — Powered by Ollama, keeping all conversations private
- 🗣️ **Natural Speech** — High-quality text-to-speech with Kokoro TTS
- ⚡ **Fast Response** — Starts speaking while still thinking for natural conversation flow
- 📊 **Rich Terminal UI** — Live status updates and conversation display
- 🔄 **Conversation Memory** — Remembers context across your conversation
- 👁️ **Vision Support** — Can see through your webcam with vision-capable models
- 🔍 **Web Search** — Can search for current information when needed
- 🔒 **Fully Offline** — No internet required after initial setup (except for web search)

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
git clone https://github.com/your-username/iris.git
cd iris
uv sync
uv run iris
```

### First Run Setup

On first run, Iris includes a **setup wizard** that checks all dependencies and offers to download any missing models:

```
┌─────────────────────────────────────────────────────────────────┐
│ Iris Setup                                                      │
│                                                                 │
│ Checking dependencies and downloading required models...        │
└─────────────────────────────────────────────────────────────────┘

✓  Ollama                    running
↓  LLM (qwen3:1.7b)          needs download
↓  Whisper (small)           needs download
✓  kokoro-tts CLI            installed
↓  Kokoro TTS models         needs download

The following components need to be downloaded:
  • Ollama model (qwen3:1.7b)
  • Whisper model (small)
  • Kokoro TTS models (~360 MB)

Download locations:
  • Ollama models: ~/.ollama/models/
  • Whisper model: ~/.cache/huggingface/hub/
  • Kokoro models: ~/.iris/models/

Download now? [Y/n]:
```

The wizard will show progress bars during downloads and tell you exactly where files are being stored.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE -->

## Usage

Start the voice assistant:

```bash
iris
```

The assistant will listen for your voice, transcribe what you say, generate a response, and speak it back to you.

Press `Ctrl+C` to exit.

### Command-line Options

```
iris --help

Options:
  --whisper-model {tiny,base,small,medium,large-v2,large-v3}
                        Whisper model size (default: small)
  --llm-model LLM_MODEL
                        Ollama model name (default: qwen3:1.7b)
  --device {auto,cpu,cuda}
                        Device for Whisper (default: auto)
  --persona {iris,hal,computer}
                        Persona/character for the assistant (default: iris)
  --enable-vision       Enable camera/vision tool (requires vision model)
  --enable-web-search   Enable web search tool for current information
  --skip-setup          Skip the setup wizard
```

### Examples

```bash
# Try different LLMs — experiment to find what works best for you!
iris --llm-model llama3:8b
iris --llm-model mistral
iris --llm-model phi3

# Use a larger Whisper model for improved transcription
iris --whisper-model medium

# Enable vision with a vision-capable model
ollama pull llama3.2-vision
iris --llm-model llama3.2-vision --enable-vision

# Enable web search for current information
iris --enable-web-search

# Use a different persona
iris --persona hal
```

### Vision Capabilities

Iris supports **intelligent vision switching** - it uses a fast text model for normal conversations and automatically switches to a vision model only when it needs to see something.

```bash
# First, ensure you have both models
ollama pull qwen2.5:1.5b        # Fast text model
ollama pull qwen3-vl:8b         # Vision model (default)

# Start iris with vision enabled
iris --enable-vision
```

**How it works:**
- Uses the fast primary model (`qwen2.5:1.5b`) for normal conversations
- When you ask it to look at something, it automatically:
  1. Captures an image with your webcam
  2. Switches to the vision model (`qwen3-vl:8b`) 
  3. Processes the image and responds
  4. Switches back to the fast model for the next conversation

**Example conversations:**

```
You: "Tell me a joke"
→ Uses fast text model (qwen2.5:1.5b)

You: "What am I holding?"
→ Captures image → Switches to vision model (qwen3-vl:8b) → Responds

You: "That's funny, tell me another joke"  
→ Back to fast text model (qwen2.5:1.5b)
```

**Custom vision model:**

```bash
# Use a different vision model
ollama pull llama3.2-vision:11b
iris --enable-vision --vision-model llama3.2-vision:11b

# Or use a smaller one
ollama pull qwen3-vl:2b
iris --enable-vision --vision-model qwen3-vl:2b
```

You'll see a "Looking" indicator appear in the status bar when the camera captures an image.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- TROUBLESHOOTING -->

## Troubleshooting

<details>
<summary><b>"Ollama connection failed"</b></summary>

Make sure Ollama is running:

**macOS:**
```bash
brew services start ollama
```

**Linux:**
```bash
ollama serve
```

**Windows:** Launch the Ollama application from the Start menu.
</details>

<details>
<summary><b>Poor transcription accuracy</b></summary>

The default `small` model should work well for most use cases. If you're experiencing issues, try speaking more clearly or reducing background noise. For challenging audio environments, you can try a larger model:
```bash
iris --whisper-model medium
```
</details>

<details>
<summary><b>Audio issues on macOS</b></summary>

Grant microphone permissions to your terminal in **System Preferences → Security & Privacy → Microphone**.
</details>

<details>
<summary><b>Audio issues on Linux</b></summary>

Ensure you have PortAudio installed:
```bash
# Debian/Ubuntu
sudo apt install portaudio19-dev

# Fedora
sudo dnf install portaudio-devel

# Arch
sudo pacman -S portaudio
```
</details>

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

This project is provided as-is for personal and educational use.

<p align="right">(<a href="#top">back to top</a>)</p>

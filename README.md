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

> **macOS note:** On first use, you may be prompted to grant screen recording permission in **System Settings > Privacy & Security > Screen Recording**. Restart Iris after granting permission.

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

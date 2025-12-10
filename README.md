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
- 🔒 **Fully Offline** — No internet required after initial setup

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

On first run, Iris will prompt you to download the required TTS models (~360 MB).

### Installation

```bash
git clone https://github.com/your-username/iris.git
cd iris
uv sync
uv run iris
```

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
```

### Examples

```bash
# Try different LLMs — experiment to find what works best for you!
iris --llm-model llama3:8b
iris --llm-model mistral
iris --llm-model phi3

# Use a larger Whisper model for improved transcription
iris --whisper-model medium
```

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

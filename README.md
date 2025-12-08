# Iris Voice Assistant

A voice-controlled AI assistant that listens, understands, thinks, and speaks. Iris combines speech-to-text (Whisper), large language models (Ollama), and text-to-speech (Kokoro) into a seamless conversational interface.

## Features

- 🎤 **Voice Input**: Real-time microphone recording with automatic silence detection
- 🧠 **Speech Recognition**: Powered by OpenAI's Whisper (supports GPU acceleration)
- 💬 **Intelligent Conversation**: Uses Ollama for natural language understanding
- 🗣️ **Natural Speech**: High-quality text-to-speech with Kokoro TTS
- ⚡ **Parallel Processing**: Simultaneous speech generation and playback for fast response
- 📊 **Rich UI**: Beautiful status indicators and conversation display
- 🔄 **Context Management**: Automatic conversation summarization to stay within token limits

## Architecture

Iris is built with a modular architecture:

- `audio.py` - Audio recording and playback
- `transcription.py` - Whisper-based speech-to-text
- `llm.py` - Conversation management with Ollama
- `tts.py` - Text-to-speech pipeline with Kokoro
- `ui.py` - Rich terminal UI components
- `assistant.py` - Main coordinator orchestrating all components
- `cli.py` - Command-line interface entry point

## Prerequisites

1. **Python 3.11+**
2. **Ollama** - For LLM inference
   ```bash
   brew install ollama
   brew services start ollama
   ollama pull qwen3:8b
   ```
3. **Kokoro TTS** - For text-to-speech
   ```bash
   pip install kokoro
   ```
4. **Kokoro Models** - See [models/README.md](models/README.md) for download instructions

## Installation

### Using uv (recommended)

```bash
# Clone or navigate to the iris directory
cd /Users/cls/Documents/Work/RnD/iris

# Install dependencies
uv sync

# Run iris
uv run iris
```

### Using pip

```bash
# Install in development mode
pip install -e .

# Run iris
iris
```

## Usage

### Basic Usage

Simply run iris to start the voice assistant:

```bash
iris
```

The assistant will:
1. Listen for your voice input
2. Transcribe your speech to text
3. Send it to the LLM for processing
4. Stream the response while simultaneously generating speech
5. Speak the response aloud

Press `Ctrl+C` to exit.

### Command-line Options

```bash
iris --help

Options:
  --whisper-model {tiny,base,small,medium,large-v2,large-v3}
                        Whisper model size (default: base)
  --llm-model LLM_MODEL
                        Ollama model name (default: qwen3:8b)
  --device {auto,cpu,cuda}
                        Device for Whisper (default: auto)
```

### Examples

Use a larger Whisper model for better accuracy:
```bash
iris --whisper-model medium
```

Use a different LLM:
```bash
iris --llm-model llama3:8b
```

Force CPU for Whisper:
```bash
iris --device cpu
```

## Configuration

### Model Locations

Iris searches for Kokoro TTS models in the following order:

1. `$IRIS_MODELS_DIR` (if set)
2. `~/.iris/models/`
3. `~/.local/share/iris/models/`
4. `./models/` (in current directory)

See [models/README.md](models/README.md) for detailed setup instructions.

### Environment Variables

- `IRIS_MODELS_DIR` - Custom location for Kokoro TTS models

## How It Works

### Conversation Flow

1. **Listening**: Waits for speech, then records until silence is detected
2. **Understanding**: Transcribes audio using Whisper
3. **Thinking**: Streams response from LLM
4. **Speaking**: Generates and plays speech in parallel with streaming

### Parallel Speech Pipeline

Iris uses a sophisticated pipeline for responsive speech:

```
Text chunks → [Generation Queue] → TTS Worker → [Audio Queue] → Playback Worker
```

This allows the assistant to start speaking while still generating the rest of the response.

### Context Management

The assistant automatically manages conversation context:
- Tracks token usage for all messages
- Summarizes old messages when approaching context limit
- Keeps recent conversation intact for coherent responses

## Development

### Project Structure

```
iris/
├── src/iris/
│   ├── __init__.py
│   ├── cli.py          # Entry point
│   ├── assistant.py    # Main coordinator
│   ├── audio.py        # Recording + playback
│   ├── transcription.py # Whisper STT
│   ├── llm.py          # Ollama conversation
│   ├── tts.py          # Kokoro TTS pipeline
│   └── ui.py           # Terminal UI
├── models/             # TTS model files
├── tests/              # Test suite
├── pyproject.toml      # Project configuration
└── README.md
```

### Running Tests

```bash
uv run pytest
```

## Troubleshooting

### "kokoro-tts CLI not found"

Install kokoro:
```bash
pip install kokoro
```

### "Could not find kokoro models"

Download the models as described in [models/README.md](models/README.md).

### "Ollama connection failed"

Make sure Ollama is running:
```bash
brew services start ollama
```

### Poor transcription accuracy

Try a larger Whisper model:
```bash
iris --whisper-model medium
```

### Audio issues on macOS

Grant microphone permissions to your terminal in System Preferences → Security & Privacy → Microphone.

## Credits

- **Whisper**: OpenAI's speech recognition model
- **Ollama**: Local LLM inference
- **Kokoro**: High-quality TTS engine
- **Rich**: Terminal UI framework

## License

This project is provided as-is for personal and educational use.


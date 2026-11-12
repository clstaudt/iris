# Kokoro TTS Models

Iris uses the Kokoro TTS models for text-to-speech generation. You need to download these model files before using the voice assistant.

## Required Model Files

- `kokoro-v1.0.onnx` - The main TTS model
- `voices-v1.0.bin` - Voice data for different speakers

## Installation

### Option 1: Download to this directory

Place the model files in this directory:
```
models/
├── kokoro-v1.0.onnx
└── voices-v1.0.bin
```

### Option 2: User directory

Place the model files in your home directory:
```
~/.iris/models/
├── kokoro-v1.0.onnx
└── voices-v1.0.bin
```

### Option 3: System-wide installation

Place the model files in:
```
~/.local/share/iris/models/
├── kokoro-v1.0.onnx
└── voices-v1.0.bin
```

### Option 4: Custom location

Set the `IRIS_MODELS_DIR` environment variable to point to your model directory:
```bash
export IRIS_MODELS_DIR=/path/to/your/models
```

## Where to Get the Models

The Kokoro TTS models are available from the [kokoro-tts releases](https://github.com/nazdridoy/kokoro-tts/releases).

Download links:
- Model: https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
- Voices: https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin

You can download them with:
```bash
# Create the models directory
mkdir -p ~/.iris/models

# Download the models
wget -O ~/.iris/models/kokoro-v1.0.onnx https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
wget -O ~/.iris/models/voices-v1.0.bin https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
```

## Verify Installation

After placing the files, you can verify they're found by running:
```bash
iris --help
```

If the models are missing, Iris will show an error message indicating where it looked for them.


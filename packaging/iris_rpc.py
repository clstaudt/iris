"""Entry point of the frozen Python core inside the desktop app.

`iris-rpc --check` imports each engine and reports what fails, a smoke test
for packaged builds (a missing library would otherwise only show up as a
quietly disabled feature).
"""

import importlib
import multiprocessing
import sys
import traceback

from iris.rpc_server import main

# What the core needs at runtime, by feature
ENGINES = {
    "speech recognition (MLX, Apple GPU)": "mlx_whisper",
    "speech recognition (CPU)": "faster_whisper",
    "voice activity detection": "faster_whisper.vad",
    "turn detection": "iris.turns",
    "echo cancellation": "livekit.rtc",
    "speech synthesis": "kokoro_onnx",
    "web search": "ddgs",
    "web pages": "trafilatura",
    "camera": "cv2",
    "screen capture": "mss",
    "audio devices": "sounddevice",
}


def check() -> int:
    failed = 0
    for feature, module in ENGINES.items():
        try:
            importlib.import_module(module)
            print(f"ok      {feature} ({module})")
        except Exception:
            failed += 1
            print(f"FAILED  {feature} ({module})")
            traceback.print_exc(file=sys.stdout)
    return failed


if __name__ == "__main__":
    # Libraries that start worker processes (joblib via librosa) need this
    # when frozen, or each worker would start another copy of the core
    multiprocessing.freeze_support()
    if "--check" in sys.argv:
        sys.exit(1 if check() else 0)
    main()

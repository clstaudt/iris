"""Entry point of the frozen Python core inside the desktop app.

`iris-rpc --check` imports each engine and reports what fails, a smoke test
for packaged builds (a missing library would otherwise only show up as a
quietly disabled feature). It also has the voices' readers read a line, as
a missing table or dictionary shows only once they read.
"""

import importlib
import multiprocessing
import re
import sys
import traceback

from iris.numbers import numbers_in_words
from iris.rpc_server import main
from iris.tts import READERS

# What the core needs at runtime, by feature
ENGINES = {
    "speech recognition (MLX, Apple GPU)": "mlx_whisper",
    "speech recognition (CPU)": "faster_whisper",
    "voice activity detection": "faster_whisper.vad",
    "turn detection": "iris.turns",
    "echo cancellation": "livekit.rtc",
    "speech synthesis (Pocket TTS)": "pocket_tts",
    "speech synthesis (Kokoro)": "kokoro_onnx",
    "the voice (both engines)": "iris.tts",
    "OpenAI-compatible model servers": "iris.llm_client",
    "web search": "ddgs",
    "web pages": "trafilatura",
    "camera": "cv2",
    "screen capture": "mss",
    "audio devices": "sounddevice",
    "numbers in words": "num2words",
    "Japanese in kana": "pykakasi",
}

# What the voices are given to read, and what it must not hold any more
READS = {
    "numbers in words": (lambda: numbers_in_words("am 27. November um 10:30 Uhr", "de"), r"\d"),
    "Japanese in kana": (lambda: READERS.kana("東京の天気は晴れ、18度"), r"[\d\u4e00-\u9fff]"),
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
    for feature, (read, left) in READS.items():
        try:
            said = read()
            if not said.strip() or re.search(left, said):
                raise ValueError(f"read as {said!r}")
            print(f"ok      {feature} reads")
        except Exception:
            failed += 1
            print(f"FAILED  {feature} reads")
            traceback.print_exc(file=sys.stdout)
    return failed


if __name__ == "__main__":
    # Libraries that start worker processes (joblib via librosa) need this
    # when frozen, or each worker would start another copy of the core
    multiprocessing.freeze_support()
    if "--check" in sys.argv:
        sys.exit(1 if check() else 0)
    main()

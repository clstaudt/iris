# Development

```bash
uv run pytest                  # unit tests
uv run pytest -m integration   # integration tests, which need Ollama, Kokoro and Whisper
```

The unit tests also run on CI's slow, shared runners, where a sleep of 0.3 s can last half a second. A test that measures time must hold there too, without widening its numbers. Where the code reads a clock, give it one the test moves on (the `Clock` in `test_playback_levels` and `test_voicetest`), so what is measured is what the test said. Wait for an event or a condition with a generous ceiling instead of sleeping and hoping the other thread got there. Where real time must pass, as when the virtual microphone delivers blocks in real time, check what the code promises: a block never comes before it was heard, measured on the stream's own times, rather than "ten blocks in 0.32 s". Bound a wait by what the test guards against (a call cut off before the model's first words), not by how fast your machine is. A bound like that still fails when the behaviour breaks.

The walkthrough opens by itself only once per settings folder (Settings > Walkthrough runs it again with Iris running). To go through it as a new user:

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

The test starts with empty settings and goes through the walkthrough (a server typed without `/v1`, the downloads, the sound check, a voice preview, the wake word, Iris's greeting), then opens every stage of the pipeline, switches models, turn detection and tools, plays voices, edits personas, switches setups, has a typed conversation, stops replies, opens presence as a page of its own and runs the walkthrough again. Its core has a home folder of its own, so your personas, memory and settings are left alone, and it sleeps until a wake word nobody says. Its microphone and speaker are a virtual room (`irisdev.e2e_core`): nothing sounds, and the room is not heard. Screenshots of failed steps and the core's log go to `ui/e2e/output`.

In a terminal, `npm run e2e` shows the steps under their areas as they end, the one running with its time and about how long the rest will take, each failure with its reason and screenshot, and at the end the slowest steps and an `E2E_ONLY` that runs the failures again (`src/irisdev/e2e_report.py`). Piped to a file, it writes the same as plain lines. `npm run e2e -- --verbose` adds the core's own log lines; `npm run e2e:plain` runs the test without the reporter.

Other environment variables: `OLLAMA_HOST` (the model server when none is set in the app), `IRIS_LLM_API_KEY` (a key for servers that need one), `IRIS_MEMORY_DB`, `IRIS_PERSONAS_DIR`, `IRIS_PLUGINS_DIR`, and `IRIS_MODELS_DIR` (where Kokoro's files are looked for).

The app icon is the fully open iris of the startup animation, drawn by the same code (`ui/src/lib/iris.ts`); `npm run icon` re-renders `ui/build/icon.png` and, on macOS, `icon.icns`. `iris-benchmark` compares kokoro-onnx with the kokoro-tts command line.

> **Cursor and VS Code terminals:** if Electron exits immediately with `Cannot set properties of undefined`, your terminal exports `ELECTRON_RUN_AS_NODE=1`. The `dev` script already clears it; when launching `electron .` manually, prefix it with `ELECTRON_RUN_AS_NODE=`.

**The microphone in development on macOS.** If the orb never reacts to your voice, macOS is returning pure silence instead of audio, the symptom of a missing microphone grant. In development the grant does not belong to Electron: macOS assigns it to the app that started the dev server (Cursor, VS Code, Terminal or iTerm), which is also why "Electron" may not appear in the Privacy list at all. Check what your shell captures:

```bash
.venv/bin/python -c "import sounddevice as sd, numpy as np; r = sd.rec(int(3*16000), samplerate=16000, channels=1); sd.wait(); print('max amplitude:', float(np.abs(r).max()))"
```

A working microphone reports a non-zero value (a quiet room still has a noise floor); exactly `0.0` means capture is being denied. Enable the microphone for the launching app under **System Settings > Privacy & Security > Microphone**, then quit and reopen that app (a reload is not enough). The app says so itself: a red line appears in the conversation when the microphone delivers only silence for a few seconds.

"""JSON-RPC 2.0 stdin/stdout transport for the Electron UI shell.

Reads newline-delimited JSON-RPC requests from stdin and writes responses to
stdout. Assistant activity (state changes, chat chunks, mic levels) is pushed
as JSON-RPC notifications: ``{"jsonrpc": "2.0", "method": "event",
"params": {"type": ..., "data": ...}}``.

Stdout carries RPC frames only — prints, and native libraries writing to
file descriptor 1, go to stderr so they cannot corrupt the framing.

Usage::

    python -m iris.rpc_server
"""

import json
import logging
import os
import queue
import signal
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from iris.controller import AssistantController
from iris.events import EventBus


def _coerce(obj):
    """Last resort for values json cannot encode, notably numpy scalars.

    Audio levels arrive as numpy floats; without this an unserializable value
    silently kills the event, since EventBus swallows subscriber exceptions.
    """
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def encode_frame(frame: dict) -> str:
    """Serialize one JSON-RPC frame to a single line of JSON."""
    return json.dumps(frame, default=_coerce)


def _start_parent_watchdog(interval: float = 2.0):
    """Exit if the parent process (Electron) disappears.

    Polls os.getppid(); when it changes to 1 (re-parented to init/launchd)
    the parent has died and we must not linger as an orphan.
    """
    parent_pid = os.getppid()

    def _watch():
        while True:
            if os.getppid() != parent_pid:
                os.kill(os.getpid(), signal.SIGTERM)
                return
            threading.Event().wait(interval)

    threading.Thread(target=_watch, daemon=True).start()


# Controls that take effect the moment they arrive, whatever the core is busy
# with: Stop must never wait behind a slow request (switching the server or
# the persona, checking servers), and neither must pausing or the microphone
IMMEDIATE = frozenset(
    {"assistant.interrupt", "assistant.pause", "assistant.resume", "assistant.hold", "mic.set"}
)

# Slow requests that depend on nothing else and nothing depends on: each is
# answered on a thread of its own, so it holds up neither the requests in
# order nor the controls. Checking the servers takes seconds when one is away,
# and the app checks them again and again while their list is shown; the
# walkthrough also looks for servers and asks one for its models. So are
# voice cloning's: asking Hugging Face, downloading its model, reading a
# sample, making a voice, and the recording (stopped while the rest waits).
# Renaming or deleting a cloned voice is not: it comes in order, after a
# persona saved with it. Making voices ready for a language only starts
# its jobs, which run on a thread of their own (iris.readiness)
CONCURRENT = frozenset(
    {
        "llm.check_hosts",
        "llm.find_servers",
        "llm.probe",
        "voice.clone_access",
        "voice.clone_connect",
        "voice.clone_fetch",
        "voice.record",
        "voice.sample",
        "voice.clone",
    }
)


def serve(read_line, methods: dict, write_frame, immediate=IMMEDIATE, concurrent=CONCURRENT):
    """Answer requests until the input ends.

    Requests are answered one after the other, in the order they came, on a
    worker thread; the immediate ones are answered as they arrive, and the
    concurrent ones each on a thread of its own.
    """
    work: queue.Queue = queue.Queue()

    def handle(request: dict):
        req_id = request.get("id")
        try:
            method = request.get("method", "")
            handler = methods.get(method)
            if handler is None:
                raise ValueError(f"Unknown method: {method}")
            result = handler(request.get("params", {}) or {})
            write_frame({"jsonrpc": "2.0", "id": req_id, "result": result})
        except Exception as exc:
            traceback.print_exc()
            write_frame(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32603,
                        "message": str(exc),
                        "data": traceback.format_exc(),
                    },
                }
            )

    def worker():
        while True:
            request = work.get()
            if request is None:
                return
            handle(request)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while True:
        line = read_line()
        if not line:
            break  # input closed: the parent is gone
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            write_frame(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
            )
            continue
        method = request.get("method")
        if method in immediate:
            handle(request)
        elif method in concurrent:
            threading.Thread(target=handle, args=(request,), daemon=True).start()
        else:
            work.put(request)
    work.put(None)
    thread.join(timeout=5)


def log_the_voice():
    """Keep what the voice did over speech (each hold, what it was judged to
    be, where the voice went on) in ~/.iris/voice.log, to tell why a voice
    stuttered in a real room. IRIS_VOICE_LOG names another file; empty, it
    keeps none."""
    path = os.environ.get("IRIS_VOICE_LOG", str(Path.home() / ".iris" / "voice.log"))
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=1, encoding="utf-8")
    except OSError:
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log = logging.getLogger("iris.voice")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def main():
    # Stdout purity: RPC frames get a private copy of the real stdout, and
    # file descriptor 1 is pointed at stderr. That catches native libraries
    # writing to fd 1 as well as Python prints, so nothing splits a frame.
    sys.stdout.flush()
    rpc_out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    log_the_voice()

    bus = EventBus()
    controller = AssistantController(bus)

    write_lock = threading.Lock()

    def write_frame(frame: dict):
        try:
            encoded = encode_frame(frame)
        except (TypeError, ValueError) as exc:
            print(f"[rpc] dropped unserializable frame: {exc}", file=sys.stderr)
            return
        with write_lock:
            rpc_out.write(encoded + "\n")
            rpc_out.flush()

    def forward_event(event_type: str, data: dict):
        write_frame(
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {"type": event_type, "data": data},
            }
        )

    bus.subscribe(forward_event)

    def debug_emit(params: dict) -> dict:
        bus.emit(params.get("type", "debug"), params.get("data", {}))
        return {"ok": True}

    methods = {
        "app.info": lambda params: controller.info(),
        "assistant.start": lambda params: controller.start(params),
        "assistant.stop": lambda params: controller.stop(),
        "assistant.interrupt": lambda params: controller.interrupt(),
        "assistant.pause": lambda params: controller.pause(),
        "assistant.resume": lambda params: controller.resume(),
        "audio.list_inputs": lambda params: controller.list_input_devices(),
        "audio.set_input": lambda params: controller.set_input_device(params.get("index")),
        "llm.list": lambda params: controller.list_models(),
        "llm.set": lambda params: controller.set_model(params.get("name", "")),
        "llm.set_host": lambda params: controller.set_ollama_host(params.get("url", "")),
        "llm.check_hosts": lambda params: controller.check_servers(params.get("urls", [])),
        "llm.find_servers": lambda params: controller.find_servers(),
        "llm.probe": lambda params: controller.probe_server(params.get("url", "")),
        "chat.send_text": lambda params: controller.send_text(
            params.get("text", ""), params.get("attachments")
        ),
        "chat.history": lambda params: controller.history(),
        "memory.topics": lambda params: controller.memory_topics(),
        "memory.topic": lambda params: controller.memory_topic(params.get("id", -1)),
        "memory.forget": lambda params: controller.memory_forget(params.get("id", -1)),
        "memory.set_amnesia": lambda params: controller.set_amnesia(bool(params.get("on"))),
        "assistant.hold": lambda params: controller.hold_listening(bool(params.get("on"))),
        "mic.set": lambda params: controller.set_mic(bool(params.get("on", True))),
        # {"on": bool} switches the persona's wake word on or off; {"word"}
        # listens for a word of its own (a test's)
        "wake.set": lambda params: (
            controller.set_wake(bool(params["on"]))
            if "on" in params
            else controller.set_wake_word(params.get("word", ""))
        ),
        "wake.adopt": lambda params: controller.adopt_wake_words(
            params.get("persona"), params.get("wake_word"), params.get("sleep_word")
        ),
        "persona.words": lambda params: controller.set_persona_words(
            params.get("id"), params.get("wake_word"), params.get("sleep_word")
        ),
        "setup.info": lambda params: controller.setup_info(),
        "setup.download": lambda params: controller.setup_download(params),
        "mic.check": lambda params: controller.sound_check(
            bool(params.get("on", True)), params.get("index")
        ),
        "assistant.greet": lambda params: controller.greet(),
        "models.check": lambda params: controller.check_model_paths(
            params.get("stt_path"), params.get("tts_dir")
        ),
        "models.set": lambda params: controller.set_model_paths(
            params.get("stt_path"), params.get("tts_dir")
        ),
        "wake.options": lambda params: controller.set_wake_options(
            params.get("window"), params.get("sleep_word")
        ),
        "voice.set_output": lambda params: controller.set_voice_output(
            bool(params.get("enabled", True))
        ),
        "voice.set_barge_in": lambda params: controller.set_barge_in(
            bool(params.get("enabled", True))
        ),
        "tools.set": lambda params: controller.set_tool(
            params.get("name", ""), bool(params.get("enabled", True))
        ),
        "stt.models": lambda params: controller.stt_models(),
        "stt.set": lambda params: controller.set_stt_model(params.get("model", "")),
        "turn.set": lambda params: controller.set_turn_detection(params.get("mode", "")),
        "plugins.list": lambda params: controller.plugins_list(),
        "engines.list": lambda params: controller.engines_list(),
        "persona.list": lambda params: controller.list_personas(),
        "persona.set": lambda params: controller.set_persona(params.get("name", "")),
        "persona.get": lambda params: controller.persona_get(params.get("id", "")),
        "persona.voices": lambda params: controller.persona_voices(),
        "persona.save": lambda params: controller.persona_save(params),
        "persona.delete": lambda params: controller.persona_delete(params.get("id", "")),
        "persona.preview": lambda params: controller.persona_preview(params),
        # Cloning a voice for a persona (iris.cloning)
        "voice.clone_access": lambda params: controller.cloning.access(),
        "voice.clone_connect": lambda params: controller.cloning.connect(params.get("token", "")),
        "voice.clone_fetch": lambda params: controller.cloning.fetch(),
        "voice.record": lambda params: controller.cloning.record(
            bool(params.get("on")), params.get("device"), bool(params.get("keep", True))
        ),
        "voice.sample": lambda params: controller.cloning.use_file(
            params.get("data", ""), params.get("name", "")
        ),
        "voice.clone": lambda params: controller.voice_clone(params),
        # The Voices page: every voice, the cloned ones managed, and each
        # made ready for Iris's languages
        "voice.list": lambda params: controller.voice_list(),
        "voice.rename": lambda params: controller.cloning.rename(
            params.get("voice", ""), params.get("name", "")
        ),
        "voice.delete": lambda params: controller.voice_delete(
            params.get("voice", ""), bool(params.get("reassign"))
        ),
        "voice.ready": lambda params: controller.voice_make_ready(
            params.get("engine", ""), params.get("voice", ""), params.get("languages")
        ),
        "voice.retry": lambda params: controller.voice_retry(params.get("id", "")),
        "voice.jobs": lambda params: {"jobs": controller.readier.jobs()},
        "languages.set": lambda params: controller.set_languages(params.get("languages")),
        "debug.emit": debug_emit,
        "debug.inject_audio": lambda params: controller.inject_audio(params.get("path", "")),
    }

    _start_parent_watchdog()
    serve(sys.stdin.readline, methods, write_frame)


if __name__ == "__main__":
    main()

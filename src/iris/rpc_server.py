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
import os
import signal
import sys
import threading
import traceback

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


def main():
    # Stdout purity: RPC frames get a private copy of the real stdout, and
    # file descriptor 1 is pointed at stderr. That catches native libraries
    # writing to fd 1 as well as Python prints, so nothing splits a frame.
    sys.stdout.flush()
    rpc_out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

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
        "chat.send_text": lambda params: controller.send_text(params.get("text", "")),
        "chat.history": lambda params: controller.history(),
        "memory.topics": lambda params: controller.memory_topics(),
        "memory.topic": lambda params: controller.memory_topic(params.get("id", -1)),
        "memory.forget": lambda params: controller.memory_forget(params.get("id", -1)),
        "memory.set_amnesia": lambda params: controller.set_amnesia(bool(params.get("on"))),
        "assistant.hold": lambda params: controller.hold_listening(bool(params.get("on"))),
        "mic.set": lambda params: controller.set_mic(bool(params.get("on", True))),
        "wake.set": lambda params: controller.set_wake_word(params.get("word", "")),
        "setup.info": lambda params: controller.setup_info(),
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
        "persona.list": lambda params: controller.list_personas(),
        "persona.set": lambda params: controller.set_persona(params.get("name", "")),
        "debug.emit": debug_emit,
        "debug.inject_audio": lambda params: controller.inject_audio(params.get("path", "")),
    }

    _start_parent_watchdog()

    while True:
        line = sys.stdin.readline()
        if not line:
            break  # stdin closed — parent is gone
        line = line.strip()
        if not line:
            continue

        req_id = None
        try:
            request = json.loads(line)
            req_id = request.get("id")
            method = request.get("method", "")
            params = request.get("params", {})

            handler = methods.get(method)
            if handler is None:
                raise ValueError(f"Unknown method: {method}")

            result = handler(params)
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


if __name__ == "__main__":
    main()

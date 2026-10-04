"""Thread-safe pub/sub event bus used to stream assistant activity to a GUI.

The terminal UI (Rich) does not use this bus; it is attached only when an
external UI (e.g. the Electron shell) is driving the assistant. Subscribers
must be fast and non-blocking: events are emitted synchronously, sometimes
while StatusManager.status_lock is held.

Event types:
    state        {"state": "idle"|"starting"|"listening"|"understanding"|"thinking"|
                  "speaking"|"looking"|"capturing_screen"|"searching"|"browsing"|
                  "paused"|"muted"|"typing"|"dormant"}
                                                     paused: whole conversation;
                                                     muted: mic off;
                                                     dormant: waiting for the wake word
    flags        {"flags": {flag_name: bool, ...}}   raw StatusManager snapshot
    audio.level  {"rms": float}                      mic input level 0.0-1.0
    chat.user    {"text": str, "language": str, "source": "voice"|"text", "id": str}
                                                     id only for typed turns
    chat.chunk   {"text": str}                       streaming LLM token chunk
    chat.done    {"text": str, "interrupted": bool}
    tool.use     {"tool": str, "detail": str}
    wake         {"word": str}                       the wake word was heard
    wake.window  {"remaining": float}                seconds until it falls asleep;
                                                     0 once it is asleep
    sleep        {}                                  it fell asleep (window over,
                                                     or the sleep word)
    error        {"message": str, "kind": str}         kind is optional
    setup.progress {"model": str, "label": str,      model download at first start
                    "done": int, "total": int}       (model: stt, wake or tts);
                                                     an empty label when done
    memory.updated {}                                topics were filed or forgotten
"""

import threading
from collections.abc import Callable


class EventBus:
    """Minimal synchronous pub/sub bus, safe for cross-thread emit/subscribe."""

    def __init__(self):
        self._subscribers: list[Callable[[str, dict], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[str, dict], None]):
        """Register a callback(event_type, data) invoked on every event."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, dict], None]):
        """Remove a previously registered callback."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def emit(self, event_type: str, data: dict | None = None):
        """Emit an event to all subscribers. Subscriber exceptions are swallowed."""
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event_type, data or {})
            except Exception:
                pass

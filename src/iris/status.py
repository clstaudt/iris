"""What the assistant is doing, as flags the app shows."""

import threading

from iris.events import EventBus

STATUS_FLAG_NAMES = (
    "listening",
    "dormant",  # listening only for the wake word
    "looking",
    "capturing_screen",
    "searching",
    "browsing",
    "understanding",
    "thinking",
    "summarizing",
)


class StatusManager:
    """Status flags, each change pushed to the app as a "flags" event."""

    def __init__(self, event_bus: EventBus | None = None):
        """
        Args:
            event_bus: Where flag changes are emitted; None emits nothing
        """
        # Backing fields; accessed through the properties below so that the
        # event bus is notified on every change
        for _flag in STATUS_FLAG_NAMES:
            setattr(self, f"_status_{_flag}", False)
        self.status_lock = threading.Lock()
        self.event_bus = event_bus

    def flags_snapshot(self) -> dict:
        """Return a {flag_name: bool} snapshot of all status flags."""
        with self.status_lock:
            return {flag: getattr(self, f"_status_{flag}") for flag in STATUS_FLAG_NAMES}

    def _notify_flags_changed(self):
        """Emit a flags event on the bus. Callers may hold status_lock, so this
        must not acquire it (single-threaded flag mutation makes the unlocked
        read a benign race)."""
        bus = self.event_bus
        if bus is not None:
            snapshot = {flag: getattr(self, f"_status_{flag}") for flag in STATUS_FLAG_NAMES}
            bus.emit("flags", {"flags": snapshot})


def _make_status_property(flag: str) -> property:
    """Create a property for a status flag that notifies the bus on change."""

    def getter(self):
        return getattr(self, f"_status_{flag}")

    def setter(self, value):
        value = bool(value)
        if getattr(self, f"_status_{flag}") == value:
            return
        setattr(self, f"_status_{flag}", value)
        self._notify_flags_changed()

    return property(getter, setter)


for _flag in STATUS_FLAG_NAMES:
    setattr(StatusManager, f"status_{_flag}", _make_status_property(_flag))
del _flag

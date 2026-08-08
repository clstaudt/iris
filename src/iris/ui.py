"""UI components for status display and message panels."""

import threading
import warnings

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from iris.events import EventBus

warnings.filterwarnings("ignore")

STATUS_FLAG_NAMES = (
    "listening",
    "looking",
    "capturing_screen",
    "searching",
    "browsing",
    "understanding",
    "thinking",
    "summarizing",
)


class DynamicStatusBar:
    """A renderable that re-evaluates status on each render."""

    def __init__(self, status_manager, speech_pipeline=None):
        """
        Initialize dynamic status bar.

        Args:
            status_manager: StatusManager instance
            speech_pipeline: Optional SpeechPipeline to check for speech activity
        """
        self.status_manager = status_manager
        self.speech_pipeline = speech_pipeline

    def __rich_console__(self, console, options):
        """Yield fresh status bar on each render."""
        yield self.status_manager.make_status_bar(self.speech_pipeline)


class StatusManager:
    """Manages status bar with activity indicators."""

    def __init__(self, console: Console):
        """
        Initialize status manager.

        Args:
            console: Rich Console instance
        """
        self.console = console

        # Status flags (backing fields; accessed through properties below so
        # that an attached EventBus is notified on every change)
        for _flag in STATUS_FLAG_NAMES:
            setattr(self, f"_status_{_flag}", False)
        self.status_lock = threading.Lock()

        # Optional event bus (attached by the GUI controller; None in CLI mode)
        self.event_bus: EventBus | None = None

        # Create persistent spinners (reused across updates for animation)
        self.spinner_listening = Spinner("dots", text="Listening", style="red")
        self.spinner_looking = Spinner("dots", text="Looking", style="bright_magenta")
        self.spinner_capturing_screen = Spinner("dots", text="Capturing Screen", style="bright_magenta")
        self.spinner_searching = Spinner("dots", text="Searching", style="bright_cyan")
        self.spinner_browsing = Spinner("dots", text="Browsing", style="bright_cyan")
        self.spinner_understanding = Spinner("dots", text="Understanding", style="magenta")
        self.spinner_thinking = Spinner("dots", text="Thinking", style="cyan")
        self.spinner_speaking = Spinner("dots", text="Speaking", style="yellow")
        self.spinner_summarizing = Spinner("dots", text="Summarizing", style="blue")

        # Live display
        self.status_bar_live = None
        self.status_bar_active = False

    def make_status_bar(self, speech_pipeline=None) -> Panel:
        """
        Create the bottom status bar showing active activities.

        Args:
            speech_pipeline: Optional SpeechPipeline to check for speech activity

        Returns:
            Panel with status indicators
        """
        status_parts = []

        with self.status_lock:
            if self.status_listening:
                status_parts.append(self.spinner_listening)
            if self.status_looking:
                status_parts.append(self.spinner_looking)
            if self.status_capturing_screen:
                status_parts.append(self.spinner_capturing_screen)
            if self.status_searching:
                status_parts.append(self.spinner_searching)
            if self.status_browsing:
                status_parts.append(self.spinner_browsing)
            if self.status_understanding:
                status_parts.append(self.spinner_understanding)
            if self.status_summarizing:
                status_parts.append(self.spinner_summarizing)
            if self.status_thinking:
                status_parts.append(self.spinner_thinking)

        # Check speech pipeline activities
        if speech_pipeline and speech_pipeline.is_active:
            status_parts.append(self.spinner_speaking)

        if not status_parts:
            status_text = Text("Ready", style="dim")
        else:
            # Create table to show spinners side by side
            table = Table.grid(padding=(0, 2))
            for spinner in status_parts:
                table.add_column()
            table.add_row(*status_parts)
            status_text = table

        return Panel(
            Align.center(status_text),
            border_style="dim",
            style="on #1a1a1a",
            padding=(0, 2),
        )

    def update_status_bar(self, speech_pipeline=None):
        """
        Update the status bar display.

        Args:
            speech_pipeline: Optional SpeechPipeline to check for speech activity
        """
        if self.status_bar_live and self.status_bar_active:
            self.status_bar_live.update(self.make_status_bar(speech_pipeline))

    def make_status_bar_with_level(self, level: float, speech_pipeline=None) -> Panel:
        """
        Create the bottom status bar with audio level indicator.

        Args:
            level: Audio level from 0.0 to 1.0
            speech_pipeline: Optional SpeechPipeline to check for speech activity

        Returns:
            Panel with status indicators and level meter
        """
        # Create slim level meter (thin bar characters)
        bar_width = 12
        filled = int(level * bar_width)

        # Color based on level
        if level < 0.3:
            bar_color = "red"
        elif level < 0.7:
            bar_color = "green"
        else:
            bar_color = "yellow"

        # Use thin bar characters for slim appearance
        level_text = Text()
        level_text.append("─" * filled, style=bar_color)
        level_text.append("─" * (bar_width - filled), style="dim")

        # Build status parts: spinner + level bar
        status_parts = [self.spinner_listening, level_text]

        # Check speech pipeline activities
        if speech_pipeline and speech_pipeline.is_active:
            status_parts.append(self.spinner_speaking)

        # Create table to show items side by side
        table = Table.grid(padding=(0, 2))
        for part in status_parts:
            table.add_column()
        table.add_row(*status_parts)

        return Panel(
            Align.center(table),
            border_style="dim",
            style="on #1a1a1a",
            padding=(0, 2),
        )

    def start(self, speech_pipeline=None):
        """
        Start the status bar display.

        Args:
            speech_pipeline: Optional SpeechPipeline to check for speech activity
        """
        if not self.status_bar_active:
            # Use DynamicStatusBar so speech state is re-evaluated on each refresh
            self.status_bar_live = Live(
                DynamicStatusBar(self, speech_pipeline),
                console=self.console,
                refresh_per_second=10,
                auto_refresh=True,
            )
            self.status_bar_live.start()
            self.status_bar_active = True

    def stop(self):
        """Stop the status bar display."""
        if self.status_bar_live and self.status_bar_active:
            self.status_bar_live.stop()
            self.status_bar_active = False

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


class DisplayManager:
    """Manages message display panels and streaming updates."""

    def __init__(self, console: Console, status_manager: StatusManager):
        """
        Initialize display manager.

        Args:
            console: Rich Console instance
            status_manager: StatusManager instance
        """
        self.console = console
        self.status_manager = status_manager

    def update_audio_level_display(self, level: float, speech_pipeline=None):
        """
        Update status bar to show audio input level visualization.

        Args:
            level: Audio level from 0.0 to 1.0
            speech_pipeline: Optional SpeechPipeline to check for speech activity
        """
        # Update the status bar with the level indicator
        if self.status_manager.status_bar_live and self.status_manager.status_bar_active:
            self.status_manager.status_bar_live.update(
                self.status_manager.make_status_bar_with_level(level, speech_pipeline)
            )

    def show_user_message(self, text: str):
        """
        Display user message panel.

        Args:
            text: User message text
        """
        user_panel = Panel(
            Text(text, style="bold cyan"),
            title="[bold cyan]You[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
            width=int(self.console.width * 0.85),
        )
        self.console.print(Align.left(user_panel, pad=False))
        self.console.print()

    def show_tool_activity(self, tool_name: str, detail: str = ""):
        """
        Display a compact indicator that a tool was used.

        Args:
            tool_name: Name of the tool (e.g. "search", "browse")
            detail: Brief detail string (e.g. query or URL)
        """
        labels = {
            "search": "Searched",
            "browse": "Browsed",
            "look": "Looked",
            "capture_screen": "Captured screen",
        }
        label = labels.get(tool_name, tool_name)
        content = f"{label}: {detail}" if detail else label

        panel = Panel(
            Text(content, style="dim"),
            border_style="dim",
            padding=(0, 1),
            expand=False,
        )

        if self.status_manager.status_bar_live and self.status_manager.status_bar_active:
            self.status_manager.status_bar_live.console.print(Align.center(panel))
        else:
            self.console.print(Align.center(panel))

    def update_streaming_display(self, message: str, speech_pipeline=None):
        """
        Update display during streaming to show message above status bar.

        Args:
            message: Current streaming message text
            speech_pipeline: Optional SpeechPipeline to check for speech activity
        """
        # Create assistant panel
        assistant_panel = Panel(
            Text(message, style="green"),
            title="[bold green]Assistant[/bold green]",
            border_style="green",
            padding=(0, 1),
            width=int(self.console.width * 0.85),
        )

        # Combine message and status bar
        display_group = Group(
            Align.right(assistant_panel, pad=False),
            Text(""),  # Blank line
            self.status_manager.make_status_bar(speech_pipeline),
        )

        # Update the existing Live display
        if self.status_manager.status_bar_live and self.status_manager.status_bar_active:
            self.status_manager.status_bar_live.update(display_group)

    def finalize_message(self, message: str, speech_pipeline=None):
        """
        Finalize streaming message display and restart status bar.

        Args:
            message: Final message text
            speech_pipeline: Optional SpeechPipeline to check for speech activity
        """
        # Show only the message (without status bar)
        if self.status_manager.status_bar_live and self.status_manager.status_bar_active:
            assistant_panel = Panel(
                Text(message, style="bold green"),
                title="[bold green]Assistant[/bold green]",
                border_style="green",
                padding=(0, 1),
                width=int(self.console.width * 0.85),
            )
            # Update to show only message
            self.status_manager.status_bar_live.update(Align.right(assistant_panel, pad=False))
            # Stop it - leaves just the message on screen
            self.status_manager.status_bar_live.stop()
            self.status_manager.status_bar_active = False

            # Add spacing
            self.console.print()

            # Start fresh status bar below
            self.status_manager.start(speech_pipeline)

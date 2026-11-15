"""The app's end-to-end test (ui/e2e/run.mjs), shown as it runs.

`npm run e2e` starts this, which starts the test's Node runner with its
events on (E2E_EVENTS, told in ui/e2e/report.mjs) and shows them with rich.
In a terminal: the steps under their areas as they end, a live line for the
step running, with how long it has taken and how long the rest should take,
failures in a panel, and a summary at the end. Elsewhere, in a log: the same
as plain lines. Arguments and the environment go on to the runner, but for

    --verbose   the core's own log lines too (also E2E_VERBOSE)

`npm run e2e:plain` runs the runner alone, in its own plain lines.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Table
from rich.text import Text

UI = Path(__file__).resolve().parents[2] / "ui"
PREFIX = "@@e2e "
SLOWEST = 5
# How many of the runner's own lines are kept, to show when it ends early
TAIL = 30
# Escape sequences and the control characters a log line may carry; a carriage
# return keeps only what was written after it, as a terminal would show
ESCAPES = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x00-\x08\x0b-\x1f\x7f]"
)


def clean(line: str) -> str:
    return ESCAPES.sub("", line.rstrip("\r\n").rsplit("\r", 1)[-1])


def seconds(value: float) -> str:
    """A step's duration, as the runner gives it."""
    return f"{value:.1f} s"


def duration(value: float) -> str:
    """A longer stretch of time: the run's, or what is left of it."""
    value = round(value)
    if value < 60:
        return f"{value} s"
    if value < 3600:
        return f"{value // 60} min {value % 60:02d} s"
    return f"{value // 3600} h {value % 3600 // 60:02d} min"


def left(value: float) -> str:
    if value < 2.5:
        return ""
    if value < 60:
        return f"about {max(5, round(value / 5) * 5)} s left"
    if value < 3600:
        return f"about {round(value / 60)} min left"
    return f"about {duration(value)} left"


def heading(group: str) -> str:
    return group[:1].upper() + group[1:]


@dataclass
class Step:
    name: str
    group: str
    ok: bool
    seconds: float
    reason: str = ""
    states: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    page: list[str] = field(default_factory=list)
    screenshot: str = ""

    @property
    def detail(self) -> str:
        """The reason with what happened meanwhile, on one line, as the runner tells it."""
        return (
            self.reason
            + f" [states: {' > '.join(self.states) or 'none'}]"
            + (f" [errors: {' | '.join(self.errors)}]" if self.errors else "")
            + (f" [page: {' | '.join(self.page)}]" if self.page else "")
        )


@dataclass
class Run:
    """What the runner told so far."""

    plan: list[dict] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    skipped: int = 0
    current: str | None = None
    current_group: str = ""
    current_since: float = 0.0
    started: set[str] = field(default_factory=set)
    began: float = field(default_factory=time.monotonic)
    done: bool = False
    tail: deque[str] = field(default_factory=lambda: deque(maxlen=TAIL))

    @property
    def total(self) -> int:
        planned = sum(1 for step in self.plan if step["run"])
        return max(planned, len(self.steps) + (1 if self.current else 0))

    @property
    def failures(self) -> list[Step]:
        return [step for step in self.steps if not step.ok]

    def typical(self, name: str) -> float | None:
        return next((step["typical"] for step in self.plan if step["name"] == name), None)

    def left(self, now: float) -> float | None:
        """How long the steps still to run should take, from their durations
        in earlier runs; a step with none counts as the mean of the others."""
        known = [step["typical"] for step in self.plan if step["typical"] is not None]
        measured = [step.seconds for step in self.steps if step.ok]
        guess = statistics.mean(known or measured) if known or measured else None
        if guess is None:
            return None
        ahead = [s for s in self.plan if s["run"] and s["name"] not in self.started]
        rest = sum(guess if s["typical"] is None else s["typical"] for s in ahead)
        if self.current:
            typical = self.typical(self.current)
            rest += max((guess if typical is None else typical) - (now - self.current_since), 0)
        return rest

    def rerun(self) -> str:
        """E2E_ONLY for the failed steps: for each, the shortest start of its
        name (three words at least) that no other step's name holds."""
        names = [step["name"].lower() for step in self.plan] or [s.name.lower() for s in self.steps]
        parts = []
        for step in self.failures:
            words = step.name.split(" ")
            part = step.name
            for count in range(3, len(words) + 1):
                candidate = " ".join(words[:count]).rstrip(",;:")
                lower = candidate.lower()
                if not any(lower in name for name in names if name != step.name.lower()):
                    part = candidate
                    break
            if part not in parts:
                parts.append(part)
        return f"E2E_ONLY={shlex.quote('|'.join(parts))} npm run e2e"


class PlainView:
    """The run as plain lines, for a log: no colours, no control codes."""

    def __init__(self, console: Console, run: Run, verbose: bool):
        self.console = console
        self.run = run
        self.verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def line(self, text: str = "") -> None:
        self.console.print(text, markup=False, soft_wrap=True)

    def log(self, source: str, line: str) -> None:
        if source == "core" and not self.verbose:
            return
        self.line(f"[core] {clean(line)}" if source == "core" else clean(line))

    def heading(self, group: str) -> None:
        self.line()
        self.line(heading(group))

    def step(self, step: Step, index: int) -> None:
        mark = "ok  " if step.ok else "FAIL"
        self.line(f"  {mark}  {step.name} ({seconds(step.seconds)}, {index}/{self.run.total})")
        if not step.ok:
            self.line(f"        {clean(step.detail)}")
            self.line(f"        {step.screenshot}")

    def summary(self, code: int, interrupted: bool) -> None:
        run = self.run
        self.line()
        self.ending(code, interrupted)
        failed = len(run.failures)
        passed = len(run.steps) - failed
        total = duration(time.monotonic() - run.began)
        self.line(f"{passed} passed, {failed} failed, {run.skipped} skipped in {total}")
        slowest = sorted(run.steps, key=lambda step: step.seconds, reverse=True)[:SLOWEST]
        if slowest:
            self.line()
            self.line("Slowest:")
            for step in slowest:
                self.line(f"  {seconds(step.seconds):>8}  {step.name}")
        if run.failures:
            self.line()
            self.line("Failures:")
            for step in run.failures:
                self.line(f"  FAIL  {step.name}")
                self.line(f"        {clean(step.detail)}")
                self.line(f"        {step.screenshot}")
            self.line()
            self.line(f"Rerun the failures: {run.rerun()}")

    def ending(self, code: int, interrupted: bool) -> None:
        if interrupted:
            self.line(f"Interrupted after {len(self.run.steps)} of {self.run.total} steps")
            self.line()
        elif not self.run.done:
            self.line(f"The runner stopped without finishing (exit code {code}); its last lines:")
            for line in self.run.tail:
                self.line(f"  {clean(line)}")
            self.line()


class TerminalView(PlainView):
    """The run in a terminal: a live line for the step running, colours, panels."""

    def __init__(self, console: Console, run: Run, verbose: bool):
        super().__init__(console, run, verbose)
        self.spinner = Spinner("dots", style="cyan")
        self.live = Live(
            console=console, get_renderable=self.status, refresh_per_second=12, transient=True
        )

    def __enter__(self):
        self.live.start()
        return self

    def __exit__(self, *_):
        self.live.stop()
        return False

    def status(self) -> Table:
        run = self.run
        now = time.monotonic()
        index = len(run.steps) + (1 if run.current else 0)
        counter = f"{index}/{run.total}" if run.total else ""
        # Widths that hold still as the numbers change
        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True, justify="right", min_width=2 * len(str(run.total)) + 1)
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True, overflow="ellipsis", ratio=1)
        grid.add_column(no_wrap=True, style="dim", justify="right", min_width=7)
        grid.add_column(no_wrap=True, style="dim", min_width=17)
        bar = ProgressBar(
            total=max(run.total, 1),
            completed=len(run.steps),
            width=16,
            complete_style="red" if run.failures else "green",
            finished_style="red" if run.failures else "green",
        )
        if run.current:
            name = Text(self.short(run.current, run.current_group))
            elapsed = seconds(now - run.current_since)
        else:
            if not run.steps:
                name = Text("starting", style="dim")
            elif len(run.steps) >= run.total:
                name = Text("finishing", style="dim")
            else:
                name = Text("")
            elapsed = ""
        rest = run.left(now)
        grid.add_row(
            self.spinner,
            counter,
            bar,
            name,
            elapsed,
            left(rest) if rest is not None else "",
        )
        return grid

    @staticmethod
    def short(name: str, group: str) -> str:
        """A step's name under its heading, without the heading's own word."""
        prefix = f"{group.lower()}: "
        return name[len(prefix) :] if name.lower().startswith(prefix) else name

    def log(self, source: str, line: str) -> None:
        if source == "core" and not self.verbose:
            return
        text = Text("  core  " if source == "core" else "  ", style="dim")
        text.append(clean(line), style="dim")
        # Wrapped by the terminal, not at its spaces: a path stays whole
        self.console.print(text, soft_wrap=True)

    def heading(self, group: str) -> None:
        self.console.print()
        self.console.print(Text(heading(group), style="bold"))

    def step(self, step: Step, index: int) -> None:
        row = Table.grid(padding=(0, 1))
        row.add_column(no_wrap=True)
        row.add_column()
        mark = Text("  ✓", style="green") if step.ok else Text("  ✗", style="red")
        name = Text.assemble(
            (self.short(step.name, step.group), "" if step.ok else "red"),
            (f"  {seconds(step.seconds)}", "dim"),
        )
        row.add_row(mark, name)
        self.console.print(row)
        if not step.ok:
            panel = Panel(self.failure(step), border_style="red", box=box.ROUNDED, expand=False)
            self.console.print(Padding(panel, (0, 0, 0, 4)))

    @staticmethod
    def link(path: str) -> Text:
        """A path a terminal opens by a click, as a link where it can."""
        return Text(
            path, style=Style(dim=True, link=Path(path).resolve().as_uri() if path else None)
        )

    def failure(self, step: Step) -> Group:
        """Why a step failed, what went on meanwhile, and its screenshot."""
        lines = [Text(clean(step.reason))]
        labelled = (
            ("states", " > ".join(step.states) or "none"),
            ("errors", " | ".join(step.errors)),
            ("page", " | ".join(step.page)),
        )
        for label, value in labelled:
            if value:
                lines.append(Text.assemble((f"{label:<7}", "dim"), clean(value)))
        lines.append(self.link(step.screenshot))
        return Group(*lines)

    def summary(self, code: int, interrupted: bool) -> None:
        run = self.run
        self.console.print()
        if interrupted:
            self.console.print(
                Text(f"Interrupted after {len(run.steps)} of {run.total} steps", style="bold")
            )
            self.console.print()
        elif not run.done:
            self.console.print(
                Panel(
                    Group(*(Text(clean(line)) for line in run.tail)),
                    title=Text(f"The runner stopped without finishing (exit code {code})"),
                    title_align="left",
                    border_style="red",
                    box=box.ROUNDED,
                )
            )
            self.console.print()
        failed = len(run.failures)
        passed = len(run.steps) - failed
        table = Table(
            title=Text("Summary", style="bold"),
            title_justify="left",
            box=box.ROUNDED,
            show_header=False,
            border_style="dim",
            padding=(0, 1),
        )
        table.add_column(style="dim", no_wrap=True)
        table.add_column(overflow="fold")
        counts = Text.assemble(
            (f"{passed} passed", "green" if passed else ""),
            ", ",
            (f"{failed} failed", "red" if failed else ""),
            ", ",
            f"{run.skipped} skipped",
        )
        table.add_row("Steps", counts)
        table.add_row("Time", duration(time.monotonic() - run.began))
        slowest = sorted(run.steps, key=lambda step: step.seconds, reverse=True)[:SLOWEST]
        if slowest:
            rows = Table.grid(padding=(0, 2))
            rows.add_column(justify="right", style="dim", no_wrap=True)
            rows.add_column()
            for step in slowest:
                rows.add_row(seconds(step.seconds), step.name)
            table.add_row("Slowest", rows)
        if run.failures:
            failures = []
            for step in run.failures:
                failures.append(Text(step.name, style="red"))
                failures.append(Padding(self.failure(step), (0, 0, 0, 2)))
            table.add_row("Failures", Group(*failures))
        self.console.print(table)
        if run.failures:
            # Outside the table, so that it copies as one line
            hint = Text.assemble(("Rerun the failures: ", "dim"), run.rerun())
            self.console.print(hint, soft_wrap=True)


def follow(lines: Iterable[str], run: Run, view: PlainView) -> None:
    """Show what the runner tells, line by line, as it comes."""
    shown = None
    for raw in lines:
        line = raw.rstrip("\n")
        event = None
        if line.startswith(PREFIX):
            try:
                event = json.loads(line[len(PREFIX) :])
            except ValueError:
                pass
        if not isinstance(event, dict):
            run.tail.append(line)
            view.log("node", line)
            continue
        kind = event.get("type")
        if kind == "run":
            run.plan = event.get("steps", [])
        elif kind == "start":
            run.current = event["name"]
            run.current_group = event["group"]
            run.current_since = time.monotonic()
            run.started.add(event["name"])
            if event["group"] != shown:
                shown = event["group"]
                view.heading(shown)
        elif kind == "step":
            step = Step(
                name=event["name"],
                group=event["group"],
                ok=event["ok"],
                seconds=event["seconds"],
                reason=event.get("reason", ""),
                states=event.get("states", []),
                errors=event.get("errors", []),
                page=event.get("page", []),
                screenshot=event.get("screenshot", ""),
            )
            run.steps.append(step)
            run.current = None
            view.step(step, len(run.steps))
        elif kind == "skip":
            run.skipped += 1
        elif kind == "log":
            if event.get("source") != "core":
                run.tail.append(event.get("line", ""))
            view.log(event.get("source", "runner"), event.get("line", ""))
        elif kind == "done":
            run.done = True


def exit_code(code: int) -> int:
    """A process's return code as a shell gives it: 128 and the signal for one ended by it."""
    return 128 - code if code < 0 else code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="npm run e2e --",
        description="Run the app's end-to-end test and show its steps as they run.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=bool(os.environ.get("E2E_VERBOSE")),
        help="show the core's own log lines too (E2E_VERBOSE)",
    )
    options, rest = parser.parse_known_args(argv)
    console = Console(highlight=False, emoji=False)
    run = Run()
    terminal = console.is_terminal and not console.is_dumb_terminal
    view = (TerminalView if terminal else PlainView)(console, run, options.verbose)
    process = subprocess.Popen(
        [shutil.which("node") or "node", "e2e/run.mjs", *rest],
        cwd=UI,
        env={**os.environ, "E2E_EVENTS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    interrupted = False
    with view:
        try:
            follow(process.stdout, run, view)
        except KeyboardInterrupt:
            # The runner had Ctrl+C too, and ends of it
            interrupted = True
    try:
        code = process.wait(timeout=30 if interrupted else None)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        process.kill()
        code = process.wait()
    view.summary(exit_code(code), interrupted)
    return 128 + signal.SIGINT if interrupted else exit_code(code)


if __name__ == "__main__":
    sys.exit(main())

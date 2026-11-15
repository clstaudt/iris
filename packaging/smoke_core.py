"""Smoke test of the frozen Python core in the built app.

The core is started as the app starts it and asked what the app asks first:
app.info, setup.info, plugins.list and persona.voices. Each must answer, and
the answers must show what a working bundle has: the shipped personas, MLX
for speech recognition, every shipped plugin, and both speech engines. Then
`iris-rpc --check` must find every engine. A bundle that lacks a library, or
one that signing broke, fails here instead of on someone's Mac.

    python packaging/smoke_core.py ui/release/mac-arm64/Iris.app
    python packaging/smoke_core.py packaging/dist/iris-rpc/iris-rpc
    python packaging/smoke_core.py .venv/bin/python packaging/iris_rpc.py

It needs no network and no models: the core runs offline in an empty home
folder, with empty model folders, and on macOS in a sandbox that refuses it
every connection over the network.
"""

import argparse
import json
import os
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# The frozen core inside the app, where the app starts it from
APP_CORE = Path("Contents/Resources/iris-rpc/iris-rpc")

# The first answer waits for the core to load (torch, MLX, ...), which takes
# longest the first time a new bundle runs, while macOS scans its libraries
FIRST_ANSWER_S = 300
ANSWER_S = 60
EXIT_S = 30
CHECK_S = 600

PLUGINS = ("memory", "camera", "screen", "web", "sleep")
# Kyutai's voices ship with Iris; Kokoro's come with its model, which this
# test does without
SPEECH_ENGINES = ("kyutai", "kokoro")

# Refuses the core every connection over IP; all else is as in the app
SANDBOX = "/usr/bin/sandbox-exec"
NO_NETWORK = "(version 1)(allow default)(deny network-outbound (remote ip))"


def core_command(given: list[str]) -> tuple[list[str], Path | None]:
    """The command that runs the core, and the folder it runs in (the
    frozen core's own, as in the app)."""
    if len(given) > 1:
        return given, None
    program = Path(os.path.abspath(given[0]))
    if program.suffix == ".app":
        program = program / APP_CORE
    if not program.is_file():
        sys.exit(f"No core at {program}")
    return [str(program)], program.parent


def hermetic_env(home: Path) -> dict:
    """A fresh computer's environment, offline: an empty home folder, no
    model, and none of this computer's Iris settings."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IRIS_", "OLLAMA_", "HF_"))}
    for folder in ("huggingface", "models", "tmp"):
        (home / folder).mkdir()
    env.update(
        HOME=str(home),
        TMPDIR=str(home / "tmp"),
        HF_HOME=str(home / "huggingface"),
        HF_HUB_OFFLINE="1",
        IRIS_MODELS_DIR=str(home / "models"),
        PYTHONUNBUFFERED="1",
    )
    return env


def sandboxed(command: list[str]) -> list[str]:
    if sys.platform == "darwin" and Path(SANDBOX).exists():
        return [SANDBOX, "-p", NO_NETWORK, *command]
    return command


class Core:
    """The core as the app runs it, answering JSON-RPC requests on stdin."""

    def __init__(self, command: list[str], env: dict, cwd: Path | None, log):
        self.proc = subprocess.Popen(
            sandboxed(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
        )
        self.answers: queue.Queue = queue.Queue()
        self.events: list[dict] = []
        # Lines on stdout that are no RPC frame: the app cannot read them,
        # and one written halfway through a frame would cost its answer
        self.stray: list[str] = []
        self.next_id = 1
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                frame = json.loads(line)
            except ValueError:
                self.stray.append(line.rstrip("\n"))
                continue
            if frame.get("method") == "event":
                self.events.append(frame.get("params") or {})
            else:
                self.answers.put(frame)
        self.answers.put(None)  # stdout closed: the core is gone

    def call(self, method: str, timeout: float) -> dict:
        request_id, self.next_id = self.next_id, self.next_id + 1
        request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": {}}
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                frame = self.answers.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise TimeoutError(f"no answer within {timeout:.0f} s") from None
            if frame is None:
                raise RuntimeError(f"the core exited with code {self.proc.wait()}")
            if frame.get("id") != request_id:
                continue
            if "error" in frame:
                raise RuntimeError(frame["error"].get("message") or "an error without a message")
            return frame.get("result") or {}

    def close(self) -> str | None:
        """Close its input, as the app's end does; why it did not end well, if so."""
        self.proc.stdin.close()
        try:
            code = self.proc.wait(timeout=EXIT_S)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            return f"it did not exit within {EXIT_S} s of its input closing"
        return f"it exited with code {code}" if code else None


def check_app_info(info: dict) -> tuple[str, list[str]]:
    personas = info.get("personas") or []
    problems = [] if info.get("version") else ["it tells no version"]
    if "iris" not in personas:
        problems.append(f"the shipped persona 'iris' is missing (found: {personas})")
    return f"version {info.get('version')}, personas {', '.join(personas)}", problems


def check_setup_info(info: dict) -> tuple[str, list[str]]:
    stt, tts = info.get("stt") or {}, info.get("tts") or {}
    problems = []
    if sys.platform == "darwin" and platform.machine() == "arm64" and stt.get("engine") != "mlx":
        problems.append(f"speech recognition runs on {stt.get('engine')!r}: MLX did not load")
    if stt.get("cached") or tts.get("found"):
        problems.append("it found a model, which this test must do without")
    return f"speech recognition on {stt.get('engine')}, model {stt.get('model')}", problems


def check_plugins(listed: dict) -> tuple[str, list[str]]:
    plugins = {plugin.get("id"): plugin for plugin in listed.get("plugins") or []}
    problems = []
    for plugin_id in PLUGINS:
        if plugin_id not in plugins:
            problems.append(f"the plugin '{plugin_id}' is missing")
        elif not plugins[plugin_id].get("available"):
            problems.append(f"the plugin '{plugin_id}' lacks what it needs in the bundle")
    problems += [f"a plugin failed to load: {failure}" for failure in listed.get("failures") or []]
    available = [plugin_id for plugin_id, plugin in plugins.items() if plugin.get("available")]
    return f"available: {', '.join(available)}", problems


def check_voices(listed: dict) -> tuple[str, list[str]]:
    engines = listed.get("engines") or {}
    problems = [f"the speech engine '{e}' is missing" for e in SPEECH_ENGINES if e not in engines]
    if "kyutai" in engines and not engines["kyutai"]:
        problems.append("Kyutai lists no voices")
    counts = ", ".join(f"{engine} {len(voices)}" for engine, voices in engines.items())
    return f"voices: {counts}", problems


CHECKS = {
    "app.info": check_app_info,
    "setup.info": check_setup_info,
    "plugins.list": check_plugins,
    "persona.voices": check_voices,
}


def report(ok: bool, what: str):
    print(f"{'ok    ' if ok else 'FAILED'}  {what}", flush=True)


def answers(command: list[str], env: dict, cwd: Path | None, log_path: Path) -> int:
    """Ask the core what the app asks first; the number of failures."""
    failed = 0
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log:
        core = Core(command, env, cwd, log)
        try:
            for i, (method, check) in enumerate(CHECKS.items()):
                asked = time.monotonic()
                try:
                    result = core.call(method, FIRST_ANSWER_S if i == 0 else ANSWER_S)
                except Exception as e:
                    failed += 1
                    report(False, f"{method}: {e}")
                    # A core that is gone or stuck answers nothing more
                    if isinstance(e, TimeoutError) or core.proc.poll() is not None:
                        break
                    continue
                summary, problems = check(result)
                took = time.monotonic() - asked
                report(not problems, f"{method} answered in {took:.1f} s; {summary}")
                for problem in problems:
                    print(f"          {problem}")
                failed += bool(problems)
        finally:
            ended = core.close() if core.proc.poll() is None else None
        if ended:
            failed += 1
            report(False, f"the core's end: {ended}")
        if core.stray:
            failed += 1
            report(False, f"the core wrote {len(core.stray)} line(s) to stdout outside RPC frames:")
            for line in core.stray[:10]:
                print(f"          {line[:200]}")
        for event in core.events:
            if event.get("type") == "error":
                print(f"note    the core told the app of an error: {event.get('data')}")
    print(f"        (the session took {time.monotonic() - started:.1f} s)")
    if failed:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"-- the core's stderr, last {min(len(lines), 80)} of {len(lines)} lines:")
        print("\n".join(lines[-80:]))
        print("--")
    return failed


def engines_check(command: list[str], env: dict, cwd: Path | None) -> int:
    """`iris-rpc --check`: every engine imports, and the voices' readers read."""
    print(f"== {' '.join([Path(command[-1]).name, '--check'])}", flush=True)
    try:
        done = subprocess.run(sandboxed([*command, "--check"]), cwd=cwd, env=env, timeout=CHECK_S)
    except subprocess.TimeoutExpired:
        report(False, f"--check did not finish within {CHECK_S} s")
        return 1
    report(done.returncode == 0, f"--check exited with code {done.returncode}")
    return int(done.returncode != 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "core",
        nargs="+",
        help="the built app (Iris.app), its iris-rpc, or a command running packaging/iris_rpc.py",
    )
    command, cwd = core_command(parser.parse_args().core)
    with tempfile.TemporaryDirectory(prefix="iris-smoke-") as home:
        env = hermetic_env(Path(home))
        print(f"== {' '.join(command)}", flush=True)
        failed = answers(command, env, cwd, Path(home) / "core.log")
        failed += engines_check(command, env, cwd)
    print("The core works." if not failed else f"{failed} check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

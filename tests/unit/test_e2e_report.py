"""The end-to-end test's reporter (irisdev.e2e_report), from events as the
runner (ui/e2e/report.mjs) tells them: in a log, plain lines a search finds,
with no colours or control codes; the failures to run again; the time left."""

import io
import json

from rich.console import Console

from irisdev import e2e_report

NAMES = [
    "first run: Iris starts with the choices, and greets the user",
    "threads: a new thread starts from its empty screen, with a persona chosen and heard",
    "threads: a restart opens the thread used last",
    "voices: a voice plays from its tile, through the editor's preview, and reaches the speaker",
]


def event(**fields):
    return e2e_report.PREFIX + json.dumps(fields) + "\n"


def group(name):
    return name.split(":")[0]


def stream():
    steps = [{"name": n, "group": group(n), "run": True, "typical": 2.0} for n in NAMES]
    yield event(type="run", steps=steps)
    yield event(type="log", source="app", line="[python-bridge] dev mode, spawning: python")
    yield event(type="log", source="core", line="Fetching 4 files: \x1b[1m0%\r100%")
    yield "a line Node wrote itself\n"
    for name in NAMES:
        yield event(type="start", name=name, group=group(name))
        if name == NAMES[2]:
            yield event(
                type="step",
                name=name,
                group=group(name),
                ok=False,
                seconds=8.4,
                reason="Timed out waiting for the reply",
                states=["thinking", "speaking"],
                errors=[],
                page=[],
                screenshot="/out/3-threads-a-restart.png",
            )
        else:
            yield event(type="step", name=name, group=group(name), ok=True, seconds=1.5)
    yield event(type="done", passed=3, failed=1, skipped=0, seconds=12.9)


def plain(verbose=False):
    out = io.StringIO()
    console = Console(file=out, highlight=False, emoji=False)
    run = e2e_report.Run()
    view = e2e_report.PlainView(console, run, verbose)
    with view:
        e2e_report.follow(stream(), run, view)
    view.summary(1, interrupted=False)
    return out.getvalue(), run


def test_a_log_holds_plain_lines_with_every_step_failure_and_summary():
    text, run = plain()
    lines = text.splitlines()
    assert "\x1b" not in text and "\r" not in text
    assert "[python-bridge] dev mode, spawning: python" in lines
    assert "a line Node wrote itself" in lines
    assert "Threads" in lines
    assert f"  ok    {NAMES[0]} (1.5 s, 1/4)" in lines
    assert f"  FAIL  {NAMES[2]} (8.4 s, 3/4)" in lines
    assert "        Timed out waiting for the reply [states: thinking > speaking]" in lines
    assert "        /out/3-threads-a-restart.png" in lines
    assert any(line.startswith("3 passed, 1 failed, 0 skipped in ") for line in lines)
    assert f"     8.4 s  {NAMES[2]}" in lines
    assert "Rerun the failures: E2E_ONLY='threads: a restart' npm run e2e" in lines
    assert run.done


def test_the_cores_lines_show_only_when_asked():
    assert "Fetching" not in plain()[0]
    assert "[core] 100%" in plain(verbose=True)[0].splitlines()


def test_a_rerun_names_each_failure_by_the_shortest_start_no_other_step_holds():
    run = e2e_report.Run(
        plan=[{"name": n, "group": group(n), "run": True, "typical": None} for n in NAMES]
    )
    run.steps = [
        e2e_report.Step(name=NAMES[1], group="threads", ok=False, seconds=1),
        e2e_report.Step(name=NAMES[3], group="voices", ok=False, seconds=1),
    ]
    # Three words at least; "threads: a new" is held by no other step
    assert run.rerun() == "E2E_ONLY='threads: a new|voices: a voice' npm run e2e"


def test_the_time_left_counts_the_steps_ahead_by_their_typical_durations():
    plan = [
        {"name": n, "group": group(n), "run": True, "typical": t}
        for n, t in zip(NAMES, (10, 20, None, 30))
    ]
    run = e2e_report.Run(plan=plan)
    run.current, run.current_since = NAMES[1], 100.0
    run.started = {NAMES[0], NAMES[1]}
    # 5 s of the running step's 20, the next as the others' mean (20), and 30
    assert run.left(now=115.0) == 5 + 20 + 30
    assert e2e_report.Run().left(now=0) is None

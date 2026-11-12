"""What the scenarios conclude across conversations, and the scenarios that
are built from a measurement (a delay calibrated on this computer, the
Whisper sizes downloaded here).

Each report function takes the results of the conversations held and gives
lines for the report; each run function holds conversations itself.
"""

from statistics import median
from typing import Callable

from irisdev.voicetest.conversations import (
    DISTANCES,
    OTHER_VOICES,
    WHISPER_SIZES,
    addition_while_thinking,
    whisper_cells,
)
from irisdev.voicetest.script import Conversation, Turn
from irisdev.voicetest.scripted import ScriptedConversations, TurnResult, stats

CALIBRATION = Conversation(
    "calibrating the time to the first word",
    [Turn(["What color is the sky on a clear day?"]), Turn(["How many legs does a spider have?"]),
     Turn(["What is the opposite of hot?"])],
    tags=["turn-taking", "calibration"],
)


def _by_conversation(results: list[TurnResult]) -> dict:
    found: dict = {}
    for result in results:
        found.setdefault(result.conversation, []).append(result)
    return found


def _passed(turns: list[TurnResult]) -> bool:
    return bool(turns) and all(r.passed and not r.skipped for r in turns)


def smallest_passing_shift(results: list[TurnResult]) -> list[str]:
    """Answering at once: the smallest shift of the gaps at which every
    check passed."""
    held = {name: turns for name, turns in _by_conversation(results).items() if name.startswith("answering at once")}
    if not held:
        return []
    passing = sorted(
        float(next((t.split()[1] for t in turns[0].tags if t.startswith("shift ")), "0"))
        for name, turns in held.items() if _passed(turns)
    )
    if not passing:
        return ["answering at once: no shift of the gaps passed every check"]
    return [f"answering at once: every check passed from a shift of +{passing[0]:g} s"]


def farthest_distance(results: list[TurnResult]) -> list[str]:
    """A distant user: the farthest distance at which every check passed."""
    held = _by_conversation(results)
    passing = [label for label in DISTANCES if _passed(held.get(f"a user {label}", []))]
    if not any(f"a user {label}" in held for label in DISTANCES):
        return []
    return [f"a distant user: every check passed up to {passing[-1] if passing else 'no distance'}"]


def per_voice(results: list[TurnResult]) -> list[str]:
    """Other voices: the median share heard, the worst first word and the
    worst barge-in for each voice."""
    lines = []
    for voice in OTHER_VOICES:
        turns = [r for r in results if voice in r.tags and not r.skipped]
        if not turns:
            continue
        shares = [r.heard_share for r in turns if r.heard]
        firsts = [r.first_word for r in turns if r.first_word is not None]
        barges = [r.barge_in for r in turns if r.barge_in is not None]
        lines.append(
            f"{voice}: heard {median(shares):.0%} (median)" if shares else f"{voice}: nothing heard"
        )
        if firsts:
            lines[-1] += f", first word at worst {max(firsts):.2f} s"
        if barges:
            lines[-1] += f", silent at worst {max(barges):.2f} s after the user began"
    return lines


def whisper_table(results: list[TurnResult], budget: float) -> list[str]:
    """Whisper's sizes by turn detection: p50 and p90 of the turn end,
    recognition, first word and voice pipeline; and the smallest size whose
    voice pipeline median is within budget with nine words in ten heard."""
    lines, good = [], []
    for size in WHISPER_SIZES:
        for mode in ("smart_turn", "silence"):
            turns = [r for r in results if f"cell {size} {mode}" in r.tags and not r.skipped]
            if not turns:
                continue

            def cell(values):
                found = stats(values)
                return f"{found['median']:.2f}/{found['p90']:.2f}" if found else "-"

            ends = cell(r.stages.get("turn_end", 0) / 1000 for r in turns if r.stages)
            recognition = cell(r.stages.get("recognition", 0) / 1000 for r in turns if r.stages)
            firsts = cell(r.first_word for r in turns)
            pipeline = stats(r.voice_pipeline for r in turns)
            lines.append(f"{size:>15} {mode:>10}: turn end {ends}, recognition {recognition}, first word {firsts}, "
                         f"voice pipeline {cell(r.voice_pipeline for r in turns)} (p50/p90 s)")
            heard = median([r.heard_share for r in turns]) if turns else 0
            if pipeline and pipeline["median"] <= budget and heard >= 0.9:
                good.append(size)
    if good:
        lines.append(f"the smallest size within budget, hearing nine words in ten: {good[0]}")
    return lines


def report_all(results: list[TurnResult], budgets: dict) -> list[str]:
    """Every conclusion the results allow."""
    return (smallest_passing_shift(results) + farthest_distance(results) + per_voice(results)
            + whisper_table(results, budgets.get("voice_pipeline", 2.5)))


def calibrated_addition(runner: ScriptedConversations) -> list[TurnResult]:
    """Going on while Iris thinks, just before it would answer: the time to
    the first word (T) is measured over three quick questions, then the
    addition is said T - 0.3 s after the question, where either a merged
    turn or a short answer to the first part passes."""
    calibration = runner.hold(CALIBRATION)
    firsts = [r.first_word for r in calibration if r.first_word is not None]
    if not firsts:
        return calibration
    delay = max(0.3, median(firsts) - 0.3)
    return calibration + runner.hold(addition_while_thinking(round(delay, 2), either=True))


def downloaded_whisper_sizes(assistant) -> list[str]:
    """The Whisper sizes on this computer, as Iris reports them."""
    downloaded = getattr(assistant, "stt_downloaded", None)
    return downloaded() if downloaded is not None else []


def whisper_matrix(runner: ScriptedConversations, report: Callable[[str], None] = print) -> list[TurnResult]:
    """Each downloaded Whisper size with each way of ending turns."""
    sizes = downloaded_whisper_sizes(runner.assistant)
    results: list[TurnResult] = []
    for conversation in whisper_cells([s for s in WHISPER_SIZES if s in sizes]):
        results += runner.hold(conversation)
    for line in whisper_table(results, runner.budgets["voice_pipeline"]):
        report(line)
    return results


# Runs the CLI can start by name
RUNS = {
    "calibrated-addition": calibrated_addition,
    "whisper-matrix": whisper_matrix,
}

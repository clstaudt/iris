"""Talk to Iris in a virtual room, and report how it went.

    python -m irisdev.voicetest conversations [--llm-host URL --llm-model NAME] [--only TEXT]
    python -m irisdev.voicetest selftalk --llm-host URL --llm-model NAME [--minutes 5]

Without a server, conversations use a scripted model (StubModel), which
tests the voice without a model's time; replies are then not checked. Iris
runs with a home folder of its own, so the user's personas, memory and
settings are left alone; the speech models already on this computer are
used. Nothing is played on the speakers or heard from the microphone.

Each run writes report.json and, per conversation, what each microphone
heard and each speaker played as WAV files, to listen to.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

REAL_HOME = Path.home()


def isolate() -> Path:
    """Give Iris a home folder of its own, before any of it is imported; the
    speech models stay where they are."""
    home = Path(tempfile.mkdtemp(prefix="iris-voicetest-"))
    os.environ.setdefault("HF_HOME", str(REAL_HOME / ".cache" / "huggingface"))
    os.environ.setdefault("IRIS_MODELS_DIR", str(REAL_HOME / ".iris" / "models"))
    os.environ.update(
        HOME=str(home),
        USERPROFILE=str(home),
        IRIS_MEMORY_DB=str(home / "memory.db"),
        IRIS_PLUGINS_DIR=str(home / "plugins"),
    )
    return home


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def save_audio(folder: Path, endpoints, start: float, end: float):
    """What each microphone heard and each speaker played."""
    import soundfile as sf

    from irisdev.voicetest.acoustics import RATE

    folder.mkdir(parents=True, exist_ok=True)
    for endpoint in endpoints:
        sf.write(folder / f"{slug(endpoint.name)}-microphone.wav", endpoint.recording(start, end), RATE)
        sf.write(folder / f"{slug(endpoint.name)}-speaker.wav", endpoint.speaker.recording(start, end), RATE)


def conversations(args, out: Path) -> int:
    import soundfile as sf

    from irisdev.voicetest import (
        CONVERSATIONS,
        IrisUnderTest,
        ScriptedConversations,
        Space,
        StubModel,
        UserVoice,
        summarize,
    )
    from irisdev.voicetest.acoustics import RATE
    from irisdev.voicetest.scripted import budgets_from
    from irisdev.voicetest.stub_llm import MODEL

    chosen = [c for c in CONVERSATIONS if not args.only or any(o.lower() in c.name for o in args.only)]
    if not chosen:
        print(f"No conversation matches {args.only}; there are: " + ", ".join(c.name for c in CONVERSATIONS))
        return 2
    stub = None if args.llm_host else StubModel(first_token=args.first_token)
    space = Space(echo_gain=args.echo_gain)
    endpoint = space.endpoint(args.persona)
    voice = UserVoice(args.voice_cache)
    iris = IrisUnderTest(endpoint, args.llm_host or stub.url, args.llm_model or MODEL, args.persona)
    print(f"Starting {args.persona} ({'scripted model' if stub else args.llm_model})")
    iris.start()
    warm = out / "warm.wav"
    sf.write(warm, voice.utterance(["Hello there."]), RATE)
    iris.warm_up(str(warm))
    warm.unlink()
    runner = ScriptedConversations(
        iris, space, voice, check_replies=stub is None, budgets=budgets_from(args.budgets),
        reply_timeout=args.reply_timeout or (300.0 if stub is None else 180.0),
    )
    results = []
    try:
        for round_ in range(args.repeat):
            for number, conversation in enumerate(chosen, 1):
                begun = time.monotonic()
                results += runner.hold(conversation)
                save_audio(out / f"{round_ + 1}-{number:02d}-{slug(conversation.name)}", [endpoint],
                           begun, time.monotonic())
    finally:
        iris.close()
        if stub:
            stub.close()
    summary = summarize(results)
    report = {
        "model": "scripted" if stub else args.llm_model,
        "persona": args.persona,
        "summary": summary,
        "turns": [asdict(r) | {"passed": r.passed} for r in results],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(_summary_lines(summary))
    print(f"Report and recordings: {out}")
    return 0 if summary["passed"] == summary["turns"] else 1


def selftalk(args, out: Path) -> int:
    from irisdev.voicetest import IrisUnderTest, SelfTalk, Space

    if not args.llm_host:
        print("Self-talk needs a real model: give --llm-host and --llm-model")
        return 2
    space = Space(echo_gain=args.echo_gain, across_gain=args.across_gain)
    first, second = args.personas
    a = IrisUnderTest(space.endpoint(first), args.llm_host, args.llm_model, first)
    b = IrisUnderTest(space.endpoint(second), args.llm_host, args.llm_model, second)
    print(f"Starting {first} and {second} ({args.llm_model})")
    a.start()
    b.start()
    opening = (
        f"You are talking out loud with another voice assistant, {b.assistant.persona.name}. "
        f"Greet it and ask it something you would like to know."
    )
    begun = time.monotonic()
    try:
        result = SelfTalk(a, b).run(opening, seconds=args.minutes * 60, replies=args.replies)
    finally:
        save_audio(out / "selftalk", space.endpoints, begun, time.monotonic())
        a.close()
        b.close()
    (out / "report.json").write_text(json.dumps(asdict(result), indent=2))
    shares = [e.heard_share for e in result.exchanges]
    gaps = sorted(e.gap for e in result.exchanges if e.gap is not None)
    print(
        f"{len(result.exchanges)} replies heard in {result.duration:.0f} s"
        + (f" · words heard: median {sorted(shares)[len(shares) // 2]:.0%}" if shares else "")
        + (f" · answered after: median {gaps[len(gaps) // 2]:.1f} s, worst {gaps[-1]:.1f} s" if gaps else "")
        + f" · both talking at once {result.overlap:.1f} s"
        + f" · cut itself off {result.self_cuts} · interrupted {result.interrupted}"
        + (" · the conversation stalled" if result.stalled else "")
    )
    for error in result.errors:
        print(f"  error: {error}")
    print(f"Report and recordings: {out}")
    return 0 if result.exchanges and not result.errors and not result.stalled else 1


def _summary_lines(summary: dict) -> str:
    def stat(name, label):
        value = summary.get(name)
        return f"{label}: median {value['median']:.2f} s, worst {value['worst']:.2f} s" if value else None

    parts = [f"{summary['passed']} of {summary['turns']} turns passed"]
    parts += [p for p in (stat("first_word", "first word"), stat("voice_pipeline", "voice pipeline"),
                          stat("barge_in", "silent after being interrupted"), stat("stop", "silent after Stop"))
              if p]
    if summary["taken_early"]:
        parts.append(f"turns taken too early: {summary['taken_early']}")
    return " · ".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m irisdev.voicetest", description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm-host", default=os.environ.get("IRIS_VOICETEST_HOST"),
                        help="the language model's server (or IRIS_VOICETEST_HOST)")
    parser.add_argument("--llm-model", default=os.environ.get("IRIS_VOICETEST_MODEL"),
                        help="the model (or IRIS_VOICETEST_MODEL)")
    parser.add_argument("--echo-gain", type=float, default=0.3, help="how loud a speaker is in its own microphone")
    parser.add_argument("--out", type=Path, default=Path("voicetest-output"))
    parser.add_argument("--voice-cache", type=Path, default=REAL_HOME / ".cache" / "iris-voicetest",
                        help="where the user's synthesized utterances are kept")
    commands = parser.add_subparsers(dest="command", required=True)

    talk = commands.add_parser("conversations", help="scripted conversations with one Iris")
    talk.add_argument("--only", action="append", help="conversations whose name contains this")
    talk.add_argument("--persona", default="iris")
    talk.add_argument("--repeat", type=int, default=1, help="hold the conversations this many times")
    talk.add_argument("--reply-timeout", type=float, help="seconds a reply may take")
    talk.add_argument("--first-token", type=float, default=0.3, help="the scripted model's first-token delay")
    talk.add_argument("--budgets", default=os.environ.get("IRIS_VOICETEST_BUDGETS", ""),
                      help="budgets to use, as voice_pipeline=3.5,barge_in=1 (or IRIS_VOICETEST_BUDGETS)")

    pair = commands.add_parser("selftalk", help="two Irises talking to each other")
    pair.add_argument("--personas", nargs=2, default=["iris", "hal"])
    pair.add_argument("--minutes", type=float, default=5.0)
    pair.add_argument("--replies", type=int, default=12, help="stop after this many replies")
    pair.add_argument("--across-gain", type=float, default=0.6, help="how loud each hears the other")

    args = parser.parse_args(argv)
    if bool(args.llm_host) != bool(args.llm_model):
        parser.error("give --llm-host and --llm-model together")
    out = args.out / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    isolate()
    return conversations(args, out) if args.command == "conversations" else selftalk(args, out)


if __name__ == "__main__":
    sys.exit(main())

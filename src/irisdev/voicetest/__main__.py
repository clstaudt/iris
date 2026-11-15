"""Talk to Iris in a virtual room, and report how it went.

    python -m irisdev.voicetest conversations [--group NAME] [--only TEXT] [--tag TAG]
    python -m irisdev.voicetest conversations --run whisper-matrix
    python -m irisdev.voicetest conversations --list
    python -m irisdev.voicetest selftalk --llm-host URL --llm-model NAME [--minutes 5]

Without a server, conversations use a scripted model (StubModel), which
tests the voice without a model's time; replies are then not checked, apart
from facts that come from the scripted model or the scripted web. Web
searches go to a scripted web (FakeWeb) unless --web real is given. Iris
keeps its personas, memory and plugins in a folder of its own, so the
user's are left alone; the speech models already on this computer are used
(irisdev.voicetest.isolation). Nothing is played on the speakers or heard
from the microphone.

Each run writes report.json and, per conversation, what each microphone
heard and each speaker played as WAV files, to listen to.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import soundfile as sf

from irisdev.voicetest import (
    IrisUnderTest,
    ScriptedConversations,
    SelfTalk,
    Space,
    StubModel,
    UserVoice,
    summarize,
)
from irisdev.voicetest.acoustics import RATE
from irisdev.voicetest.conversations import GROUPS, group
from irisdev.voicetest.resources import Resources
from irisdev.voicetest.runners import RUNS, report_all
from irisdev.voicetest.script import Echo, Noise
from irisdev.voicetest.scripted import budgets_from
from irisdev.voicetest.stub_llm import MODEL


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def save_audio(folder: Path, endpoints, start: float, end: float):
    """What each microphone heard and each speaker played."""
    folder.mkdir(parents=True, exist_ok=True)
    for endpoint in endpoints:
        sf.write(folder / f"{slug(endpoint.name)}-microphone.wav", endpoint.recording(start, end), RATE)
        sf.write(folder / f"{slug(endpoint.name)}-speaker.wav", endpoint.speaker.recording(start, end), RATE)


def choose(args) -> list:
    """The conversations asked for: of the groups given (basic by default),
    those whose name or a tag holds an --only text, with every --tag."""
    chosen = group(*(args.group or ["basic"]))
    if args.only:
        wanted = [o.lower() for o in args.only]
        chosen = [c for c in chosen if any(o in c.name.lower() or any(o == t.lower() for t in c.tags) for o in wanted)]
    if args.tag:
        chosen = [c for c in chosen if all(t in c.tags for t in args.tag)]
    if args.noise or args.echo_room is not None:
        changed = []
        for conversation in chosen:
            conversation = type(conversation)(**conversation.__dict__)
            if args.noise:
                sound, _, snr = args.noise.partition(":")
                conversation.noise = list(conversation.noise) + [Noise(sound, snr=float(snr or 15))]
            if args.echo_room is not None:
                conversation.echo = Echo(gain=args.echo_gain, room=args.echo_room, drive=args.echo_drive)
            changed.append(conversation)
        chosen = changed
    return chosen


def conversations(args, out: Path) -> int:
    if args.list:
        for name, members in GROUPS.items():
            print(f"{name}:")
            for conversation in members:
                print(f"  {conversation.name}  [{', '.join(conversation.tags)}]")
        return 0
    chosen = [] if args.run else choose(args)
    if not chosen and not args.run:
        print("No conversation matches; see --list for the groups and their conversations")
        return 2
    if args.web == "real":
        os.environ["IRIS_VOICETEST_WEB"] = "real"
    stub = None if args.llm_host else StubModel(first_token=args.first_token)
    space = Space(echo_gain=args.echo_gain, echo_room=args.echo_room, echo_drive=args.echo_drive)
    endpoint = space.endpoint(args.persona)
    voice = UserVoice(args.voice_cache)
    iris = IrisUnderTest(endpoint, args.llm_host or stub.url, args.llm_model or MODEL, args.persona, stub=stub)
    print(f"Starting {args.persona} ({'scripted model' if stub else args.llm_model})")
    iris.start()
    warm = out / "warm.wav"
    sf.write(warm, voice.utterance(["Hello there."]), RATE)
    iris.warm_up(str(warm))
    warm.unlink()
    runner = ScriptedConversations(
        iris, space, voice, check_replies=stub is None, budgets=budgets_from(args.budgets),
        reply_timeout=args.reply_timeout or (300.0 if stub is None else 180.0), stub=stub,
        resources=Resources(), transcribe_replies=args.transcribe,
    )
    results = []
    try:
        for round_ in range(args.repeat):
            for name in args.run or []:
                begun = time.monotonic()
                results += RUNS[name](runner)
                save_audio(out / f"{round_ + 1}-{slug(name)}", [endpoint], begun, time.monotonic())
            for number, conversation in enumerate(chosen, 1):
                print(f"{conversation.name}")
                begun = time.monotonic()
                results += runner.hold(conversation)
                save_audio(out / f"{round_ + 1}-{number:03d}-{slug(conversation.name)}", [endpoint],
                           begun, time.monotonic())
    finally:
        iris.close()
        if stub:
            stub.close()
    summary = summarize(results)
    conclusions = report_all(results, runner.budgets)
    report = {
        "model": "scripted" if stub else args.llm_model,
        "persona": args.persona,
        "web": args.web,
        "summary": summary,
        "conclusions": conclusions,
        "turns": [asdict(r) | {"passed": r.passed} for r in results],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(_summary_lines(summary))
    for line in conclusions:
        print(line)
    print(f"Report and recordings: {out}")
    return 0 if summary["passed"] == summary["turns"] else 1


def selftalk(args, out: Path) -> int:
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
        if not value:
            return None
        return f"{label}: median {value['median']:.2f} s, p90 {value['p90']:.2f} s, worst {value['worst']:.2f} s"

    parts = [f"{summary['passed']} of {summary['turns']} turns passed"]
    if summary.get("expected_failures"):
        parts.append(f"{summary['expected_failures']} with failures known today")
    parts += [p for p in (stat("first_word", "first word"), stat("voice_pipeline", "voice pipeline"),
                          stat("barge_in", "silent after being interrupted"), stat("stop", "silent after Stop"))
              if p]
    if summary["taken_early"]:
        parts.append(f"turns taken too early: {summary['taken_early']}")
    lines = [" · ".join(parts)]
    lines += [f"now passes: {entry}" for entry in summary.get("now_passes", [])]
    lines += [f"skipped: {entry}" for entry in summary.get("skipped", [])]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m irisdev.voicetest", description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm-host", default=os.environ.get("IRIS_VOICETEST_HOST"),
                        help="the language model's server (or IRIS_VOICETEST_HOST)")
    parser.add_argument("--llm-model", default=os.environ.get("IRIS_VOICETEST_MODEL"),
                        help="the model (or IRIS_VOICETEST_MODEL)")
    parser.add_argument("--echo-gain", type=float, default=0.3, help="how loud a speaker is in its own microphone")
    parser.add_argument("--echo-room", type=float, help="the speaker's echo rings in a room with this RT60 (seconds)")
    parser.add_argument("--echo-drive", type=float, help="the speaker distorts at this drive (tanh)")
    parser.add_argument("--out", type=Path, default=Path("voicetest-output"))
    parser.add_argument("--voice-cache", type=Path, default=Path.home() / ".cache" / "iris-voicetest",
                        help="where the user's synthesized utterances are kept")
    commands = parser.add_subparsers(dest="command", required=True)

    talk = commands.add_parser("conversations", help="scripted conversations with one Iris")
    talk.add_argument("--group", action="append", help="a group of conversations (basic by default; see --list)")
    talk.add_argument("--only", action="append", help="conversations whose name contains this, or with this tag")
    talk.add_argument("--tag", action="append", help="only conversations with this tag")
    talk.add_argument("--list", action="store_true", help="list the groups and their conversations")
    talk.add_argument("--run", action="append", choices=["calibrated-addition", "whisper-matrix"],
                      help="a scenario built from a measurement on this computer")
    talk.add_argument("--noise", help="a sound in the room for every conversation, as fan:15 (sound:SNR in dB)")
    talk.add_argument("--web", choices=["fake", "real"], default="fake",
                      help="search the scripted web, or the real one (or IRIS_VOICETEST_WEB=real)")
    talk.add_argument("--transcribe", action="store_true", help="transcribe every reply's voice afterwards")
    talk.add_argument("--persona", default="iris")
    talk.add_argument("--repeat", type=int, default=1, help="hold the conversations this many times")
    talk.add_argument("--reply-timeout", type=float, help="seconds a reply may take")
    talk.add_argument("--first-token", type=float, default=0.3, help="the scripted model's first-token delay")
    talk.add_argument("--budgets", default=os.environ.get("IRIS_VOICETEST_BUDGETS", ""),
                      help="budgets to use, as voice_pipeline=3.5,barge_in=1 (or IRIS_VOICETEST_BUDGETS)")

    pair = commands.add_parser("selftalk", help="two Irises talking to each other")
    pair.add_argument("--personas", nargs=2, default=["iris", "test-kokoro"])
    pair.add_argument("--minutes", type=float, default=5.0)
    pair.add_argument("--replies", type=int, default=12, help="stop after this many replies")
    pair.add_argument("--across-gain", type=float, default=0.6, help="how loud each hears the other")

    args = parser.parse_args(argv)
    if bool(args.llm_host) != bool(args.llm_model):
        parser.error("give --llm-host and --llm-model together")
    out = args.out / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return conversations(args, out) if args.command == "conversations" else selftalk(args, out)


if __name__ == "__main__":
    sys.exit(main())

"""How Iris's voice keeps time while it speaks, measured.

Iris speaks a reply a piece at a time: a thread synthesizes each piece while
the one before plays, and the player plays each as a clip. The voice comes
out uneven where that falls behind: a gap between two pieces when the next
is not ready, or the clip before takes long to end; a dropout inside a clip
when the audio callback cannot run in time.

This puts together what runs while Iris speaks a long reply, as the
assistant does: the speech pipeline and the player, the microphone listening
over the voice through the echo canceller and the VAD (holding the voice for
speech), and events for the app. It runs in a virtual room
(irisdev.voicetest.acoustics), where the voice comes back into the
microphone and nothing is heard. Given an output device nobody hears
(--output, a loopback device say), each clip is also played on it as silence
of the same length, by Iris's own speaker code, so a real audio callback, as
PortAudio calls it, is timed.

Measured:

- the real-time factor of synthesis: seconds to synthesize per second of
  audio, by piece (speak) and on its own (rtf);
- each gap between two pieces, and what made it: waiting for synthesis, the
  clip before ending (its stream's tail), the next one starting;
- dropouts inside a clip: the stream falling behind its own timeline, and
  PortAudio's underflow flags, with how long the callback waited for Python;
- the CPU each thread of the process used while speaking (macOS).

A slower computer is emulated by a steady load beside Iris (--load: busy
loops at normal priority taking that share of the performance cores), or by
Iris pushed towards the efficiency cores (--clamp utility, or background).
The busy loops end with the measurement, and after LIMIT seconds whatever
happens.

    python -m irisdev.speech_timing rtf --engine kokoro
    python -m irisdev.speech_timing speak --output <device> --load 0.25
"""

import argparse
import ctypes
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from iris.audio import AudioPlayer, AudioRecorder, SoundDevice, voice_log
from iris.echo import EchoCanceller
from iris.events import EventBus
from iris.llm import split_speech
from iris.tts import SpeechGenerator, SpeechPipeline
from iris.turns import SAMPLE_RATE, SmartTurn
from irisdev.voicetest.acoustics import Space

# A long reply as a model writes one: a short first sentence, then more,
# which Iris speaks in pieces of several sentences
REPLY = {
    "en": (
        "Here is a little story for you. The old lighthouse stood at the edge of the cliff, "
        "its lamp dark for as long as anyone could remember. Every evening the fishermen looked "
        "up at it on their way home, and wondered who had kept it burning in the years before "
        "the harbour was built. One autumn night, a light appeared in the tower again. It was "
        "faint at first, no more than a candle behind the glass, but it grew steadier as the "
        "hours went by. By midnight the whole village had come out onto the quay to see it. "
        "Nobody had a key to the door, and the path up the cliff had been washed away in a storm "
        "long ago. The children said it was a ghost, and the old men said it was the wind "
        "playing tricks with the moon. In the morning the light was gone, and the door was "
        "still locked. But on the step outside lay a basket of apples, and a note that said "
        "only: thank you for waiting."
    ),
    "fr": (
        "Voici une petite histoire pour vous. Le vieux phare se dressait au bord de la falaise, "
        "sa lampe éteinte depuis aussi longtemps que l'on s'en souvenait. Chaque soir, les "
        "pêcheurs le regardaient en rentrant au port, et se demandaient qui l'avait allumé "
        "autrefois. Une nuit d'automne, une lumière apparut de nouveau dans la tour. Elle était "
        "faible au début, à peine une bougie derrière la vitre, mais elle devint plus stable au "
        "fil des heures. À minuit, tout le village était sorti sur le quai pour la voir. "
        "Personne n'avait la clé de la porte, et le chemin de la falaise avait été emporté par "
        "une tempête il y a bien longtemps. Les enfants disaient que c'était un fantôme, et les "
        "anciens que c'était le vent qui jouait avec la lune. Au matin, la lumière avait "
        "disparu, et la porte était toujours fermée. Mais sur la marche, il y avait un panier "
        "de pommes, et un mot qui disait seulement : merci d'avoir attendu."
    ),
}
# For the real-time factor on its own: a short sentence, and a long piece
SHORT = {
    "en": "Sure, I can help you with that today.",
    "fr": "Bien sûr, je peux vous aider avec cela.",
}
LONG = {language: text[: text.index(".", 240) + 1] for language, text in REPLY.items()}

# A gap between two pieces longer than this is more than a pause between
# sentences
GAP = 0.25
# The stream fell this far behind its own timeline at a callback: audio was
# missing there
DROPOUT = 0.002
# Seconds a measurement may take at most; the load ends then whatever happens
LIMIT = 240.0


# Threads' CPU time (macOS) ------------------------------------------------------

_THREAD_EXTENDED_INFO, _THREAD_IDENTIFIER_INFO = 5, 4


class _ExtendedInfo(ctypes.Structure):
    _fields_ = [
        ("user_time", ctypes.c_uint64),
        ("system_time", ctypes.c_uint64),
        ("cpu_usage", ctypes.c_int32),
        ("policy", ctypes.c_int32),
        ("run_state", ctypes.c_int32),
        ("flags", ctypes.c_int32),
        ("sleep_time", ctypes.c_int32),
        ("curpri", ctypes.c_int32),
        ("priority", ctypes.c_int32),
        ("maxpriority", ctypes.c_int32),
        ("name", ctypes.c_char * 64),
    ]


class _IdentifierInfo(ctypes.Structure):
    _fields_ = [
        ("thread_id", ctypes.c_uint64),
        ("handle", ctypes.c_uint64),
        ("queue", ctypes.c_uint64),
    ]


_libc = ctypes.CDLL(None) if sys.platform == "darwin" else None


def _info(port: int, flavor: int, info: ctypes.Structure) -> bool:
    size = ctypes.c_uint(ctypes.sizeof(info) // 4)
    return _libc.thread_info(port, flavor, ctypes.byref(info), ctypes.byref(size)) == 0


def thread_times() -> dict[int, dict]:
    """Each thread of this process by its id: its name, the CPU seconds it
    used so far, and its scheduling priority. Empty but on macOS."""
    if _libc is None:
        return {}
    task = ctypes.c_uint.in_dll(_libc, "mach_task_self_").value
    threads = ctypes.POINTER(ctypes.c_uint)()
    count = ctypes.c_uint(0)
    if _libc.task_threads(task, ctypes.byref(threads), ctypes.byref(count)) != 0:
        return {}
    names = {t.native_id: t.name for t in threading.enumerate()}
    found = {}
    for i in range(count.value):
        port = threads[i]
        info, ident = _ExtendedInfo(), _IdentifierInfo()
        if _info(port, _THREAD_EXTENDED_INFO, info) and _info(port, _THREAD_IDENTIFIER_INFO, ident):
            name = names.get(ident.thread_id) or info.name.decode(errors="replace") or "(unnamed)"
            found[ident.thread_id] = {
                "name": name,
                "cpu": (info.user_time + info.system_time) / 1e9,
                "priority": info.curpri,
            }
        _libc.mach_port_deallocate(task, port)
    _libc.vm_deallocate(task, ctypes.cast(threads, ctypes.c_void_p), count.value * 4)
    return found


class ThreadSampler:
    """The CPU each thread uses from start() to stop(), sampled every
    quarter second so that threads which end in between (Pocket TTS starts
    two for each piece) count too; and the whole process's."""

    def __init__(self, every: float = 0.25):
        self.every = every
        self._first: dict[int, dict] = {}
        self._last: dict[int, dict] = {}
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._sample, name="thread sampler", daemon=True)

    def start(self):
        self._first = thread_times()
        self._last = dict(self._first)
        self._began, self._process = time.monotonic(), time.process_time()
        self._thread.start()

    def _sample(self):
        while not self._stopped.wait(self.every):
            self._last.update(thread_times())

    def stop(self) -> tuple[list[tuple[str, float, int]], float]:
        """The threads that used CPU (name, share of one core, priority),
        the busiest first, and the process's share of one core."""
        self._stopped.set()
        self._thread.join()
        self._last.update(thread_times())
        seconds = time.monotonic() - self._began
        used: dict[tuple[str, int], float] = {}
        for tid, last in self._last.items():
            key = (last["name"], last["priority"])
            used[key] = used.get(key, 0.0) + last["cpu"] - self._first.get(tid, {"cpu": 0.0})["cpu"]
        busy = [
            (name, cpu / seconds, priority)
            for (name, priority), cpu in used.items()
            if cpu > 0.005 * seconds
        ]
        process = (time.process_time() - self._process) / seconds
        return sorted(busy, key=lambda row: -row[1]), process


# A slower computer ----------------------------------------------------------------

# A busy loop: busy for its share of every 10 ms, until its time is up or the
# measurement that started it is gone
BUSY = """
import os, sys, time
duty, period, parent = float(sys.argv[1]), 0.01, os.getppid()
until = time.monotonic() + float(sys.argv[2])
while time.monotonic() < until and os.getppid() == parent:
    start = time.perf_counter()
    while time.perf_counter() - start < duty * period:
        pass
    time.sleep(period * (1 - duty))
"""
CLAMPS = {"utility": ["taskpolicy", "-c", "utility"], "background": ["taskpolicy", "-b"]}


def performance_cores() -> int:
    try:
        found = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"], capture_output=True, text=True, check=True
        )
        return int(found.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return os.cpu_count() or 1


class Load:
    """Busy loops at normal priority, one per performance core, each busy for
    this share of its time: together that share of the performance cores."""

    def __init__(self, share: float):
        self.share = share
        self._loops: list[subprocess.Popen] = []

    def __enter__(self):
        if self.share > 0:
            command = [sys.executable, "-c", BUSY, str(self.share), str(LIMIT)]
            self._loops = [subprocess.Popen(command) for _ in range(performance_cores())]
        return self

    def __exit__(self, *exc):
        for loop in self._loops:
            loop.terminate()
        for loop in self._loops:
            loop.wait()
        return False


# The player's output, timed ---------------------------------------------------------


@dataclass
class Callback:
    """A call of a stream's callback: when Python was entered, when PortAudio
    called it, when its first sample is due, its frames, and whether
    PortAudio flagged an underflow."""

    entered: float
    called: float
    due: float
    frames: int
    underflow: bool


@dataclass
class Clip:
    """A clip the player played: when it was asked to, its length and rate,
    and the callbacks of its stream (none in the virtual room alone)."""

    asked: float
    seconds: float
    rate: int
    latency: float = 0.0
    callbacks: list[Callback] = field(default_factory=list)

    @property
    def start(self) -> float:
        """When its first sample was due at the device. Told by the second
        callback where there is one: a stream that joins a device another
        stream plays on is told up to a cycle of the device early when its
        first block is due."""
        if len(self.callbacks) > 1:
            return self.callbacks[1].due - self.callbacks[0].frames / self.rate
        return self.callbacks[0].due if self.callbacks else self.asked + self.latency

    def slips(self) -> list[float]:
        """How far behind its own timeline the stream fell at each callback
        after the second (seconds): audio the device did not have in time."""
        out = []
        if len(self.callbacks) > 1:
            expected = self.callbacks[1].due
            for previous, callback in zip(self.callbacks[1:], self.callbacks[2:]):
                expected += previous.frames / self.rate
                out.append(max(0.0, callback.due - expected))
                expected = max(expected, callback.due)
        return out

    @property
    def end(self) -> float:
        """When its last sample was due."""
        return self.start + self.seconds + sum(self.slips())


class _TimedStream(sd.OutputStream):
    """sounddevice's output stream, its callback timed (into log)."""

    log: list = []

    def __init__(self, *args, callback=None, **kwargs):
        log = _TimedStream.log

        def timed(outdata, frames, time_info, status):
            log.append(Callback(time.monotonic(), time_info.currentTime,
                                time_info.outputBufferDacTime, frames,
                                bool(status.output_underflow)))
            callback(outdata, frames, time_info, status)

        super().__init__(*args, callback=timed, **kwargs)


class TimedOutput:
    """The room's endpoint as the player's device, with a real stream too:
    each clip is also played, as silence of the same length, on an output
    device by Iris's own SoundDevice, which says when it is over. The
    callbacks of its streams are timed."""

    def __init__(self, endpoint, output: str | None):
        self.endpoint = endpoint
        self.output = _output_device(output) if output is not None else None
        self.sound = SoundDevice()
        if self.output is not None:
            sd.OutputStream = _TimedStream
        self.clips: list[Clip] = []

    def __getattr__(self, name):
        return getattr(self.endpoint, name)

    def play(self, audio, sample_rate, **options) -> float:
        latency = self.endpoint.play(audio, sample_rate, **options)
        clip = Clip(time.monotonic(), len(audio) / sample_rate, sample_rate, latency)
        self.clips.append(clip)
        if self.output is None:
            return latency
        _TimedStream.log = clip.callbacks
        silence = np.zeros_like(audio)
        clip.latency = self.sound.play(silence, sample_rate, device=self.output, **options)
        return clip.latency

    def playing(self) -> bool:
        return self.sound.playing() if self.output is not None else self.endpoint.playing()

    def stop(self):
        self.endpoint.stop()
        if self.output is not None:
            self.sound.stop()


def _output_device(output: str) -> int:
    """The output device of this name (or index); never the speaker."""
    for index, device in enumerate(sd.query_devices()):
        named = output.lower() in device["name"].lower()
        if device["max_output_channels"] > 0 and (named or output == str(index)):
            if index == sd.default.device[1]:
                raise SystemExit(f"{device['name']} is the speaker: choose a device nobody hears")
            return index
    raise SystemExit(f"No output device named {output}")


# Speaking a reply -------------------------------------------------------------------


@dataclass
class Piece:
    """A piece of the reply: when it was synthesized and played, its audio's
    length and file, and its clip."""

    text: str
    synthesized: tuple[float, float] | None = None
    played: tuple[float, float] | None = None
    seconds: float = 0.0
    path: str = ""
    clip: Clip | None = None


class _Lines(logging.Handler):
    """The voice log's lines."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def speak(engine: str, voice: str, language: str, output: str | None, rate: float,
          listen: bool = True) -> dict:
    """Speak REPLY[language] as Iris does, streamed at rate characters a
    second, and measure it."""
    endpoint = Space().endpoint("Iris")
    device = TimedOutput(endpoint, output)
    echo = EchoCanceller()
    player = AudioPlayer(audio=device)
    player.echo = echo
    generator = SpeechGenerator(engine, voice, 1.25 if engine == "kokoro" else 1.0)
    generator.prepare_language(language)
    pipeline = SpeechPipeline(generator, player)
    # Events for the app, as the core writes them
    sink = open(os.devnull, "w")
    bus = EventBus()
    bus.subscribe(lambda kind, data: sink.write(json.dumps({"event": kind, "data": data}) + "\n"))
    pipeline.on_output_level = lambda level: bus.emit("audio.output", {"rms": level})

    pieces: list[Piece] = []
    generate, play = generator.generate, player.play

    def timed_generate(text, language="en", **options):
        piece = next(p for p in pieces if p.text == text and p.synthesized is None)
        started = time.monotonic()
        path = generate(text, language=language, **options)
        piece.synthesized = (started, time.monotonic())
        if path:
            piece.seconds, piece.path = sf.info(path).duration, path
        return path

    def timed_play(audio_file, **options):
        piece = next(p for p in pieces if p.path == audio_file)
        started, clips = time.monotonic(), len(device.clips)
        play(audio_file, **options)
        piece.played = (started, time.monotonic())
        piece.clip = device.clips[clips] if len(device.clips) > clips else None

    generator.generate, player.play = timed_generate, timed_play

    lines = _Lines()
    voice_log.addHandler(lines)
    voice_log.setLevel(logging.DEBUG)
    voice_log.propagate = False
    done = threading.Event()
    barge_ins: list[float] = []

    def taken_for_the_user() -> bool:
        """The voice was taken for the user speaking: Iris would stop its
        reply here. Counted, and the reply goes on."""
        barge_ins.append(time.monotonic())
        pipeline.resume()
        return True

    listener = None
    if listen:
        recorder = AudioRecorder(echo=echo, audio=endpoint, turn_end=SmartTurn.load(download=False))
        turn = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(turn, np.zeros(SAMPLE_RATE // 2, dtype=np.float32), SAMPLE_RATE)

        def listening():
            while not done.is_set():
                heard = recorder.listen_on(
                    turn,
                    on_speech=taken_for_the_user,
                    until=done.is_set,
                    should_stop=done.is_set,
                    on_level=lambda level: bus.emit("audio.level", {"rms": level}),
                    speaking=lambda: pipeline.is_active,
                    hold=pipeline.pause,
                    release=pipeline.resume,
                )
                if heard:
                    Path(heard).unlink(missing_ok=True)
            recorder.close_microphone()
            Path(turn).unlink(missing_ok=True)

        listener = threading.Thread(target=listening, name="listening", daemon=True)
        listener.start()
        time.sleep(1.0)

    text, buffer, last = REPLY[language], "", 0
    sampler = ThreadSampler()
    sampler.start()
    began = time.monotonic()
    step = max(1, int(rate * 0.05))
    for i in range(0, len(text), step):
        chunk = text[i : i + step]
        buffer += chunk
        bus.emit("chat.chunk", {"text": chunk})
        while True:
            piece, buffer = split_speech(buffer, last)
            if not piece:
                break
            last = len(piece)
            pieces.append(Piece(piece))
            pipeline.queue_speech(piece, language=language)
        time.sleep(0.05)
    if buffer.strip():
        pieces.append(Piece(buffer.strip()))
        pipeline.queue_speech(buffer.strip(), language=language)
    while pipeline.is_active and time.monotonic() < began + LIMIT - 30:
        time.sleep(0.1)
    finished = not pipeline.is_active
    pipeline.stop()
    seconds = time.monotonic() - began
    threads, process = sampler.stop()
    done.set()
    if listener is not None:
        listener.join(timeout=5)
    voice_log.removeHandler(lines)
    return {
        "pieces": pieces, "seconds": seconds, "finished": finished, "real": output is not None,
        "threads": threads, "process": process, "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "holds": [line for line in lines.lines if line.startswith("held")],
        "barge_ins": barge_ins, "lines": lines.lines,
    }


def report(result: dict):
    pieces = result["pieces"]
    print(f"{'piece':>5} {'chars':>5} {'audio':>6} {'synth':>6} {'RTF':>5} {'gap':>6} "
          f"{'waited':>6} {'tail':>6} {'start':>6} {'slips':>6} {'wait max':>8}")
    gaps, dropouts, waits = [], [], []
    for i, piece in enumerate(pieces):
        synth = piece.synthesized[1] - piece.synthesized[0] if piece.synthesized else 0.0
        clip, gap, waited, tail, start = piece.clip, 0.0, 0.0, 0.0, 0.0
        if i and clip is not None and pieces[i - 1].clip is not None:
            # The gap, and what made it: waiting for this piece to be
            # synthesized, the clip before ending, this one starting
            before = pieces[i - 1]
            gap = clip.start - before.clip.end
            waited = max(0.0, piece.synthesized[1] - before.played[1])
            tail = before.played[1] - before.clip.end
            start = clip.start - piece.played[0]
            gaps.append((gap, waited, tail, start))
        slips = clip.slips() if clip else []
        dropouts += [s for s in slips if s > DROPOUT]
        late = [c.entered - c.called for c in (clip.callbacks if clip else [])]
        waits += late
        rtf = synth / piece.seconds if piece.seconds else 0.0
        print(f"{i:5d} {len(piece.text):5d} {piece.seconds:6.2f} {synth:6.2f} {rtf:5.2f} "
              f"{gap:6.2f} {waited:6.2f} {tail:6.2f} {start:6.2f} {sum(slips):6.3f} "
              f"{max(late, default=0.0) * 1000:6.1f}ms")
    rtfs = [(p.synthesized[1] - p.synthesized[0]) / p.seconds for p in pieces if p.seconds]
    print(f"\nspoke {sum(p.seconds for p in pieces):.1f} s of audio in {result['seconds']:.1f} s")
    if not result["finished"]:
        print(f"THE REPLY WAS NOT SPOKEN TO ITS END within {LIMIT - 30:.0f} s")
    print(f"RTF while speaking: mean {np.mean(rtfs):.2f}, max {max(rtfs):.2f}")
    if gaps:
        print(f"gaps between pieces: mean {np.mean([g[0] for g in gaps]):.2f} s, "
              f"max {max(g[0] for g in gaps):.2f} s, "
              f"{sum(g[0] > GAP for g in gaps)} of {len(gaps)} over {GAP} s; of them waiting for "
              f"synthesis {sum(g[1] for g in gaps):.2f} s, stream tails "
              f"{sum(g[2] for g in gaps):.2f} s, starts {sum(g[3] for g in gaps):.2f} s")
    if result["real"]:
        flagged = sum(c.underflow for p in pieces if p.clip for c in p.clip.callbacks)
        print(f"dropouts inside clips: {len(dropouts)} ({sum(dropouts):.3f} s), underflow flags "
              f"{flagged}; callback waited for Python up to {max(waits, default=0) * 1000:.1f} ms "
              f"({sum(w > 0.010 for w in waits)} of {len(waits)} over 10 ms)")
    print(f"holds of the voice: {len(result['holds'])}, the voice taken for the user "
          f"(Iris would stop): {len(result['barge_ins'])}")
    for line in result["lines"][:12]:
        print(f"  voice log: {line}")
    print(f"CPU while speaking: {result['process'] * 100:.0f}% of one core in all, at nice "
          f"{result['nice']}; by thread (share of one core, priority):")
    for name, share, priority in result["threads"][:12]:
        print(f"  {share * 100:5.1f}%  pri {priority:3d}  {name}")


def measure_rtf(engine: str, voice: str, language: str, reps: int = 3):
    """The real-time factor of synthesis on its own, for a short sentence
    and a long piece."""
    generator = SpeechGenerator(engine, voice, 1.25 if engine == "kokoro" else 1.0)
    generator.prepare_language(language)
    for label, text in (("short", SHORT[language]), ("long", LONG[language])):
        took = []
        for _ in range(reps):
            started = time.monotonic()
            path = generator.generate(text, language=language)
            took.append(time.monotonic() - started)
            seconds = sf.info(path).duration
            Path(path).unlink()
        print(f"{engine}:{voice} {language} {label}: {seconds:.2f} s of audio in "
              f"{np.mean(took):.2f} s (RTF {np.mean(took) / seconds:.2f}, "
              f"best {min(took) / seconds:.2f})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("rtf", "speak"))
    parser.add_argument("--engine", default="kyutai")
    parser.add_argument("--voice", default="")
    parser.add_argument("--language", default="en", choices=sorted(REPLY))
    parser.add_argument("--output", help="an output device nobody hears, for a real stream")
    parser.add_argument("--rate", type=float, default=150.0,
                        help="characters a second the reply streams at")
    parser.add_argument("--load", type=float, default=0.0,
                        help="share of the performance cores kept busy")
    parser.add_argument("--clamp", choices=sorted(CLAMPS))
    parser.add_argument("--quiet-room", action="store_true", help="speak without listening")
    args = parser.parse_args(argv)
    voice = args.voice or ("af_bella" if args.engine == "kokoro" else "eve")
    with Load(args.load):
        if args.clamp:
            # Measured in a process of its own, the load beside it unclamped
            child = list(sys.argv[1:] if argv is None else argv)
            for option in ("--clamp", "--load"):
                if option in child:
                    del child[child.index(option) : child.index(option) + 2]
            command = [sys.executable, "-m", "irisdev.speech_timing", *child]
            return subprocess.call(CLAMPS[args.clamp] + command)
        if args.what == "rtf":
            measure_rtf(args.engine, voice, args.language)
        else:
            report(speak(args.engine, voice, args.language, args.output, args.rate,
                         not args.quiet_room))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())

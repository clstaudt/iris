"""The audio devices changing while Iris runs: AirPods connecting or
disconnecting change the devices, and the system's defaults, underneath
PortAudio, which read them once.

A speaker that cannot be opened has the devices read anew and the clip
played once more, on the default as it is now; a microphone whose stream is
closed for that, or stops delivering, is opened again, the one chosen found
by its name. Neither is tried again and again. On macOS a change of the
system's defaults is followed before it fails. A reply the speaker cannot
play is told once, the rest of it is not spoken, and Iris goes on; the
assistant's loop survives whatever fails in it.

sounddevice is replaced by a stand-in: no device is opened, nothing sounds.
"""

import queue
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice as sd
import soundfile as sf

from iris import audio
from iris.assistant import VoiceAssistant
from iris.audio import AudioPlayer, AudioRecorder, SoundDevice, SpeakerError
from iris.echo import STEP
from iris.events import EventBus
from iris.tts import SpeechPipeline
from iris.turns import SAMPLE_RATE

BLOCK = 3 * STEP

AIRPODS, MICROPHONE, SPEAKERS = "AirPods", "Built-in Microphone", "Built-in Speakers"
USB = "USB Microphone"
# What each microphone hears, to tell which one a frame came from
LEVELS = {AIRPODS: 0.1, MICROPHONE: 0.3, USB: 0.5}


def _wait_for(condition, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return bool(condition())


class Computer:
    """The computer's audio devices, and PortAudio's view of them, as
    sounddevice gives it: the devices and the defaults as they were when it
    was last initialized.

    A clip played on a device that is gone fails as on macOS ("Internal
    PortAudio error"), as does opening a microphone that is gone; one that
    goes away while open stops delivering.
    """

    def __init__(self, devices: dict, default_input: str, default_output: str):
        self.devices = dict(devices)  # name: (inputs, outputs)
        self.default_input, self.default_output = default_input, default_output
        # Microphones that are there but deliver nothing
        self.silent: set[str] = set()
        self.output_broken = False
        self.initialized = 0
        self.played: list[str] = []
        self.plays = 0
        self.streams: list[Stream] = []
        self._initialize()

    # The computer --------------------------------------------------------

    def plug(self, name: str, inputs: int, outputs: int, first=False, default=False):
        devices = {name: (inputs, outputs)}
        self.devices = {**devices, **self.devices} if first else {**self.devices, **devices}
        if default:
            self.default_input = name if inputs else self.default_input
            self.default_output = name if outputs else self.default_output

    def unplug(self, name: str, default_input=None, default_output=None):
        del self.devices[name]
        if self.default_input == name:
            self.default_input = None
        if self.default_output == name:
            self.default_output = None
        self.default_input = default_input or self.default_input
        self.default_output = default_output or self.default_output

    def system_defaults(self):
        return (self.default_input, self.default_output)

    # PortAudio, as sounddevice exposes it --------------------------------

    def _initialize(self):
        self.initialized += 1
        self.known = [
            {"name": name, "max_input_channels": inputs, "max_output_channels": outputs}
            for name, (inputs, outputs) in self.devices.items()
        ]
        names = [d["name"] for d in self.known]

        def index(name):
            return names.index(name) if name in names else None

        self.known_input, self.known_output = index(self.default_input), index(self.default_output)

    def _terminate(self):
        pass

    def query_devices(self, device=None, kind=None):
        if kind is not None:
            index = self.known_input if kind == "input" else self.known_output
            if index is None:
                raise sd.PortAudioError("Error querying device -1")
            return dict(self.known[index])
        if device is not None:
            return dict(self.known[device])
        return [dict(d) for d in self.known]

    def play(self, data, samplerate, **options):
        self.plays += 1
        name = self.known[self.known_output]["name"] if self.known_output is not None else None
        if self.output_broken or name not in self.devices:
            raise sd.PortAudioError("Error opening OutputStream: Internal PortAudio error", -9986)
        self.played.append(name)

    def stop(self, ignore_errors=True):
        pass

    def get_stream(self):
        return SimpleNamespace(latency=0.0, active=False, abort=lambda: None)

    def InputStream(self, device=None, callback=None, **options):  # noqa: N802 (sounddevice's)
        index = self.known_input if device is None else device
        name = self.known[index]["name"] if index is not None else None
        if name not in self.devices:
            raise sd.PortAudioError("Error opening InputStream: Invalid device", -9996)
        stream = Stream(self, name, callback)
        self.streams.append(stream)
        return stream


class Stream:
    """An input stream: a block of its microphone's level every 30 ms while
    the microphone is there and open."""

    def __init__(self, computer: Computer, name: str, callback):
        self.computer, self.name, self.callback = computer, name, callback
        self.open = self.closed = False

    def __enter__(self):
        self.open = True
        threading.Thread(target=self._feed, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.open, self.closed = False, True

    def _feed(self):
        while self.open:
            if self.name in self.computer.devices and self.name not in self.computer.silent:
                now = time.monotonic()
                info = SimpleNamespace(currentTime=now, inputBufferAdcTime=now - 0.01)
                block = np.full((BLOCK, 1), LEVELS[self.name], dtype=np.float32)
                self.callback(block, BLOCK, info, None)
            time.sleep(BLOCK / SAMPLE_RATE)


@pytest.fixture
def computer(monkeypatch):
    """AirPods, the computer's own microphone and speakers; the AirPods are
    the defaults. Defaults that change are not followed unless a test says
    so (as off macOS)."""
    machine = Computer(
        {MICROPHONE: (1, 0), SPEAKERS: (0, 2), AIRPODS: (1, 2)},
        default_input=AIRPODS,
        default_output=AIRPODS,
    )
    portaudio = ("_initialize", "_terminate", "query_devices", "play", "stop", "get_stream")
    for name in (*portaudio, "InputStream"):
        monkeypatch.setattr(audio.sd, name, getattr(machine, name))
    monkeypatch.setattr(audio, "system_defaults", lambda: None)
    monkeypatch.setattr(audio, "_PORTAUDIO", audio._PortAudio())
    machine.initialized = 0
    yield machine
    for stream in machine.streams:
        stream.open = False


def _follow_defaults(monkeypatch, computer: Computer):
    """As on macOS: the system's defaults can be told."""
    monkeypatch.setattr(audio, "system_defaults", computer.system_defaults)
    audio._PORTAUDIO.defaults = computer.system_defaults()


def _heard(frames, level: float, timeout=5.0) -> bool:
    """Whether frames of this level come within the timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            frame = frames.get(timeout=0.05)
        except queue.Empty:
            continue
        if np.isclose(frame, level).all():
            return True
    return False


class Quiet:
    """A VAD that hears no speech."""

    probability = 0.0

    def reset(self):
        pass

    def __call__(self, frame):
        return 0.0


# The speaker ------------------------------------------------------------------


def test_a_speaker_that_is_gone_is_found_where_it_is_now(computer):
    """The AirPods disconnect: PortAudio still opens them, as its default,
    and fails. The devices are read anew and the clip plays on the
    computer's speakers."""
    computer.unplug(AIRPODS, default_input=MICROPHONE, default_output=SPEAKERS)
    SoundDevice().play(np.zeros(240, dtype=np.float32), 24000)
    assert computer.played == [SPEAKERS]
    assert computer.initialized == 1  # read anew once
    assert SoundDevice().generation == 1


def test_a_speaker_that_cannot_be_opened_is_tried_once_more_only(computer):
    computer.output_broken = True
    with pytest.raises(sd.PortAudioError):
        SoundDevice().play(np.zeros(240, dtype=np.float32), 24000)
    assert computer.plays == 2 and computer.initialized == 1

    clip = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(clip, np.full(2400, 0.1, dtype=np.float32), 24000)
    with pytest.raises(SpeakerError, match="PaErrorCode -9986"):
        AudioPlayer(audio=SoundDevice()).play(clip)
    assert computer.plays == 4 and computer.initialized == 2


def test_on_macos_a_new_default_speaker_is_followed_before_the_clip(computer, monkeypatch):
    """The AirPods connect and become the default: the computer's speakers
    are still there, so nothing fails, yet the clip plays in the AirPods."""
    computer.unplug(AIRPODS, default_input=MICROPHONE, default_output=SPEAKERS)
    SoundDevice().refresh()
    _follow_defaults(monkeypatch, computer)
    SoundDevice().play(np.zeros(240, dtype=np.float32), 24000)
    assert computer.played == [SPEAKERS]

    computer.plug(AIRPODS, 1, 2, default=True)
    SoundDevice().play(np.zeros(240, dtype=np.float32), 24000)
    assert computer.played == [SPEAKERS, AIRPODS]
    assert computer.plays == 2  # no clip failed for it
    # Once followed, the clips after it do not read the devices again
    SoundDevice().play(np.zeros(240, dtype=np.float32), 24000)
    assert computer.initialized == 2


# The microphone ---------------------------------------------------------------


def test_the_microphone_open_when_the_speaker_fails_is_opened_again(computer):
    """Reading the devices anew closes every stream, the microphone's too:
    the listener goes on, on the new default."""
    recorder = AudioRecorder(audio=SoundDevice())
    try:
        with recorder._microphone() as frames:
            assert _heard(frames, LEVELS[AIRPODS])
            computer.unplug(AIRPODS, default_input=MICROPHONE, default_output=SPEAKERS)
            SoundDevice().play(np.zeros(240, dtype=np.float32), 24000)
            assert _heard(frames, LEVELS[MICROPHONE])
        assert computer.streams[0].closed
        assert [s.name for s in computer.streams] == [AIRPODS, MICROPHONE]
        assert computer.initialized == 1
    finally:
        recorder.close_microphone()


def test_a_microphone_that_stops_delivering_is_opened_where_it_is_now(computer, monkeypatch):
    monkeypatch.setattr(audio, "STALLED", 0.3)
    recorder = AudioRecorder(audio=SoundDevice())
    try:
        with recorder._microphone() as frames:
            assert _heard(frames, LEVELS[AIRPODS])
            # Gone while open: its stream just stops delivering
            computer.unplug(AIRPODS, default_input=MICROPHONE, default_output=SPEAKERS)
            assert _heard(frames, LEVELS[MICROPHONE])
        assert computer.initialized == 1
        assert [s.name for s in computer.streams] == [AIRPODS, MICROPHONE]
        assert computer.streams[0].closed
    finally:
        recorder.close_microphone()


def test_a_microphone_that_delivers_nothing_is_given_up(computer, monkeypatch):
    """Opened again after it stopped delivering, the microphone delivers
    nothing either: it is given up, not opened again and again."""
    monkeypatch.setattr(audio, "STALLED", 0.3)
    monkeypatch.setattr(audio, "STARTING", 0.3)
    monkeypatch.setattr(audio, "SpeechDetector", Quiet)
    computer.silent.add(MICROPHONE)
    recorder = AudioRecorder(audio=SoundDevice())
    give_up = time.monotonic() + 10

    def unplug_soon():
        time.sleep(0.3)
        computer.unplug(AIRPODS, default_input=MICROPHONE, default_output=SPEAKERS)

    threading.Thread(target=unplug_soon, daemon=True).start()
    try:
        assert recorder.listen(should_stop=lambda: time.monotonic() > give_up) is None
        assert time.monotonic() < give_up
        assert recorder._device_error
        assert [s.name for s in computer.streams] == [AIRPODS, MICROPHONE]
        assert computer.initialized == 1
    finally:
        recorder.close_microphone()


def test_a_microphone_that_cannot_be_opened_is_tried_once_more_only(computer, monkeypatch):
    monkeypatch.setattr(audio, "SpeechDetector", Quiet)
    recorder = AudioRecorder(audio=SoundDevice())
    # PortAudio's default is gone, and there is no microphone left
    computer.unplug(MICROPHONE)
    computer.unplug(AIRPODS, default_output=SPEAKERS)
    assert recorder.listen(should_stop=lambda: False) is None
    assert recorder._device_error
    assert computer.streams == [] and computer.initialized == 1


def test_the_microphone_chosen_is_found_again_by_its_name(computer):
    computer.plug(USB, 1, 0)
    SoundDevice().refresh()
    recorder = AudioRecorder(audio=SoundDevice())
    recorder.set_device(3)
    assert recorder.current_device_name() == USB
    try:
        # A device connects ahead of it: its index changes
        computer.plug("Headset", 1, 2, first=True, default=True)
        LEVELS["Headset"] = 0.7
        SoundDevice().refresh()
        assert recorder.current_device_name() == USB
        with recorder._microphone() as frames:
            assert _heard(frames, LEVELS[USB])
        assert computer.streams[-1].name == USB

        # Gone: the default is listened to until it is back
        computer.unplug(USB)
        SoundDevice().refresh()
        assert recorder.current_device_name() == "Headset"
        with recorder._microphone() as frames:
            assert _heard(frames, LEVELS["Headset"])
    finally:
        recorder.close_microphone()
        LEVELS.pop("Headset")


def test_on_macos_a_new_default_microphone_is_followed_while_listening(computer, monkeypatch):
    """The user picks the AirPods while Iris listens to the computer's
    microphone, which goes on delivering: Iris listens to the AirPods."""
    monkeypatch.setattr(audio, "FOLLOW_EVERY", 0.1)
    computer.unplug(AIRPODS, default_input=MICROPHONE, default_output=SPEAKERS)
    SoundDevice().refresh()
    _follow_defaults(monkeypatch, computer)
    recorder = AudioRecorder(audio=SoundDevice())
    try:
        with recorder._microphone() as frames:
            assert _heard(frames, LEVELS[MICROPHONE])
            computer.plug(AIRPODS, 1, 2, default=True)
            assert _heard(frames, LEVELS[AIRPODS])
        assert [s.name for s in computer.streams] == [MICROPHONE, AIRPODS]
        assert computer.initialized == 2  # read anew once for it
        # Followed, it is not read anew again
        with recorder._microphone() as frames:
            time.sleep(0.3)
            assert _heard(frames, LEVELS[AIRPODS])
        assert computer.initialized == 2
    finally:
        recorder.close_microphone()


# Telling the user ---------------------------------------------------------------


def _bare_assistant() -> VoiceAssistant:
    """An assistant of nothing but its audio and its events."""
    a = object.__new__(VoiceAssistant)
    a.event_bus = EventBus()
    a.events = []
    a.event_bus.subscribe(lambda kind, data: a.events.append((kind, dict(data))))
    a.pause_event = threading.Event()
    a.config_info = lambda: {}
    return a


def test_the_user_is_told_where_iris_speaks_and_listens_after_a_change(computer):
    a = _bare_assistant()
    a.player = AudioPlayer(audio=SoundDevice())
    a.recorder = AudioRecorder(audio=SoundDevice())
    a._in_use = a._devices_in_use()
    SoundDevice().watch(a._devices_changed)
    try:
        computer.unplug(AIRPODS, default_input=MICROPHONE, default_output=SPEAKERS)
        SoundDevice().play(np.zeros(240, dtype=np.float32), 24000)
        notices = [d["message"] for k, d in a.events if k == "notice"]
        assert notices == [f"Now speaking through {SPEAKERS} and listening through {MICROPHONE}."]
        assert ("config", {}) in a.events

        # Read anew with nothing changed: nothing to tell
        SoundDevice().refresh()
        assert len([k for k, _ in a.events if k == "notice"]) == 1
    finally:
        SoundDevice().unwatch(a._devices_changed)


# The reply ----------------------------------------------------------------------


class Speaker:
    """A speaker that cannot be opened until it is fixed."""

    def __init__(self):
        self.broken = True
        self.tries = 0
        self.played = 0

    def play(self, audio, sample_rate, **options) -> float:
        self.tries += 1
        if self.broken:
            raise sd.PortAudioError("Error opening OutputStream: Internal PortAudio error", -9986)
        self.played += 1
        return 0.0

    def stop(self):
        pass

    def playing(self) -> bool:
        return False


class Voice:
    """Speaks every piece as a short clip, at once."""

    def generate(self, text, language="en", **options):
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(path, np.full(2400, 0.1, dtype=np.float32), 24000)
        return path


def _spoken(pipeline: SpeechPipeline, *pieces, sound=None):
    """Speak the pieces and wait until the pipeline is done with them."""
    for piece in pieces:
        pipeline.queue_speech(piece, sound=sound)
    done = threading.Thread(target=pipeline.wait_for_speech, daemon=True)
    done.start()
    done.join(5)
    assert not done.is_alive()


def test_a_reply_the_speaker_cannot_play_is_told_once_and_not_spoken():
    speaker = Speaker()
    pipeline = SpeechPipeline(Voice(), AudioPlayer(audio=speaker))
    told = []
    pipeline.on_speaker_error = told.append

    pipeline.begin_reply()
    _spoken(pipeline, "One.", "Two.", "Three.")
    assert [str(e) for e in told] == [
        "Error opening OutputStream: Internal PortAudio error [PaErrorCode -9986]"
    ]
    # The rest of the reply is dropped, not tried clip by clip
    assert speaker.tries == 1 and not pipeline.is_active
    assert pipeline.said() == ""

    # The next reply tries again, and is not told again
    pipeline.begin_reply()
    _spoken(pipeline, "Four.")
    assert speaker.tries == 2 and len(told) == 1

    # Fixed, a preview plays
    speaker.broken = False
    pipeline.stop()
    _spoken(pipeline, "Hi, I'm Iris.", sound={"engine": "kokoro", "voice": "af_heart"})
    assert speaker.played == 1

    # Failing once more after it played, it is told again
    speaker.broken = True
    pipeline.begin_reply()
    _spoken(pipeline, "Five.")
    assert len(told) == 2


# The assistant's loop --------------------------------------------------------------


class Status:
    def __init__(self):
        self.status_lock = threading.Lock()
        self.status_listening = False
        self.status_dormant = False


class Recorder:
    """Listened to, it has the conversation paused; closed for that, it
    fails, as a microphone whose device is gone may; listened to again, it
    has Iris shut down."""

    silence_threshold = 0.015
    _device_error = False

    def __init__(self, assistant):
        self.assistant = assistant
        self.listens = 0
        self.closes_failed = 0

    def check_audio_devices(self):
        return True, ""

    def close_microphone(self):
        if self.assistant.suspend_event.is_set():
            self.assistant.suspend_event.clear()
            self.closes_failed += 1
            raise sd.PortAudioError("Error closing stream", -9986)

    def listen(self, on_level=None, should_stop=None, silence_duration=None):
        self.listens += 1
        if self.listens == 1:
            self.assistant.suspend_event.set()
        else:
            self.assistant.stop_event.set()
        return None


def test_the_loop_goes_on_when_the_audio_fails_in_it():
    """The falling chime cannot be played (the loop used to end there), and
    closing the microphone fails: Iris goes on listening."""
    a = _bare_assistant()
    a.status_manager = Status()
    a.stop_event, a.suspend_event = threading.Event(), threading.Event()
    a.hold_event, a.interrupt_event = threading.Event(), threading.Event()
    a.speak_replies = True
    a.wake_word, a.wake_window, a.awake_until = "Iris", 30.0, 0.0
    a._was_dormant = False  # awake until now: it falls asleep, with a chime
    a._last_window_emit = 0.0
    a.turn_queue = queue.Queue()
    a.transcriber = object()
    speaker = Speaker()
    a.player = SimpleNamespace(audio=speaker)
    a.speech_pipeline = SimpleNamespace(is_active=False, stop=lambda: None)
    a.tools = a.memory = None
    a.recorder = Recorder(a)

    loop = threading.Thread(target=a.run_loop, daemon=True)
    loop.start()
    loop.join(5)
    assert not loop.is_alive()
    assert speaker.tries == 1 and a.recorder.closes_failed == 1
    assert a.recorder.listens == 2
    assert [k for k, _ in a.events].count("sleep") == 1

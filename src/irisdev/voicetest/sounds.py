"""Sounds of a room, synthesized: none has to be recorded or downloaded.

- fan: brown noise with a 100 Hz hum and the tone of its blades;
- babble: six voices other than the user's, reading unrelated text from
  random places, as in a café;
- music: chords, a bass line and a kick drum, with a voice over them;
- tv: one voice reading the news;
- cough and throat_clear: band-passed noise bursts with a two-stage envelope;
- door: a click and a low thud;
- keyboard: typing, 6 to 12 keys a second;
- mug: a mug set down on a table, ringing at 2.4 and 5.1 kHz.

The voices are synthesized by a UserVoice and kept on disk; the rest is
computed and kept in memory. A sound is placed in a Space with
Space.add_source, at a level set by gain_for_snr.
"""

import functools
from typing import Callable, Optional

import numpy as np
from scipy.signal import butter, sosfilt

from iris.turns import FRAME, SPEECH_THRESHOLD, SpeechDetector
from irisdev.voicetest.acoustics import RATE

SEED = 11
# Voices for the chatter of a café, none of them the user's (am_michael)
BABBLE_VOICES = ("af_heart", "bf_isabella", "am_adam", "bm_george", "af_nova", "am_eric")
BABBLE_TEXT = (
    "We took the early train and still missed the ferry, so we had lunch by the harbour instead.",
    "My sister repainted the kitchen yellow, and honestly it looks better than I expected.",
    "The garden needs rain; the tomatoes have been sulking for two weeks now.",
    "He said the meeting moved to Thursday, but nobody told the people upstairs.",
    "I finally finished that novel about the orchestra, and the ending surprised me.",
    "If the bakery is still open, could you pick up a loaf on your way home?",
)
TV_VOICE = "bm_daniel"
TV_TEXT = (
    "Good evening. Our top story tonight: the city council has approved a new plan for the "
    "harbour district, with work set to begin next spring. Residents gathered outside the "
    "town hall as the vote was announced. In other news, the regional orchestra opens its "
    "season on Friday with a programme of works by local composers, and forecasters expect "
    "a mild weekend with sunshine in the east and showers along the coast."
)
MUSIC_VOICE = "af_sky"
MUSIC_TEXT = "Oh the river runs so slowly, and the evening light is golden, la la la."


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng(SEED + sum(map(ord, name)))


def _band(audio: np.ndarray, low: float, high: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=RATE, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _normal(audio: np.ndarray, peak: float = 0.5) -> np.ndarray:
    top = float(np.abs(audio).max()) if len(audio) else 0.0
    return (audio * (peak / top)).astype(np.float32) if top else audio.astype(np.float32)


def _fit(audio: np.ndarray, seconds: float) -> np.ndarray:
    """The audio repeated or cut to so many seconds."""
    frames = int(seconds * RATE)
    if len(audio) == 0:
        return np.zeros(frames, dtype=np.float32)
    return np.resize(audio, frames).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _fan(seconds: float) -> np.ndarray:
    rng = _rng("fan")
    frames = int(seconds * RATE)
    t = np.arange(frames) / RATE
    brown = np.cumsum(rng.standard_normal(frames))
    brown -= np.convolve(brown, np.ones(RATE // 10) / (RATE // 10), mode="same")  # no drift
    brown = _normal(brown, 0.3)
    hum = 0.08 * np.sin(2 * np.pi * 100 * t)
    # Five blades at 24 turns a second
    blades = 0.05 * np.sin(2 * np.pi * 120 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 24 * t))
    return _normal(brown + hum + blades, 0.4)


def fan(seconds: float = 20.0) -> np.ndarray:
    return _fan(float(seconds)).copy()


@functools.lru_cache(maxsize=None)
def _cough() -> np.ndarray:
    rng = _rng("cough")
    frames = int(0.45 * RATE)
    t = np.arange(frames) / RATE
    noise = _band(rng.standard_normal(frames), 300, 3500)
    # A sharp burst, then a softer decaying tail
    burst = np.exp(-((t - 0.03) ** 2) / (2 * 0.015**2))
    tail = 0.45 * np.exp(-np.maximum(0.0, t - 0.08) / 0.12) * (t > 0.06)
    return _normal(noise * (burst + tail), 0.6)


def cough() -> np.ndarray:
    return _cough().copy()


@functools.lru_cache(maxsize=None)
def _throat_clear() -> np.ndarray:
    rng = _rng("throat")
    frames = int(0.6 * RATE)
    t = np.arange(frames) / RATE
    noise = _band(rng.standard_normal(frames), 150, 1800)
    rasp = 0.6 + 0.4 * np.sin(2 * np.pi * 35 * t)
    envelope = np.where(t < 0.25, np.sin(np.pi * t / 0.25) * 0.6, 0.0)
    envelope += np.where(t >= 0.3, np.exp(-(t - 0.3) / 0.08) * (t < 0.55), 0.0)
    return _normal(noise * rasp * envelope, 0.45)


def throat_clear() -> np.ndarray:
    return _throat_clear().copy()


@functools.lru_cache(maxsize=None)
def _door() -> np.ndarray:
    rng = _rng("door")
    frames = int(0.4 * RATE)
    t = np.arange(frames) / RATE
    click = _band(rng.standard_normal(frames), 1500, 6000) * np.exp(-t / 0.005) * (t < 0.02)
    sweep = 150 - 70 * np.minimum(1.0, t / 0.25)  # 150 Hz falling to 80 Hz
    thud = np.sin(2 * np.pi * np.cumsum(sweep) / RATE) * np.exp(-np.maximum(0.0, t - 0.02) / 0.07)
    thud *= (t >= 0.02) & (t < 0.27)
    return _normal(0.6 * click + thud, 0.7)


def door() -> np.ndarray:
    return _door().copy()


@functools.lru_cache(maxsize=None)
def _keyboard(seconds: float) -> np.ndarray:
    rng = _rng("keyboard")
    frames = int(seconds * RATE)
    out = np.zeros(frames, dtype=np.float32)
    click_frames = int(0.012 * RATE)
    tc = np.arange(click_frames) / RATE
    at = 0.05
    while at < seconds - 0.02:
        start = int(at * RATE)
        click = _band(rng.standard_normal(click_frames), 2000, 7000) * np.exp(-tc / 0.003)
        out[start : start + click_frames] += click[: frames - start] * rng.uniform(0.5, 1.0)
        at += 1.0 / rng.uniform(6, 12)
    return _normal(out, 0.35)


def keyboard(seconds: float = 2.0) -> np.ndarray:
    return _keyboard(float(seconds)).copy()


@functools.lru_cache(maxsize=None)
def _mug() -> np.ndarray:
    frames = int(0.3 * RATE)
    t = np.arange(frames) / RATE
    ring = np.sin(2 * np.pi * 2400 * t) * np.exp(-t / 0.06) + 0.6 * np.sin(2 * np.pi * 5100 * t) * np.exp(-t / 0.035)
    knock = _band(_rng("mug").standard_normal(frames), 400, 2000) * np.exp(-t / 0.004)
    return _normal(ring + 0.8 * knock, 0.5)


def mug() -> np.ndarray:
    return _mug().copy()


def babble(voices: Callable[[str], object], seconds: float = 20.0) -> np.ndarray:
    """Six voices talking among themselves, each from a random place.
    voices(name) gives a UserVoice speaking with that Kokoro voice."""
    rng = _rng("babble")
    out = np.zeros(int(seconds * RATE), dtype=np.float32)
    for name, text in zip(BABBLE_VOICES, BABBLE_TEXT):
        speech = voices(name).utterance([text, 0.4, text])
        speech = np.roll(_fit(speech, seconds), int(rng.integers(0, len(out))))
        out += speech * rng.uniform(0.6, 1.0)
    return _normal(out, 0.5)


def tv(voices: Callable[[str], object], seconds: float = 30.0) -> np.ndarray:
    """A news reader, as from a television across the room."""
    speech = voices(TV_VOICE).utterance([TV_TEXT])
    return _normal(_band(_fit(speech, seconds), 150, 6000), 0.5)


def music(voices: Optional[Callable[[str], object]] = None, seconds: float = 20.0) -> np.ndarray:
    """Chords, bass and a kick drum at 120 beats a minute, with a voice
    over them when voices are given."""
    frames = int(seconds * RATE)
    t = np.arange(frames) / RATE
    roots = (220.0, 174.61, 261.63, 196.0)  # A, F, C, G
    chord_of = (t // 2.0).astype(int) % len(roots)
    root = np.take(roots, chord_of)
    phase = 2 * np.pi * np.cumsum(root) / RATE
    chords = sum(np.sin(phase * ratio) / (n + 1) for n, ratio in enumerate((1.0, 1.26, 1.5, 2.0)))
    bass = 0.6 * np.sin(phase / 2)
    beat = t % 0.5
    kick = np.sin(2 * np.pi * 60 * beat) * np.exp(-beat / 0.08)
    out = 0.3 * chords + 0.3 * bass + 0.5 * kick
    if voices is not None:
        vocal = voices(MUSIC_VOICE).utterance([MUSIC_TEXT, 1.0, MUSIC_TEXT])
        out = out + 0.6 * _normal(_fit(vocal, seconds), 0.8)
    return _normal(out, 0.5)


# The sounds needing no voices, and those that do, by name
SPEECHLESS = {
    "fan": fan,
    "cough": lambda seconds=None: cough(),
    "throat_clear": lambda seconds=None: throat_clear(),
    "door": lambda seconds=None: door(),
    "keyboard": keyboard,
    "mug": lambda seconds=None: mug(),
}
SPOKEN = {"babble": babble, "tv": tv, "music": music}
NAMES = tuple(SPEECHLESS) + tuple(SPOKEN)


def sound(name: str, seconds: Optional[float] = None, voices: Optional[Callable[[str], object]] = None) -> np.ndarray:
    """A sound by name, so many seconds long where its length can be chosen."""
    if name in SPEECHLESS:
        make = SPEECHLESS[name]
        return make(seconds) if seconds is not None else make()
    if name in SPOKEN:
        if voices is None and name != "music":
            raise ValueError(f"The sound {name!r} needs voices to speak with")
        return SPOKEN[name](voices, seconds or 20.0)
    raise ValueError(f"No sound called {name!r}; there are {', '.join(NAMES)}")


def active_rms(audio: np.ndarray, floor: float = 0.02) -> float:
    """The RMS of the audio's loud parts (speech, not its pauses): 20 ms
    windows whose RMS is within 30 dB of the loudest, and above a floor."""
    step = RATE // 50
    windows = len(audio) // step
    if windows == 0:
        return 0.0
    rms = np.sqrt(np.mean(audio[: windows * step].reshape(windows, step) ** 2, axis=1))
    active = rms[(rms > rms.max() * 10 ** (-30 / 20)) & (rms > floor * rms.max())]
    return float(np.sqrt(np.mean(active**2))) if len(active) else 0.0


def gain_for_snr(noise: np.ndarray, speech: np.ndarray, snr_db: float) -> float:
    """The gain that puts the noise snr_db below the speech, measured on the
    speech while it is active, as it arrives at the microphone."""
    noise_rms = float(np.sqrt(np.mean(noise**2))) if len(noise) else 0.0
    speech_rms = active_rms(speech)
    if not noise_rms or not speech_rms:
        return 0.0
    return speech_rms / (noise_rms * 10 ** (snr_db / 20))


def vad_profile(audio: np.ndarray) -> dict:
    """How much Silero, Iris's voice detector, takes the audio for speech:
    its highest probability, and the longest run at or above 0.5."""
    detector = SpeechDetector()
    longest = run = 0
    highest = 0.0
    for start in range(0, len(audio) - FRAME + 1, FRAME):
        probability = detector(audio[start : start + FRAME].astype(np.float32))
        highest = max(highest, probability)
        run = run + 1 if probability >= SPEECH_THRESHOLD else 0
        longest = max(longest, run)
    return {"max": round(highest, 3), "longest": round(longest * FRAME / RATE, 3)}

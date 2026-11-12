"""Stop must silence a reply completely, including audio still being made."""

import tempfile
import threading
import time

from iris.tts import SpeechPipeline


class SlowGenerator:
    """Generates a file per text, each taking a while, like Kokoro."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, text, language="en"):
        self.started.set()
        self.release.wait(5)
        return tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name


class RecordingPlayer:
    def __init__(self):
        self.played = []

    def play(self, audio_file, on_level=None, cancelled=None):
        if cancelled is None or not cancelled():
            self.played.append(audio_file)

    def stop(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def progress(self):
        return 0.0


def _wait(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_audio_generated_after_stop_is_never_played():
    generator, player = SlowGenerator(), RecordingPlayer()
    pipeline = SpeechPipeline(generator, player)
    pipeline.queue_speech("The lighthouse keeper rose with the dawn.")
    pipeline.queue_speech("A storm was coming.")
    assert generator.started.wait(2)

    pipeline.stop()
    # Still generating, but for a stopped reply: not speaking any more
    assert not pipeline.is_active

    generator.release.set()
    time.sleep(0.3)
    assert player.played == []
    pipeline.wait_for_speech()  # returns: nothing left behind


def test_speech_queued_after_a_stop_plays_normally():
    generator, player = SlowGenerator(), RecordingPlayer()
    generator.release.set()
    pipeline = SpeechPipeline(generator, player)
    pipeline.stop()
    pipeline.queue_speech("A new reply.")
    assert _wait(lambda: len(player.played) == 1)


def test_a_stop_racing_the_start_of_a_clip_silences_it():
    """The player is told the clip was cancelled even if stop() lands after
    the pipeline picked the clip up."""
    generator = SlowGenerator()
    generator.release.set()
    seen = []

    class Player(RecordingPlayer):
        def play(self, audio_file, on_level=None, cancelled=None):
            pipeline.stop()  # stop arrives just as the clip starts
            seen.append(cancelled())

    pipeline = SpeechPipeline(generator, Player())
    pipeline.queue_speech("Hello.")
    assert _wait(lambda: seen == [True])


"""Unit tests for audio configuration."""

from iris.audio import AudioRecorder


class TestAudioRecorderConfig:
    def test_default_values(self):
        recorder = AudioRecorder()
        assert recorder.sample_rate == 16000
        assert recorder.silence_threshold == 0.02
        assert recorder.silence_duration == 2.5

    def test_custom_values(self):
        recorder = AudioRecorder(
            sample_rate=44100,
            silence_threshold=0.05,
            silence_duration=3.0,
        )
        assert recorder.sample_rate == 44100
        assert recorder.silence_threshold == 0.05
        assert recorder.silence_duration == 3.0

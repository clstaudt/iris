"""Voice cloning via Kanade voice conversion on top of Kokoro TTS."""

import tempfile
from pathlib import Path

import soundfile as sf


class VoiceCloner:
    """
    Applies Kanade voice conversion to Kokoro TTS output.

    Loads the Kanade model eagerly on initialization so that
    the download and setup happen at assistant startup time.
    """

    def __init__(self):
        import torch
        from kanade_tokenizer import KanadeModel, load_vocoder

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._kanade = KanadeModel.from_pretrained("frothywater/kanade-12.5hz").to(self._device).eval()
        self._vocoder = load_vocoder(self._kanade.config.vocoder_name).to(self._device)
        self._sample_rate = self._kanade.config.sample_rate

    def convert(self, source_audio_path: str | Path, reference_audio_path: str | Path) -> tuple:
        """
        Apply voice conversion to source audio using a reference voice.

        Args:
            source_audio_path: Path to the source audio (Kokoro TTS output)
            reference_audio_path: Path to the reference voice audio

        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        import torch
        from kanade_tokenizer import load_audio, vocode

        source_wav = load_audio(str(source_audio_path), sample_rate=self._sample_rate).to(self._device)
        ref_wav = load_audio(str(reference_audio_path), sample_rate=self._sample_rate).to(self._device)

        with torch.inference_mode():
            converted_mel = self._kanade.voice_conversion(
                source_waveform=source_wav, reference_waveform=ref_wav
            )
            converted_wav = vocode(self._vocoder, converted_mel.unsqueeze(0))

        return converted_wav.squeeze().cpu().numpy(), self._sample_rate

    def clone_to_file(
        self, source_audio_path: str | Path, reference_audio_path: str | Path, output_path: str | Path
    ) -> str:
        """
        Apply voice conversion and save to a file.

        Args:
            source_audio_path: Path to the source audio
            reference_audio_path: Path to the reference voice audio
            output_path: Path to save the converted audio

        Returns:
            Path to the output file
        """
        audio_data, sample_rate = self.convert(source_audio_path, reference_audio_path)
        sf.write(str(output_path), audio_data, sample_rate)
        return str(output_path)

    def convert_to_tempfile(
        self, source_audio_path: str | Path, reference_audio_path: str | Path
    ) -> str:
        """
        Apply voice conversion and save to a temporary file.

        Args:
            source_audio_path: Path to the source audio
            reference_audio_path: Path to the reference voice audio

        Returns:
            Path to the temporary output file
        """
        audio_data, sample_rate = self.convert(source_audio_path, reference_audio_path)
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp.close()
        sf.write(temp.name, audio_data, sample_rate)
        return temp.name

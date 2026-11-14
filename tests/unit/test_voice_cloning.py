"""Voices cloned from a sample, with Kyutai's gated Pocket TTS model: the
access to it, the sample, the voice made of it in every language, and the
persona editor's part. Hugging Face, the models and the microphone are
stand-ins: nothing is downloaded, logged in to, or recorded."""

import base64
import contextlib
import io
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import httpx
import numpy as np
import pytest
import soundfile as sf
import torch
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from iris import audio as iris_audio
from iris import cloning, personas, rpc_server
from iris.cloning import (
    LOGIN,
    READY,
    TERMS,
    CloneAccessError,
    Cloning,
    SampleError,
    VoiceNotReadyError,
    access,
    cloned_voices,
    cloning_weights,
    connect,
    forget_voice,
    made_for,
    make_language,
    make_voice,
    prepare_sample,
    read_audio,
    voice_folder,
    voice_state,
)
from iris.controller import AssistantController
from iris.events import EventBus
from iris.tts import PocketGenerator, SpeechGenerator

TOKEN = "hf_stand_in_secret_0123456789"


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api"))


class StandInModel:
    """A Pocket TTS model: in the form that clones, or the open one."""

    sample_rate = 24000

    def __init__(self, language: str, clones: bool):
        self.language = language
        self.has_voice_cloning = clones
        self.prompts: list = []

    def get_state_for_audio_prompt(self, prompt):
        self.prompts.append(prompt)
        if str(prompt).endswith(".safetensors"):
            return {"read": {"from": torch.zeros(1)}}
        if not isinstance(prompt, Path):
            return {"catalogue": {"voice": torch.zeros(1)}}
        if not self.has_voice_cloning:
            raise ValueError("this model cannot clone")
        return {"made": {"cache": torch.ones(3)}}

    def generate_audio(self, state, text):
        return torch.zeros(240)


def _gated() -> GatedRepoError:
    message = "Access to model kyutai/pocket-tts is restricted"
    return GatedRepoError(message, response=_response(403))


class StandInKyutai:
    """Hugging Face and Kyutai's gated repo, as cloning sees them. The open
    repo's files (the models without cloning, the tokenizers, the voices)
    are downloaded, as Iris's first run leaves them."""

    def __init__(self, token=TOKEN, valid=True, terms=True, cached=(), online=True):
        self._token = token
        self.valid = valid
        self.terms = terms
        self.online = online
        self.files = {cloning_weights(language) for language in cached}
        self.loaded: list[StandInModel] = []
        self.fetched: list[str] = []

    def _online(self):
        if not self.online:
            raise httpx.ConnectError("no route to huggingface.co")

    def token(self):
        return self._token

    def whoami(self):
        self._online()
        if not self.valid:
            raise HfHubHTTPError("Invalid user token.", response=_response(401))
        return {"name": "someone"}

    def check_access(self):
        self._online()
        if not self.terms:
            raise _gated()

    def login(self, token):
        self._online()
        if not token.startswith("hf_"):
            raise HfHubHTTPError("Invalid user token.", response=_response(401))
        self._token, self.valid = token, True

    def cached(self, file):
        return file in self.files or file.repo != cloning.CLONING_REPO

    def fetch(self, files, on_progress):
        self._online()
        gated = [file for file in files if file.repo == cloning.CLONING_REPO]
        if gated and (not self.terms or not self._token):
            raise _gated()
        on_progress(100, 200)
        on_progress(200, 200)
        self.files.update(files)
        self.fetched += [file.filename.split("/")[1] for file in gated]

    def load_model(self, pocket_lang):
        model = StandInModel(pocket_lang, clones=cloning_weights(pocket_lang) in self.files)
        self.loaded.append(model)
        return model


@pytest.fixture(autouse=True)
def nothing_forgotten(monkeypatch):
    """Each test names its voices afresh."""
    monkeypatch.setattr(cloning, "_forgotten", set())


@pytest.fixture
def kyutai(monkeypatch):
    stand_in = StandInKyutai()
    monkeypatch.setattr(cloning, "KYUTAI", stand_in)
    return stand_in


def energy(frame: np.ndarray) -> float:
    """Speech, to the tests: anything louder than a whisper."""
    return 1.0 if np.sqrt(np.mean(frame**2)) > 0.01 else 0.0


@pytest.fixture
def heard_by_energy(monkeypatch):
    """The sample's speech is told by its energy, not by Silero."""
    monkeypatch.setattr(cloning, "SpeechDetector", lambda: energy)


def speech(seconds: float, rate: int = 48000, loudness: float = 0.1) -> np.ndarray:
    """Something with a voice's rhythm: a tone in syllables."""
    t = np.arange(int(seconds * rate)) / rate
    syllables = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 2.5 * t))
    return (loudness * np.sqrt(2) * syllables * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def silence(seconds: float, rate: int = 48000) -> np.ndarray:
    return np.random.default_rng(1).normal(0, 1e-4, int(seconds * rate)).astype(np.float32)


def wav(audio: np.ndarray, rate: int = 48000, channels: int = 1) -> bytes:
    out = io.BytesIO()
    data = np.stack([audio] * channels, axis=1) if channels > 1 else audio
    sf.write(out, data, rate, format="WAV")
    return out.getvalue()


def a_voice(name="Ada", seconds=8.0) -> Path:
    return make_voice(speech(seconds, rate=cloning.RATE), name)


# ---------------------------------------------------------------------------
# Access: no login, terms not accepted, or ready
# ---------------------------------------------------------------------------


def test_without_a_login_cloning_asks_for_one(kyutai):
    kyutai._token = None
    assert access()["state"] == LOGIN


def test_a_refused_token_asks_for_another(kyutai):
    kyutai.valid = False
    result = access()
    assert result["state"] == LOGIN
    assert "did not accept" in result["error"]


def test_a_login_without_kyutais_terms_asks_for_them(kyutai):
    kyutai.terms = False
    assert access() == {"state": TERMS, "cached": False, "error": ""}


def test_with_the_terms_accepted_cloning_is_ready(kyutai):
    assert access() == {"state": READY, "cached": False, "error": ""}


def test_offline_a_downloaded_model_still_clones(kyutai):
    kyutai.online = False
    kyutai.files.add(cloning_weights("english"))
    assert access()["state"] == READY


def test_offline_without_the_model_says_so(kyutai):
    kyutai.online = False
    result = access()
    assert result["state"] == TERMS
    assert result["error"] == "No connection to Hugging Face"


def test_connecting_logs_in_and_tells_access(kyutai):
    kyutai._token = None
    assert connect(TOKEN)["state"] == READY
    assert kyutai.token() == TOKEN


def test_a_refused_token_is_neither_returned_nor_logged(kyutai, caplog):
    kyutai._token = None
    caplog.set_level(logging.DEBUG)
    result = connect("not-a-token-but-secret")
    assert result["state"] == LOGIN
    assert result["error"] == "Hugging Face did not accept the token"
    assert "secret" not in repr(result)
    assert "secret" not in caplog.text


def test_an_empty_token_is_asked_for(kyutai):
    kyutai._token = None
    assert connect("  ")["error"] == "Paste an access token"


def test_the_cloning_model_is_the_gated_one_of_its_language():
    file = cloning_weights("german")
    assert file.repo == "kyutai/pocket-tts"
    assert file.filename == "languages/german/model.safetensors"
    assert file.revision


# ---------------------------------------------------------------------------
# The sample: trimmed, levelled, long enough
# ---------------------------------------------------------------------------


def test_a_sample_is_trimmed_its_pauses_shortened_and_levelled(heard_by_energy):
    audio = np.concatenate([silence(1.5), speech(4), silence(2.5), speech(4), silence(1.5)])
    sample = prepare_sample(audio, 48000)

    seconds = len(sample) / cloning.RATE
    # 8 s of speech, a little silence around it and in place of the pause
    assert 8.3 < seconds < 9.6
    level = 20 * np.log10(np.sqrt(np.mean(sample**2)))
    assert -22 < level < -18.5
    assert np.max(np.abs(sample)) <= 10 ** (cloning.PEAK_DB / 20) + 1e-6
    # It starts and ends where the speech does, not a second away
    lead = np.argmax(np.abs(sample) > 0.02) / cloning.RATE
    assert lead < cloning.EDGE + 0.1


def test_too_little_speech_is_refused(heard_by_energy):
    audio = np.concatenate([silence(2), speech(3), silence(4)])
    with pytest.raises(SampleError, match="Too short: 3 s of speech"):
        prepare_sample(audio, 48000)


def test_a_silent_recording_is_refused(heard_by_energy):
    with pytest.raises(SampleError, match="Too short: 0 s"):
        prepare_sample(silence(10), 48000)
    with pytest.raises(SampleError, match="Nothing was recorded"):
        prepare_sample(np.zeros(0, dtype=np.float32), 48000)


def test_a_long_sample_is_cut_to_thirty_seconds(heard_by_energy):
    sample = prepare_sample(speech(45, rate=16000), 16000)
    assert len(sample) == pytest.approx(cloning.MAX_SAMPLE * cloning.RATE, abs=10)


def test_a_quiet_voice_is_raised(heard_by_energy):
    sample = prepare_sample(speech(8, loudness=0.012), 48000)
    assert 20 * np.log10(np.sqrt(np.mean(sample**2))) > -22


def test_the_speech_is_told_by_silero_by_default():
    # A tone is no voice to Silero, however long
    with pytest.raises(SampleError, match="Too short"):
        prepare_sample(np.sin(np.arange(48000 * 8) * 0.05).astype(np.float32) * 0.2, 48000)


def test_wav_and_m4a_files_are_read():
    audio, rate = read_audio(wav(speech(1), channels=2))
    assert rate == 48000 and audio.ndim == 1 and len(audio) == 48000

    out = io.BytesIO()
    with av.open(out, "w", format="ipod") as container:
        stream = container.add_stream("aac", rate=44100, layout="mono")
        sound = speech(1, rate=44100)[np.newaxis, :]
        frame = av.AudioFrame.from_ndarray(sound, format="fltp", layout="mono")
        frame.rate = 44100
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            container.mux(packet)
    audio, rate = read_audio(out.getvalue())
    assert rate == 44100 and len(audio) >= 44100


def test_a_file_that_is_no_audio_is_refused():
    with pytest.raises(SampleError, match="cannot read"):
        read_audio(b"%PDF-1.4 not a sound")


# ---------------------------------------------------------------------------
# The voices: where they live and how personas name them
# ---------------------------------------------------------------------------


def test_a_voice_is_kept_in_a_folder_named_for_the_persona(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_VOICES_DIR", str(tmp_path))
    folder = a_voice("Ada Lovelace")
    assert folder == tmp_path / "ada-lovelace"
    audio, rate = sf.read(folder / cloning.SAMPLE_FILE)
    assert rate == cloning.RATE and len(audio) == 8 * cloning.RATE
    assert cloned_voices() == [
        {"id": "clone:ada-lovelace", "name": "Ada Lovelace", "language": "Cloned"}
    ]


def test_another_voice_of_the_same_name_gets_a_number():
    a_voice("Ada")
    second = a_voice("Ada")
    assert second.name == "ada-2"
    assert [v["name"] for v in cloned_voices()] == ["Ada", "Ada 2"]


def test_a_deleted_voice_is_gone_and_its_folder_not_taken_again():
    folder = a_voice("Ada")
    assert forget_voice("clone:ada")
    assert not folder.exists() and cloned_voices() == []
    # A voice loaded from that folder may still be in memory
    assert a_voice("Ada").name == "ada-2"


def test_a_new_voice_takes_no_folder_a_persona_names():
    """A persona that names a voice no longer there would speak with the new
    one unasked."""
    named = {"clone:ada"}.__contains__
    assert make_voice(speech(8, rate=cloning.RATE), "Ada", named).name == "ada-2"


def test_only_folders_of_the_voices_folder_are_cloned_voices():
    assert voice_folder("clone:ada") == cloning.voices_dir() / "ada"
    assert voice_folder(str(cloning.voices_dir() / "ada")) == cloning.voices_dir() / "ada"
    for voice in ("clone:../ada", "clone:", "clone:/etc", "anna", "/tmp/ada", "af_bella"):
        assert voice_folder(voice) is None


# ---------------------------------------------------------------------------
# Every language: made by its own cloning model the first time, then kept
# ---------------------------------------------------------------------------


def test_a_voice_is_made_for_a_language_by_its_cloning_model_and_kept(kyutai):
    folder = a_voice()
    progress, making = [], []

    make_language(folder, "french", lambda *got: progress.append(got), lambda: making.append(1))

    # Its cloning model downloaded first, with progress, then the voice made
    assert kyutai.fetched == ["french"] and progress[-1] == (200, 200) and making
    [french] = [model for model in kyutai.loaded if model.language == "french"]
    assert french.has_voice_cloning and french.prompts == [folder / cloning.SAMPLE_FILE]
    assert (folder / "french.safetensors").is_file()
    assert made_for(folder) == ["french"]


def test_a_voice_not_made_for_a_language_is_not_made_when_it_is_to_speak_it(kyutai):
    folder = a_voice()
    with pytest.raises(VoiceNotReadyError, match="Ada is not made for French yet"):
        voice_state({"french": StandInModel("french", clones=False)}, "french", folder)
    assert not kyutai.fetched and not kyutai.loaded
    assert not (folder / "french.safetensors").exists()


def test_a_kept_voice_is_read_by_whichever_model_is_loaded(kyutai):
    folder = a_voice()
    (folder / "english.safetensors").write_bytes(b"made before")
    models = {"english": StandInModel("english", clones=False)}

    voice_state(models, "english", folder)

    assert models["english"].prompts == [str(folder / "english.safetensors")]
    assert not kyutai.fetched and not kyutai.loaded


def test_without_access_a_new_language_says_why(kyutai):
    kyutai.terms = False
    folder = a_voice()
    with pytest.raises(CloneAccessError, match="Making Ada for German needs Kyutai's cloning"):
        make_language(folder, "german")
    assert not (folder / "german.safetensors").exists()


def test_a_voice_whose_folder_is_gone_says_so(kyutai):
    with pytest.raises(FileNotFoundError, match="is gone"):
        make_language(cloning.voices_dir() / "nobody", "english")


@pytest.fixture
def pocket(kyutai):
    """Pocket TTS's models, open ones unless their cloning form is cached."""
    with patch("iris.tts.TTSModel") as mock:
        mock.load_model.side_effect = lambda language: kyutai.load_model(language)
        yield mock


def test_a_persona_with_a_cloned_voice_speaks_english_with_it(pocket):
    folder = a_voice()
    (folder / "english.safetensors").write_bytes(b"made when cloned")
    gen = PocketGenerator(voice="clone:ada")

    path = gen.generate("Hello there.", language="en")

    assert path is not None
    Path(path).unlink()
    assert gen.voice == str(folder)
    assert pocket.load_model.call_args.kwargs == {"language": "english"}


def test_a_cloned_voice_stays_itself_in_another_language(pocket, kyutai):
    folder = a_voice()
    (folder / "english.safetensors").write_bytes(b"made when cloned")
    make_language(folder, "french")
    gen = PocketGenerator(voice="clone:ada")

    Path(gen.generate("Bonjour.", language="fr")).unlink()

    # Not the French model's own voice: the clone, made for French
    french = [m for m in kyutai.loaded if m.language == "french"]
    read = [p for m in french for p in m.prompts if str(p).endswith(".safetensors")]
    assert read == [str(folder / "french.safetensors")]
    # The next reply reads it from memory
    Path(gen.generate("Merci.", language="fr")).unlink()
    assert [p for m in french for p in m.prompts if str(p).endswith(".safetensors")] == read


def test_a_language_not_made_is_not_spoken_and_no_other_voice_speaks_it(pocket, kyutai):
    folder = a_voice()
    (folder / "english.safetensors").write_bytes(b"made when cloned")
    told = []
    gen = SpeechGenerator("kyutai", "clone:ada", prepare=False)
    gen.on_failure = told.append

    assert gen.generate("Guten Tag.", language="de") is None

    # Nothing downloaded or made for it, and no other voice spoke in its place
    assert not kyutai.fetched and not (folder / "german.safetensors").exists()
    assert not [p for m in kyutai.loaded if m.language == "german" for p in m.prompts]
    # English goes on in the voice
    english = gen.generate("Hello.", language="en")
    assert english is not None
    Path(english).unlink()


def test_cloned_voices_are_listed_with_the_kyutai_voices():
    a_voice("Ada")
    listed = PocketGenerator.voices()
    assert listed[-1] == {"id": "clone:ada", "name": "Ada", "language": "Cloned"}
    assert any(v["id"] == "anna" for v in listed)


# ---------------------------------------------------------------------------
# The persona editor's part: record or choose, make, drop
# ---------------------------------------------------------------------------


class Microphone:
    """The computer's audio, with a microphone the test speaks into."""

    def __init__(self, rate=48000):
        self.rate = rate
        self.opened: list[dict] = []
        self._callback = None

    def query_devices(self, *args, kind=None, **kwargs):
        info = {"name": "Test microphone", "default_samplerate": self.rate, "max_input_channels": 1}
        return info if kind == "input" else [info] * 3

    def input_stream(self, callback=None, **options):
        self.opened.append(options)
        self._callback = callback
        return contextlib.nullcontext()

    def say(self, audio: np.ndarray, block: int = 4800):
        """Audio as the microphone delivers it, faster than it is said."""
        for start in range(0, len(audio), block):
            time.sleep(0.002)
            now = time.monotonic()
            info = SimpleNamespace(currentTime=now, inputBufferAdcTime=now - 0.01)
            piece = audio[start : start + block]
            self._callback(piece.reshape(-1, 1), len(piece), info, None)


class Assistant:
    def __init__(self):
        self.hold_event = threading.Event()
        self.recorder = SimpleNamespace(device=2)


@pytest.fixture
def microphone(monkeypatch):
    device = Microphone()
    monkeypatch.setattr(iris_audio, "SYSTEM", device)
    return device


def until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_a_recording_holds_iris_from_listening_and_makes_a_voice(
    microphone, kyutai, heard_by_energy
):
    kyutai.files.add(cloning_weights("english"))
    events, interrupted = [], []
    assistant = Assistant()
    service = Cloning(
        lambda t, d: events.append((t, d)), lambda: assistant, lambda: interrupted.append(1)
    )

    assert service.record(True) == {"ok": True}
    # Iris stops speaking and does not listen; its own microphone is used
    assert assistant.hold_event.is_set() and interrupted
    assert microphone.opened[0]["device"] == 2 and microphone.opened[0]["samplerate"] == 48000
    microphone.say(np.concatenate([silence(0.5), speech(9), silence(0.5)]))
    assert until(lambda: any(t == "clone.level" and d["seconds"] > 5 for t, d in events))
    assert max(d["level"] for t, d in events if t == "clone.level") > 0.3

    result = service.record(False)
    assert result["ok"] and 9 < result["seconds"] < 10
    assert not assistant.hold_event.is_set()

    made = service.clone("Ada")
    assert made == {"ok": True, "voice": {"id": "clone:ada", "name": "Ada", "language": "Cloned"}}
    folder = voice_folder("clone:ada")
    assert (folder / cloning.SAMPLE_FILE).is_file() and (folder / "english.safetensors").is_file()


def test_a_hold_the_user_set_typing_stays(microphone, kyutai, heard_by_energy):
    assistant = Assistant()
    assistant.hold_event.set()
    service = Cloning(lambda t, d: None, lambda: assistant)
    service.record(True)
    service.record(False, keep=False)
    assert assistant.hold_event.is_set()


def test_a_redo_drops_what_was_recorded(microphone, kyutai, heard_by_energy):
    service = Cloning(lambda t, d: None)
    service.record(True)
    microphone.say(speech(9))
    assert service.record(False, keep=False) == {"ok": True}
    assert service.clone("Ada")["ok"] is False


def test_a_microphone_that_cannot_open_is_told_and_iris_listens_again(monkeypatch):
    class Broken(Microphone):
        def input_stream(self, **options):
            raise OSError("no input device")

    monkeypatch.setattr(iris_audio, "SYSTEM", Broken())
    assistant = Assistant()
    result = Cloning(lambda t, d: None, lambda: assistant).record(True)
    assert result["ok"] is False and "could not be opened" in result["error"]
    assert not assistant.hold_event.is_set()


def test_a_chosen_file_makes_a_voice(kyutai, heard_by_energy):
    kyutai.files.add(cloning_weights("english"))
    service = Cloning(lambda t, d: None)
    data = base64.b64encode(wav(np.concatenate([speech(6), silence(1), speech(4)]))).decode()

    assert service.use_file(data, "grandpa.wav")["ok"]
    assert service.clone("Klaus")["voice"]["id"] == "clone:klaus"


def test_a_short_file_or_no_audio_is_refused(kyutai, heard_by_energy):
    service = Cloning(lambda t, d: None)
    short = base64.b64encode(wav(speech(2))).decode()
    assert "Too short" in service.use_file(short)["error"]
    assert "cannot read" in service.use_file(base64.b64encode(b"hello").decode())["error"]
    assert "cannot read" in service.use_file("not base64!")["error"]


def test_without_access_no_voice_is_made(kyutai, heard_by_energy):
    kyutai.terms = False
    service = Cloning(lambda t, d: None)
    service.use_file(base64.b64encode(wav(speech(8))).decode())
    result = service.clone("Ada")
    assert result["ok"] is False and result["access"] is True
    assert cloned_voices() == []


def test_the_cloning_model_downloads_with_progress(kyutai):
    events = []
    assert Cloning(lambda t, d: events.append((t, d))).fetch() == {"ok": True}
    assert ("clone.progress", {"done": 200, "total": 200}) in events
    assert kyutai.fetched == ["english"]


def test_saving_a_persona_with_a_voice_cloned_again_keeps_the_one_before(kyutai, caplog):
    """Iris removed the voice cloned before, and so lost voices the user
    liked. Every voice stays until it is deleted on the Voices page."""
    a_voice("Ada")
    a_voice("Ada")
    controller = AssistantController(EventBus())
    ada = {"name": "Ada", "engine": "kyutai", "description": "", "speed": 1.0}
    first = controller.persona_save({**ada, "voice": "clone:ada"})
    assert personas.find_persona(first["id"]).voice == "clone:ada"

    with caplog.at_level(logging.INFO, logger="iris.cloning"):
        controller.persona_save({**ada, "id": first["id"], "voice": "clone:ada-2"})
        controller.persona_save({**ada, "id": first["id"], "voice": "anna"})

    assert [v["id"] for v in cloned_voices()] == ["clone:ada", "clone:ada-2"]
    assert personas.load_all_personas()[first["id"]].voice == "anna"
    assert not [r for r in caplog.records if "removing" in r.getMessage()]


def test_a_voice_cloned_and_not_saved_with_a_persona_stays(kyutai, heard_by_energy):
    kyutai.files.add(cloning_weights("english"))
    service = Cloning(lambda t, d: None)
    for _ in range(2):
        service.use_file(base64.b64encode(wav(speech(8))).decode())
        assert service.clone("Ada")["ok"]
    assert [v["id"] for v in cloned_voices()] == ["clone:ada", "clone:ada-2"]


def test_a_cloned_voice_that_is_not_here_is_not_given_to_a_persona():
    """The editor offered a voice removed meanwhile, and the persona saved
    with it named a voice that was not there."""
    controller = AssistantController(EventBus())
    noam = {"name": "Noam", "engine": "kyutai", "description": "", "speed": 1.0}

    refused = controller.persona_save({**noam, "voice": "clone:noam-2"})

    assert refused["ok"] is False
    assert refused["error"] == "The voice Noam 2 cannot be found: choose another voice"
    assert "noam" not in personas.load_all_personas()
    # Nor in place of the voice a persona speaks with
    persona_id = controller.persona_save({**noam, "voice": "anna"})["id"]
    instead = controller.persona_save({**noam, "id": persona_id, "voice": "clone:noam-2"})
    assert instead["ok"] is False
    assert personas.load_all_personas()[persona_id].voice == "anna"


def test_a_persona_whose_cloned_voice_is_not_here_still_saves_its_other_changes():
    """Its wake word, say: its voice is for the user to choose again."""
    noam = {"name": "Noam", "engine": "kyutai", "voice": "clone:noam-2", "description": ""}
    persona_id = personas.save_persona(noam)
    personas.reload_personas()
    controller = AssistantController(EventBus())

    assert controller.persona_save({**noam, "id": persona_id, "speed": 1.2})["ok"]
    assert controller.set_persona_words(persona_id, wake_word="Hey Noam")["ok"]
    saved = personas.load_all_personas()[persona_id]
    assert (saved.voice, saved.speed, saved.wake_word) == ("clone:noam-2", 1.2, "Hey Noam")


def test_slow_cloning_requests_do_not_hold_up_the_others():
    slow = ("voice.clone_access", "voice.clone_connect", "voice.clone_fetch", "voice.clone")
    for method in (*slow, "voice.record", "voice.sample"):
        assert method in rpc_server.CONCURRENT

"""Unit tests for persona files: a persona exported to a file of its own,
with its cloned voice and the languages made of it, and imported from one.
A file comes from anywhere: one Iris does not take is refused with why, and
leaves nothing behind. Hugging Face and the models are stand-ins."""

import io
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from pocket_tts import export_model_state
from safetensors.torch import save_file

from iris import cloning, persona_files, personas, readiness
from iris.controller import AssistantController
from iris.events import EventBus
from iris.persona_files import (
    NEWER,
    NOT_A_PERSONA,
    STRANGE,
    TOO_LARGE,
    UNREADABLE,
    VOICE_DAMAGED,
    PersonaFileError,
    export_persona,
    import_persona,
)
from iris.tts import PocketGenerator
from tests.unit.test_voice_cloning import StandInKyutai, a_voice, speech


@pytest.fixture(autouse=True)
def nothing_forgotten(monkeypatch):
    monkeypatch.setattr(cloning, "_forgotten", set())


@pytest.fixture
def kyutai(monkeypatch):
    stand_in = StandInKyutai()
    monkeypatch.setattr(cloning, "KYUTAI", stand_in)
    return stand_in


def voice_state(path: Path, frames: int = 4) -> Path:
    """A voice as Pocket TTS exports one: each layer's attention cache."""
    state = {
        f"transformer.layers.{layer}.self_attn": {
            "cache": torch.randn(2, 1, frames, 2, 4),
            "offset": torch.tensor([frames]),
            "pad": torch.tensor([0]),
        }
        for layer in range(2)
    }
    export_model_state(state, path)
    return path


def cloned(name="Noam", languages=("english", "german")) -> Path:
    """A voice cloned here, made for these languages."""
    folder = a_voice(name)
    for language in languages:
        voice_state(folder / f"{language}.safetensors")
    return folder


NOAM = {
    "name": "Noam",
    "engine": "kyutai",
    "voice": "alba",
    "speed": 1.1,
    "audio_filter": "radio",
    "description": "You are Noam.\nYou take your time.",
    "wake_word": "Hey Noam",
    "sleep_word": "Goodnight",
}


def saved(**changes) -> personas.Persona:
    """A persona of the user's, saved."""
    return personas.find_persona(personas.save_persona({**NOAM, **changes}))


def exported(persona, folder: Path) -> Path:
    path = folder / f"{persona.name}.irispersona"
    export_persona(persona, path)
    return path


def held(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return sorted(archive.namelist())


def library() -> list[str]:
    """Every folder in the voices folder, hidden ones too."""
    folder = cloning.voices_dir()
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def user_personas() -> list[str]:
    folder = personas.USER_DIR
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


# ---------------------------------------------------------------------------
# To a file and back
# ---------------------------------------------------------------------------


def test_a_persona_with_a_voice_of_its_engine_goes_to_a_file_and_back(tmp_path):
    persona = saved()
    before = persona.to_dict()

    path = exported(persona, tmp_path)

    assert held(path) == ["persona.yaml"]
    with zipfile.ZipFile(path) as archive:
        data = yaml.safe_load(archive.read("persona.yaml"))
    assert data["format"] == persona_files.FORMAT
    assert data["voice"] == "alba" and data["wake_word"] == "Hey Noam"

    personas.delete_persona(persona.id)
    persona_id = import_persona(path)

    assert persona_id == "noam"
    assert personas.find_persona(persona_id).to_dict() == before
    assert library() == []


def test_a_persona_with_a_cloned_voice_takes_it_along_with_the_languages_made_of_it(tmp_path):
    folder = cloned("Noam")
    persona = saved(voice="clone:noam")
    sample = (folder / "sample.wav").read_bytes()
    states = {p.name: p.read_bytes() for p in folder.glob("*.safetensors")}

    path = exported(persona, tmp_path)

    assert held(path) == [
        "persona.yaml",
        "voice/english.safetensors",
        "voice/german.safetensors",
        "voice/sample.wav",
        "voice/voice.yaml",
    ]

    # Gone from here, the persona and its voice, and back from the file
    personas.delete_persona(persona.id)
    assert cloning.forget_voice("clone:noam")
    persona_id = import_persona(path)

    imported = personas.find_persona(persona_id)
    assert imported.name == "Noam"
    # The folder of a voice deleted while Iris runs is not taken again
    assert imported.voice == "clone:noam-2"
    folder = cloning.voice_folder(imported.voice)
    assert (folder / "sample.wav").read_bytes() == sample
    assert {p.name: p.read_bytes() for p in folder.glob("*.safetensors")} == states
    assert cloning.cloned_voices() == [{"id": "clone:noam-2", "name": "Noam", "language": "Cloned"}]
    voice = next(
        v
        for v in AssistantController(EventBus()).voice_list()["engines"]["kyutai"]
        if v["id"] == "clone:noam-2"
    )
    assert voice["cloned"]["made"] == ["en", "de"]
    assert voice["personas"] == [{"id": "noam", "name": "Noam"}]
    assert library() == ["noam-2"]


def test_an_imported_voice_speaks_the_languages_made_of_it_at_once(tmp_path, kyutai):
    cloned("Noam")
    path = exported(saved(voice="clone:noam"), tmp_path)
    personas.delete_persona("noam")
    cloning.forget_voice("clone:noam")
    voice = personas.find_persona(import_persona(path)).voice

    with patch("iris.tts.TTSModel") as models:
        models.load_model.side_effect = lambda language: kyutai.load_model(language)
        Path(PocketGenerator(voice=voice).generate("Guten Tag.", language="de")).unlink()

    # Read from the file: nothing downloaded, nothing made of the sample
    folder = cloning.voice_folder(voice)
    german = [p for m in kyutai.loaded if m.language == "german" for p in m.prompts]
    assert german == [str(folder / "german.safetensors")]
    assert not kyutai.fetched


def test_imported_where_its_name_and_voice_are_taken_both_get_a_number(tmp_path):
    cloned("Noam")
    path = exported(saved(voice="clone:noam", wake_word=""), tmp_path)

    second = import_persona(path)
    third = import_persona(path)

    assert (second, third) == ("noam-2", "noam-3")
    found = [personas.find_persona(persona_id) for persona_id in ("noam", second, third)]
    assert [p.name for p in found] == ["Noam", "Noam 2", "Noam 3"]
    # Woken by its name, as the persona it came from
    assert [p.wake_word for p in found] == ["Noam", "Noam 2", "Noam 3"]
    assert [p.voice for p in found] == ["clone:noam", "clone:noam-2", "clone:noam-3"]
    assert [v["name"] for v in cloning.cloned_voices()] == ["Noam", "Noam 2", "Noam 3"]


def test_a_persona_named_as_one_iris_ships_gets_a_number(tmp_path):
    path = exported(personas.find_persona("iris"), tmp_path)

    persona_id = import_persona(path)

    assert persona_id == "iris-2"
    assert personas.find_persona("iris-2").name == "Iris 2"
    assert personas.find_persona("iris").builtin


def test_a_voice_whose_name_another_voice_has_gets_a_number(tmp_path):
    cloned("Noam")
    path = exported(saved(voice="clone:noam"), tmp_path)
    personas.delete_persona("noam")
    # Another voice, called Noam since, in another folder
    cloning.rename_voice("clone:noam", "Someone")
    cloned("Other")
    cloning.rename_voice("clone:other", "Noam")

    voice = personas.find_persona(import_persona(path)).voice

    assert voice == "clone:noam-2"
    assert cloning.voice_name(cloning.voice_folder(voice)) == "Noam 2"


# ---------------------------------------------------------------------------
# Files Iris does not take
# ---------------------------------------------------------------------------


def persona_yaml(**changes) -> bytes:
    data = {
        "format": 1,
        "name": "Noam",
        "engine": "kyutai",
        "voice": "clone:noam",
        "speed": 1.0,
        "description": "You are Noam.\n",
    }
    data.update(changes)
    return yaml.safe_dump({k: v for k, v in data.items() if v is not ...}).encode()


def sample_wav(seconds=8.0, channels=1) -> bytes:
    audio = speech(seconds, rate=cloning.RATE)
    if channels > 1:
        audio = np.stack([audio] * channels, axis=1)
    out = io.BytesIO()
    sf.write(out, audio, cloning.RATE, format="WAV", subtype="PCM_16")
    return out.getvalue()


@pytest.fixture
def state_bytes(tmp_path) -> bytes:
    return voice_state(tmp_path / "state.safetensors").read_bytes()


@pytest.fixture
def files(state_bytes) -> dict:
    """What a persona file with a cloned voice holds, as Iris writes it."""
    return {
        "persona.yaml": persona_yaml(),
        "voice/voice.yaml": b"name: Noam\n",
        "voice/sample.wav": sample_wav(),
        "voice/english.safetensors": state_bytes,
    }


def written(folder: Path, files: dict, extra: list | None = None) -> Path:
    """A persona file holding these files, and these entries besides."""
    path = folder / "crafted.irispersona"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # a name given twice
            for info, data in extra or []:
                archive.writestr(info, data)
    return path


def refused(path: Path, message: str):
    """The file is refused with this message, and nothing of it is kept."""
    known = set(personas.PERSONAS)
    with pytest.raises(PersonaFileError) as refusal:
        import_persona(path)
    assert str(refusal.value) == message
    assert library() == [] and user_personas() == []
    assert set(personas.reload_personas()) == known
    result = AssistantController(EventBus()).persona_import(str(path))
    assert result == {"ok": False, "error": message}
    assert library() == [] and user_personas() == []


def test_a_crafted_file_as_iris_writes_it_is_taken(tmp_path, files):
    persona_id = import_persona(written(tmp_path, files))

    assert personas.find_persona(persona_id).voice == "clone:noam"
    assert cloning.made_for(cloning.voices_dir() / "noam") == ["english"]


def _link(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.external_attr = 0o120777 << 16
    return info


@pytest.mark.parametrize(
    "entry",
    [
        "../persona.yaml",
        "voice/../../outside.yaml",
        "/etc/outside.yaml",
        "voice/sample.wav/../../x",
        "notes.txt",
        "voice/notes.txt",
        "voice/klingon.safetensors",
        "persona.yaml",  # a second time
    ],
)
def test_a_file_holding_another_path_is_refused(tmp_path, files, entry):
    path = written(tmp_path, files, extra=[(zipfile.ZipInfo(entry), b"x")])
    assert entry in zipfile.ZipFile(path).namelist()
    refused(path, STRANGE)


def test_a_link_is_refused(tmp_path, files):
    del files["voice/sample.wav"]
    path = written(tmp_path, files, extra=[(_link("voice/sample.wav"), b"/etc/passwd")])
    refused(path, STRANGE)


def test_a_voice_beside_a_persona_that_does_not_speak_with_it_is_refused(tmp_path, files):
    files["persona.yaml"] = persona_yaml(voice="alba")
    refused(written(tmp_path, files), STRANGE)


@pytest.mark.parametrize("limit", ["MAX_TEXT", "MAX_SAMPLE", "MAX_STATE", "MAX_TOTAL", "MAX_FILE"])
def test_a_file_too_large_is_refused(tmp_path, files, monkeypatch, limit):
    monkeypatch.setattr(persona_files, limit, 50)
    refused(written(tmp_path, files), TOO_LARGE)


@pytest.mark.parametrize(
    "text",
    [
        b"name: [unclosed",
        b"- a list\n- of things\n",
        b"\xff\xfe not text",
        b"!!python/object/apply:os.system ['echo taken']",
        persona_yaml(name=...),
        persona_yaml(name=""),
        persona_yaml(name=12),
        persona_yaml(name="Line\nbreak"),
        persona_yaml(name="N" * 101),
        persona_yaml(description=["not", "text"]),
        persona_yaml(speed="fast"),
        persona_yaml(speed=True),
        persona_yaml(speed=9.0),
        persona_yaml(speed=float("nan")),
        persona_yaml(audio_filter="underwater"),
        persona_yaml(wake_word={"a": "mapping"}),
        persona_yaml(engine="kokoro", voice="../../voices.bin"),
    ],
)
def test_a_persona_that_cannot_be_read_is_refused(tmp_path, files, text):
    files["persona.yaml"] = text
    refused(written(tmp_path, files), UNREADABLE)


@pytest.mark.parametrize("voice", ["/etc/passwd", "https://example.com/voice.wav", "nobody"])
def test_a_voice_that_is_a_path_or_an_address_is_refused(tmp_path, voice):
    path = written(tmp_path, {"persona.yaml": persona_yaml(voice=voice)})
    refused(path, UNREADABLE)


def test_a_persona_of_an_engine_this_iris_lacks_is_refused(tmp_path):
    path = written(tmp_path, {"persona.yaml": persona_yaml(engine="espeak", voice="en")})
    refused(path, "The persona speaks with espeak, which this Iris does not have")


def test_a_damaged_voice_state_is_refused(tmp_path, files):
    files["voice/german.safetensors"] = b"\x08\x00\x00\x00\x00\x00\x00\x00{broken}"
    refused(written(tmp_path, files), VOICE_DAMAGED)


def test_a_cut_off_voice_state_is_refused(tmp_path, files, state_bytes):
    files["voice/english.safetensors"] = state_bytes[: len(state_bytes) // 2]
    refused(written(tmp_path, files), VOICE_DAMAGED)


@pytest.mark.parametrize(
    "tensors",
    [
        # Tensors of no module, or of a module within a module
        {"flat": torch.zeros(3), "m/cache": torch.zeros(2)},
        {"a/b/c": torch.zeros(3), "m/cache": torch.zeros(2)},
        # No attention cache, one that is not all numbers, or text
        {"m/offset": torch.tensor([4])},
        {"m/cache": torch.full((2, 3), float("nan"))},
        {"m/cache": torch.zeros(2), "m/text": torch.zeros(4, dtype=torch.uint8)},
    ],
)
def test_a_voice_state_not_as_pocket_tts_keeps_one_is_refused(tmp_path, files, tensors):
    state = tmp_path / "odd.safetensors"
    save_file(tensors, state)
    files["voice/english.safetensors"] = state.read_bytes()
    refused(written(tmp_path, files), VOICE_DAMAGED)


@pytest.mark.parametrize(
    "sample",
    [b"RIFF and nothing", sample_wav(channels=2), sample_wav(seconds=40.0), sample_wav(0.3)],
)
def test_a_sample_that_is_not_one_is_refused(tmp_path, files, monkeypatch, sample):
    monkeypatch.setattr(persona_files, "MAX_SAMPLE", 10_000_000)
    files["voice/sample.wav"] = sample
    refused(written(tmp_path, files), VOICE_DAMAGED)


@pytest.mark.parametrize("missing", ["voice/sample.wav", "voice/voice.yaml"])
def test_a_voice_missing_its_sample_or_name_is_refused(tmp_path, files, missing):
    del files[missing]
    refused(written(tmp_path, files), VOICE_DAMAGED)


@pytest.mark.parametrize("info", [b"name: [unclosed", b"- Noam\n", b"name: ''\n", b"name: 12\n"])
def test_a_voice_whose_name_cannot_be_read_is_refused(tmp_path, files, info):
    files["voice/voice.yaml"] = info
    refused(written(tmp_path, files), VOICE_DAMAGED)


def test_a_cloned_voice_left_out_is_refused(tmp_path):
    refused(written(tmp_path, {"persona.yaml": persona_yaml()}), VOICE_DAMAGED)


def test_a_file_whose_contents_changed_is_refused(tmp_path, files, state_bytes):
    path = written(tmp_path, files)
    data = bytearray(path.read_bytes())
    at = bytes(data).index(state_bytes[-64:])
    data[at : at + 8] = b"\x00" * 8 if data[at : at + 8] != b"\x00" * 8 else b"\x01" * 8
    path.write_bytes(bytes(data))
    refused(path, persona_files.DAMAGED)


def test_what_is_no_persona_file_is_refused(tmp_path):
    other = tmp_path / "notes.irispersona"
    other.write_text("Just some notes")
    refused(other, NOT_A_PERSONA)
    refused(written(tmp_path, {}), NOT_A_PERSONA)


def test_nothing_is_left_where_no_voices_were_before(tmp_path, files, monkeypatch):
    monkeypatch.setenv("IRIS_VOICES_DIR", str(tmp_path / "voices"))
    files["voice/english.safetensors"] = b"broken"
    with pytest.raises(PersonaFileError):
        import_persona(written(tmp_path, files))
    assert not (tmp_path / "voices").exists()


# ---------------------------------------------------------------------------
# The format's version
# ---------------------------------------------------------------------------


def test_a_file_of_a_newer_format_is_refused_as_such(tmp_path):
    path = written(tmp_path, {"persona.yaml": persona_yaml(format=2, voice="alba")})
    refused(path, NEWER)


@pytest.mark.parametrize("version", [..., "1", True, 0, 1.0, None])
def test_a_file_without_a_version_this_iris_writes_is_no_persona_file(tmp_path, version):
    path = written(tmp_path, {"persona.yaml": persona_yaml(format=version, voice="alba")})
    refused(path, NOT_A_PERSONA)


def test_a_file_of_the_version_this_iris_writes_is_taken(tmp_path):
    path = written(tmp_path, {"persona.yaml": persona_yaml(format=1, voice="alba")})
    assert personas.find_persona(import_persona(path)).voice == "alba"


# ---------------------------------------------------------------------------
# As the app asks
# ---------------------------------------------------------------------------


def test_an_imported_voice_is_made_ready_for_iriss_other_languages(tmp_path):
    cloned("Noam", languages=("english",))
    path = exported(saved(voice="clone:noam"), tmp_path)
    personas.delete_persona("noam")
    cloning.forget_voice("clone:noam")
    controller = AssistantController(EventBus())
    controller.languages = ["en", "fr"]
    asked = []
    controller.readier.request = lambda *args: asked.append(args)

    result = controller.persona_import(str(path))

    assert result["ok"] and result["id"] == "noam"
    assert "noam" in result["personas"] and result["names"]["noam"] == "Noam"
    assert ("kyutai", "clone:noam-2", "fr", "Noam", ["Noam"], readiness.VOICES) in asked


def test_the_app_exports_a_persona_as_saved(tmp_path):
    saved()
    controller = AssistantController(EventBus())
    path = tmp_path / "Noam.irispersona"

    assert controller.persona_export("noam", str(path)) == {"ok": True}
    assert held(path) == ["persona.yaml"]
    assert controller.persona_export("nobody", str(path))["ok"] is False


def test_a_persona_whose_cloned_voice_is_gone_is_not_exported(tmp_path):
    personas.save_persona({**NOAM, "voice": "clone:gone"})
    result = AssistantController(EventBus()).persona_export("noam", str(tmp_path / "x.irispersona"))
    assert result == {"ok": False, "error": "The voice Gone cannot be found"}
    assert list(tmp_path.iterdir()) == []


def test_a_file_that_cannot_be_written_leaves_nothing(tmp_path):
    saved()
    path = tmp_path / "gone" / "Noam.irispersona"
    result = AssistantController(EventBus()).persona_export("noam", str(path))
    assert result["ok"] is False and result["error"].startswith("The file could not be written")
    assert list(tmp_path.iterdir()) == []

"""Unit tests for personas: engines, and saving and deleting them."""

import pytest
import yaml

from iris import personas
from iris.personas import Persona, delete_persona, find_persona, save_persona, shipped


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """A built-in folder holding one persona, and an empty user folder."""
    builtin, user = tmp_path / "builtin", tmp_path / "user"
    builtin.mkdir()
    (builtin / "iris.yaml").write_text(
        'name: "Iris"\nengine: "kyutai"\nvoice: "alba"\ndescription: |\n  Warm.\n'
    )
    monkeypatch.setattr(personas, "BUILTIN_DIR", builtin)
    monkeypatch.setattr(personas, "USER_DIR", user)
    monkeypatch.delenv("IRIS_PERSONAS_DIR", raising=False)
    saved = dict(personas.PERSONAS)
    personas.reload_personas()
    yield builtin, user
    personas.PERSONAS.clear()
    personas.PERSONAS.update(saved)


SETTINGS = {
    "name": "Jeeves",
    "engine": "kokoro",
    "voice": "bm_george",
    "speed": 1.1,
    "audio_filter": "",
    "description": "A discreet butler.\nNever flustered.",
}


def test_a_kokoro_voice_without_engine_means_kokoro():
    assert Persona("HAL", "Calm.", voice="bm_lewis").engine == "kokoro"
    assert Persona("Iris", "Warm.", voice="alba").engine == "kyutai"
    assert Persona("Iris", "Warm.", voice="alba", engine="kyutai").engine == "kyutai"


def test_a_new_persona_is_saved_in_the_user_folder(dirs):
    _, user = dirs
    persona_id = save_persona(SETTINGS)

    assert persona_id == "jeeves"
    data = yaml.safe_load((user / "jeeves.yaml").read_text())
    assert data == {
        "name": "Jeeves",
        "engine": "kokoro",
        "voice": "bm_george",
        "speed": 1.1,
        "description": "A discreet butler.\nNever flustered.\n",
    }
    persona = find_persona("jeeves")
    assert (persona.engine, persona.voice, persona.personality) == (
        "kokoro",
        "bm_george",
        "A discreet butler.\nNever flustered.\n",
    )


def test_the_personality_is_written_as_a_block(dirs):
    _, user = dirs
    save_persona(SETTINGS)
    assert "description: |" in (user / "jeeves.yaml").read_text()


def test_a_new_persona_never_replaces_another(dirs):
    assert save_persona({**SETTINGS, "name": "Iris"}) == "iris-2"
    assert save_persona({**SETTINGS, "name": "Iris"}) == "iris-3"
    assert find_persona("iris").voice == "alba"


def test_editing_a_shipped_persona_saves_a_copy_that_replaces_it(dirs):
    builtin, user = dirs
    save_persona({**SETTINGS, "name": "Iris", "engine": "kyutai", "voice": "eve"}, "iris")

    assert (user / "iris.yaml").exists()
    assert "alba" in (builtin / "iris.yaml").read_text()
    assert find_persona("iris").voice == "eve"
    assert shipped("iris") and not find_persona("iris").builtin


def test_editing_a_user_persona_rewrites_its_file(dirs):
    _, user = dirs
    save_persona(SETTINGS)
    save_persona({**SETTINGS, "name": "Jeeves the Butler"}, "jeeves")

    assert sorted(p.name for p in user.iterdir()) == ["jeeves.yaml"]
    assert find_persona("jeeves").name == "Jeeves the Butler"


def test_deleting_a_user_persona_removes_it(dirs):
    save_persona(SETTINGS)
    assert delete_persona("jeeves") is False
    assert find_persona("jeeves") is None


def test_deleting_an_edited_shipped_persona_resets_it(dirs):
    save_persona({**SETTINGS, "name": "Iris"}, "iris")
    assert delete_persona("iris") is True
    assert find_persona("iris").voice == "alba"


def test_a_shipped_persona_cannot_be_deleted(dirs):
    builtin, _ = dirs
    assert delete_persona("iris") is True
    assert (builtin / "iris.yaml").exists()


def test_to_dict_is_what_the_editor_shows(dirs):
    assert find_persona("iris").to_dict() == {
        "name": "Iris",
        "engine": "kyutai",
        "voice": "alba",
        "speed": 1.25,
        "audio_filter": "",
        "description": "Warm.\n",
        "wake_word": "Iris",
        "sleep_word": "",
    }

"""The Voices page: every voice with the personas that speak with it, and
the cloned voices managed there: renamed, cloned again from a new sample,
and deleted, a voice in use only once its personas have another. Hugging
Face and the models are stand-ins: nothing is downloaded."""

import base64
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from iris import cloning, personas, rpc_server
from iris.cloning import Cloning, cloned_voices, made_for, voice_folder
from iris.controller import AssistantController
from iris.events import EventBus
from iris.personas import slug as persona_slug
from iris.tts import PocketGenerator
from tests.unit.test_voice_cloning import StandInKyutai, a_voice, speech, wav


@pytest.fixture(autouse=True)
def nothing_forgotten(monkeypatch):
    monkeypatch.setattr(cloning, "_forgotten", set())


@pytest.fixture
def kyutai(monkeypatch):
    stand_in = StandInKyutai()
    monkeypatch.setattr(cloning, "KYUTAI", stand_in)
    return stand_in


@pytest.fixture
def heard_by_energy(monkeypatch):
    monkeypatch.setattr(
        cloning, "SpeechDetector", lambda: lambda frame: float(np.sqrt(np.mean(frame**2)) > 0.01)
    )


def made(name="Ada") -> Path:
    """A cloned voice made for English."""
    folder = a_voice(name)
    (folder / "english.safetensors").write_bytes(b"made when cloned")
    return folder


def speaking(controller, persona_id="ada", voice="clone:ada", name="Ada"):
    """A persona of the user's that speaks with a voice."""
    settings = {"name": name, "engine": "kyutai", "voice": voice, "description": "", "speed": 1.0}
    return controller.persona_save(settings)["id"]


def by_id(listing, engine="kyutai"):
    return {voice["id"]: voice for voice in listing["engines"][engine]}


# ---------------------------------------------------------------------------
# Every voice
# ---------------------------------------------------------------------------


def test_every_voice_is_listed_with_the_personas_that_speak_with_it():
    made("Ada")
    controller = AssistantController(EventBus())
    speaking(controller)

    listing = controller.voice_list()

    kyutai = by_id(listing)
    assert kyutai["eve"]["personas"] == [{"id": "iris", "name": "Iris"}]
    assert kyutai["alba"]["personas"] == []
    assert kyutai["clone:ada"]["personas"] == [{"id": "ada", "name": "Ada"}]
    # The languages each speaks, and those it is ready for
    assert kyutai["eve"]["languages"] == ["en", "fr", "it", "de", "es", "pt", "nl"]
    assert kyutai["eve"]["ready"] == kyutai["eve"]["languages"]
    assert listing["chosen"] == ["en"] and listing["fallback"] == "eve"
    assert "cloned" not in kyutai["eve"]


def test_a_cloned_voice_shows_its_folder_and_the_languages_made_of_it():
    folder = made("Ada")
    (folder / "german.safetensors").write_bytes(b"made later")

    ada = by_id(AssistantController(EventBus()).voice_list())["clone:ada"]

    assert ada["cloned"] == {"folder": str(folder), "made": ["en", "de"]}
    assert ada["name"] == "Ada" and ada["language"] == "Cloned"


# ---------------------------------------------------------------------------
# Renaming
# ---------------------------------------------------------------------------


def test_a_cloned_voice_is_renamed_and_its_personas_keep_it():
    made("Ada")
    controller = AssistantController(EventBus())
    speaking(controller)

    result = controller.cloning.rename("clone:ada", "  Ada   Lovelace ")

    assert result == {
        "ok": True,
        "voice": {"id": "clone:ada", "name": "Ada Lovelace", "language": "Cloned"},
    }
    assert personas.load_all_personas()["ada"].voice == "clone:ada"
    assert by_id(controller.voice_list())["clone:ada"]["name"] == "Ada Lovelace"


def test_a_voice_needs_a_name_of_its_own():
    made("Ada")
    made("Bea")
    service = Cloning(lambda t, d: None)
    assert service.rename("clone:ada", "  ") == {"ok": False, "error": "The voice needs a name"}
    assert service.rename("clone:ada", "bea") == {
        "ok": False,
        "error": "Another voice is called bea",
    }
    assert service.rename("clone:nobody", "Cy") == {"ok": False, "error": "The voice is gone"}
    assert service.rename("anna", "Cy")["ok"] is False
    assert [v["name"] for v in cloned_voices()] == ["Ada", "Bea"]


# ---------------------------------------------------------------------------
# Cloning again, from a new sample
# ---------------------------------------------------------------------------


def _sampled(service):
    data = base64.b64encode(wav(np.concatenate([speech(6), speech(4)]))).decode()
    assert service.use_file(data, "again.wav")["ok"]


def test_cloned_again_a_voice_keeps_its_name_and_personas_and_drops_the_old(
    kyutai, heard_by_energy, caplog
):
    folder = made("Ada")
    (folder / "german.safetensors").write_bytes(b"made of the old sample")
    old_sample = (folder / cloning.SAMPLE_FILE).read_bytes()
    changed = []
    service = Cloning(lambda t, d: None, changed=changed.append)
    _sampled(service)

    with caplog.at_level(logging.INFO, logger="iris.cloning"):
        result = service.clone("Someone else", replace="clone:ada")

    assert result == {"ok": True, "voice": {"id": "clone:ada", "name": "Ada", "language": "Cloned"}}
    assert (folder / cloning.SAMPLE_FILE).read_bytes() != old_sample
    # English made again of the new sample, the rest to be made again
    assert made_for(folder) == ["english"]
    assert (folder / "english.safetensors").read_bytes() != b"made when cloned"
    assert changed == ["clone:ada"]
    assert [v["id"] for v in cloned_voices()] == ["clone:ada"]
    # What it replaces is logged, with the folder
    [said] = [r.getMessage() for r in caplog.records if "replaced" in r.getMessage()]
    assert "clone:ada" in said and str(folder) in said


def test_a_voice_that_cannot_be_made_again_stays_as_it_was(kyutai, heard_by_energy):
    kyutai.terms = False
    folder = made("Ada")
    before = sorted(path.name for path in folder.iterdir())
    service = Cloning(lambda t, d: None)
    _sampled(service)

    result = service.clone("Ada", replace="clone:ada")

    assert result["ok"] is False and result["access"] is True
    assert sorted(path.name for path in folder.iterdir()) == before
    assert [v["id"] for v in cloned_voices()] == ["clone:ada"]


@pytest.mark.readiness
def test_a_voice_cloned_is_made_ready_for_the_languages_confirmed_with_it(kyutai, heard_by_energy):
    """The languages picked as the clone is confirmed are the ones made,
    English always: one of Iris's left out is not made."""
    controller = AssistantController(EventBus())
    controller.languages = ["en", "fr"]
    _sampled(controller.cloning)

    result = controller.voice_clone({"name": "Ada", "languages": ["de"]})

    assert result["ok"] and result["voice"]["id"] == "clone:ada"
    assert [job["language"] for job in result["jobs"]] == ["de"]
    assert result["jobs"][0]["origin"] == "clone"
    # German's cloning model is not here: it downloads first, as its job shows
    assert controller.readier.wait(result["jobs"][0]["id"], 5)
    assert "german" in kyutai.fetched
    assert made_for(voice_folder("clone:ada")) == ["english", "german"]


@pytest.mark.readiness
def test_a_voice_cloned_without_languages_named_is_made_for_iriss(kyutai, heard_by_energy):
    controller = AssistantController(EventBus())
    controller.languages = ["en", "fr"]
    _sampled(controller.cloning)

    result = controller.voice_clone({"name": "Ada"})

    assert [job["language"] for job in result["jobs"]] == ["fr"]
    assert controller.readier.wait(result["jobs"][0]["id"], 5)
    assert made_for(voice_folder("clone:ada")) == ["english", "french"]


@pytest.mark.readiness
def test_english_alone_needs_no_job_beside_the_clone(kyutai, heard_by_energy):
    controller = AssistantController(EventBus())
    controller.languages = ["en", "fr"]
    _sampled(controller.cloning)

    result = controller.voice_clone({"name": "Ada", "languages": []})

    assert result["ok"] and result["jobs"] == []
    assert made_for(voice_folder("clone:ada")) == ["english"]


def test_a_generator_drops_a_voice_made_again(kyutai):
    folder = made("Ada")
    with patch("iris.tts.TTSModel") as model:
        model.load_model.side_effect = lambda language: kyutai.load_model(language)
        generator = PocketGenerator("clone:ada")
        generator._voice_state("english", generator.voice)
        assert generator._voice_states

        generator.voice_changed("clone:ada")

    assert not generator._voice_states
    assert generator.voice == str(folder)


# ---------------------------------------------------------------------------
# Deleting
# ---------------------------------------------------------------------------


def test_a_cloned_voice_no_persona_speaks_with_is_deleted():
    folder = made("Ada")
    result = AssistantController(EventBus()).voice_delete("clone:ada")
    assert result["ok"] and not folder.exists() and cloned_voices() == []


def test_a_voice_in_use_is_deleted_only_once_its_personas_have_another():
    folder = made("Ada")
    controller = AssistantController(EventBus())
    speaking(controller)

    refused = controller.voice_delete("clone:ada")

    assert refused["ok"] is False and refused["error"] == "Ada speaks with this voice"
    assert refused["personas"] == [{"id": "ada", "name": "Ada"}]
    assert folder.exists()

    assert controller.voice_delete("clone:ada", reassign=True)["ok"]
    assert not folder.exists()
    ada = personas.load_all_personas()["ada"]
    assert (ada.engine, ada.voice) == ("kyutai", "eve")


def test_only_a_cloned_voice_is_deleted():
    assert AssistantController(EventBus()).voice_delete("anna")["ok"] is False


def test_a_voice_deleted_is_logged_with_its_folder(caplog):
    folder = made("Ada")
    with caplog.at_level(logging.INFO, logger="iris.cloning"):
        assert AssistantController(EventBus()).voice_delete("clone:ada")["ok"]
    [said] = [r.getMessage() for r in caplog.records if "removing" in r.getMessage()]
    assert "clone:ada" in said and str(folder) in said


def test_a_cloned_voice_that_is_not_here_does_not_play():
    result = AssistantController(EventBus()).persona_preview(
        {"name": "Noam", "engine": "kyutai", "voice": "clone:noam-2"}
    )
    assert result == {"ok": False, "error": "The voice Noam 2 cannot be found"}


# ---------------------------------------------------------------------------
# Names, and the methods
# ---------------------------------------------------------------------------


def test_a_voice_named_with_other_letters_gets_a_folder_that_reads_well():
    assert cloning.slug("Der Führer") == "der-fuehrer"
    assert cloning.slug("Zoë Ångström") == "zoe-angstroem"
    assert cloning.slug("Straße") == "strasse"
    assert cloning.slug("アンナ") == "voice"
    assert persona_slug("Der Führer") == "der-fuehrer"
    assert a_voice("Der Führer").name == "der-fuehrer"


def test_a_folder_named_before_still_is_a_voice():
    folder = cloning.voices_dir() / "der-f-hrer"
    folder.mkdir(parents=True)
    (folder / cloning.SAMPLE_FILE).write_bytes(b"")
    assert voice_folder("clone:der-f-hrer") == folder
    assert [v["id"] for v in cloned_voices()] == ["clone:der-f-hrer"]


def test_managing_voices_comes_in_order_with_the_personas_saved():
    for method in ("voice.list", "voice.rename", "voice.delete"):
        assert method not in rpc_server.CONCURRENT

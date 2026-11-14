"""Every voice, with the personas that speak with it and the languages it
is ready for, and a voice cloned made ready for the languages chosen with
it. Hugging Face and the models are stand-ins: nothing is downloaded."""

import base64
from pathlib import Path

import numpy as np
import pytest

from iris import cloning
from iris.cloning import made_for, voice_folder
from iris.controller import AssistantController
from iris.events import EventBus
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
    assert kyutai["anna"]["personas"] == [{"id": "iris", "name": "Iris"}]
    assert kyutai["alba"]["personas"] == []
    assert kyutai["clone:ada"]["personas"] == [{"id": "ada", "name": "Ada"}]
    # The languages each speaks, and those it is ready for
    assert kyutai["anna"]["languages"] == ["en", "fr", "it", "de", "es", "pt", "nl"]
    assert kyutai["anna"]["ready"] == kyutai["anna"]["languages"]
    assert listing["chosen"] == ["en"]


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------


def _sampled(service):
    data = base64.b64encode(wav(np.concatenate([speech(6), speech(4)]))).decode()
    assert service.use_file(data, "again.wav")["ok"]


@pytest.mark.readiness
def test_a_voice_cloned_is_made_ready_for_the_languages_chosen_with_it(kyutai, heard_by_energy):
    controller = AssistantController(EventBus())
    controller.languages = ["en", "fr"]
    _sampled(controller.cloning)

    result = controller.voice_clone({"name": "Ada", "languages": ["de"]})

    assert result["ok"] and result["voice"]["id"] == "clone:ada"
    assert sorted(job["language"] for job in result["jobs"]) == ["de", "fr"]
    for job in result["jobs"]:
        assert controller.readier.wait(job["id"], 5)
    assert made_for(voice_folder("clone:ada")) == ["english", "french", "german"]

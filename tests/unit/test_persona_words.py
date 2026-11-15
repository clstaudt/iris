"""Unit tests for each persona's wake word and sleep word: their defaults,
switching persona, saving them, and taking on the words an earlier Iris
kept for all personas."""

import time

import pytest
import yaml

from iris import personas
from iris.assistant import VoiceAssistant
from iris.audio import AudioPlayer
from iris.controller import AssistantController
from iris.events import EventBus
from iris.personas import Persona, find_persona, get_persona, save_persona
from iris.tts import SpeechGenerator

from tests.unit.test_controller import FakeAssistant, _wait_for


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Two built-in personas, one with words of its own, and an empty user
    folder."""
    builtin, user = tmp_path / "builtin", tmp_path / "user"
    builtin.mkdir()
    (builtin / "iris.yaml").write_text(
        'name: "Iris"\nengine: "kyutai"\nvoice: "alba"\ndescription: |\n  Warm.\n'
    )
    (builtin / "hal.yaml").write_text(
        'name: "HAL"\nengine: "kokoro"\nvoice: "bm_lewis"\nwake_word: "Hey HAL"\n'
        'sleep_word: "Goodbye, HAL"\ndescription: |\n  Calm.\n'
    )
    monkeypatch.setattr(personas, "BUILTIN_DIR", builtin)
    monkeypatch.setattr(personas, "USER_DIR", user)
    monkeypatch.delenv("IRIS_PERSONAS_DIR", raising=False)
    saved = dict(personas.PERSONAS)
    personas.reload_personas()
    yield builtin, user
    personas.PERSONAS.clear()
    personas.PERSONAS.update(saved)


class RunningAssistant(FakeAssistant):
    """A running assistant whose persona can be switched and edited."""

    def __init__(self, wake_word=None, sleep_word=None):
        super().__init__(wake_word=wake_word, sleep_word=sleep_word)
        self.persona = get_persona("iris")
        self.conversation.add_note = lambda note: None
        self.tts_generator = SpeechGenerator.__new__(SpeechGenerator)
        self.tts_generator.prepare = lambda: None
        self.player = AudioPlayer()

    def config_info(self):
        return {"wake_word": self.wake_word or "", "sleep_word": self.sleep_word or ""}


def _controller(wake_word=None, sleep_word=None):
    controller = AssistantController(EventBus())
    controller.assistant = RunningAssistant(wake_word, sleep_word)
    return controller


# Defaults -----------------------------------------------------------------


def test_the_wake_word_is_the_persona_s_name_and_there_is_no_sleep_word():
    persona = Persona("Klaus", "Dry.", voice="alba")
    assert persona.wake_word == "Klaus"
    assert persona.sleep_word == ""


def test_a_persona_file_can_give_its_own_words(dirs):
    hal = find_persona("hal")
    assert (hal.wake_word, hal.sleep_word) == ("Hey HAL", "Goodbye, HAL")
    iris = find_persona("iris")
    assert (iris.wake_word, iris.sleep_word) == ("Iris", "")


def test_the_shipped_personas_are_woken_by_their_names():
    for persona in personas.load_all_personas().values():
        if persona.builtin:
            assert persona.wake_word == persona.name
            assert persona.sleep_word == ""


def test_iris_starts_with_the_persona_s_words_when_the_wake_word_is_on(dirs, monkeypatch):
    monkeypatch.setattr("iris.controller.VoiceAssistant", FakeAssistant)
    controller = AssistantController(EventBus())
    try:
        controller.start({"persona": "hal", "wake_on": True})
        assert _wait_for(lambda: controller.assistant is not None)
        assert controller.assistant.kwargs["wake_word"] == "Hey HAL"
        assert controller.assistant.kwargs["sleep_word"] == "Goodbye, HAL"
    finally:
        controller.stop()


def test_with_the_wake_word_off_iris_starts_listening_to_everything(dirs, monkeypatch):
    monkeypatch.setattr("iris.controller.VoiceAssistant", FakeAssistant)
    controller = AssistantController(EventBus())
    try:
        controller.start({"persona": "hal", "wake_on": False})
        assert _wait_for(lambda: controller.assistant is not None)
        assert controller.assistant.kwargs["wake_word"] is None
    finally:
        controller.stop()


def test_the_switch_turns_on_the_persona_s_word(dirs):
    controller = _controller()
    controller.set_persona("hal")

    assert controller.set_wake(True)["ok"] is True
    assert controller.assistant.wake_word == "Hey HAL"
    controller.set_wake(False)
    assert controller.assistant.wake_word is None


# Switching persona ----------------------------------------------------------


def test_switching_persona_switches_the_words(dirs):
    controller = _controller(wake_word="Iris")

    controller.set_persona("hal")
    assert controller.assistant.wake_word == "Hey HAL"
    assert controller.assistant.sleep_word == "Goodbye, HAL"

    controller.set_persona("iris")
    assert controller.assistant.wake_word == "Iris"
    assert controller.assistant.sleep_word is None


def test_switching_persona_leaves_the_wake_word_off(dirs):
    controller = _controller()
    controller.set_persona("hal")
    assert controller.assistant.wake_word is None
    assert controller.assistant.sleep_word == "Goodbye, HAL"


def _awake(wake_word: str) -> VoiceAssistant:
    a = object.__new__(VoiceAssistant)
    a.wake_word = wake_word
    a.sleep_word = None
    a.awake_until = time.monotonic() + 30
    return a


def test_iris_stays_awake_through_a_switch():
    a = _awake("Iris")
    a.set_persona_words("Hey HAL", "Goodbye, HAL")
    assert (a.wake_word, a.sleep_word) == ("Hey HAL", "Goodbye, HAL")
    assert a.awake_until > time.monotonic()


# Saving ---------------------------------------------------------------------


SETTINGS = {
    "name": "Jeeves",
    "engine": "kokoro",
    "voice": "bm_george",
    "speed": 1.0,
    "audio_filter": "",
    "description": "A butler.",
}


def test_words_that_are_the_defaults_are_not_written(dirs):
    _, user = dirs
    save_persona({**SETTINGS, "wake_word": "jeeves", "sleep_word": ""})
    data = yaml.safe_load((user / "jeeves.yaml").read_text())
    assert "wake_word" not in data and "sleep_word" not in data


def test_words_of_its_own_are_written_and_read_back(dirs):
    _, user = dirs
    save_persona({**SETTINGS, "wake_word": "Butler", "sleep_word": "Thank you, Jeeves"})
    data = yaml.safe_load((user / "jeeves.yaml").read_text())
    assert (data["wake_word"], data["sleep_word"]) == ("Butler", "Thank you, Jeeves")
    persona = find_persona("jeeves")
    assert (persona.wake_word, persona.sleep_word) == ("Butler", "Thank you, Jeeves")


def test_a_wake_word_that_is_the_name_follows_a_rename(dirs):
    persona_id = save_persona({**SETTINGS, "wake_word": "Jeeves"})
    save_persona({**SETTINGS, "name": "Alfred", "wake_word": ""}, persona_id)
    assert find_persona(persona_id).wake_word == "Alfred"


def test_setting_a_word_saves_it_to_the_current_persona_and_applies_it(dirs):
    _, user = dirs
    controller = _controller(wake_word="Iris")

    result = controller.set_persona_words(sleep_word="Goodnight")

    assert result["ok"] is True
    data = yaml.safe_load((user / "iris.yaml").read_text())
    assert data["sleep_word"] == "Goodnight"
    assert controller.assistant.sleep_word == "Goodnight"

    controller.set_persona_words(wake_word="Zanzibar")
    assert controller.assistant.wake_word == "Zanzibar"
    controller.set_persona_words(wake_word="")  # its name again
    assert controller.assistant.wake_word == "Iris"


def test_an_unchanged_word_writes_nothing(dirs):
    _, user = dirs
    controller = _controller()
    assert controller.set_persona_words("iris", wake_word="Iris", sleep_word="")["ok"] is True
    assert not (user / "iris.yaml").exists()


def test_words_are_set_before_iris_runs(dirs):
    _, user = dirs
    controller = AssistantController(EventBus())
    assert controller.set_persona_words("hal", wake_word="Computer")["ok"] is True
    assert find_persona("hal").wake_word == "Computer"
    assert (user / "hal.yaml").exists()


def test_an_unknown_persona_is_refused(dirs):
    assert AssistantController(EventBus()).set_persona_words("nobody", wake_word="Hi")["ok"] is False


# Taking on the words kept for all personas before -----------------------------


def test_a_wake_word_and_sleep_word_set_before_go_to_the_persona_in_use(dirs):
    _, user = dirs
    controller = AssistantController(EventBus())

    result = controller.adopt_wake_words("iris", "Jarvis", "That's all")

    assert result == {**result, "ok": True, "wake_on": True}
    data = yaml.safe_load((user / "iris.yaml").read_text())
    assert (data["wake_word"], data["sleep_word"]) == ("Jarvis", "That's all")
    # The other personas keep theirs
    assert find_persona("hal").wake_word == "Hey HAL"


def test_the_old_default_wake_word_changes_no_file(dirs):
    _, user = dirs
    result = AssistantController(EventBus()).adopt_wake_words("iris", "Iris", "")
    assert result["wake_on"] is True
    assert not user.exists() or not list(user.iterdir())


def test_no_wake_word_before_means_the_switch_is_off(dirs):
    _, user = dirs
    result = AssistantController(EventBus()).adopt_wake_words("hal", "", "")
    assert result == {"ok": True, "wake_on": False}
    assert not user.exists() or not list(user.iterdir())


def test_iris_starts_with_the_words_taken_on(dirs, monkeypatch):
    monkeypatch.setattr("iris.controller.VoiceAssistant", FakeAssistant)
    controller = AssistantController(EventBus())
    controller.adopt_wake_words("hal", "Jarvis", "Goodnight")
    try:
        controller.start({"persona": "hal", "wake_on": True})
        assert _wait_for(lambda: controller.assistant is not None)
        assert controller.assistant.kwargs["wake_word"] == "Jarvis"
        assert controller.assistant.kwargs["sleep_word"] == "Goodnight"
    finally:
        controller.stop()


def test_the_words_of_a_persona_gone_since_go_to_iris(dirs):
    _, user = dirs
    result = AssistantController(EventBus()).adopt_wake_words("klaus", "Jarvis", "")
    assert result["ok"] is True and result["wake_on"] is True
    assert find_persona("iris").wake_word == "Jarvis"

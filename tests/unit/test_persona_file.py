"""Unit tests for a persona file changed on disk while Iris runs: the
persona in use takes it on, between replies."""

from iris.controller import AssistantController
from iris.events import EventBus

from tests.unit.test_persona_words import RunningAssistant, dirs  # noqa: F401


EDITED = (
    'name: "Iris"\nengine: "kyutai"\nvoice: "anna"\nspeed: 1.0\n'
    'wake_word: "Hey Iris"\nsleep_word: "Goodnight"\ndescription: |\n  Warmer.\n'
)


def _running():
    controller = AssistantController(EventBus())
    controller.assistant = RunningAssistant(wake_word="Iris")
    controller.set_persona("iris")  # as taken on at the start
    return controller


def test_the_voice_follows_the_persona_file_changed_on_disk(dirs):  # noqa: F811
    _, user = dirs
    controller = _running()
    assistant = controller.assistant
    assert assistant.tts_generator.voice == "alba"

    # The built-in Iris is replaced by the user's own, with another voice
    user.mkdir()
    (user / "iris.yaml").write_text(EDITED)
    controller._follow_persona_file()

    assert (assistant.tts_generator.engine, assistant.tts_generator.voice) == ("kyutai", "anna")
    assert assistant.persona.personality == "Warmer.\n"
    assert (assistant.wake_word, assistant.sleep_word) == ("Hey Iris", "Goodnight")

    # And that file edited again
    (user / "iris.yaml").write_text(EDITED.replace("anna", "marius"))
    controller._follow_persona_file()
    assert assistant.tts_generator.voice == "marius"


def test_a_reply_being_given_keeps_its_voice_to_the_end(dirs):  # noqa: F811
    _, user = dirs
    controller = _running()
    assistant = controller.assistant
    user.mkdir()
    (user / "iris.yaml").write_text(EDITED)

    assistant._turn_active.set()
    controller._follow_persona_file()
    assert assistant.tts_generator.voice == "alba"

    assistant._turn_active.clear()
    controller._follow_persona_file()
    assert assistant.tts_generator.voice == "anna"


def test_an_edit_undone_brings_the_shipped_persona_back(dirs):  # noqa: F811
    _, user = dirs
    controller = _running()
    user.mkdir()
    (user / "iris.yaml").write_text(EDITED)
    controller._follow_persona_file()

    (user / "iris.yaml").unlink()
    controller._follow_persona_file()

    assert controller.assistant.tts_generator.voice == "alba"
    assert controller.assistant.wake_word == "Iris"


def test_an_unchanged_file_is_left_alone(dirs):  # noqa: F811
    controller = _running()
    switched = []
    controller.set_persona = lambda name: switched.append(name)
    controller._follow_persona_file()
    assert switched == []

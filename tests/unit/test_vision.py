"""Camera and screen: the image is captured as the plugins capture it, and
goes to a model that can see it. The model of the conversation answers it
when it can see; otherwise the server's vision model does, and without one
the image is left out and the user is told. Images attached to a typed
message take the same way.

The camera and the screen are the scripted ones of the voice test (OpenCV
and mss replaced, the plugins' own code running), and the model server is
the scripted model, with or without a model that sees."""

import base64
import io
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from iris.assistant import NO_VISION, VoiceAssistant
from iris.attachments import prepare_all
from iris.controller import AssistantController
from iris.events import EventBus
from iris.llm import ConversationManager
from iris.plugins import camera, screen
from iris.tools import Tools
from irisdev.voicetest.fakesight import FakeCamera, FakeScreen, color_of, image_size
from irisdev.voicetest.stub_llm import MODEL, NO_IMAGES, VISION_MODEL, Rule, StubModel, images_of


@pytest.fixture
def slept(monkeypatch):
    """The camera's warm-up, without the wait."""
    waits = []
    monkeypatch.setattr(camera, "time", SimpleNamespace(sleep=waits.append))
    return waits


def _seen(messages) -> str:
    """What the scripted model says of the image it was shown."""
    images = images_of(messages)
    if not images:
        return "I cannot see anything."
    return f"I see something {color_of(images[0])}."


LOOK = Rule("holding|looking at", tool=("look", lambda m: {}), lead_in="Let me look.", after_tool=_seen)
SCREEN = Rule("screen", tool=("capture_screen", lambda m: {}), after_tool=_seen)


@pytest.fixture
def stub():
    model = StubModel(rules=[LOOK, SCREEN], first_token=0.0, per_word=0.0)
    yield model
    model.close()


def _conversation(stub, vision_model=None) -> ConversationManager:
    return ConversationManager(
        MODEL,
        host=stub.url,
        tools=Tools([camera.plugin, screen.plugin]),
        vision_model=vision_model,
    )


class Pipeline:
    current_language = "en"

    def begin_reply(self):
        pass

    def queue_speech(self, text, language=None, sound=None):
        pass

    def stop(self):
        pass

    def wait_for_speech(self):
        pass

    def said(self):
        return ""


def _assistant(conversation) -> VoiceAssistant:
    assistant = object.__new__(VoiceAssistant)
    assistant.event_bus = MagicMock()
    assistant.speak_replies = True
    assistant.conversation = conversation
    assistant.speech_pipeline = Pipeline()
    assistant.tts_generator = SimpleNamespace(languages=lambda: frozenset({"en"}))
    assistant.interrupt_event = threading.Event()
    assistant.suspend_event = threading.Event()
    assistant.tools = conversation.tools
    assistant.wake_word = None
    assistant.conversation_language = "en"
    assistant._timing = None

    class Status:
        status_lock = threading.Lock()
        status_thinking = False

    assistant.status_manager = Status()
    return assistant


def _errors(assistant) -> list[dict]:
    return [c.args[1] for c in assistant.event_bus.emit.call_args_list if c.args[0] == "error"]


# The camera ---------------------------------------------------------------------


def test_the_camera_takes_a_frame_after_warming_up_and_lets_go_of_it(slept):
    with FakeCamera(color="red", size=(640, 480)) as fake:
        result = camera.look()
    assert result["success"] is True and result["message"] == "Image captured"
    assert base64.b64decode(result["image"])[:3] == b"\xff\xd8\xff"
    assert image_size(result["image"]) == (640, 480)
    # Turned from OpenCV's BGR, the red stays red
    assert color_of(result["image"]) == "red"
    [capture] = fake.captures
    # Five frames thrown away while it warms up, then the one kept
    assert capture.reads == 6 and sum(slept) >= 1.5
    # Let go at once, so the camera's light goes off
    assert capture.released


def test_without_a_camera_the_model_is_told_and_nothing_is_held(slept):
    with FakeCamera(available=False) as fake:
        result = camera.look()
    assert result["success"] is False and result["message"] == "Camera not available"
    assert fake.captures[0].released and fake.captures[0].reads == 0


def test_a_camera_that_gives_no_frame_or_breaks_is_let_go(slept):
    with FakeCamera(frames=False) as fake:
        assert camera.look() == {"success": False, "message": "Failed to capture image"}
    assert fake.captures[0].released
    with FakeCamera(breaks=RuntimeError("device lost")) as fake:
        result = camera.look()
    assert result == {"success": False, "message": "Error capturing image: device lost"}
    assert fake.captures[0].released


def test_on_a_mac_the_user_is_told_where_to_allow_the_camera(slept, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with FakeCamera(available=False):
        result = camera.look()
    assert "Privacy & Security > Camera" in result["notice"]
    # Elsewhere there is nothing to allow
    monkeypatch.setattr(sys, "platform", "linux")
    with FakeCamera(available=False):
        assert "notice" not in camera.look()


def test_the_camera_switch_brings_the_look_tool():
    with FakeCamera():
        tools = Tools([camera.plugin], choices={"camera": False})
        assert tools.names() == []
        assert tools.set_enabled("camera", True)
        assert tools.names() == ["look"]


# The screen ---------------------------------------------------------------------


def test_the_primary_screen_is_captured_and_scaled_down():
    with FakeScreen(color="blue", size=(2880, 1800)) as fake:
        result = screen.capture_screen()
    assert result["success"] is True and result["message"] == "Screen captured"
    # The first entry of mss's list is all monitors together; the primary
    # one comes next
    assert fake.grabbed == [1]
    assert image_size(result["image"]) == (1920, 1200)
    assert color_of(result["image"]) == "blue"


def test_a_screen_that_fits_is_kept_as_it_is():
    with FakeScreen(size=(1440, 900)):
        assert image_size(screen.capture_screen()["image"]) == (1440, 900)


@pytest.mark.parametrize("refused", [{"black": True}, {"empty": True}])
def test_without_screen_recording_a_mac_user_is_told_how_to_allow_it(refused, monkeypatch):
    """macOS gives an app that may not record the screen a black or empty
    capture. The model is told, and so is the user: only they can allow it."""
    monkeypatch.setattr(sys, "platform", "darwin")
    with FakeScreen(**refused):
        result = screen.capture_screen()
    assert result["success"] is False and "image" not in result
    assert "Privacy & Security > Screen Recording" in result["message"]
    assert "Privacy & Security > Screen Recording" in result["notice"]
    assert "reopen Iris" in result["notice"]


def test_elsewhere_a_black_screen_names_no_mac_settings(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with FakeScreen(black=True):
        result = screen.capture_screen()
    assert result["success"] is False and "notice" not in result
    assert "System Settings" not in result["message"]
    with FakeScreen(breaks=RuntimeError("no display")):
        result = screen.capture_screen()
    assert result == {"success": False, "message": "Error capturing screen: no display"}


# Which model answers an image ------------------------------------------------------


def test_a_camera_image_is_answered_by_the_vision_model_without_tools(stub, slept):
    stub.configure(vision=True)
    with FakeCamera(color="red"):
        reply = "".join(_conversation(stub, VISION_MODEL).chat_stream("What am I holding?"))
    assert reply == "Let me look.I see something red."
    asked, answered = stub.requests
    assert asked.model == MODEL and asked.tools == ["look", "capture_screen"] and not asked.images
    # The image goes to the model that sees, which is offered no tools
    assert answered.model == VISION_MODEL and len(answered.images) == 1
    assert answered.tools == []


def test_a_model_that_sees_answers_the_image_itself(stub, slept):
    stub.configure(vision=True, capabilities=("completion", "tools", "vision"))
    with FakeCamera(color="green"):
        reply = "".join(_conversation(stub, VISION_MODEL).chat_stream("What am I holding?"))
    assert reply.endswith("I see something green.")
    asked, answered = stub.requests
    assert answered.model == MODEL and len(answered.images) == 1
    # It keeps its tools: it may look again
    assert answered.tools == ["look", "capture_screen"]


def test_the_screen_goes_to_the_vision_model_too(stub):
    stub.configure(vision=True)
    with FakeScreen(color="yellow"):
        reply = "".join(_conversation(stub, VISION_MODEL).chat_stream("What is on my screen?"))
    assert reply == "I see something yellow."
    answered = stub.requests[-1]
    assert answered.model == VISION_MODEL and image_size(answered.images[0]) == (1920, 1200)


def test_without_a_model_that_sees_the_image_is_left_out_and_the_user_told(stub, slept):
    """No model on the server can see: the text model answers, told that the
    image cannot be shown, and the app says why."""
    conversation = _conversation(stub)
    assert conversation.image_model() is None
    assistant = _assistant(conversation)
    with FakeCamera():
        reply = assistant.chat("What am I holding?")
    assert reply == "Let me look.I cannot see anything."
    answered = stub.requests[-1]
    assert answered.model == MODEL and not answered.images and answered.status is None
    assert "cannot be shown to you" in answered.tool_messages()[-1]
    assert {"message": NO_VISION, "kind": "no_vision"} in _errors(assistant)


def test_the_screen_permission_reaches_the_user_beside_the_reply(stub, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    stub.configure(vision=True)
    assistant = _assistant(_conversation(stub, VISION_MODEL))
    with FakeScreen(black=True):
        reply = assistant.chat("What is on my screen?")
    # The model is told why it sees nothing, and says what it makes of it
    assert reply == "I cannot see anything."
    assert "Screen Recording" in stub.requests[-1].tool_messages()[-1]
    [error] = _errors(assistant)
    assert "Privacy & Security > Screen Recording" in error["message"]


def test_stop_while_the_camera_looks_lets_it_go_and_shows_no_model_the_image(stub, monkeypatch):
    """Stop pressed while the camera warms up: the capture cannot be cut
    short, but the camera is let go and no model is asked about the image."""
    stub.configure(vision=True)
    conversation = _conversation(stub, VISION_MODEL)
    warming, stopped = threading.Event(), threading.Event()

    def warm_up(seconds):
        warming.set()
        stopped.wait(5)

    monkeypatch.setattr(camera, "time", SimpleNamespace(sleep=warm_up))
    said = []
    with FakeCamera() as fake:
        reply = threading.Thread(target=lambda: said.extend(conversation.chat_stream("What am I holding?")))
        reply.start()
        assert warming.wait(5)
        conversation.cancel()
        stopped.set()
        reply.join(5)
    assert not reply.is_alive()
    assert "".join(said) == "Let me look."
    assert fake.captures[0].released
    assert [r.model for r in stub.requests] == [MODEL]


# Images attached to a typed message ------------------------------------------------


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (210, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _attached():
    data = base64.b64encode(_png()).decode("ascii")
    return prepare_all([{"name": "photo.png", "mime": "image/png", "data": data}])


def test_an_attached_image_goes_to_the_vision_model(stub):
    stub.configure(vision=True)
    stub.set_rules([Rule("photo", _seen)])
    conversation = _conversation(stub, VISION_MODEL)
    "".join(conversation.chat_stream("What is in this photo?", attachments=_attached()))
    [request] = stub.requests
    assert request.model == VISION_MODEL and len(request.images) == 1
    assert color_of(request.images[0]) == "red"


def test_an_attached_image_without_a_model_that_sees_is_told(stub):
    stub.set_rules([Rule("photo", _seen)])
    assistant = _assistant(_conversation(stub))
    assistant._answer_typed({"text": "What is in this photo?", "id": "t1", "attachments": _attached()})
    [request] = stub.requests
    assert request.model == MODEL and not request.images and request.status is None
    assert {"message": NO_VISION, "kind": "no_vision"} in _errors(assistant)


def test_the_scripted_server_refuses_images_for_a_model_that_cannot_see(stub):
    """So the tests above would notice an image sent to the wrong model."""
    stub.configure(vision=True)
    conversation = _conversation(stub, VISION_MODEL)
    conversation.vision_model = MODEL  # as if the text model could see
    with pytest.raises(Exception, match=NO_IMAGES):
        list(conversation.chat_stream("What is in this photo?", attachments=_attached()))


# Switching servers -----------------------------------------------------------------


def test_switching_servers_takes_the_new_servers_vision_model(monkeypatch):
    """The model that sees is picked again on every server: one where there
    is none had kept the last server's, which the new one does not have."""
    plain, seeing = StubModel(), StubModel(vision=True)
    try:
        controller = AssistantController(EventBus())
        controller._ollama_host = plain.url
        controller.assistant = SimpleNamespace(
            conversation=ConversationManager(MODEL, host=plain.url),
            config_info=lambda: {},
        )
        conversation = controller.assistant.conversation
        assert controller.set_ollama_host(seeing.url)["ok"]
        assert conversation.vision_model == VISION_MODEL
        assert controller.set_ollama_host(plain.url)["ok"]
        assert conversation.vision_model is None
    finally:
        plain.close()
        seeing.close()

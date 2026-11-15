"""Unit tests for files attached to typed messages: how they are read, what
the model gets, and that only their names are remembered."""

import base64
import io
import threading
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from pypdf import PdfWriter

from iris.assistant import VoiceAssistant
from iris.attachments import AttachmentError, prepare, prepare_all, recorded_text
from iris.controller import AssistantController
from iris.events import EventBus
from iris.llm import ConversationManager
from iris.tts import SpeechGenerator


def _upload(name, mime, data: bytes) -> dict:
    return {"name": name, "mime": mime, "data": base64.b64encode(data).decode("ascii")}


def _png(size=(40, 30), mode="RGBA") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, (200, 30, 30, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_image_becomes_jpeg_for_the_model_with_a_preview():
    attachment = prepare(_upload("photo.png", "image/png", _png()))
    assert attachment.kind == "image"
    assert base64.b64decode(attachment.image)[:3] == b"\xff\xd8\xff"
    assert attachment.preview.startswith("data:image/jpeg;base64,")
    assert attachment.describe() == {
        "name": "photo.png",
        "kind": "image",
        "mime": "image/png",
        "preview": attachment.preview,
    }


def test_large_image_is_scaled_down():
    attachment = prepare(_upload("big.png", "image/png", _png((4000, 1000), "RGB")))
    image = Image.open(io.BytesIO(base64.b64decode(attachment.image)))
    assert max(image.size) == 1568


def test_text_file_is_read_as_a_document():
    attachment = prepare(_upload("notes.md", "", "# Plan\nShip it".encode()))
    assert attachment.kind == "document"
    assert attachment.text == "# Plan\nShip it"
    assert "preview" not in attachment.describe()


def test_pdf_without_text_says_so():
    attachment = prepare(_upload("scan.pdf", "application/pdf", _blank_pdf()))
    assert attachment.kind == "document"
    assert "no text" in attachment.text


@pytest.mark.parametrize(
    "upload, reason",
    [
        (_upload("a.bin", "application/octet-stream", b"\x00\x01\x02"), "cannot read"),
        (_upload("fake.png", "image/png", b"not an image"), "not an image"),
        (_upload("empty.txt", "text/plain", b""), "empty"),
        ({"name": "x.txt", "mime": "text/plain", "data": "%%%"}, "damaged"),
    ],
)
def test_unreadable_files_are_refused(upload, reason):
    with pytest.raises(AttachmentError, match=reason):
        prepare(upload)


def test_too_many_files_are_refused():
    with pytest.raises(AttachmentError):
        prepare_all([_upload(f"{i}.txt", "text/plain", b"x") for i in range(11)])


def test_a_file_over_25_mb_is_refused():
    limit = 25 * 1024 * 1024
    assert prepare(_upload("notes.txt", "text/plain", b"a" * limit)).kind == "document"
    with pytest.raises(AttachmentError, match="notes.txt is larger than 25 MB"):
        prepare(_upload("notes.txt", "text/plain", b"a" * (limit + 1)))


def test_only_names_are_recorded():
    files = prepare_all([_upload("a.txt", "text/plain", b"secret contents")])
    assert recorded_text("Summarize", files) == "Summarize\n\n(Attached: a.txt)"
    assert recorded_text("", files) == "(Attached: a.txt)"
    assert recorded_text("Hi", []) == "Hi"


def test_send_text_refuses_a_bad_file_before_queueing():
    controller = AssistantController(EventBus())
    result = controller.send_text("look", [_upload("a.bin", "", b"\x00\x00")])
    assert result["ok"] is False
    assert controller._turns.empty()


def test_send_text_accepts_files_without_text():
    controller = AssistantController(EventBus())
    result = controller.send_text("", [_upload("a.txt", "text/plain", b"hello")])
    assert result["ok"] is True
    turn = controller._turns.get_nowait()
    assert turn["text"] == "" and turn["attachments"][0].text == "hello"


def _manager(mock_ollama, capabilities, vision_model=None):
    mock_ollama.show.return_value = MagicMock(
        modelinfo={}, details=MagicMock(family="test"), capabilities=capabilities
    )
    mock_ollama.list.return_value = []
    mock_ollama.chat.side_effect = lambda **kwargs: iter([{"message": {"content": "ok"}}])
    return ConversationManager(model="test", vision_model=vision_model)


def _image():
    return prepare_all([_upload("p.png", "image/png", _png())])


def _sent_user_message(mock_ollama) -> dict:
    return mock_ollama.chat.call_args.kwargs["messages"][-1]


@patch("iris.llm.ollama")
def test_document_goes_to_the_model_but_only_its_name_is_kept(mock_ollama):
    cm = _manager(mock_ollama, ["completion"])
    files = prepare_all([_upload("plan.txt", "text/plain", b"step one")])
    list(cm.chat_stream("What is in it?", attachments=files))

    content = _sent_user_message(mock_ollama)["content"]
    assert content.startswith("What is in it?")
    assert '<file name="plan.txt" shared="with this message">\nstep one\n</file>' in content
    assert cm.conversation_history[0]["content"] == "What is in it?\n\n(Attached: plan.txt)"

    # Still at hand for a follow-up, marked as shared earlier
    list(cm.chat_stream("And then?"))
    assert 'shared="earlier"' in _sent_user_message(mock_ollama)["content"]


@patch("iris.llm.ollama")
def test_a_spoken_reply_keeps_only_the_names_too(mock_ollama):
    """The assistant keeps an exchange once it knows how much of the reply
    was heard; the prompt it keeps names the files as well."""
    assistant = object.__new__(VoiceAssistant)
    assistant.conversation = _manager(mock_ollama, ["completion"])
    assistant.status_manager = MagicMock()
    assistant.speech_pipeline = MagicMock()
    assistant.tts_generator = SpeechGenerator.__new__(SpeechGenerator)
    assistant.tts_generator.engine = "kyutai"
    assistant.tts_generator.voice = "anna"
    assistant.event_bus = None
    assistant.memory = None
    assistant.speak_replies = True
    assistant.suspend_event = threading.Event()
    assistant.interrupt_event = threading.Event()
    files = prepare_all([_upload("plan.txt", "text/plain", b"step one")])

    assistant.chat("What is in it?", source="text", attachments=files)

    assert assistant.conversation.conversation_history[0]["content"] == (
        "What is in it?\n\n(Attached: plan.txt)"
    )


@patch("iris.llm.ollama")
def test_shared_files_fade_after_a_few_turns(mock_ollama):
    cm = _manager(mock_ollama, ["completion"])
    list(cm.chat_stream("Read", attachments=prepare_all([_upload("a.txt", "", b"x")])))
    for _ in range(6):
        list(cm.chat_stream("more"))
    assert "a.txt" not in _sent_user_message(mock_ollama)["content"]
    assert cm.shared == []


@patch("iris.llm.ollama")
def test_images_go_to_a_model_that_can_see(mock_ollama):
    cm = _manager(mock_ollama, ["completion", "vision"])
    list(cm.chat_stream("What is this?", attachments=_image()))
    assert mock_ollama.chat.call_args.kwargs["model"] == "test"
    assert len(_sent_user_message(mock_ollama)["images"]) == 1


@patch("iris.llm.ollama")
def test_images_switch_to_the_vision_model_when_needed(mock_ollama):
    cm = _manager(mock_ollama, ["completion"], vision_model="seer")
    list(cm.chat_stream("What is this?", attachments=_image()))
    assert mock_ollama.chat.call_args.kwargs["model"] == "seer"


@patch("iris.llm.ollama")
def test_without_a_vision_model_the_images_are_left_out(mock_ollama):
    cm = _manager(mock_ollama, ["completion"])
    assert cm.image_model() is None
    list(cm.chat_stream("What is this?", attachments=_image()))
    message = _sent_user_message(mock_ollama)
    assert "images" not in message
    assert "cannot be shown to you" in message["content"]


@patch("iris.llm.ollama")
def test_a_model_that_cannot_see_is_told_so_with_the_pictures_only(mock_ollama):
    """The pictures stay among the files shared for a few turns; the next
    messages, which have none, must not have it said again, or the model
    tells the user once more on every reply that it cannot see them."""
    cm = _manager(mock_ollama, ["completion"])
    list(cm.chat_stream("What is this?", attachments=_image()))
    list(cm.chat_stream("And what time is it?"))
    message = _sent_user_message(mock_ollama)
    assert "cannot be shown" not in message["content"]
    assert "images" not in message

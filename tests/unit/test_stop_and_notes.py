"""Stop acts at once, whatever the model is doing, and a system message is
only ever the first one."""

import queue
import threading
import time
from unittest.mock import MagicMock, patch

from iris.assistant import VoiceAssistant
from iris.llm import ConversationManager, system_first


class TestSystemFirst:
    def test_a_note_goes_into_the_user_message_after_it(self):
        messages = [
            {"role": "system", "content": "You are Iris."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "system", "content": "The persona changed to HAL."},
            {"role": "user", "content": "Who are you?"},
        ]
        sent = system_first(messages)
        assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
        assert sent[-1]["content"] == "(Note: The persona changed to HAL.)\n\nWho are you?"
        # The conversation itself is left as it was
        assert messages[3]["role"] == "system"
        assert messages[4]["content"] == "Who are you?"

    def test_several_notes_and_what_the_message_carries(self):
        sent = system_first(
            [
                {"role": "system", "content": "You are Iris."},
                {"role": "system", "content": "Summary: weather talk."},
                {"role": "system", "content": "Respond in French."},
                {"role": "user", "content": "Salut", "images": ["x"]},
            ]
        )
        assert len(sent) == 2
        assert sent[1]["content"].startswith("(Note: Summary: weather talk.)\n\n(Note: Respond")
        assert sent[1]["images"] == ["x"]

    def test_a_note_at_the_end_becomes_a_message_of_its_own(self):
        sent = system_first(
            [
                {"role": "system", "content": "You are Iris."},
                {"role": "user", "content": "Search for it"},
                {"role": "tool", "content": "results"},
                {"role": "system", "content": "No more tools can be used."},
            ]
        )
        assert sent[-1] == {"role": "user", "content": "(Note: No more tools can be used.)"}

    def test_nothing_changes_without_notes(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}]
        assert system_first(messages) == messages


def _conversation():
    with patch("iris.llm.ollama") as mock_ollama:
        mock_ollama.show.return_value = MagicMock(modelinfo={}, details=MagicMock(family="test"))
        mock_ollama.list.return_value = []
        return ConversationManager(model="test")


def test_a_stored_note_reaches_the_model_inside_the_user_message():
    conversation = _conversation()
    conversation.conversation_history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    conversation.add_note("The persona changed to HAL.")
    conversation.client = MagicMock()
    conversation.client.chat.return_value = iter([{"message": {"content": "I am HAL."}}])
    list(conversation.chat_stream("Who are you?"))
    sent = conversation.client.chat.call_args.kwargs["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert "The persona changed to HAL." in sent[-1]["content"]


class TestCancel:
    def test_the_request_is_cut_off_and_a_new_client_is_ready(self):
        conversation = _conversation()
        old = conversation.client = MagicMock()
        with patch("iris.llm.ollama_client") as make:
            conversation.cancel()
        old._client.close.assert_called_once()
        assert conversation.client is make.return_value

    def test_no_tool_runs_once_cancelled(self):
        conversation = _conversation()
        tools = MagicMock()
        tools.has_tools.return_value = True
        tools.guidance.return_value = ""
        tools.get_schemas.return_value = []
        conversation.tools = tools
        call = {"function": {"name": "search", "arguments": {}}}

        def chat(**kwargs):
            yield {"message": {"content": "", "tool_calls": [call]}}
            conversation._cancelled.set()  # Stop, while the model asked for the tool

        conversation.client = MagicMock()
        conversation.client.chat.side_effect = chat
        assert list(conversation.chat_stream("Look it up", keep=False)) == []
        tools.execute.assert_not_called()
        assert conversation.client.chat.call_count == 1


class Pipeline:
    current_language = "en"

    def __init__(self):
        self.stopped = False

    def begin_reply(self):
        pass

    def queue_speech(self, text):
        pass

    def stop(self):
        self.stopped = True

    def wait_for_speech(self):
        pass

    def said(self):
        return ""


class SilentModel:
    """A model that says nothing for a long time, as one that reasons first."""

    reply = None
    model = "test"
    host = None

    def __init__(self):
        self.cancelled = threading.Event()

    def chat_stream(self, prompt, **kwargs):
        self.cancelled.wait(30)
        return
        yield

    def cancel(self):
        self.cancelled.set()

    def clean_markdown(self, text):
        return text

    def keep(self, *args):
        raise AssertionError("nothing was said, so nothing is kept")


def _assistant(conversation):
    assistant = object.__new__(VoiceAssistant)
    assistant.event_bus = MagicMock()
    assistant.speak_replies = True
    assistant.conversation = conversation
    assistant.speech_pipeline = Pipeline()
    assistant.interrupt_event = threading.Event()
    assistant.suspend_event = threading.Event()
    assistant.tools = MagicMock()

    class Status:
        status_lock = threading.Lock()
        status_thinking = False

    assistant.status_manager = Status()
    return assistant


def test_stop_ends_a_reply_the_model_has_not_begun():
    conversation = SilentModel()
    assistant = _assistant(conversation)
    threading.Timer(0.2, assistant.interrupt_event.set).start()
    started = time.monotonic()
    assistant.chat("Explain haiku")
    assert time.monotonic() - started < 2
    assert conversation.cancelled.is_set()
    assert assistant.speech_pipeline.stopped
    assert not assistant.status_manager.status_thinking
    assistant.event_bus.emit.assert_any_call("chat.done", {"text": "", "interrupted": True})


class ForgettableModel(SilentModel):
    """A model thinking over its reply, in a conversation that can be
    forgotten."""

    def __init__(self):
        super().__init__()
        self.forgotten = threading.Event()

    def forget(self):
        self.forgotten.set()


def test_forgetting_everything_stops_the_reply_first_and_waits_for_the_turn():
    """As Stop stops it; everything is forgotten only once the turn is over,
    so nothing of it is kept or shown after."""
    conversation = ForgettableModel()
    assistant = _assistant(conversation)
    assistant.memory = None
    assistant.turn_queue = queue.Queue()
    assistant._turn_lock = threading.Lock()
    assistant.conversation_language = "de"

    def turn():  # as the run loop takes one
        with assistant._turn_lock:
            assistant.chat("Tell me a long story")

    thread = threading.Thread(target=turn)
    thread.start()
    time.sleep(0.2)  # the model is thinking
    assistant.turn_queue.put({"text": "And another one", "id": "t2", "attachments": []})
    started = time.monotonic()
    assistant.forget_everything()

    assert time.monotonic() - started < 2
    thread.join(2)
    assert not thread.is_alive()
    assert conversation.cancelled.is_set() and conversation.forgotten.is_set()
    told = [call.args for call in assistant.event_bus.emit.call_args_list]
    stopped = told.index(("chat.done", {"text": "", "interrupted": True}))
    assert stopped < told.index(("memory.forgotten", {}))
    # The message waiting goes too, and the next one is not stopped
    assert assistant.turn_queue.empty()
    assert not assistant.interrupt_event.is_set()
    assert assistant.conversation_language == "en"


def test_an_error_from_the_model_is_still_shown():
    class Failing(SilentModel):
        def chat_stream(self, prompt, **kwargs):
            raise RuntimeError("no such model")
            yield

    assistant = _assistant(Failing())
    assert assistant.chat("Hi") == "The language model could not answer: no such model"


def test_the_output_stream_is_only_ever_touched_by_one_thread(monkeypatch):
    """PortAudio aborts the process when two threads close the shared stream
    at once, which Stop did: the app's command, the reply's thread and the
    playback thread all stop the sound."""
    from iris import audio

    inside, overlaps = [0], []
    guard = threading.Lock()

    def slow(*args, **kwargs):
        with guard:
            inside[0] += 1
            if inside[0] > 1:
                overlaps.append(inside[0])
        time.sleep(0.01)
        with guard:
            inside[0] -= 1

    monkeypatch.setattr(audio.sd, "stop", slow)
    monkeypatch.setattr(audio.sd, "play", slow)
    monkeypatch.setattr(audio.sd, "get_stream", lambda: MagicMock(latency=0.0, active=False))
    player = audio.AudioPlayer()
    threads = [threading.Thread(target=player.stop) for _ in range(6)]
    threads += [threading.Thread(target=audio.stop_sound) for _ in range(3)]
    threads += [threading.Thread(target=audio.play_sound, args=(None, 24000)) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert overlaps == []

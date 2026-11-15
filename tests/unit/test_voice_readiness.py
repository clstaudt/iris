"""Voices made ready to speak a language before they speak it, where the user
sees it: nothing is downloaded or made while Iris speaks, a reply in a
language its voice is not ready for is shown, and the jobs that make voices
ready tell their progress, a stall, and a failure to retry. Hugging Face and
the models are stand-ins: nothing is downloaded."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils._runtime import is_xet_available

import iris.tts as tts
from iris import cloning, downloads, pocket, readiness
from iris.assistant import VoiceAssistant
from iris.controller import AssistantController
from iris.events import EventBus
from iris.readiness import Readier
from iris.tts import PocketGenerator, SpeechGenerator, voice_ready
from tests.unit.test_voice_cloning import StandInKyutai, _gated, a_voice

pytestmark = pytest.mark.readiness


class Downloads(StandInKyutai):
    """Hugging Face with only some of the open repo downloaded: what the
    test says, and what is fetched. Each fetch may be held, fail, or stall."""

    def __init__(self, open_files=(), **kwargs):
        super().__init__(**kwargs)
        self.files |= set(open_files)
        self.calls: list[list] = []
        self.failures = 0
        self.hold: threading.Event | None = None

    def cached(self, file):
        return file in self.files

    def fetch(self, files, on_progress):
        self.calls.append(list(files))
        if any(file.repo == cloning.CLONING_REPO for file in files) and not self.terms:
            raise _gated()
        on_progress(50, 200)
        if self.hold is not None:
            self.hold.wait(5)
        if self.failures:
            self.failures -= 1
            raise ConnectionError("The connection was reset")
        on_progress(200, 200)
        self.files.update(files)


def english_here():
    """The English model and Eve's voice for it, as the first run leaves them."""
    return [*pocket.model_files("english"), pocket.voice_file("english", "eve")]


@pytest.fixture
def hub(monkeypatch):
    stand_in = Downloads(open_files=english_here())
    monkeypatch.setattr(cloning, "KYUTAI", stand_in)
    return stand_in


class Events:
    def __init__(self):
        self.seen: list[tuple[str, dict]] = []

    def __call__(self, event_type, data):
        self.seen.append((event_type, dict(data)))

    def jobs(self, **match):
        return [
            d for t, d in self.seen if t == "voice.job" and all(d[k] == v for k, v in match.items())
        ]


def until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# Speaking fetches nothing
# ---------------------------------------------------------------------------


def test_pocket_tts_takes_its_files_from_the_cache_only():
    """Pocket TTS would download a language's model when first asked to
    speak it, the gated form first: everything of it it loads comes from the
    cache, or not at all."""
    with patch.object(tts, "hf_hub_download", return_value="/cache/model") as fetch:
        tts.pocket_model_module.download_if_necessary(
            "hf://kyutai/pocket-tts/a/model.safetensors@1"
        )
        tts.pocket_text_module.download_if_necessary("hf://kyutai/open/b/tokenizer.json@2")
    assert all(call.kwargs["local_files_only"] for call in fetch.call_args_list)
    with pytest.raises(FileNotFoundError):
        tts.pocket_model_module.download_if_necessary("https://example.com/voice.wav")


def test_a_voice_is_ready_for_a_language_once_its_model_and_voice_are_here(hub):
    assert voice_ready("kyutai", "eve", "en")
    assert not voice_ready("kyutai", "eve", "de")
    assert not voice_ready("kyutai", "paul", "en")
    # The language's model in either form, and the voice for it
    hub.files.update(
        [*pocket.model_files("german", clones=True), pocket.voice_file("german", "eve")]
    )
    assert voice_ready("kyutai", "eve", "de")
    # A language the voice does not speak properly is never ready
    assert not voice_ready("kyutai", "eve", "ja")


def test_a_cloned_voice_is_ready_for_a_language_once_it_is_made_for_it(hub):
    folder = a_voice()
    assert not voice_ready("kyutai", "clone:ada", "en")
    (folder / "english.safetensors").write_bytes(b"made")
    assert voice_ready("kyutai", "clone:ada", "en")
    hub.files.update(pocket.model_files("german"))
    assert not voice_ready("kyutai", "clone:ada", "de")


def test_hugging_face_downloads_go_over_http_which_resumes():
    """A file interrupted resumes from its .incomplete part over HTTP, and a
    silent connection gives up; Xet would start the file over and wait."""
    assert downloads.hf_constants.HF_HUB_DISABLE_XET
    assert not is_xet_available()


# ---------------------------------------------------------------------------
# A reply in a language its voice is not ready for
# ---------------------------------------------------------------------------


def _speaker(ready: bool):
    assistant = object.__new__(VoiceAssistant)
    assistant.speak_replies = True
    assistant.conversation = SimpleNamespace(clean_markdown=lambda text: text)
    assistant.speech_pipeline = MagicMock()
    assistant.tts_generator = SimpleNamespace(
        engine="kyutai", voice="eve", speaks=lambda language: True, ready=lambda language: ready
    )
    assistant._speech_language = lambda text: "de"
    assistant._reply_voiced = assistant._reply_unspoken = False
    assistant._unspoken_told = set()
    assistant._report_timing = lambda: None
    assistant.persona = SimpleNamespace(name="Iris")
    assistant.event_bus = MagicMock()
    return assistant


def test_a_reply_its_voice_is_not_ready_for_is_shown_and_its_voice_made_ready():
    assistant = _speaker(ready=False)
    asked = []
    assistant.on_voice_not_ready = lambda *voice: asked.append(voice)

    assistant._speak("Guten Tag.")

    assistant.speech_pipeline.queue_speech.assert_not_called()
    assert asked == [("kyutai", "eve", "de")]
    assert assistant._reply_unspoken


def test_without_the_app_the_user_is_told_once():
    assistant = _speaker(ready=False)
    assistant._speak("Guten Tag.")
    assistant._speak("Wie geht's?")
    [notice] = [c for c in assistant.event_bus.emit.call_args_list if c.args[0] == "notice"]
    assert notice.args[1]["kind"] == "not_ready"
    assert "Iris's voice is not ready for German yet" in notice.args[1]["message"]


def test_once_the_voice_is_ready_the_language_is_spoken():
    assistant = _speaker(ready=True)
    assistant._speak("Guten Tag.")
    assistant.speech_pipeline.queue_speech.assert_called_once_with("Guten Tag.", language="de")


def test_a_persona_whose_cloned_voice_is_not_here_is_told_to_choose_another():
    """The conversation said the voice for English could not be loaded and
    the cloned voice was gone. It says once which persona's voice it is,
    and where to choose another; nothing is made ready for it."""
    assistant = _speaker(ready=False)
    assistant.tts_generator.voice = "clone:noam-2"
    assistant.persona = SimpleNamespace(name="Noam")
    asked = []
    assistant.on_voice_not_ready = lambda *voice: asked.append(voice)

    assistant._speak("Guten Tag.")
    assistant._speak("Wie geht's?")

    assistant.speech_pipeline.queue_speech.assert_not_called()
    assert asked == [] and assistant._reply_unspoken
    [notice] = [c.args[1] for c in assistant.event_bus.emit.call_args_list if c.args[0] == "notice"]
    assert notice == {
        "message": "Noam's voice, Noam 2, cannot be found. "
        "Choose another voice for Noam in the persona editor.",
        "kind": "voice_missing",
    }


# ---------------------------------------------------------------------------
# The jobs that make voices ready
# ---------------------------------------------------------------------------


def _readier(events, **options):
    return Readier(events, threading.RLock(), **options)


def test_a_voice_is_made_ready_with_its_progress_told(hub):
    events = Events()
    jobs = _readier(events)

    job = jobs.request("kyutai", "eve", "de", "Eve", ["Iris"], readiness.LANGUAGES)

    assert job["title"] == "Downloading the German voice model"
    assert job["purpose"] == "Iris will speak German once it's done"
    assert jobs.wait(job["id"], 5)
    assert voice_ready("kyutai", "eve", "de")
    states = [d["state"] for d in events.jobs(id=job["id"])]
    assert states[0] == "queued" and "downloading" in states and states[-1] == "ready"
    assert {"done": 50, "total": 200} in [
        {"done": d["done"], "total": d["total"]} for d in events.jobs()
    ]
    # Ready, it needs no other
    assert jobs.request("kyutai", "eve", "de", "Eve") is None


def test_a_voice_asked_for_again_is_the_same_job_and_the_conversations(hub):
    hub.hold = threading.Event()
    jobs = _readier(Events())
    first = jobs.request("kyutai", "eve", "de", "Eve", ["Iris"], readiness.LANGUAGES)
    again = jobs.request("kyutai", "eve", "de", "Eve", ["Iris"], readiness.CONVERSATION)
    hub.hold.set()
    assert first["id"] == again["id"] and again["origin"] == "conversation"
    assert jobs.wait(first["id"], 5)
    assert len(hub.calls) == 1


def test_a_stalled_download_is_said_and_a_failed_one_is_tried_again(hub):
    hub.hold, hub.failures = threading.Event(), 1
    events = Events()
    jobs = _readier(events, stall_after=0.2)
    job = jobs.request("kyutai", "eve", "de", "Eve", ["Iris"])

    # Nothing came for a while: said, while it still runs
    assert until(lambda: events.jobs(id=job["id"], stalled=True), 5)
    hub.hold.set()
    assert not jobs.wait(job["id"], 5)
    [failed] = [j for j in jobs.jobs() if j["id"] == job["id"]]
    assert failed["state"] == "failed" and failed["error"] == "No connection"
    assert failed["title"] == "Eve's voice for German could not be downloaded"

    retried = jobs.retry(job["id"])
    assert jobs.wait(retried["id"], 5)
    assert voice_ready("kyutai", "eve", "de")


def test_a_cloned_voice_is_downloaded_for_then_made(hub):
    folder = a_voice()
    (folder / "english.safetensors").write_bytes(b"made")
    hub.files.update(pocket.model_files("german"))
    events = Events()
    jobs = _readier(events)

    job = jobs.request("kyutai", "clone:ada", "de", "Ada", ["Ada"], readiness.CONVERSATION)
    assert jobs.wait(job["id"], 5)

    titles = [d["title"] for d in events.jobs(id=job["id"])]
    assert "Downloading Kyutai's cloning model for German" in titles
    assert "Making Ada's voice for German" in titles
    assert titles[-1] == "Ada's voice is ready for German"
    assert (folder / "german.safetensors").is_file()
    assert voice_ready("kyutai", "clone:ada", "de")


def test_a_cloned_voice_that_is_not_here_is_not_made_ready(hub):
    events = Events()
    jobs = _readier(events)
    assert jobs.request("kyutai", "clone:noam-2", "de", "Noam 2", ["Noam"]) is None
    assert jobs.jobs() == [] and events.jobs() == []


def test_without_access_a_cloned_voice_fails_and_says_where_to_connect(hub):
    hub.terms = False
    folder = a_voice()
    (folder / "english.safetensors").write_bytes(b"made")
    jobs = _readier(Events())
    job = jobs.request("kyutai", "clone:ada", "de", "Ada", ["Ada"])
    assert not jobs.wait(job["id"], 5)
    [failed] = jobs.jobs()
    assert "connect under Settings, Voices" in failed["error"]


# ---------------------------------------------------------------------------
# Iris's languages, and the conversation
# ---------------------------------------------------------------------------


def test_choosing_a_language_makes_every_personas_voice_ready_for_it(hub, monkeypatch):
    monkeypatch.setattr(tts.KokoroGenerator, "ready", classmethod(lambda cls, *a, **k: True))
    controller = AssistantController(EventBus())

    result = controller.set_languages(["de", "xx"])

    assert result["languages"] == ["en", "de"]
    [job] = result["jobs"]
    assert (job["voice"], job["language"], job["personas"]) == ("eve", "de", ["Iris"])
    assert controller.readier.wait(job["id"], 5)


def test_the_languages_chosen_are_the_users_for_whisper(hub, monkeypatch):
    """Whisper leaves the conversation's language for one of them more
    readily than for another (iris.transcription.language_heard)."""
    monkeypatch.setattr(tts.KokoroGenerator, "ready", classmethod(lambda cls, *a, **k: True))
    controller = AssistantController(EventBus())
    controller.assistant = SimpleNamespace(tts_generator=SimpleNamespace(models_dir=None))

    result = controller.set_languages(["it", "de"])

    assert controller.assistant.chosen_languages == {"en", "it", "de"}
    for job in result["jobs"]:
        controller.readier.wait(job["id"], 5)


def test_a_conversation_makes_the_voice_ready_but_leaves_a_failure_to_the_user(hub):
    hub.failures = 1
    controller = AssistantController(EventBus())
    controller.assistant = SimpleNamespace(
        persona=SimpleNamespace(name="Iris"), tts_generator=SimpleNamespace(models_dir=None)
    )

    controller._voice_not_ready("kyutai", "eve", "de")
    job = controller.readier.find("kyutai", "eve", "de")
    assert job.origin == "conversation" and job.personas == ["Iris"]
    assert not controller.readier.wait(job.id, 5)

    # The next reply in German does not try again unasked
    controller._voice_not_ready("kyutai", "eve", "de")
    assert controller.readier.find("kyutai", "eve", "de").state == "failed"
    assert len(hub.calls) == 1


def test_a_voice_played_that_is_not_here_yet_plays_once_it_is(hub):
    controller = AssistantController(EventBus())
    controller.assistant = MagicMock()

    result = controller.persona_preview({"name": "Paul", "engine": "kyutai", "voice": "paul"})

    assert result["ok"] and result["job"]["purpose"] == "Paul plays once it's done"
    assert until(lambda: controller.assistant.speech_pipeline.queue_speech.called)
    line = controller.assistant.speech_pipeline.queue_speech.call_args.args[0]
    assert line.startswith("Hi, I'm Paul.")


def test_a_generator_never_prepares_a_language_it_is_not_ready_for(hub):
    generator = SpeechGenerator("kyutai", "eve", prepare=False)
    with patch.object(PocketGenerator, "generate") as speak:
        generator.prepare_language("de")
    speak.assert_not_called()
    assert not hub.calls

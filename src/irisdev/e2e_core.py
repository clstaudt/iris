"""The core as the app's end-to-end test (ui/e2e) runs it: iris.rpc_server,
with a virtual room for its audio in place of the computer's microphone and
speaker, and a stand-in for Kyutai's gated cloning model.

The room holds nothing but Iris's own voice, which comes back into its
microphone as from a laptop's speakers. What goes on in the real room (a video
playing, people talking) never becomes a turn, the test still hears the voice
come back as the app would, and nothing sounds on the speakers.

Cloning a voice needs a Hugging Face account that accepted Kyutai's terms,
which a test has not: the stand-in logs in with the test's token, has the
terms accepted when the test says so (in a file of the test's home), counts
its download without fetching, and "clones" every sample as one of Pocket
TTS's own voices, which the open model reads. All the rest of cloning is
Iris's own: the sample, the voice's folder, the persona that speaks with it.

The test can break the room's speaker (in a file of the test's home): while
it says so, opening the speaker fails as it does on macOS when its device has
gone (AirPods disconnecting), so the test sees what Iris does then.

The test looks into the core through methods of its own:

- e2e.speaker {since, until}: how much sound the virtual speaker played
  between two moments, so the test knows a voice was heard, not only that
  the core reported its level;
- e2e.model: when the model's streamed replies came in, its reasoning
  included, which the app is never shown: a reasoning model may think for
  minutes before it writes a word;
- e2e.file_memory: files the conversation into topics now, as Iris does once
  it has been quiet for a while, so the test need not wait for that;
- e2e.memory: what the core remembers (the turns and topics stored, the words
  in their search index) and what the next prompt would carry of the
  conversation.

The moments of e2e.speaker and e2e.model are the test's: milliseconds since
the epoch.

How well the model answers does not matter to the test, so every chat
request asks it to answer without reasoning first: an OpenAI-compatible
server through its chat template (enable_thinking off), Ollama with think
off. Iris itself asks for no such thing.

    python -m irisdev.e2e_core
"""

import json
import os
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path

import httpx
import sounddevice as sd
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from pocket_tts import TTSModel

from iris import audio, cloning, rpc_server
from iris.controller import AssistantController
from iris.memory import DEFAULT_PATH
from irisdev.voicetest.acoustics import Space

# The token the test logs in with, the file where it says what the account
# did, and the voice every sample becomes
TOKEN = "hf_e2e_test_token"
STATE = Path.home() / ".e2e-huggingface.json"
CLONED_AS = "anna"
# Where the test says which of the room's devices it broke
FAULTS = Path.home() / ".e2e-audio.json"
# Requests whose reply the model streams: OpenAI-compatible servers', Ollama's
CHAT_PATHS = ("/chat/completions", "/api/chat")
# Streamed replies kept, the latest
STREAMS_KEPT = 50

# The room, and Iris's laptop in it
SPACE = Space()
LAPTOP = SPACE.endpoint("Iris")


def _refused(status: int, message: str, gated: bool = False):
    response = httpx.Response(status, request=httpx.Request("GET", cloning.MODEL_PAGE))
    return (GatedRepoError if gated else HfHubHTTPError)(message, response=response)


class StandInCloner:
    """The open model, as if it cloned: a sample becomes CLONED_AS."""

    has_voice_cloning = True

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def get_state_for_audio_prompt(self, prompt, truncate=False):
        if isinstance(prompt, Path) and prompt.suffix == ".wav":
            prompt = CLONED_AS
        return self._model.get_state_for_audio_prompt(prompt)


class StandInKyutai(cloning.Kyutai):
    """Kyutai's gated repo, and an account the test controls.

    The open models are this computer's, but for the languages the test
    hides (in the same file, "hidden": ["italian"]): those are not
    downloaded until a download Iris shows fetches them, slowly, so the test
    sees its progress while a reply waits on it."""

    # How long a hidden language's model takes to download, in seconds
    slow_download = 12.0

    def __init__(self, state: Path = STATE):
        self.state = state

    def _read(self) -> dict:
        try:
            return json.loads(self.state.read_text())
        except (OSError, ValueError):
            return {}

    def _write(self, **changes):
        self.state.write_text(json.dumps({**self._read(), **changes}))

    def token(self):
        return self._read().get("token")

    def _hidden(self, file) -> bool:
        """A file of a language the test hides, not downloaded yet."""
        hidden = self._read().get("hidden", [])
        return any(f"languages/{lang}/" in file.filename for lang in hidden) and (
            file.filename not in self._read().get("fetched_open", [])
        )

    def whoami(self):
        if self.token() != TOKEN:
            raise _refused(401, "Invalid user token.")
        return {"name": "e2e"}

    def check_access(self):
        if not self._read().get("terms"):
            raise _refused(403, "Access to model kyutai/pocket-tts is restricted", gated=True)

    def login(self, token):
        if token != TOKEN:
            raise _refused(401, "Invalid user token.")
        self._write(token=token)

    def cached(self, file):
        if file.repo != cloning.CLONING_REPO:
            return not self._hidden(file) and super().cached(file)
        return file.filename in self._read().get("fetched", [])

    def fetch(self, files, on_progress):
        gated = [file for file in files if file.repo == cloning.CLONING_REPO]
        hidden = [file for file in files if file not in gated and self._hidden(file)]
        others = [file for file in files if file not in gated and file not in hidden]
        if gated:
            self.check_access()
        total = 219_000_000
        steps = 40 if hidden else 10
        pause = self.slow_download / steps if hidden else 0.1
        if gated or hidden:
            for step in range(1, steps + 1):
                time.sleep(pause)
                on_progress(total * step // steps, total)
        state = self._read()
        self._write(
            fetched=[*state.get("fetched", []), *(file.filename for file in gated)],
            fetched_open=[*state.get("fetched_open", []), *(file.filename for file in hidden)],
        )
        # What this computer does not have of the hidden ones comes for real
        others += [file for file in hidden if not super().cached(file)]
        if others:
            super().fetch(others, on_progress)

    def load_model(self, pocket_lang):
        model = TTSModel.load_model(language=pocket_lang)
        if self.cached(cloning.cloning_weights(pocket_lang)):
            return StandInCloner(model)
        return model


class Breakable:
    """The room's microphone and speaker, as one audio device, with a
    speaker the test can break: {"speaker": "broken"} in the faults file."""

    def __init__(self, endpoint, faults: Path = FAULTS):
        self._endpoint = endpoint
        self._faults = faults

    def __getattr__(self, name):
        return getattr(self._endpoint, name)

    def _broken(self, part: str) -> bool:
        try:
            return json.loads(self._faults.read_text()).get(part) == "broken"
        except (OSError, ValueError):
            return False

    def play(self, data, sample_rate, **options) -> float:
        if self._broken("speaker"):
            raise sd.PortAudioError("Error opening OutputStream: Internal PortAudio error", -9986)
        return self._endpoint.play(data, sample_rate, **options)


def _monotonic(moment_ms: float) -> float:
    """A moment of the test's clock on time.monotonic()'s."""
    return time.monotonic() - (time.time() - moment_ms / 1000)


def speaker(params: dict) -> dict:
    """How much sound the laptop's speaker played between two moments
    (since, and until or now): the seconds it was audible, and the clips it
    was given."""
    start = _monotonic(float(params.get("since") or 0))
    end = _monotonic(float(params["until"])) if params.get("until") else time.monotonic()
    played = LAPTOP.speaker
    return {
        "seconds": round(played.sound_heard(start, end), 3),
        "clips": len(played.clips_between(start, end)),
    }


class ModelWatch:
    """When the lines of each reply the model streams come in, whatever they
    hold: words, reasoning or tool calls."""

    def __init__(self):
        self._streams: deque = deque(maxlen=STREAMS_KEPT)
        self._lock = threading.Lock()

    def watch(self, response: httpx.Response, lines):
        """The lines of a response, noted as they pass when it is a reply."""
        try:
            chat = response.request.url.path.endswith(CHAT_PATHS)
        except RuntimeError:  # a response without its request
            chat = False
        stream = None
        for line in lines:
            if chat and line.strip():
                now = time.time() * 1000
                with self._lock:
                    if stream is None:
                        stream = {"first": now, "last": now, "lines": 0}
                        self._streams.append(stream)
                    stream["last"] = now
                    stream["lines"] += 1
            yield line

    def report(self, params: dict) -> dict:
        """The replies streamed lately: when their first and last lines came
        (milliseconds since the epoch), and how many lines they had."""
        with self._lock:
            return {"streams": [dict(stream) for stream in self._streams]}


MODEL = ModelWatch()
_iter_lines = httpx.Response.iter_lines


def _watched_lines(self, *args, **kwargs):
    return MODEL.watch(self, _iter_lines(self, *args, **kwargs))


def without_reasoning(path: str, body: dict) -> dict:
    """A chat request's body, asking the model to answer at once: for an
    OpenAI-compatible server (/chat/completions) through its chat template,
    for Ollama (/api/chat) with think off. Any other request is as it was."""
    if path.endswith("/chat/completions"):
        template = {**(body.get("chat_template_kwargs") or {}), "enable_thinking": False}
        return {**body, "chat_template_kwargs": template}
    if path.endswith("/api/chat"):
        return {**body, "think": False}
    return body


_build_request = httpx.Client.build_request


def _built_without_reasoning(self, method, url, *args, **kwargs):
    if isinstance(kwargs.get("json"), dict):
        kwargs["json"] = without_reasoning(httpx.URL(url).path, kwargs["json"])
    return _build_request(self, method, url, *args, **kwargs)


class Controller(AssistantController):
    """Iris's own controller, kept where the test's methods find it."""

    current: "Controller | None" = None

    def __init__(self, bus):
        super().__init__(bus)
        Controller.current = self


def _assistant():
    assistant = Controller.current.assistant if Controller.current else None
    if assistant is None or assistant.memory is None:
        raise RuntimeError("Iris is not running with memory")
    return assistant


def file_memory(params: dict) -> dict:
    """File what was said into topics, on the conversation's model, as
    Iris does after a quiet spell, and say which topics there are then. The
    test asks while the conversation rests, so filing does not wait for it."""
    memory = _assistant().memory
    with memory._filing:
        changed = memory.filer.file()
    if changed and memory.on_change:
        memory.on_change()
    return {"topics": [topic.title for topic in memory.store.topics()]}


def _words_indexed() -> int:
    """How many words the search index of the memory file holds, read as
    another program would read the file."""
    db = sqlite3.connect(os.environ.get("IRIS_MEMORY_DB") or DEFAULT_PATH)
    try:
        db.execute("CREATE VIRTUAL TABLE temp.words USING fts5vocab(main, turns_fts, row)")
        return db.execute("SELECT COUNT(*) FROM temp.words").fetchone()[0]
    finally:
        db.close()


def remembered(params: dict) -> dict:
    """What the core remembers, and what the next prompt would carry: the
    messages of the conversation after the system prompt, and the whole
    prompt as text (the system prompt holds the topics and those recalled,
    the files shared go with the next message)."""
    assistant = _assistant()
    store = assistant.memory.store
    conversation = assistant.conversation
    messages = conversation.manage_context()
    prompt = "\n\n".join(m["content"] for m in messages) + conversation._files_context([])
    return {
        "turns": store.turn_count(),
        "topics": len(store.topics()),
        "words": _words_indexed(),
        "context": messages[1:],
        "prompt": prompt,
    }


def main():
    # The microphone and the speaker are taken from here when the assistant
    # is built, after the app has asked for it
    audio.SYSTEM = Breakable(LAPTOP)
    cloning.KYUTAI = StandInKyutai()
    # Both Ollama's client and Iris's own for other servers read a streamed
    # reply line by line, and build each request through httpx
    httpx.Response.iter_lines = _watched_lines
    httpx.Client.build_request = _built_without_reasoning
    rpc_server.AssistantController = Controller
    rpc_server.main(
        {
            "e2e.speaker": speaker,
            "e2e.model": MODEL.report,
            "e2e.file_memory": file_memory,
            "e2e.memory": remembered,
        }
    )


if __name__ == "__main__":
    main()

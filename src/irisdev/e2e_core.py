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

    python -m irisdev.e2e_core
"""

import json
import time
from pathlib import Path

import httpx
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from pocket_tts import TTSModel

from iris import audio, cloning, rpc_server
from irisdev.voicetest.acoustics import Space

# The token the test logs in with, the file where it says what the account
# did, and the voice every sample becomes
TOKEN = "hf_e2e_test_token"
STATE = Path.home() / ".e2e-huggingface.json"
CLONED_AS = "anna"


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
    """Kyutai's gated repo, and an account the test controls."""

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
        return file.filename in self._read().get("fetched", [])

    def fetch(self, file, on_progress):
        self.check_access()
        total = 219_000_000
        for step in range(1, 11):
            time.sleep(0.1)
            on_progress(total * step // 10, total)
        self._write(fetched=[*self._read().get("fetched", []), file.filename])

    def load_model(self, pocket_lang):
        model = TTSModel.load_model(language=pocket_lang)
        if self.cached(cloning.cloning_weights(pocket_lang)):
            return StandInCloner(model)
        return model


def main():
    # The microphone and the speaker are taken from here when the assistant
    # is built, after the app has asked for it
    audio.SYSTEM = Space().endpoint("Iris")
    cloning.KYUTAI = StandInKyutai()
    rpc_server.main()


if __name__ == "__main__":
    main()

"""The unit tests call no model and download none, on any computer.

Starting Iris fetches missing speech models: through iris.downloads (the
Kokoro voices come from GitHub, beyond Hugging Face's offline switch) and
through Hugging Face (Whisper, Smart Turn). Here every download fails unless
a test puts its own in place, and Hugging Face is offline, so a test that
comes to need a model fails the same way on a fresh computer as in CI,
instead of passing only where the model happens to be cached. A test of the
download code itself, with the network replaced by stand-ins, is marked
`downloads`.

Whether a voice is ready to speak a language depends on what this computer
has downloaded, so every voice counts as ready here, unless a test of
readiness itself, marked `readiness`, has its own stand-ins say what is.

The keys model servers want are kept in each test's own folder, never in the
user's.
"""

import huggingface_hub.constants as hf_constants
import pytest

from iris import downloads, llm_client
from iris.tts import KokoroGenerator, PocketGenerator


def _refused(*args, **kwargs):
    raise RuntimeError("The unit tests download nothing; put a stand-in in place in the test")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "downloads: tests the download code itself, with the network replaced by stand-ins"
    )
    config.addinivalue_line(
        "markers", "readiness: tests whether voices are ready, with stand-ins for the downloads"
    )


@pytest.fixture(autouse=True)
def no_downloads(request, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    if request.node.get_closest_marker("downloads"):
        return
    monkeypatch.setattr(downloads, "download", _refused)
    monkeypatch.setattr(downloads, "download_files", _refused)


@pytest.fixture(autouse=True)
def voices_ready(request, monkeypatch):
    if request.node.get_closest_marker("readiness"):
        return
    for engine in (PocketGenerator, KokoroGenerator):
        monkeypatch.setattr(engine, "ready", classmethod(lambda cls, *args, **kwargs: True))


@pytest.fixture(autouse=True)
def server_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_client, "KEYS_PATH", tmp_path / ".iris" / "server-keys.json")

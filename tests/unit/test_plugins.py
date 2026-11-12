"""Unit tests for finding plugins: shipped, installed and in the user's
folder, and the engines they register."""

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from iris import engines, plugins
from iris.engines import RecognitionEngine, SpeechEngine
from iris.plugin import Plugin, PluginContext
from iris.tools import Tools
from iris.tts import SpeechGenerator, engine_speaks, list_voices

CLOCK = '''
    import time

    from iris.plugin import Plugin, Tool


    class Clock(Plugin):
        id = "clock"
        name = "Clock"

        def tools(self, context):
            return [
                Tool(
                    name="current_time",
                    description="The current local time.",
                    run=lambda **_: {"success": True, "result": "12:34"},
                    label="Looked at the clock",
                )
            ]


    plugin = Clock()
'''


@pytest.fixture(autouse=True)
def registries():
    """Engines a test registers are gone after it."""
    speech = dict(engines._speech), dict(engines._speech_sources)
    recognition = dict(engines._recognition)
    yield
    engines._speech.clear()
    engines._speech.update(speech[0])
    engines._speech_sources.clear()
    engines._speech_sources.update(speech[1])
    engines._recognition.clear()
    engines._recognition.update(recognition)


def _folder(tmp_path, **modules):
    for name, source in modules.items():
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(source))
    return tmp_path


def _load(folder, monkeypatch, entries=()):
    monkeypatch.setattr(plugins, "entry_points", lambda group: list(entries))
    return plugins.load(folder)


def test_the_shipped_plugins_come_first(tmp_path, monkeypatch):
    loaded = _load(tmp_path, monkeypatch)
    assert [p.id for p in loaded.plugins] == ["memory", "camera", "screen", "web"]
    assert loaded.failures == []


def test_a_plugin_in_the_folder_gives_the_model_a_tool(tmp_path, monkeypatch):
    loaded = _load(_folder(tmp_path, clock=CLOCK), monkeypatch)
    clock = loaded.plugins[-1]
    assert clock.id == "clock" and clock.source == str(tmp_path / "clock.py")

    tools = Tools([clock], PluginContext())
    assert tools.states()["clock"] == {"enabled": True, "available": True, "name": "Clock"}
    assert tools.execute("current_time") == {"success": True, "result": "12:34"}
    assert tools.describe("current_time", {}) == ("Looked at the clock", "")


def test_broken_plugins_are_reported_and_skipped(tmp_path, monkeypatch):
    folder = _folder(
        tmp_path,
        broken="raise RuntimeError('boom')",
        empty="x = 1",
        twin=CLOCK.replace('"Clock"', '"Clock twin"'),
        clock=CLOCK,
        _private="raise RuntimeError('never run')",
    )
    loaded = _load(folder, monkeypatch)
    assert [p.id for p in loaded.plugins][-1] == "clock"
    errors = {f["source"].rsplit("/", 1)[-1]: f["error"] for f in loaded.failures}
    assert errors["broken.py"] == "boom"
    assert "no 'plugin'" in errors["empty.py"]
    assert "already loaded" in errors["twin.py"]
    assert "_private.py" not in errors


def test_a_shipped_id_cannot_be_taken(tmp_path, monkeypatch):
    loaded = _load(_folder(tmp_path, cam=CLOCK.replace('id = "clock"', 'id = "camera"')), monkeypatch)
    assert "already loaded" in loaded.failures[0]["error"]


class Whisperer(SpeechEngine):
    id = "whisperer"
    name = "Whisperer"
    languages = frozenset({"en", "de"})

    def __init__(self, models_dir=None):
        self.spoken = []

    @classmethod
    def voices(cls, models_dir=None):
        return [{"id": "soft", "name": "Soft", "language": "English"}]

    def generate(self, text, language="en"):
        self.spoken.append((text, language, self.voice))
        return "/tmp/whisper.wav"


class Voices(Plugin):
    id = "voices"
    name = "Voices"

    def speech_engines(self):
        return [Whisperer]

    def recognition_engines(self):
        return [RecognitionEngine("ears", "Ears", lambda options: SimpleNamespace(**options))]


def test_an_installed_plugin_registers_its_engines(tmp_path, monkeypatch):
    entry = SimpleNamespace(name="voices", dist=SimpleNamespace(name="iris-voices"), load=Voices)
    loaded = _load(tmp_path, monkeypatch, [entry])
    assert loaded.plugins[-1].source == "package iris-voices"

    assert "whisperer" in engines.speech_engines()
    assert engine_speaks("whisperer", "de") and not engine_speaks("whisperer", "fr")
    assert list_voices("whisperer") == [{"id": "soft", "name": "Soft", "language": "English"}]
    generator = SpeechGenerator("whisperer", "soft")
    assert generator.generate("Hallo", language="de") == "/tmp/whisper.wav"
    assert generator._engine("whisperer").spoken == [("Hallo", "de", "soft")]

    ears = engines.recognition_engine("ears")
    assert ears.source == "voices" and ears.create({"model": "x"}).model == "x"
    described = engines.describe(speech="whisperer", recognition="ears")
    assert {"id": "whisperer", "label": "Whisperer", "active": True}.items() <= next(
        e for e in described["tts"] if e["id"] == "whisperer"
    ).items()
    assert next(e for e in described["stt"] if e["id"] == "ears")["source"] == "voices"


def test_an_engine_id_that_is_taken_fails_the_plugin_whole(tmp_path, monkeypatch):
    class Clash(Voices):
        id = "clash"

        def speech_engines(self):
            return [Whisperer, type("Kyutai2", (Whisperer,), {"id": "kyutai"})]

    entry = SimpleNamespace(name="clash", dist=None, load=Clash)
    loaded = _load(tmp_path, monkeypatch, [entry])
    assert "already registered" in loaded.failures[0]["error"]
    # Nothing of it was registered
    assert "whisperer" not in engines.speech_engines()


def test_the_built_in_engines_are_registered():
    assert list(engines.speech_engines())[:2] == ["kyutai", "kokoro"]
    assert "whisper" in engines.recognition_engines()
    # Kyutai borrows Kokoro's voice for Japanese, Kokoro Kyutai's for German
    assert engines.speaking_engine("kyutai", "ja") == ("kokoro", "jf_alpha")
    assert engines.speaking_engine("kokoro", "de") == ("kyutai", "juergen")
    assert engines.speaking_engine("kokoro", "ru") is None


def test_the_example_plugin_loads_from_its_folder(monkeypatch):
    folder = Path(__file__).parents[2] / "examples" / "plugins"
    loaded = _load(folder, monkeypatch)
    clock = next(p for p in loaded.plugins if p.id == "clock")
    tools = Tools([clock])
    assert tools.execute("current_time")["success"] is True
    assert "'current_time' function" in tools.guidance()

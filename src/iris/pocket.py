"""The files Kyutai's Pocket TTS loads for each of its languages, as Iris
fetches them.

Pocket TTS would fetch a language's model itself the first time it is
asked to speak it, unseen, and the gated form that clones first when the
account may have it. Iris loads its files from the Hugging Face cache only
(iris.tts) and downloads them beforehand, with progress (iris.readiness):
nothing is fetched while Iris speaks.
"""

from functools import lru_cache

import yaml
from pocket_tts.utils.config import CONFIGS_DIR
from pocket_tts.utils.utils import get_predefined_voice

from iris.downloads import HubFile

# Pocket TTS's languages: the Whisper code, and its name for the language's model
LANGUAGES = {
    "en": "english",
    "fr": "french",
    "it": "italian",
    "de": "german",
    "es": "spanish",
    "pt": "portuguese",
    "nl": "dutch",
}


@lru_cache(maxsize=None)
def config(pocket_lang: str) -> dict:
    """A language's config, as Pocket TTS ships it (read once; not to be
    changed)."""
    return yaml.safe_load((CONFIGS_DIR / f"{pocket_lang}.yaml").read_text())


def hub_paths(node) -> list[str]:
    """Every hf:// path in a Pocket TTS config, however deep."""
    if isinstance(node, dict):
        return [path for value in node.values() for path in hub_paths(value)]
    if isinstance(node, list):
        return [path for value in node for path in hub_paths(value)]
    return [node] if isinstance(node, str) and node.startswith("hf://") else []


WEIGHTS = ("weights_path", "weights_path_without_voice_cloning")


def model_paths(pocket_lang: str, clones: bool = False) -> list[str]:
    """The hf:// paths of a language's model: its weights, in the form that
    clones (from the gated repo) or the open one, then the rest its config
    names (the tokenizer)."""
    settings = config(pocket_lang)
    if clones:
        weights = settings.get("weights_path")
    else:
        weights = settings.get("weights_path_without_voice_cloning") or settings.get("weights_path")
    others = {key: value for key, value in settings.items() if key not in WEIGHTS}
    return ([weights] if weights else []) + hub_paths(others)


def model_files(pocket_lang: str, clones: bool = False) -> list[HubFile]:
    """A language's model, as files to fetch (see model_paths)."""
    return [HubFile.parse(path) for path in model_paths(pocket_lang, clones)]


def voice_path(pocket_lang: str, voice: str) -> str:
    """One of Pocket TTS's own voices, as a language's model reads it."""
    return get_predefined_voice(pocket_lang, voice)


def voice_file(pocket_lang: str, voice: str) -> HubFile:
    return HubFile.parse(voice_path(pocket_lang, voice))

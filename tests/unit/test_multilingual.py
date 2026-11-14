"""Unit tests for multilingual support."""

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from kokoro_onnx.config import SUPPORTED_LANGUAGES
from kokoro_onnx.config import VOCAB as KOKORO_VOCAB
from kokoro_onnx.tokenizer import Tokenizer
from pocket_tts.default_parameters import DEFAULT_VOICE_FOR_LANGUAGE

from iris.controller import PREVIEW_LINES, AssistantController
from iris.events import EventBus
from iris.language import language_of
from iris.personas import BUILTIN_DIR, Persona
from iris.tts import (
    KOKORO_LANGUAGES,
    KOKORO_V1_SYMBOLS,
    KOKORO_VOICES_SPEAK,
    POCKET_LANGUAGES,
    KokoroGenerator,
    PocketGenerator,
    SpeechGenerator,
    voice_language,
    voice_languages,
)

# The languages Iris speaks its replies in, as Whisper codes
SPOKEN = ("en", "fr", "it", "de", "es", "pt", "nl", "hi", "ja", "zh")
# The measurements the voices' languages come from
MEASURED = json.loads((Path(__file__).parent / "fixtures" / "voice_languages.json").read_text())


# ---------------------------------------------------------------------------
# Kokoro's languages
# ---------------------------------------------------------------------------


def test_kokoro_speaks_no_german_or_dutch():
    assert set(KOKORO_LANGUAGES) == {"en", "fr", "it", "es", "pt", "hi", "ja", "zh"}


@patch.object(KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin"))
@patch("iris.tts.Kokoro")
def _kokoro(voice, kokoro, _):
    kokoro.return_value.create.return_value = (np.zeros(100, dtype="float32"), 24000)
    kokoro.return_value.tokenizer.phonemize.return_value = "fəʊnimz"
    return KokoroGenerator(voice=voice, speed=1.0), kokoro.return_value


def test_a_kokoro_voice_speaks_its_own_language_in_itself():
    generator, kokoro = _kokoro("ff_siwis")
    os.unlink(generator.generate("Bonjour", language="fr"))
    assert kokoro.create.call_args.kwargs["lang"] == "fr-fr"
    assert kokoro.create.call_args.kwargs["voice"] == "ff_siwis"


def test_a_language_the_voice_does_not_speak_is_not_said_by_another_voice():
    """af_bella's French replies were said by ff_siwis, another woman."""
    generator, kokoro = _kokoro("af_bella")
    assert generator.generate("Bonjour", language="fr") is None
    assert generator.generate("こんにちは", language="ja") is None
    assert generator.generate("Hallo", language="de") is None
    assert not kokoro.create.called


def test_english_by_default():
    generator, kokoro = _kokoro("af_bella")
    os.unlink(generator.generate("Hello"))
    assert kokoro.create.call_args.kwargs["lang"] == "en-us"


class TestPersonaVoice:
    """A persona's voice is never swapped for another."""

    def test_english_keeps_the_persona_voice(self):
        assert _kokoro("af_sarah")[0].code_for("en") == "en-us"

    def test_british_voice_gets_british_english(self):
        assert _kokoro("bm_lewis")[0].code_for("en") == "en-gb"

    def test_a_language_the_voice_does_not_speak_has_no_code(self):
        assert _kokoro("bm_lewis")[0].code_for("ja") is None
        assert _kokoro("bm_lewis")[0].code_for("de") is None

    @pytest.mark.parametrize("voice", [v for v in KOKORO_VOICES_SPEAK if v != "jf_alpha"])
    def test_every_kokoro_voice_speaks_english(self, voice):
        """Whisper heard every one of them word for word in English, but
        jf_alpha, whose English it took for Japanese one time in two."""
        assert _kokoro(voice)[0].code_for("en") in ("en-us", "en-gb")
        assert _kokoro("jf_alpha")[0].code_for("en") is None

    @pytest.mark.parametrize(
        "voice, language, code",
        [("ef_dora", "es", "es"), ("em_alex", "es", "es"), ("pf_dora", "pt", "pt-br"),
         ("pm_alex", "pt", "pt-br"), ("hf_alpha", "hi", "hi"), ("hm_omega", "hi", "hi")],
    )  # fmt: skip
    def test_a_spanish_portuguese_or_hindi_voice_speaks_its_language_itself(
        self, voice, language, code
    ):
        assert _kokoro(voice)[0].code_for(language) == code

    def test_voice_switch_applies_to_the_next_sentence(self):
        """Switching persona swaps generator.voice; the next reply must use it."""
        generator = _kokoro("af_sarah")[0]
        generator.voice = "bm_lewis"
        assert generator.code_for("en") == "en-gb"


# ---------------------------------------------------------------------------
# Which languages each voice speaks properly: measured
# ---------------------------------------------------------------------------


def _measured_speaks(engine: str, voice: str, language: str) -> bool:
    """The rule the measurements are judged by (MEASURED["about"]): the
    language heard in 3 of 4 of the voice's sentences, and on average about
    as many of their words as the voices made for the language manage. In
    the language a voice was made for, its words are enough."""
    cell = MEASURED[engine][voice][language]
    bar = MEASURED["bar"][engine].get(language, MEASURED["bar"]["default"])
    heard = [d == language or p >= 0.5 for d, p in zip(cell["detected"], cell["p"])]
    own = language == voice_language(engine, voice)
    words = sum(cell["shares"]) / len(cell["shares"])
    return (own or sum(heard) / len(heard) >= MEASURED["bar"]["detected"]) and words >= bar


@pytest.mark.parametrize("voice", sorted(MEASURED["kokoro"]))
def test_each_kokoro_voice_speaks_the_languages_measured(voice):
    measured = {lang for lang in MEASURED["kokoro"][voice] if lang in KOKORO_LANGUAGES
                and _measured_speaks("kokoro", voice, lang)}
    assert set(voice_languages("kokoro", voice)) == measured


def test_every_kokoro_voice_was_measured_in_every_language_it_has():
    voices = {v["id"] for v in KokoroGenerator.voices()} or set(KOKORO_VOICES_SPEAK)
    assert set(KOKORO_VOICES_SPEAK) == set(MEASURED["kokoro"]) >= voices
    for voice, cells in MEASURED["kokoro"].items():
        assert set(KOKORO_LANGUAGES) <= set(cells), voice


@pytest.mark.parametrize("voice", sorted(MEASURED["kyutai"]))
def test_the_kyutai_voices_measured_speak_each_language_measured(voice):
    """Each through the language's model with its own prompt. The voices not
    measured are taken to do as these do: they are prompts like them."""
    for language in MEASURED["kyutai"][voice]:
        speaks = _measured_speaks("kyutai", voice, language)
        assert speaks == (language in voice_languages("kyutai", voice)), language


# Kyutai voices measured in English and another language
PITCHED = sorted(v for v, cells in MEASURED["kyutai"].items() if "en" in cells and len(cells) > 1)


@pytest.mark.parametrize("voice", PITCHED)
def test_a_kyutai_voice_stays_itself_in_each_language(voice):
    """Its median pitch in each language is within a third of its English
    one: Juergen, who spoke Iris's German, has half Anna's pitch."""
    cells = MEASURED["kyutai"][voice]
    english = cells["en"]["pitch"]
    for language in voice_languages("kyutai", voice):
        if language in cells:
            assert abs(cells[language]["pitch"] - english) <= english / 3, language


# ---------------------------------------------------------------------------
# Who speaks each language, for each persona shipped with Iris
# ---------------------------------------------------------------------------

# The engine's code for each language: a reply is spoken with the persona's
# own engine and voice, through it, or not at all
KOKORO_CODE = {**KOKORO_LANGUAGES}
POCKET_MODEL = {**POCKET_LANGUAGES}
SHIPPED = {p.stem: Persona.from_file(p) for p in BUILTIN_DIR.glob("*.yaml")}


def spoken_with(persona_id: str, language: str):
    """(engine, voice, code) a shipped persona's reply in this language is to
    be spoken with, or None when its voice does not speak the language."""
    persona = SHIPPED[persona_id]
    if language not in voice_languages(persona.engine, persona.voice):
        return None
    if persona.engine == "kyutai":
        return "kyutai", persona.voice, POCKET_MODEL[language]
    code = KOKORO_CODE[language]
    if code == "en-us" and persona.voice.startswith("b"):
        code = "en-gb"
    return "kokoro", persona.voice, code


# The languages each shipped persona's voice speaks, pinned
SHIPPED_SPEAK = {
    "iris": ["en", "fr", "it", "de", "es", "pt", "nl"],
    "hal": ["en", "it", "es", "pt", "zh"],
    "computer": ["en", "es", "pt", "zh"],
}
# The language each code of the engines stands for
CODE_LANGUAGE = {
    **{code: lang for lang, code in KOKORO_LANGUAGES.items()},
    "en-gb": "en",
    **{model: lang for lang, model in POCKET_LANGUAGES.items()},
}
# Kokoro's voices for the languages kokoro-onnx refuses by name
KOKORO_ESPEAK_VOICES = ["ef_dora", "em_alex", "pf_dora", "pm_alex", "hf_alpha", "hm_omega"]


def test_the_persona_editor_is_told_irises_languages_and_each_voices():
    voices = AssistantController(EventBus()).persona_voices()
    assert [language["code"] for language in voices["languages"]] == list(SPOKEN)
    assert voices["languages"][3] == {"code": "de", "name": "German"}
    kyutai = {voice["id"]: voice for voice in voices["engines"]["kyutai"]}
    assert kyutai["anna"]["languages"] == SHIPPED_SPEAK["iris"]
    # The language a voice was made for comes first
    assert kyutai["juergen"]["languages"][0] == "de"


class PreviewPipeline:
    """Takes what a voice preview queues, instead of speaking it."""

    def __init__(self):
        self.queued = []

    def stop(self):
        pass

    def queue_speech(self, text, language=None, sound=None):
        self.queued.append((text, language, sound))


@pytest.fixture
def engines_used():
    """Kokoro and Pocket TTS stand-ins, which note the voice and the
    language code they are asked to speak with (no models, no sound).

    Yields spoken_with(persona_id, language) for a shipped persona, and
    spoken_with(engine, language, voice=...) for a persona on that voice;
    preview holds the engine, voice and speed SpeechGenerator.generate is
    given for a voice preview."""
    with patch("iris.tts.TTSModel") as pocket, patch("iris.tts.Kokoro") as kokoro, patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        model = pocket.load_model.return_value
        model.sample_rate = 24000
        model.generate_audio.return_value = torch.zeros(100)
        kokoro.return_value.create.return_value = (np.zeros(100, dtype="float32"), 24000)
        phonemize = kokoro.return_value.tokenizer.phonemize
        phonemize.return_value = "fəʊnimz"

        def spoken_with(who: str, language: str, voice: str = "", preview: dict | None = None):
            if voice:
                generator = SpeechGenerator(who, voice)
            else:
                persona = Persona.from_file(BUILTIN_DIR / f"{who}.yaml")
                generator = SpeechGenerator(persona.engine, persona.voice, persona.speed)
            pocket.reset_mock()
            kokoro.reset_mock()
            path = generator.generate("Text.", language=language, **(preview or {}))
            if path is None:
                # Not spoken, and by no other engine
                assert not kokoro.return_value.create.called
                assert not model.generate_audio.called
                return None
            os.unlink(path)
            if kokoro.return_value.create.called:
                spoken = kokoro.return_value.create.call_args.kwargs
                if "phonemes" in spoken and language == "zh":
                    # Kokoro's own Mandarin phonemes, misaki's
                    return "kokoro", spoken["voice"], "cmn"
                if "phonemes" in spoken:
                    # Spoken from the phonemes espeak made in the language's code
                    assert spoken["phonemes"] is phonemize.return_value
                    return "kokoro", spoken["voice"], phonemize.call_args.args[1]
                return "kokoro", spoken["voice"], spoken["lang"]
            # The model the audio came from: loaded now, or with the persona
            # (English), and the voice its state was made for
            voice = model.get_state_for_audio_prompt.call_args.args[0]
            loaded = [c.kwargs["language"] for c in pocket.load_model.call_args_list]
            return "kyutai", voice, loaded[-1] if loaded else "english"

        yield spoken_with


@pytest.mark.parametrize("persona_id", sorted(SHIPPED))
def test_each_shipped_persona_speaks_the_languages_pinned(persona_id):
    persona = SHIPPED[persona_id]
    assert voice_languages(persona.engine, persona.voice) == SHIPPED_SPEAK[persona_id]


@pytest.mark.parametrize("persona_id", sorted(SHIPPED))
@pytest.mark.parametrize("language", SPOKEN)
def test_each_language_is_spoken_with_the_personas_engine_and_voice_or_not_at_all(
    engines_used, persona_id, language
):
    """Iris's German replies were Juergen's, a man's voice, and HAL's
    German replies Pocket TTS's: the voice changed mid-conversation."""
    assert engines_used(persona_id, language) == spoken_with(persona_id, language)


@pytest.mark.parametrize("persona_id", sorted(SHIPPED))
@pytest.mark.parametrize("language", SPOKEN)
def test_no_language_is_spoken_with_the_code_of_another(engines_used, persona_id, language):
    """Italian once came out of the Portuguese model: it sounded Portuguese."""
    spoken = engines_used(persona_id, language)
    if spoken is not None:
        engine, voice, code = spoken
        assert CODE_LANGUAGE[code] == language
        assert (engine, voice) == (SHIPPED[persona_id].engine, SHIPPED[persona_id].voice)


@pytest.mark.parametrize("persona_id", sorted(SHIPPED))
def test_a_language_no_voice_speaks_is_not_spoken(engines_used, persona_id):
    assert engines_used(persona_id, "ru") is None


@pytest.mark.parametrize("voice", PocketGenerator.catalogue())
@pytest.mark.parametrize("language", list(POCKET_LANGUAGES))
def test_every_kyutai_voice_speaks_each_language_through_its_model(engines_used, voice, language):
    """Or not at all, a language it does not speak properly."""
    model = POCKET_LANGUAGES[language]
    spoken = ("kyutai", voice, model) if language in voice_languages("kyutai", voice) else None
    assert engines_used("kyutai", language, voice=voice) == spoken


@pytest.mark.parametrize("voice", KOKORO_ESPEAK_VOICES)
def test_a_persona_on_a_spanish_portuguese_or_hindi_voice_is_heard(engines_used, voice):
    """These voices were never heard: kokoro-onnx refused their languages,
    and af_bella spoke in their place, in English as in their own language."""
    language = voice_language("kokoro", voice)
    code = KOKORO_LANGUAGES[language]
    assert engines_used("kokoro", language, voice=voice) == ("kokoro", voice, code)
    assert engines_used("kokoro", "en", voice=voice) == ("kokoro", voice, "en-us")


@pytest.mark.parametrize("voice", KOKORO_ESPEAK_VOICES + ["ff_siwis", "bm_lewis"])
def test_a_preview_says_its_line_in_the_voice_itself_and_its_language(engines_used, voice):
    controller = AssistantController(EventBus())
    controller.assistant = MagicMock()
    controller.assistant.speech_pipeline = pipeline = PreviewPipeline()
    controller.persona_preview({"name": "Nova", "engine": "kokoro", "voice": voice})

    [(line, language, sound)] = pipeline.queued
    assert language == voice_language("kokoro", voice)
    assert line == PREVIEW_LINES[language].format(name="Nova")
    # Rendered as the speech pipeline renders it, in the voice and language given
    preview = {key: sound[key] for key in ("engine", "voice", "speed")}
    engine, spoken, code = engines_used("iris", language, preview=preview)
    assert (engine, spoken, CODE_LANGUAGE[code]) == ("kokoro", voice, language)


@pytest.mark.parametrize("voice", ["alba", "estelle", "giovanni", "juergen", "lola", "rafael", "daan"])
def test_a_kyutai_voice_previews_its_line_in_its_own_language(engines_used, voice):
    controller = AssistantController(EventBus())
    controller.assistant = MagicMock()
    controller.assistant.speech_pipeline = pipeline = PreviewPipeline()
    controller.persona_preview({"name": "Nova", "engine": "kyutai", "voice": voice})

    [(line, language, sound)] = pipeline.queued
    assert language == voice_language("kyutai", voice)
    assert line == PREVIEW_LINES[language].format(name="Nova")
    preview = {key: sound[key] for key in ("engine", "voice", "speed")}
    engine, spoken, code = engines_used("iris", language, preview=preview)
    assert (engine, spoken, CODE_LANGUAGE[code]) == ("kyutai", voice, language)


def test_kokoro_onnx_is_only_given_languages_it_takes_and_phonemes_for_the_rest():
    """kokoro-onnx refuses any other language by name: a reply in it would
    not be heard."""
    with patch("iris.tts.Kokoro") as kokoro, patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        kokoro.return_value.create.return_value = (np.zeros(100, dtype="float32"), 24000)
        phonemize = kokoro.return_value.tokenizer.phonemize
        phonemize.return_value = "fəʊnimz"
        for language, code in KOKORO_LANGUAGES.items():
            # A voice that speaks the language
            voice = next(v for v, spoken in KOKORO_VOICES_SPEAK.items() if language in spoken)
            generator = KokoroGenerator(voice=voice)
            kokoro.reset_mock()
            os.unlink(generator.generate("Text.", language=language))
            spoken = kokoro.return_value.create.call_args.kwargs
            assert spoken["lang"] in SUPPORTED_LANGUAGES
            if code not in SUPPORTED_LANGUAGES:
                phonemize.assert_called_once_with("Text.", code)
                assert spoken["phonemes"] is phonemize.return_value
            if language == "zh":
                # Kokoro's own Mandarin phonemes, the English word in it by espeak
                assert spoken["phonemes"] and spoken["lang"] == "en-us"


@pytest.mark.parametrize(
    "code, text, word",
    [("fr-fr", "J'aime le football.", "fˈʊtbɔːl"), ("hi", "मैं Nova हूँ।", "nˈəʊvə"),
     ("cmn", "我喜欢Queen的歌。", "kwˈiːn")],
)  # fmt: skip
def test_a_word_espeak_reads_as_english_is_said_without_the_switch(code, text, word):
    """espeak marks where it reads a word as English, "(en)fˈʊtbɔːl(fr)";
    Kokoro said the marks: "en football fr". Real espeak, through Kokoro's
    tokenizer as Iris sets it up."""
    try:
        phonemes = Tokenizer().phonemize(text, code)
    except RuntimeError as error:
        pytest.skip(f"espeak cannot run here: {error}")
    assert word in phonemes
    assert "en" + word not in phonemes


def _given_to_kokoro(text: str, language: str, voice: str) -> dict:
    """What KokoroGenerator gives kokoro-onnx to say a text."""
    with patch("iris.tts.Kokoro") as kokoro, patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        kokoro.return_value.create.return_value = (np.zeros(100, dtype="float32"), 24000)
        kokoro.return_value.tokenizer = Tokenizer()
        os.unlink(KokoroGenerator(voice=voice).generate(text, language=language))
        return kokoro.return_value.create.call_args.kwargs


def test_a_japanese_reply_is_read_not_called_chinese_letters():
    """Kokoro speaks Japanese from espeak's phonemes, and espeak calls every
    kanji a "Chinese letter", in English: a Japanese reply was gibberish
    ("Chinese letter, Chinese letter, no Chinese letter..."). Given in kana,
    numbers too, it is read. Real espeak, through Kokoro's tokenizer."""
    try:
        given = _given_to_kokoro("東京の天気は晴れ、18度です。", "ja", "jf_alpha")
        phonemes = Tokenizer().phonemize(given["text"], given["lang"])
    except RuntimeError as error:
        pytest.skip(f"espeak cannot run here: {error}")
    assert given["text"] == "とうきょうのてんきははれ、じゅうはちどです。"
    assert "tʃˈaɪniːz" not in phonemes and "dʒˈapəniːz" not in phonemes, phonemes


def test_mandarin_is_given_as_kokoros_own_phonemes():
    """espeak's Mandarin was a third intelligible. misaki's phonemes are
    what Kokoro was trained on: every symbol of them is on the token id
    Kokoro v1.0 has for it, numbers are read in Chinese, and an English word
    is read by espeak's English."""
    try:
        given = _given_to_kokoro("北京的门票是23.5元，我喜欢Queen的歌。", "zh", "zf_xiaoxiao")
    except RuntimeError as error:
        pytest.skip(f"espeak cannot run here: {error}")
    phonemes = given["phonemes"]
    assert given["lang"] == "en-us" and "↗" in phonemes and "kwˈiːn" in phonemes
    assert all(symbol in KOKORO_VOCAB for symbol in phonemes), phonemes
    assert not any(c.isdigit() for c in phonemes)
    # ʨ and ʦ, which misaki writes, on Kokoro v1.0's ids 21 and 20
    tokens = Tokenizer().tokenize(phonemes)
    assert KOKORO_VOCAB["ʨ".translate(KOKORO_V1_SYMBOLS)] == 21 and 21 in tokens
    assert KOKORO_VOCAB["ʦ".translate(KOKORO_V1_SYMBOLS)] == 20


@pytest.mark.parametrize(
    "language, text, read",
    [("fr", "Le billet coûte 16,50 €.", "Le billet coûte seize euros cinquante."),
     ("it", "Ci saranno 21 gradi.", "Ci saranno ventuno gradi.")],
)  # fmt: skip
def test_kokoro_is_given_numbers_in_words_where_it_reads_them_as_english(language, text, read):
    """kokoro-onnx reads numbers with English rules first: "16,50" was 1650."""
    voice = {"fr": "ff_siwis", "it": "if_sara"}[language]
    assert _given_to_kokoro(text, language, voice)["text"] == read


def test_pocket_tts_is_given_numbers_in_words_except_in_english():
    """Only Pocket TTS's English model reads digits: "am 27. November" was
    "am Gas im November"."""
    with patch("iris.tts.TTSModel") as pocket:
        model = pocket.load_model.return_value
        model.sample_rate = 24000
        model.generate_audio.return_value = torch.zeros(100)
        generator = PocketGenerator("alba")
        for language, text in [("de", "Am 27. November um 10 Uhr."), ("en", "On November 27 at 10.")]:
            model.generate_audio.reset_mock()
            os.unlink(generator.generate(text, language=language))
            said = " ".join(c.args[1] for c in model.generate_audio.call_args_list)
            assert said == ("Am siebenundzwanzigsten November um zehn Uhr." if language == "de" else text)


def test_each_pocket_language_has_a_voice_made_for_it():
    for lang, model in POCKET_LANGUAGES.items():
        if lang != "en":
            assert voice_language("kyutai", DEFAULT_VOICE_FOR_LANGUAGE[model]) == lang


def test_a_kokoro_personas_german_loads_no_pocket_tts():
    with patch("iris.tts.TTSModel") as pocket, patch("iris.tts.Kokoro"), patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        generator = SpeechGenerator("kokoro", "af_sarah")
        assert generator.generate("Guten Tag.", language="de") is None
        generator.prepare_language("de")

        assert not pocket.load_model.called


def test_the_voice_for_a_language_is_prepared_once_ahead_of_its_first_reply():
    """A first German reply waited seconds for Pocket TTS's German model
    and the voice's prompt for it; prepared while the model thinks, they
    are ready when the reply comes. A language the voice does not speak
    has nothing to prepare, and no other engine loads for it."""
    with patch("iris.tts.TTSModel") as pocket, patch("iris.tts.Kokoro") as kokoro, patch.object(
        KokoroGenerator, "_locate_models", return_value=("/m.onnx", "/v.bin")
    ):
        model = pocket.load_model.return_value
        model.sample_rate = 24000
        model.generate_audio.return_value = torch.zeros(100)
        kokoro.return_value.create.return_value = (np.zeros(100, dtype="float32"), 24000)
        generator = SpeechGenerator("kyutai", "alba")

        generator.prepare_language("de")
        generator.prepare_language("de")
        loaded = [call(language="english"), call(language="german")]
        assert pocket.load_model.call_args_list == loaded
        assert model.get_state_for_audio_prompt.call_args_list == [call("alba")]
        assert model.generate_audio.call_count == 1

        generator.prepare_language("ja")
        generator.prepare_language("ru")
        assert not kokoro.called
        assert model.generate_audio.call_count == 1

        # Another voice is prepared anew
        generator.voice = "anna"
        generator.prepare_language("de")
        assert model.get_state_for_audio_prompt.call_args_list == [call("alba"), call("anna")]


# ---------------------------------------------------------------------------
# The language a text is written in
# ---------------------------------------------------------------------------


class TestLanguageOf:
    @pytest.mark.parametrize(
        "language, text",
        [
            ("en", "What would you like to talk about today?"),
            ("fr", "De quoi voulez-vous parler aujourd'hui ?"),
            ("it", "Di cosa vorresti parlare oggi?"),
            ("de", "Worüber möchtest du heute sprechen?"),
            ("es", "¿De qué quieres hablar hoy?"),
            ("pt", "Sobre o que você quer falar hoje?"),
            ("nl", "Waar wil je vandaag over praten?"),
            ("ja", "今日は何について話したいですか？"),
            ("zh", "你今天想聊些什么？"),
        ],
    )
    def test_each_language_is_told(self, language, text):
        assumed = "de" if language == "en" else "en"
        assert language_of(text, assumed, SPOKEN) == language

    def test_italian_heard_as_portuguese_is_italian(self):
        assert language_of("Parliamo italiano.", "pt", SPOKEN) == "it"

    @pytest.mark.parametrize(
        "piece",
        [
            "Sì, certamente.",
            "Posso parlare italiano.",
            "Va bene, parliamo italiano.",
            "Di cosa vorresti parlare?",
        ],
    )
    def test_an_italian_reply_to_english_is_italian(self, piece):
        assert language_of(piece, "en", SPOKEN) == "it"

    @pytest.mark.parametrize(
        "text, assumed",
        [("Okay.", "de"), ("Ciao!", "en"), ("Sure.", "it"), ("Fitzcarraldo.", "en"),
         ("42", "fr"), ("...", "en"), ("", "en")],
    )  # fmt: skip
    def test_a_word_or_two_keep_the_language(self, text, assumed):
        assert language_of(text, assumed, SPOKEN) == assumed

    @pytest.mark.parametrize(
        "text",
        [
            "Werner Herzog is a German director, born in Munich in 1942.",
            "Who is Jürgen Klopp?",
            "Look up tickets for the Musée d'Orsay.",
            "Search for the opening hours of Schloss Neuschwanstein.",
        ],
    )
    def test_names_from_elsewhere_keep_a_sentence_english(self, text):
        assert language_of(text, "en", SPOKEN) == "en"

    @pytest.mark.parametrize("text", ["Опиши себя коротко.", "안녕하세요, 잘 지내세요?"])
    def test_a_language_none_speaks_is_not_taken_for_one_of_them(self, text):
        assert language_of(text, "en", SPOKEN) == "en"

    def test_only_the_languages_given_are_told(self):
        assert language_of("Worüber möchtest du heute sprechen?", "en", {"en", "fr"}) == "en"


# ---------------------------------------------------------------------------
# SpeechPipeline passes language through
# ---------------------------------------------------------------------------


class TestSpeechPipelineLanguage:
    def test_current_language_defaults_to_english(self):
        from iris.tts import SpeechPipeline

        mock_gen = MagicMock()
        mock_player = MagicMock()
        pipeline = SpeechPipeline(mock_gen, mock_player)
        assert pipeline.current_language == "en"
        pipeline.stop()

    def test_queue_speech_sends_language_tuple(self):
        from iris.tts import SpeechPipeline

        mock_gen = MagicMock()
        mock_player = MagicMock()
        pipeline = SpeechPipeline(mock_gen, mock_player)

        pipeline.current_language = "ja"
        pipeline.text_queue = MagicMock()
        pipeline.queue_speech("テスト")
        # (text, language, the speech epoch it was queued in)
        pipeline.text_queue.put.assert_called_once_with(("テスト", "ja", 0, None))
        pipeline.stop()


# ---------------------------------------------------------------------------
# Transcription returns language
# ---------------------------------------------------------------------------


class TestTranscriptionLanguage:
    @patch("iris.transcription.WhisperModel")
    def test_transcribe_returns_language(self, MockWhisper):
        from iris.transcription import WhisperTranscriber

        mock_info = MagicMock()
        mock_info.language = "fr"
        mock_segment = MagicMock()
        mock_segment.text = "Bonjour"
        mock_segment.no_speech_prob = 0.02
        MockWhisper.return_value.transcribe.return_value = ([mock_segment], mock_info)

        with patch("iris.transcription.torch", create=True) as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.cuda.is_available.return_value = False
            transcriber = WhisperTranscriber(
                model="tiny", device="cpu", engine="faster-whisper"
            )

        text, lang = transcriber.transcribe("/fake/audio.wav")
        assert text == "Bonjour"
        assert lang == "fr"

    @patch("iris.transcription.WhisperModel")
    def test_transcribe_drops_segments_rated_as_silence(self, MockWhisper):
        """Noise transcribed as "You" must not become a user turn."""
        from iris.transcription import WhisperTranscriber

        mock_info = MagicMock()
        mock_info.language = "en"
        phantom = MagicMock()
        phantom.text = "You"
        phantom.no_speech_prob = 0.9
        MockWhisper.return_value.transcribe.return_value = ([phantom], mock_info)

        transcriber = WhisperTranscriber(
            model="tiny", device="cpu", engine="faster-whisper"
        )
        text, _ = transcriber.transcribe("/fake/audio.wav")
        assert text == ""


# ---------------------------------------------------------------------------
# LLM language instruction
# ---------------------------------------------------------------------------


class TestLLMLanguageInstruction:
    @patch("iris.llm.ollama")
    def test_chat_stream_injects_language_for_non_english(self, mock_ollama):
        mock_ollama.show.return_value = MagicMock(
            modelinfo={}, details=MagicMock(family="test")
        )
        mock_ollama.list.return_value = []

        # Make chat return a simple stream
        mock_ollama.chat.return_value = iter(
            [{"message": {"content": "Bonjour!"}}]
        )

        from iris.llm import ConversationManager

        cm = ConversationManager(model="test", tools=None)
        # Consume the generator
        list(cm.chat_stream("Salut", response_language="fr"))

        # Check the messages passed to ollama.chat
        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        # Told with the user's message: a system message is only ever first
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "Respond in French" in messages[-1]["content"]
        assert messages[-1]["content"].endswith("Salut")

    @patch("iris.llm.ollama")
    def test_chat_stream_no_language_instruction_for_english(self, mock_ollama):
        mock_ollama.show.return_value = MagicMock(
            modelinfo={}, details=MagicMock(family="test")
        )
        mock_ollama.list.return_value = []
        mock_ollama.chat.return_value = iter(
            [{"message": {"content": "Hello!"}}]
        )

        from iris.llm import ConversationManager

        cm = ConversationManager(model="test", tools=None)
        list(cm.chat_stream("Hi", response_language="en"))

        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        # No language instruction should be injected for English
        lang_instructions = [
            m for m in messages
            if m["role"] == "system" and "speaking in" in m.get("content", "").lower()
        ]
        assert len(lang_instructions) == 0

    @patch("iris.llm.ollama")
    def test_back_in_english_the_model_is_told_to_answer_in_english(self, mock_ollama):
        """After German turns, the model went on answering in German: the
        conversation it is given is German, and nothing named English."""
        mock_ollama.show.return_value = MagicMock(modelinfo={}, details=MagicMock(family="test"))
        mock_ollama.list.return_value = []
        mock_ollama.chat.side_effect = lambda **kwargs: iter([{"message": {"content": "Gut."}}])

        from iris.llm import ConversationManager

        cm = ConversationManager(model="test", tools=None)

        def told(prompt, language):
            list(cm.chat_stream(prompt, response_language=language))
            messages = mock_ollama.chat.call_args.kwargs["messages"]
            return re.findall(r"Respond in ([\w ()]+)\.", messages[-1]["content"])

        assert told("Hi", "en") == []
        assert told("Wie spät ist es?", "de") == ["German"]
        assert told("Und morgen?", "de") == ["German"]
        assert told("Back to English, please. What time is it?", "en") == ["English"]
        # Once back, English goes on without a note, as at the start
        assert told("And tomorrow?", "en") == []

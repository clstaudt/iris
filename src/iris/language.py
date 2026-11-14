"""The language a text is written in, told from its words.

Whisper tells the language of what the user said from the sound, and a few
words in an accent can fool it: "Parliamo italiano." was taken for
Portuguese. A reply need not be in the user's language either: asked in
English whether it speaks Italian, the model answers in Italian. The voice
must speak a text in the language it is written in, or it speaks it with a
foreign accent, so that language is told from the text itself.
"""

import threading
from collections.abc import Collection

from py3langid.langid import MODEL_FILE, LanguageIdentifier

# A text is taken to be in another language than the one assumed only when it
# is more likely in it than not, and this many times likelier in it than in
# the one assumed: a word or two ("Okay.", "Ciao!", "Fitzcarraldo.") tells
# little, and a name in an English sentence must not switch the voice
LIKELY = 0.5
LIKELIER = 10.0

_identifier: LanguageIdentifier | None = None
_lock = threading.Lock()


def prepare() -> LanguageIdentifier:
    """The language identifier, loaded the first time (a third of a second)."""
    global _identifier
    with _lock:
        if _identifier is None:
            _identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
        return _identifier


def language_of(text: str, assumed: str, languages: Collection[str]) -> str:
    """The language a text is written in, of the given ones.

    Args:
        text: Words heard or to be spoken
        assumed: Whisper code of the language taken until the words clearly
            say otherwise: the one Whisper heard, or the one the reply was
            in so far
        languages: Whisper codes of the languages it may be in, e.g. those
            the voice speaks

    Returns:
        One of languages, or assumed when the words do not tell
    """
    if not any(c.isalpha() for c in text):
        return assumed
    likelihood = dict(prepare().rank(text))
    likeliest = max(likelihood, key=likelihood.__getitem__)
    # Most likely in a language that is not among them: none of them is meant
    if likeliest not in languages and likelihood[likeliest] >= LIKELY:
        return assumed
    among = {language: likelihood.get(language, 0.0) for language in languages}
    total = sum(among.values())
    if total <= 0.0:
        return assumed
    language = max(among, key=among.__getitem__)
    if among[language] >= LIKELY * total and among[language] >= LIKELIER * among.get(assumed, 0.0):
        return language
    return assumed

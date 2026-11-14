"""File names made from what the user named something: a persona, a voice."""

import re
import unicodedata

# Letters spelled out rather than stripped of their marks: the German ones as
# German writes them without, and those that have no marks to strip
SPELLED_OUT = str.maketrans(
    {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "æ": "ae",
        "œ": "oe",
        "ø": "o",
        "å": "a",
        "ł": "l",
        "đ": "d",
        "ð": "d",
        "þ": "th",
    }
)


def slug(name: str, fallback: str) -> str:
    """A file name for something named this: lower case, in plain letters
    and digits, its words joined by dashes. "Der Führer" is der-fuehrer and
    "Zoë" zoe; a name of no such letters at all is the fallback."""
    text = unicodedata.normalize("NFKC", name).lower().translate(SPELLED_OUT)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or fallback

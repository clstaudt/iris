"""What was heard against what was said, word by word.

Recognition writes the same words in many ways: "four" or "4", "L H four
oh two" or "LH402", "Musée" or "Musee", "3rd" or "third". Both sides are
brought to one form before they are compared:

- words are Unicode letters and digits, lower case, accents folded
  ("übermorgen" is one word, "Musée" is "musee");
- numbers are digits: "twenty six" is 26, "two thousand twenty seven" is
  2027, "oh" inside a run of digits is 0, and ordinals are numbers ("3rd"
  and "third" are 3), while digits said one by one stay apart ("four nine
  two" is 4 9 2);
- codes are joined: letters and digits said apart, with only spaces, dots
  or hyphens between them, are one word ("L H four oh two" is lh402);
- fillers (um, uh, er, hmm) are not expected to be heard.

Japanese and Chinese have no spaces to split words at: they are compared
character by character (character error rate).
"""

import re
import unicodedata

UNITS = {
    word: n
    for n, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
TENS = {
    word: 10 * n
    for n, word in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)
}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}
ORDINAL_DIGITS = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
FILLERS = frozenset({"um", "umm", "uh", "uhm", "er", "erm", "hmm", "hm", "mm", "mmm", "ah"})
# A word: letters and digits, with an apostrophe inside ("what's")
TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
# What may lie between the pieces of a code
CODE_JOIN = re.compile(r"^[\s.\-]+$")
# Languages compared character by character
CHARACTER_LANGUAGES = ("ja", "zh")


def fold(text: str) -> str:
    """Lower case, accents taken off: "Musée" is "musee"."""
    text = unicodedata.normalize("NFKD", (text or "").replace("ß", "ss").replace("’", "'"))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _tokens(text: str) -> list[tuple[str, str, str]]:
    """(word folded, word as written, what came before it) for each word."""
    text = unicodedata.normalize("NFKC", text or "").replace("’", "'")
    out, last = [], 0
    for match in TOKEN.finditer(text):
        out.append((fold(match.group(0)), match.group(0), text[last : match.start()]))
        last = match.end()
    return out


def _digit_word(word: str) -> bool:
    return word.isdigit() or (word in UNITS and UNITS[word] < 10)


def _small(words: list[str], i: int) -> tuple:
    """A number below a hundred at words[i]: (value, next index, ordinal),
    or None."""
    word = words[i]
    if word in TENS:
        value, j = TENS[word], i + 1
        if j < len(words) and words[j] in UNITS and 0 < UNITS[words[j]] < 10:
            return value + UNITS[words[j]], j + 1, False
        if j < len(words) and words[j] in ORDINALS and ORDINALS[words[j]] < 10:
            return value + ORDINALS[words[j]], j + 1, True
        return value, j, False
    if word in UNITS:
        return UNITS[word], i + 1, False
    if word in ORDINALS:
        return ORDINALS[word], i + 1, True
    return None


def _number(words: list[str], i: int) -> tuple:
    """A number said in words from words[i]: (value, next index), or None.
    Digits said one by one ("four nine two") are not run together."""
    first = _small(words, i)
    if first is None:
        return None
    current, j, ordinal = first
    total = 0
    while not ordinal and j < len(words) and words[j] in SCALES:
        scale = SCALES[words[j]]
        j += 1
        if scale == 100:
            current *= 100
        else:
            total += current * scale
            current = 0
        if j + 1 < len(words) and words[j] == "and" and _small(words, j + 1) is not None:
            j += 1
        following = _small(words, j) if j < len(words) else None
        if following is not None:
            value, j, ordinal = following
            current += value
    return total + current, j


def _numbers(tokens: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """The tokens with numbers said in words as digits."""
    words = [t[0] for t in tokens]
    out = []
    i = 0
    while i < len(words):
        word, written, before = tokens[i]
        found = ORDINAL_DIGITS.match(word)
        if found:
            out.append((found.group(1), found.group(1), before))
            i += 1
            continue
        if word == "oh":
            previous = bool(out) and out[-1][0].isdigit()
            following = [_digit_word(w) for w in words[i + 1 : i + 3]]
            if (previous and following[:1] == [True]) or following == [True, True]:
                out.append(("0", "0", before))
                i += 1
                continue
        if word in UNITS or word in TENS or word in ORDINALS:
            value, j = _number(words, i)
            out.append((str(value), str(value), before))
            i = j
            continue
        out.append(tokens[i])
        i += 1
    return out


def _code_piece(token: tuple[str, str, str]) -> str:
    """What kind of code piece a token is: "digits", "letter" (a capital
    said alone), "letters" (a few capitals), "mixed", or ""."""
    word, written, _ = token
    if word.isdigit():
        return "digits"
    if len(written) == 1 and written.isalpha() and written.isupper():
        return "letter"
    if written.isalpha() and written.isupper() and len(written) <= 3:
        return "letters"
    if word.isalnum() and any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
        return "mixed"
    return ""


def _codes(tokens: list[tuple[str, str, str]]) -> list[str]:
    """The words, with codes said in pieces joined into one."""
    out: list[str] = []
    run: list[tuple[str, str]] = []  # (word, kind)

    def close():
        kinds = [k for _, k in run]
        letters = [w for w, k in run if k == "letter"]
        # "I" and "a" said alone are words, not the letters of a code
        code_letters = [w for w in letters if w not in ("i", "a")]
        has_letters = bool(code_letters) or "letters" in kinds or "mixed" in kinds
        joinable = len(run) > 1 and (
            ("digits" in kinds and has_letters)
            or kinds.count("digits") > 1
            or "mixed" in kinds
            or (len(letters) == len(run) and bool(code_letters))
        )
        if joinable:
            out.append("".join(w for w, _ in run))
        else:
            out.extend(w for w, _ in run)
        run.clear()

    for token in tokens:
        kind = _code_piece(token)
        if kind and run and CODE_JOIN.match(token[2] or " "):
            run.append((token[0], kind))
        else:
            close()
            if kind:
                run.append((token[0], kind))
            else:
                out.append(token[0])
    close()
    return out


def words(text: str) -> list[str]:
    """The words of a text in the one form both sides are compared in."""
    return _codes(_numbers(_tokens(text)))


def normalized(text: str) -> str:
    """The words of a text as one line, for terms to be searched in."""
    return " ".join(words(text))


def units(text: str) -> list[str]:
    """The words, with every code and number split into its characters, so
    "492", "4 9 2" and "4-9-2" count the same."""
    out = []
    for word in words(text):
        if any(c.isdigit() for c in word):
            out.extend(word)
        else:
            out.append(word)
    return out


def expected(said: str) -> list[str]:
    """What should be heard of what was said: its units, fillers left out."""
    return [w for w in units(said) if w not in FILLERS]


def characters(text: str) -> list[str]:
    """The letters and digits of a text, for languages without spaces."""
    return [c for c in fold(text) if c.isalnum()]


def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def character_error_rate(heard: str, said: str) -> float:
    said_chars = characters(said)
    if not said_chars:
        return 0.0
    return edit_distance(characters(heard), said_chars) / len(said_chars)


def heard_share(heard: str, said: str, language: str = "en") -> float:
    """How much of what was said is in what was heard, word by word (and
    character by character for Japanese and Chinese)."""
    if language in CHARACTER_LANGUAGES:
        return max(0.0, 1.0 - character_error_rate(heard, said))
    wanted = expected(said)
    if not wanted:
        return 1.0
    got = set(units(heard))
    return sum(1 for word in wanted if word in got) / len(wanted)


def heard_precision(heard: str, said: str) -> float:
    """How much of what was heard was said: words the recognizer added
    (another voice, a TV) bring it down."""
    got = units(heard)
    if not got:
        return 1.0
    said_units = set(units(said))
    return sum(1 for word in got if word in said_units or word in FILLERS) / len(got)


def same_words(heard: str, said: str) -> bool:
    """Whether what was heard is exactly what was said, once normalized:
    "Yes." for "yes", "4" for "four"; not "yes yes" or "thank you"."""
    return [w for w in units(heard) if w not in FILLERS] == expected(said)


def contains(text: str, item: str) -> bool:
    """Whether a word or code (normalized) is among the text's words."""
    wanted = words(item)
    have = words(text)
    if len(wanted) == 1:
        return wanted[0] in have
    line = " " + " ".join(have) + " "
    return " " + " ".join(wanted) + " " in line


def in_order(text: str, items: list[str]) -> bool:
    """Whether the items are heard in this order (each a word, or a letter
    or digit of a code)."""
    have = units(text)
    position = 0
    for item in items:
        wanted = units(item)
        for unit in wanted:
            try:
                position = have.index(unit, position) + 1
            except ValueError:
                return False
    return True


def digits(text: str) -> str:
    """The digits heard, in order."""
    return "".join(c for w in words(text) for c in w if c.isdigit())


def term_found(text: str, term: str) -> bool:
    """Whether a regular expression is found in the normalized text."""
    return re.search(term, normalized(text)) is not None

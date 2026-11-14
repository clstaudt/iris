"""Numbers written out as words, the way they are said in each language.

Pocket TTS's models for other languages than English read no digits ("am
27. November" comes out as "am Gas im November"), and Kokoro's text goes
through kokoro-onnx's English normalization before espeak, which drops the
decimal comma ("16,50 €" is read as 1650) and gives times and money English
words. So a reply is given to them with its numbers in words: dates, times,
ordinals, decimals, money, percentages, degrees and ranges each as the
language says them, the number words themselves from num2words.
"""

import re
from decimal import Decimal

from num2words import num2words

# num2words' code for each language numbers are written out in (Brazilian
# Portuguese, as Kokoro's Portuguese voices speak it)
NUM2WORDS = {"de": "de", "fr": "fr", "it": "it", "es": "es", "pt": "pt_BR", "nl": "nl", "ja": "ja"}
LANGUAGES = frozenset(NUM2WORDS)

MONTHS = {
    "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
           "November", "Dezember"],
    "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
           "novembre", "décembre"],
    "it": ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre",
           "ottobre", "novembre", "dicembre"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
           "noviembre", "diciembre"],
    "pt": ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
           "novembro", "dezembro"],
    "nl": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober",
           "november", "december"],
}
# The decimal comma, as said
COMMA = {"de": "Komma", "fr": "virgule", "it": "virgola", "es": "coma", "pt": "vírgula", "nl": "komma"}
PERCENT = {"de": "Prozent", "fr": "pour cent", "it": "per cento", "es": "por ciento", "pt": "por cento",
           "nl": "procent", "ja": "パーセント"}
# Degrees: one, more
DEGREES = {"de": ("Grad", "Grad"), "fr": ("degré", "degrés"), "it": ("grado", "gradi"), "es": ("grado", "grados"),
           "pt": ("grau", "graus"), "nl": ("graad", "graden"), "ja": ("度", "度")}
MINUS = {"de": "minus", "fr": "moins", "it": "meno", "es": "menos", "pt": "menos", "nl": "min", "ja": "マイナス"}
# Between the two ends of a range: "10–12" as "zehn bis zwölf"
RANGE = {"de": "bis", "fr": "à", "it": "a", "es": "a", "pt": "a", "nl": "tot", "ja": "から"}
# Money: the symbol's unit (one, more), and how cents follow the units
CURRENCIES = {
    "de": {"€": ("Euro", "Euro"), "$": ("Dollar", "Dollar"), "£": ("Pfund", "Pfund")},
    "fr": {"€": ("euro", "euros"), "$": ("dollar", "dollars"), "£": ("livre", "livres")},
    "it": {"€": ("euro", "euro"), "$": ("dollaro", "dollari"), "£": ("sterlina", "sterline")},
    "es": {"€": ("euro", "euros"), "$": ("dólar", "dólares"), "£": ("libra", "libras")},
    "pt": {"€": ("euro", "euros"), "$": ("dólar", "dólares"), "£": ("libra", "libras"), "R$": ("real", "reais")},
    "nl": {"€": ("euro", "euro"), "$": ("dollar", "dollar"), "£": ("pond", "pond")},
    "ja": {"€": ("ユーロ", "ユーロ"), "$": ("ドル", "ドル"), "£": ("ポンド", "ポンド"), "¥": ("円", "円")},
}
CENTS_JOIN = {"de": " ", "fr": " ", "it": " e ", "es": " con ", "pt": " e ", "nl": " "}

# A number as written in a language with a decimal comma: 1.000 or 1 000 a
# thousand, 3,5 three and a half (and 2.5, as models also write it)
NUMBER = r"\d{1,3}(?:[.\u00a0\u202f ]\d{3})+(?:,\d+)?(?!\d)|\d+,\d+|\d+\.\d{1,2}(?![\d.])|\d+"
# In Japanese: 1,000 a thousand, 3.5 three and a half
JA_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d+\.\d+|\d+"
# Where a number cannot be part of another
BEFORE = r"(?<![\d,.:/])"
AFTER = r"(?![\d:/]|[.,]\d)"


def numbers_in_words(text: str, language: str) -> str:
    """The text with its numbers as words in this language (one of
    LANGUAGES); other languages' text as it is."""
    if language not in LANGUAGES or not any(c.isdigit() for c in text):
        return text
    if language == "ja":
        return _japanese(text)
    return _spaced(text, language)


# Number words ------------------------------------------------------------------------------


def _words(value, language: str, to: str = "cardinal") -> str:
    said = num2words(value, lang=NUM2WORDS[language], to=to)
    # num2words writes Brazilian Portuguese thousands with a comma
    return said.replace(",", "") if language == "pt" else said


def _parts(written: str) -> tuple[int, str]:
    """A number as written with a decimal comma: its whole part, and the
    digits after the comma ("" if there are none). 1.000 is a thousand,
    and 2.5 two and a half, as models write it in these languages too."""
    if "," in written:
        whole, fraction = written.rsplit(",", 1)
    elif re.fullmatch(r"\d+\.\d{1,2}", written):
        whole, fraction = written.split(".")
    else:
        whole, fraction = written, ""
    return int(re.sub(r"\D", "", whole)), fraction


def _number(written: str, language: str) -> str:
    """A number, decimals read digit by digit after the comma."""
    whole, fraction = _parts(written)
    if not fraction:
        return _words(whole, language)
    digits = " ".join(_words(int(d), language) for d in fraction)
    return f"{_words(whole, language)} {COMMA[language]} {digits}"


def _before_noun(said: str, language: str, feminine: bool = False) -> str:
    """A number word as it is said before a noun: "ein Euro", not "eins
    Euro"; "una hora", "duas horas"."""
    if language == "de":
        return re.sub(r"eins$", "ein", said)
    if language == "es":
        said = re.sub(r"veintiuno$", "veintiuna" if feminine else "veintiún", said)
        return re.sub(r"\buno$", "una" if feminine else "un", said)
    if language == "it":
        return re.sub(r"\buno$", "una" if feminine else "un", said)
    if language == "fr" and feminine:
        return re.sub(r"\bun$", "une", said)
    if language == "pt" and feminine:
        return re.sub(r"\bdois$", "duas", re.sub(r"\bum$", "uma", said))
    return said


def _ordinal(day: int, language: str) -> str:
    return _words(day, language, to="ordinal")


# Languages with spaces ----------------------------------------------------------------------


def _spaced(text: str, language: str) -> str:
    months = MONTHS[language]
    month = "|".join(months + [m[:3] + r"\." for m in months])
    text = _money(text, language)
    text = _times(text, language)
    text = _dates(text, language, month)
    text = re.sub(rf"{BEFORE}({NUMBER})\s?%", lambda m: f"{_number(m[1], language)} {PERCENT[language]}", text)
    text = re.sub(rf"{BEFORE}([-−]?)({NUMBER})\s?°\s?[CF]?(?![\w])", lambda m: _degrees(m, language), text)
    text = _ordinals(text, language, month)
    # A range's dash is spaced on both sides or neither ("8 -9" is eight,
    # minus nine), and a chain of them is no range (2026-11-13)
    text = re.sub(rf"{BEFORE}({NUMBER})(?:[–-]|\s[–-]\s)({NUMBER}){AFTER}(?![–-]\d)",
                  lambda m: f"{_number(m[1], language)} {RANGE[language]} {_number(m[2], language)}", text)
    text = re.sub(rf"(?<![\w\d])[-−]({NUMBER})", lambda m: f"{MINUS[language]} {_number(m[1], language)}", text)
    if language in ("de", "nl"):
        text = re.sub(rf"{BEFORE}\b(1[1-9]\d\d)\b{AFTER}", lambda m: _words(int(m[1]), language, to="year"), text)
    # A number in a word ("LH402") is said apart from its letters
    text = re.sub(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", text)
    return re.sub(rf"{BEFORE}({NUMBER}){AFTER}", lambda m: _counted(m, language), text)


# Nouns after which a number is feminine ("une heure", "una hora")
FEMININE = {"fr": r"heures?|minutes?|secondes?|fois|semaines?|personnes?|nuits?",
            "es": r"horas?|semanas?|personas?|veces|noches?",
            "it": r"ore|ora|settimane?|persone?|volte|notti?",
            "pt": r"horas?|semanas?|pessoas?|vezes|noites?"}


def _counted(match: re.Match, language: str) -> str:
    """A number, in agreement with the noun it counts where that shows:
    "ein Ticket", "une heure", "duas horas"."""
    said = _number(match[1], language)
    following = match.string[match.end():]
    if language == "de" and re.match(r"\s+[A-ZÄÖÜ]", following):
        return _before_noun(said, language)
    feminine = FEMININE.get(language)
    if feminine and re.match(rf"\s+(?:{feminine})\b", following, re.IGNORECASE):
        return _before_noun(said, language, feminine=True)
    return said


# Unit words of money after a number: "12,50 euro" is twelve euros fifty
MONEY_WORDS = {"de": "Euro|Dollar|Pfund|Franken", "fr": "euros?|dollars?|livres?",
               "it": "euro|dollar[oi]|sterlin[ae]", "es": "euros?|dólar(?:es)?|libras?",
               "pt": "euros?|dólar(?:es)?|reais|real|libras?", "nl": "euro|dollar|pond"}


def _money(text: str, language: str) -> str:
    currencies = CURRENCIES[language]
    symbol = "|".join(re.escape(s) for s in sorted(currencies, key=len, reverse=True))

    def amount(written: str, one: str, more: str) -> str:
        units, fraction = _parts(written)
        cents = int(fraction[:2].ljust(2, "0")) if fraction else 0
        feminine = one in ("livre", "libra", "sterlina")
        out = f"{_before_noun(_words(units, language), language, feminine)} {one if units == 1 else more}"
        return out + (f"{CENTS_JOIN[language]}{_words(cents, language)}" if cents else "")

    text = re.sub(rf"({symbol})\s?({NUMBER}){AFTER}", lambda m: amount(m[2], *currencies[m[1]]), text)
    text = re.sub(rf"{BEFORE}({NUMBER})\s?({symbol})", lambda m: amount(m[1], *currencies[m[2]]), text)
    # With its unit in words, an amount with cents is read as one too
    return re.sub(rf"{BEFORE}(\d+(?:[.  ]\d{{3}})*,\d{{1,2}})\s+({MONEY_WORDS[language]})\b",
                  lambda m: amount(m[1], m[2], m[2]), text)


def _degrees(match: re.Match, language: str) -> str:
    one, more = DEGREES[language]
    said = _number(match[2], language)
    unit = one if said == _words(1, language) else more
    minus = f"{MINUS[language]} " if match[1] else ""
    return f"{minus}{said} {unit}"


def _hour(hour: int, language: str) -> str:
    return _before_noun(_words(hour, language), language, feminine=language in ("fr", "it", "es", "pt"))


def _time(hour: int, minute: int, language: str, hours_word: bool = False) -> str:
    """A time of day as the language says it."""
    h = _hour(hour, language)
    m = _words(minute, language) if minute else ""
    if language == "de":
        return f"{h} Uhr {m}".strip()
    if language == "nl":
        return f"{h} uur {m}".strip()
    if language == "fr":
        return f"{h} {'heure' if hour == 1 else 'heures'} {m}".strip()
    if language == "pt" and (hours_word or not minute):
        return f"{h} {'hora' if hour == 1 else 'horas'}" + (f" e {m}" if minute else "")
    joined = {"it": "e", "es": "y", "pt": "e"}[language]
    return f"{h} {joined} {m}" if minute else h


def _times(text: str, language: str) -> str:
    hour = r"([01]?\d|2[0-3])"
    # A word for o'clock after the time is part of it
    word = r"(?:\s?(?:Uhr|uur|h|heures?|horas?)\b)?"
    if language in ("de", "nl"):
        # "von 9.00 bis 17.00 Uhr": the first time has no word of its own
        text = re.sub(rf"{BEFORE}{hour}[.:]([0-5]\d)(\s*(?:[–-]|bis|tot)\s*){hour}[.:]([0-5]\d){AFTER}{word}",
                      lambda m: f"{_time(int(m[1]), int(m[2]), language)} {RANGE[language]} "
                                f"{_time(int(m[4]), int(m[5]), language)}", text)
    text = re.sub(rf"{BEFORE}{hour}:([0-5]\d){AFTER}{word}",
                  lambda m: _time(int(m[1]), int(m[2]), language), text)
    if language in ("de", "nl"):
        text = re.sub(rf"{BEFORE}{hour}\.([0-5]\d)\s?(?:Uhr|uur|h)\b",
                      lambda m: _time(int(m[1]), int(m[2]), language), text)
    if language in ("fr", "pt", "es", "it"):
        text = re.sub(rf"{BEFORE}{hour}\s?h([0-5]\d)?\b",
                      lambda m: _time(int(m[1]), int(m[2] or 0), language, hours_word=True), text)
    return text


def _day(day: int, language: str, case: str = "") -> str:
    """The day of a date: an ordinal in German (with its ending), the first
    as one in French, Italian and Portuguese, a cardinal otherwise."""
    if language == "de":
        return _ordinal(day, language) + case
    if day == 1 and language in ("fr", "it", "pt"):
        return _ordinal(day, language)
    return _words(day, language)


def _date(day: int, month: int, year: str, language: str, case: str = "") -> str:
    name = MONTHS[language][month - 1]
    said_year = ""
    if year:
        said_year = _words(int(year), language, to="year" if language in ("de", "nl") else "cardinal")
    of = " de " if language in ("es", "pt") else " "
    out = f"{_day(day, language, case)}{of}{name}"
    return out + (f"{of}{said_year}" if said_year else "")


# German puts an ordinal in the dative after these
DATIVE = r"(?:am|im|vom|zum|beim|zur|dem|den|des|seit dem|ab dem|bis zum)"


def _case(text: str, start: int) -> str:
    """The ending of a German ordinal by the word before it."""
    return "n" if re.search(rf"\b{DATIVE}\s+$", text[:start], re.IGNORECASE) else ""


def _dates(text: str, language: str, month: str) -> str:
    def numeric(match: re.Match) -> str:
        day, month_, year = int(match[1]), int(match[2]), match[3] or ""
        if not (1 <= day <= 31 and 1 <= month_ <= 12):
            return match[0]
        if len(year) == 2:
            year = ""
        case = _case(match.string, match.start()) if language == "de" else ""
        return _date(day, month_, year, language, case)

    if language == "de":
        text = re.sub(rf"{BEFORE}(\d{{1,2}})\.(\d{{1,2}})\.(\d{{4}}|\d{{2}})?(?![\d])", numeric, text)
        text = re.sub(rf"{BEFORE}(\d{{1,2}})/(\d{{1,2}})/(\d{{4}}|\d{{2}})(?![\d])", numeric, text)
        text = re.sub(rf"{BEFORE}(\d{{1,2}})\.\s?(?=(?:{month}))",
                      lambda m: _day(int(m[1]), "de", _case(m.string, m.start())) + " ", text)
    else:
        separator = "[/-]" if language == "nl" else "/"
        text = re.sub(rf"{BEFORE}(\d{{1,2}}){separator}(\d{{1,2}}){separator}(\d{{4}}|\d{{2}})(?![\d])", numeric, text)
        if language in ("fr", "it", "pt"):
            of = r"(?:de\s+)?" if language == "pt" else ""
            text = re.sub(rf"{BEFORE}\b1(?:º|°)?\s+(?={of}(?:{month}))", lambda m: _day(1, language) + " ", text)
    return text


def _ordinals(text: str, language: str, month: str) -> str:
    if language == "de":
        # After an article or a preposition, before a noun: "der 3. Platz"
        return re.sub(r"\b(der|die|das|dem|den|des|am|im|vom|zum|beim|zur)\s+(\d+)\.(?=\s+[A-ZÄÖÜ])",
                      lambda m: f"{m[1]} {_ordinal(int(m[2]), 'de')}{_case(m.string, m.start(2))}", text,
                      flags=re.IGNORECASE)
    if language == "fr":
        def french(match: re.Match) -> str:
            n, suffix = int(match[1]), match[2]
            if n == 1:
                return "première" if suffix == "re" else "premier"
            return _ordinal(n, "fr")
        return re.sub(r"\b(\d+)(er|re|ème|e|è)\b", french, text)
    if language == "nl":
        return re.sub(r"\b(\d+)(ste|de|e)\b", lambda m: _ordinal(int(m[1]), "nl"), text)

    # Spanish, Portuguese, Italian: 1.º, 1º, 1ª, 1°
    def southern(match: re.Match) -> str:
        said = _ordinal(int(match[1]), language)
        if match[2] == "ª":
            return re.sub(r"o\b", "a", said)
        # "el primer piso", "el tercer día"
        if language == "es" and re.match(r"\s+[^\W\d_]", match.string[match.end():]):
            return re.sub(r"\b(prim|terc)ero$", r"\1er", said)
        return said
    return re.sub(r"\b(\d+)\.?([ºª]|°(?=\s+[^\W\d_]))", southern, text)


# Japanese ----------------------------------------------------------------------------------


def _japanese(text: str) -> str:
    """Japanese numbers in kanji, which are then read in kana like any
    other kanji: 18度 as 十八度, 14:30 as 十四時三十分, ¥1,000 as 千円."""
    currencies = CURRENCIES["ja"]
    symbol = "|".join(re.escape(s) for s in currencies)

    def number(written: str) -> str:
        value = Decimal(written.replace(",", ""))
        return _words(int(value) if value == int(value) else float(value), "ja")

    text = re.sub(rf"({symbol})\s?({JA_NUMBER})", lambda m: number(m[2]) + currencies[m[1]][0], text)
    text = re.sub(rf"({JA_NUMBER})\s?({symbol})", lambda m: number(m[1]) + currencies[m[2]][0], text)
    text = re.sub(rf"({JA_NUMBER})\s?%", lambda m: number(m[1]) + PERCENT["ja"], text)
    text = re.sub(rf"([-−]?)({JA_NUMBER})\s?°\s?[CF]?",
                  lambda m: (MINUS["ja"] if m[1] else "") + number(m[2]) + DEGREES["ja"][0], text)
    text = re.sub(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)",
                  lambda m: number(m[1]) + "時" + (number(m[2]) + "分" if int(m[2]) else ""), text)
    text = re.sub(rf"(?<![\w\d])[-−]({JA_NUMBER})", lambda m: MINUS["ja"] + number(m[1]), text)
    return re.sub(JA_NUMBER, lambda m: number(m[0]), text)

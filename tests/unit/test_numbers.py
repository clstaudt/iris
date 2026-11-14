"""Unit tests for numbers written out as words, as each language says them:
the voices for other languages than English read no digits, or read them
with English rules."""

import pytest

from iris.numbers import LANGUAGES, numbers_in_words


@pytest.mark.parametrize(
    "language, text, said",
    [
        # Dates: German days are ordinals, in the dative after "am"
        ("de", "Der Markt öffnet am 27. November um 10 Uhr.",
         "Der Markt öffnet am siebenundzwanzigsten November um zehn Uhr."),
        ("de", "Heute ist der 1. Advent.", "Heute ist der erste Advent."),
        ("de", "Am 3.12.2026 um 9.30 Uhr.", "Am dritten Dezember zweitausendsechsundzwanzig um neun Uhr dreißig."),
        ("fr", "Le marché ouvre le 21 novembre à 11h30.",
         "Le marché ouvre le vingt et un novembre à onze heures trente."),
        ("fr", "C'est le 1er décembre, le 13/11/2026.",
         "C'est le premier décembre, le treize novembre deux mille vingt-six."),
        ("it", "Il 1° dicembre alle 9:30.", "Il primo dicembre alle nove e trenta."),
        ("es", "Sale el 21 de noviembre, el 13/11/2026.",
         "Sale el veintiuno de noviembre, el trece de noviembre de dos mil veintiséis."),
        ("pt", "No 1º de dezembro às 21:30.", "No primeiro de dezembro às vinte e uma e trinta."),
        ("nl", "Op 13-11-2026, de 1e keer.", "Op dertien november tweeduizendzesentwintig, de eerste keer."),
        # Times, and ranges of them
        ("de", "Wir treffen uns um 14:30 Uhr.", "Wir treffen uns um vierzehn Uhr dreißig."),
        ("de", "Geöffnet von 9.00 bis 17.00 Uhr.", "Geöffnet von neun Uhr bis siebzehn Uhr."),
        ("nl", "Open van 9.00 tot 17.00 uur.", "Open van negen uur tot zeventien uur."),
        ("fr", "À 21h, puis à 13:05.", "À vingt et une heures, puis à treize heures cinq."),
        ("es", "Sale a las 10:15.", "Sale a las diez y quince."),
        ("pt", "Abre das 10h às 18h.", "Abre das dez horas às dezoito horas."),
        # Money, with the decimal comma
        ("de", "Das Ticket kostet 23,50 €.", "Das Ticket kostet dreiundzwanzig Euro fünfzig."),
        ("de", "Der Preis: 1 €.", "Der Preis: ein Euro."),
        ("fr", "Le billet coûte 16,50 €.", "Le billet coûte seize euros cinquante."),
        ("it", "Il biglietto costa 12,50 euro.", "Il biglietto costa dodici euro e cinquanta."),
        ("es", "Cuesta 1 € o 23,50 €.", "Cuesta un euro o veintitrés euros con cincuenta."),
        ("pt", "Custa 15,90 € ou R$ 2,00.", "Custa quinze euros e noventa ou dois reais."),
        ("nl", "Een kaartje kost € 23,50.", "Een kaartje kost drieëntwintig euro vijftig."),
        # Decimals, thousands, years, percentages, degrees, ranges, minus
        ("de", "1.000 Leute, 2,5 km, 1990.", "eintausend Leute, zwei Komma fünf km, neunzehnhundertneunzig."),
        ("de", "Bei 20 % und -5 °C.", "Bei zwanzig Prozent und minus fünf Grad."),
        ("de", "Geöffnet 10–12 Uhr.", "Geöffnet zehn bis zwölf Uhr."),
        ("fr", "Il fait 18 °C au 2e étage.", "Il fait dix-huit degrés au deuxième étage."),
        ("it", "Ci saranno 21 gradi e 3,75 km.", "Ci saranno ventuno gradi e tre virgola sette cinque km."),
        ("pt", "Desde 1960.", "Desde mil novecentos e sessenta."),
        ("nl", "Het is 12 graden, sinds 1990.", "Het is twaalf graden, sinds negentien honderd negentig."),
        # A number agrees with the noun it counts where that shows
        ("de", "1 Ticket, 21 Tickets, Seite 1 ist gut.", "ein Ticket, einundzwanzig Tickets, Seite eins ist gut."),
        ("fr", "Pendant 1 heure.", "Pendant une heure."),
        ("es", "El 1.º piso, 1 hora, 21 horas.", "El primer piso, una hora, veintiuna horas."),
        ("pt", "Por 2 horas.", "Por duas horas."),
        # A number in a word is said apart from its letters
        ("de", "Flug LH402.", "Flug LH vierhundertzwei."),
        # Japanese numbers in kanji, read in kana like other kanji
        ("ja", "明日は晴れ、18度。", "明日は晴れ、十八度。"),
        ("ja", "14:30に¥1,000で、気温は-2°C、確率は20%です。",
         "十四時三十分に千円で、気温はマイナス二度、確率は二十パーセントです。"),
        ("ja", "2026年11月13日", "二千二十六年十一月十三日"),
    ],
)  # fmt: skip
def test_numbers_are_said_as_the_language_says_them(language, text, said):
    assert numbers_in_words(text, language) == said


def test_text_without_numbers_and_other_languages_are_left_alone():
    assert numbers_in_words("Guten Tag.", "de") == "Guten Tag."
    # English and Hindi voices read their digits right
    assert numbers_in_words("It costs $23.50 at 10:15.", "en") == "It costs $23.50 at 10:15."
    assert numbers_in_words("तापमान 18 डिग्री", "hi") == "तापमान 18 डिग्री"
    assert LANGUAGES == {"de", "fr", "it", "es", "pt", "nl", "ja"}


@pytest.mark.parametrize("language", sorted(LANGUAGES))
def test_no_digit_is_left(language):
    text = "1 2,5 3.000 4:30 5 % 6 °C 7–8 -9 10. 11/12/2026 13h 14ª 15º 16° 17e 18er €19,20 21 € X22 2026"
    said = numbers_in_words(text, language)
    assert not any(c.isdigit() for c in said), said

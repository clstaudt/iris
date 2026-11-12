"""Unit tests for preparing reply text for speech: lists must be read with
a pause after each item, not run together as one sentence."""

from iris.llm import (
    PIECE_GROWTH,
    PIECE_SENTENCE_FROM,
    PIECE_SENTENCE_TO,
    ConversationManager,
    speech_break,
    split_speech,
)

clean = ConversationManager.__new__(ConversationManager).clean_markdown


class TestListPauses:
    def test_bullet_items_end_in_a_pause(self):
        text = clean("Some options:\n\n- pasta\n- risotto\n- soup")
        assert text == "Some options:\n\npasta.\nrisotto.\nsoup."

    def test_list_right_under_its_intro(self):
        text = clean("Steps:\n1. Boil water\n2. Add the tea")
        assert text == "Steps:\n\nBoil water.\nAdd the tea."

    def test_existing_punctuation_kept(self):
        text = clean("- Is it ripe?\n- Yes, it is.\n- He said \"go!\"")
        assert text == "Is it ripe?\nYes, it is.\nHe said \"go!\""

    def test_heading_ends_in_a_pause(self):
        assert clean("# Weather\nSunny all day") == "Weather.\nSunny all day."

    def test_plain_sentence_unchanged(self):
        assert clean("It's sunny today.") == "It's sunny today."

    def test_emoji_only_line_dropped(self):
        assert clean("Done\n\n😀") == "Done."


class TestSpeechBreak:
    def test_cuts_after_last_sentence(self):
        text = "One. Two. Three"
        assert text[: speech_break(text, 0, len(text))] == "One. Two."

    def test_cuts_at_line_break(self):
        text = "First item\nSecond item"
        assert text[: speech_break(text, 0, len(text))] == "First item\n"

    def test_not_after_a_list_number(self):
        text = "Steps:\n1. Boil water\n2. Add the tea"
        assert text[: speech_break(text, 0, len(text))] == "Steps:\n1. Boil water\n"

    def test_none_before_start(self):
        assert speech_break("One. Two", 5, 8) == -1


def pieces(stream: list[str]) -> list[str]:
    """The pieces a reply streamed in these chunks is spoken in, as chat() cuts them."""
    spoken, buffer = [], ""
    for chunk in stream:
        buffer += chunk
        while True:
            piece, buffer = split_speech(buffer, len(spoken[-1]) if spoken else 0)
            if not piece:
                break
            spoken.append(piece)
    return spoken + ([buffer.strip()] if buffer.strip() else [])


def words(text: str) -> list[str]:
    """Text streamed word by word, the way tokens arrive."""
    return [word + " " for word in text.split(" ")]


LONG = (
    "The capital of France is Paris. It has been the capital for centuries, "
    "and it is home to more than two million people. The city is known for "
    "its museums, its cafes and its architecture. Many visitors come for the "
    "Louvre, which is the largest art museum in the world. Others come for "
    "the food, which is famous across Europe."
)


class TestSplitSpeech:
    def test_first_sentence_is_spoken_at_once(self):
        assert pieces(words(LONG))[0] == "The capital of France is Paris."

    def test_pieces_grow_to_the_usual_size(self):
        spoken = pieces(words(LONG * 3))
        for before, piece in zip(spoken, spoken[1:-1]):
            assert len(piece) <= PIECE_GROWTH * len(before) + 1
        # ...and then stay the usual size
        assert all(len(piece) > PIECE_SENTENCE_FROM for piece in spoken[2:-1])
        assert all(len(piece) <= PIECE_SENTENCE_TO for piece in spoken)

    def test_nothing_is_lost_or_reordered(self):
        assert " ".join(pieces(words(LONG))) == LONG

    def test_a_short_opening_joins_the_next_sentence(self):
        reply = "Sure. The capital of France is Paris. It is large."
        assert pieces(words(reply))[0] == "Sure. The capital of France is Paris."

    def test_first_sentence_also_when_text_arrives_at_once(self):
        # Tool rounds can deliver a reply in one chunk
        assert pieces([LONG])[0] == "The capital of France is Paris."

    def test_waits_for_the_sentence_to_end(self):
        assert split_speech("The capital of France is Paris", 0) == (
            "",
            "The capital of France is Paris",
        )

    def test_a_paragraph_is_a_piece(self):
        assert split_speech("Hi!\n\nThe capital", 0) == ("Hi!", "The capital")

    def test_blank_paragraphs_are_skipped(self):
        assert split_speech("\n\n\n\nHi!\n\nMore", 100) == ("Hi!", "More")

    def test_list_intro_is_the_first_piece(self):
        reply = "Here are some ideas for you:\n1. Pasta\n2. Risotto\n"
        assert pieces(words(reply))[0] == "Here are some ideas for you:"

    def test_one_long_sentence_ends_where_it_ends(self):
        sentence = "This sentence goes on " + "and on " * 50 + "until it stops. Next one."
        spoken = pieces(words("A first short sentence here. " + sentence))
        assert spoken[1] == sentence[: -len(" Next one.")]


def characters(text: str, size: int = 2) -> list[str]:
    """Text streamed a few characters at a time, the way tokens of a language
    without spaces arrive."""
    return [text[i : i + size] for i in range(0, len(text), size)]


JAPANESE = (
    "東京の明日の天気は晴れで、最高気温は十八度の予報です。"
    "朝は少し冷えるので、上着を持って出かけるといいでしょう。"
    "午後からは風が強くなり、夕方には北の風が吹く見込みです。"
    "週末にかけては雲が増えて、日曜日には雨が降るかもしれません。"
)
CHINESE = (
    "北京明天是晴天，最高气温十八度左右，适合出门散步。"
    "早上比较冷，出门的时候最好带一件外套！"
    "你想知道周末的天气怎么样吗？"
    "周末可能会下雨，记得带伞。"
)
HINDI = (
    "कल दिल्ली में मौसम साफ़ रहेगा और तापमान अठारह डिग्री होगा। "
    "सुबह थोड़ी ठंड होगी, इसलिए जैकेट साथ रखें। "
    "शाम को हवा तेज़ चलेगी।"
)


class TestSentencesWithoutSpaces:
    """Japanese and Chinese end a sentence with 。！？ and go on without a
    space: a reply must still be spoken from its first sentence, not once it
    has arrived whole."""

    def test_japanese_is_spoken_from_its_first_sentence(self):
        spoken = pieces(characters(JAPANESE))
        assert spoken[0] == "東京の明日の天気は晴れで、最高気温は十八度の予報です。"
        assert "".join(spoken) == JAPANESE

    def test_chinese_ends_sentences_at_full_stops_exclamations_and_questions(self):
        cut = speech_break(CHINESE, 0, len(CHINESE), first=True)
        assert CHINESE[:cut] == "北京明天是晴天，最高气温十八度左右，适合出门散步。"
        spoken = pieces(characters(CHINESE))
        assert spoken[0] == CHINESE[:cut]
        assert "".join(spoken) == CHINESE
        assert all(piece[-1] in "。！？" for piece in spoken)

    def test_a_closing_quote_stays_with_its_sentence(self):
        text = "彼は「明日また来ます。」と言いました。それから帰りました。"
        cut = speech_break(text, 0, len(text), first=True)
        assert text[:cut] == "彼は「明日また来ます。」"

    def test_a_comma_is_no_sentence_end(self):
        assert split_speech("東京の天気は晴れで、最高気温は十八度の", 0) == (
            "",
            "東京の天気は晴れで、最高気温は十八度の",
        )

    def test_hindi_ends_sentences_at_the_danda(self):
        spoken = pieces(words(HINDI))
        assert spoken[0] == "कल दिल्ली में मौसम साफ़ रहेगा और तापमान अठारह डिग्री होगा।"
        assert " ".join(spoken) == HINDI

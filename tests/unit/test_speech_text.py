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

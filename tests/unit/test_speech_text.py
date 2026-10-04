"""Unit tests for preparing reply text for speech: lists must be read with
a pause after each item, not run together as one sentence."""

from iris.llm import ConversationManager, speech_break

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

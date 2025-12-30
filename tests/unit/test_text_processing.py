"""Unit tests for text processing utilities."""

import re
import markdown
from bs4 import BeautifulSoup


# Extract pure functions for testing (same logic as ConversationManager)
def estimate_tokens(text: str) -> int:
    """~4 characters per token heuristic."""
    return len(text) // 4


def clean_markdown(text: str) -> str:
    """Remove markdown and emojis for TTS."""
    html = markdown.markdown(text)
    clean_text = BeautifulSoup(html, "html.parser").get_text()

    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", clean_text).strip()


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        assert estimate_tokens("hello") == 1  # 5 chars

    def test_longer_text(self):
        assert estimate_tokens("a" * 100) == 25


class TestCleanMarkdown:
    def test_bold(self):
        assert clean_markdown("**bold**") == "bold"

    def test_headers(self):
        assert clean_markdown("# Header") == "Header"

    def test_links(self):
        assert clean_markdown("[text](http://example.com)") == "text"

    def test_emojis_removed(self):
        assert clean_markdown("Hello 😀 World") == "Hello  World"

    def test_plain_text_unchanged(self):
        assert clean_markdown("plain text") == "plain text"

    def test_combined(self):
        result = clean_markdown("**Hello** 🎉 [world](http://x.com)")
        assert "Hello" in result
        assert "world" in result
        assert "🎉" not in result

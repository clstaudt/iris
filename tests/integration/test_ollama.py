"""Integration tests for Ollama LLM."""

import pytest
from iris.llm import ConversationManager


@pytest.fixture
def conversation():
    """Create ConversationManager - requires running Ollama."""
    return ConversationManager("qwen3:1.7b")


@pytest.mark.integration
class TestConversationManager:
    def test_connection(self, conversation):
        """Ollama connects and detects context length."""
        assert conversation.max_context_tokens > 0

    def test_chat_stream(self, conversation):
        """Streaming chat returns response chunks."""
        chunks = list(conversation.chat_stream("Say hello in one word."))
        assert len(chunks) > 0
        response = "".join(chunks)
        assert len(response) > 0

    def test_clean_markdown_method(self, conversation):
        """clean_markdown works on instance."""
        assert conversation.clean_markdown("**bold**") == "bold"

    def test_context_management(self, conversation):
        """Context includes system prompt."""
        messages = conversation.manage_context()
        assert any("Iris" in m.get("content", "") for m in messages)





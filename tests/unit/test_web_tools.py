"""Unit tests for web search and browse tools."""

import threading
from unittest.mock import ANY, MagicMock, patch

import pytest

from iris.tools import Tools


class FakeStatusManager:
    """Minimal stand-in for StatusManager used by tools."""

    def __init__(self):
        self.status_lock = threading.Lock()
        self.status_searching = False
        self.status_browsing = False


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestEnableWebSearch:
    def test_registers_search_and_browse(self):
        tools = Tools()
        assert tools.enable_web_search() is True
        assert "search" in tools._enabled_tools
        assert "browse" in tools._enabled_tools

    def test_has_tools_after_enable(self):
        tools = Tools()
        tools.enable_web_search()
        assert tools.has_tools() is True

    def test_schemas_include_both_tools(self):
        tools = Tools()
        tools.enable_web_search()
        schemas = tools.get_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert names == {"search", "browse"}

    def test_search_schema_requires_query(self):
        tools = Tools()
        tools.enable_web_search()
        search_schema = tools._enabled_tools["search"]["schema"]["function"]
        assert "query" in search_schema["parameters"]["required"]

    def test_browse_schema_requires_url(self):
        tools = Tools()
        tools.enable_web_search()
        browse_schema = tools._enabled_tools["browse"]["schema"]["function"]
        assert "url" in browse_schema["parameters"]["required"]

    def test_creates_trafilatura_config_with_browser_ua(self):
        tools = Tools()
        tools.enable_web_search()
        ua = tools._traf_config.get("DEFAULT", "USER_AGENTS")
        assert "Mozilla" in ua

    def test_execute_unknown_tool(self):
        tools = Tools()
        result = tools.execute("nonexistent")
        assert result["success"] is False
        assert "Unknown tool" in result["message"]


# ---------------------------------------------------------------------------
# _search
# ---------------------------------------------------------------------------


class TestSearch:
    @patch("iris.tools.DDGS")
    def test_returns_formatted_results(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.return_value = [
            {"title": "Result 1", "body": "Snippet one", "href": "https://one.com"},
            {"title": "Result 2", "body": "Snippet two", "href": "https://two.com"},
        ]

        tools = Tools()
        tools.enable_web_search()
        result = tools.execute("search", query="test query")

        assert result["success"] is True
        assert "Result 1" in result["result"]
        assert "Result 2" in result["result"]
        assert "https://one.com" in result["result"]
        mock_ddgs_cls.return_value.text.assert_called_once_with("test query", max_results=5)

    @patch("iris.tools.DDGS")
    def test_empty_results(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.return_value = []

        tools = Tools()
        tools.enable_web_search()
        result = tools.execute("search", query="obscure query")

        assert result["success"] is True
        assert "No results found" in result["result"]

    @patch("iris.tools.DDGS")
    def test_exception_returns_failure(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.side_effect = RuntimeError("network down")

        tools = Tools()
        tools.enable_web_search()
        result = tools.execute("search", query="anything")

        assert result["success"] is False
        assert "network down" in result["message"]

    @patch("iris.tools.DDGS")
    def test_status_manager_cleared_after_search(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.return_value = []
        sm = FakeStatusManager()

        tools = Tools(status_manager=sm)
        tools.enable_web_search()

        assert sm.status_searching is False
        tools.execute("search", query="q")
        assert sm.status_searching is False

    @patch("iris.tools.DDGS")
    def test_status_cleared_on_error(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.side_effect = RuntimeError("boom")
        sm = FakeStatusManager()

        tools = Tools(status_manager=sm)
        tools.enable_web_search()
        tools.execute("search", query="q")

        assert sm.status_searching is False


# ---------------------------------------------------------------------------
# _browse
# ---------------------------------------------------------------------------


class TestBrowse:
    @patch("iris.tools.trafilatura")
    def test_returns_extracted_text(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html><body>Hello world</body></html>"
        mock_traf.extract.return_value = "Hello world"

        tools = Tools()
        tools.enable_web_search()
        result = tools.execute("browse", url="https://example.com")

        assert result["success"] is True
        assert result["result"] == "Hello world"
        mock_traf.fetch_url.assert_called_once_with("https://example.com", config=ANY)

    @patch("iris.tools.trafilatura")
    def test_prepends_https_if_missing(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = "content"

        tools = Tools()
        tools.enable_web_search()
        tools.execute("browse", url="example.com")

        mock_traf.fetch_url.assert_called_once_with("https://example.com", config=ANY)

    @patch("iris.tools.trafilatura")
    def test_preserves_explicit_http(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = "content"

        tools = Tools()
        tools.enable_web_search()
        tools.execute("browse", url="http://example.com")

        mock_traf.fetch_url.assert_called_once_with("http://example.com", config=ANY)

    @patch("iris.tools.trafilatura")
    def test_fetch_failure(self, mock_traf):
        mock_traf.fetch_url.return_value = None

        tools = Tools()
        tools.enable_web_search()
        result = tools.execute("browse", url="https://bad.url")

        assert result["success"] is False
        assert "Could not fetch" in result["message"]

    @patch("iris.tools.trafilatura")
    def test_no_content_extracted(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = ""

        tools = Tools()
        tools.enable_web_search()
        result = tools.execute("browse", url="https://empty.com")

        assert result["success"] is False
        assert "No content" in result["message"]

    @patch("iris.tools.trafilatura")
    def test_long_content_truncated(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = "x" * 5000

        tools = Tools()
        tools.enable_web_search()
        result = tools.execute("browse", url="https://long.com")

        assert result["success"] is True
        assert len(result["result"]) < 5000
        assert result["result"].endswith("...(truncated)")

    @patch("iris.tools.trafilatura")
    def test_short_content_not_truncated(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = "Short page."

        tools = Tools()
        tools.enable_web_search()
        result = tools.execute("browse", url="https://short.com")

        assert result["result"] == "Short page."
        assert "truncated" not in result["result"]

    @patch("iris.tools.trafilatura")
    def test_exception_returns_failure(self, mock_traf):
        mock_traf.fetch_url.side_effect = ConnectionError("timeout")

        tools = Tools()
        tools.enable_web_search()
        result = tools.execute("browse", url="https://fail.com")

        assert result["success"] is False
        assert "timeout" in result["message"]

    @patch("iris.tools.trafilatura")
    def test_status_manager_cleared_after_browse(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = "ok"
        sm = FakeStatusManager()

        tools = Tools(status_manager=sm)
        tools.enable_web_search()

        assert sm.status_browsing is False
        tools.execute("browse", url="https://example.com")
        assert sm.status_browsing is False

    @patch("iris.tools.trafilatura")
    def test_status_cleared_on_error(self, mock_traf):
        mock_traf.fetch_url.side_effect = RuntimeError("boom")
        sm = FakeStatusManager()

        tools = Tools(status_manager=sm)
        tools.enable_web_search()
        tools.execute("browse", url="https://x.com")

        assert sm.status_browsing is False


# ---------------------------------------------------------------------------
# LLM system prompt integration
# ---------------------------------------------------------------------------


class TestLLMSearchPrompt:
    """Verify manage_context injects search/browse instructions."""

    @patch("iris.llm.ollama")
    def test_system_prompt_includes_search_instructions(self, mock_ollama):
        mock_ollama.show.return_value = MagicMock(
            modelinfo={}, details=MagicMock(family="test")
        )
        mock_ollama.list.return_value = []

        from iris.llm import ConversationManager

        tools = Tools()
        tools.enable_web_search()

        cm = ConversationManager(model="test", tools=tools)
        messages = cm.manage_context()
        system = messages[0]["content"]

        assert "'search' function" in system
        assert "'browse' function" in system

    @patch("iris.llm.ollama")
    def test_system_prompt_omits_search_without_tools(self, mock_ollama):
        mock_ollama.show.return_value = MagicMock(
            modelinfo={}, details=MagicMock(family="test")
        )
        mock_ollama.list.return_value = []

        from iris.llm import ConversationManager

        cm = ConversationManager(model="test", tools=None)
        messages = cm.manage_context()
        system = messages[0]["content"]

        assert "'search' function" not in system
        assert "'browse' function" not in system

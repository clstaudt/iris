"""Unit tests for web search and browse tools."""

import threading
from unittest.mock import ANY, MagicMock, patch

import pytest
from ddgs.exceptions import DDGSException, TimeoutException

from iris.plugins import web
from iris.tools import Tools


def _web(status_manager=None):
    """The tools with the web plugin switched on."""
    return Tools([web.plugin], status_manager=status_manager)


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
        tools = _web()
        assert tools.names() == ["search", "browse"]

    def test_has_tools_after_enable(self):
        tools = _web()
        assert tools.has_tools() is True

    def test_schemas_include_both_tools(self):
        tools = _web()
        schemas = tools.get_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert names == {"search", "browse"}

    def test_search_schema_requires_query(self):
        tools = _web()
        search_schema = tools.tool("search").schema()["function"]
        assert "query" in search_schema["parameters"]["required"]

    def test_browse_schema_requires_url(self):
        tools = _web()
        browse_schema = tools.tool("browse").schema()["function"]
        assert "url" in browse_schema["parameters"]["required"]

    def test_creates_trafilatura_config_with_browser_ua(self):
        ua = web.Pages().config.get("DEFAULT", "USER_AGENTS")
        assert "Mozilla" in ua

    def test_execute_unknown_tool(self):
        tools = Tools([])
        result = tools.execute("nonexistent")
        assert result["success"] is False
        assert "Unknown tool" in result["message"]


# ---------------------------------------------------------------------------
# _search
# ---------------------------------------------------------------------------


class TestSearch:
    @patch("iris.plugins.web.DDGS")
    def test_returns_formatted_results(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.return_value = [
            {"title": "Result 1", "body": "Snippet one", "href": "https://one.com"},
            {"title": "Result 2", "body": "Snippet two", "href": "https://two.com"},
        ]

        tools = _web()
        result = tools.execute("search", query="test query")

        assert result["success"] is True
        assert "Result 1" in result["result"]
        assert "Result 2" in result["result"]
        assert "https://one.com" in result["result"]
        mock_ddgs_cls.return_value.text.assert_called_once_with(
            "test query", max_results=5, backend="yahoo"
        )

    @patch("iris.plugins.web.DDGS")
    def test_empty_results(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.return_value = []

        tools = _web()
        result = tools.execute("search", query="obscure query")

        assert result["success"] is True
        assert "No results found" in result["result"]

    @patch("iris.plugins.web.DDGS")
    def test_exception_returns_failure(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.side_effect = RuntimeError("network down")

        tools = _web()
        result = tools.execute("search", query="anything")

        assert result["success"] is False
        assert "network down" in result["message"]

    @patch("iris.plugins.web.DDGS")
    def test_status_manager_cleared_after_search(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.return_value = []
        sm = FakeStatusManager()

        tools = _web(sm)

        assert sm.status_searching is False
        tools.execute("search", query="q")
        assert sm.status_searching is False

    @patch("iris.plugins.web.DDGS")
    def test_status_cleared_on_error(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.side_effect = RuntimeError("boom")
        sm = FakeStatusManager()

        tools = _web(sm)
        tools.execute("search", query="q")

        assert sm.status_searching is False


# ---------------------------------------------------------------------------
# _browse
# ---------------------------------------------------------------------------


class TestBrowse:
    @patch("iris.plugins.web.trafilatura")
    def test_returns_extracted_text(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html><body>Hello world</body></html>"
        mock_traf.extract.return_value = "Hello world"

        tools = _web()
        result = tools.execute("browse", url="https://example.com")

        assert result["success"] is True
        assert result["result"] == "Hello world"
        mock_traf.fetch_url.assert_called_once_with("https://example.com", config=ANY)

    @patch("iris.plugins.web.trafilatura")
    def test_prepends_https_if_missing(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = "content"

        tools = _web()
        tools.execute("browse", url="example.com")

        mock_traf.fetch_url.assert_called_once_with("https://example.com", config=ANY)

    @patch("iris.plugins.web.trafilatura")
    def test_preserves_explicit_http(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = "content"

        tools = _web()
        tools.execute("browse", url="http://example.com")

        mock_traf.fetch_url.assert_called_once_with("http://example.com", config=ANY)

    @patch("iris.plugins.web.trafilatura")
    def test_fetch_failure(self, mock_traf):
        mock_traf.fetch_url.return_value = None

        tools = _web()
        result = tools.execute("browse", url="https://bad.url")

        assert result["success"] is False
        assert "Could not fetch" in result["message"]

    @patch("iris.plugins.web.trafilatura")
    def test_no_content_extracted(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = ""

        tools = _web()
        result = tools.execute("browse", url="https://empty.com")

        assert result["success"] is False
        assert "No content" in result["message"]

    @patch("iris.plugins.web.trafilatura")
    def test_long_content_truncated(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = "x" * 5000

        tools = _web()
        result = tools.execute("browse", url="https://long.com")

        assert result["success"] is True
        assert len(result["result"]) < 5000
        assert result["result"].endswith("...(truncated)")

    @patch("iris.plugins.web.trafilatura")
    def test_short_content_not_truncated(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = "Short page."

        tools = _web()
        result = tools.execute("browse", url="https://short.com")

        assert result["result"] == "Short page."
        assert "truncated" not in result["result"]

    @patch("iris.plugins.web.trafilatura")
    def test_exception_returns_failure(self, mock_traf):
        mock_traf.fetch_url.side_effect = ConnectionError("timeout")

        tools = _web()
        result = tools.execute("browse", url="https://fail.com")

        assert result["success"] is False
        assert "timeout" in result["message"]

    @patch("iris.plugins.web.trafilatura")
    def test_status_manager_cleared_after_browse(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = "ok"
        sm = FakeStatusManager()

        tools = _web(sm)

        assert sm.status_browsing is False
        tools.execute("browse", url="https://example.com")
        assert sm.status_browsing is False

    @patch("iris.plugins.web.trafilatura")
    def test_status_cleared_on_error(self, mock_traf):
        mock_traf.fetch_url.side_effect = RuntimeError("boom")
        sm = FakeStatusManager()

        tools = _web(sm)
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

        tools = _web()

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


class TestSearchTriedAgain:
    @patch("iris.plugins.web.DDGS")
    def test_engines_that_refused_are_asked_again(self, mock_ddgs_cls):
        """ddgs says "No results found." when the engines it asked refused,
        or answered in a page it cannot read; asked again, they answer."""
        mock_ddgs_cls.return_value.text.side_effect = [
            DDGSException("No results found."),
            [{"title": "Ali G Indahouse - Quotes", "body": "Keep it real", "href": "https://q.example"}],
        ]
        result = _web().execute("search", query="Ali G Indahouse quotes")
        assert result["success"] is True
        assert "Keep it real" in result["result"]

    @patch("iris.plugins.web.DDGS")
    def test_yahoo_six_times_then_every_engine(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.side_effect = DDGSException("No results found.")
        result = _web().execute("search", query="something nobody wrote")
        backends = [call.kwargs["backend"] for call in mock_ddgs_cls.return_value.text.call_args_list]
        assert backends == ["yahoo"] * 6 + ["auto"]
        # Nothing found is said as such, not as a broken service
        assert result["success"] is True
        assert "No results found" in result["result"]

    @patch("iris.plugins.web.DDGS")
    def test_each_try_asks_as_another_browser(self, mock_ddgs_cls):
        """ddgs picks the browser it asks as when it is made: a fresh one per
        try, so a page for a phone that it cannot read is not asked for again."""
        mock_ddgs_cls.return_value.text.side_effect = [DDGSException("No results found.")] * 2 + [
            [{"title": "Reykjavik", "body": "About 140,000 people", "href": "https://r.example"}]
        ]
        assert "140,000" in _web().execute("search", query="population of Reykjavik")["result"]
        assert mock_ddgs_cls.call_count == 3

    @patch("iris.plugins.web.DDGS")
    def test_a_network_failure_is_tried_again_then_reported(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.side_effect = [RuntimeError("network down")] * 2 + [
            [{"title": "Back", "body": "up", "href": "https://b.example"}]
        ]
        assert _web().execute("search", query="anything")["success"] is True

    @pytest.mark.parametrize("error", [
        DDGSException("ConnectError: Connection refused"),
        TimeoutException("operation timed out"),
    ])
    @patch("iris.plugins.web.DDGS")
    def test_engines_out_of_reach_are_a_failure_not_nothing_found(self, mock_ddgs_cls, error):
        """ddgs raises its own exception when it cannot reach the engines
        or they time out: the search could not be made, which the model
        says, rather than that nothing was found."""
        mock_ddgs_cls.return_value.text.side_effect = error
        result = _web().execute("search", query="weather in Lisbon")
        assert result["success"] is False
        assert str(error) in result["message"]
        assert mock_ddgs_cls.return_value.text.call_count == len(web.SEARCH_TRIES)

    @patch("iris.plugins.web.DDGS")
    def test_nothing_found_by_engines_that_answered_is_nothing_found(self, mock_ddgs_cls):
        mock_ddgs_cls.return_value.text.side_effect = [
            DDGSException("ConnectError: Connection refused"),
            DDGSException("No results found."),
            TimeoutException("operation timed out"),
            [],
        ] + [DDGSException("ConnectError: Connection refused")] * 3
        result = _web().execute("search", query="something nobody wrote")
        assert result["success"] is True
        assert result["result"] == "No results found."

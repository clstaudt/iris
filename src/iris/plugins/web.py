"""The web: Iris searches it for current facts and reads pages."""

import logging

from iris.plugin import Plugin, PluginContext, Tool

try:
    import trafilatura
    from ddgs import DDGS
    from trafilatura.settings import use_config

    HAS_SEARCH = True
except ImportError:
    HAS_SEARCH = False

logger = logging.getLogger(__name__)

# Sites answer a browser better than a script
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# A page is cut to this many characters for the model
PAGE_CHARS = 4000

SEARCH_GUIDANCE = (
    "- 'search' function: Search the web for CURRENT information.\n"
    "  * YOU MUST call this for questions about current events, news, weather, prices, or recent facts\n"
    "  * Use when the user asks about something you're unsure of or that changes over time\n"
    "  * NEVER guess at current facts - search first\n"
)
BROWSE_GUIDANCE = (
    "- 'browse' function: Fetch and read the contents of a specific webpage.\n"
    "  * When the user names a website (e.g. 'The Guardian', 'BBC News'), infer its URL and call this\n"
    "  * Use to read, scan, or check any web page, article, or headline\n"
    "  * Use to read full article content from URLs found via 'search'\n"
    "  * Provide the full URL including https://\n"
)


def search(query: str) -> dict:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query string

    Returns:
        Dict with 'success', 'result', and 'message' keys
    """
    logger.info("Searching: %s", query)
    try:
        results = DDGS().text(query, max_results=5)

        if not results:
            logger.warning("Search returned no results for: %s", query)
            return {"success": True, "result": "No results found.", "message": "No results"}

        logger.info("Search got %d results for: %s", len(results), query)
        formatted = [
            f"- {r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})" for r in results
        ]
        return {
            "success": True,
            "result": "\n".join(formatted),
            "message": f"Search results for: {query}",
        }

    except Exception as e:
        logger.error("Search failed for '%s': %s", query, e, exc_info=True)
        return {"success": False, "result": "Search failed", "message": f"Error: {e}"}


class Pages:
    """Reads web pages as text, with trafilatura set up like a browser."""

    def __init__(self):
        self.config = use_config()
        self.config.set("DEFAULT", "USER_AGENTS", USER_AGENT)

    def browse(self, url: str) -> dict:
        """
        Fetch and extract readable text from a webpage.

        Args:
            url: URL to fetch

        Returns:
            Dict with 'success', 'result', and 'message' keys
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        logger.info("Browsing: %s", url)
        try:
            downloaded = trafilatura.fetch_url(url, config=self.config)
            if downloaded is None:
                logger.warning("Browse could not fetch: %s", url)
                return {"success": False, "result": "", "message": f"Could not fetch {url}"}

            text = trafilatura.extract(
                downloaded,
                include_links=True,
                include_tables=True,
                output_format="txt",
                config=self.config,
            )
            if not text:
                logger.warning("Browse fetched but no content extracted: %s", url)
                return {"success": False, "result": "", "message": "No content extracted"}

            logger.info("Browse got %d chars from %s", len(text), url)
            if len(text) > PAGE_CHARS:
                text = text[:PAGE_CHARS] + "\n...(truncated)"
            return {"success": True, "result": text, "message": f"Content from: {url}"}

        except Exception as e:
            logger.error("Browse failed for '%s': %s", url, e, exc_info=True)
            return {"success": False, "result": "Browse failed", "message": f"Error: {e}"}


class Web(Plugin):
    id = "web"
    name = "Web search"

    def available(self) -> bool:
        return HAS_SEARCH

    def tools(self, context: PluginContext) -> list[Tool]:
        pages = Pages()
        return [
            Tool(
                name="search",
                description=(
                    "Search the web for current information. Use when you need up-to-date "
                    "facts, news, or information you don't know."
                ),
                run=search,
                parameters={"query": {"type": "string", "description": "Search query"}},
                required=("query",),
                guidance=SEARCH_GUIDANCE,
                activity="searching",
                label="Searched the web",
            ),
            Tool(
                name="browse",
                description=(
                    "Fetch and read the contents of a webpage."
                    " Use when the user mentions a specific website"
                    " or you need to read a URL from search results."
                ),
                run=pages.browse,
                parameters={
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch (e.g. https://www.theguardian.com)",
                    }
                },
                required=("url",),
                guidance=BROWSE_GUIDANCE,
                activity="browsing",
                label="Read a web page",
            ),
        ]


plugin = Web()

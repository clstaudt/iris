"""The web: Iris searches it for current facts and reads pages."""

import logging

from iris.plugin import Plugin, PluginContext, Tool

try:
    import trafilatura
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException
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


# Tries of a search, each with a fresh DDGS, until one finds something.
# Measured with ddgs 9.16, most of its engines turn a script away
# (DuckDuckGo with a bot check or by not answering, Brave with a rate
# limit, Google with 403, Mojeek with a captcha), which ddgs reports as
# nothing found. Yahoo answers, but ddgs asks as a browser picked at
# random, and Yahoo's page for a phone (and for some Safari versions) is
# one ddgs cannot read: a third to a half of its answers come back empty
# although Yahoo found plenty. Such an answer comes quickly, and a fresh
# DDGS is another browser, so Yahoo is asked up to six times, then every
# engine ddgs has (auto), for a network where others answer.
SEARCH_TRIES = ("yahoo",) * 6 + ("auto",)
# Results the model gets
MAX_RESULTS = 5
# What ddgs raises when the engines answered and none found anything; any
# other DDGSException is an engine it could not reach, or that timed out
NOTHING_FOUND = "No results found."


def search(query: str) -> dict:
    """
    Search the web with ddgs's engines.

    Args:
        query: Search query string

    Returns:
        Dict with 'success', 'result', and 'message' keys. Nothing found is
        a success with "No results found.", so the model says so instead
        of blaming the service; only a search that could not be made (no
        try reached an engine) fails.
    """
    logger.info("Searching: %s", query)
    error, answered = None, False
    for backend in SEARCH_TRIES:
        try:
            results = DDGS().text(query, max_results=MAX_RESULTS, backend=backend)
        except Exception as e:
            if isinstance(e, DDGSException) and str(e) == NOTHING_FOUND:
                logger.info("Search found nothing (%s): %s", backend, query)
                answered = True
            else:
                logger.warning("Search failed (%s): %s", backend, e)
                error = e
            continue
        if results:
            logger.info("Search got %d results (%s) for: %s", len(results), backend, query)
            formatted = [
                f"- {r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})" for r in results
            ]
            return {
                "success": True,
                "result": "\n".join(formatted),
                "message": f"Search results for: {query}",
            }
        answered = True

    if error is not None and not answered:
        logger.error("Search failed for '%s': %s", query, error)
        return {"success": False, "result": "Search failed", "message": f"Error: {error}"}
    logger.warning("Search found nothing for: %s", query)
    return {"success": True, "result": "No results found.", "message": "No results"}


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

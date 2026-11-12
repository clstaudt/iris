"""A scripted web for Iris's web search and page reading.

Iris's web plugin (iris.plugins.web) searches with ddgs's DDGS and reads
pages with trafilatura. FakeWeb puts scripted stand-ins in their place for
as long as it is entered: a search answers from regular expressions on the
query, after a delay; a page read returns a scripted page, or None as a page
that cannot be fetched does. Every call is recorded with when it started and
ended and what it returned, so a test can tell what Iris searched for, and
how its states and voice lined up with the search. Iris tries a search again
when it finds nothing or fails (iris.plugins.web.SEARCH_TRIES): its tries
are one call, from the first try's start to the last one's end; nothing
found is raised as ddgs raises it.

The delay is slept with time.sleep, which nothing interrupts, as a socket
waiting for a slow site would: Stop cannot cut a search short in Iris either.

With IRIS_VOICETEST_WEB=real (or real=True) the network stays in place:
the real DDGS.text and fetch_url run, and are recorded the same way.
"""

import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

from iris.plugins import web

try:
    from ddgs.exceptions import DDGSException
except ImportError:
    DDGSException = None
from irisdev.voicetest.wording import normalized

# A search or page with no delay given takes this long
DELAY = 1.0
# The names patched in iris.plugins.web
PATCHED = ("DDGS", "trafilatura")
_MISSING = object()


class Each(list):
    """One value per call, in order; the last one is kept for later calls
    (Iris's tries of one search get the same)."""


@dataclass
class WebCall:
    """A search or a page read."""

    kind: str  # "search" or "browse"
    query: str  # what was searched for, or the page's address
    start: float
    end: Optional[float] = None
    # "results", "nothing" or "error" for a search; "page" or "failed" for
    # a page read
    outcome: str = ""
    # The results, the page's text or the error
    result: object = None
    # How many times Iris tried the search (see iris.plugins.web.SEARCH_TRIES)
    tries: int = 1
    # The search's answer, its delay, and the frame of Iris's search making it
    _answer: object = field(default=None, repr=False)
    _delay: float = field(default=0.0, repr=False)
    _frame: object = field(default=None, repr=False)


def as_results(value) -> list[dict]:
    """A scripted search answer as DDGS gives results. A text becomes one
    result: "Title: body (href)", the title and address optional."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("- "):
            text = text[2:]
        href = ""
        found = re.search(r"\s*\(([^()\s]+)\)\s*$", text)
        if found:
            href, text = found.group(1), text[: found.start()]
        title, colon, body = text.partition(": ")
        if not colon:
            title, body = "", text
        return [{"title": title, "body": body, "href": href}]
    return [dict(r) for r in value or []]


def latest_match(patterns, text: str) -> Optional[str]:
    """Of these regular expressions, the one whose match lies latest in the
    text (case ignored): "Lisbon, no, Porto" gives the one for Porto."""
    best, where = None, -1
    for pattern in patterns:
        starts = [m.start() for m in re.finditer(pattern, text or "", re.IGNORECASE)]
        if starts and starts[-1] > where:
            best, where = pattern, starts[-1]
    return best


def match_query(patterns, query: str) -> Optional[str]:
    """The pattern for a query, as written or normalized (numbers as digits,
    codes joined: "L H four oh two" is "lh402")."""
    return latest_match(patterns, query) or latest_match(patterns, normalized(query))


class FakeWeb:
    """Scripted search results and pages, in place of the web while entered."""

    def __init__(
        self,
        results: Optional[dict] = None,
        pages: Optional[dict] = None,
        delay=DELAY,
        browse_delay: float = DELAY,
        real: Optional[bool] = None,
    ):
        """
        Args:
            results: {regex: results}, the regex searched in the query as
                written and normalized (see match_query), where results is a list of
                {title, body, href}, a text (one result), an Exception (the
                search fails with it), [] (nothing found) or Each([...]) of
                those, one per call
            pages: {url: text}, None for a page that cannot be fetched;
                an address not listed cannot be fetched either
            delay: Seconds a search (each of its tries) takes: one number,
                {regex: seconds}, or a list of seconds per call (the last
                one kept)
            browse_delay: Seconds a page takes to fetch
            real: Search the real web and record it; by default when
                IRIS_VOICETEST_WEB is "real"
        """
        self.results = dict(results or {})
        self.pages = dict(pages or {})
        self.delay = delay
        self.browse_delay = browse_delay
        self.real = os.environ.get("IRIS_VOICETEST_WEB", "") == "real" if real is None else real
        self.calls: list[WebCall] = []
        self._lock = threading.Lock()
        self._saved: dict = {}
        self._searches = 0
        self._served: dict = {}

    # Entering and leaving ------------------------------------------------------

    def __enter__(self) -> "FakeWeb":
        self._saved = {name: getattr(web, name, _MISSING) for name in PATCHED}
        if self.real:
            web.DDGS = self._recorded_ddgs(self._saved["DDGS"])
            real = self._saved["trafilatura"]
            web.trafilatura = SimpleNamespace(fetch_url=self._recorded_fetch(real.fetch_url),
                                              extract=real.extract)
        else:
            web.DDGS = self._fake_ddgs()
            web.trafilatura = SimpleNamespace(fetch_url=self._fetch, extract=lambda downloaded, **_: downloaded)
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            if value is _MISSING:
                if hasattr(web, name):
                    delattr(web, name)
            else:
                setattr(web, name, value)
        self._saved = {}

    # The records -----------------------------------------------------------------

    def searches(self) -> list[WebCall]:
        with self._lock:
            return [c for c in self.calls if c.kind == "search"]

    def browses(self) -> list[WebCall]:
        with self._lock:
            return [c for c in self.calls if c.kind == "browse"]

    def between(self, start: float, end: float = float("inf")) -> list[WebCall]:
        with self._lock:
            return [c for c in self.calls if start <= c.start <= end]

    def _begin(self, kind: str, query: str) -> WebCall:
        call = WebCall(kind, query, time.monotonic())
        with self._lock:
            self.calls.append(call)
        return call

    # Searching -------------------------------------------------------------------

    def _delay_for(self, query: str, n: int) -> float:
        delay = self.delay
        if isinstance(delay, dict):
            key = match_query(delay, query)
            return float(delay[key]) if key is not None else DELAY
        if isinstance(delay, (list, tuple)):
            return float(delay[min(n, len(delay) - 1)]) if delay else 0.0
        return float(delay)

    def _answer(self, query: str):
        key = match_query(self.results, query)
        if key is None:
            return []
        value = self.results[key]
        if isinstance(value, Each):
            n = self._served.get(key, 0)
            self._served[key] = n + 1
            value = value[min(n, len(value) - 1)] if value else []
        return value

    @staticmethod
    def _searching_frame():
        """The frame of Iris's web search making this call, if it is one."""
        frame = sys._getframe(2)
        code = web.search.__code__
        while frame is not None and frame.f_code is not code:
            frame = frame.f_back
        return frame

    def _search_call(self, query: str, frame) -> WebCall:
        """The call this try belongs to: the search's own when Iris tries it
        again, a new one otherwise."""
        with self._lock:
            last = next((c for c in reversed(self.calls) if c.kind == "search"), None)
            if frame is not None and last is not None and last._frame is frame and last.query == query:
                last.tries += 1
                return last
        call = self._begin("search", query)
        call._frame = frame
        return call

    def search(self, query: str, max_results: int = 5, frame=None) -> list[dict]:
        """What the fake DDGS().text(query) returns, after the delay; nothing
        found is raised as ddgs raises it."""
        call = self._search_call(query, frame)
        if call.tries == 1:
            with self._lock:
                n = self._searches
                self._searches += 1
                call._answer = self._answer(query)
            call._delay = self._delay_for(query, n)
        try:
            time.sleep(call._delay)
            answer = call._answer
            if isinstance(answer, BaseException):
                call.outcome, call.result = "error", repr(answer)
                raise answer
            found = as_results(answer)[:max_results]
            call.outcome, call.result = ("results" if found else "nothing"), found
            if not found and DDGSException is not None:
                raise DDGSException(web.NOTHING_FOUND)
            return found
        finally:
            call.end = time.monotonic()
            if call.outcome == "results":
                call._frame = None

    def _fake_ddgs(self):
        fake = self

        class DDGS:
            def __init__(self, *args, **kwargs):
                pass

            def text(self, query, max_results=5, **kwargs):
                return fake.search(query, max_results, fake._searching_frame())

        return DDGS

    def _recorded_ddgs(self, real):
        fake = self

        class DDGS:
            def __init__(self, *args, **kwargs):
                self._real = real(*args, **kwargs)

            def text(self, query, max_results=5, **kwargs):
                call = fake._search_call(query, fake._searching_frame())
                try:
                    found = self._real.text(query, max_results=max_results, **kwargs)
                    call.outcome, call.result = ("results" if found else "nothing"), found
                    return found
                except Exception as error:
                    nothing = (DDGSException is not None and isinstance(error, DDGSException)
                               and str(error) == web.NOTHING_FOUND)
                    call.outcome, call.result = ("nothing" if nothing else "error"), repr(error)
                    raise
                finally:
                    call.end = time.monotonic()
                    if call.outcome == "results":
                        call._frame = None

        return DDGS

    # Reading pages -----------------------------------------------------------------

    def page(self, url: str) -> Optional[str]:
        """The scripted page at this address (a trailing slash aside)."""
        for key in (url, url.rstrip("/"), url.rstrip("/") + "/"):
            if key in self.pages:
                return self.pages[key]
        return None

    def _fetch(self, url, config=None, **kwargs) -> Optional[str]:
        call = self._begin("browse", url)
        try:
            time.sleep(self.browse_delay)
            text = self.page(url)
            call.outcome, call.result = ("page" if text is not None else "failed"), text
            return text
        finally:
            call.end = time.monotonic()

    def _recorded_fetch(self, fetch_url):
        def fetch(url, *args, **kwargs):
            call = self._begin("browse", url)
            try:
                downloaded = fetch_url(url, *args, **kwargs)
                call.outcome = "page" if downloaded is not None else "failed"
                call.result = None if downloaded is None else len(downloaded)
                return downloaded
            except Exception as error:
                call.outcome, call.result = "failed", repr(error)
                raise
            finally:
                call.end = time.monotonic()

        return fetch

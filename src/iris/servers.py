"""Whether the model servers the user keeps are reachable.

A server is Ollama, answering at its root, or any server with the
OpenAI-compatible API, answering at .../v1. The check is a plain request with
a short timeout, so a list of servers can be checked while the app shows it.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import httpx

from iris.llm import effective_host

# Seconds a server gets to answer before it counts as unreachable
PROBE_TIMEOUT = 3.0
# A server that answers but wants a key is there
ANSWERS = (200, 401, 403)


def reachable(url: str, timeout: float = PROBE_TIMEOUT) -> bool:
    """Whether a model server answers at this URL ("" for the default one)."""
    base = (url or "").strip().rstrip("/") or effective_host(None)
    if "://" not in base:
        base = f"http://{base}"
    root = base[: -len("/v1")].rstrip("/") if base.endswith("/v1") else base
    headers = {}
    if key := os.environ.get("IRIS_LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    for probe in (f"{root}/api/version", f"{root}/v1/models"):
        try:
            if httpx.get(probe, timeout=timeout, headers=headers).status_code in ANSWERS:
                return True
        except Exception:
            continue
    return False


def check(urls: list[str], timeout: float = PROBE_TIMEOUT) -> list[dict]:
    """Each server's reachability, as {url, reachable}, checked side by side."""
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        results = list(pool.map(lambda url: reachable(url, timeout), urls))
    return [{"url": url, "reachable": ok} for url, ok in zip(urls, results)]

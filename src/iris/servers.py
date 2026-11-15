"""Whether the model servers the user keeps are reachable.

A server is Ollama, answering at its root, or any server with the
OpenAI-compatible API, answering at .../v1. The check is a plain request with
a short timeout, so a list of servers can be checked while the app shows it.
The walkthrough also looks for servers on this computer at the ports they
usually take.
"""

from concurrent.futures import ThreadPoolExecutor

import httpx

from iris.llm import effective_host
from iris.llm_client import auth_headers

# Seconds a server gets to answer before it counts as unreachable
PROBE_TIMEOUT = 3.0
# A server that answers but wants a key is there
ANSWERS = (200, 401, 403)

# Servers with the OpenAI-compatible API, where they usually listen on this
# computer: LM Studio, llama.cpp's server (and MLX's), vLLM, Jan
LOCAL_SERVERS = (
    "http://localhost:1234/v1",
    "http://localhost:8080/v1",
    "http://localhost:8000/v1",
    "http://localhost:1337/v1",
)
# Seconds each gets: on this computer an answer is quick, or none comes
DISCOVER_TIMEOUT = 1.0


def reachable(url: str, timeout: float = PROBE_TIMEOUT) -> bool:
    """Whether a model server answers at this URL ("" for the default one)."""
    base = (url or "").strip().rstrip("/") or effective_host(None)
    if "://" not in base:
        base = f"http://{base}"
    root = base[: -len("/v1")].rstrip("/") if base.endswith("/v1") else base
    headers = auth_headers(root)
    for probe in (f"{root}/api/version", f"{root}/v1/models"):
        try:
            if httpx.get(probe, timeout=timeout, headers=headers).status_code in ANSWERS:
                return True
        except Exception:
            continue
    return False


def model_server(url: str, timeout: float = DISCOVER_TIMEOUT) -> bool:
    """Whether a server with the OpenAI-compatible API answers at this URL
    (.../v1) with a list of models, not merely something on that port. One
    that wants a key is found once it has one."""
    try:
        response = httpx.get(
            f"{url.rstrip('/')}/models", timeout=timeout, headers=auth_headers(url)
        )
        return response.status_code == 200 and isinstance(response.json().get("data"), list)
    except Exception:
        return False


def discover(
    candidates: tuple[str, ...] = LOCAL_SERVERS, timeout: float = DISCOVER_TIMEOUT
) -> list[str]:
    """The model servers that answer on this computer at their usual ports,
    besides Ollama (the default server), asked side by side."""
    with ThreadPoolExecutor(max_workers=len(candidates) or 1) as pool:
        found = list(pool.map(lambda url: model_server(url, timeout), candidates))
    return [url for url, ok in zip(candidates, found) if ok]


def check(urls: list[str], timeout: float = PROBE_TIMEOUT) -> list[dict]:
    """Each server's reachability, as {url, reachable}, checked side by side."""
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        results = list(pool.map(lambda url: reachable(url, timeout), urls))
    return [{"url": url, "reachable": ok} for url, ok in zip(urls, results)]

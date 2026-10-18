"""Fetch model repos from Hugging Face with progress, before they are needed.

The libraries would download missing models themselves, but silently: on a
fresh machine the first launch would sit at "Loading models" for minutes while
1.6 GB arrive. Fetching them here first lets the UI show what is happening.
"""

import threading
import time
from collections.abc import Callable
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.auto import tqdm

# on_progress(done_bytes, total_bytes)
Progress = Callable[[int, int], None]


def is_cached(repo: str, filename: str | None = None) -> bool:
    """Whether a repo, or one file of it, is in the Hugging Face cache."""
    try:
        if filename:
            hf_hub_download(repo, filename, local_files_only=True)
        else:
            snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def download(
    repo: str, on_progress: Progress, interval: float = 0.5, filename: str | None = None
) -> str:
    """Download a repo (or only one file of it) into the Hugging Face cache,
    reporting bytes summed over all its files at most every `interval`
    seconds. Returns its path."""
    lock = threading.Lock()
    totals: dict[int, int] = {}
    done: dict[int, int] = {}
    last = [0.0]

    def report(force: bool = False):
        now = time.monotonic()
        if force or now - last[0] >= interval:
            last[0] = now
            with lock:
                on_progress(sum(done.values()), sum(totals.values()))

    class ByteProgress(tqdm):
        """Hugging Face opens one bar per file (plus one counting files);
        only the byte bars are summed."""

        def __init__(self, *args, **kwargs):
            kwargs["disable"] = False
            kwargs.pop("name", None)  # huggingface_hub's own bars take a name
            super().__init__(*args, **kwargs)
            if self.unit == "B":
                with lock:
                    totals[id(self)] = self.total or 0
                    done[id(self)] = 0

        def update(self, n=1):
            super().update(n)
            if self.unit == "B":
                with lock:
                    totals[id(self)] = max(totals.get(id(self), 0), self.total or 0)
                    done[id(self)] = self.n
                report()

    path = snapshot_download(
        repo, tqdm_class=ByteProgress, allow_patterns=[filename] if filename else None
    )
    report(force=True)
    return path


def download_files(files: list[tuple[str, Path]], on_progress: Progress, interval: float = 0.5):
    """Download (url, destination) pairs, reporting bytes over all of them.

    Each file lands under a temporary name first, so an interrupted download
    never leaves a half file where the app would look for it.
    """
    sizes = []
    for url, _ in files:
        head = requests.head(url, allow_redirects=True, timeout=30)
        sizes.append(int(head.headers.get("content-length", 0)))
    total, done, last = sum(sizes), 0, 0.0
    for url, dest in files:
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(part, "wb") as out:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if time.monotonic() - last >= interval:
                        last = time.monotonic()
                        on_progress(done, total)
        part.rename(dest)
    on_progress(done, total)

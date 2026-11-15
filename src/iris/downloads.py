"""Fetch model repos from Hugging Face with progress, before they are needed.

The libraries would download missing models themselves, but silently: on a
fresh machine the first launch would sit at "Loading models" for minutes while
1.6 GB arrive. Fetching them here first lets the UI show what is happening.

Progress is in bytes over everything a download fetches. The total is asked
of the server before the first byte (a dry run), so a bar never runs
backwards when the next file of a model starts.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.auto import tqdm

# on_progress(done_bytes, total_bytes)
Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class HubFile:
    """One file of a Hugging Face repo, at a revision (None for the latest)."""

    repo: str
    filename: str
    revision: str | None = None

    @classmethod
    def parse(cls, path: str) -> "HubFile":
        """A file named as hf://owner/repo/path/in/repo[@revision], the way
        Pocket TTS names the files it loads."""
        parts = path.removeprefix("hf://").split("/")
        filename, _, revision = "/".join(parts[2:]).partition("@")
        return cls("/".join(parts[:2]), filename, revision or None)


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


def files_cached(files: list[HubFile]) -> bool:
    """Whether all these files are in the Hugging Face cache."""
    try:
        for file in files:
            hf_hub_download(
                file.repo, file.filename, revision=file.revision, local_files_only=True
            )
        return True
    except Exception:
        return False


class _ByteCount:
    """Bytes received over all the files of a download, reported as
    (done, total) at most every `interval` seconds."""

    def __init__(self, total: int, on_progress: Progress, interval: float):
        self.total = total
        self.on_progress = on_progress
        self.interval = interval
        self._lock = threading.Lock()
        self._done: dict[int, int] = {}
        self._last = 0.0
        self.bar = self._bar_class()

    def report(self, force: bool = False):
        now = time.monotonic()
        if not force and now - self._last < self.interval:
            return
        self._last = now
        with self._lock:
            done = sum(self._done.values())
        self.on_progress(min(done, self.total) if self.total else done, self.total)

    def _bar_class(self):
        count = self

        class ByteProgress(tqdm):
            """Hugging Face opens one bar per file (plus one counting files);
            only the byte bars are summed. Xet downloads update them from
            threads of their own."""

            def __init__(self, *args, **kwargs):
                kwargs["disable"] = False
                kwargs.pop("name", None)  # huggingface_hub's own bars take a name
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                super().update(n)
                if self.unit == "B":
                    with count._lock:
                        count._done[id(self)] = self.n
                    count.report()

        return ByteProgress


def download(
    repo: str, on_progress: Progress, interval: float = 0.5, filename: str | None = None
) -> str:
    """Download a repo (or only one file of it) into the Hugging Face cache,
    reporting bytes summed over all its files at most every `interval`
    seconds. Returns its path."""
    patterns = [filename] if filename else None
    planned = snapshot_download(repo, allow_patterns=patterns, dry_run=True)
    count = _ByteCount(
        sum(f.file_size for f in planned if f.will_download), on_progress, interval
    )
    path = snapshot_download(repo, tqdm_class=count.bar, allow_patterns=patterns)
    count.report(force=True)
    return path


def download_hub_files(files: list[HubFile], on_progress: Progress, interval: float = 0.5):
    """Download single files of Hugging Face repos into its cache, reporting
    bytes over all of them; files already there are left as they are."""
    planned = [
        hf_hub_download(f.repo, f.filename, revision=f.revision, dry_run=True) for f in files
    ]
    count = _ByteCount(
        sum(p.file_size for p in planned if p.will_download), on_progress, interval
    )
    for file, plan in zip(files, planned):
        if plan.will_download:
            hf_hub_download(
                file.repo, file.filename, revision=file.revision, tqdm_class=count.bar
            )
    count.report(force=True)


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

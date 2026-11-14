"""What the process holds as a session goes on: memory, threads, open
files, temporary recordings left behind, CPU time, and the memory of MLX
(Apple's array framework, which Whisper runs on there) when it is in use.

A long session that leaks shows here: memory climbing turn after turn,
threads that outlive the turn that started them, WAV files nobody deleted.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional


def rss_mb(pid: Optional[int] = None) -> Optional[float]:
    """The process's resident memory in MB, as ps reports it."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid or os.getpid())],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return round(int(out) / 1024, 1) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def open_files() -> Optional[int]:
    """How many file descriptors the process has open."""
    for folder in ("/dev/fd", f"/proc/{os.getpid()}/fd"):
        try:
            return len(os.listdir(folder))
        except OSError:
            continue
    return None


def temporary_wavs() -> set[str]:
    """The temporary WAV files Iris and its parts leave in the temp folder."""
    return {p.name for p in Path(tempfile.gettempdir()).glob("tmp*.wav")}


def mlx_memory() -> Optional[dict]:
    """MLX's active and cached memory in MB, when MLX is loaded."""
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return None
    metal = getattr(mx, "metal", None)
    active = getattr(mx, "get_active_memory", None) or getattr(metal, "get_active_memory", None)
    cache = getattr(mx, "get_cache_memory", None) or getattr(metal, "get_cache_memory", None)
    if active is None or cache is None:
        return None
    try:
        return {"active_mb": round(active() / 2**20, 1), "cache_mb": round(cache() / 2**20, 1)}
    except Exception:
        return None


class Resources:
    """Samples the process now and then; CPU is given over the time since
    the sample before (a silence, say)."""

    def __init__(self):
        self.wavs_before = temporary_wavs()
        self._cpu = os.times()
        self._wall = time.monotonic()
        self.samples: list[dict] = []

    def sample(self, label: str = "") -> dict:
        times, wall = os.times(), time.monotonic()
        busy = (times.user + times.system) - (self._cpu.user + self._cpu.system)
        elapsed = max(1e-6, wall - self._wall)
        self._cpu, self._wall = times, wall
        sample = {
            "label": label,
            "rss_mb": rss_mb(),
            "threads": threading.active_count(),
            "thread_names": sorted(t.name for t in threading.enumerate()),
            "fds": open_files(),
            "new_wavs": len(temporary_wavs() - self.wavs_before),
            "cpu_percent": round(100 * busy / elapsed, 1),
            "mlx": mlx_memory(),
        }
        self.samples.append(sample)
        return sample

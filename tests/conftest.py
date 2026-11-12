"""Pytest configuration."""

import sys
from pathlib import Path

import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: tests requiring external services (ollama, kokoro, whisper)"
    )


@pytest.fixture(autouse=True)
def no_user_plugins(tmp_path_factory, monkeypatch):
    """Plugins in the user's own folder stay out of the tests."""
    monkeypatch.setenv("IRIS_PLUGINS_DIR", str(tmp_path_factory.mktemp("plugins")))

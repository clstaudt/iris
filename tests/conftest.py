"""Pytest configuration."""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests requiring external services (ollama, kokoro, whisper)")




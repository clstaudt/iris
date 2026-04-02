#!/bin/bash
set -e

# =============================================================================
# Iris Voice Assistant – Docker Entrypoint
# =============================================================================
# Waits for the Ollama service to be reachable, then exec's the given command.
# All iris CLI arguments are passed through as-is.
# =============================================================================

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"

# ---------------------------------------------------------------------------
# Wait for Ollama to become reachable
# ---------------------------------------------------------------------------
echo "Waiting for Ollama at ${OLLAMA_HOST} ..."
python3 -c "
import urllib.request, time, sys
for i in range(60):
    try:
        urllib.request.urlopen('${OLLAMA_HOST}', timeout=2)
        print('Ollama is ready.')
        sys.exit(0)
    except Exception:
        time.sleep(1)
print('Error: Ollama did not become ready within 60s')
sys.exit(1)
"

# ---------------------------------------------------------------------------
# Exec the command (defaults to "iris", see CMD in Dockerfile)
# ---------------------------------------------------------------------------
exec "$@"

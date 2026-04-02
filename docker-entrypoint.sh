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
max_retries=60
retry=0
until curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
    retry=$((retry + 1))
    if [ "$retry" -ge "$max_retries" ]; then
        echo "Error: Ollama did not become ready within ${max_retries}s"
        exit 1
    fi
    sleep 1
done
echo "Ollama is ready."

# ---------------------------------------------------------------------------
# Exec the command (defaults to "iris", see CMD in Dockerfile)
# ---------------------------------------------------------------------------
exec "$@"

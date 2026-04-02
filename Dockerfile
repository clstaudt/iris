::# syntax=docker/dockerfile:1

# =============================================================================
# Iris Voice Assistant - Docker Image
# =============================================================================
# Multi-stage build for a lean production image.
# No models are baked in — the iris setup wizard downloads them on first run
# and they are persisted via Docker volumes.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 – Builder: install dependencies with uv
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

# System packages needed to build native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
        libportaudio2 \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project metadata first for layer caching
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Install the project and all dependencies into a virtual-env
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2 – Runtime: lean image with only what we need
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libportaudio2 \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ src/
COPY pyproject.toml ./

# Make sure the virtualenv is on PATH
ENV PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/app/.venv" \
    PYTHONUNBUFFERED=1

# Entrypoint: waits for Ollama, then exec's iris with any arguments
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["iris"]

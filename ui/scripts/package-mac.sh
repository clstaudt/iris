#!/usr/bin/env bash
# Build Iris.dmg for Apple Silicon Macs: the frozen Python core, the UI, and
# the app around them. Signed ad hoc only; see README "macOS app".
#
#   cd ui && npm run dist:mac      ->  ui/release/Iris-<version>-macOS-arm64.dmg
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "== Python core (PyInstaller)"
cd "$ROOT/packaging"
# The core is frozen from a build environment of its own, on uv's standalone
# Python and the locked dependencies, so the app never depends on whichever
# interpreter backs the development .venv
export UV_PROJECT_ENVIRONMENT="$ROOT/packaging/.venv"
uv sync --project "$ROOT" --managed-python --python 3.13 --locked --no-dev --quiet
uv pip install --quiet --python .venv "pyinstaller>=6.11"
.venv/bin/pyinstaller --noconfirm --clean --log-level WARN iris-rpc.spec

# Every engine must import inside the bundle; a missing one would otherwise
# only show up on the recipient's Mac
./dist/iris-rpc/iris-rpc --check

echo "== UI"
cd "$ROOT/ui"
npm run build

echo "== App and disk image"
npx electron-builder --mac dmg --arm64
ls -lh release/*.dmg

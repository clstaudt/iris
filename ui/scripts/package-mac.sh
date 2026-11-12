#!/usr/bin/env bash
# Build Iris.dmg for Apple Silicon Macs: the frozen Python core, the UI, and
# the app around them; see README "macOS app".
#
#   cd ui && npm run dist:mac      ->  ui/release/Iris-<version>-macOS-arm64.dmg
#
# Signing: with a Developer ID Application certificate (CSC_LINK and
# CSC_KEY_PASSWORD, or one in the login keychain) the app is signed with it
# under the hardened runtime, and notarized when APPLE_ID,
# APPLE_APP_SPECIFIC_PASSWORD and APPLE_TEAM_ID are set. Without a
# certificate, or with CSC_IDENTITY_AUTO_DISCOVERY=false, it is signed ad hoc
# and not notarized.
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
signing=()
if [ -z "${CSC_LINK:-}" ] && { [ "${CSC_IDENTITY_AUTO_DISCOVERY:-true}" = "false" ] ||
    ! security find-identity -v -p codesigning 2>/dev/null | grep -q "Developer ID Application"; }; then
  echo "Signing ad hoc, without notarization (no Developer ID certificate in use)"
  signing=(-c.mac.identity=- -c.mac.notarize=false)
fi
npx electron-builder --mac dmg --arm64 --publish never ${signing[@]+"${signing[@]}"}
ls -lh release/*.dmg

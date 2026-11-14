# Building the macOS app

```bash
cd ui
npm run dist:mac        # -> ui/release/Iris-<version>-macOS-arm64.dmg
```

`ui/scripts/package-mac.sh` does three things:

1. It freezes the Python core with PyInstaller (`packaging/iris-rpc.spec`) into a standalone `iris-rpc` folder. The core's own Python, from a separate build environment in `packaging/.venv`, comes along, so the recipient needs no Python. `iris-rpc --check` then imports every engine inside the bundle (speech recognition on MLX and on the CPU, voice activity detection, turn detection, echo cancellation, Pocket TTS and Kokoro, the voice itself, OpenAI-compatible model servers, web search, web pages, camera, screen capture, audio devices) and stops the build if one is missing.
2. It builds the UI.
3. It runs electron-builder, which puts the core into the app's `Resources`, signs the app under the hardened runtime (`ui/build/entitlements.mac.plist`) and writes the `.dmg`.

What is not in the app: the speech models, which are downloaded at first launch (or taken from folders the recipient points to), and the language model server.

**Signing and notarizing.** With a Developer ID Application certificate the app is signed with it: one in the login keychain is found by itself, or give it as `CSC_LINK` (a base64-encoded `.p12`, or its path) and `CSC_KEY_PASSWORD`. It is then notarized by Apple when `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD` (an app-specific password of that Apple ID) and `APPLE_TEAM_ID` are set; electron-builder also takes a stored `notarytool` profile as `APPLE_KEYCHAIN_PROFILE`. Without a certificate, or with `CSC_IDENTITY_AUTO_DISCOVERY=false`, the app is signed ad hoc and not notarized.

**In CI.** `.github/workflows/pack-electron.yml` builds the disk image on a macOS runner for every push to `main` (signed ad hoc; its disk image replaces the one in the rolling "latest" pre-release, so the newest build can always be downloaded) and for every tag `v*` (signed with the Developer ID and notarized, using the repository secrets `CSC_LINK`, `CSC_KEY_PASSWORD`, `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD` and `APPLE_TEAM_ID`). A tag's disk image is uploaded to its GitHub release; a draft release is made if there is none. To release, set the version in `ui/package.json` and push a matching tag, e.g. `v0.2.0`.

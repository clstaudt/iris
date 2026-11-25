"""The app's third-party notices: every piece of software others made that
the app ships, with its license and the license texts it comes with.

ui/scripts/package-mac.sh runs this with the Python of the core's build
environment once the ui is built. It writes packaging/dist/
THIRD_PARTY_NOTICES.txt, which electron-builder puts beside the app's own
LICENSE. In it:

- Python, which PyInstaller freezes into the core;
- every package of the build environment, which uv synced from the lock file
  without the dev dependencies: what the core imports, and PyInstaller, whose
  bootloader starts it;
- libraries those packages carry without their license text;
- the packages from node_modules that the ui's bundle holds, as the ui's build
  lists them (bundled-packages.json, see vite.config.ts);
- Electron. Chromium's notices, a page of their own in Electron, go beside the
  file as LICENSES.chromium.html.

    python notices.py [--ui <ui folder>] [--out <file>]
"""

import argparse
import json
import re
import sys
from importlib import metadata
from pathlib import Path

try:
    # PyAV, which carries FFmpeg, knows which FFmpeg it was built with
    import av

    FFMPEG_VERSION = av.ffmpeg_version_info
except ImportError:
    FFMPEG_VERSION = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Files that hold a license, or notices a license asks to keep with it
LICENSE_NAME = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|COPYRIGHT|THIRD.?PARTY)", re.I)
# ...unless they are code (a license.py) or a library
NOT_TEXT = {".py", ".pyc", ".pyi", ".pyx", ".pxd", ".so", ".dylib", ".dll", ".js", ".ts"}

RULE = "=" * 78
LINE = "-" * 78

# GNU licenses, for the libraries below, which come without their text
GNU_TEXTS = {
    "GPL-3.0": HERE / "licenses" / "GPL-3.0.txt",
    "LGPL-3.0": HERE / "licenses" / "LGPL-3.0.txt",
}


def is_license_file(path: str) -> bool:
    name = Path(path).name
    return bool(LICENSE_NAME.match(name)) and Path(name).suffix.lower() not in NOT_TEXT


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def license_of(meta) -> str:
    """A package's license as its metadata states it: the SPDX expression,
    else a one-line License field, else its license classifiers."""
    if meta.get("License-Expression"):
        return meta["License-Expression"]
    field = (meta.get("License") or "").strip()
    if field and "\n" not in field and len(field) <= 100:
        return field
    classifiers = [
        c.split(" :: ")[-1] for c in meta.get_all("Classifier") or [] if c.startswith("License ::")
    ]
    return ", ".join(classifiers) or "not stated"


def home_of(meta) -> str:
    if meta.get("Home-page"):
        return meta["Home-page"]
    urls = dict(
        (label.strip().lower(), url.strip())
        for label, _, url in (entry.partition(",") for entry in meta.get_all("Project-URL") or [])
    )
    for label in ("homepage", "home", "source", "source code", "repository", "github"):
        if label in urls:
            return urls[label]
    return next(iter(urls.values()), "")


def entry(title: str, license: str, home: str, texts: list[str], missing: str) -> str:
    head = [LINE, title, f"License: {license}"]
    if home:
        head.append(f"Home: {home}")
    body = texts or [missing]
    return "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n"


def python_entry() -> str:
    version = ".".join(map(str, sys.version_info[:3]))
    lib = Path(sys.base_prefix) / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}"
    return entry(
        f"Python {version}",
        "PSF-2.0",
        "https://www.python.org",
        [read(lib / "LICENSE.txt")],
        "",
    )


def package_texts(dist) -> list[str]:
    seen, texts = set(), []
    for file in sorted(dist.files or [], key=str):
        if not is_license_file(str(file)):
            continue
        path = Path(dist.locate_file(file))
        if not path.is_file():
            continue
        text = read(path)
        if text and text not in seen:
            seen.add(text)
            texts.append(f"[{file}]\n\n{text}")
    if not texts:
        # Some packages put the license's text in the License field itself
        field = (dist.metadata.get("License") or "").strip()
        if "\n" in field or len(field) > 100:
            texts.append(field)
    return texts


def python_packages(skip: set[str]) -> list[str]:
    dists = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name and name.lower() not in skip:
            dists.setdefault(name.lower(), dist)
    return [
        entry(
            f"{dist.metadata['Name']} {dist.version}",
            license_of(dist.metadata),
            home_of(dist.metadata),
            package_texts(dist),
            "The package ships no license text; its license is the one named above.",
        )
        for _, dist in sorted(dists.items())
    ]


def vendored() -> list[str]:
    """Libraries packages carry in their wheels without the library's
    license text: the license, and the source of the version shipped."""
    entries = []
    try:
        loader = Path(metadata.distribution("espeakng-loader").locate_file("espeakng_loader"))
        found = sorted(loader.glob("libespeak-ng.*.*.*.dylib")) + sorted(
            loader.glob("libespeak-ng.so.*.*.*")
        )
        version = re.search(r"(\d+\.\d+\.\d+)", found[0].name).group(1) if found else ""
        entries.append(
            entry(
                f"espeak-ng {version}".strip() + ", in espeakng-loader",
                "GPL-3.0-or-later (the text is at the end of this file)",
                "https://github.com/espeak-ng/espeak-ng",
                [
                    "Source: https://github.com/espeak-ng/espeak-ng/tree/"
                    + (version or "master")
                ],
                "",
            )
        )
    except metadata.PackageNotFoundError:
        pass
    if FFMPEG_VERSION is not None:
        entries.append(
            entry(
                f"FFmpeg {FFMPEG_VERSION}, in av (PyAV)",
                "LGPL-3.0-or-later (the texts of the LGPL and the GPL are at the end of this file)",
                "https://ffmpeg.org",
                [
                    f"Source: https://ffmpeg.org/releases/ffmpeg-{FFMPEG_VERSION}.tar.xz, as "
                    "PyAV builds it for its wheels with https://github.com/PyAV-Org/pyav-ffmpeg"
                ],
                "",
            )
        )
    return entries


def ui_packages(ui: Path) -> list[str]:
    listed = ui / "dist" / "bundled-packages.json"
    if not listed.is_file():
        raise SystemExit(f"{listed} is missing: build the ui first (npm run build)")
    entries = []
    for name in sorted(json.loads(listed.read_text())):
        folder = ui / "node_modules" / name
        package = json.loads((folder / "package.json").read_text())
        texts = [
            f"[{path.name}]\n\n{read(path)}"
            for path in sorted(folder.iterdir())
            if path.is_file() and is_license_file(path.name)
        ]
        repository = package.get("repository")
        if isinstance(repository, dict):
            repository = repository.get("url")
        entries.append(
            entry(
                f"{name} {package.get('version', '')}".strip(),
                str(package.get("license") or "not stated"),
                package.get("homepage") or repository or "",
                texts,
                "The package ships no license text; its license is the one named above.",
            )
        )
    return entries


def electron_entry(ui: Path) -> str:
    folder = ui / "node_modules" / "electron"
    version = json.loads((folder / "package.json").read_text())["version"]
    return entry(
        f"Electron {version}",
        "MIT",
        "https://www.electronjs.org",
        [
            read(folder / "dist" / "LICENSE"),
            "Chromium, which Electron is built on, and the libraries it includes have their "
            "notices in LICENSES.chromium.html, beside this file.",
        ],
        "",
    )


def notices(ui: Path) -> str:
    parts = [
        "Third-party notices",
        RULE,
        "Iris is free software under the GNU Affero General Public License, version 3 or "
        "later (LICENSE, beside this file). It includes the software below, which others "
        "made. Each part comes under its own license, named with it, with the license and "
        "notice texts it ships.",
        "",
        RULE,
        "The core",
        "",
        python_entry(),
        *python_packages(skip={"iris"}),
        RULE,
        "Libraries in the packages above that come without their license text",
        "",
        *vendored(),
        RULE,
        "The window",
        "",
        electron_entry(ui),
        *ui_packages(ui),
        RULE,
        "License texts the libraries above refer to",
        "",
        *(f"{LINE}\n{name}\n\n{read(path)}\n" for name, path in GNU_TEXTS.items()),
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ui", type=Path, default=ROOT / "ui")
    parser.add_argument("--out", type=Path, default=HERE / "dist" / "THIRD_PARTY_NOTICES.txt")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(notices(args.ui), encoding="utf-8")
    print(f"{args.out}: {args.out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()

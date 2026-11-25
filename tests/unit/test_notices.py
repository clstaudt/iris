"""The app's third-party notices (packaging/notices.py): each package with its
license and the license texts it ships, from the build environment and the
ui's bundle."""

import importlib.util
import json
from email.message import Message
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("notices", ROOT / "packaging" / "notices.py")
notices = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(notices)


def _meta(**fields) -> Message:
    meta = Message()
    for key, value in fields.items():
        for item in value if isinstance(value, list) else [value]:
            meta[key.replace("_", "-")] = item
    return meta


def _dist(folder: Path, name: str, version: str, license_text: str | None, **fields):
    """A package installed in folder, as pip writes one: its metadata, a
    module, and, when it has one, its license file."""
    info = folder / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    meta = _meta(Metadata_Version="2.4", Name=name, Version=version, **fields)
    (info / "METADATA").write_text(meta.as_string())
    (folder / name).mkdir()
    (folder / name / "__init__.py").write_text("")
    (folder / name / "license.py").write_text("# code, not a license\n")
    records = [f"{name}/__init__.py,,", f"{name}/license.py,,", f"{info.name}/METADATA,,"]
    if license_text is not None:
        (info / "licenses").mkdir()
        (info / "licenses" / "LICENSE").write_text(license_text)
        records.append(f"{info.name}/licenses/LICENSE,,")
    (info / "RECORD").write_text("\n".join(records) + "\n")
    return metadata.PathDistribution(info)


def test_license_files_are_told_from_code_and_other_files():
    for path in (
        "felix-1.0.dist-info/licenses/LICENSE",
        "LICENSE.txt",
        "COPYING",
        "NOTICE.md",
        "LICENSES_bundled.txt",
        "ThirdPartyNotices.txt",
        "LICENSE-3RD-PARTY.txt",
    ):
        assert notices.is_license_file(path), path
    for path in ("felix/license.py", "README.md", "felix/_licenses.cpython-313-darwin.so"):
        assert not notices.is_license_file(path), path


def test_a_license_is_named_as_the_metadata_best_states_it():
    expression = _meta(License_Expression="MIT", License="MIT License")
    assert notices.license_of(expression) == "MIT"
    assert notices.license_of(_meta(License="BSD 3-Clause")) == "BSD 3-Clause"
    classified = _meta(
        License="Copyright (c) Ada\n\nPermission is hereby granted...",
        Classifier=["License :: OSI Approved :: MIT License", "Programming Language :: Python"],
    )
    assert notices.license_of(classified) == "MIT License"
    assert notices.license_of(_meta()) == "not stated"


def test_every_package_comes_with_its_license_texts(tmp_path, monkeypatch):
    felix = _dist(tmp_path, "felix", "1.2.0", "Copyright (c) Felix\n\nThe text of MIT.",
                  License_Expression="MIT", Home_page="https://example.org/felix")
    robin = _dist(
        tmp_path, "robin", "0.3", None, Classifier="License :: OSI Approved :: MIT License"
    )
    own = _dist(tmp_path / "own", "iris", "1.0.0", "AGPL")
    monkeypatch.setattr(notices.metadata, "distributions", lambda: [robin, own, felix])

    entries = notices.python_packages(skip={"iris"})

    assert len(entries) == 2
    first, second = entries
    assert "felix 1.2.0\nLicense: MIT\nHome: https://example.org/felix" in first
    assert "The text of MIT." in first
    assert "code, not a license" not in first
    assert "robin 0.3\nLicense: MIT License" in second
    assert "ships no license text" in second


def test_the_ui_bundles_packages_with_their_licenses(tmp_path):
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundled-packages.json").write_text(json.dumps(["ada-ui"]))
    package = tmp_path / "node_modules" / "ada-ui"
    package.mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps({"name": "ada-ui", "version": "2.0.1", "license": "ISC",
                    "repository": {"url": "https://example.org/ada-ui"}})
    )
    (package / "LICENSE").write_text("ISC License, Copyright (c) Ada")
    (package / "index.js").write_text("export {}")

    (entry,) = notices.ui_packages(tmp_path)

    assert "ada-ui 2.0.1\nLicense: ISC\nHome: https://example.org/ada-ui" in entry
    assert "ISC License, Copyright (c) Ada" in entry


def test_without_the_ui_built_the_notices_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="build the ui first"):
        notices.ui_packages(tmp_path)

"""The core packed into the app imports without what its spec leaves out.

PyInstaller freezes the app's core from packaging/iris-rpc.spec, whose
`excludes` keeps modules out of the bundle that the core is thought never
to need. When the core comes to import one after all (voice cloning reached
typer through huggingface_hub's login), the core in the app cannot start,
and only the macOS app build would show it. Here the core is imported as
the app starts it, in a Python that finds none of the excluded modules nor
their submodules, as in the bundle: the entry point (packaging/iris_rpc.py),
iris.rpc_server, and each engine `iris-rpc --check` imports.
"""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "packaging" / "iris-rpc.spec"
ENTRY = ROOT / "packaging" / "iris_rpc.py"
PROBE = Path(__file__).with_name("import_as_bundled.py")

# Importing torch, MLX and the rest takes a while the first time
IMPORT_S = 600


def spec_excludes(path: Path = SPEC) -> tuple[list[str], list[str]]:
    """The modules the spec's Analysis excludes, and the entries of its
    excludes that could not be read, each with why."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # Names the spec sets to a literal at its top, such as UNUSED_JIEBA
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        names[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError, SyntaxError):
                        names.pop(target.id, None)
    analyses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Analysis"
    ]
    assert len(analyses) == 1, f"{path.name} should call Analysis once, not {len(analyses)} times"
    listed = next((k.value for k in analyses[0].keywords if k.arg == "excludes"), None)
    if listed is None:
        return [], []
    if not isinstance(listed, ast.List | ast.Tuple):
        return [], [f"excludes={ast.unparse(listed)} is no list the test can read"]
    excludes, unread = [], []
    for item in listed.elts:
        starred = isinstance(item, ast.Starred)
        node = item.value if starred else item
        try:
            value = names[node.id] if isinstance(node, ast.Name) else ast.literal_eval(node)
        except (KeyError, ValueError, TypeError, SyntaxError):
            unread.append(f"{ast.unparse(item)}: not a string, nor a name the spec sets to strings")
            continue
        values = list(value) if starred else [value]
        if not all(isinstance(v, str) for v in values):
            unread.append(f"{ast.unparse(item)}: not module names ({value!r})")
            continue
        excludes += values
    return excludes, unread


def entry_engines(path: Path = ENTRY) -> dict[str, str]:
    """ENGINES of the entry point: each feature and the module it needs."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ENGINES" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise LookupError(f"{path.name} sets no ENGINES")


ENGINES = entry_engines()
MODULES = ["iris.rpc_server", *dict.fromkeys(ENGINES.values())]
DEPENDENCIES = [
    Requirement(line)
    for line in tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
]


@pytest.fixture(scope="module")
def excludes():
    return spec_excludes()


@pytest.fixture(scope="module")
def imported(excludes, tmp_path_factory):
    """What imported, in one Python refusing the excluded modules."""
    folder = tmp_path_factory.mktemp("bundle-imports")
    request, results = folder / "request.json", folder / "results.json"
    asked = {"src": str(ROOT / "src"), "entry": str(ENTRY), "modules": MODULES}
    request.write_text(json.dumps({**asked, "excludes": excludes[0]}), encoding="utf-8")
    # -I: neither the environment's PYTHONPATH nor the folders it runs in
    # put anything on the path before the source folder the probe adds
    done = subprocess.run(
        [sys.executable, "-I", str(PROBE), str(request), str(results)],
        cwd=folder,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=IMPORT_S,
    )
    if not results.exists():
        pytest.fail(
            f"The import probe ended with code {done.returncode} and no results:\n"
            f"{done.stdout[-4000:]}\n{done.stderr[-4000:]}",
            pytrace=False,
        )
    return json.loads(results.read_text(encoding="utf-8"))


def shown(file: str) -> str:
    """A file of the chain, short: in the repo, or in its package."""
    path = Path(file)
    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    parts = path.parts
    if "site-packages" in parts:
        return str(Path(*parts[parts.index("site-packages") + 1 :]))
    return file


def explain(what: str, result: dict) -> str:
    lines = [f"{what} does not import in the app's core: {result['error']}", ""]
    if result["excluded"]:
        lines.append(
            f"{result['missing']} is not in the bundle: packaging/iris-rpc.spec excludes "
            f"{result['excluded']!r}. It is imported through:"
        )
    else:
        lines.append("Through:")
    for step in result["chain"]:
        lines.append(f"  {step['module']} ({shown(step['file'])}:{step['line']}): {step['code']}")
    if result["missing"]:
        lines.append(f"  -> {result['missing']}")
    return "\n".join([*lines, "", result["traceback"]])


def not_for_this_computer(result: dict) -> str | None:
    """Why a module that did not import is not meant to be installed here,
    if that is why: pyproject.toml declares it for other platforms only (as
    MLX, for Apple Silicon), and it is not installed. A module the spec
    excludes, one that is installed, or one meant for here fails instead."""
    missing = result["missing"]
    if result["excluded"] or not missing:
        return None
    package = missing.partition(".")[0]
    if importlib.util.find_spec(package) is not None:
        return None
    for requirement in DEPENDENCIES:
        if (
            canonicalize_name(requirement.name) == canonicalize_name(package)
            and requirement.marker is not None
            and not requirement.marker.evaluate()
        ):
            return (
                f"{package} is not installed here: pyproject.toml declares {requirement.name} "
                f"only where {requirement.marker}"
            )
    return None


def check(what: str, result: dict):
    if result["ok"]:
        return
    reason = not_for_this_computer(result)
    if reason:
        pytest.skip(reason)
    pytest.fail(explain(what, result), pytrace=False)


def test_every_exclude_is_read(excludes):
    """Each entry of the spec's excludes is refused below; one the test
    cannot read (not a string, nor a name the spec sets to strings) is
    named here, instead of passing unnoticed."""
    found, unread = excludes
    assert found, "packaging/iris-rpc.spec excludes nothing the test could read"
    if unread:
        pytest.skip("Entries of the spec's excludes left out of this test: " + "; ".join(unread))


def test_the_entry_point_imports(imported):
    """packaging/iris_rpc.py and what it imports at its top: the RPC server,
    and the readers --check has the voices read with."""
    check("packaging/iris_rpc.py", imported["entry"])


@pytest.mark.parametrize("module", MODULES)
def test_the_core_and_each_engine_import(imported, module):
    check(module, imported["modules"][module])

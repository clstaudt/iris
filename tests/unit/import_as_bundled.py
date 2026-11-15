"""Imports the core as the app's bundle holds it: without the modules its
PyInstaller spec excludes. test_bundle_imports.py runs this in a Python of
its own, so that only the standard library is loaded before those modules
are refused.

    python -I tests/unit/import_as_bundled.py REQUEST.json RESULTS.json

The request names the source folder, the entry point, the modules to import
after it and the modules excluded. The results tell for each whether it
imported, and if not, why and through which modules.
"""

import importlib
import importlib.util
import json
import linecache
import sys
import traceback
from pathlib import Path

# Frames of the import system itself, which are no step of an import chain
IMPORT_SYSTEM = {
    "importlib",
    "importlib.util",
    "importlib._bootstrap",
    "importlib._bootstrap_external",
}


class Excluded:
    """The excluded modules; with each, its submodules, as PyInstaller
    leaves them out."""

    def __init__(self, names: list[str]):
        self.names = tuple(names)

    def rule(self, module: str) -> str | None:
        """The excluded name that leaves this module out, if one does."""
        return next((n for n in self.names if module == n or module.startswith(n + ".")), None)


class Refusing:
    """One of the import system's finders, finding no excluded module: to
    the code importing one it is not installed, as in the bundle. A lookup
    (importlib.util.find_spec) finds nothing either, instead of failing."""

    def __init__(self, finder, excluded: Excluded):
        self.finder = finder
        self.excluded = excluded

    def find_spec(self, name, path=None, target=None):
        if self.excluded.rule(name):
            return None
        find_spec = getattr(self.finder, "find_spec", None)
        return find_spec(name, path, target) if find_spec else None

    def __getattr__(self, attribute):
        # invalidate_caches, find_distributions, ...
        return getattr(self.finder, attribute)


def refuse(excluded: Excluded):
    """Every finder refuses the excluded modules, also those added since."""
    sys.meta_path[:] = [
        finder if isinstance(finder, Refusing) else Refusing(finder, excluded)
        for finder in sys.meta_path
    ]


def load_entry(path: str):
    """The entry point as the bundle runs it, but not its main()."""
    spec = importlib.util.spec_from_file_location("iris_rpc", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["iris_rpc"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules["iris_rpc"]
        raise


def errors_of(error: BaseException) -> list[BaseException]:
    """The error, and those it was raised from or while handling."""
    errors = []
    while error is not None and error not in errors:
        errors.append(error)
        error = error.__cause__ or error.__context__
    return errors


def steps(error: BaseException) -> list[dict]:
    """The modules an error's traceback goes through, each with the line it
    went on to the next on."""
    found = []
    tb = error.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        module = frame.f_globals.get("__name__", "?")
        filename = frame.f_code.co_filename
        if module != "__main__" and module not in IMPORT_SYSTEM and not filename.startswith("<"):
            code = linecache.getline(filename, tb.tb_lineno).strip()
            step = {"module": module, "file": filename, "line": tb.tb_lineno, "code": code}
            # Of a module's frames in a row, the last is where it imports
            if found and found[-1]["module"] == module:
                found[-1] = step
            else:
                found.append(step)
        tb = tb.tb_next
    return found


def attempt(load, target: str, excluded: Excluded) -> dict:
    refuse(excluded)
    try:
        load(target)
    except Exception as error:
        errors = errors_of(error)
        missing = [e for e in errors if isinstance(e, ModuleNotFoundError) and e.name]
        refused = next((e for e in missing if excluded.rule(e.name)), None)
        cause = refused or (missing[0] if missing else None)
        # From the target down: the error raised, then those it was raised
        # from, as far as the module that is missing
        chain = []
        for e in errors[: errors.index(cause) + 1] if cause else errors[:1]:
            for step in steps(e):
                if chain and chain[-1]["module"] == step["module"]:
                    chain[-1] = step
                else:
                    chain.append(step)
        return {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "missing": cause.name if cause else None,
            "excluded": excluded.rule(cause.name) if cause else None,
            "chain": chain,
            "traceback": "".join(traceback.format_exception(error)),
        }
    return {"ok": True}


def main(request_path: str, results_path: str):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    excluded = Excluded(request["excludes"])
    # Nothing excluded may be loaded already (by a .pth file, say)
    for name in [name for name in sys.modules if excluded.rule(name)]:
        del sys.modules[name]
    sys.path.insert(0, request["src"])
    results = {"entry": attempt(load_entry, request["entry"], excluded), "modules": {}}
    for module in request["modules"]:
        results["modules"][module] = attempt(importlib.import_module, module, excluded)
    Path(results_path).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])

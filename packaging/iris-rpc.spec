# PyInstaller spec for the Python core of the desktop app: a standalone
# `iris-rpc` folder that Electron starts from the app's Resources.
# Built by ui/scripts/package-mac.sh; Apple Silicon only (MLX).

import importlib.util
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# MLX looks for its GPU kernels (mlx.metallib) beside the real libmlx.dylib.
# Where PyInstaller puts that library on its own depends on the Python it
# runs on, so it is placed explicitly in mlx/lib (anything at the top is then
# only a symlink) with the kernels beside it. Kernels in the wrong place make
# the first import of mlx.core fail; the optional-MLX guard swallows that and
# the next import aborts the process ("duplicate key ... DeviceType"). Never
# collect_all("mlx"): a second copy of the library aborts it the same way.
mlx_lib = os.path.join(importlib.util.find_spec("mlx").submodule_search_locations[0], "lib")
hiddenimports += collect_submodules("mlx")
binaries += [(os.path.join(mlx_lib, "libmlx.dylib"), "mlx/lib")]
datas += [(os.path.join(mlx_lib, "mlx.metallib"), "mlx/lib")]

# Packages that need their data files (models, tables, stubs) or native
# libraries at runtime, beyond what PyInstaller finds through imports
for package in (
    "iris",  # personas
    "mlx_whisper",
    "faster_whisper",
    "ctranslate2",
    "kokoro_onnx",
    "espeakng_loader",
    "phonemizer",
    "language_tags",
    "segments",
    "csvw",
    "librosa",
    "trafilatura",
    "justext",
    "courlan",
    "htmldate",
    "tiktoken",
    "tiktoken_ext",
    "ddgs",
    "primp",
    "livekit",  # its native library, for echo cancellation
    "pocket_tts",  # the configs of its models
    "py3langid",  # its model, which tells the language a reply is written in
    "num2words",  # numbers in words, for voices that read no digits
):
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports

a = Analysis(
    ["iris_rpc.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # Installed for other tools but never used by the core. torch stays:
    # Pocket TTS runs on it, and the voice imports it
    excludes=[
        "torchvision",
        "torchaudio",
        "pandas",
        "matplotlib",
        "tkinter",
        "IPython",
        "pytest",
        "typer",
        "irisdev",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="iris-rpc",
    console=True,
    target_arch="arm64",
)
coll = COLLECT(exe, a.binaries, a.datas, name="iris-rpc")

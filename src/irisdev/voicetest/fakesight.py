"""A scripted camera and screen for Iris's camera and screen tools.

Iris's camera plugin (iris.plugins.camera) opens the camera with OpenCV and
its screen plugin (iris.plugins.screen) captures the monitor with mss.
FakeCamera and FakeScreen put stand-ins in their place for as long as they
are entered, so the plugins' own code runs: the camera is opened, warmed up,
read and released, the screen grabbed, scaled and encoded. What they show
is a scene: a colored object in the middle of a gray room, which a scripted
model can tell from the image it is shown (color_of). Each capture is
recorded with when it started and ended, and whether the camera was let go.

They can stand for what goes wrong too: no camera, or one the system does
not let Iris open (available=False), a camera that gives no frame or breaks,
and a screen that comes back black or empty, as macOS returns it to an app
that may not record the screen.
"""

import base64
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from types import SimpleNamespace
from typing import Optional

import numpy as np
from PIL import Image

from iris.plugins import camera, screen

# The colors a scene's object can have, as RGB
COLORS = {
    "red": (210, 30, 30),
    "green": (30, 170, 60),
    "blue": (40, 60, 210),
    "yellow": (230, 200, 30),
}
ROOM = (120, 120, 120)
_MISSING = object()


def scene(color: str = "red", size: tuple = (640, 480)) -> np.ndarray:
    """A gray room with an object of this color in its middle, as RGB."""
    width, height = size
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = ROOM
    image[height // 4 : 3 * height // 4, width // 3 : 2 * width // 3] = COLORS[color]
    return image


def color_of(image_b64: str) -> Optional[str]:
    """The color of the object in the middle of an image (base64, as Iris
    shows it to the model), or None if it holds none of COLORS."""
    try:
        image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
    except Exception:
        return None
    width, height = image.size
    middle = np.asarray(image.crop((width * 2 // 5, height * 2 // 5, width * 3 // 5, height * 3 // 5)))
    mean = middle.reshape(-1, 3).mean(axis=0)
    name, distance = min(
        ((name, float(np.linalg.norm(mean - np.array(rgb)))) for name, rgb in COLORS.items()),
        key=lambda pair: pair[1],
    )
    return name if distance < 60 else None


def image_size(image_b64: str) -> tuple[int, int]:
    return Image.open(BytesIO(base64.b64decode(image_b64))).size


@dataclass
class Capture:
    """One use of the camera or the screen."""

    start: float
    end: Optional[float] = None
    # Whether the camera was let go afterwards (the screen holds nothing)
    released: bool = False
    # How many frames were read, the warm-up's included
    reads: int = 0


class FakeCamera:
    """A camera showing a scene, in place of OpenCV's while entered."""

    def __init__(
        self,
        color: str = "red",
        size: tuple = (640, 480),
        available: bool = True,
        frames: bool = True,
        breaks: Optional[Exception] = None,
    ):
        """
        Args:
            color: The color of the object the user holds up
            size: The camera's frame size
            available: False for no camera, or one the system does not let
                Iris open: it never opens
            frames: False for a camera that opens but gives no frame
            breaks: Raised by the camera when a frame is read
        """
        self.color = color
        self.size = size
        self.available = available
        self.frames = frames
        self.breaks = breaks
        self.captures: list[Capture] = []
        self._lock = threading.Lock()
        self._saved: dict = {}

    def __enter__(self) -> "FakeCamera":
        self._saved = {"cv2": getattr(camera, "cv2", _MISSING), "HAS_CAMERA": camera.HAS_CAMERA}
        fake = self

        class VideoCapture:
            def __init__(self, index):
                self.capture = Capture(time.monotonic())
                with fake._lock:
                    fake.captures.append(self.capture)

            def isOpened(self):
                return fake.available

            def read(self):
                self.capture.reads += 1
                if fake.breaks is not None:
                    raise fake.breaks
                if not fake.frames:
                    return False, None
                # OpenCV's frames are BGR
                return True, scene(fake.color, fake.size)[:, :, ::-1].copy()

            def release(self):
                self.capture.released = True
                self.capture.end = time.monotonic()

        camera.cv2 = SimpleNamespace(
            VideoCapture=VideoCapture,
            COLOR_BGR2RGB=4,
            cvtColor=lambda frame, code: frame[:, :, ::-1].copy(),
            setLogLevel=lambda level: None,
        )
        camera.HAS_CAMERA = True
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            if value is _MISSING:
                if hasattr(camera, name):
                    delattr(camera, name)
            else:
                setattr(camera, name, value)
        self._saved = {}

    def between(self, start: float, end: float = float("inf")) -> list[Capture]:
        with self._lock:
            return [c for c in self.captures if start <= c.start <= end]


class FakeScreen:
    """A monitor showing a scene, in place of mss while entered."""

    def __init__(
        self,
        color: str = "blue",
        size: tuple = (2880, 1800),
        black: bool = False,
        empty: bool = False,
        breaks: Optional[Exception] = None,
    ):
        """
        Args:
            color: The color of the window in the middle of the screen
            size: The primary monitor's size in pixels
            black: The capture comes back black, as macOS returns it to an
                app that may not record the screen
            empty: The capture comes back with no pixels at all
            breaks: Raised when the screen is grabbed
        """
        self.color = color
        self.size = size
        self.black = black
        self.empty = empty
        self.breaks = breaks
        self.captures: list[Capture] = []
        # The monitors grabbed, by their index in mss's list
        self.grabbed: list[int] = []
        self._lock = threading.Lock()
        self._saved: dict = {}

    def __enter__(self) -> "FakeScreen":
        self._saved = {"mss": getattr(screen, "mss", _MISSING), "HAS_SCREEN": screen.HAS_SCREEN}
        fake = self
        width, height = self.size
        # mss lists every monitor together first, then each one
        monitors = [
            {"left": 0, "top": 0, "width": width * 2, "height": height},
            {"left": 0, "top": 0, "width": width, "height": height},
            {"left": width, "top": 0, "width": width, "height": height},
        ]

        class Grabber:
            def __init__(self):
                self.monitors = monitors

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def grab(self, monitor):
                capture = Capture(time.monotonic())
                with fake._lock:
                    fake.captures.append(capture)
                    fake.grabbed.append(monitors.index(monitor))
                try:
                    if fake.breaks is not None:
                        raise fake.breaks
                    if fake.empty:
                        return SimpleNamespace(size=(0, 0), rgb=b"")
                    pixels = scene(fake.color, (monitor["width"], monitor["height"]))
                    if fake.black:
                        pixels = np.zeros_like(pixels)
                    return SimpleNamespace(size=(monitor["width"], monitor["height"]), rgb=pixels.tobytes())
                finally:
                    capture.end = time.monotonic()

        screen.mss = SimpleNamespace(mss=Grabber)
        screen.HAS_SCREEN = True
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            if value is _MISSING:
                if hasattr(screen, name):
                    delattr(screen, name)
            else:
                setattr(screen, name, value)
        self._saved = {}

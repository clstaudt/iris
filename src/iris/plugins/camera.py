"""The camera: Iris looks through it when what it is asked needs eyes."""

import base64
import os
import sys
import time
import warnings
from io import BytesIO

from iris.plugin import Plugin, PluginContext, Tool

# Suppress OpenCV/AVCapture warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import cv2
    from PIL import Image

    # Suppress macOS AVCapture deprecation warnings
    cv2.setLogLevel(0)
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False

DESCRIPTION = (
    "Use your camera to see what the user is showing you. Call this function whenever you "
    "need to see something visually to answer the user's question. This is your vision - use "
    'it when they ask "what do you see?", "look at this", "what am I holding?", or any '
    "question requiring visual information."
)
GUIDANCE = (
    "- 'look' function: Your CAMERA. You CANNOT see through the camera without calling this.\n"
    "  * YOU MUST call 'look' for ANY question about physical objects, gestures, or camera view\n"
    "  * NEVER describe what the camera sees without calling this function first\n"
    "  * Call it EVERY TIME they say 'look again' - always capture fresh\n"
)
# Where macOS lets an app use the camera
MAC_PERMISSION = "If there is one, allow Iris under System Settings > Privacy & Security > Camera."


def _unavailable() -> dict:
    """No camera, or one the system does not let Iris open, which look the
    same from here. On macOS the user may have to allow it, which only they
    can do: they are told as well as the model."""
    result = {"success": False, "message": "Camera not available"}
    if sys.platform == "darwin":
        result["notice"] = f"Iris cannot open a camera. {MAC_PERMISSION}"
    return result


def look(**kwargs) -> dict:
    """
    Capture a single image from the camera.

    Args:
        **kwargs: Ignored - accepts any arguments the model might pass

    Returns:
        Dict with 'success', 'image', and 'message' keys
    """
    camera = None
    try:
        # OpenCV reports on stderr while it opens the camera; silence it
        # without getting in the way of the permission prompt
        old_stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(old_stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, old_stderr_fd)
        os.close(devnull_fd)

        try:
            # Open camera temporarily
            camera = cv2.VideoCapture(0)
        finally:
            # Restore stderr
            os.dup2(saved_stderr_fd, old_stderr_fd)
            os.close(saved_stderr_fd)

        if not camera.isOpened():
            return _unavailable()

        # Give camera time to warm up (important for first frame quality)
        # Also makes the "Looking" indicator clearly visible
        time.sleep(1.5)

        # Discard first few frames (they're often dark/poor quality)
        for _ in range(5):
            camera.read()

        # Small additional pause to ensure "Looking" is noticeable
        time.sleep(0.3)

        # Capture the actual frame
        ret, frame = camera.read()

        if not ret or frame is None:
            return {"success": False, "message": "Failed to capture image"}

        # Convert BGR to RGB, then encode as JPEG in memory
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"success": True, "image": image_base64, "message": "Image captured"}

    except Exception as e:
        return {"success": False, "message": f"Error capturing image: {e}"}
    finally:
        # Release camera immediately (not for video)
        if camera is not None:
            camera.release()


class Camera(Plugin):
    id = "camera"
    name = "Camera"

    def available(self) -> bool:
        return HAS_CAMERA

    def tools(self, context: PluginContext) -> list[Tool]:
        return [
            Tool(
                name="look",
                description=DESCRIPTION,
                run=look,
                guidance=GUIDANCE,
                activity="looking",
                label="Looked through the camera",
            )
        ]


plugin = Camera()

"""The screen: Iris looks at what is on the user's monitor when asked."""

import base64
from io import BytesIO

import numpy as np

from iris.plugin import Plugin, PluginContext, Tool

try:
    import mss
    import mss.tools
    from PIL import Image

    HAS_SCREEN = True
except ImportError:
    HAS_SCREEN = False

DESCRIPTION = (
    "Take a screenshot of the user's physical computer screen."
    " Use ONLY when they ask about what is on their screen,"
    " monitor, or desktop. NEVER use this to read websites"
    " - use 'browse' for that."
)
GUIDANCE = (
    "- 'capture_screen' function: Screenshot of the user's PHYSICAL computer screen.\n"
    "  * ONLY use when they ask about what is on their screen, desktop, or open applications\n"
    "  * NEVER use this to read websites - use 'browse' for websites\n"
    "  * 'page', 'article', 'headline', 'website' refer to the WEB, not the screen\n"
)


def capture_screen(**kwargs) -> dict:
    """
    Capture a screenshot of the user's primary screen.

    Args:
        **kwargs: Ignored - accepts any arguments the model might pass

    Returns:
        Dict with 'success', 'image', and 'message' keys
    """
    try:
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[1])

            # Empty or black: screen recording was refused
            if screenshot.size[0] == 0 or screenshot.size[1] == 0:
                return {
                    "success": False,
                    "message": "Screen capture failed - screen recording permission may be "
                    "required. Grant permission in System Settings > Privacy & Security > "
                    "Screen Recording.",
                }

            image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            if np.array(image).max() == 0:
                return {
                    "success": False,
                    "message": "Screen capture returned black image - please grant screen "
                    "recording permission in System Settings > Privacy & Security > Screen "
                    "Recording, then restart the application.",
                }

            # Vision models don't need full resolution
            max_dimension = 1920
            if image.width > max_dimension or image.height > max_dimension:
                ratio = min(max_dimension / image.width, max_dimension / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return {"success": True, "image": image_base64, "message": "Screen captured"}

    except Exception as e:
        return {
            "success": False,
            "message": f"Error capturing screen: {e}. You may need to grant screen "
            "recording permission in System Settings.",
        }


class Screen(Plugin):
    id = "screen"
    name = "Screen"

    def available(self) -> bool:
        return HAS_SCREEN

    def tools(self, context: PluginContext) -> list[Tool]:
        return [
            Tool(
                name="capture_screen",
                description=DESCRIPTION,
                run=capture_screen,
                guidance=GUIDANCE,
                activity="capturing_screen",
                label="Looked at your screen",
            )
        ]


plugin = Screen()

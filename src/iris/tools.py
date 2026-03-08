"""Simple tool system for LLM function calling."""

import base64
import logging
import os
import sys
import time
import warnings
from io import BytesIO

import numpy as np

logger = logging.getLogger(__name__)

# Suppress OpenCV/AVCapture warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Note: We handle stderr redirection in _look() to suppress warnings without blocking permissions

try:
    from PIL import Image
    import cv2

    # Suppress macOS AVCapture deprecation warnings
    cv2.setLogLevel(0)
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False

try:
    import mss
    import mss.tools

    HAS_SCREEN = True
except ImportError:
    HAS_SCREEN = False

try:
    from ddgs import DDGS
    import trafilatura

    HAS_SEARCH = True
except ImportError:
    HAS_SEARCH = False


class Tools:
    """Simple container for LLM tools."""

    def __init__(self, console=None, status_manager=None):
        """
        Initialize tools.

        Args:
            console: Optional Rich console for status messages
            status_manager: Optional StatusManager for UI updates
        """
        self.console = console
        self.status_manager = status_manager
        self._camera = None
        self._enabled_tools = {}

    def enable_camera(self):
        """
        Enable camera/vision tool.

        Returns:
            True if enabled successfully, False if dependencies missing
        """
        if HAS_CAMERA:
            self._enabled_tools["look"] = {
                "function": self._look,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "look",
                        "description": 'Use your camera to see what the user is showing you. Call this function whenever you need to see something visually to answer the user\'s question. This is your vision - use it when they ask "what do you see?", "look at this", "what am I holding?", or any question requiring visual information.',
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                },
            }
            return True
        return False

    def enable_web_search(self):
        """
        Enable web search and browse tools.

        Returns:
            True if enabled successfully, False if dependencies missing
        """
        if HAS_SEARCH:
            from trafilatura.settings import use_config

            self._traf_config = use_config()
            self._traf_config.set(
                "DEFAULT",
                "USER_AGENTS",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            self._enabled_tools["search"] = {
                "function": self._search,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web for current information. Use when you need up-to-date facts, news, or information you don't know.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"}
                            },
                            "required": ["query"],
                        },
                    },
                },
            }
            self._enabled_tools["browse"] = {
                "function": self._browse,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "browse",
                        "description": (
                            "Fetch and read the contents of a webpage."
                            " Use when the user mentions a specific website"
                            " or you need to read a URL from search results."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "Full URL to fetch (e.g. https://www.theguardian.com)",
                                }
                            },
                            "required": ["url"],
                        },
                    },
                },
            }
            return True
        return False

    def enable_screen_capture(self):
        """
        Enable screen capture tool.

        Returns:
            True if enabled successfully, False if dependencies missing
        """
        if HAS_SCREEN:
            self._enabled_tools["capture_screen"] = {
                "function": self._capture_screen,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "capture_screen",
                        "description": (
                            "Take a screenshot of the user's physical computer screen."
                            " Use ONLY when they ask about what is on their screen,"
                            " monitor, or desktop. NEVER use this to read websites"
                            " - use 'browse' for that."
                        ),
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                },
            }
            return True
        return False

    def _look(self, **kwargs) -> dict:
        """
        Capture a single image from camera.
        Note: Looking status should already be set by LLM before this is called.

        Args:
            **kwargs: Ignored - accepts any arguments LLM might pass

        Returns:
            Dict with 'success', 'image', and 'message' keys
        """
        camera = None
        try:
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
                return {"success": False, "message": "Camera not available"}

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

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)

            # Encode as JPEG in memory
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

            # Clear looking status after camera is done
            if self.status_manager:
                with self.status_manager.status_lock:
                    self.status_manager.status_looking = False

    def _search(self, query: str) -> dict:
        """
        Search the web using DuckDuckGo.

        Args:
            query: Search query string

        Returns:
            Dict with 'success', 'result', and 'message' keys
        """
        logger.info("Searching: %s", query)

        if self.status_manager:
            with self.status_manager.status_lock:
                self.status_manager.status_searching = True

        try:
            results = DDGS().text(query, max_results=5)

            if not results:
                logger.warning("Search returned no results for: %s", query)
                return {"success": True, "result": "No results found.", "message": "No results"}

            logger.info("Search got %d results for: %s", len(results), query)

            formatted = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                formatted.append(f"- {title}: {body} ({href})")

            result_text = "\n".join(formatted)
            return {
                "success": True,
                "result": result_text,
                "message": f"Search results for: {query}",
            }

        except Exception as e:
            logger.error("Search failed for '%s': %s", query, e, exc_info=True)
            return {"success": False, "result": "Search failed", "message": f"Error: {e}"}
        finally:
            if self.status_manager:
                with self.status_manager.status_lock:
                    self.status_manager.status_searching = False

    def _browse(self, url: str) -> dict:
        """
        Fetch and extract readable text from a webpage using trafilatura.

        Args:
            url: URL to fetch

        Returns:
            Dict with 'success', 'result', and 'message' keys
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        logger.info("Browsing: %s", url)

        if self.status_manager:
            with self.status_manager.status_lock:
                self.status_manager.status_browsing = True

        try:
            downloaded = trafilatura.fetch_url(url, config=self._traf_config)
            if downloaded is None:
                logger.warning("Browse could not fetch: %s", url)
                return {"success": False, "result": "", "message": f"Could not fetch {url}"}

            text = trafilatura.extract(
                downloaded,
                include_links=True,
                include_tables=True,
                output_format="txt",
                config=self._traf_config,
            )
            if not text:
                logger.warning("Browse fetched but no content extracted: %s", url)
                return {"success": False, "result": "", "message": "No content extracted"}

            logger.info("Browse got %d chars from %s", len(text), url)

            if len(text) > 4000:
                text = text[:4000] + "\n...(truncated)"

            return {
                "success": True,
                "result": text,
                "message": f"Content from: {url}",
            }

        except Exception as e:
            logger.error("Browse failed for '%s': %s", url, e, exc_info=True)
            return {"success": False, "result": "Browse failed", "message": f"Error: {e}"}
        finally:
            if self.status_manager:
                with self.status_manager.status_lock:
                    self.status_manager.status_browsing = False

    def _capture_screen(self, **kwargs) -> dict:
        """
        Capture a screenshot of the user's screen.
        Note: Capturing screen status should already be set by LLM before this is called.

        Args:
            **kwargs: Ignored - accepts any arguments LLM might pass

        Returns:
            Dict with 'success', 'image', and 'message' keys
        """
        try:
            # Capture the primary monitor
            with mss.mss() as sct:
                # Get the primary monitor
                monitor = sct.monitors[1]

                # Capture screenshot
                screenshot = sct.grab(monitor)

                # Check if screenshot is valid (not empty/black from permission denial)
                if screenshot.size[0] == 0 or screenshot.size[1] == 0:
                    return {
                        "success": False,
                        "message": "Screen capture failed - screen recording permission may be required. Grant permission in System Settings > Privacy & Security > Screen Recording.",
                    }

                # Convert to PIL Image
                image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

                img_array = np.array(image)
                if img_array.max() == 0:  # Completely black
                    return {
                        "success": False,
                        "message": "Screen capture returned black image - please grant screen recording permission in System Settings > Privacy & Security > Screen Recording, then restart the application.",
                    }

                # Resize to reasonable size (vision models don't need full resolution)
                max_dimension = 1920
                if image.width > max_dimension or image.height > max_dimension:
                    ratio = min(max_dimension / image.width, max_dimension / image.height)
                    new_size = (int(image.width * ratio), int(image.height * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)

                # Encode as JPEG in memory
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=85)
                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                return {"success": True, "image": image_base64, "message": "Screen captured"}

        except Exception as e:
            error_msg = f"Error capturing screen: {e}. You may need to grant screen recording permission in System Settings."
            return {"success": False, "message": error_msg}
        finally:
            if self.status_manager:
                with self.status_manager.status_lock:
                    self.status_manager.status_capturing_screen = False

    def get_schemas(self) -> list[dict]:
        """
        Get Ollama tool schemas for all enabled tools.

        Returns:
            List of tool schema dicts
        """
        return [tool["schema"] for tool in self._enabled_tools.values()]

    def execute(self, tool_name: str, **kwargs) -> dict:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool execution result dict
        """
        if tool_name not in self._enabled_tools:
            return {"success": False, "message": f"Unknown tool: {tool_name}"}

        return self._enabled_tools[tool_name]["function"](**kwargs)

    def has_tools(self) -> bool:
        """
        Check if any tools are enabled.

        Returns:
            True if at least one tool is enabled
        """
        return len(self._enabled_tools) > 0

    def cleanup(self):
        """Cleanup resources (camera already released per-use)."""
        pass

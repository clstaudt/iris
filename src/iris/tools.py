"""Simple tool system for LLM function calling."""

import base64
import os
import sys
import warnings
from io import BytesIO
from typing import Optional

# Suppress OpenCV/AVCapture warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

try:
    from PIL import Image
    import cv2
    # Suppress macOS AVCapture deprecation warnings
    cv2.setLogLevel(0)
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


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
            self._enabled_tools['look'] = {
                'function': self._look,
                'schema': {
                    'type': 'function',
                    'function': {
                        'name': 'look',
                        'description': 'Use your camera to see what the user is showing you. Call this function whenever you need to see something visually to answer the user\'s question. This is your vision - use it when they ask "what do you see?", "look at this", "what am I holding?", or any question requiring visual information.',
                        'parameters': {
                            'type': 'object',
                            'properties': {},
                            'required': []
                        }
                    }
                }
            }
            return True
        return False
    
    def enable_web_search(self):
        """
        Enable web search tool.
        
        Returns:
            True if enabled successfully, False if dependencies missing
        """
        if HAS_REQUESTS:
            self._enabled_tools['search'] = {
                'function': self._search,
                'schema': {
                    'type': 'function',
                    'function': {
                        'name': 'search',
                        'description': 'Search the web for current information. Use when you need up-to-date facts, news, or information you don\'t know.',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'query': {
                                    'type': 'string',
                                    'description': 'Search query'
                                }
                            },
                            'required': ['query']
                        }
                    }
                }
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
        import time
        import contextlib
        
        camera = None
        try:
            # Redirect stderr to suppress AVCapture warnings at OS level
            import sys
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
                return {'success': False, 'message': 'Camera not available'}
            
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
                return {'success': False, 'message': 'Failed to capture image'}
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            
            # Encode as JPEG in memory
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return {
                'success': True,
                'image': image_base64,
                'message': 'Image captured'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error capturing image: {e}'
            }
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
        Search the web.
        
        Args:
            query: Search query string
            
        Returns:
            Dict with 'success', 'result', and 'message' keys
        """
        if self.console:
            self.console.print(f"[dim cyan]🔍 Searching: {query}[/dim cyan]")
        
        try:
            # Use DuckDuckGo instant answer API
            response = requests.get(
                'https://api.duckduckgo.com/',
                params={'q': query, 'format': 'json'},
                timeout=5
            )
            data = response.json()
            
            # Try to get meaningful result
            result = data.get('AbstractText') or data.get('Answer') or 'No results found'
            
            if self.console:
                self.console.print("[dim green]✓ Search complete[/dim green]")
            
            return {
                'success': True,
                'result': result,
                'message': f'Search results for: {query}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'result': 'Search failed',
                'message': f'Error: {e}'
            }
    
    def get_schemas(self) -> list[dict]:
        """
        Get Ollama tool schemas for all enabled tools.
        
        Returns:
            List of tool schema dicts
        """
        return [tool['schema'] for tool in self._enabled_tools.values()]
    
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
            return {'success': False, 'message': f'Unknown tool: {tool_name}'}
        
        return self._enabled_tools[tool_name]['function'](**kwargs)
    
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


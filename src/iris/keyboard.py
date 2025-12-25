"""Keyboard input handler for pause/interrupt controls."""

import logging
import threading
from collections.abc import Callable
from typing import Optional

# Suppress pynput logging warnings about accessibility
logging.getLogger("pynput").setLevel(logging.ERROR)

# Try to import keyboard library (pynput is cross-platform)
try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False


class KeyboardHandler:
    """Handles keyboard input for pause and interrupt controls."""
    
    def __init__(self):
        """Initialize keyboard handler."""
        self.listener: Optional[keyboard.Listener] = None
        self.on_space_pressed: Optional[Callable[[], None]] = None
        self._lock = threading.Lock()
        self._enabled = False
        
    def start(self, on_space: Optional[Callable[[], None]] = None) -> bool:
        """
        Start listening for keyboard input. If already started, just update callback.
        
        Args:
            on_space: Callback function to call when space bar is pressed
            
        Returns:
            True if listener started successfully, False otherwise
        """
        if not KEYBOARD_AVAILABLE:
            return False
        
        # If already running, just update the callback
        if self.listener and self._enabled:
            if on_space:
                with self._lock:
                    self.on_space_pressed = on_space
            return True
            
        # Set callback
        if on_space:
            with self._lock:
                self.on_space_pressed = on_space
        
        def on_press(key):
            """Handle key press events."""
            try:
                # Check if space bar was pressed
                if key == keyboard.Key.space:
                    with self._lock:
                        if self.on_space_pressed:
                            self.on_space_pressed()
            except Exception:
                pass
        
        try:
            # Start keyboard listener in non-blocking mode
            self.listener = keyboard.Listener(
                on_press=on_press,
                suppress=False  # Don't suppress the space key
            )
            self.listener.daemon = True
            self.listener.start()
            self._enabled = True
            return True
        except Exception:
            # Likely permissions issue on macOS
            self._enabled = False
            return False
    
    def stop(self):
        """Stop listening for keyboard input."""
        if self.listener:
            try:
                self.listener.stop()
            except Exception:
                pass
            self.listener = None
        with self._lock:
            self.on_space_pressed = None
        self._enabled = False
    
    def update_callback(self, on_space: Optional[Callable[[], None]]):
        """
        Update the callback function.
        
        Args:
            on_space: New callback function, or None to disable
        """
        with self._lock:
            self.on_space_pressed = on_space

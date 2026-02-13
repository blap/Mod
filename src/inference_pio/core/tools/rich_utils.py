"""
Rich Console Utility with Real Fallback
"""
import sys
import contextlib

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class FallbackConsole:
        """
        Functional Console implementation using standard I/O when Rich is unavailable.
        Not a 'Mock' - it performs real output operations.
        """
        def print(self, *args, **kwargs):
            # Basic print implementation
            # Removing 'style' or other rich-specific kwargs
            clean_kwargs = {k:v for k,v in kwargs.items() if k in ['sep', 'end', 'file', 'flush']}
            print(*args, **clean_kwargs)

        def log(self, *args, **kwargs):
            # Simple log format
            print("[LOG]", *args)

        def status(self, message: str):
            """
            Context manager for status updates.
            Prints start message and end message.
            """
            print(f"[STATUS] {message}...")
            @contextlib.contextmanager
            def _status():
                yield
                print(f"[STATUS] Done.")
            return _status()

    console = FallbackConsole()

__all__ = ["console"]

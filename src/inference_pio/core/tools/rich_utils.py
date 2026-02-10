"""
Rich Console Utility
"""
try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            # Strip color codes if any (simple implementation)
            msg = " ".join([str(a) for a in args])
            print(msg)

    console = MockConsole()

__all__ = ["console"]

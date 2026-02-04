import logging
import os
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for Inference-PIO
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "error": "bold red",
    "success": "bold green",
    "highlight": "yellow"
})

console = Console(theme=custom_theme)

def setup_rich_logging(debug: bool = False, log_file: str = None):
    """
    Configure logging to use Rich for console output and standard logging for file.

    Args:
        debug: Enable debug logging
        log_file: Path to log file
    """
    level = logging.DEBUG if debug else logging.INFO

    # Create log directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True),
            *( [file_handler] if log_file else [] )
        ]
    )

    return console

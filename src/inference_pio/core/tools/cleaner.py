import os
import shutil
from pathlib import Path
from .rich_utils import console

def clean_project():
    """Clean temporary files and caches."""
    root_dir = Path(os.getcwd())

    # Patterns to clean
    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/.pytest_cache",
        "**/.mypy_cache",
        "**/temp_activations",
        "**/offloaded_parts",
        "**/logs/*.log"
    ]

    console.print("[bold yellow]Cleaning project...[/bold yellow]")

    cleaned_count = 0
    cleaned_size = 0

    for pattern in patterns:
        for path in root_dir.glob(pattern):
            try:
                if path.is_file():
                    size = path.stat().st_size
                    path.unlink()
                    cleaned_size += size
                    cleaned_count += 1
                    console.print(f"Removed file: {path.relative_to(root_dir)}", style="dim")
                elif path.is_dir():
                    # Calculate size roughly
                    size = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
                    shutil.rmtree(path)
                    cleaned_size += size
                    cleaned_count += 1
                    console.print(f"Removed directory: {path.relative_to(root_dir)}", style="dim")
            except Exception as e:
                console.print(f"[red]Failed to remove {path}: {e}[/red]")

    console.print(f"[bold green]Clean complete! Removed {cleaned_count} items ({cleaned_size / (1024*1024):.2f} MB).[/bold green]")

"""
Model Loader and Path Resolution Utility

This module handles finding model weights, prioritizing the H: drive as requested,
and managing model downloads from Hugging Face Hub if needed.
"""

import os
import platform
import logging
from typing import Optional, List
from pathlib import Path

# Try to import huggingface_hub for downloads, but don't crash if missing
try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from .tools.rich_utils import console

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Handles resolution of model paths with specific priority logic.
    Priority:
    1. H:/models/<model_name> (Windows) or /mnt/h/models/<model_name> (Linux/WSL)
    2. H:/<model_name>
    3. User provided path / Local cache
    4. Hugging Face Hub (Download)
    """

    @staticmethod
    def get_h_drive_base() -> Optional[Path]:
        """Detect the H: drive mount point."""
        system = platform.system()
        if system == 'Windows':
            path = Path('H:/')
            if path.exists():
                return path
        else:
            # Linux/WSL common mount points
            paths = [Path('/mnt/h'), Path('/media/h'), Path('/drives/h')]
            for p in paths:
                if p.exists():
                    return p
        return None

    @staticmethod
    def resolve_model_path(model_name: str, hf_repo_id: Optional[str] = None) -> str:
        """
        Resolve the path to the model weights.

        Args:
            model_name: Name of the model (e.g., 'qwen3-0.6b')
            hf_repo_id: Hugging Face Repository ID (e.g., 'Qwen/Qwen1.5-0.5B')

        Returns:
            Resolved path as string, or HF repo ID if using remote loading
        """
        # Normalize model name for directory search
        model_dir_name = model_name.replace('-', '_').lower()
        model_dir_name_alt = model_name # Original

        # 1. Check H: drive
        h_drive = ModelLoader.get_h_drive_base()
        if h_drive:
            potential_paths = [
                h_drive / "models" / model_dir_name,
                h_drive / "models" / model_dir_name_alt,
                h_drive / model_dir_name,
                h_drive / model_dir_name_alt,
                h_drive / "AI" / "models" / model_dir_name # Common variation
            ]

            for path in potential_paths:
                if path.exists() and (path / "config.json").exists():
                    logger.info(f"Found model in H: drive priority path: {path}")
                    console.print(f"[green]Found model locally on H: drive:[/green] {path}")
                    return str(path)

        # 2. Check standard local paths
        local_paths = [
            Path("models") / model_dir_name,
            Path.home() / ".cache" / "inference_pio" / model_dir_name
        ]

        for path in local_paths:
            if path.exists() and (path / "config.json").exists():
                logger.info(f"Found model in local path: {path}")
                return str(path)

        # 3. If not found locally, suggest download
        if not hf_repo_id:
             # Try to guess or return name to let plugin decide
             logger.warning(f"Model {model_name} not found locally and no HF repo ID provided.")
             return model_name

        logger.info(f"Model {model_name} not found locally. Preparing for HF Hub.")

        # If we are here, we might want to download or just return the repo ID
        # for transformers to handle.
        # But per requirements, let's offer to download to H: if available?

        return ModelLoader.download_model_interactive(model_name, hf_repo_id, h_drive)

    @staticmethod
    def download_model_interactive(model_name: str, repo_id: str, h_drive: Optional[Path]) -> str:
        """Interactive model download manager."""
        if not HF_HUB_AVAILABLE:
            console.print("[yellow]huggingface_hub not installed. Returning repo ID for automatic cache download.[/yellow]")
            return repo_id

        console.print(f"[bold yellow]Model '{model_name}' not found locally.[/bold yellow]")

        # Default download path
        if h_drive:
            target_dir = h_drive / "models" / model_name.replace('-', '_')
            console.print(f"H: drive detected. Recommended download path: [cyan]{target_dir}[/cyan]")
        else:
            target_dir = Path("models") / model_name.replace('-', '_')

        if os.environ.get("PIO_AUTO_DOWNLOAD", "0") == "1":
             should_download = True
        else:
             response = input(f"Do you want to download '{repo_id}' to '{target_dir}'? [y/N/cache]: ").lower()
             if response == 'cache':
                 return repo_id # Let transformers handle caching
             should_download = response == 'y'

        if should_download:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                console.print(f"[green]Downloading {repo_id} to {target_dir}...[/green]")
                snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
                console.print("[bold green]Download complete![/bold green]")
                return str(target_dir)
            except Exception as e:
                console.print(f"[bold red]Download failed: {e}[/bold red]")
                return repo_id # Fallback

        return repo_id

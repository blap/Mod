"""
Model Loader and Path Resolution Utility

This module handles finding model weights, prioritizing the H: drive as requested,
and managing model downloads from Hugging Face Hub if needed.
"""

import os
import platform
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Try to import huggingface_hub for downloads, but don't crash if missing
try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from .tools.rich_utils import console
from .engine.backend import load_safetensors, Tensor

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
    def __init__(self, model_path: str = ""):
        self.model_path = model_path

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
    def resolve_model_path(
        model_name: str, hf_repo_id: Optional[str] = None
    ) -> str:
        # Normalize model name for directory search
        model_dir_name = model_name.replace('-', '_').lower()
        model_dir_name_alt = model_name  # Original

        # 1. Check H: drive
        h_drive = ModelLoader.get_h_drive_base()
        if h_drive:
            potential_paths = [
                h_drive / "models" / model_dir_name,
                h_drive / "models" / model_dir_name_alt,
                h_drive / model_dir_name,
                h_drive / model_dir_name_alt,
                h_drive / "AI" / "models" / model_dir_name  # Common variation
            ]

            for path in potential_paths:
                if path.exists() and (path / "config.json").exists():
                    logger.info(f"Found model in H: drive priority path: {path}")
                    console.print(
                        f"[green]Found model locally on H: drive:[/green] {path}"
                    )
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
            logger.warning(
                f"Model {model_name} not found locally and no HF repo ID provided."
            )
            return model_name

        logger.info(f"Model {model_name} not found locally. Preparing for HF Hub.")

        return ModelLoader.download_model_interactive(
            model_name, hf_repo_id, h_drive
        )

    @staticmethod
    def download_model_interactive(
        model_name: str, repo_id: str, h_drive: Optional[Path]
    ) -> str:
        """Interactive model download manager."""
        if not HF_HUB_AVAILABLE:
            console.print(
                "[yellow]huggingface_hub not installed. "
                "Returning repo ID for automatic cache download.[/yellow]"
            )
            return repo_id

        console.print(
            f"[bold yellow]Model '{model_name}' not found locally.[/bold yellow]"
        )

        # Default download path
        if h_drive:
            target_dir = h_drive / "models" / model_name.replace('-', '_')
            console.print(
                f"H: drive detected. Recommended download path: "
                f"[cyan]{target_dir}[/cyan]"
            )
        else:
            target_dir = Path("models") / model_name.replace('-', '_')

        if os.environ.get("PIO_AUTO_DOWNLOAD", "0") == "1":
            should_download = True
        else:
            # Interactive skipped for automation safety, default to cache if not explicit
            should_download = False

        if should_download:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                console.print(
                    f"[green]Downloading {repo_id} to {target_dir}...[/green]"
                )
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False
                )
                console.print("[bold green]Download complete![/bold green]")
                return str(target_dir)
            except Exception as e:
                console.print(f"[bold red]Download failed: {e}[/bold red]")
                return repo_id  # Fallback

        return repo_id

    def load_into_module(self, module: Any):
        """
        Standard load: Loads all tensors found in safetensors to the module parameters.
        Default to CPU.
        """
        if not self.model_path or not os.path.exists(self.model_path):
            return

        filepath = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(filepath):
            load_safetensors(filepath, module._parameters)
            # Also recurse modules? load_safetensors expects a flat dict or handling of hierarchy?
            # Our custom Model puts all params in module._parameters or recurses manually.
            # Real impl relies on the module structure matching the file.
            pass

class SmartModelLoader(ModelLoader):
    def load_into_module(self, module: Any, max_gpu_mem_gb: float = 10.0):
        """
        Smart Load: Loads weights and moves to GPU until limit is reached.
        """
        if not self.model_path or not os.path.exists(self.model_path):
            return

        filepath = os.path.join(self.model_path, "model.safetensors")
        if not os.path.exists(filepath):
            return

        # 1. Load everything to CPU first (simplest safe path)
        # In a real heavy implementation, we would mmap and selective load.
        # But 'load_safetensors' (C) is efficient.
        super().load_into_module(module)

        # 2. Distribute to GPU
        current_gpu_bytes = 0
        limit_bytes = int(max_gpu_mem_gb * 1024**3)

        # Iterate all modules recursively to find Tensors
        # Simple DFS
        queue = [module]
        while queue:
            curr = queue.pop(0)

            # Move parameters
            for name, tensor in curr._parameters.items():
                if tensor and tensor.device == "cpu":
                    size = tensor.size * 4
                    if current_gpu_bytes + size < limit_bytes:
                        # Move to GPU
                        new_tensor = tensor.to("cuda")
                        curr._parameters[name] = new_tensor
                        current_gpu_bytes += size
                    else:
                        # Keep on CPU
                        pass

            # Recurse children
            for m in curr._modules.values():
                queue.append(m)

        logger.info(f"Smart Load: Placed {current_gpu_bytes / 1024**2:.2f} MB on GPU")

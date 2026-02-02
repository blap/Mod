"""
Global Configuration for Inference-PIO

This module provides global configuration settings for the Inference-PIO system.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GlobalConfig:
    """
    Global configuration settings for the Inference-PIO system.
    """

    # Performance settings
    default_batch_size: int = 1
    max_batch_size: int = 32
    inference_timeout: int = 300  # seconds
    memory_fraction: float = 0.9  # Fraction of GPU memory to use

    # Model loading settings
    default_device: str = (
        "cuda" if os.environ.get("USE_CUDA", "true").lower() == "true" else "cpu"
    )
    enable_tensor_parallelism: bool = False
    tensor_parallel_size: int = 1

    # Optimization settings
    enable_flash_attention: bool = True
    enable_tensor_fusion: bool = True
    enable_disk_offloading: bool = False

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Cache settings
    cache_directory: str = "./model_cache"
    enable_model_caching: bool = True

    # Plugin settings
    plugin_directory: str = "./plugins"
    auto_discover_plugins: bool = True

    def __post_init__(self):
        """
        Validate and finalize configuration settings.
        """
        if self.default_device == "cuda" and not self._is_cuda_available():
            self.default_device = "cpu"

        if not os.path.exists(self.cache_directory):
            os.makedirs(self.cache_directory, exist_ok=True)

    def _is_cuda_available(self) -> bool:
        """
        Check if CUDA is available.

        Returns:
            True if CUDA is available, False otherwise.
        """
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False


# Global configuration instance
global_config = GlobalConfig()


def get_global_config() -> GlobalConfig:
    """
    Get the global configuration instance.

    Returns:
        GlobalConfig: The global configuration instance.
    """
    return global_config


__all__ = ["GlobalConfig", "get_global_config", "global_config"]

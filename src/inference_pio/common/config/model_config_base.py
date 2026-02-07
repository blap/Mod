
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch

def get_default_model_path(model_name: str) -> str:
    return f"H:/{model_name}"

class ModelConfigError(Exception):
    pass

@dataclass
class BaseConfig:
    """
    Base configuration class for all models.
    """

    # Model identification
    model_path: str = ""
    model_name: str = "base_model"

    # Device settings
    device: Optional[str] = None
    device_map: str = "auto"

    # Data type
    torch_dtype: str = "float16"

    # Memory optimization settings
    gradient_checkpointing: bool = True
    use_cache: bool = True
    low_cpu_mem_usage: bool = True
    max_memory: Optional[Dict] = None
    enable_disk_offloading: bool = False
    offload_folder: str = "offload"

    # Hardware optimizations
    use_tensor_parallelism: bool = False

    # Attention mechanism settings
    use_flash_attention_2: bool = True
    use_sdpa: bool = True

    # KV Cache
    use_paged_kv_cache: bool = True
    paged_attention_page_size: int = 16

    # Throughput
    use_continuous_batching: bool = True

    def __post_init__(self):
        """Post-initialization adjustments."""
        # Set default model path if not provided
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)

@dataclass
class OptimizationConfig:
    """
    Configuration for optimization strategies.
    """
    enable_quantization: bool = False
    quantization_bits: int = 8

def get_optimal_config_for_hardware(model_name: str = None) -> BaseConfig:
    """
    Generate an optimal configuration based on detected hardware.
    """
    config = BaseConfig()
    config.model_name = model_name or "optimal_model"

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        if gpu_mem > 20 * 1024**3: # >20GB
            config.device_map = "cuda:0"
            config.offload_folder = None
        else:
            config.device_map = "auto"
            config.enable_disk_offloading = True
    else:
        config.device = "cpu"
        config.enable_disk_offloading = True

    return config


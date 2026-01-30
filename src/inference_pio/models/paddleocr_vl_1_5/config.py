"""
PaddleOCR-VL-1.5 Configuration

This module provides the configuration for the PaddleOCR-VL-1.5 model in the
self-contained plugin architecture for the Inference-PIO system.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch

@dataclass
class PaddleOCRVL15Config:
    """
    Configuration class for the PaddleOCR-VL-1.5 model with all optimization parameters.
    """
    # Model identification
    model_name: str = "PaddleOCR-VL-1.5"
    model_path_h: str = "H:/PaddleOCR-VL-1.5"
    model_path_temp: str = "src/inference_pio/models/paddleocr_vl_1_5/temp_models/PaddleOCR-VL-1.5"
    hf_repo_id: str = "PaddlePaddle/PaddleOCR-VL-1.5"

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"

    # Optimization Floor Flags
    enable_paged_kv_cache: bool = True
    enable_flash_attention: bool = True
    enable_continuous_batching: bool = True
    enable_tensor_pagination: bool = True

    # Vision Optimizations
    enable_optimized_image_processing: bool = True
    enable_dynamic_resizing: bool = True
    spotting_upscale_threshold: int = 1500

    # Memory Management
    max_memory: Optional[Dict] = None
    low_cpu_mem_usage: bool = True

    # Task specific defaults
    default_task: str = "ocr"

    # Model specific architecture params (0.9B model)
    hidden_size: int = 2048 # Estimated/Placeholder, will be loaded from actual model config

    # Paged Attention settings
    paged_attention_page_size: int = 16
    max_num_batched_tokens: int = 4096

    def __post_init__(self):
        pass

def create_paddleocr_vl_1_5_config(**kwargs) -> PaddleOCRVL15Config:
    """
    Factory function to create a PaddleOCR-VL-1.5 configuration.
    """
    config = PaddleOCRVL15Config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

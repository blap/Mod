"""
Qwen3-VL-2B Configuration
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...common.config.model_config_base import BaseConfig, get_default_model_path

@dataclass
class Qwen3VL2BConfig(BaseConfig):
    model_path: str = ""
    model_name: str = "qwen3_vl_2b"

    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    intermediate_size: int = 5632
    layer_norm_eps: float = 1e-6

    # Vision
    vision_hidden_size: int = 1024
    vision_num_attention_heads: int = 16
    vision_num_hidden_layers: int = 24
    vision_patch_size: int = 14
    vision_image_size: int = 448
    vision_intermediate_size: int = 2816

    use_flash_attention_2: bool = True
    enable_intelligent_pagination: bool = True
    enable_disk_offloading: bool = True
    enable_continuous_nas: bool = False

    def __post_init__(self):
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)
        # super().__post_init__()

def create_qwen3_vl_2b_config(**kwargs):
    config = Qwen3VL2BConfig()
    for k, v in kwargs.items():
        if hasattr(config, k): setattr(config, k, v)
    return config

__all__ = ["Qwen3VL2BConfig", "create_qwen3_vl_2b_config"]

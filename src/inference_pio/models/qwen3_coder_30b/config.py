"""
Qwen3-Coder-30B Configuration - Self-Contained Version
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ...common.config.model_config_base import BaseConfig, get_default_model_path

@dataclass
class Qwen3Coder30BConfig(BaseConfig):
    model_path: str = ""
    model_name: str = "qwen3_coder_30b"

    # 30B MoE: 32 Q heads, 4 KV heads (GQA 8).
    # Hidden Size: 4096 (Estimated based on heads/dim)
    # Layers: 48
    # Context: 128K
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    num_hidden_layers: int = 48
    max_position_embeddings: int = 131072 # 128k context
    rope_theta: float = 1000000.0
    intermediate_size: int = 11008
    vocab_size: int = 152064
    layer_norm_eps: float = 1e-06

    # MoE Config
    num_experts: int = 128
    num_experts_per_tok: int = 8

    def __post_init__(self):
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)

__all__ = ["Qwen3Coder30BConfig"]

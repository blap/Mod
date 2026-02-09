"""
Qwen3-Coder-Next Configuration
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...common.config.model_config_base import BaseConfig, get_default_model_path

@dataclass
class Qwen3CoderNextConfig(BaseConfig):
    model_path: str = ""
    model_name: str = "qwen3_coder_next"

    hidden_size: int = 2048
    num_hidden_layers: int = 48
    max_position_embeddings: int = 262144
    vocab_size: int = 152064
    layer_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Hybrid Architecture
    hybrid_block_pattern: List[str] = field(default_factory=lambda: (["deltanet"] * 3 + ["attention"]) * 12)

    # Attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    attention_head_dim: int = 256
    attention_rope_dim: int = 64

    # DeltaNet
    deltanet_value_heads: int = 32
    deltanet_query_key_heads: int = 16
    deltanet_head_dim: int = 128

    # MoE
    num_experts: int = 512
    num_activated_experts: int = 10
    num_shared_experts: int = 1
    expert_intermediate_size: int = 512
    intermediate_size: int = 512

    def __post_init__(self):
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)

def create_qwen3_coder_next_config(**kwargs):
    config = Qwen3CoderNextConfig()
    for k, v in kwargs.items():
        if hasattr(config, k): setattr(config, k, v)
    return config

__all__ = ["Qwen3CoderNextConfig", "create_qwen3_coder_next_config"]

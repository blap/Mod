"""
Qwen3-Coder-Next Configuration - Self-Contained Version
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ...common.config.model_config_base import BaseConfig, get_default_model_path

@dataclass
class Qwen3CoderNextConfig(BaseConfig):
    model_path: str = ""
    model_name: str = "qwen3_coder_next"

    # Core
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    max_position_embeddings: int = 262144
    vocab_size: int = 152064
    rope_theta: float = 1000000.0
    layer_norm_eps: float = 1e-06
    intermediate_size: int = 512 # Expert Intermediate (from snippet)

    # Hybrid Pattern
    # 12 blocks of (3 Delta, 1 Attn)
    hybrid_block_pattern: List[str] = field(default_factory=lambda: ["deltanet", "deltanet", "deltanet", "attention"] * 12)

    # Gated Attention Params
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    attention_head_dim: int = 256
    attention_rope_dim: int = 64

    # Gated DeltaNet Params
    deltanet_query_key_heads: int = 16
    deltanet_value_heads: int = 32
    deltanet_head_dim: int = 128

    # MoE Params
    num_experts: int = 512
    num_experts_per_tok: int = 10
    num_shared_experts: int = 1

    def __post_init__(self):
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)

__all__ = ["Qwen3CoderNextConfig"]

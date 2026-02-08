"""
Qwen3-Coder-Next Configuration - Dependency Free Version
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import logging

# Removed torch import

try:
    from ...common.config import PretrainedConfig, BaseConfig, get_default_model_path
except ImportError:
    # Fallback to local definition if needed, but in full structure it should work
    from dataclasses import dataclass
    @dataclass
    class BaseConfig: pass
    def get_default_model_path(x): return ""

logger = logging.getLogger(__name__)

@dataclass
class Qwen3CoderNextConfig(BaseConfig):
    """
    Configuration class for the Qwen3-Coder-Next model with all optimization parameters.
    """
    # Model identification
    model_path: str = ""
    model_name: str = "qwen3_coder_next"

    # General Architecture Parameters
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
    intermediate_size: int = 512 # Compatibility alias

    # Standard outputs
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_return_dict: bool = True
    use_cache: bool = True

    # Generation Defaults
    pad_token_id: Optional[int] = 151643

    # Intelligent Cache settings (Simplified)
    intelligent_cache_enabled: bool = True
    intelligent_cache_max_size: int = 256 * 1024 * 1024

    def __post_init__(self):
        # Validation logic
        pass

def create_qwen3_coder_next_config(**kwargs) -> Qwen3CoderNextConfig:
    config = Qwen3CoderNextConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

__all__ = ["Qwen3CoderNextConfig", "create_qwen3_coder_next_config"]

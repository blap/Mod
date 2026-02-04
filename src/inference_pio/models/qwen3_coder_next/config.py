"""
Qwen3-Coder-Next Configuration

This module defines the configuration class for the Qwen3-Coder-Next model,
incorporating specialized settings for its hybrid DeltaNet/Attention/MoE architecture.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ...common.config.model_config_base import (
    BaseConfig,
    ModelConfigError,
    get_default_model_path,
)
from ...common.config.config_factory import register_model_config

@dataclass
class Qwen3CoderNextConfig(BaseConfig):
    """
    Configuration for Qwen3-Coder-Next model.
    """

    # Model Identity
    model_name: str = "qwen3_coder_next"
    model_path: str = ""  # Set in post_init

    # General Architecture Parameters
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    max_position_embeddings: int = 262144  # 256k context
    vocab_size: int = 152064  # Assuming same as 30B, verify if different
    layer_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Hybrid Architecture Layout
    # 12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))
    hybrid_block_pattern: List[str] = field(default_factory=lambda: (["deltanet"] * 3 + ["attention"]) * 12)

    # Gated Attention Parameters
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    attention_head_dim: int = 256
    attention_rope_dim: int = 64

    # Gated DeltaNet Parameters
    deltanet_value_heads: int = 32
    deltanet_query_key_heads: int = 16
    deltanet_head_dim: int = 128

    # Mixture of Experts (MoE) Parameters
    num_experts: int = 512
    num_activated_experts: int = 10
    num_shared_experts: int = 1
    expert_intermediate_size: int = 512

    # Optimization Flags
    enable_deltanet_kernel: bool = True
    enable_moe_kernel: bool = True
    enable_attention_kernel: bool = True

    # Standard outputs
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_return_dict: bool = True
    use_cache: bool = True

    # Explicitly disable Thinking Mode features
    thinking_mode: bool = False
    enable_thinking: bool = False

    # Standard Qwen Optimization Defaults
    use_flash_attention_2: bool = True
    use_fused_layer_norm: bool = True
    use_rms_norm: bool = True

    # Default Generation Parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 4096
    pad_token_id: Optional[int] = 151643 # Default Qwen pad token

    def __post_init__(self):
        """
        Post-initialization adjustments.
        """
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)

        super().__post_init__()

        # Validation
        if len(self.hybrid_block_pattern) != self.num_hidden_layers:
            # If pattern doesn't match layer count, try to repeat it or raise warning
            if len(self.hybrid_block_pattern) < self.num_hidden_layers:
                 # Auto-fill assuming the pattern repeats
                 base_pattern = ["deltanet"] * 3 + ["attention"]
                 repeats = self.num_hidden_layers // len(base_pattern)
                 self.hybrid_block_pattern = base_pattern * repeats

        # Ensure thinking mode is disabled
        self.thinking_mode = False
        self.enable_thinking = False

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return model-specific parameters."""
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_experts": self.num_experts,
            "num_activated_experts": self.num_activated_experts,
            "hybrid_pattern_len": len(self.hybrid_block_pattern)
        }

register_model_config("qwen3_coder_next", Qwen3CoderNextConfig)

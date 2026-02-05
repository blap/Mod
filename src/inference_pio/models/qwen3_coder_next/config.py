"""
Qwen3-Coder-Next Configuration - Self-Contained Version

This module provides the configuration for the Qwen3-Coder-Next model in the
self-contained plugin architecture for the Inference-PIO system. Each model plugin
is completely independent with its own configuration, tests, and benchmarks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

try:
    from inference_pio.common.config.model_config_base import (
        BaseConfig,
        ModelConfigError,
        get_default_model_path,
    )
except ImportError:
    # Fallback
    try:
        from ...common.model_config_base import (
            BaseConfig,
            ModelConfigError,
            get_default_model_path,
        )
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from src.inference_pio.common.config.model_config_base import (
            BaseConfig,
            ModelConfigError,
            get_default_model_path,
        )


@dataclass
class Qwen3CoderNextConfig(BaseConfig):
    """
    Configuration class for the Qwen3-Coder-Next model with all optimization parameters.

    This configuration class defines all the parameters needed for the Qwen3-Coder-Next model,
    including memory management, attention mechanisms, hybrid architecture settings,
    and hardware-specific optimizations.
    """

    # Model identification - override defaults
    model_path: str = ""  # Will be set in __post_init__ if not provided
    model_name: str = "qwen3_coder_next"

    # General Architecture Parameters
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    max_position_embeddings: int = 262144  # 256k context
    vocab_size: int = 152064
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
    pad_token_id: Optional[int] = 151643  # Default Qwen pad token

    # Memory optimization settings
    enable_disk_offloading: bool = False  # Enable for weak hardware
    offload_folder: str = "offload"

    # Hardware/Floor optimizations
    use_tensor_parallelism: bool = False

    # Attention mechanism settings
    use_sdpa: bool = True  # Scaled Dot Product Attention
    use_fused_rope: bool = True
    use_long_sequence_rope: bool = True  # float32 precision for long ctx

    # Fused Kernels
    use_fused_rms_norm: bool = True
    use_fused_mlp: bool = True  # SwiGLU

    # KV Cache
    use_paged_kv_cache: bool = True
    paged_attention_page_size: int = 16

    # Throughput
    use_continuous_batching: bool = True

    # Advanced
    use_quantization: bool = False
    quantization_bits: int = 8

    # System adaptivity
    enable_intelligent_pagination: bool = True

    # Intelligent Cache settings
    intelligent_cache_enabled: bool = True  # Enable intelligent caching system
    intelligent_cache_max_size: int = 1024 * 1024 * 256  # 256MB
    intelligent_cache_precision: str = "float16"  # Cache precision
    intelligent_cache_compression_enabled: bool = True  # Enable compression
    intelligent_cache_compression_method: str = "intelligent"  # Compression method
    intelligent_cache_policy: str = "intelligent"  # Cache policy: lru, fifo, lfu, predictive, intelligent
    intelligent_cache_enable_prefetching: bool = True  # Enable prefetching
    intelligent_cache_prefetch_distance: int = 1  # Distance for prefetching
    intelligent_cache_max_prefix_length: int = 2048  # Max length for cached prefixes
    intelligent_cache_min_prefix_length: int = 8  # Min length for cached prefixes
    intelligent_cache_warmup_threshold: int = 2  # Threshold for warming up cache entries
    intelligent_cache_prediction_horizon: int = 12  # Number of steps to predict ahead
    intelligent_cache_prediction_confidence_threshold: float = 0.68  # Minimum confidence for predictions
    intelligent_cache_enable_adaptive_eviction: bool = True  # Enable adaptive eviction
    intelligent_cache_enable_adaptive_prefetching: bool = True  # Enable adaptive prefetching
    intelligent_cache_adaptive_window_size: int = 150  # Window size for adaptive algorithms
    intelligent_cache_enable_performance_monitoring: bool = True  # Enable performance monitoring
    intelligent_cache_performance_log_interval: int = 75  # Log interval for performance metrics

    # Intelligent scheduling settings
    enable_intelligent_scheduling: bool = True
    intelligent_scheduling_max_concurrent_ops: int = 48
    intelligent_scheduling_policy: str = "intelligent"  # Options: "fifo", "priority", "round_robin", "predictive", "intelligent"
    intelligent_scheduling_enable_prediction: bool = True
    intelligent_scheduling_prediction_horizon: int = 18
    intelligent_scheduling_enable_adaptive: bool = True
    intelligent_scheduling_adaptive_window: int = 200
    intelligent_scheduling_enable_resource_opt: bool = True
    intelligent_scheduling_resource_buffer: float = 0.18
    intelligent_scheduling_enable_priority_boost: bool = True
    intelligent_scheduling_priority_decay: float = 0.91
    intelligent_scheduling_enable_load_balancing: bool = True
    intelligent_scheduling_load_balance_interval: float = 0.06
    intelligent_scheduling_performance_log_interval: int = 85

    # Cross-Alignment Optimization settings
    enable_cross_alignment: bool = True  # Enable cross-alignment optimization
    cross_alignment_temperature: float = 0.5  # Temperature for alignment computation
    cross_alignment_lambda: float = 0.1  # Weight for alignment loss in total loss
    use_cross_alignment_contrastive: bool = True  # Whether to use contrastive alignment loss
    enable_dynamic_cross_alignment: bool = True  # Whether to enable dynamic alignment based on input complexity
    cross_alignment_frequency: int = 10  # Frequency of alignment updates (every N steps)
    cross_alignment_threshold: float = 0.8  # Threshold for alignment quality
    use_cross_alignment_attention: bool = True  # Whether to use attention-based alignment
    use_cross_alignment_learned: bool = True  # Whether to use learned alignment projections
    cross_alignment_projection_dim: int = 512  # Dimension for alignment projections
    enable_cross_alignment_similarity: bool = True  # Whether to enable similarity-based alignment
    cross_alignment_method: str = "qwen3_coder_next_specific"  # Default alignment method

    def __post_init__(self):
        """
        Post-initialization adjustments.
        """
        # Set default model path if not provided
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)

        # Ensure the model path points to the H drive for Qwen3-Coder-Next model
        if (
            not self.model_path
            or "qwen3_coder_next" in self.model_path.lower()
            or "qwen3-coder-next" in self.model_path.lower()
            or "qwen3_coder" in self.model_path.lower()
            or "qwen3-coder" in self.model_path.lower()
        ):
            self.model_path = "H:/Qwen3-Coder-Next"

        # Call parent's post_init to validate config
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

        # Configure memory settings based on available system resources
        self._configure_memory_settings()

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return model-specific parameters."""
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_experts": self.num_experts,
            "num_activated_experts": self.num_activated_experts,
            "hybrid_pattern_len": len(self.hybrid_block_pattern),
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
        }

    def _configure_memory_settings(self):
        """
        Configure memory settings based on available system resources.
        """
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = 512 * 1024 * 1024  # 512MB reserved
                available_memory = gpu_memory - reserved_memory

                if available_memory > 0:
                    max_memory_gb = available_memory / (1024**3)
                    self.max_memory = {0: f"{max_memory_gb:.1f}GB", "cpu": "20GB"}
        except Exception:
            """Implement the required functionality."""
        # This is a placeholder implementation
        # In a real implementation, this would contain the actual logic
        return None


def create_qwen3_coder_next_config(**kwargs) -> Qwen3CoderNextConfig:
    """
    Factory function to create a Qwen3-Coder-Next configuration.
    """
    config = Qwen3CoderNextConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


# Configurável para integração com o sistema de configuração dinâmica
class Qwen3CoderNextDynamicConfig(Qwen3CoderNextConfig):
    """
    Extends the base Qwen3-Coder-Next configuration with dynamic configuration capabilities.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Adiciona capacidades de configuração dinâmica se necessário
        """Implement the required functionality."""
        # This is a placeholder implementation
        # In a real implementation, this would contain the actual logic
        return None


# Register this configuration with the factory
try:
    from inference_pio.common.config.config_factory import register_model_config
except ImportError:
    try:
        from ...common.config_factory import register_model_config
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from src.inference_pio.common.config.config_factory import register_model_config

register_model_config("qwen3_coder_next", Qwen3CoderNextConfig)


__all__ = ["Qwen3CoderNextConfig", "Qwen3CoderNextDynamicConfig", "create_qwen3_coder_next_config"]
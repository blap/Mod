"""
Qwen3-0.6B Configuration - Self-Contained Version

This module provides the configuration for the Qwen3-0.6B model in the
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
        from ...common.config.model_config_base import (
            BaseConfig,
            ModelConfigError,
            get_default_model_path,
        )
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from ...common.config.model_config_base import (
            BaseConfig,
            ModelConfigError,
            get_default_model_path,
        )


@dataclass
class Qwen3_0_6B_Config(BaseConfig):
    """
    Configuration class for the Qwen3-0.6B model with all optimization parameters.

    This configuration class defines all the parameters needed for the Qwen3-0.6B model,
    including memory management, attention mechanisms, thinking mode settings,
    and hardware-specific optimizations.
    """

    # Model identification - override defaults
    model_path: str = ""  # Will be set in __post_init__ if not provided
    model_name: str = "qwen3_0_6b"

    # Device settings - inherit from BaseConfig
    # device: Optional[str] = None  # Will be set dynamically (inherited)
    # device_map: str = "auto"     # (inherited)

    # Model architecture settings (Qwen3-0.6B spec)
    hidden_size: int = 896
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    num_hidden_layers: int = 24
    intermediate_size: int = 4864
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Data type - inherit from BaseConfig
    # torch_dtype: str = "float16"  # (inherited)

    # Thinking Mode Settings
    enable_thinking: bool = True  # Default enabled as per docs
    thinking_temperature: float = 0.6
    thinking_top_p: float = 0.95
    thinking_top_k: int = 20
    thinking_min_p: float = 0.0
    thinking_presence_penalty: float = 0.0  # Standard

    # Non-Thinking Mode Settings
    non_thinking_temperature: float = 0.7
    non_thinking_top_p: float = 0.8
    non_thinking_top_k: int = 20
    non_thinking_min_p: float = 0.0

    # Thinking Specific Optimizations
    enable_thought_compression: bool = True
    dynamic_repetition_penalty: bool = True
    thinking_repetition_penalty_value: float = 1.5

    # Memory optimization settings (The Floor) - some inherited from BaseConfig
    # gradient_checkpointing: bool = True  # (inherited)
    # use_cache: bool = True              # (inherited)
    # low_cpu_mem_usage: bool = True      # (inherited)
    # max_memory: Optional[Dict] = None    # (inherited)
    enable_disk_offloading: bool = False  # Enable for weak hardware
    offload_folder: str = "offload"

    # Hardware/Floor optimizations - some inherited from BaseConfig
    # use_tensor_parallelism: bool = False  # (inherited)

    # Attention mechanism settings (The Floor) - some inherited from BaseConfig
    # use_flash_attention_2: bool = True    # (inherited)
    # use_sdpa: bool = True                 # Scaled Dot Product Attention (inherited)
    use_fused_rope: bool = True
    use_long_sequence_rope: bool = True  # float32 precision for long ctx

    # Fused Kernels (The Floor)
    use_fused_rms_norm: bool = True
    use_fused_mlp: bool = True  # SwiGLU

    # KV Cache (The Floor) - some inherited from BaseConfig
    # use_paged_kv_cache: bool = True       # (inherited)
    # paged_attention_page_size: int = 16   # (inherited) Small page size for 0.6B model efficiency

    # Throughput (The Floor) - inherited from BaseConfig
    # use_continuous_batching: bool = True  # (inherited)

    # Advanced
    use_quantization: bool = False
    quantization_bits: int = 8

    # System adaptivity
    enable_intelligent_pagination: bool = True

    # Intelligent Cache settings
    intelligent_cache_enabled: bool = True  # Enable intelligent caching system
    intelligent_cache_max_size: int = 1024 * 1024 * 128  # 128MB for smaller model
    intelligent_cache_precision: str = "float16"  # Cache precision
    intelligent_cache_compression_enabled: bool = True  # Enable compression
    intelligent_cache_compression_method: str = "fp16"  # Compression method
    intelligent_cache_policy: str = "intelligent"  # Cache policy: lru, fifo, lfu, predictive, intelligent
    intelligent_cache_enable_prefetching: bool = True  # Enable prefetching
    intelligent_cache_prefetch_distance: int = 1  # Distance for prefetching
    intelligent_cache_max_prefix_length: int = 1024  # Max length for cached prefixes (smaller for smaller model)
    intelligent_cache_min_prefix_length: int = 4  # Min length for cached prefixes
    intelligent_cache_warmup_threshold: int = 2  # Threshold for warming up cache entries
    intelligent_cache_prediction_horizon: int = 8  # Number of steps to predict ahead
    intelligent_cache_prediction_confidence_threshold: float = 0.7  # Minimum confidence for predictions (higher for smaller model)
    intelligent_cache_enable_adaptive_eviction: bool = True  # Enable adaptive eviction
    intelligent_cache_enable_adaptive_prefetching: bool = True  # Enable adaptive prefetching
    intelligent_cache_adaptive_window_size: int = 50  # Window size for adaptive algorithms (smaller for smaller model)
    intelligent_cache_enable_performance_monitoring: bool = True  # Enable performance monitoring
    intelligent_cache_performance_log_interval: int = 50  # Log interval for performance metrics

    # Intelligent scheduling settings
    enable_intelligent_scheduling: bool = True
    intelligent_scheduling_max_concurrent_ops: int = 12
    intelligent_scheduling_policy: str = "intelligent"  # Options: "fifo", "priority", "round_robin", "predictive", "intelligent"
    intelligent_scheduling_enable_prediction: bool = True
    intelligent_scheduling_prediction_horizon: int = 8
    intelligent_scheduling_enable_adaptive: bool = True
    intelligent_scheduling_adaptive_window: int = 75
    intelligent_scheduling_enable_resource_opt: bool = True
    intelligent_scheduling_resource_buffer: float = 0.08
    intelligent_scheduling_enable_priority_boost: bool = True
    intelligent_scheduling_priority_decay: float = 0.95
    intelligent_scheduling_enable_load_balancing: bool = True
    intelligent_scheduling_load_balance_interval: float = 0.12
    intelligent_scheduling_performance_log_interval: int = 40

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
    cross_alignment_method: str = "qwen3_small_specific"  # Default alignment method

    def __post_init__(self):
        """Post-initialization adjustments."""
        # Set default model path if not provided
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)

        # Ensure the model path points to the H drive for Qwen3-0.6B model
        if not self.model_path or "qwen3_0_6b" in self.model_path.lower():
            self.model_path = "H:/Qwen3-0.6B"

        # Call parent's post_init to validate config
        super().__post_init__()

        # Configure memory settings based on available system resources
        self._configure_memory_settings__()

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return model-specific parameters."""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
        }

    def _configure_memory_settings__(self):
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


def create_qwen3_0_6b_config(**kwargs) -> Qwen3_0_6B_Config:
    """
    Factory function to create a Qwen3-0.6B configuration.
    """
    config = Qwen3_0_6B_Config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


# Configurável para integração com o sistema de configuração dinâmica
class Qwen3_0_6B_DynamicConfig(Qwen3_0_6B_Config):
    """
    Extends the base Qwen3-0.6B configuration with dynamic configuration capabilities.
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

register_model_config("qwen3_0_6b", Qwen3_0_6B_Config)


__all__ = ["Qwen3_0_6B_Config", "Qwen3_0_6B_DynamicConfig", "create_qwen3_0_6b_config"]

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
    from ...common.model_config_base import (
        BaseConfig,
        ModelConfigError,
        get_default_model_path,
    )
except ImportError:
    # Fallback para quando os imports relativos não funcionam
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from src.common.model_config_base import (
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
            pass


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
        pass


# Register this configuration with the factory
try:
    from ...common.config_factory import register_model_config
except ImportError:
    # Fallback para quando os imports relativos não funcionam
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from src.common.config_factory import register_model_config

register_model_config("qwen3_0_6b", Qwen3_0_6B_Config)


__all__ = ["Qwen3_0_6B_Config", "Qwen3_0_6B_DynamicConfig", "create_qwen3_0_6b_config"]

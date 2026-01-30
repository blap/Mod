"""
Qwen3-0.6B Configuration - Self-Contained Version

This module provides the configuration for the Qwen3-0.6B model in the
self-contained plugin architecture for the Inference-PIO system.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Qwen3_0_6B_Config:
    """
    Configuration class for the Qwen3-0.6B model with all optimization parameters.

    This configuration class defines all the parameters needed for the Qwen3-0.6B model,
    including memory management, attention mechanisms, thinking mode settings,
    and hardware-specific optimizations.
    """
    # Model identification
    model_path: str = "H:/Qwen/Qwen3-0.6B"  # Default local path

    # Device settings
    device: Optional[str] = None  # Will be set dynamically
    device_map: str = "auto"

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

    # Data type
    torch_dtype: str = "float16"

    # Thinking Mode Settings
    enable_thinking: bool = True  # Default enabled as per docs
    thinking_temperature: float = 0.6
    thinking_top_p: float = 0.95
    thinking_top_k: int = 20
    thinking_min_p: float = 0.0
    thinking_presence_penalty: float = 0.0 # Standard

    # Non-Thinking Mode Settings
    non_thinking_temperature: float = 0.7
    non_thinking_top_p: float = 0.8
    non_thinking_top_k: int = 20
    non_thinking_min_p: float = 0.0

    # Thinking Specific Optimizations
    enable_thought_compression: bool = True
    dynamic_repetition_penalty: bool = True
    thinking_repetition_penalty_value: float = 1.5

    # Memory optimization settings (The Floor)
    gradient_checkpointing: bool = True
    use_cache: bool = True
    low_cpu_mem_usage: bool = True
    max_memory: Optional[Dict] = None
    enable_disk_offloading: bool = False # Enable for weak hardware
    offload_folder: str = "offload"

    # Hardware/Floor optimizations
    use_tensor_parallelism: bool = False

    # Attention mechanism settings (The Floor)
    use_flash_attention_2: bool = True
    use_sdpa: bool = True # Scaled Dot Product Attention
    use_fused_rope: bool = True
    use_long_sequence_rope: bool = True # float32 precision for long ctx

    # Fused Kernels (The Floor)
    use_fused_rms_norm: bool = True
    use_fused_mlp: bool = True # SwiGLU

    # KV Cache (The Floor)
    use_paged_kv_cache: bool = True
    paged_attention_page_size: int = 16 # Small page size for 0.6B model efficiency

    # Throughput (The Floor)
    use_continuous_batching: bool = True

    # Advanced
    use_quantization: bool = False
    quantization_bits: int = 8

    # System adaptivity
    enable_intelligent_pagination: bool = True

    def __post_init__(self):
        """Post-initialization adjustments."""
        pass

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
                    max_memory_gb = available_memory / (1024 ** 3)
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

__all__ = [
    "Qwen3_0_6B_Config",
    "create_qwen3_0_6b_config"
]

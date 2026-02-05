"""
Optimization Configuration for Inference-PIO System

This module provides configuration classes for various optimization strategies.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class OptimizationConfig:
    """
    Base configuration class for optimization strategies in the Inference-PIO system.
    """

    # Memory optimization settings
    gradient_checkpointing: bool = True
    use_cache: bool = True
    torch_dtype: str = "float16"
    device_map: str = "auto"
    low_cpu_mem_usage: bool = True
    max_memory: Optional[Dict] = None

    # Attention mechanism settings
    use_flash_attention_2: bool = True
    use_sdpa: bool = True
    use_fused_rope: bool = True
    use_long_sequence_rope: bool = True

    # Fused kernels settings
    use_fused_rms_norm: bool = True
    use_fused_mlp: bool = True

    # KV Cache settings
    use_paged_kv_cache: bool = True
    paged_attention_page_size: int = 16

    # Throughput settings
    use_continuous_batching: bool = True

    # Advanced optimization settings
    use_quantization: bool = False
    quantization_bits: int = 8
    use_tensor_parallelism: bool = False
    tensor_parallel_size: int = 1

    # System adaptivity settings
    enable_intelligent_pagination: bool = True

    def _get_h_drive_path(self):
        """Get the H drive path for this model if available."""
        import os
        import platform
        
        # Determine the model-specific path on H drive
        model_name_clean = self.model_name.replace("_", "-").replace(" ", "")
        h_drive_paths = [
            f"H:/{model_name_clean}",
            f"H:/models/{model_name_clean}",
            f"H:/AI/models/{model_name_clean}",
        ]
        
        # Check platform-specific paths
        if platform.system() == 'Windows':
            for path in h_drive_paths:
                if os.path.exists(path):
                    return path
        else:
            # For Linux/WSL, common mount points
            mount_points = ['/mnt/h', '/media/h', '/drives/h']
            for mount_point in mount_points:
                for path in h_drive_paths:
                    alt_path = path.replace('H:/', f'{mount_point}/')
                    if os.path.exists(alt_path):
                        return alt_path
        
        return None


    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def save_to_file(self, file_path: str):
        """Save configuration to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def load_from_file(cls, file_path: str):
        """Load configuration from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)


@dataclass
class GLM47OptimizationConfig(OptimizationConfig):
    """Optimization configuration specific to GLM-4.7 model."""

    # GLM-specific optimizations
    use_glm_attention_patterns: bool = True
    glm_attention_pattern_sparsity: float = 0.3
    glm_attention_window_size: int = 1024
    use_glm_ffn_optimization: bool = True
    glm_ffn_expansion_ratio: float = 2.6
    glm_ffn_group_size: int = 128
    use_glm_memory_efficient_kv: bool = True
    glm_kv_cache_compression_ratio: float = 0.5
    use_glm_layer_norm_fusion: bool = True
    use_glm_residual_connection_optimization: bool = True
    use_glm_quantization: bool = True
    glm_weight_bits: int = 4
    glm_activation_bits: int = 8


@dataclass
class Qwen34BOptimizationConfig(OptimizationConfig):
    """Optimization configuration specific to Qwen3-4B model."""

    # Qwen3-specific optimizations
    use_qwen3_attention_optimizations: bool = True
    use_qwen3_kv_cache_optimizations: bool = True
    use_qwen3_instruction_optimizations: bool = True
    use_qwen3_rope_optimizations: bool = True
    use_qwen3_gqa_optimizations: bool = True
    qwen3_attention_sparsity_ratio: float = 0.3
    qwen3_kv_cache_compression_ratio: float = 0.6
    qwen3_instruction_attention_scaling: float = 1.2


@dataclass
class Qwen3CoderOptimizationConfig(OptimizationConfig):
    """Optimization configuration specific to Qwen3-Coder model."""

    # Qwen3-Coder specific optimizations
    use_qwen3_coder_attention_optimizations: bool = True
    use_qwen3_coder_kv_cache_optimizations: bool = True
    use_qwen3_coder_code_optimizations: bool = True
    use_qwen3_coder_syntax_highlighting: bool = True
    qwen3_coder_attention_sparsity_ratio: float = 0.3
    qwen3_coder_kv_cache_compression_ratio: float = 0.6
    qwen3_coder_syntax_attention_scaling: float = 1.2


@dataclass
class Qwen3VLOptimizationConfig(OptimizationConfig):
    """Optimization configuration specific to Qwen3-VL model."""

    # Qwen3-VL specific optimizations
    use_qwen3_vl_attention_optimizations: bool = True
    use_qwen3_vl_kv_cache_optimizations: bool = True
    use_qwen3_vl_vision_optimizations: bool = True
    use_qwen3_vl_cross_modal_optimizations: bool = True
    qwen3_vl_attention_sparsity_ratio: float = 0.3
    qwen3_vl_kv_cache_compression_ratio: float = 0.6
    qwen3_vl_cross_modal_attention_scaling: float = 1.2


from enum import Enum


class ModelFamily(Enum):
    """
    Enum for different model families supported by the optimization system.
    """

    GENERAL = "general"
    GLM = "glm"
    QWEN = "qwen"
    LLAMA = "llama"
    MISTRAL = "mistral"


class ModelOptimizationConfig:
    """
    Configuration for model-specific optimizations.
    """

    def __init__(
        self,
        model_family: ModelFamily,
        optimizations: list = None,
        priority_order: list = None,
    ):
        self.model_family = model_family
        self.optimizations = optimizations or []
        self.priority_order = priority_order or []
        self.enabled = True


def get_config_manager():
    """
    Placeholder function to get the configuration manager.
    This would be implemented in a real system to manage optimization configurations.
    """

    # In a real implementation, this would return an instance of a config manager
    # For now, we'll return a mock object that satisfies the import
    class MockConfigManager:
        def set_active_profile(self, profile_name):
            return True

        def apply_profile_to_model_config(self, model_family):
            return ModelOptimizationConfig(model_family)

    return MockConfigManager()


__all__ = [
    "OptimizationConfig",
    "GLM47OptimizationConfig",
    "Qwen34BOptimizationConfig",
    "Qwen3CoderOptimizationConfig",
    "Qwen3VLOptimizationConfig",
    "get_config_manager",
    "ModelFamily",
    "ModelOptimizationConfig",
]

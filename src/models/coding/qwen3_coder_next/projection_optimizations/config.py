"""
Configuration for Qwen3-Coder-Next Projection Optimizations

This module defines the configuration settings for Qwen3-Coder-Next projection optimizations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Qwen3CoderNextProjectionConfig:
    """
    Configuration for Qwen3-Coder-Next projection optimizations.
    """
    # Basic projection settings
    use_projection_layer_optimization: bool = True
    projection_layer_use_bias: bool = True
    projection_layer_activation: str = "silu"
    projection_layer_dropout: float = 0.1
    projection_layer_use_residual: bool = True
    
    # Low-rank projection settings
    projection_layer_use_low_rank: bool = False
    projection_layer_low_rank_dim: Optional[int] = None
    
    # Normalization settings
    projection_layer_use_group_norm: bool = False
    projection_layer_group_norm_num_groups: int = 32
    
    # Architecture settings
    projection_layer_intermediate_dim: Optional[int] = None
    projection_layer_num_layers: int = 2
    
    # Cross-attention settings
    projection_layer_use_cross_attention: bool = False
    projection_layer_cross_attention_heads: int = 8
    
    # Performance settings
    use_memory_efficient_attention: bool = True
    attention_chunk_size: int = 4096
    
    # Quantization settings
    use_quantized_projections: bool = False
    quantization_bits: int = 8
    
    # Sparsity settings
    use_sparse_projections: bool = False
    sparsity_ratio: float = 0.1
    
    # Next-generation settings
    use_adaptive_computation: bool = True
    use_mixed_precision: bool = True
    use_sparse_attention: bool = True
    sparse_attention_density: float = 0.5
    use_dynamic_routing: bool = True
    num_experts: int = 4
    expert_capacity: int = 1024
    
    # Code-specific settings
    use_code_specific_optimizations: bool = True
    code_context_window_size: int = 128
    use_syntax_aware_projections: bool = True
    syntax_embedding_dim: int = 64
    use_advanced_code_features: bool = True
    code_feature_extraction_depth: int = 3


# Default configuration
DEFAULT_QWEN3_CODER_NEXT_PROJECTION_CONFIG = Qwen3CoderNextProjectionConfig()

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


def get_qwen3_coder_next_projection_config(**kwargs) -> Qwen3CoderNextProjectionConfig:
    """
    Get Qwen3-Coder-Next projection configuration with optional overrides.
    
    Args:
        **kwargs: Configuration overrides
        
    Returns:
        Qwen3CoderNextProjectionConfig: Configuration object
    """
    config = DEFAULT_QWEN3_CODER_NEXT_PROJECTION_CONFIG
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
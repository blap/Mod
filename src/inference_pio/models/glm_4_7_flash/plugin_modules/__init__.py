"""
GLM-4.7 Plugin Package - Self-Contained Version

This module provides the initialization for the GLM-4.7 model plugin package.
"""

from .glm47_attention import GLM47FlashAttention2, create_glm47_flash_attention_2
from .glm47_bias_removal import (
    apply_bias_removal_during_model_loading,
    apply_bias_removal_to_model,
)
from .glm47_cuda_kernels import apply_glm47_optimizations_to_model
from .glm47_fused_layers import (
    FusedLayerNorm,
    FusedRMSNorm,
    replace_layer_norm_in_model,
    replace_rms_norm_in_model,
)
from .glm47_kv_cache import apply_compressed_kv_cache_to_model
from .glm47_multi_query_attention import (
    GLM47GroupedQueryAttention,
    GLM47MultiQueryAttention,
    create_mqa_gqa_attention,
)
from .glm47_paged_attention import GLM47PagedAttention, create_glm47_paged_attention
from .glm47_prefix_cache import apply_prefix_cache_to_model
from .glm47_rotary_embeddings import GLM47RotaryEmbedding
from .glm47_sliding_window_attention import (
    GLM47SlidingWindowAttention,
    create_glm47_sliding_window_attention,
)
from .glm47_sparse_attention import GLM47SparseAttention, create_glm47_sparse_attention
from .glm47_specific_optimizations import (
    GLM47OptimizationConfig,
    apply_glm47_specific_optimizations,
    get_glm47_optimization_report,
)
from .glm47_tensor_parallel import safe_convert_to_tensor_parallel

__all__ = [
    "GLM47FlashAttention2",
    "create_glm47_flash_attention_2",
    "GLM47SparseAttention",
    "create_glm47_sparse_attention",
    "GLM47SlidingWindowAttention",
    "create_glm47_sliding_window_attention",
    "GLM47MultiQueryAttention",
    "GLM47GroupedQueryAttention",
    "create_mqa_gqa_attention",
    "GLM47PagedAttention",
    "create_glm47_paged_attention",
    "GLM47RotaryEmbedding",
    "FusedLayerNorm",
    "FusedRMSNorm",
    "replace_layer_norm_in_model",
    "replace_rms_norm_in_model",
    "apply_bias_removal_to_model",
    "apply_bias_removal_during_model_loading",
    "safe_convert_to_tensor_parallel",
    "apply_compressed_kv_cache_to_model",
    "apply_prefix_cache_to_model",
    "apply_glm47_optimizations_to_model",
    "apply_glm47_specific_optimizations",
    "get_glm47_optimization_report",
    "GLM47OptimizationConfig",
]

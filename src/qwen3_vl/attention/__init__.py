"""
Complete Attention System for Qwen3-VL Model
Combines all attention mechanisms with lifecycle management and hardware-specific optimizations.
"""
from .consolidated_attention_complete import (
    StandardAttention, FlashAttention2, SM61OptimizedFlashAttention2, TrueSparseAttention, 
    BlockSparseAttention, DynamicSparseAttention, Qwen3VLAttention, AttentionMechanismSelector, 
    Qwen3VLRotaryEmbedding, rotate_half, apply_rotary_pos_emb, repeat_kv
)

# Import the lifecycle management components
from .consolidated_tensor_lifecycle import (
    TensorLifecycleTracker, LifetimePredictor, AccessPatternAnalyzer, 
    EnhancedPredictiveTensorLifecycleManager, create_optimized_lifecycle_manager, 
    integrate_with_existing_systems, TensorType, TensorState, TensorMetadata
)

# Import sparse attention components
from .consolidated_sparse_attention import (
    VectorizedSparseAttention, OptimizedDynamicSparseAttention, VisionDynamicSparseAttention, 
    OptimizedVisionDynamicSparseAttention, BlockSparseAttentionFactory
)

# Import flash attention components
from .consolidated_flash_attention import (
    KVCacheOptimizedFlashAttention2, SM61OptimizedKVCacheFlashAttention2, 
    FlashAttention2TransformerLayer, create_optimized_flash_attention_with_cache
)

# Import rotary embeddings components
from .consolidated_rotary_embeddings import (
    OptimizedRotaryEmbedding, ApproximatedRotaryEmbedding, CachedRotaryEmbedding, 
    InterpolatedRotaryEmbedding, RotaryEmbeddingOptimizer
)

# Import linear attention components
from .linear_attention import PerformerAttention

# Import predictive tensor lifecycle components
from .predictive_tensor_lifecycle_manager import IntegratedTensorLifecycleManager

__all__ = [
    # Core attention mechanisms
    'StandardAttention',
    'FlashAttention2',
    'SM61OptimizedFlashAttention2',
    'TrueSparseAttention',
    'BlockSparseAttention',
    'DynamicSparseAttention',
    'Qwen3VLAttention',

    # Attention mechanism selector/factory
    'AttentionMechanismSelector',

    # Rotary embeddings
    'Qwen3VLRotaryEmbedding',
    'rotate_half',
    'apply_rotary_pos_emb',
    'repeat_kv',

    # Tensor lifecycle management
    'TensorLifecycleTracker',
    'LifetimePredictor',
    'AccessPatternAnalyzer',
    'EnhancedPredictiveTensorLifecycleManager',
    'create_optimized_lifecycle_manager',
    'integrate_with_existing_systems',
    'TensorType',
    'TensorState',
    'TensorMetadata',

    # Additional attention components
    'VectorizedSparseAttention',
    'OptimizedDynamicSparseAttention',
    'VisionDynamicSparseAttention',
    'OptimizedVisionDynamicSparseAttention',
    'BlockSparseAttentionFactory',
    'KVCacheOptimizedFlashAttention2',
    'SM61OptimizedKVCacheFlashAttention2',
    'FlashAttention2TransformerLayer',
    'create_optimized_flash_attention_with_cache',

    # Additional rotary embeddings
    'OptimizedRotaryEmbedding',
    'ApproximatedRotaryEmbedding',
    'CachedRotaryEmbedding',
    'InterpolatedRotaryEmbedding',
    'RotaryEmbeddingOptimizer',

    # Linear attention
    'PerformerAttention',

    # Predictive tensor lifecycle
    'IntegratedTensorLifecycleManager'
]
"""
Memory configuration classes for Qwen3-VL model components.

This module contains configuration classes specifically for memory management
components with clear separation of concerns.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryConfig:
    """
    Configuration class for memory management components.
    """
    # Memory pooling configuration
    use_memory_pooling: bool = True
    memory_pool_initial_size: int = 1024 * 1024 * 256  # 256MB initial pool
    memory_pool_max_size: int = 1024 * 1024 * 1024  # 1GB max pool
    memory_pool_growth_factor: float = 1.5  # Growth factor when expanding pool
    use_buddy_allocation: bool = True  # Use buddy allocation system
    memory_defragmentation_enabled: bool = True  # Enable memory defragmentation
    memory_defragmentation_threshold: float = 0.3  # Defragment when fragmentation > 30%

    # KV cache optimization configuration
    kv_cache_strategy: str = "hybrid"  # Options: "low_rank", "sliding_window", "hybrid"
    use_low_rank_kv_cache: bool = True  # Enable low-rank KV cache compression
    kv_cache_window_size: int = 1024  # Window size for sliding window attention
    kv_low_rank_dimension: int = 64  # Rank for low-rank approximation
    kv_cache_max_length: int = 32768  # Maximum sequence length for KV cache

    # Memory optimization configuration
    use_gradient_checkpointing: bool = True  # Memory efficiency for training
    use_activation_sparsity: bool = False  # Activation sparsity for memory reduction
    sparsity_ratio: float = 0.5  # Ratio of activations to keep (1 - sparsity)
    use_pre_allocated_tensors: bool = True  # Use pre-allocated tensor caches
    pre_allocated_cache_size: int = 1024 * 1024 * 128  # 128MB for pre-allocated tensors

    # Memory management configuration
    memory_efficient_attention: bool = True  # Use memory-efficient attention
    use_vision_memory_optimization: bool = True  # Optimize vision encoder memory
    vision_memory_chunk_size: int = 512  # Chunk size for vision memory processing
    use_tensor_fusion: bool = True  # Fuse tensors where possible to reduce overhead

    def __post_init__(self):
        """Validate memory configuration after initialization."""
        if self.kv_cache_window_size <= 0:
            raise ValueError(f"kv_cache_window_size must be positive, got {self.kv_cache_window_size}")

        if self.kv_low_rank_dimension <= 0:
            raise ValueError(f"kv_low_rank_dimension must be positive, got {self.kv_low_rank_dimension}")

        if not 0.0 <= self.sparsity_ratio <= 1.0:
            raise ValueError(f"sparsity_ratio must be between 0.0 and 1.0, got {self.sparsity_ratio}")

        if self.memory_pool_growth_factor <= 1.0:
            raise ValueError(f"memory_pool_growth_factor must be > 1.0, got {self.memory_pool_growth_factor}")
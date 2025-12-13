"""
Attention configuration classes for Qwen3-VL model components.

This module contains configuration classes specifically for attention mechanisms
with clear separation of concerns.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AttentionConfig:
    """
    Configuration class for attention mechanisms.
    """
    # Basic attention configuration
    attention_implementation: str = "eager"  # Options: "eager", "flash_attention_2", "sdpa", "kv_cache_optimized"
    use_flash_attention_2: bool = False  # Enable FlashAttention-2
    flash_attention_causal: bool = True  # Whether to use causal masking in flash attention
    attention_dropout_prob: float = 0.0  # Dropout probability for attention weights
    use_memory_efficient_attention: bool = False  # Use memory-efficient attention implementations

    # Dynamic sparse attention configuration (Phase 7)
    use_dynamic_sparse_attention: bool = False  # Enable dynamic sparse attention with learned routing
    sparse_attention_sparsity_ratio: float = 0.5  # Ratio of tokens to attend to (top-k selection)
    vision_sparse_attention_sparsity_ratio: float = 0.4  # Sparsity ratio for vision attention
    sparse_attention_pattern: str = "top_k"  # Options: "top_k", "random", "local", "strided"
    sparse_attention_num_blocks: int = 32  # Number of blocks for block-sparse attention

    # Rotary embedding configuration
    rope_theta: float = 1000000.0  # Base value for RoPE calculation
    use_rotary_embedding: bool = True  # Enable rotary embeddings
    use_approximated_rotary_embeddings: bool = False  # Use approximated rotary embeddings for speed
    rotary_embedding_scaling_factor: float = 1.0  # Scaling factor for rotary embeddings

    # Multi-head attention configuration
    num_attention_heads: int = 32  # Number of attention heads (preserved for full capacity)
    num_key_value_heads: Optional[int] = None  # For GQA (Grouped Query Attention)
    head_dim: Optional[int] = None  # Dimension of each attention head

    # Advanced attention features (Phase 9)
    use_block_sparse_attention: bool = False  # Enable block-sparse attention patterns
    block_sparse_block_size: int = 64  # Size of blocks for block-sparse attention
    use_learned_attention_routing: bool = False  # Use learned routing for attention patterns
    learned_routing_temperature: float = 1.0  # Temperature for learned routing softmax

    def __post_init__(self):
        """Validate attention configuration after initialization."""
        valid_implementations = ["eager", "flash_attention_2", "sdpa", "kv_cache_optimized"]
        if self.attention_implementation not in valid_implementations:
            raise ValueError(f"attention_implementation must be one of {valid_implementations}, got {self.attention_implementation}")

        if not 0.0 <= self.attention_dropout_prob <= 1.0:
            raise ValueError(f"attention_dropout_prob must be between 0.0 and 1.0, got {self.attention_dropout_prob}")

        if not 0.0 <= self.sparse_attention_sparsity_ratio <= 1.0:
            raise ValueError(f"sparse_attention_sparsity_ratio must be between 0.0 and 1.0, got {self.sparse_attention_sparsity_ratio}")

        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")

        if self.head_dim is not None and self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
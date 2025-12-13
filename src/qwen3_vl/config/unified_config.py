"""
Unified configuration class for Qwen3-VL model with all configuration options in a single file.

This configuration maintains full capacity with 32 transformer layers
and 32 attention heads while enabling efficiency optimizations.
Eliminates the need for multiple configuration files by including all
configuration options directly in this single class.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Qwen3VLConfig:
    """
    Main unified configuration class for Qwen3-VL model.

    This configuration maintains full capacity with 32 transformer layers
    and 32 attention heads while enabling efficiency optimizations.
    All configuration options are defined directly in this class to 
    eliminate dependencies on multiple configuration files.
    """
    # Language model configuration
    vocab_size: int = 152064  # Standard for Qwen models
    hidden_size: int = 2048
    num_hidden_layers: int = 32  # Preserved for full capacity
    num_attention_heads: int = 32  # Preserved for full capacity
    num_key_value_heads: Optional[int] = None  # If using GQA
    intermediate_size: int = 11008  # Standard FFN intermediate size
    hidden_act: str = "silu"
    hidden_dropout_prob: float = 0.0
    max_position_embeddings: int = 32768  # Standard for Qwen models
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    pad_token_id: int = 0
    tie_word_embeddings: bool = False

    # Vision model configuration
    vision_model_type: str = "clip_vision_model"
    vision_hidden_size: int = 1152
    vision_num_hidden_layers: int = 24
    vision_num_attention_heads: int = 16
    vision_intermediate_size: int = 4304
    vision_patch_size: int = 14
    vision_image_size: int = 448
    vision_window_size: int = 14
    vision_num_channels: int = 3
    vision_qkv_bias: bool = True  # Bias for QKV projection in vision attention

    # Multimodal configuration
    num_query_tokens: int = 64  # Number of query tokens for vision-language fusion
    vision_projection_dim: int = 2048
    language_projection_dim: int = 2048

    # Additional parameters
    torch_dtype: Optional[str] = None  # e.g., "float16", "bfloat16"
    pretraining_tp: int = 1  # Parameter for pretraining tensor parallelism
    use_cache: bool = True

    # Parameter-efficient adaptation parameters
    use_adapters: bool = False  # Enable parameter-efficient adaptation

    # Early exit parameters
    use_early_exit: bool = False  # Enable early exit mechanisms
    exit_threshold: float = 0.8  # Confidence threshold for early exit

    # Adaptive depth parameters (Phase 7)
    use_adaptive_depth: bool = False  # Enable adaptive depth networks with input complexity assessment
    use_vision_adaptive_depth: bool = False  # Enable adaptive depth for vision transformer
    use_multimodal_adaptive_depth: bool = False  # Enable adaptive depth for multimodal fusion
    min_depth_ratio: float = 0.2  # Minimum ratio of layers to use (20%)
    max_depth_ratio: float = 1.0  # Maximum ratio of layers to use (100%)
    vision_min_depth_ratio: float = 0.3  # Minimum ratio of vision layers to use (30%)
    vision_max_depth_ratio: float = 1.0  # Maximum ratio of vision layers to use (100%)
    depth_temperature: float = 1.0  # Temperature for soft depth selection

    # Context-adaptive positional encoding parameters (Phase 7)
    use_context_adaptive_positional_encoding: bool = False  # Enable learned context-adaptive positional representations
    use_cross_modal_positional_encoding: bool = False  # Enable cross-modal context-adaptive positional encoding

    # Conditional feature extraction parameters (Phase 7)
    use_conditional_feature_extraction: bool = False  # Enable conditional feature extraction based on input modality

    # Advanced features and optimizations
    use_moe: bool = False  # Mixture of Experts
    moe_num_experts: int = 4  # Number of experts for MoE
    moe_top_k: int = 2  # Top-k routing for MoE
    use_sparsity: bool = False  # Activation sparsity
    sparsity_ratio: float = 0.5  # Ratio of activations to keep (1 - sparsity)
    use_adaptive_precision: bool = False  # Adaptive precision optimization
    use_cross_modal_compression: bool = False  # Cross-modal compression
    compression_ratio: float = 0.5  # Compression ratio
    use_cross_layer_memory_sharing: bool = False  # Cross-layer memory sharing
    use_mixed_precision: bool = False  # Mixed precision training
    gpu_memory_size: int = 6 * 1024 * 1024 * 1024  # GPU memory size in bytes (6GB)
    use_inference_memory_efficient: bool = False  # Memory efficient inference

    # Attention configuration
    attention_implementation: str = "eager"  # Options: "eager", "flash_attention_2", "sdpa", "kv_cache_optimized"
    use_flash_attention_2: bool = False  # Enable FlashAttention-2
    flash_attention_causal: bool = True  # Whether to use causal masking in flash attention
    attention_dropout_prob: float = 0.0  # Dropout probability for attention weights
    use_memory_efficient_attention: bool = False  # Use memory-efficient attention implementations
    use_dynamic_sparse_attention: bool = False  # Enable dynamic sparse attention with learned routing
    sparse_attention_sparsity_ratio: float = 0.5  # Ratio of tokens to attend to (top-k selection)
    vision_sparse_attention_sparsity_ratio: float = 0.4  # Sparsity ratio for vision attention
    sparse_attention_pattern: str = "top_k"  # Options: "top_k", "random", "local", "strided"
    sparse_attention_num_blocks: int = 32  # Number of blocks for block-sparse attention
    rope_theta: float = 1000000.0  # Base value for RoPE calculation
    use_rotary_embedding: bool = True  # Enable rotary embeddings
    use_approximated_rotary_embeddings: bool = False  # Use approximated rotary embeddings for speed
    rotary_embedding_scaling_factor: float = 1.0  # Scaling factor for rotary embeddings
    head_dim: Optional[int] = None  # Dimension of each attention head
    use_block_sparse_attention: bool = False  # Enable block-sparse attention patterns
    block_sparse_block_size: int = 64  # Size of blocks for block-sparse attention
    use_learned_attention_routing: bool = False  # Use learned routing for attention patterns
    learned_routing_temperature: float = 1.0  # Temperature for learned routing softmax

    # Memory management configuration
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
    memory_efficient_attention: bool = True  # Use memory-efficient attention
    use_vision_memory_optimization: bool = True  # Optimize vision encoder memory
    vision_memory_chunk_size: int = 512  # Chunk size for vision memory processing
    use_tensor_fusion: bool = True  # Fuse tensors where possible to reduce overhead
    use_pre_allocated_tensors: bool = True  # Use pre-allocated tensor caches
    pre_allocated_cache_size: int = 1024 * 1024 * 128  # 128MB for pre-allocated tensors

    # Hardware-specific configuration
    hardware_compute_capability: tuple = (6, 1)  # Default for NVIDIA SM61
    memory_size_gb: float = 8.0  # Default memory size in GB
    use_nvme_cache: bool = True  # Use NVMe for caching
    
    # Routing configuration (if needed)
    routing_method: str = "static"  # Options: "static", "dynamic", "learned"
    routing_top_k: int = 2  # For learned routing
    routing_temperature: float = 1.0  # For learned routing

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure full capacity is preserved
        if self.num_hidden_layers != 32:
            raise ValueError(f"num_hidden_layers must be 32 to preserve full capacity, got {self.num_hidden_layers}")

        if self.num_attention_heads != 32:
            raise ValueError(f"num_attention_heads must be 32 to preserve full capacity, got {self.num_attention_heads}")

        # Validate memory configuration
        if self.kv_cache_window_size <= 0:
            raise ValueError(f"kv_cache_window_size must be positive, got {self.kv_cache_window_size}")

        if self.kv_low_rank_dimension <= 0:
            raise ValueError(f"kv_low_rank_dimension must be positive, got {self.kv_low_rank_dimension}")

        if not 0.0 <= self.sparsity_ratio <= 1.0:
            raise ValueError(f"sparsity_ratio must be between 0.0 and 1.0, got {self.sparsity_ratio}")

        if self.memory_pool_growth_factor <= 1.0:
            raise ValueError(f"memory_pool_growth_factor must be > 1.0, got {self.memory_pool_growth_factor}")

        # Validate attention configuration
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


def get_default_config() -> Qwen3VLConfig:
    """Get a default configuration for Qwen3-VL model."""
    return Qwen3VLConfig()


def get_hardware_optimized_config(hardware_compute_capability: tuple = (6, 1),
                                 memory_size_gb: float = 8.0) -> Qwen3VLConfig:
    """Get a hardware-optimized configuration based on compute capability and memory size."""
    config = Qwen3VLConfig()
    config.hardware_compute_capability = hardware_compute_capability
    config.memory_size_gb = memory_size_gb
    
    # Adjust memory settings based on available memory
    if memory_size_gb < 8.0:
        config.memory_pool_initial_size = int(1024 * 1024 * 128 * (memory_size_gb / 8.0))  # Scale down initial pool
        config.memory_pool_max_size = int(1024 * 1024 * 512 * (memory_size_gb / 8.0))  # Scale down max pool
        config.pre_allocated_cache_size = int(1024 * 1024 * 64 * (memory_size_gb / 8.0))  # Scale down cache
    else:
        config.memory_pool_initial_size = 1024 * 1024 * 256
        config.memory_pool_max_size = 1024 * 1024 * 1024
        config.pre_allocated_cache_size = 1024 * 1024 * 128
    
    # For lower memory systems, enable more aggressive optimizations
    if memory_size_gb < 6.0:
        config.use_gradient_checkpointing = True
        config.use_activation_sparsity = True
        config.sparsity_ratio = 0.3  # Higher sparsity for lower memory
        config.kv_cache_strategy = "low_rank"  # More memory efficient
        
    return config
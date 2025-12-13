"""
Configuration component for Qwen3-VL
"""
from dataclasses import dataclass
from typing import Optional
from ..adapters.adapter_layers import AdapterConfig


@dataclass
class Qwen3VLConfig:
    """
    Configuration class for Qwen3-VL model.

    This configuration maintains full capacity with 32 transformer layers
    and 32 attention heads while enabling efficiency optimizations.
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
    attention_dropout_prob: float = 0.0
    max_position_embeddings: int = 32768  # Standard for Qwen models
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    pad_token_id: int = 0
    tie_word_embeddings: bool = False
    rope_theta: float = 1000000.0
    use_cache: bool = True

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
    use_gradient_checkpointing: bool = True
    attention_implementation: str = "flash"  # Options: "eager", "flash_attention_2", "sdpa"
    pretraining_tp: int = 1  # Parameter for pretraining tensor parallelism

    # Parameter-efficient adaptation parameters
    use_adapters: bool = False  # Enable parameter-efficient adaptation
    adapter_config: Optional[AdapterConfig] = None  # Configuration for adapters

    # Activation sparsity and early exit parameters
    use_sparsity: bool = False  # Enable activation sparsity and early exit mechanisms
    sparsity_ratio: float = 0.5  # Ratio of activations to keep (1 - sparsity)
    exit_threshold: float = 0.8  # Confidence threshold for early exit

    # Memory-efficient transformer variants parameters (Phase 2.75)
    use_moe: bool = False  # Enable Mixture of Experts
    moe_num_experts: int = 4  # Number of experts in MoE (2-4 as specified)
    moe_top_k: int = 2  # Top-k routing for MoE (top-2 as specified)
    use_flash_attention_2: bool = False  # Enable FlashAttention-2
    use_parameter_sharing: bool = False  # Enable parameter sharing between alternate layers

    # KV Cache optimization parameters (Phase 2.85)
    kv_cache_strategy: str = "hybrid"  # Options: "low_rank", "sliding_window", "hybrid"
    use_low_rank_kv_cache: bool = True  # Enable low-rank KV cache compression
    kv_cache_window_size: int = 1024  # Window size for sliding window attention
    kv_low_rank_dimension: int = 64  # Rank for low-rank approximation

    # Dynamic sparse attention parameters (Phase 7)
    use_dynamic_sparse_attention: bool = False  # Enable dynamic sparse attention with learned routing
    sparse_attention_sparsity_ratio: float = 0.5  # Ratio of tokens to attend to (top-k selection)
    vision_sparse_attention_sparsity_ratio: float = 0.4  # Sparsity ratio for vision attention

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

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure full capacity is preserved
        if self.num_hidden_layers != 32:
            raise ValueError(f"num_hidden_layers must be 32 to preserve full capacity, got {self.num_hidden_layers}")

        if self.num_attention_heads != 32:
            raise ValueError(f"num_attention_heads must be 32 to preserve full capacity, got {self.num_attention_heads}")

        # Initialize adapter config if using adapters but no config provided
        if self.use_adapters and self.adapter_config is None:
            self.adapter_config = AdapterConfig()


__all__ = ["Qwen3VLConfig"]
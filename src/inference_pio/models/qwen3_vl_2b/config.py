"""
Qwen3-VL-2B Configuration - Self-Contained Version

This module provides the configuration for the Qwen3-VL-2B model in the
self-contained plugin architecture for the Inference-PIO system.
"""

import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class Qwen3VL2BConfig:
    """
    Configuration class for the Qwen3-VL-2B model with all optimization parameters.

    This configuration class defines all the parameters needed for the Qwen3-VL-2B model,
    including memory management, attention mechanisms, and hardware-specific optimizations.
    """
    # Model identification
    model_path: str = "H:/Qwen3-VL-2B-Instruct"  # Local model path on drive H
    
    # Device settings for dynamic hybrid execution
    device: Optional[str] = None  # Will be set dynamically during initialization
    device_map: str = "auto"
    
    # Model architecture settings
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: int = 5504
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0  # Qwen3 specific value
    
    # Vision-specific parameters for Qwen3-VL-2B
    vision_hidden_size: int = 1024
    vision_num_attention_heads: int = 16
    vision_num_hidden_layers: int = 24
    vision_intermediate_size: int = 2816
    vision_patch_size: int = 14
    vision_image_size: int = 448
    vision_layer_norm_eps: float = 1e-6
    use_vision_flash_attention: bool = True
    
    # Data type for model computations
    torch_dtype: str = "float16"  # Use string to avoid torch dependency at config level
    
    # Memory optimization settings
    gradient_checkpointing: bool = True
    use_cache: bool = True
    low_cpu_mem_usage: bool = True
    max_memory: Optional[Dict] = None  # Will be set dynamically based on available GPU memory
    
    # Hardware optimization settings
    use_tensor_parallelism: bool = False
    tensor_parallel_size: int = 1
    tensor_parallel_local_rank: int = 0
    tensor_parallel_world_size: int = 1
    tensor_parallel_init_method: str = "tcp://localhost:29500"
    
    # Attention mechanism settings
    use_flash_attention_2: bool = True
    use_sparse_attention: bool = False
    use_sliding_window_attention: bool = False
    use_multi_query_attention: bool = False
    use_grouped_query_attention: bool = True  # Qwen3-VL-2B uses GQA
    use_paged_attention: bool = False
    paged_attention_page_size: int = 256
    sliding_window_size: int = 4096
    
    # Sparse attention settings
    sparse_attention_pattern: str = "longformer"
    sparse_attention_sparsity_ratio: float = 0.25
    sparse_attention_block_size: int = 64
    sparse_attention_local_window_size: int = 128
    use_global_attention: bool = True
    global_attention_indices: List[int] = None  # Will be set to [0] in __post_init__
    
    # Multi-query attention settings
    num_key_value_heads: int = 2  # For GQA - Qwen3-VL-2B specific
    grouped_query_attention_num_kv_heads: int = 2  # For GQA - Qwen3-VL-2B specific
    
    # Additional attention parameters
    attention_dropout_prob: float = 0.0
    remove_bias_in_attention: bool = False
    is_causal: bool = True
    
    # Multimodal attention settings
    use_multimodal_attention: bool = True  # Enable multimodal attention for vision-language models
    multimodal_attention_sparsity_ratio: float = 0.3  # Sparsity ratio for multimodal attention
    multimodal_attention_local_window_size: int = 128  # Local window size for multimodal attention
    multimodal_attention_global_indices: List[int] = None  # Global attention indices for multimodal attention
    multimodal_dropout: float = 0.1  # Dropout for multimodal attention
    alignment_method: str = "qwen3_vl_specific"  # Alignment method for cross-modal processing
    modalities: List[str] = None  # Will be set to ["text", "image"] in __post_init__
    
    # Cross-modal fusion settings
    use_cross_modal_fusion: bool = True  # Enable cross-modal fusion optimization
    cross_modal_fusion_method: str = "qwen3_vl_specific"  # Fusion method for cross-modal processing
    cross_modal_fusion_temperature: float = 0.5  # Temperature for fusion computation
    cross_modal_fusion_lambda: float = 0.1  # Weight for fusion loss
    use_cross_modal_contrastive_fusion: bool = True  # Whether to use contrastive fusion loss
    cross_modal_contrastive_margin: float = 0.2  # Margin for contrastive fusion loss
    enable_dynamic_cross_modal_fusion: bool = True  # Whether to enable dynamic fusion based on input complexity
    cross_modal_fusion_frequency: int = 10  # Frequency of fusion updates (every N steps)
    cross_modal_fusion_threshold: float = 0.8  # Threshold for fusion quality (above which fusion is considered good enough)
    use_cross_modal_attention_fusion: bool = True  # Whether to use attention-based fusion
    use_cross_modal_learned_fusion: bool = True  # Whether to use learned fusion projections
    cross_modal_fusion_projection_dim: int = 512  # Dimension for fusion projections
    enable_cross_modal_similarity_fusion: bool = True  # Whether to enable similarity-based fusion
    cross_modal_similarity_method: str = 'cosine'  # Method for similarity computation ('cosine', 'dot_product', 'euclidean')
    
    # Cross-modal alignment settings
    use_cross_modal_alignment: bool = True  # Enable cross-modal alignment optimization
    cross_modal_alignment_temperature: float = 0.5  # Temperature for alignment computation (controls sharpness of alignment distribution)
    alignment_temperature: float = 0.5 # Alias for compatibility
    cross_modal_alignment_lambda: float = 0.1  # Weight for alignment loss in total loss
    alignment_lambda: float = 0.1 # Alias for compatibility
    use_cross_modal_contrastive_alignment: bool = True  # Whether to use contrastive alignment loss
    use_contrastive_alignment: bool = True # Alias for compatibility
    cross_modal_contrastive_margin: float = 0.2  # Margin for contrastive loss
    enable_dynamic_cross_modal_alignment: bool = True  # Whether to enable dynamic alignment based on input complexity
    cross_modal_alignment_frequency: int = 10  # Frequency of alignment updates (every N steps)
    cross_modal_alignment_threshold: float = 0.8  # Threshold for alignment quality (above which alignment is considered good enough)
    use_cross_modal_attention_alignment: bool = True  # Whether to use attention-based alignment
    use_cross_modal_learned_alignment: bool = True  # Whether to use learned alignment projections
    cross_modal_alignment_projection_dim: int = 512  # Dimension for alignment projections
    enable_cross_modal_similarity_alignment: bool = True  # Whether to enable similarity-based alignment
    cross_modal_similarity_method: str = 'cosine'  # Method for similarity computation ('cosine', 'dot_product', 'euclidean')
    cross_modal_alignment_method: str = 'qwen3_vl_specific'  # Default alignment method ('contrastive', 'attention', 'learned_projection', 'similarity_based', 'qwen3_vl_specific')

    # Qwen3-VL specific optimization settings
    use_qwen3_vl_attention_optimizations: bool = True  # Enable Qwen3-VL specific attention optimizations
    use_qwen3_vl_kv_cache_optimizations: bool = True  # Enable Qwen3-VL specific KV-cache optimizations
    use_qwen3_vl_vision_optimizations: bool = True  # Enable Qwen3-VL specific vision optimizations
    use_qwen3_vl_cross_modal_optimizations: bool = True  # Enable Qwen3-VL specific cross-modal optimizations
    qwen3_vl_attention_sparsity_ratio: float = 0.3  # Sparsity ratio for Qwen3-VL attention optimizations
    qwen3_vl_kv_cache_compression_ratio: float = 0.6  # Compression ratio for Qwen3-VL KV-cache optimizations
    qwen3_vl_cross_modal_attention_scaling: float = 1.2  # Scaling factor for cross-modal attention
    qwen3_vl_extended_context_optimization: bool = True  # Enable optimizations for extended context lengths
    qwen3_vl_speculative_decoding_enabled: bool = False  # Enable speculative decoding for faster inference
    qwen3_vl_speculative_draft_model_ratio: float = 0.5  # Ratio of draft model size to main model
    qwen3_vl_speculative_max_tokens: int = 5  # Max speculative tokens to generate
    qwen3_vl_vision_prompt_enhancement: bool = True  # Enable prompt enhancement for vision tasks
    qwen3_vl_vision_quality_optimization: bool = True  # Enable optimizations for vision quality
    qwen3_vl_memory_efficient_inference: bool = True  # Enable memory-efficient inference optimizations
    qwen3_vl_compute_efficient_inference: bool = True  # Enable compute-efficient inference optimizations

    # Kernel Fusion Patterns
    kernel_fusion_patterns: Optional[Dict[str, bool]] = None
    multimodal_preserve_modalities: Optional[List[str]] = None
    multimodal_preserve_components: Optional[List[str]] = None
    vision_language_visual_device_mapping: Optional[List[int]] = None
    vision_language_textual_device_mapping: Optional[List[int]] = None

    # Additional Qwen3-VL specific attributes that may be set by profiles
    use_glm_attention_patterns: bool = False  # Placeholder for compatibility
    use_glm_ffn_optimization: bool = False  # Placeholder for compatibility
    use_glm_memory_efficient_kv: bool = False  # Placeholder for compatibility
    use_glm_layer_norm_fusion: bool = False  # Placeholder for compatibility
    use_glm_residual_connection_optimization: bool = False  # Placeholder for compatibility
    use_glm_quantization: bool = False  # Placeholder for compatibility
    glm_attention_pattern_sparsity: float = 0.0  # Placeholder for compatibility
    glm_attention_window_size: int = 0  # Placeholder for compatibility
    glm_ffn_expansion_ratio: float = 0.0  # Placeholder for compatibility
    glm_ffn_group_size: int = 0  # Placeholder for compatibility
    glm_kv_cache_compression_ratio: float = 0.0  # Placeholder for compatibility
    glm_weight_bits: int = 0  # Placeholder for compatibility
    glm_activation_bits: int = 0  # Placeholder for compatibility
    use_qwen3_attention_optimizations: bool = False  # Placeholder for compatibility
    use_qwen3_kv_cache_optimizations: bool = False  # Placeholder for compatibility
    use_qwen3_instruction_optimizations: bool = False  # Placeholder for compatibility
    use_qwen3_rope_optimizations: bool = False  # Placeholder for compatibility
    use_qwen3_gqa_optimizations: bool = False  # Placeholder for compatibility
    qwen3_attention_sparsity_ratio: float = 0.0  # Placeholder for compatibility
    qwen3_kv_cache_compression_ratio: float = 0.0  # Placeholder for compatibility
    qwen3_instruction_attention_scaling: float = 0.0  # Placeholder for compatibility
    use_qwen3_coder_attention_optimizations: bool = False  # Placeholder for compatibility
    use_qwen3_coder_kv_cache_optimizations: bool = False  # Placeholder for compatibility
    use_qwen3_coder_code_optimizations: bool = False  # Placeholder for compatibility
    use_qwen3_coder_syntax_highlighting: bool = False  # Placeholder for compatibility
    qwen3_coder_attention_sparsity_ratio: float = 0.0  # Placeholder for compatibility
    qwen3_coder_kv_cache_compression_ratio: float = 0.0  # Placeholder for compatibility
    qwen3_coder_syntax_attention_scaling: float = 0.0  # Placeholder for compatibility

    def __post_init__(self):
        """Post-initialization to set default values for lists and other fields."""
        if self.global_attention_indices is None:
            self.global_attention_indices = [0]
        if self.modalities is None:
            self.modalities = ["text", "image"]
        if self.multimodal_attention_global_indices is None:
            self.multimodal_attention_global_indices = [0]
        if self.kernel_fusion_patterns is None:
            self.kernel_fusion_patterns = ["linear_relu", "linear_gelu", "matmul_add", "add_layer_norm"]
        if self.multimodal_preserve_modalities is None:
            self.multimodal_preserve_modalities = []
        if self.multimodal_preserve_components is None:
            self.multimodal_preserve_components = []
        if self.vision_language_visual_device_mapping is None:
            self.vision_language_visual_device_mapping = [0]
        if self.vision_language_textual_device_mapping is None:
            self.vision_language_textual_device_mapping = [0]

    def _configure_memory_settings(self):
        """
        Configure memory settings based on available system resources.
        """
        try:
            if torch.cuda.is_available():
                # Get GPU memory info
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                # Reserve some memory for overhead (about 1GB)
                reserved_memory = 1024 * 1024 * 1024  # 1GB in bytes
                available_memory = gpu_memory - reserved_memory

                # Set max_memory to use most of available GPU memory but leave some room
                if available_memory > 0:
                    # Convert to GB for max_memory specification
                    max_memory_gb = available_memory / (1024 ** 3)
                    self.max_memory = {0: f"{max_memory_gb:.1f}GB", "cpu": "20GB"}

        except Exception as e:
            # If we can't determine memory, use defaults
            pass


def create_qwen3_vl_2b_config(**kwargs) -> Qwen3VL2BConfig:
    """
    Factory function to create a Qwen3-VL-2B configuration.

    Args:
        **kwargs: Configuration parameters to override defaults

    Returns:
        Qwen3VL2BConfig instance with specified parameters
    """
    config_dict = Qwen3VL2BConfig.__annotations__.copy()
    config = Qwen3VL2BConfig()

    # Update config with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"'{key}' is not a valid configuration parameter for Qwen3VL2BConfig")

    return config


__all__ = [
    "Qwen3VL2BConfig",
    "create_qwen3_vl_2b_config"
]

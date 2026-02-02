"""
Qwen3-VL-2B Configuration - Self-Contained Version

This module provides the configuration for the Qwen3-VL-2B model in the
self-contained plugin architecture for the Inference-PIO system.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from ...common.model_config_base import (
    BaseConfig,
    ModelConfigError,
    get_default_model_path,
)


@dataclass
class Qwen3VL2BConfig(BaseConfig):
    """
    Configuration class for the Qwen3-VL-2B model with all optimization parameters.

    This configuration class defines all the parameters needed for the Qwen3-VL-2B model,
    including memory management, attention mechanisms, and hardware-specific optimizations.
    """

    # Model identification - override defaults
    model_path: str = ""  # Will be set in __post_init__ if not provided
    model_name: str = "qwen3_vl_2b"

    # Device settings for dynamic hybrid execution - inherit from BaseConfig
    # device: Optional[str] = None  # Will be set dynamically during initialization (inherited)
    # device_map: str = "auto"  # (inherited)

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
    layer_norm_eps: float = 1e-6
    use_vision_flash_attention: bool = True

    # Data type for model computations - inherit from BaseConfig
    # torch_dtype: str = "float16"  # Use string to avoid torch dependency at config level (inherited)

    # Memory optimization settings - some inherited from BaseConfig
    # gradient_checkpointing: bool = True  # (inherited)
    # use_cache: bool = True  # (inherited)
    # low_cpu_mem_usage: bool = True  # (inherited)
    # max_memory: Optional[Dict] = None  # Will be set dynamically based on available GPU memory (inherited)
    enable_disk_offloading: bool = False
    offload_folder: str = "offload"
    enable_intelligent_pagination: bool = False

    # Hardware optimization settings - some inherited from BaseConfig
    # use_tensor_parallelism: bool = False  # (inherited)
    tensor_parallel_size: int = 1
    tensor_parallel_local_rank: int = 0
    tensor_parallel_world_size: int = 1
    tensor_parallel_init_method: str = "tcp://localhost:29500"

    # Attention mechanism settings - some inherited from BaseConfig
    # use_flash_attention_2: bool = True  # (inherited)
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
    use_multimodal_attention: bool = (
        True  # Enable multimodal attention for vision-language models
    )
    multimodal_attention_sparsity_ratio: float = (
        0.3  # Sparsity ratio for multimodal attention
    )
    multimodal_attention_local_window_size: int = (
        128  # Local window size for multimodal attention
    )
    multimodal_attention_global_indices: List[int] = (
        None  # Global attention indices for multimodal attention
    )
    multimodal_dropout: float = 0.1  # Dropout for multimodal attention
    alignment_method: str = (
        "qwen3_vl_specific"  # Alignment method for cross-modal processing
    )
    modalities: List[str] = None  # Will be set to ["text", "image"] in __post_init__

    # Cross-modal fusion settings
    use_cross_modal_fusion: bool = True  # Enable cross-modal fusion optimization
    cross_modal_fusion_method: str = (
        "qwen3_vl_specific"  # Fusion method for cross-modal processing
    )
    cross_modal_fusion_temperature: float = 0.5  # Temperature for fusion computation
    cross_modal_fusion_lambda: float = 0.1  # Weight for fusion loss
    use_cross_modal_contrastive_fusion: bool = (
        True  # Whether to use contrastive fusion loss
    )
    cross_modal_contrastive_margin: float = 0.2  # Margin for contrastive fusion loss
    enable_dynamic_cross_modal_fusion: bool = (
        True  # Whether to enable dynamic fusion based on input complexity
    )
    cross_modal_fusion_frequency: int = (
        10  # Frequency of fusion updates (every N steps)
    )
    cross_modal_fusion_threshold: float = (
        0.8  # Threshold for fusion quality (above which fusion is considered good enough)
    )
    use_cross_modal_attention_fusion: bool = (
        True  # Whether to use attention-based fusion
    )
    use_cross_modal_learned_fusion: bool = (
        True  # Whether to use learned fusion projections
    )
    cross_modal_fusion_projection_dim: int = 512  # Dimension for fusion projections
    enable_cross_modal_similarity_fusion: bool = (
        True  # Whether to enable similarity-based fusion
    )
    cross_modal_similarity_method: str = (
        "cosine"  # Method for similarity computation ('cosine', 'dot_product', 'euclidean')
    )

    # Cross-modal alignment settings
    use_cross_modal_alignment: bool = True  # Enable cross-modal alignment optimization
    cross_modal_alignment_temperature: float = (
        0.5  # Temperature for alignment computation (controls sharpness of alignment distribution)
    )
    alignment_temperature: float = 0.5  # Alias for compatibility
    cross_modal_alignment_lambda: float = 0.1  # Weight for alignment loss in total loss
    alignment_lambda: float = 0.1  # Alias for compatibility
    use_cross_modal_contrastive_alignment: bool = (
        True  # Whether to use contrastive alignment loss
    )
    use_contrastive_alignment: bool = True  # Alias for compatibility
    cross_modal_contrastive_margin: float = 0.2  # Margin for contrastive loss
    enable_dynamic_cross_modal_alignment: bool = (
        True  # Whether to enable dynamic alignment based on input complexity
    )
    cross_modal_alignment_frequency: int = (
        10  # Frequency of alignment updates (every N steps)
    )
    cross_modal_alignment_threshold: float = (
        0.8  # Threshold for alignment quality (above which alignment is considered good enough)
    )
    use_cross_modal_attention_alignment: bool = (
        True  # Whether to use attention-based alignment
    )
    use_cross_modal_learned_alignment: bool = (
        True  # Whether to use learned alignment projections
    )
    cross_modal_alignment_projection_dim: int = (
        512  # Dimension for alignment projections
    )
    enable_cross_modal_similarity_alignment: bool = (
        True  # Whether to enable similarity-based alignment
    )
    cross_modal_similarity_method: str = (
        "cosine"  # Method for similarity computation ('cosine', 'dot_product', 'euclidean')
    )
    cross_modal_alignment_method: str = (
        "qwen3_vl_specific"  # Default alignment method ('contrastive', 'attention', 'learned_projection', 'similarity_based', 'qwen3_vl_specific')
    )

    # Legacy/Alias attributes for compatibility
    contrastive_margin: float = 0.2
    enable_dynamic_alignment: bool = True
    alignment_frequency: int = 10
    alignment_threshold: float = 0.8
    use_attention_alignment: bool = True
    use_learned_alignment: bool = True
    alignment_projection_dim: int = 512
    enable_similarity_alignment: bool = True
    similarity_method: str = "cosine"

    # Qwen3-VL specific optimization settings
    use_qwen3_vl_attention_optimizations: bool = (
        True  # Enable Qwen3-VL specific attention optimizations
    )
    use_qwen3_vl_kv_cache_optimizations: bool = (
        True  # Enable Qwen3-VL specific KV-cache optimizations
    )
    use_qwen3_vl_vision_optimizations: bool = (
        True  # Enable Qwen3-VL specific vision optimizations
    )
    use_qwen3_vl_cross_modal_optimizations: bool = (
        True  # Enable Qwen3-VL specific cross-modal optimizations
    )
    qwen3_vl_attention_sparsity_ratio: float = (
        0.3  # Sparsity ratio for Qwen3-VL attention optimizations
    )
    qwen3_vl_kv_cache_compression_ratio: float = (
        0.6  # Compression ratio for Qwen3-VL KV-cache optimizations
    )
    qwen3_vl_cross_modal_attention_scaling: float = (
        1.2  # Scaling factor for cross-modal attention
    )
    qwen3_vl_extended_context_optimization: bool = (
        True  # Enable optimizations for extended context lengths
    )
    qwen3_vl_speculative_decoding_enabled: bool = (
        False  # Enable speculative decoding for faster inference
    )
    qwen3_vl_speculative_draft_model_ratio: float = (
        0.5  # Ratio of draft model size to main model
    )
    qwen3_vl_speculative_max_tokens: int = 5  # Max speculative tokens to generate
    qwen3_vl_vision_prompt_enhancement: bool = (
        True  # Enable prompt enhancement for vision tasks
    )
    qwen3_vl_vision_quality_optimization: bool = (
        True  # Enable optimizations for vision quality
    )
    qwen3_vl_memory_efficient_inference: bool = (
        True  # Enable memory-efficient inference optimizations
    )
    qwen3_vl_compute_efficient_inference: bool = (
        True  # Enable compute-efficient inference optimizations
    )

    # Kernel Fusion Patterns
    kernel_fusion_patterns: Optional[Dict[str, bool]] = None
    multimodal_preserve_modalities: Optional[List[str]] = None
    multimodal_preserve_components: Optional[List[str]] = None

    # System-wide optimization flags (added for compatibility)
    linear_bias_optimization_enabled: bool = False
    enable_continuous_nas: bool = False
    enable_sequence_parallelism: bool = False
    enable_vision_language_parallelism: bool = False
    enable_async_multimodal_processing: bool = False
    async_max_concurrent_requests: int = 4
    async_buffer_size: int = 100
    async_batch_timeout: float = 0.1
    enable_async_batching: bool = True
    async_processing_device: Optional[str] = None
    enable_image_tokenization: bool = True
    enable_image_patch_caching: bool = True
    enable_image_batch_processing: bool = True
    enable_memory_efficient_image_processing: bool = True
    enable_image_quantization: bool = False
    image_quantization_bits: int = 8
    enable_image_compression: bool = True
    image_compression_ratio: float = 0.5
    image_token_dim: int = 1024
    max_image_tokens: int = 1024
    image_size: int = 448
    patch_size: int = 14
    use_quantization: bool = False
    enable_intelligent_multimodal_caching: bool = False
    intelligent_multimodal_cache_size_gb: float = 2.0
    intelligent_multimodal_cache_eviction_policy: str = "predictive"
    intelligent_multimodal_cache_enable_similarity: bool = True
    intelligent_multimodal_cache_similarity_threshold: float = 0.85
    intelligent_multimodal_cache_enable_ttl: bool = True
    intelligent_multimodal_cache_default_ttl: float = 7200.0
    intelligent_multimodal_cache_enable_compression: bool = True
    intelligent_multimodal_cache_compression_ratio: float = 0.6
    enable_multimodal_preprocessing_pipeline: bool = False
    multimodal_pipeline_cache_size: int = 1000
    enable_multimodal_pipeline_caching: bool = True
    max_text_length: int = 32768
    enable_visual_resource_compression: bool = False
    visual_compression_method: str = "quantization"
    visual_compression_ratio: float = 0.5
    visual_quantization_bits: int = 8
    visual_enable_compression_cache: bool = True
    visual_compression_cache_size: int = 1000
    visual_enable_adaptive_compression: bool = True
    use_snn_conversion: bool = False
    use_cuda_kernels: bool = False
    use_prefix_caching: bool = False
    use_kv_cache_compression: bool = False
    use_bias_removal_optimization: bool = False
    use_fused_layer_norm: bool = False
    use_structured_pruning: bool = False
    use_tensor_decomposition: bool = False
    use_adaptive_batching: bool = False
    vision_language_visual_device_mapping: Optional[List[int]] = None
    vision_language_textual_device_mapping: Optional[List[int]] = None

    # Additional Qwen3-VL specific attributes that may be set by profiles
    use_glm_attention_patterns: bool = False  # Placeholder for compatibility
    use_glm_ffn_optimization: bool = False  # Placeholder for compatibility
    use_glm_memory_efficient_kv: bool = False  # Placeholder for compatibility
    use_glm_layer_norm_fusion: bool = False  # Placeholder for compatibility
    use_glm_residual_connection_optimization: bool = (
        False  # Placeholder for compatibility
    )
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
    use_qwen3_coder_attention_optimizations: bool = (
        False  # Placeholder for compatibility
    )
    use_qwen3_coder_kv_cache_optimizations: bool = (
        False  # Placeholder for compatibility
    )
    use_qwen3_coder_code_optimizations: bool = False  # Placeholder for compatibility
    use_qwen3_coder_syntax_highlighting: bool = False  # Placeholder for compatibility
    qwen3_coder_attention_sparsity_ratio: float = 0.0  # Placeholder for compatibility
    qwen3_coder_kv_cache_compression_ratio: float = 0.0  # Placeholder for compatibility
    qwen3_coder_syntax_attention_scaling: float = 0.0  # Placeholder for compatibility

    # Multimodal Attention Optimization Parameters
    use_multimodal_attention_optimization: bool = True
    multimodal_attention_temperature: float = 1.0
    multimodal_attention_lambda: float = 0.1
    multimodal_attention_window_size: int = 512
    multimodal_attention_use_flash: bool = True
    multimodal_attention_use_sparse: bool = False
    multimodal_attention_use_sliding_window: bool = False
    multimodal_attention_use_mqa_gqa: bool = True
    multimodal_attention_use_paged: bool = False
    multimodal_attention_cross_modal_fusion_method: str = "gated"
    multimodal_attention_cross_modal_alignment_method: str = "contrastive"
    multimodal_attention_enable_dynamic_fusion: bool = True
    multimodal_attention_enable_adaptive_compression: bool = True
    multimodal_attention_compression_ratio: float = 0.8
    multimodal_attention_enable_tensor_fusion: bool = True
    multimodal_attention_tensor_fusion_method: str = "bilinear"
    multimodal_attention_enable_quantization: bool = False
    multimodal_attention_quantization_bits: int = 8
    multimodal_attention_enable_lora: bool = False
    multimodal_attention_lora_rank: int = 16
    multimodal_attention_lora_alpha: float = 32.0

    def __post_init__(self):
        """Post-initialization to set default values for lists and other fields."""
        # Set default model path if not provided
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)

        # Ensure the model path points to the H drive for Qwen3-VL-2B model
        if (
            not self.model_path
            or "qwen3_vl_2b" in self.model_path.lower()
            or "qwen3-vl" in self.model_path.lower()
        ):
            self.model_path = "H:/Qwen3-VL-2B-Instruct"

        # Call parent's post_init to validate config
        super().__post_init__()

        if self.global_attention_indices is None:
            self.global_attention_indices = [0]
        if self.modalities is None:
            self.modalities = ["text", "image"]
        if self.multimodal_attention_global_indices is None:
            self.multimodal_attention_global_indices = [0]
        if self.kernel_fusion_patterns is None:
            self.kernel_fusion_patterns = [
                "linear_relu",
                "linear_gelu",
                "matmul_add",
                "add_layer_norm",
            ]
        if self.multimodal_preserve_modalities is None:
            self.multimodal_preserve_modalities = []
        if self.multimodal_preserve_components is None:
            self.multimodal_preserve_components = []
        if self.vision_language_visual_device_mapping is None:
            self.vision_language_visual_device_mapping = [0]
        if self.vision_language_textual_device_mapping is None:
            self.vision_language_textual_device_mapping = [0]

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return model-specific parameters."""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "vision_hidden_size": self.vision_hidden_size,
            "vision_num_attention_heads": self.vision_num_attention_heads,
            "vision_num_hidden_layers": self.vision_num_hidden_layers,
            "vision_intermediate_size": self.vision_intermediate_size,
            "vision_patch_size": self.vision_patch_size,
            "vision_image_size": self.vision_image_size,
            "vision_layer_norm_eps": self.vision_layer_norm_eps,
            "num_key_value_heads": self.num_key_value_heads,
            "modalities": self.modalities,
        }

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
                    max_memory_gb = available_memory / (1024**3)
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
    config = Qwen3VL2BConfig(**kwargs)
    return config


# Configurável para integração com o sistema de configuração dinâmica
class Qwen3VLDynamicConfig(Qwen3VL2BConfig):
    """
    Extends the base Qwen3-VL-2B configuration with dynamic configuration capabilities.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Adiciona capacidades de configuração dinâmica se necessário
        pass


# Register this configuration with the factory
from ...common.config_factory import register_model_config

register_model_config("qwen3_vl_2b", Qwen3VL2BConfig)


__all__ = ["Qwen3VL2BConfig", "Qwen3VLDynamicConfig", "create_qwen3_vl_2b_config"]

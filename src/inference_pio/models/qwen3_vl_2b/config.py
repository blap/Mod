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
    cross_modal_alignment_lambda: float = 0.1  # Weight for alignment loss in total loss
    use_cross_modal_contrastive_alignment: bool = True  # Whether to use contrastive alignment loss
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
    
    # Rotary embedding settings
    rope_theta: float = 1000000.0  # Qwen3-VL-2B specific theta value
    
    # Optimization settings
    use_fused_layer_norm: bool = True
    use_bias_removal_optimization: bool = True
    linear_bias_optimization_enabled: bool = True
    
    # KV-cache optimization settings
    use_kv_cache_compression: bool = True
    kv_cache_compression_method: str = "quantization"  # Options: "quantization", "low_rank", "sparse"
    kv_cache_quantization_bits: int = 8
    kv_cache_low_rank_dimension: int = 64
    kv_cache_adaptive_precision_threshold: float = 0.1
    kv_cache_sparse_compression_ratio: float = 0.5
    kv_cache_enable_dynamic_compression: bool = True
    
    # Prefix caching settings
    use_prefix_caching: bool = True
    prefix_cache_max_size: int = 1000  # Max number of cached prefixes
    prefix_cache_precision: str = "float16"  # Options: "float16", "float32", "bfloat16"
    prefix_cache_compression_enabled: bool = True
    prefix_cache_eviction_policy: str = "LRU"  # Options: "LRU", "LFU", "FIFO"
    prefix_cache_enable_prefetching: bool = True
    prefix_cache_prefetch_distance: int = 3
    prefix_cache_max_prefix_length: int = 1024
    prefix_cache_min_prefix_length: int = 8
    prefix_cache_warmup_threshold: int = 2  # Min accesses before caching
    
    # CUDA kernels settings
    use_cuda_kernels: bool = True
    
    # Kernel fusion settings
    enable_kernel_fusion: bool = True
    kernel_fusion_patterns: List[str] = None  # Will be set in __post_init__
    use_custom_cuda_kernels: bool = True
    custom_kernel_fallback_enabled: bool = True
    kernel_fusion_verbose: bool = False
    
    # Runtime memory optimization settings
    torch_compile_mode: str = "reduce-overhead"  # Options: "reduce-overhead", "max-autotune", "default"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = True
    enable_cudnn_benchmark: bool = True
    enable_memory_efficient_attention: bool = True
    
    # Memory management and paging settings
    enable_memory_management: bool = True  # Enable advanced memory management
    max_memory_ratio: float = 0.8  # Maximum ratio of system memory to use
    swap_directory: Optional[str] = None  # Directory for swap files (None for temp)
    page_size_mb: int = 16  # Size of memory pages in MB for tensor paging
    eviction_policy: str = "predictive"  # Page eviction policy: "lru", "fifo", "priority", "predictive"
    enable_tensor_paging: bool = True  # Enable tensor paging between RAM and disk
    enable_smart_swap: bool = True  # Enable smart swap configuration
    tensor_paging_priority: str = "medium"  # Priority for paged tensors: "low", "medium", "high", "critical"
    pin_optimizer_states: bool = False  # Whether to pin optimizer states in memory
    pin_embeddings: bool = True  # Whether to pin embedding layers in memory
    pin_attention_weights: bool = False  # Whether to pin attention weights in memory
    memory_cleanup_interval: int = 300  # Interval in seconds for memory cleanup (0 to disable)
    
    # Predictive memory management settings
    enable_predictive_management: bool = True  # Enable predictive memory management using ML algorithms
    prediction_horizon_seconds: int = 30  # Time horizon for memory usage predictions (seconds)
    proactive_management_interval: float = 5.0  # Interval for proactive memory management (seconds)
    memory_prediction_threshold: float = 0.9  # Threshold for triggering proactive management (percentage of max_memory_ratio)
    
    # Disk offloading settings
    enable_disk_offloading: bool = True  # Enable disk offloading for model components
    offload_directory: Optional[str] = None  # Directory for offload files (None for temp)
    offloading_priority: str = "medium"  # Priority for offloaded tensors: "low", "medium", "high", "critical"
    offload_attention_weights: bool = False  # Whether to offload attention weights to disk
    enable_predictive_offloading: bool = True  # Enable predictive offloading based on access patterns
    proactive_offloading_interval: float = 5.0  # Interval for proactive offloading (seconds)
    
    # Adaptive batching settings
    enable_adaptive_batching: bool = True  # Enable adaptive batching for dynamic batch size adjustment
    initial_batch_size: int = 1  # Initial batch size for adaptive batching
    min_batch_size: int = 1  # Minimum batch size allowed
    max_batch_size: int = 16  # Maximum batch size allowed
    memory_threshold_ratio: float = 0.85  # Memory usage ratio that triggers batch size adjustment
    performance_window_size: int = 10  # Number of recent samples to consider for performance evaluation
    batch_adjustment_factor: float = 0.1  # Factor controlling how aggressively to adjust batch size
    batch_cooldown_period: float = 5.0  # Time in seconds to wait between batch size adjustments
    performance_target: float = 0.8  # Target performance score (0.0 to 1.0)
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    pad_token_id: int = 151643  # Default for Qwen models
    
    # Tensor compression settings
    enable_tensor_compression: bool = True  # Enable tensor compression for model weights
    tensor_compression_method: str = "incremental_pca"  # Compression method: "incremental_pca", "svd", "auto"
    tensor_compression_ratio: float = 0.5  # Target compression ratio (0.0 to 1.0, where 0.5 = 50% reduction)
    tensor_compression_max_components: int = 256  # Maximum number of components to keep during compression
    compression_memory_threshold_high: float = 0.8  # Memory threshold for high compression (0.0 to 1.0)
    compression_memory_threshold_critical: float = 0.9  # Memory threshold for critical compression (0.0 to 1.0)
    enable_adaptive_compression: bool = True  # Enable adaptive compression that adjusts based on memory usage
    enable_activation_compression: bool = True  # Enable compression of model activations during inference
    compression_update_frequency: int = 100  # How often to update compression models (in terms of batches)
    
    # Activation offloading settings
    enable_activation_offloading: bool = True  # Enable activation offloading for intermediate activations
    activation_max_memory_ratio: float = 0.7  # Maximum ratio of system memory to use for activations
    activation_offload_directory: Optional[str] = None  # Directory for activation offload files (None for temp)
    activation_page_size_mb: int = 8  # Size of activation pages in MB
    activation_eviction_policy: str = "predictive"  # Activation eviction policy: "lru", "fifo", "priority", "predictive"
    activation_offloading_priority: str = "medium"  # Priority for offloaded activations: "low", "medium", "high", "critical"
    enable_predictive_activation_offloading: bool = True  # Enable predictive activation offloading based on access patterns
    proactive_activation_offloading_interval: float = 5.0  # Interval for proactive activation offloading (seconds)
    
    # Tensor decomposition settings
    use_tensor_decomposition: bool = False  # Enable tensor decomposition for model compression
    tensor_decomposition_method: str = "cp_decomposition"  # Decomposition method: "cp_decomposition", "tucker_decomposition", "tensor_train", "matrix_svd"
    tensor_decomposition_rank_ratio: float = 0.5  # Target rank ratio (0.0 to 1.0, where 0.5 = 50% of original rank)
    
    # Structured pruning settings
    use_structured_pruning: bool = False  # Enable structured pruning for model compression
    pruning_ratio: float = 0.2  # Ratio of blocks/layers to remove (0.0 to 1.0)
    pruning_method: str = "layer_removal"  # Pruning method: "layer_removal", "block_removal", "head_removal", "mlp_removal", "adaptive_pruning"
    pruning_block_size: int = 1  # Size of blocks to remove (for block pruning)
    
    # Continuous NAS settings
    enable_continuous_nas: bool = False  # Enable continuous NAS for architecture adaptation
    nas_strategy: str = "combined_adaptive"  # NAS strategy: "depth_adaptive", "width_adaptive", "combined_adaptive", "latency_based", "memory_based"
    nas_min_depth_ratio: float = 0.3  # Minimum depth as percentage of original
    nas_max_depth_ratio: float = 1.0  # Maximum depth as percentage of original
    nas_min_width_ratio: float = 0.3  # Minimum width as percentage of original
    nas_max_width_ratio: float = 1.0  # Maximum width as percentage of original
    nas_latency_target_ms: float = 100.0  # Target latency in milliseconds
    nas_memory_budget_mb: float = 2048.0  # Memory budget in MB
    nas_accuracy_tradeoff_factor: float = 0.7  # Trade-off factor between accuracy and speed (0-1)
    nas_adaptation_frequency: int = 10  # How often to adapt (in terms of inference calls)
    
    # Intelligent Pagination settings
    enable_intelligent_pagination: bool = True  # Enable intelligent pagination for multimodal data
    pagination_swap_directory: str = "./tensor_swap"  # Directory for pagination swap files
    pagination_page_size_mb: int = 16  # Size of pagination pages in MB
    pagination_eviction_policy: str = "intelligent"  # Pagination eviction policy: "lru", "fifo", "priority", "intelligent"
    pagination_max_memory_ratio: float = 0.8  # Maximum ratio of system memory to use for pagination
    paginate_attention_weights: bool = True  # Whether to paginate attention weights
    paginate_kv_cache: bool = True  # Whether to paginate KV cache tensors
    paginate_activations: bool = True  # Whether to paginate activation tensors
    enable_proactive_pagination: bool = True  # Enable proactive pagination based on access patterns
    proactive_pagination_interval: float = 5.0  # Interval for proactive pagination (seconds)
    
    # Quantization settings - added for memory optimization
    use_quantization: bool = False
    quantization_scheme: str = 'int8'
    quantization_bits: int = 8
    quantization_symmetric: bool = True
    quantization_per_channel: bool = True
    
    # Sequence parallelism settings - added for sequence parallel execution
    enable_sequence_parallelism: bool = False
    sequence_parallel_num_segments: int = 1
    sequence_parallel_split_method: str = 'chunk'
    sequence_parallel_enable_overlap: bool = True
    sequence_parallel_overlap_size: int = 64
    sequence_parallel_algorithm: str = '1d'
    
    # Vision-Language parallelism settings - added for multimodal parallel execution
    enable_vision_language_parallelism: bool = False
    vision_language_num_visual_stages: int = 1
    vision_language_num_textual_stages: int = 1
    vision_language_visual_device_mapping: Optional[List[int]] = None
    vision_language_textual_device_mapping: Optional[List[int]] = None
    vision_language_enable_cross_modal_communication: bool = True
    vision_language_pipeline_schedule: str = 'interleaved'
    
    # Multimodal Model Surgery settings
    enable_multimodal_model_surgery: bool = False  # Enable multimodal model surgery for component removal
    multimodal_surgery_enabled: bool = True  # Whether to perform surgery when enabled
    multimodal_auto_identify_components: bool = True  # Auto-identify components to remove
    multimodal_surgery_priority_threshold: int = 10  # Priority threshold for surgery (lower = higher priority)
    multimodal_analysis_only: bool = False  # Only analyze model without performing surgery
    multimodal_preserve_modalities: List[str] = None  # Modalities to preserve during surgery (e.g., ['vision', 'text'])
    multimodal_preserve_components: List[str] = None  # Specific components to preserve during surgery
    multimodal_components_to_remove: Optional[List[str]] = None  # Specific components to remove (None means auto-detect)
    
    # Additional attributes for compatibility
    use_snn_conversion: bool = False  # SNN conversion for energy efficiency
    
    # Projection layer optimization settings
    use_projection_layer_optimization: bool = True  # Enable projection layer optimization for vision-language alignment
    projection_layer_use_bias: bool = True  # Whether to use bias in projection layers
    projection_layer_activation: str = "silu"  # Activation function for projection layers ('silu', 'relu', 'swish', 'linear')
    projection_layer_dropout: float = 0.1  # Dropout probability for projection layers
    projection_layer_use_residual: bool = True  # Whether to use residual connections in projection layers
    projection_layer_use_low_rank: bool = False  # Whether to use low-rank projection for memory efficiency
    projection_layer_low_rank_dim: int = 512  # Dimension for low-rank projection
    projection_layer_use_group_norm: bool = False  # Whether to use group normalization instead of layer normalization
    projection_layer_group_norm_num_groups: int = 32  # Number of groups for group normalization
    projection_layer_intermediate_dim: int = 4096  # Intermediate dimension for multimodal fusion
    projection_layer_num_layers: int = 2  # Number of projection layers to stack
    projection_layer_use_cross_attention: bool = True  # Whether to include cross-attention between modalities
    projection_layer_cross_attention_heads: int = 8  # Number of attention heads for cross-attention
    
    # Intelligent Multimodal Caching settings
    enable_intelligent_multimodal_caching: bool = True  # Enable intelligent caching for multimodal data
    intelligent_multimodal_cache_size_gb: float = 2.0  # Cache size in GB for multimodal caching
    intelligent_multimodal_cache_eviction_policy: str = "predictive"  # Options: "lru", "lfu", "fifo", "predictive"
    intelligent_multimodal_cache_enable_similarity: bool = True  # Enable similarity-based caching
    intelligent_multimodal_cache_similarity_threshold: float = 0.85  # Threshold for considering content similar
    intelligent_multimodal_cache_enable_ttl: bool = True  # Enable time-to-live for cache entries
    intelligent_multimodal_cache_default_ttl: float = 7200.0  # Default TTL in seconds (2 hours)
    intelligent_multimodal_cache_enable_compression: bool = True  # Enable compression for cached data
    intelligent_multimodal_cache_compression_ratio: float = 0.6  # Target compression ratio (0.0 to 1.0)
    
    # Asynchronous Multimodal Processing settings
    enable_async_multimodal_processing: bool = True  # Enable asynchronous multimodal processing
    async_max_concurrent_requests: int = 8  # Maximum concurrent requests for async processing
    async_buffer_size: int = 200  # Size of the async request buffer
    async_batch_timeout: float = 0.05  # Timeout for batching async requests (in seconds)
    enable_async_batching: bool = True  # Enable batching in async processing
    async_processing_device: str = "cuda:0" if torch.cuda.is_available() else "cpu"  # Device for async processing
    
    # Dynamic Multimodal Batching settings
    enable_dynamic_multimodal_batching: bool = False  # Enable dynamic batching for multimodal inputs
    text_weight: float = 0.4  # Weight of text complexity in combined complexity calculation
    image_weight: float = 0.6  # Weight of image complexity in combined complexity calculation
    complexity_threshold_low: float = 0.3  # Below this complexity, use max batch size
    complexity_threshold_high: float = 0.7  # Above this complexity, use min batch size
    
    # Multimodal Preprocessing Pipeline settings
    enable_multimodal_preprocessing_pipeline: bool = True  # Enable multimodal preprocessing pipeline
    max_text_length: int = 32768  # Maximum length for text sequences
    image_size: int = 448  # Size of processed images
    patch_size: int = 14  # Size of image patches
    enable_multimodal_pipeline_caching: bool = True  # Enable caching for preprocessing results
    multimodal_pipeline_cache_size: int = 1000  # Size of the preprocessing cache
    
    # Vision Encoder Optimization settings
    enable_vision_patch_embedding_optimization: bool = True  # Enable optimization for patch embedding
    enable_vision_attention_optimization: bool = True  # Enable optimization for vision attention
    enable_vision_mlp_optimization: bool = True  # Enable optimization for vision MLP
    enable_vision_block_optimization: bool = True  # Enable optimization for vision transformer blocks
    use_vision_convolution_fusion: bool = True  # Use convolution fusion for patch embedding
    enable_vision_gradient_checkpointing: bool = True  # Enable gradient checkpointing for vision encoder
    enable_vision_memory_efficient_attention: bool = True  # Enable memory-efficient attention
    enable_vision_tensor_fusion: bool = True  # Enable tensor fusion for memory efficiency
    enable_vision_sparse_attention: bool = False  # Enable sparse attention in vision encoder
    vision_sparse_attention_density: float = 0.5  # Density for sparse attention (0.0 to 1.0)
    enable_vision_encoder_quantization: bool = False  # Enable quantization for vision encoder
    vision_encoder_quantization_bits: int = 8  # Number of bits for vision encoder quantization
    vision_encoder_quantization_method: str = 'linear'  # Quantization method: 'linear', 'log', 'affine'
    enable_vision_encoder_lora: bool = False  # Enable LoRA adaptation for vision encoder
    vision_encoder_lora_rank: int = 16  # Rank for LoRA adaptation
    vision_encoder_lora_alpha: int = 32  # Alpha parameter for LoRA adaptation
    enable_vision_sparse_convolution: bool = False  # Enable sparse convolution in vision encoder
    vision_sparse_convolution_density: float = 0.5  # Density for sparse convolution (0.0 to 1.0)
    
    # Visual Resource Compression settings
    enable_visual_resource_compression: bool = True  # Enable visual resource compression for memory efficiency
    visual_compression_method: str = 'quantization'  # Compression method: 'pca', 'svd', 'quantization', 'sparse_coding', 'autoencoder'
    visual_compression_ratio: float = 0.5  # Target compression ratio (0.0 to 1.0, where 0.5 = 50% reduction)
    visual_quantization_bits: int = 8  # Number of bits for quantization
    visual_quantization_method: str = 'linear'  # Quantization method: 'linear', 'log', 'kmeans'
    enable_visual_compression_cache: bool = True  # Enable caching of compressed representations
    visual_compression_cache_size: int = 1000  # Maximum number of cached compressed representations
    enable_adaptive_visual_compression: bool = True  # Enable adaptive compression based on input characteristics
    visual_pca_components_ratio: float = 0.7  # Ratio of components to keep for PCA (0.0 to 1.0)
    visual_svd_rank_ratio: float = 0.5  # Ratio of rank to keep for SVD (0.0 to 1.0)
    visual_autoencoder_latent_dim_ratio: float = 0.5  # Ratio of latent dimension to original for autoencoder
    visual_sparse_coding_sparsity: float = 0.1  # Sparsity level for sparse coding (0.0 to 1.0)
    visual_sparse_coding_dictionary_size: int = 256  # Size of dictionary for sparse coding
    
    # Image Tokenization settings
    enable_image_tokenization: bool = True  # Enable efficient image tokenization for vision processing
    max_image_tokens: int = 1024  # Maximum number of image tokens
    token_dim: int = 1024  # Dimension of each image token
    enable_image_patch_caching: bool = True  # Enable caching of processed image patches
    enable_image_batch_processing: bool = True  # Enable batch processing for images
    enable_memory_efficient_image_processing: bool = True  # Enable memory-efficient image processing
    enable_image_quantization: bool = False  # Enable quantization for faster image processing
    image_quantization_bits: int = 8  # Number of bits for image quantization
    enable_image_compression: bool = True  # Enable compression for image storage efficiency
    image_compression_ratio: float = 0.5  # Compression ratio for images (0.0 to 1.0)
    
    # Cross-Modal Alignment Optimization settings
    enable_cross_modal_alignment: bool = True  # Enable cross-modal alignment optimization for vision-language models
    cross_modal_alignment_temperature: float = 0.5  # Temperature for alignment computation (controls sharpness of alignment distribution)
    cross_modal_alignment_lambda: float = 0.1  # Weight for alignment loss in total loss
    use_cross_modal_contrastive_alignment: bool = True  # Whether to use contrastive alignment loss
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
"""
Qwen3-4B-Instruct-2507 Configuration - Self-Contained Version

This module defines the configuration class for the Qwen3-4B-Instruct-2507 model in the
self-contained plugin architecture for the Inference-PIO system.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

try:
    from ...common.config.model_config_base import (
        BaseConfig,
        ModelConfigError,
        get_default_model_path,
    )
except ImportError:
    # Fallback import
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from src.inference_pio.common.config.model_config_base import (
        BaseConfig,
        ModelConfigError,
        get_default_model_path,
    )


@dataclass
class Qwen34BInstruct2507Config(BaseConfig):
    """
    Configuration class for the Qwen3-4B-Instruct-2507 model with all parameters consolidated.

    This is the main configuration class for the Qwen3-4B-Instruct-2507 model. It contains all
    parameters needed to define the model architecture and optimization settings.
    This class extends BaseConfig with Qwen3-4B-Instruct-2507 specific parameters.
    """

    # Model identification - override defaults
    model_path: str = ""  # Will be set in __post_init__ if not provided
    model_name: str = "qwen3_4b_instruct_2507"

    # Device settings for dynamic hybrid execution - inherit from BaseConfig
    # device: str = None  # Will be set dynamically during initialization (inherited)
    # device_map: str = "auto"  # (inherited)

    # Model architecture parameters
    hidden_size: int = 2560  # Qwen3-4B specific
    num_attention_heads: int = 32  # Qwen3-4B specific
    num_hidden_layers: int = 32  # Qwen3-4B specific
    max_position_embeddings: int = 32768  # Extended for Qwen3 models
    rope_theta: float = 1000000.0
    intermediate_size: int = 6912  # Qwen3-4B specific
    vocab_size: int = 152064  # Qwen3-4B specific
    layer_norm_eps: float = 1e-06  # Qwen3-4B specific
    attention_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    num_key_value_heads: int = (
        32  # Should match num_attention_heads for standard attention
    )
    initializer_range: float = 0.02

    # Memory optimization settings - some inherited from BaseConfig
    # gradient_checkpointing: bool = True  # (inherited)
    # use_cache: bool = True  # (inherited)
    # torch_dtype: str = "float16"  # Use half precision for memory efficiency (inherited)
    # device_map: str = "auto"  # Auto-distribute across available devices (inherited)
    # low_cpu_mem_usage: bool = True  # (inherited)
    # max_memory: Optional[dict] = None  # Will be set dynamically based on available GPU memory (inherited)

    # Qwen3-4B-Instruct-2507 specific generation parameters
    temperature: float = 0.7  # Balanced temperature for Qwen3-4B
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1024  # Reasonable default for Qwen3-4B
    do_sample: bool = True
    pad_token_id: Optional[int] = None  # Will be set from tokenizer

    # Optimization flags - some inherited from BaseConfig
    # use_flash_attention_2: bool = True  # Enable FlashAttention 2.0 optimization (inherited)
    use_sparse_attention: bool = (
        True  # Enable sparse attention optimization (overrides FlashAttention if True)
    )
    sparse_attention_pattern: str = (
        "longformer"  # Options: 'longformer', 'bigbird', 'block_sparse', 'local', 'random', 'strided'
    )
    sparse_attention_sparsity_ratio: float = (
        0.25  # Ratio of tokens to attend to (for random/local patterns)
    )
    sparse_attention_block_size: int = 64  # Block size for block sparse attention
    sparse_attention_local_window_size: int = 128  # Window size for local attention
    use_global_attention: bool = (
        True  # Whether to use global attention in sparse patterns
    )
    global_attention_indices: List[int] = None  # Indices for global attention tokens
    use_multi_pattern_attention: bool = (
        False  # Whether to use multi-pattern adaptive attention
    )
    use_sparse_attention_with_fallback: bool = (
        True  # Whether to fallback to FlashAttention when sparse is not applicable
    )
    use_multi_query_attention: bool = (
        True  # Enable Multi-Query Attention (MQA) optimization
    )
    use_grouped_query_attention: bool = (
        True  # Enable Grouped-Query Attention (GQA) optimization
    )
    use_paged_attention: bool = (
        True  # Enable paged attention for efficient KV-cache management
    )
    paged_attention_page_size: int = 16  # Size of each page in paged attention
    use_sliding_window_attention: bool = (
        True  # Use sliding window attention (separate from paged attention)
    )
    sliding_window_size: int = 4096  # Size of the sliding window
    attention_type: str = "gqa"  # Attention type: 'mha', 'gqa', or 'mqa'
    num_key_value_groups: int = (
        4  # Number of query heads per KV head for GQA (only used when attention_type='gqa')
    )
    use_fused_layer_norm: bool = (
        True  # Enable fused layer normalization for improved performance
    )
    use_bias_removal_optimization: bool = (
        True  # Enable bias removal optimization for linear layers
    )
    bias_removal_config: dict = None  # Configuration for bias removal optimization
    # use_tensor_parallelism: bool = False  # Enable tensor parallelism for multi-GPU support (disabled by default for compatibility) (inherited)
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    tensor_parallel_local_rank: int = 0  # Local rank for tensor parallelism
    tensor_parallel_world_size: int = 1  # World size for tensor parallelism
    tensor_parallel_init_method: str = (
        "tcp://localhost:29500"  # Initialization method for tensor parallelism
    )

    # KV-cache compression settings
    use_kv_cache_compression: bool = True
    kv_cache_compression_method: str = (
        "combined"  # Options: "quantization", "low_rank", "adaptive_precision", "sparse", "combined"
    )
    kv_cache_quantization_bits: int = 8
    kv_cache_low_rank_dimension: int = 64
    kv_cache_adaptive_precision_threshold: float = 0.01
    kv_cache_sparse_compression_ratio: float = 0.5
    kv_cache_enable_dynamic_compression: bool = True

    # Prefix caching settings
    use_prefix_caching: bool = True
    prefix_cache_max_size: int = 1024 * 1024 * 256  # 256MB
    prefix_cache_precision: str = "float16"
    prefix_cache_compression_enabled: bool = True
    prefix_cache_eviction_policy: str = "lru"  # Options: "lru", "fifo", "lfu"
    prefix_cache_enable_prefetching: bool = True
    prefix_cache_prefetch_distance: int = 1
    prefix_cache_max_prefix_length: int = 2048
    prefix_cache_min_prefix_length: int = 8
    prefix_cache_warmup_threshold: int = 3

    # CUDA kernels settings
    use_cuda_kernels: bool = True
    cuda_kernel_gelu_enabled: bool = True
    cuda_kernel_matmul_enabled: bool = True
    cuda_kernel_softmax_enabled: bool = True
    cuda_kernel_attention_enabled: bool = True
    cuda_kernel_mlp_enabled: bool = True
    cuda_kernel_layernorm_enabled: bool = True

    # Linear layer bias optimization
    linear_bias_optimization_enabled: bool = True
    remove_bias_after_norm: bool = True
    remove_bias_in_attention: bool = True
    remove_bias_in_mlp: bool = True
    remove_bias_in_embeddings: bool = False

    # Runtime memory optimization settings
    torch_compile_mode: str = (
        "reduce-overhead"  # Options: "reduce-overhead", "max-autotune", "default"
    )
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = True
    enable_cudnn_benchmark: bool = True
    enable_memory_efficient_attention: bool = True

    # Kernel fusion settings
    enable_kernel_fusion: bool = True
    kernel_fusion_patterns: List[str] = (
        None  # Options: "linear_relu", "linear_gelu", "matmul_add", "add_layer_norm", etc.
    )
    use_custom_cuda_kernels: bool = True
    custom_kernel_fallback_enabled: bool = True
    kernel_fusion_verbose: bool = False

    # Memory management and paging settings
    enable_memory_management: bool = True  # Enable advanced memory management
    max_memory_ratio: float = 0.8  # Maximum ratio of system memory to use
    swap_directory: Optional[str] = None  # Directory for swap files (None for temp)
    page_size_mb: int = 16  # Size of memory pages in MB for tensor paging
    eviction_policy: str = (
        "predictive"  # Page eviction policy: "lru", "fifo", "priority", "predictive"
    )
    enable_tensor_paging: bool = True  # Enable tensor paging between RAM and disk
    enable_smart_swap: bool = True  # Enable smart swap configuration
    tensor_paging_priority: str = (
        "medium"  # Priority for paged tensors: "low", "medium", "high", "critical"
    )
    pin_optimizer_states: bool = False  # Whether to pin optimizer states in memory
    pin_embeddings: bool = True  # Whether to pin embedding layers in memory
    pin_attention_weights: bool = False  # Whether to pin attention weights in memory
    memory_cleanup_interval: int = (
        300  # Interval in seconds for memory cleanup (0 to disable)
    )

    # Predictive memory management settings
    enable_predictive_management: bool = (
        True  # Enable predictive memory management using ML algorithms
    )
    prediction_horizon_seconds: int = (
        30  # Time horizon for memory usage predictions (seconds)
    )
    proactive_management_interval: float = (
        5.0  # Interval for proactive memory management (seconds)
    )
    memory_prediction_threshold: float = (
        0.9  # Threshold for triggering proactive management (percentage of max_memory_ratio)
    )

    # Disk offloading settings
    enable_disk_offloading: bool = True  # Enable disk offloading for model components
    offload_directory: Optional[str] = (
        None  # Directory for offload files (None for temp)
    )
    offloading_priority: str = (
        "medium"  # Priority for offloaded tensors: "low", "medium", "high", "critical"
    )
    offload_attention_weights: bool = (
        False  # Whether to offload attention weights to disk
    )
    enable_predictive_offloading: bool = (
        True  # Enable predictive offloading based on access patterns
    )
    proactive_offloading_interval: float = (
        5.0  # Interval for proactive offloading (seconds)
    )

    # Activation offloading settings
    enable_activation_offloading: bool = (
        True  # Enable activation offloading for intermediate activations
    )
    activation_max_memory_ratio: float = (
        0.7  # Maximum ratio of system memory to use for activations
    )
    activation_offload_directory: Optional[str] = (
        None  # Directory for activation offload files (None for temp)
    )
    activation_page_size_mb: int = 8  # Size of activation pages in MB
    activation_eviction_policy: str = (
        "predictive"  # Activation eviction policy: "lru", "fifo", "priority", "predictive"
    )
    activation_offloading_priority: str = (
        "medium"  # Priority for offloaded activations: "low", "medium", "high", "critical"
    )
    enable_predictive_activation_offloading: bool = (
        True  # Enable predictive activation offloading based on access patterns
    )
    proactive_activation_offloading_interval: float = (
        5.0  # Interval for proactive activation offloading (seconds)
    )

    # Adaptive batching settings
    enable_adaptive_batching: bool = (
        True  # Enable adaptive batching for dynamic batch size adjustment
    )
    initial_batch_size: int = 1  # Initial batch size for adaptive batching
    min_batch_size: int = 1  # Minimum batch size allowed
    max_batch_size: int = 16  # Maximum batch size allowed
    memory_threshold_ratio: float = (
        0.85  # Memory usage ratio that triggers batch size adjustment
    )
    performance_window_size: int = (
        10  # Number of recent samples to consider for performance evaluation
    )
    batch_adjustment_factor: float = (
        0.1  # Factor controlling how aggressively to adjust batch size
    )
    batch_cooldown_period: float = (
        5.0  # Time in seconds to wait between batch size adjustments
    )
    performance_target: float = 0.8  # Target performance score (0.0 to 1.0)

    # Tensor compression settings
    enable_tensor_compression: bool = (
        True  # Enable tensor compression for model weights
    )
    tensor_compression_method: str = (
        "incremental_pca"  # Compression method: "incremental_pca", "svd", "auto"
    )
    tensor_compression_ratio: float = (
        0.5  # Target compression ratio (0.0 to 1.0, where 0.5 = 50% reduction)
    )
    tensor_compression_max_components: int = (
        256  # Maximum number of components to keep during compression
    )
    compression_memory_threshold_high: float = (
        0.8  # Memory threshold for high compression (0.0 to 1.0)
    )
    compression_memory_threshold_critical: float = (
        0.9  # Memory threshold for critical compression (0.0 to 1.0)
    )
    enable_adaptive_compression: bool = (
        True  # Enable adaptive compression that adjusts based on memory usage
    )
    enable_activation_compression: bool = (
        True  # Enable compression of model activations during inference
    )
    compression_update_frequency: int = (
        100  # How often to update compression models (in terms of batches)
    )

    # Tensor decomposition settings
    use_tensor_decomposition: bool = (
        False  # Enable tensor decomposition for model compression
    )
    tensor_decomposition_method: str = (
        "cp_decomposition"  # Decomposition method: "cp_decomposition", "tucker_decomposition", "tensor_train", "matrix_svd"
    )
    tensor_decomposition_rank_ratio: float = (
        0.5  # Target rank ratio (0.0 to 1.0, where 0.5 = 50% of original rank)
    )

    # Structured pruning settings
    use_structured_pruning: bool = (
        False  # Enable structured pruning for model compression
    )
    pruning_ratio: float = 0.2  # Ratio of blocks/layers to remove (0.0 to 1.0)
    pruning_method: str = (
        "layer_removal"  # Pruning method: "layer_removal", "block_removal", "head_removal", "mlp_removal", "adaptive_pruning"
    )
    pruning_block_size: int = 1  # Size of blocks to remove (for block pruning)

    # Qwen3-4B-Instruct-2507 specific optimization settings
    use_qwen3_attention_optimizations: bool = (
        True  # Enable Qwen3-specific attention optimizations
    )
    use_qwen3_kv_cache_optimizations: bool = (
        True  # Enable Qwen3-specific KV-cache optimizations
    )
    use_qwen3_instruction_optimizations: bool = (
        True  # Enable Qwen3-specific instruction tuning optimizations
    )
    use_qwen3_rope_optimizations: bool = (
        True  # Enable Qwen3-specific RoPE optimizations
    )
    use_qwen3_gqa_optimizations: bool = True  # Enable Qwen3-specific GQA optimizations
    qwen3_attention_sparsity_ratio: float = (
        0.3  # Sparsity ratio for Qwen3 attention optimizations
    )
    qwen3_kv_cache_compression_ratio: float = (
        0.6  # Compression ratio for Qwen3 KV-cache optimizations
    )
    qwen3_instruction_attention_scaling: float = (
        1.2  # Scaling factor for instruction-related attention
    )
    qwen3_extended_context_optimization: bool = (
        True  # Enable optimizations for extended context lengths
    )
    qwen3_speculative_decoding_enabled: bool = (
        False  # Enable speculative decoding for faster inference
    )
    qwen3_speculative_draft_model_ratio: float = (
        0.5  # Ratio of draft model size to main model
    )
    qwen3_speculative_max_tokens: int = 5  # Max speculative tokens to generate
    qwen3_instruction_prompt_enhancement: bool = (
        True  # Enable prompt enhancement for instruction tasks
    )
    qwen3_response_quality_optimization: bool = (
        True  # Enable optimizations for response quality
    )
    qwen3_memory_efficient_inference: bool = (
        True  # Enable memory-efficient inference optimizations
    )
    qwen3_compute_efficient_inference: bool = (
        True  # Enable compute-efficient inference optimizations
    )

    # Additional Qwen3-4B specific attributes that may be set by profiles
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
    use_qwen3_vl_attention_optimizations: bool = False  # Placeholder for compatibility
    use_qwen3_vl_kv_cache_optimizations: bool = False  # Placeholder for compatibility
    use_qwen3_vl_vision_optimizations: bool = False  # Placeholder for compatibility
    use_qwen3_vl_cross_modal_optimizations: bool = (
        False  # Placeholder for compatibility
    )
    qwen3_vl_attention_sparsity_ratio: float = 0.0  # Placeholder for compatibility
    qwen3_vl_kv_cache_compression_ratio: float = 0.0  # Placeholder for compatibility
    qwen3_vl_cross_modal_attention_scaling: float = 0.0  # Placeholder for compatibility
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

    # Intelligent Pagination settings
    enable_intelligent_pagination: bool = (
        True  # Enable intelligent pagination for unimodal text data
    )
    pagination_swap_directory: str = (
        "./text_tensor_swap"  # Directory for pagination swap files
    )
    pagination_page_size_mb: int = 16  # Size of pagination pages in MB
    pagination_eviction_policy: str = (
        "intelligent"  # Pagination eviction policy: "lru", "fifo", "priority", "intelligent"
    )
    pagination_max_memory_ratio: float = (
        0.8  # Maximum ratio of system memory to use for pagination
    )
    enable_proactive_pagination: bool = (
        True  # Enable proactive pagination based on access patterns
    )
    proactive_pagination_interval: float = (
        5.0  # Interval for proactive pagination (seconds)
    )

    # Intelligent Cache settings
    intelligent_cache_enabled: bool = True  # Enable intelligent caching system
    intelligent_cache_max_size: int = 1024 * 1024 * 256  # 256MB
    intelligent_cache_precision: str = "float16"  # Cache precision
    intelligent_cache_compression_enabled: bool = True  # Enable compression
    intelligent_cache_compression_method: str = "intelligent"  # Compression method
    intelligent_cache_policy: str = "intelligent"  # Cache policy: lru, fifo, lfu, predictive, intelligent
    intelligent_cache_enable_prefetching: bool = True  # Enable prefetching
    intelligent_cache_prefetch_distance: int = 1  # Distance for prefetching
    intelligent_cache_max_prefix_length: int = 2048  # Max length for cached prefixes
    intelligent_cache_min_prefix_length: int = 8  # Min length for cached prefixes
    intelligent_cache_warmup_threshold: int = 3  # Threshold for warming up cache entries
    intelligent_cache_prediction_horizon: int = 10  # Number of steps to predict ahead
    intelligent_cache_prediction_confidence_threshold: float = 0.7  # Minimum confidence for predictions
    intelligent_cache_enable_adaptive_eviction: bool = True  # Enable adaptive eviction
    intelligent_cache_enable_adaptive_prefetching: bool = True  # Enable adaptive prefetching
    intelligent_cache_adaptive_window_size: int = 100  # Window size for adaptive algorithms
    intelligent_cache_enable_performance_monitoring: bool = True  # Enable performance monitoring
    intelligent_cache_performance_log_interval: int = 100  # Log interval for performance metrics

    # Intelligent scheduling settings
    enable_intelligent_scheduling: bool = True
    intelligent_scheduling_max_concurrent_ops: int = 32
    intelligent_scheduling_policy: str = "intelligent"  # Options: "fifo", "priority", "round_robin", "predictive", "intelligent"
    intelligent_scheduling_enable_prediction: bool = True
    intelligent_scheduling_prediction_horizon: int = 15
    intelligent_scheduling_enable_adaptive: bool = True
    intelligent_scheduling_adaptive_window: int = 150
    intelligent_scheduling_enable_resource_opt: bool = True
    intelligent_scheduling_resource_buffer: float = 0.15
    intelligent_scheduling_enable_priority_boost: bool = True
    intelligent_scheduling_priority_decay: float = 0.92
    intelligent_scheduling_enable_load_balancing: bool = True
    intelligent_scheduling_load_balance_interval: float = 0.08
    intelligent_scheduling_performance_log_interval: int = 75

    # Cross-Alignment Optimization settings
    enable_cross_alignment: bool = True  # Enable cross-alignment optimization
    cross_alignment_temperature: float = 0.5  # Temperature for alignment computation
    cross_alignment_lambda: float = 0.1  # Weight for alignment loss in total loss
    use_cross_alignment_contrastive: bool = True  # Whether to use contrastive alignment loss
    enable_dynamic_cross_alignment: bool = True  # Whether to enable dynamic alignment based on input complexity
    cross_alignment_frequency: int = 10  # Frequency of alignment updates (every N steps)
    cross_alignment_threshold: float = 0.8  # Threshold for alignment quality
    use_cross_alignment_attention: bool = True  # Whether to use attention-based alignment
    use_cross_alignment_learned: bool = True  # Whether to use learned alignment projections
    cross_alignment_projection_dim: int = 512  # Dimension for alignment projections
    enable_cross_alignment_similarity: bool = True  # Whether to enable similarity-based alignment
    cross_alignment_method: str = "qwen3_instruct_specific"  # Default alignment method

    # Continuous NAS settings
    enable_continuous_nas: bool = (
        False  # Enable continuous NAS for architecture adaptation
    )
    nas_strategy: str = (
        "combined_adaptive"  # NAS strategy: "depth_adaptive", "width_adaptive", "combined_adaptive", "latency_based", "memory_based"
    )
    nas_min_depth_ratio: float = 0.3  # Minimum depth as percentage of original
    nas_max_depth_ratio: float = 1.0  # Maximum depth as percentage of original
    nas_min_width_ratio: float = 0.3  # Minimum width as percentage of original
    nas_max_width_ratio: float = 1.0  # Maximum width as percentage of original
    nas_latency_target_ms: float = 100.0  # Target latency in milliseconds
    nas_memory_budget_mb: float = 2048.0  # Memory budget in MB
    nas_accuracy_tradeoff_factor: float = (
        0.7  # Trade-off factor between accuracy and speed (0-1)
    )
    nas_adaptation_frequency: int = (
        10  # How often to adapt (in terms of inference calls)
    )

    def __post_init__(self):
        """
        Initialize fields that depend on other fields.
        """
        # Set default model path if not provided
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)

        # Ensure the model path points to the H drive for Qwen3-4B-Instruct-2507 model
        if (
            not self.model_path
            or "qwen3_4b_instruct_2507" in self.model_path.lower()
            or "qwen3-4b" in self.model_path.lower()
        ):
            self.model_path = "H:/Qwen3-4B-Instruct-2507"

        # Call parent's post_init to validate config
        super().__post_init__()

        if self.global_attention_indices is None:
            self.global_attention_indices = [0]

        if self.bias_removal_config is None:
            self.bias_removal_config = {
                "remove_bias_after_norm": True,
                "remove_bias_in_attention": True,
                "remove_bias_in_mlp": True,
                "remove_bias_in_embeddings": False,
            }

        # Kernel fusion settings
        if (
            not hasattr(self, "kernel_fusion_patterns")
            or self.kernel_fusion_patterns is None
        ):
            self.kernel_fusion_patterns = [
                "linear_relu",
                "linear_gelu",
                "matmul_add",
                "add_layer_norm",
            ]

        # Quantization settings - added for memory optimization
        if not hasattr(self, "use_quantization"):
            self.use_quantization = False
        if not hasattr(self, "quantization_scheme"):
            self.quantization_scheme = "int8"
        if not hasattr(self, "quantization_bits"):
            self.quantization_bits = 8
        if not hasattr(self, "quantization_symmetric"):
            self.quantization_symmetric = True
        if not hasattr(self, "quantization_per_channel"):
            self.quantization_per_channel = True

        # Sequence parallelism settings - added for sequence parallel execution
        if not hasattr(self, "enable_sequence_parallelism"):
            self.enable_sequence_parallelism = False
        if not hasattr(self, "sequence_parallel_num_segments"):
            self.sequence_parallel_num_segments = 1
        if not hasattr(self, "sequence_parallel_split_method"):
            self.sequence_parallel_split_method = "chunk"
        if not hasattr(self, "sequence_parallel_enable_overlap"):
            self.sequence_parallel_enable_overlap = True
        if not hasattr(self, "sequence_parallel_overlap_size"):
            self.sequence_parallel_overlap_size = 64
        if not hasattr(self, "sequence_parallel_algorithm"):
            self.sequence_parallel_algorithm = "1d"

        # Async unimodal processing settings - added for async processing
        if not hasattr(self, "enable_async_unimodal_processing"):
            self.enable_async_unimodal_processing = False
        if not hasattr(self, "async_max_concurrent_requests"):
            self.async_max_concurrent_requests = 4
        if not hasattr(self, "async_buffer_size"):
            self.async_buffer_size = 100
        if not hasattr(self, "async_batch_timeout"):
            self.async_batch_timeout = 0.1
        if not hasattr(self, "enable_async_batching"):
            self.enable_async_batching = True
        if not hasattr(self, "async_processing_device"):
            self.async_processing_device = "cpu"

    def get_model_specific_params(self) -> Dict[str, Any]:
        """
        Return Qwen3-4B-Instruct-2507 specific parameters.

        Returns:
            A dictionary containing Qwen3-4B-Instruct-2507 specific configuration parameters.
        """
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "layer_norm_eps": self.layer_norm_eps,
            "num_key_value_heads": self.num_key_value_heads,
            "initializer_range": self.initializer_range,
        }


# Configurável para integração com o sistema de configuração dinâmica
class Qwen34BDynamicConfig(Qwen34BInstruct2507Config):
    """
    Dynamic configuration class for Qwen3-4B-Instruct model.

    Extends the base Qwen3-4B-Instruct-2507 configuration with dynamic configuration capabilities
    that allow runtime adjustment of model parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Adiciona capacidades de configuração dinâmica se necessário
        """Implement the required functionality."""
        # This is a placeholder implementation
        # In a real implementation, this would contain the actual logic
        return None


# Register this configuration with the factory
try:
    from ...common.config.config_factory import register_model_config
except ImportError:
    # Fallback import
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from src.inference_pio.common.config.config_factory import register_model_config

register_model_config("qwen3_4b_instruct_2507", Qwen34BInstruct2507Config)


__all__ = ["Qwen34BInstruct2507Config", "Qwen34BDynamicConfig"]

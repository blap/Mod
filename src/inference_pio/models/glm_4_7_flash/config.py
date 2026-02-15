"""
GLM-4.7 Configuration - Self-Contained Version

This module defines the configuration class for the GLM-4.7 model in the
self-contained plugin architecture for the Inference-PIO system.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

try:
    from src.inference_pio.common.config.model_config_base import (
        BaseConfig,
        ModelConfigError,
        get_default_model_path,
    )
except ImportError:
    from ...common.config.model_config_base import (
        BaseConfig,
        ModelConfigError,
        get_default_model_path,
    )


@dataclass
class GLM47FlashConfig(BaseConfig):
    """
    Configuration class for the GLM-4.7-Flash model with all parameters consolidated.
    """

    # Model identification
    model_path: str = ""
    model_name: str = "glm_4_7_flash"

    # Environment-specific settings
    use_mock_model: bool = False
    mock_model_size: str = "small"

    # Model architecture parameters
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 47
    max_position_embeddings: int = 202752
    rope_theta: float = 1000000.0
    intermediate_size: int = 10240
    vocab_size: int = 154880
    layer_norm_eps: float = 1e-05
    attention_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    num_key_value_heads: int = 8 # Updated to 8 per user feedback (GQA)
    initializer_range: float = 0.02

    # GLM-4.7-Flash specific architecture parameters
    attention_bias: bool = False
    pad_token_id: int = 154820
    eos_token_id: List[int] = None
    hidden_act: str = "silu"
    moe_intermediate_size: int = 1536
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    routed_scaling_factor: float = 1.8
    num_experts_per_tok: int = 4
    first_k_dense_replace: int = 1
    num_nextn_predict_layers: int = 1
    partial_rotary_factor: float = 1.0
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = False
    transformers_version: str = "5.0.0rc0"
    q_lora_rank: int = 768
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1024
    do_sample: bool = True

    # Optimization flags
    use_sparse_attention: bool = True
    sparse_attention_pattern: str = "longformer"
    sparse_attention_sparsity_ratio: float = 0.25
    sparse_attention_block_size: int = 64
    sparse_attention_local_window_size: int = 128
    use_global_attention: bool = True
    global_attention_indices: List[int] = None
    use_multi_pattern_attention: bool = False
    use_sparse_attention_with_fallback: bool = True
    use_multi_query_attention: bool = True
    use_grouped_query_attention: bool = True
    use_paged_attention: bool = True
    paged_attention_page_size: int = 16
    use_sliding_window_attention: bool = True
    sliding_window_size: int = 4096
    attention_type: str = "gqa"
    num_key_value_groups: int = 4
    use_fused_layer_norm: bool = True
    use_bias_removal_optimization: bool = True
    bias_removal_config: dict = None
    tensor_parallel_size: int = 1
    tensor_parallel_local_rank: int = 0
    tensor_parallel_world_size: int = 1
    tensor_parallel_init_method: str = "tcp://localhost:29500"

    # KV-cache settings
    use_kv_cache_compression: bool = True
    kv_cache_compression_method: str = "combined"
    kv_cache_quantization_bits: int = 8
    kv_cache_low_rank_dimension: int = 64
    kv_cache_adaptive_precision_threshold: float = 0.01
    kv_cache_sparse_compression_ratio: float = 0.5
    kv_cache_enable_dynamic_compression: bool = True

    # Caching settings
    use_prefix_caching: bool = True
    prefix_cache_max_size: int = 1024 * 1024 * 256
    prefix_cache_precision: str = "float16"
    prefix_cache_compression_enabled: bool = True
    prefix_cache_eviction_policy: str = "lru"
    prefix_cache_enable_prefetching: bool = True
    prefix_cache_prefetch_distance: int = 1
    prefix_cache_max_prefix_length: int = 2048
    prefix_cache_min_prefix_length: int = 8
    prefix_cache_warmup_threshold: int = 3

    use_intelligent_caching: bool = True
    intelligent_cache_max_size: int = 1024 * 1024 * 512
    intelligent_cache_precision: str = "float16"
    intelligent_cache_compression_enabled: bool = True
    intelligent_cache_compression_method: str = "intelligent"
    intelligent_cache_policy: str = "intelligent"
    intelligent_cache_enable_prefetching: bool = True
    intelligent_cache_prefetch_distance: int = 2
    intelligent_cache_max_prefix_length: int = 4096
    intelligent_cache_min_prefix_length: int = 4
    intelligent_cache_warmup_threshold: int = 2
    intelligent_cache_prediction_horizon: int = 10
    intelligent_cache_prediction_confidence_threshold: float = 0.7
    intelligent_cache_enable_adaptive_eviction: bool = True
    intelligent_cache_enable_adaptive_prefetching: bool = True
    intelligent_cache_adaptive_window_size: int = 100
    intelligent_cache_enable_performance_monitoring: bool = True
    intelligent_cache_performance_log_interval: int = 100

    # Scheduling
    enable_intelligent_scheduling: bool = True
    intelligent_scheduling_max_concurrent_ops: int = 16
    intelligent_scheduling_policy: str = "intelligent"
    intelligent_scheduling_enable_prediction: bool = True
    intelligent_scheduling_prediction_horizon: int = 10
    intelligent_scheduling_enable_adaptive: bool = True
    intelligent_scheduling_adaptive_window: int = 100
    intelligent_scheduling_enable_resource_opt: bool = True
    intelligent_scheduling_resource_buffer: float = 0.1
    intelligent_scheduling_enable_priority_boost: bool = True
    intelligent_scheduling_priority_decay: float = 0.95
    intelligent_scheduling_enable_load_balancing: bool = True
    intelligent_scheduling_load_balance_interval: float = 0.1
    intelligent_scheduling_performance_log_interval: int = 50

    # Cross-Alignment
    enable_cross_alignment: bool = True
    cross_alignment_temperature: float = 0.5
    cross_alignment_lambda: float = 0.1
    use_cross_alignment_contrastive: bool = True
    enable_dynamic_cross_alignment: bool = True
    cross_alignment_frequency: int = 10
    cross_alignment_threshold: float = 0.8
    use_cross_alignment_attention: bool = True
    use_cross_alignment_learned: bool = True
    cross_alignment_projection_dim: int = 512
    enable_cross_alignment_similarity: bool = True
    cross_alignment_method: str = "glm_specific"

    # Kernels
    use_cuda_kernels: bool = True
    cuda_kernel_gelu_enabled: bool = True
    cuda_kernel_matmul_enabled: bool = True
    cuda_kernel_softmax_enabled: bool = True
    cuda_kernel_attention_enabled: bool = True
    cuda_kernel_mlp_enabled: bool = True
    cuda_kernel_layernorm_enabled: bool = True

    # Linear bias opt
    linear_bias_optimization_enabled: bool = True
    remove_bias_after_norm: bool = True
    remove_bias_in_attention: bool = True
    remove_bias_in_mlp: bool = True
    remove_bias_in_embeddings: bool = False

    # Runtime
    torch_compile_mode: str = "reduce-overhead"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = True
    enable_cudnn_benchmark: bool = True
    enable_memory_efficient_attention: bool = True

    # Memory
    enable_memory_management: bool = True
    max_memory_ratio: float = 0.8
    swap_directory: Optional[str] = None
    page_size_mb: int = 16
    eviction_policy: str = "predictive"
    enable_tensor_paging: bool = True
    enable_smart_swap: bool = True
    tensor_paging_priority: str = "medium"
    pin_optimizer_states: bool = False
    pin_embeddings: bool = True
    pin_attention_weights: bool = False
    memory_cleanup_interval: int = 300

    # Predictive Memory
    enable_predictive_management: bool = True
    prediction_horizon_seconds: int = 30
    proactive_management_interval: float = 5.0
    memory_prediction_threshold: float = 0.9

    # Offloading
    enable_disk_offloading: bool = True
    offload_directory: Optional[str] = None
    offloading_priority: str = "medium"
    offload_attention_weights: bool = False
    enable_predictive_offloading: bool = True
    proactive_offloading_interval: float = 5.0

    # Fusion
    enable_kernel_fusion: bool = True
    kernel_fusion_patterns: List[str] = None
    use_custom_cuda_kernels: bool = True
    custom_kernel_fallback_enabled: bool = True
    kernel_fusion_verbose: bool = False

    # Activation Offloading
    enable_activation_offloading: bool = True
    activation_max_memory_ratio: float = 0.7
    activation_offload_directory: Optional[str] = None
    activation_page_size_mb: int = 8
    activation_eviction_policy: str = "predictive"
    activation_offloading_priority: str = "medium"
    enable_predictive_activation_offloading: bool = True
    proactive_activation_offloading_interval: float = 5.0

    # Adaptive Batching
    enable_adaptive_batching: bool = True
    initial_batch_size: int = 1
    min_batch_size: int = 1
    max_batch_size: int = 16
    memory_threshold_ratio: float = 0.85
    performance_window_size: int = 10
    batch_adjustment_factor: float = 0.1
    batch_cooldown_period: float = 5.0
    performance_target: float = 0.8

    # Compression
    enable_tensor_compression: bool = True
    tensor_compression_method: str = "incremental_pca"
    tensor_compression_ratio: float = 0.5
    tensor_compression_max_components: int = 256
    compression_memory_threshold_high: float = 0.8
    compression_memory_threshold_critical: float = 0.9
    enable_adaptive_compression: bool = True
    enable_activation_compression: bool = True
    compression_update_frequency: int = 100

    # Decomposition
    use_tensor_decomposition: bool = False
    tensor_decomposition_method: str = "cp_decomposition"
    tensor_decomposition_rank_ratio: float = 0.5

    # Pruning
    use_structured_pruning: bool = False
    pruning_ratio: float = 0.2
    pruning_method: str = "layer_removal"
    pruning_block_size: int = 1

    # Pagination
    enable_intelligent_pagination: bool = True
    pagination_swap_directory: str = "./text_tensor_swap"
    pagination_page_size_mb: int = 16
    pagination_eviction_policy: str = "intelligent"
    pagination_max_memory_ratio: float = 0.8
    enable_proactive_pagination: bool = True
    proactive_pagination_interval: float = 5.0

    # NAS
    enable_continuous_nas: bool = False
    nas_strategy: str = "combined_adaptive"
    nas_min_depth_ratio: float = 0.3
    nas_max_depth_ratio: float = 1.0
    nas_min_width_ratio: float = 0.3
    nas_max_width_ratio: float = 1.0
    nas_latency_target_ms: float = 100.0
    nas_memory_budget_mb: float = 2048.0
    nas_accuracy_tradeoff_factor: float = 0.7
    nas_adaptation_frequency: int = 10

    # GLM specific
    use_glm_attention_patterns: bool = True
    glm_attention_pattern_sparsity: float = 0.3
    glm_attention_window_size: int = 1024
    use_glm_ffn_optimization: bool = True
    glm_ffn_expansion_ratio: float = 2.6
    glm_ffn_group_size: int = 128
    use_glm_memory_efficient_kv: bool = True
    glm_kv_cache_compression_ratio: float = 0.5
    use_glm_layer_norm_fusion: bool = True
    use_glm_residual_connection_optimization: bool = True
    use_glm_quantization: bool = True
    glm_weight_bits: int = 4
    glm_activation_bits: int = 8

    # Placeholders
    use_qwen3_attention_optimizations: bool = False
    use_qwen3_kv_cache_optimizations: bool = False
    use_qwen3_instruction_optimizations: bool = False
    use_qwen3_rope_optimizations: bool = False
    use_qwen3_gqa_optimizations: bool = False
    qwen3_attention_sparsity_ratio: float = 0.0
    qwen3_kv_cache_compression_ratio: float = 0.0
    qwen3_instruction_attention_scaling: float = 0.0
    use_qwen3_vl_attention_optimizations: bool = False
    use_qwen3_vl_kv_cache_optimizations: bool = False
    use_qwen3_vl_vision_optimizations: bool = False
    use_qwen3_vl_cross_modal_optimizations: bool = False
    qwen3_vl_attention_sparsity_ratio: float = 0.0
    qwen3_vl_kv_cache_compression_ratio: float = 0.0
    qwen3_vl_cross_modal_attention_scaling: float = 0.0
    use_qwen3_coder_attention_optimizations: bool = False
    use_qwen3_coder_kv_cache_optimizations: bool = False
    use_qwen3_coder_code_optimizations: bool = False
    use_qwen3_coder_syntax_highlighting: bool = False
    qwen3_coder_attention_sparsity_ratio: float = 0.0
    qwen3_coder_kv_cache_compression_ratio: float = 0.0
    qwen3_coder_syntax_attention_scaling: float = 0.0

    def __post_init__(self):
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)

        if (
            not self.model_path
            or "glm_4_7_flash" in self.model_path.lower()
            or "glm-4.7" in self.model_path.lower()
            or "glm_4_7" in self.model_path.lower()
        ):
            self.model_path = "H:/GLM-4.7-Flash"

        if self.eos_token_id is None:
            self.eos_token_id = [154820, 154827, 154829]

        if self.global_attention_indices is None:
            self.global_attention_indices = [0]

        if self.bias_removal_config is None:
            self.bias_removal_config = {
                "remove_bias_after_norm": True,
                "remove_bias_in_attention": True,
                "remove_bias_in_mlp": True,
                "remove_bias_in_embeddings": False,
            }

        if not hasattr(self, "kernel_fusion_patterns") or self.kernel_fusion_patterns is None:
            self.kernel_fusion_patterns = ["linear_relu", "linear_gelu", "matmul_add", "add_layer_norm"]

        # Ensure dynamic attrs exist
        for attr in ["use_quantization", "enable_sequence_parallelism", "enable_async_unimodal_processing"]:
            if not hasattr(self, attr): setattr(self, attr, False)

    def get_model_specific_params(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "vocab_size": self.vocab_size,
        }

class GLM47DynamicConfig(GLM47FlashConfig):
    pass

try:
    from ...common.config.config_factory import register_model_config
    register_model_config("glm_4_7_flash", GLM47FlashConfig)
except ImportError:
    pass

__all__ = ["GLM47FlashConfig", "GLM47DynamicConfig"]

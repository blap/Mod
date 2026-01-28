"""
Common Package for Inference-PIO System

This module provides common utilities for the Inference-PIO system.
"""

from .base_plugin_interface import (
    PluginType,
    ModelPluginMetadata,
    ModelPluginInterface,
    TextModelPluginInterface,
    BaseAttention,
    logger
)
from .kernel_fusion import (
    KernelFusionPass,
    CustomCudaKernels,
    KernelFusionManager,
    get_kernel_fusion_manager,
    kernel_fusion_manager
)

from .base_model import (
    BaseModel,
    BaseTextModel,
    BaseVisionLanguageModel
)

from .config_manager import (
    ConfigManager,
    DynamicConfig,
    GLM47DynamicConfig,
    Qwen34BDynamicConfig,
    Qwen3CoderDynamicConfig,
    Qwen3VLDynamicConfig,
    get_config_manager,
    config_manager,
    register_default_templates
)

from .config_loader import (
    ConfigLoader,
    get_config_loader,
    config_loader,
    create_performance_optimized_config,
    create_memory_efficient_config,
    create_balanced_config,
    create_config_from_profile,
    CONFIG_PROFILES
)

from .config_validator import (
    ConfigValidator,
    get_config_validator,
    config_validator
)

from .config_integration import (
    ConfigurablePluginMixin,
    ConfigurableModelPlugin,
    apply_configuration_to_plugin,
    create_config_from_profile
)

from .base_attention import (
    BaseAttention as BaseAttentionImpl,
    BaseMultiHeadAttention,
    BaseCausalAttention
)

from .tensor_utils import (
    repeat_kv,
    apply_chunking_to_forward,
    gelu_new,
    silu,
    swish,
    softmax_with_temperature,
    masked_fill_with_broadcast,
    normalize_with_l2,
    pad_sequence_to_length,
    truncate_sequence_to_length,
    safe_tensor_operation,
    validate_tensor_shape
)

from .tensor_decomposition import (
    TensorDecomposer,
    AdaptiveTensorDecomposer,
    get_tensor_decomposer,
    decompose_model_weights,
    recompose_model_weights
)

from .rotary_embeddings import (
    GenericRotaryEmbedding
)

from .vision_transformer_kernels import (
    VisionTransformerConfig,
    GenericVisionPatchEmbeddingKernel,
    GenericVisionSelfAttentionKernel,
    GenericVisionMLPKernel,
    GenericVisionTransformerBlockKernel,
    GenericVisionConvolutionKernel,
    GenericVisionEncoderKernel,
    create_generic_vision_patch_embedding_kernel,
    create_generic_vision_self_attention_kernel,
    create_generic_vision_mlp_kernel,
    create_generic_vision_transformer_block_kernel,
    create_generic_vision_encoder_kernel,
    apply_generic_vision_cuda_optimizations_to_model
)

from .flash_attention_2 import (
    FlashAttention2,
    create_flash_attention_2,
    get_flash_attention_2_class
)

from .sparse_attention import (
    SparseAttention,
    create_sparse_attention,
    get_sparse_attention_class
)

from .sliding_window_attention import (
    SlidingWindowAttention,
    create_sliding_window_attention
)

from .paged_attention import (
    PagedAttention,
    create_paged_attention
)

from .adaptive_sparse_attention import (
    AdaptiveSparseAttention,
    create_adaptive_sparse_attention,
    get_adaptive_sparse_attention_class
)

from .memory_manager import (
    MemoryManager,
    TensorPagingManager,
    MemoryPriority,
    MemoryPage,
    get_memory_manager,
    create_memory_manager
)

from .disk_offloading import (
    DiskOffloader,
    TensorOffloadingManager,
    OffloadPriority,
    OffloadPage,
    AccessPattern,
    get_disk_offloader,
    create_disk_offloader
)

from .virtual_execution import (
    VirtualExecutionManager,
    PartitionConfig,
    PartitionStrategy,
    MemorySwapManager
)

from .virtual_device import (
    VirtualDeviceSimulator,
    VirtualExecutionSimulator
)

from .adaptive_batch_manager import (
    AdaptiveBatchManager,
    BatchMetrics,
    BatchSizeAdjustmentReason,
    get_adaptive_batch_manager
)

from .dynamic_text_batching import (
    DynamicTextBatchManager,
    TextBatchType,
    TextBatchInfo,
    get_dynamic_text_batch_manager
)

from .input_complexity_analyzer import (
    InputComplexityAnalyzer,
    ComplexityMetrics,
    get_complexity_analyzer
)

from .model_surgery import (
    ModelSurgerySystem,
    apply_model_surgery,
    restore_model_from_surgery,
    get_model_surgery_system
)

from .unimodal_model_surgery import (
    UnimodalModelSurgerySystem,
    apply_unimodal_model_surgery,
    analyze_unimodal_model_for_surgery,
    get_unimodal_model_surgery_system
)

from .structured_pruning import (
    StructuredPruningSystem,
    PruningMethod,
    PruningResult,
    get_structured_pruning_system,
    apply_structured_pruning
)

from .nas_controller import (
    ContinuousNASController,
    NASConfig,
    ArchitectureAdaptationStrategy,
    ArchitectureState,
    NASMetrics,
    get_nas_controller,
    nas_controller
)

from .model_adapter import (
    BaseModelAdapter,
    get_model_adapter
)


from .streaming_computation import (
    StreamRequest,
    StreamResult,
    StreamingComputationEngine,
    StreamingComputationManager,
    streaming_manager,
    create_streaming_engine
)

from .ml_optimization_selector import (
    OptimizationOutcome,
    PerformanceMetrics,
    OptimizationSelectionData,
    MLSuggestionEngine,
    AutoOptimizationSelector,
    get_auto_selector,
    auto_selector
)

from .hyperparameter_optimizer import (
    HyperparameterConfig,
    OptimizationResult,
    HyperparameterOptimizer,
    PerformanceHyperparameterOptimizer,
    get_performance_optimizer,
    performance_optimizer
)

from .unified_ml_optimization import (
    ModelType,
    MLBasedOptimizationConfig,
    UnifiedMLOptimizationSystem,
    get_ml_optimization_system,
    ml_optimization_system
)

from .feedback_controller import (
    FeedbackEventType,
    FeedbackEvent,
    PerformanceMetrics,
    OptimizationAdjustment,
    FeedbackController,
    get_feedback_controller,
    feedback_controller
)

from .feedback_integration import (
    monitor_performance,
    FeedbackIntegrationMixin,
    apply_feedback_to_model
)

from .multimodal_attention import (
    EfficientMultimodalCrossAttention,
    MultimodalAlignmentModule,
    ModalitySpecificAttention,
    MultimodalFusionLayer,
    AdaptiveMultimodalAttention
)

from .multi_query_attention import (
    MultiQueryAttention as MultiQueryAttentionImpl,
    GroupedQueryAttention as GroupedQueryAttentionImpl,
    create_mqa_gqa_attention
)

from .unimodal_cuda_kernels import (
    UnimodalAttentionKernel,
    UnimodalMLPKernel,
    UnimodalLayerNormKernel,
    UnimodalRMSNormKernel,
    UnimodalHardwareOptimizer,
    create_unimodal_cuda_kernels,
    apply_unimodal_cuda_optimizations_to_model,
    get_unimodal_cuda_optimization_report
)

from .unimodal_preprocessing import (
    TextPreprocessor as UnimodalTextPreprocessor,
    UnimodalPreprocessor,
    create_unimodal_preprocessor,
    apply_unimodal_preprocessing_to_model
)

from .unimodal_preprocessing import (
    TextPreprocessor as UnimodalTextPreprocessor,
    UnimodalPreprocessor,
    create_unimodal_preprocessor,
    apply_unimodal_preprocessing_to_model
)

from .disk_pipeline import (
    PipelineStage,
    DiskBasedPipeline,
    PipelineManager,
    create_tokenization_stage,
    create_model_inference_stage,
    create_decoding_stage
)

from .async_multimodal_processing import (
    AsyncMultimodalRequest,
    AsyncMultimodalResult,
    GenericAsyncMultimodalProcessor,
    create_generic_async_multimodal_engine,
    apply_async_multimodal_processing_to_model
)

from .intelligent_multimodal_caching import (
    CacheEvictionPolicy,
    CacheEntryType,
    CacheEntry,
    GenericIntelligentMultimodalCache,
    GenericIntelligentCachingManager,
    Qwen3VL2BIntelligentCachingManager,
    create_generic_intelligent_caching_manager,
    create_qwen3_vl_intelligent_caching_manager,
    apply_intelligent_multimodal_caching_to_model
)

from .unimodal_cuda_kernels import (
    UnimodalAttentionKernel,
    UnimodalMLPKernel,
    UnimodalLayerNormKernel,
    UnimodalRMSNormKernel,
    UnimodalHardwareOptimizer,
    create_unimodal_cuda_kernels,
    apply_unimodal_cuda_optimizations_to_model,
    get_unimodal_cuda_optimization_report
)

from .tensor_pagination import (
    DataType,
    PaginationPriority,
    TensorPage,
    AccessPatternAnalyzer,
    TensorPaginationSystem,
    MultimodalTensorPager,
    create_multimodal_pagination_system
)

from .unimodal_tensor_pagination import (
    TextDataType,
    TextAccessPatternAnalyzer,
    UnimodalTensorPaginationSystem,
    UnimodalTensorPager,
    create_unimodal_pagination_system
)

from .quantization import (
    QuantizationScheme,
    QuantizationConfig,
    QuantizationManager,
    get_quantization_manager,
    initialize_default_quantization_schemes,
    QuantizedLinear
)

from .pipeline_parallel import (
    PipelineConfig,
    PipelineStage,
    PipelineBalancer,
    PipelineParallel,
    PipelineParallelManager,
    create_pipeline_parallel_config,
    split_model_for_pipeline
)

from .sequence_parallel import (
    SequenceParallel,
    SequenceParallelConfig,
    create_sequence_parallel_config,
    split_sequence_for_parallel
)

__all__ = [
    # Base Plugin Interface
    "PluginType",
    "ModelPluginMetadata",
    "ModelPluginInterface",
    "TextModelPluginInterface",
    "BaseAttention",
    "logger",

    # Configuration Management
    "ConfigManager",
    "DynamicConfig",
    "GLM47DynamicConfig",
    "Qwen34BDynamicConfig",
    "Qwen3CoderDynamicConfig",
    "Qwen3VLDynamicConfig",
    "get_config_manager",
    "config_manager",
    "register_default_templates",

    # Configuration Loader
    "ConfigLoader",
    "get_config_loader",
    "config_loader",
    "create_performance_optimized_config",
    "create_memory_efficient_config",
    "create_balanced_config",
    "create_config_from_profile",
    "CONFIG_PROFILES",

    # Configuration Validator
    "ConfigValidator",
    "get_config_validator",
    "config_validator",

    # Configuration Integration
    "ConfigurablePluginMixin",
    "ConfigurableModelPlugin",
    "apply_configuration_to_plugin",
    "create_config_from_profile",

    # Memory Manager
    "MemoryManager",
    "TensorPagingManager",
    "MemoryPriority",
    "MemoryPage",
    "get_memory_manager",
    "create_memory_manager",

    # Disk Offloading
    "DiskOffloader",
    "TensorOffloadingManager",
    "OffloadPriority",
    "OffloadPage",
    "AccessPattern",
    "get_disk_offloader",
    "create_disk_offloader",

    # Virtual Execution
    "VirtualExecutionManager",
    "PartitionConfig",
    "PartitionStrategy",
    "MemorySwapManager",
    "VirtualDeviceSimulator",
    "VirtualExecutionSimulator",

    # Adaptive Batch Manager
    "AdaptiveBatchManager",
    "BatchMetrics",
    "BatchSizeAdjustmentReason",
    "get_adaptive_batch_manager",
    "DynamicTextBatchManager",
    "TextBatchType",
    "TextBatchInfo",
    "get_dynamic_text_batch_manager",

    # Input Complexity Analyzer
    "InputComplexityAnalyzer",
    "ComplexityMetrics",
    "get_complexity_analyzer",

    # Model Surgery
    "ModelSurgerySystem",
    "apply_model_surgery",
    "restore_model_from_surgery",
    "get_model_surgery_system",

    # Multimodal Model Surgery
    "MultimodalModelSurgerySystem",
    "apply_multimodal_model_surgery",
    "analyze_multimodal_model_for_surgery",
    "get_multimodal_model_surgery_system",

    # Unimodal Model Surgery
    "UnimodalModelSurgerySystem",
    "apply_unimodal_model_surgery",
    "analyze_unimodal_model_for_surgery",
    "get_unimodal_model_surgery_system",

    # Structured Pruning
    "StructuredPruningSystem",
    "PruningMethod",
    "PruningResult",
    "get_structured_pruning_system",
    "apply_structured_pruning",

    # NAS Controller
    "ContinuousNASController",
    "NASConfig",
    "ArchitectureAdaptationStrategy",
    "ArchitectureState",
    "NASMetrics",
    "get_nas_controller",
    "nas_controller",

    # Model Adapter
    "BaseModelAdapter",
    "get_model_adapter",

    # Streaming Computation
    "StreamRequest",
    "StreamResult",
    "StreamingComputationEngine",
    "StreamingComputationManager",
    "streaming_manager",
    "create_streaming_engine",

    # Base Model
    "BaseModel",
    "BaseTextModel",
    "BaseVisionLanguageModel",

    # Base Attention
    "BaseAttentionImpl",
    "BaseMultiHeadAttention",
    "BaseCausalAttention",

    # Flash Attention 2.0
    "FlashAttention2",
    "create_flash_attention_2",
    "get_flash_attention_2_class",

    # Sparse Attention
    "SparseAttention",
    "create_sparse_attention",
    "get_sparse_attention_class",

    # Sliding Window Attention
    "SlidingWindowAttention",
    "create_sliding_window_attention",

    # Paged Attention
    "PagedAttention",
    "create_paged_attention",

    # Adaptive Sparse Attention
    "AdaptiveSparseAttention",
    "create_adaptive_sparse_attention",
    "get_adaptive_sparse_attention_class",

    # Rotary Embeddings
    "RotaryEmbedding",

    # Vision Transformer Kernels
    "VisionTransformerConfig",
    "GenericVisionPatchEmbeddingKernel",
    "GenericVisionSelfAttentionKernel",
    "GenericVisionMLPKernel",
    "GenericVisionTransformerBlockKernel",
    "GenericVisionConvolutionKernel",
    "GenericVisionEncoderKernel",
    "create_generic_vision_patch_embedding_kernel",
    "create_generic_vision_self_attention_kernel",
    "create_generic_vision_mlp_kernel",
    "create_generic_vision_transformer_block_kernel",
    "create_generic_vision_encoder_kernel",
    "apply_generic_vision_cuda_optimizations_to_model",

    # Tensor Utils
    "repeat_kv",
    "apply_chunking_to_forward",
    "gelu_new",
    "silu",
    "swish",
    "softmax_with_temperature",
    "masked_fill_with_broadcast",
    "normalize_with_l2",
    "pad_sequence_to_length",
    "truncate_sequence_to_length",
    "safe_tensor_operation",
    "validate_tensor_shape",

    # Tensor Decomposition
    "TensorDecomposer",
    "AdaptiveTensorDecomposer",
    "get_tensor_decomposer",
    "decompose_model_weights",
    "recompose_model_weights",

    # ML Optimization Selector
    "OptimizationOutcome",
    "PerformanceMetrics",
    "OptimizationSelectionData",
    "MLSuggestionEngine",
    "AutoOptimizationSelector",
    "get_auto_selector",
    "auto_selector",

    # Hyperparameter Optimizer
    "HyperparameterConfig",
    "OptimizationResult",
    "HyperparameterOptimizer",
    "PerformanceHyperparameterOptimizer",
    "get_performance_optimizer",
    "performance_optimizer",

    # Unified ML Optimization
    "ModelType",
    "MLBasedOptimizationConfig",
    "UnifiedMLOptimizationSystem",
    "get_ml_optimization_system",
    "ml_optimization_system",

    # Feedback Controller
    "FeedbackEventType",
    "FeedbackEvent",
    "PerformanceMetrics",
    "OptimizationAdjustment",
    "FeedbackController",
    "get_feedback_controller",
    "feedback_controller",

    # Feedback Integration
    "monitor_performance",
    "FeedbackIntegrationMixin",
    "apply_feedback_to_model",

    # Multimodal Attention
    "EfficientMultimodalCrossAttention",
    "MultimodalAlignmentModule",
    "ModalitySpecificAttention",
    "MultimodalFusionLayer",
    "AdaptiveMultimodalAttention",

    # Multi-Query Attention
    "MultiQueryAttentionImpl",
    "GroupedQueryAttentionImpl",
    "create_mqa_gqa_attention",

    # Unimodal CUDA Kernels
    "UnimodalAttentionKernel",
    "UnimodalMLPKernel",
    "UnimodalLayerNormKernel",
    "UnimodalRMSNormKernel",
    "UnimodalHardwareOptimizer",
    "create_unimodal_cuda_kernels",
    "apply_unimodal_cuda_optimizations_to_model",
    "get_unimodal_cuda_optimization_report",

    # Tensor Pagination
    "DataType",
    "PaginationPriority",
    "TensorPage",
    "AccessPatternAnalyzer",
    "TensorPaginationSystem",
    "MultimodalTensorPager",
    "create_multimodal_pagination_system",
    "TextDataType",
    "TextAccessPatternAnalyzer",
    "UnimodalTensorPaginationSystem",
    "UnimodalTensorPager",
    "create_unimodal_pagination_system",

    # Quantization
    "QuantizationScheme",
    "QuantizationConfig",
    "QuantizationManager",
    "get_quantization_manager",
    "initialize_default_quantization_schemes",
    "QuantizedLinear",

    # Pipeline Parallelism
    "PipelineConfig",
    "PipelineStage",
    "PipelineBalancer",
    "PipelineParallel",
    "PipelineParallelManager",
    "create_pipeline_parallel_config",
    "split_model_for_pipeline",

    # Sequence Parallelism
    "SequenceParallel",
    "SequenceParallelConfig",
    "create_sequence_parallel_config",
    "split_sequence_for_parallel",

    # Multimodal Preprocessing
    "MultimodalTextPreprocessor",
    "ImagePreprocessor",
    "MultimodalPreprocessor",
    "create_multimodal_preprocessor",
    "apply_multimodal_preprocessing_to_model",

    # Unimodal Preprocessing
    "UnimodalTextPreprocessor",
    "UnimodalPreprocessor",
    "create_unimodal_preprocessor",
    "apply_unimodal_preprocessing_to_model",

    # Disk Pipeline
    "PipelineStage",
    "DiskBasedPipeline",
    "PipelineManager",
    "create_tokenization_stage",
    "create_model_inference_stage",
    "create_decoding_stage",

    # Async Multimodal Processing
    "AsyncMultimodalRequest",
    "AsyncMultimodalResult",
    "GenericAsyncMultimodalProcessor",
    "create_generic_async_multimodal_engine",
    "apply_async_multimodal_processing_to_model",

    # Intelligent Multimodal Caching
    "CacheEvictionPolicy",
    "CacheEntryType",
    "CacheEntry",
    "GenericIntelligentMultimodalCache",
    "GenericIntelligentCachingManager",
    "Qwen3VL2BIntelligentCachingManager",
    "create_generic_intelligent_caching_manager",
    "create_qwen3_vl_intelligent_caching_manager",
    "apply_intelligent_multimodal_caching_to_model"
]

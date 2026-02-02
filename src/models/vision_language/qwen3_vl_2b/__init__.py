"""
Qwen3-VL-2B Model Package - Self-Contained Version

This module provides the initialization for the Qwen3-VL-2B model package
in the self-contained plugin architecture for the Inference-PIO system.
Each model plugin is completely independent with its own configuration, tests, and benchmarks.
"""

from .async_multimodal_processing import (
    Qwen3VL2BAsyncMultimodalManager,
    apply_async_multimodal_processing_to_model,
    create_async_multimodal_engine,
)

# Import all model-specific components
from .attention import (
    Qwen3VL2BAdaptiveMultimodalAttention,
    Qwen3VL2BModalitySpecificAttention,
    Qwen3VL2BMultimodalAttention,
    Qwen3VL2BMultimodalFusionLayer,
    create_qwen3_vl_adaptive_multimodal_attention,
    create_qwen3_vl_modality_specific_attention,
    create_qwen3_vl_multimodal_attention,
    create_qwen3_vl_multimodal_fusion_layer,
)
from .config import Qwen3VL2BConfig, Qwen3VLDynamicConfig
from .config_integration import (
    Qwen3VL2BConfigurablePlugin,
)
from .config_integration import (
    create_qwen3_vl_2b_instruct_plugin as create_qwen3_vl_2b_configurable_plugin,
)
from .cross_modal_alignment_optimization import (
    CrossModalAlignmentManager,
    Qwen3VL2BCrossModalAlignmentOptimizer,
    apply_cross_modal_alignment_to_model,
    create_qwen3_vl_cross_modal_alignment,
)
from .cross_modal_fusion_kernels import (
    Qwen3VL2BCrossModalFusionKernel,
    Qwen3VL2BCrossModalFusionManager,
    apply_cross_modal_fusion_to_qwen3_vl_model,
    create_qwen3_vl_cross_modal_fusion,
)
from .intelligent_multimodal_caching import (
    Qwen3VL2BIntelligentCachingManager,
    apply_intelligent_multimodal_caching_to_model,
    create_qwen3_vl_intelligent_caching_manager,
)
from .model import Qwen3VL2BModel
from .multimodal_cuda_kernels import (
    MultimodalCrossAttentionKernel,
    MultimodalFusionKernel,
    MultimodalPositionEncodingKernel,
    VisionLanguageAttentionKernel,
    apply_multimodal_cuda_optimizations_to_model,
)
from .multimodal_projector import (
    Qwen3VL2BMultiModalProjector,
    Qwen3VL2BProjectionLayer,
    Qwen3VL2BVisionLanguageProjector,
    apply_qwen3_vl_projection_optimizations,
    create_qwen3_vl_multimodal_projector,
    create_qwen3_vl_projection_layer,
)
from .plugin import (
    Qwen3_VL_2B_Instruct_Plugin,
    Qwen3_VL_2B_Plugin,
    create_qwen3_vl_2b_instruct_plugin,
    create_qwen3_vl_2b_plugin,
)
from .quantized_multimodal_kernels import (
    QuantizedMultimodalPositionEncodingKernel,
    QuantizedQwen3VL2BCrossAttentionKernel,
    QuantizedQwen3VL2BFusionKernel,
    QuantizedVisionLanguageAttentionKernel,
    apply_quantized_multimodal_optimizations_to_model,
)
from .rotary_embeddings import Qwen3VL2BRotaryEmbedding
from .vision_transformer_kernels import (
    Qwen3VL2BVisionEncoderKernel,
    create_qwen3_vl_2b_vision_encoder_kernel,
)

__all__ = [
    "Qwen3VL2BModel",
    "Qwen3VL2BConfig",
    "Qwen3VLDynamicConfig",
    "Qwen3_VL_2B_Instruct_Plugin",
    "Qwen3_VL_2B_Plugin",
    "create_qwen3_vl_2b_instruct_plugin",
    "create_qwen3_vl_2b_plugin",
    "Qwen3VL2BConfigurablePlugin",
    "create_qwen3_vl_2b_configurable_plugin",
    # Attention components
    "Qwen3VL2BMultimodalAttention",
    "Qwen3VL2BModalitySpecificAttention",
    "Qwen3VL2BMultimodalFusionLayer",
    "Qwen3VL2BAdaptiveMultimodalAttention",
    "create_qwen3_vl_multimodal_attention",
    "create_qwen3_vl_modality_specific_attention",
    "create_qwen3_vl_multimodal_fusion_layer",
    "create_qwen3_vl_adaptive_multimodal_attention",
    # CUDA kernels
    "MultimodalCrossAttentionKernel",
    "MultimodalFusionKernel",
    "VisionLanguageAttentionKernel",
    "MultimodalPositionEncodingKernel",
    "apply_multimodal_cuda_optimizations_to_model",
    # Cross-modal fusion
    "Qwen3VL2BCrossModalFusionKernel",
    "Qwen3VL2BCrossModalFusionManager",
    "create_qwen3_vl_cross_modal_fusion",
    "apply_cross_modal_fusion_to_qwen3_vl_model",
    # Cross-modal alignment
    "Qwen3VL2BCrossModalAlignmentOptimizer",
    "CrossModalAlignmentManager",
    "create_qwen3_vl_cross_modal_alignment",
    "apply_cross_modal_alignment_to_model",
    # Rotary embeddings
    "Qwen3VL2BRotaryEmbedding",
    # Vision transformer kernels
    "Qwen3VL2BVisionEncoderKernel",
    "create_qwen3_vl_2b_vision_encoder_kernel",
    # Multimodal projector
    "Qwen3VL2BProjectionLayer",
    "Qwen3VL2BMultiModalProjector",
    "Qwen3VL2BVisionLanguageProjector",
    "create_qwen3_vl_projection_layer",
    "create_qwen3_vl_multimodal_projector",
    "apply_qwen3_vl_projection_optimizations",
    # Quantized kernels
    "QuantizedQwen3VL2BCrossAttentionKernel",
    "QuantizedQwen3VL2BFusionKernel",
    "QuantizedVisionLanguageAttentionKernel",
    "QuantizedMultimodalPositionEncodingKernel",
    "apply_quantized_multimodal_optimizations_to_model",
    # Async multimodal processing
    "Qwen3VL2BAsyncMultimodalManager",
    "create_async_multimodal_engine",
    "apply_async_multimodal_processing_to_model",
    # Intelligent caching
    "Qwen3VL2BIntelligentCachingManager",
    "create_qwen3_vl_intelligent_caching_manager",
    "apply_intelligent_multimodal_caching_to_model",
]

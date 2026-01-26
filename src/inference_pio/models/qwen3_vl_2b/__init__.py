"""
Qwen3-VL-2B Model Package - Self-Contained Version

This module provides the initialization for the Qwen3-VL-2B model package
in the self-contained plugin architecture for the Inference-PIO system.
"""

from .model import Qwen3VL2BModel
from .config import Qwen3VL2BConfig
from .plugin import (
    Qwen3_VL_2B_Instruct_Plugin,
    Qwen3_VL_2B_Plugin,
    create_qwen3_vl_2b_instruct_plugin,
    create_qwen3_vl_2b_plugin
)

# Import all model-specific components
from .attention import (
    Qwen3VL2BMultimodalAttention,
    Qwen3VL2BModalitySpecificAttention,
    Qwen3VL2BMultimodalFusionLayer,
    Qwen3VL2BAdaptiveMultimodalAttention,
    create_qwen3_vl_multimodal_attention,
    create_qwen3_vl_modality_specific_attention,
    create_qwen3_vl_multimodal_fusion_layer,
    create_qwen3_vl_adaptive_multimodal_attention
)

from .multimodal_cuda_kernels import (
    MultimodalCrossAttentionKernel,
    MultimodalFusionKernel,
    VisionLanguageAttentionKernel,
    MultimodalPositionEncodingKernel,
    apply_multimodal_cuda_optimizations_to_model
)

from .cross_modal_fusion_kernels import (
    Qwen3VL2BCrossModalFusionKernel,
    Qwen3VL2BCrossModalFusionManager,
    create_qwen3_vl_cross_modal_fusion,
    apply_cross_modal_fusion_to_qwen3_vl_model
)

from .cross_modal_alignment_optimization import (
    Qwen3VL2BCrossModalAlignmentOptimizer,
    CrossModalAlignmentManager,
    create_qwen3_vl_cross_modal_alignment,
    apply_cross_modal_alignment_to_model
)

from .rotary_embeddings import (
    Qwen3VL2BRotaryEmbedding
)

from .vision_transformer_kernels import (
    Qwen3VL2BVisionEncoderKernel,
    create_qwen3_vl_2b_vision_encoder_kernel
)

from .multimodal_projector import (
    Qwen3VL2BProjectionLayer,
    Qwen3VL2BMultiModalProjector,
    Qwen3VL2BVisionLanguageProjector,
    create_qwen3_vl_projection_layer,
    create_qwen3_vl_multimodal_projector,
    apply_qwen3_vl_projection_optimizations
)

from .quantized_multimodal_kernels import (
    QuantizedQwen3VL2BCrossAttentionKernel,
    QuantizedQwen3VL2BFusionKernel,
    QuantizedVisionLanguageAttentionKernel,
    QuantizedMultimodalPositionEncodingKernel,
    apply_quantized_multimodal_optimizations_to_model
)

from .async_multimodal_processing import (
    Qwen3VL2BAsyncMultimodalManager,
    create_async_multimodal_engine,
    apply_async_multimodal_processing_to_model
)

from .intelligent_multimodal_caching import (
    Qwen3VL2BIntelligentCachingManager,
    create_qwen3_vl_intelligent_caching_manager,
    apply_intelligent_multimodal_caching_to_model
)

__all__ = [
    "Qwen3VL2BModel",
    "Qwen3VL2BConfig",
    "Qwen3_VL_2B_Instruct_Plugin",
    "Qwen3_VL_2B_Plugin",
    "create_qwen3_vl_2b_instruct_plugin",
    "create_qwen3_vl_2b_plugin",
    
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
    "apply_intelligent_multimodal_caching_to_model"
]
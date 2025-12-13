"""
Qwen3-VL Model implementation with full capacity preservation and Phase 2 efficiency improvements
This file now serves as a compatibility layer that imports from the new modular structure.
"""
from src.qwen3_vl.components.models.qwen3_vl_model import Qwen3VLForConditionalGeneration
from src.qwen3_vl.components.models.qwen3_vl_model import Qwen3VLPreTrainedModel
from src.qwen3_vl.model_layers.attention_mechanisms import Qwen3VLAttention, Qwen3VLVisionAttention, repeat_kv
from src.qwen3_vl.attention.rotary_embeddings import Qwen3VLRotaryEmbedding, apply_rotary_pos_emb, rotate_half
from src.qwen3_vl.model_layers.layer_components import Qwen3VLMLP, Qwen3VLVisionMLP, Qwen3VLDecoderLayer, Qwen3VLVisionLayer
from src.qwen3_vl.model_layers.vision_transformer import Qwen3VLVisionTransformer
from src.qwen3_vl.model_layers.language_decoder import Qwen3VLDecoder
from src.qwen3_vl.model_layers.multimodal_projector import Qwen3VLMultimodalProjector
from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.optimization.adaptive_computation import AdaptiveAttention, AdaptiveMLP
from src.qwen3_vl.optimization.gradient_checkpointing import MemoryEfficientAttention, MemoryEfficientMLP
from src.qwen3_vl.attention.optimized_attention import OptimizedQwen3VLAttention
from src.qwen3_vl.optimization.adapter_layers import AdapterLayer
from src.qwen3_vl.optimization.adaptive_depth import InputComplexityAssessor, AdaptiveDepthController
from src.qwen3_vl.optimization.adaptive_depth_transformer import AdaptiveDepthTransformer, VisionAdaptiveDepthTransformer, MultimodalAdaptiveDepthFusion
from src.qwen3_vl.optimization.context_adaptive_positional_encoding import Qwen3VLContextAdaptivePositionalProcessor
from src.qwen3_vl.optimization.conditional_feature_extraction import ConditionalFeatureExtractor
from src.qwen3_vl.optimization.adaptive_precision import AdaptivePrecisionController, LayerWisePrecisionSelector, PrecisionAdaptiveLayer, AdaptivePrecisionAttention
from src.qwen3_vl.optimization.cross_modal_compression import CrossModalMemoryCompressor
from src.qwen3_vl.optimization.memory_sharing import CrossLayerMemoryManager
from src.qwen3_vl.hardware_optimization.hardware_abstraction import DeviceAwareAttention
from src.qwen3_vl.optimization.dynamic_sparse import DynamicSparseAttention, VisionDynamicSparseAttention


__all__ = [
    "Qwen3VLPreTrainedModel",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLAttention",
    "Qwen3VLVisionAttention",
    "repeat_kv",
    "Qwen3VLRotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    "Qwen3VLMLP",
    "Qwen3VLVisionMLP",
    "Qwen3VLDecoderLayer",
    "Qwen3VLVisionLayer",
    "Qwen3VLVisionTransformer",
    "Qwen3VLDecoder",
    "Qwen3VLMultimodalProjector",
    "Qwen3VLConfig",
    "AdaptiveAttention",
    "AdaptiveMLP",
    "MemoryEfficientAttention",
    "MemoryEfficientMLP",
    "OptimizedQwen3VLAttention",
    "AdapterLayer",
    "InputComplexityAssessor",
    "AdaptiveDepthController",
    "AdaptiveDepthTransformer",
    "VisionAdaptiveDepthTransformer",
    "MultimodalAdaptiveDepthFusion",
    "Qwen3VLContextAdaptivePositionalProcessor",
    "ConditionalFeatureExtractor",
    "AdaptivePrecisionController",
    "LayerWisePrecisionSelector",
    "PrecisionAdaptiveLayer",
    "AdaptivePrecisionAttention",
    "CrossModalMemoryCompressor",
    "CrossLayerMemoryManager",
    "DeviceAwareAttention",
    "DynamicSparseAttention",
    "VisionDynamicSparseAttention"
]
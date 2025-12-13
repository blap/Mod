"""
Main initialization module for Qwen3-VL with dependency injection setup.
This module configures the dependency injection container with all components.
"""

from .config.config import Qwen3VLConfig

# Import and expose top-level modules for backward compatibility
try:
    from .attention import (
        Qwen3VLAttention, FlashAttention2, BlockSparseAttention, DynamicSparseAttention, 
        OptimizedQwen3VLAttention, Qwen3VLRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
    )
except ImportError:
    pass

try:
    from .models import (
        Qwen3VLModel, Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen3VLPreTrainedModel
    )
except ImportError:
    pass

try:
    from .multimodal import (
        CrossModalCompression, CrossModalMemoryCompressor, CrossModalTokenMerger, ConditionalFeatureExtractor
    )
except ImportError:
    pass

try:
    from .vision import (
        Qwen3VLVisionTransformer, Qwen3VLVisionModel, Qwen3VLVisionAttention, HierarchicalVisionProcessor
    )
except ImportError:
    pass

try:
    from .language import (
        Qwen3VLDecoder, Qwen3VLLanguageModel
    )
except ImportError:
    pass

# Import system-level components
try:
    from . import components
    from . import optimization
    from . import memory_management
    from . import inference
    from . import training
    from . import utils
    from . import hardware
    from . import architectures
    from . import memory_optimization
    from . import config
except ImportError:
    pass

# Expose key system components
from .components.system.di_container import create_default_container, setup_qwen3_vl_system
from .components.system.pipeline import IntelOptimizedPipeline
from .components.system.interfaces import (
    ConfigurableComponent, MemoryManager, Optimizer, Preprocessor, Pipeline, 
    AttentionMechanism, MLP, Layer
)

# Expose memory optimization components
try:
    from .memory_optimization import (
        HierarchicalMemoryCompressor, CrossModalMemoryCompressor as IntegratedCrossModalCompressor, 
        MemoryEfficientAttention, MemoryEfficientMLP, GradientCheckpointingWrapper
    )
except ImportError:
    pass

# Expose optimization components
try:
    from .optimization import (
        AdaptiveAttention, AdaptiveMLP, AdaptivePrecisionController, LayerWisePrecisionSelector, 
        CrossLayerMemoryManager, DynamicSparseAttention as OptimizedDynamicSparseAttention, 
        MoeLayer, FlashAttention, LowRankKVCache, SlidingWindowKVCache, HybridKVCache, 
        OptimizedKVCachingAttention, CrossModalMemoryCompressor as OptimizationCrossModalCompressor, 
        ContextAdaptivePositionalEncoding, ConditionalFeatureExtractor as OptimizationConditionalFeatureExtractor, 
        AdaptiveDepthController, InputComplexityAssessor
    )
except ImportError:
    pass

__version__ = "3.0.0"
__author__ = "Qwen3-VL Team"

# Example usage
if __name__ == "__main__":
    # Example of how to set up the system
    print("Setting up Qwen3-VL system...")

    # Create or load configuration
    config = Qwen3VLConfig()

    print("Qwen3-VL system setup complete!")
    print(f"Config: {type(config).__name__}")
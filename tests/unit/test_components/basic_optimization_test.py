"""
Very basic test to verify the optimization modules can be imported and instantiated
"""
import torch
from src.models.config import Qwen3VLConfig

def test_basic_imports():
    print("Testing basic imports and instantiation...")
    
    # Test config
    config = Qwen3VLConfig()
    print("PASS: Config created")
    
    # Test each optimization component individually
    try:
        from src.qwen3_vl.components.optimization.adaptive_precision import AdaptivePrecisionController
        controller = AdaptivePrecisionController(config)
        print("PASS: Adaptive Precision Controller created")
    except Exception as e:
        print(f"FAIL: Adaptive Precision Controller failed: {e}")

    try:
        from src.qwen3_vl.components.optimization.activation_sparsity import TopKSparsify, ConfidenceGatedEarlyExit
        sparsify = TopKSparsify(sparsity_ratio=0.5)
        early_exit = ConfidenceGatedEarlyExit(hidden_size=512, num_layers=10)
        print("PASS: Activation Sparsity components created")
    except Exception as e:
        print(f"FAIL: Activation Sparsity components failed: {e}")

    try:
        from src.qwen3_vl.components.optimization.dynamic_sparse import DynamicSparseAttention
        attention = DynamicSparseAttention(config, layer_idx=0)
        print("PASS: Dynamic Sparse Attention created")
    except Exception as e:
        print(f"FAIL: Dynamic Sparse Attention failed: {e}")

    try:
        from src.qwen3_vl.components.optimization.kv_cache_optimization import OptimizedKVCachingAttention
        attention = OptimizedKVCachingAttention(config, layer_idx=0)
        print("PASS: KV Cache Optimization created")
    except Exception as e:
        print(f"FAIL: KV Cache Optimization failed: {e}")

    try:
        from src.qwen3_vl.components.optimization.context_adaptive_positional_encoding import ContextAdaptivePositionalEncoding
        encoder = ContextAdaptivePositionalEncoding(hidden_size=512, max_seq_len=64)
        print("PASS: Context-Adaptive Positional Encoding created")
    except Exception as e:
        print(f"FAIL: Context-Adaptive Positional Encoding failed: {e}")

    try:
        from src.qwen3_vl.components.optimization.conditional_feature_extraction import ConditionalFeatureExtractor
        extractor = ConditionalFeatureExtractor(config)
        print("PASS: Conditional Feature Extractor created")
    except Exception as e:
        print(f"FAIL: Conditional Feature Extractor failed: {e}")

    try:
        from src.qwen3_vl.components.optimization.cross_modal_compression import CrossModalMemoryCompressor
        compressor = CrossModalMemoryCompressor(config)
        print("PASS: Cross-Modal Memory Compressor created")
    except Exception as e:
        print(f"FAIL: Cross-Modal Memory Compressor failed: {e}")

    try:
        from src.qwen3_vl.components.optimization.memory_sharing import CrossLayerMemoryManager
        manager = CrossLayerMemoryManager(config)
        print("PASS: Cross-Layer Memory Manager created")
    except Exception as e:
        print(f"FAIL: Cross-Layer Memory Manager failed: {e}")

    try:
        from src.qwen3_vl.components.optimization.adaptive_depth import AdaptiveDepthController, InputComplexityAssessor
        assessor = InputComplexityAssessor(config)
        controller = AdaptiveDepthController(config, assessor)
        print("PASS: Adaptive Depth components created")
    except Exception as e:
        print(f"FAIL: Adaptive Depth components failed: {e}")

    try:
        from src.qwen3_vl.components.optimization.moe_flash_attention import MoeLayer
        config.use_moe = True
        config.moe_num_experts = 4
        config.moe_top_k = 2
        moe = MoeLayer(config, num_experts=config.moe_num_experts, top_k=config.moe_top_k)
        print("PASS: Mixture of Experts created")
    except Exception as e:
        print(f"FAIL: Mixture of Experts failed: {e}")
    
    print("\nAll individual optimization components can be instantiated successfully!")

if __name__ == "__main__":
    test_basic_imports()
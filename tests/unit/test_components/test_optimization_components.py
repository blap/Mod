"""
Test individual optimization components to verify they work correctly.
"""
import torch
import torch.nn as nn
from src.qwen3_vl.core.config import Qwen3VLConfig
from src.components.optimization.adaptive_precision import AdaptivePrecisionController
from src.components.optimization.activation_sparsity import TopKSparsify, ConfidenceGatedEarlyExit
from src.components.optimization.dynamic_sparse import DynamicSparseAttention
from src.components.optimization.kv_cache_optimization import OptimizedKVCachingAttention
from src.components.optimization.context_adaptive_positional_encoding import ContextAdaptivePositionalEncoding
from src.components.optimization.conditional_feature_extraction import ConditionalFeatureExtractor
from src.components.optimization.cross_modal_compression import CrossModalMemoryCompressor
from src.components.optimization.memory_sharing import CrossLayerMemoryManager
from src.components.optimization.adaptive_depth import AdaptiveDepthController, InputComplexityAssessor
from src.components.optimization.moe_flash_attention import MoeLayer


def test_adaptive_precision():
    """Test adaptive precision controller."""
    print("Testing Adaptive Precision Controller...")
    
    config = Qwen3VLConfig()
    controller = AdaptivePrecisionController(config)
    
    # Test profiling precision sensitivity
    dummy_input = torch.randn(2, 16, config.hidden_size)
    sensitivity = controller.profile_precision_sensitivity(dummy_input)
    
    print(f"Precision sensitivity profiled for {len(sensitivity)} layers")
    
    # Test precision selection
    requirements = {
        'computation_intensity': 0.5,
        'sensitivity_to_precision': 0.3,
        'memory_footprint': 0.7,
        'accuracy_importance': 0.8
    }
    
    precision = controller.select_optimal_precision(0, requirements)
    print(f"Selected precision for layer 0: {precision}")
    
    return True


def test_activation_sparsity():
    """Test activation sparsity mechanisms."""
    print("\nTesting Activation Sparsity...")
    
    # Test Top-K sparsification
    sparsify = TopKSparsify(sparsity_ratio=0.5)
    dummy_input = torch.randn(2, 16, 512)
    output = sparsify(dummy_input)
    
    # Check that approximately 50% of values are zero
    zero_ratio = (output == 0).float().mean()
    print(f"Zero ratio after sparsification: {zero_ratio:.3f} (expected ~0.5)")
    
    # Test early exit mechanism
    early_exit = ConfidenceGatedEarlyExit(hidden_size=512, num_layers=10, exit_threshold=0.8)
    should_exit = early_exit(dummy_input, 5)  # Test at layer 5
    
    print(f"Early exit test completed")
    
    return True


def test_dynamic_sparse_attention():
    """Test dynamic sparse attention."""
    print("\nTesting Dynamic Sparse Attention...")
    
    config = Qwen3VLConfig()
    attention = DynamicSparseAttention(config, layer_idx=0)
    
    # Create dummy inputs
    hidden_states = torch.randn(2, 10, config.hidden_size)
    position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
    
    try:
        output, _, _ = attention(
            hidden_states=hidden_states,
            position_ids=position_ids
        )
        print(f"Dynamic sparse attention output shape: {output.shape}")
        print("Dynamic sparse attention test passed")
        return True
    except Exception as e:
        print(f"Dynamic sparse attention test failed: {e}")
        return False


def test_kv_cache_optimization():
    """Test KV cache optimization."""
    print("\nTesting KV Cache Optimization...")
    
    config = Qwen3VLConfig()
    attention = OptimizedKVCachingAttention(
        config, 
        layer_idx=0,
        cache_strategy="hybrid"
    )
    
    # Create dummy inputs
    hidden_states = torch.randn(1, 8, config.hidden_size)
    position_ids = torch.arange(8).unsqueeze(0)
    
    try:
        output, _, _ = attention(
            hidden_states=hidden_states,
            position_ids=position_ids
        )
        print(f"KV cache optimized attention output shape: {output.shape}")
        print("KV cache optimization test passed")
        return True
    except Exception as e:
        print(f"KV cache optimization test failed: {e}")
        return False


def test_context_adaptive_positional_encoding():
    """Test context-adaptive positional encoding."""
    print("\nTesting Context-Adaptive Positional Encoding...")
    
    encoder = ContextAdaptivePositionalEncoding(
        hidden_size=512,
        max_seq_len=64
    )
    
    # Create dummy inputs
    hidden_states = torch.randn(2, 16, 512)
    position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)
    
    try:
        output = encoder(
            hidden_states=hidden_states,
            position_ids=position_ids
        )
        print(f"Context-adaptive positional encoding output shape: {output.shape}")
        print("Context-adaptive positional encoding test passed")
        return True
    except Exception as e:
        print(f"Context-adaptive positional encoding test failed: {e}")
        return False


def test_conditional_feature_extraction():
    """Test conditional feature extraction."""
    print("\nTesting Conditional Feature Extraction...")
    
    config = Qwen3VLConfig()
    extractor = ConditionalFeatureExtractor(config)
    
    # Create dummy text input
    text_input = torch.randint(0, 1000, (2, 16))
    
    try:
        features, modality_info = extractor(text_input=text_input)
        print(f"Conditional feature extraction output shape: {features.shape}")
        print(f"Modality: {modality_info['modality']}")
        print("Conditional feature extraction test passed")
        return True
    except Exception as e:
        print(f"Conditional feature extraction test failed: {e}")
        return False


def test_cross_modal_memory_compression():
    """Test cross-modal memory compression."""
    print("\nTesting Cross-Modal Memory Compression...")
    
    config = Qwen3VLConfig()
    compressor = CrossModalMemoryCompressor(config)
    
    # Create dummy features
    text_features = torch.randn(2, 10, config.hidden_size)
    vision_features = torch.randn(2, 10, config.vision_hidden_size)
    
    try:
        compressed_text, compressed_vision = compressor(
            text_features=text_features,
            vision_features=vision_features
        )
        print(f"Compressed text shape: {compressed_text.shape}")
        print(f"Compressed vision shape: {compressed_vision.shape}")
        print("Cross-modal memory compression test passed")
        return True
    except Exception as e:
        print(f"Cross-modal memory compression test failed: {e}")
        return False


def test_cross_layer_memory_sharing():
    """Test cross-layer memory sharing."""
    print("\nTesting Cross-Layer Memory Sharing...")
    
    config = Qwen3VLConfig()
    memory_manager = CrossLayerMemoryManager(config)
    
    # Create dummy hidden states
    hidden_states = torch.randn(2, 16, config.hidden_size)
    
    try:
        updated_states = memory_manager(
            hidden_states=hidden_states,
            layer_idx=5,
            layer_input=hidden_states
        )
        print(f"Cross-layer memory sharing output shape: {updated_states.shape}")
        print("Cross-layer memory sharing test passed")
        return True
    except Exception as e:
        print(f"Cross-layer memory sharing test failed: {e}")
        return False


def test_adaptive_depth():
    """Test adaptive depth mechanisms."""
    print("\nTesting Adaptive Depth...")
    
    config = Qwen3VLConfig()
    assessor = InputComplexityAssessor(config)
    controller = AdaptiveDepthController(config, assessor)
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 16))
    
    try:
        num_layers, complexity = controller(input_ids=input_ids)
        print(f"Adaptive depth selected {num_layers} layers for complexity {complexity:.3f}")
        print("Adaptive depth test passed")
        return True
    except Exception as e:
        print(f"Adaptive depth test failed: {e}")
        return False


def test_moe_layer():
    """Test Mixture of Experts layer."""
    print("\nTesting Mixture of Experts...")
    
    config = Qwen3VLConfig()
    config.use_moe = True
    config.moe_num_experts = 4
    config.moe_top_k = 2
    
    try:
        moe_layer = MoeLayer(
            config,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k
        )
        
        # Create dummy input
        dummy_input = torch.randn(2, 16, config.hidden_size)
        output = moe_layer(dummy_input)
        
        print(f"MoE layer output shape: {output.shape}")
        print("Mixture of Experts test passed")
        return True
    except Exception as e:
        print(f"Mixture of Experts test failed: {e}")
        return False


def run_all_tests():
    """Run all optimization component tests."""
    print("=" * 60)
    print("TESTING INDIVIDUAL OPTIMIZATION COMPONENTS")
    print("=" * 60)
    
    tests = [
        test_adaptive_precision,
        test_activation_sparsity,
        test_dynamic_sparse_attention,
        test_kv_cache_optimization,
        test_context_adaptive_positional_encoding,
        test_conditional_feature_extraction,
        test_cross_modal_memory_compression,
        test_cross_layer_memory_sharing,
        test_adaptive_depth,
        test_moe_layer
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test_func.__name__}: {status}")
    
    total_passed = sum(results)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    print(f"\nOverall success: {success}")
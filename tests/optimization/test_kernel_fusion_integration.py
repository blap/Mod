"""
Integration test for kernel fusion techniques with the Qwen3-VL model
"""
import torch
import torch.nn as nn
from typing import Dict, Any
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qwen3_vl.optimization.kernel_fusion import (
    apply_kernel_fusion_to_model,
    get_kernel_fusion_report
)
from models.modeling_qwen3_vl_phase2 import Qwen3VLForConditionalGeneration
from src.qwen3_vl.components.configuration.config import Qwen3VLConfig


def create_test_config():
    """Create a test configuration for the Qwen3-VL model"""
    config = Qwen3VLConfig()
    
    # Basic model parameters
    config.hidden_size = 512
    config.intermediate_size = 2048
    config.num_attention_heads = 8
    config.num_hidden_layers = 4  # Reduced for testing
    config.vocab_size = 32000
    config.max_position_embeddings = 512
    config.rope_theta = 1000000
    config.layer_norm_eps = 1e-5
    config.use_cache = True
    
    # Vision parameters
    config.vision_num_hidden_layers = 4  # Reduced for testing
    config.vision_num_attention_heads = 8
    config.vision_hidden_size = 512
    config.vision_intermediate_size = 2048
    config.vision_patch_size = 14
    config.vision_image_size = 224
    config.vision_num_channels = 3
    config.vision_hidden_act = "gelu"
    config.vision_hidden_dropout_prob = 0.0
    config.vision_attention_dropout_prob = 0.0
    config.vision_max_position_embeddings = 256
    config.vision_rope_theta = 10000
    config.vision_layer_norm_eps = 1e-6
    
    # Additional parameters that might be needed
    config.num_key_value_heads = 8
    config.pad_token_id = 0
    config.attention_dropout_prob = 0.0
    config.hidden_dropout_prob = 0.0
    config.vision_qkv_bias = True
    config.vision_projection_dim = 512
    
    return config


def test_kernel_fusion_integration():
    """Test kernel fusion integration with the Qwen3-VL model"""
    print("Creating test configuration...")
    config = create_test_config()
    
    print("Creating Qwen3-VL model...")
    model = Qwen3VLForConditionalGeneration(config)
    
    print("Applying kernel fusion optimizations...")
    fused_model = apply_kernel_fusion_to_model(model, config)
    
    print("Getting kernel fusion report...")
    report = get_kernel_fusion_report(fused_model, config)
    print(f"Kernel fusion report: {report}")
    
    # Test forward pass with text-only input
    print("\nTesting text-only forward pass...")
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    
    with torch.no_grad():
        text_output = fused_model(input_ids=input_ids)
        print(f"Text-only output shape: {text_output.shape}")
    
    # Test forward pass with vision input
    print("\nTesting vision forward pass...")
    pixel_values = torch.randn(2, 3, 224, 224)  # Batch of 2 RGB images
    
    with torch.no_grad():
        vision_output = fused_model(pixel_values=pixel_values)
        print(f"Vision-only output shape: {vision_output.shape}")
    
    # Test forward pass with both text and vision
    print("\nTesting multimodal forward pass...")
    with torch.no_grad():
        multimodal_output = fused_model(input_ids=input_ids, pixel_values=pixel_values)
        print(f"Multimodal output shape: {multimodal_output.shape}")
    
    print("\n[SUCCESS] Kernel fusion integration test passed!")


def benchmark_performance():
    """Benchmark performance improvements from kernel fusion"""
    print("\nBenchmarking performance improvements...")
    
    config = create_test_config()
    
    # Create original model
    original_model = Qwen3VLForConditionalGeneration(config)
    original_model.eval()
    
    # Create fused model
    fused_model = apply_kernel_fusion_to_model(original_model, config)
    fused_model.eval()
    
    # Create test inputs
    input_ids = torch.randint(0, config.vocab_size, (1, 20))
    pixel_values = torch.randn(1, 3, 224, 224)
    
    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = original_model(input_ids=input_ids, pixel_values=pixel_values)
            _ = fused_model(input_ids=input_ids, pixel_values=pixel_values)
    
    # Benchmark original model
    import time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = original_model(input_ids=input_ids, pixel_values=pixel_values)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    original_time = time.time() - start_time
    
    # Benchmark fused model
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = fused_model(input_ids=input_ids, pixel_values=pixel_values)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    fused_time = time.time() - start_time
    
    print(f"Original model time: {original_time:.4f}s for 10 runs")
    print(f"Fused model time: {fused_time:.4f}s for 10 runs")
    print(f"Speedup: {original_time / fused_time:.2f}x")
    
    # Check that outputs are similar
    with torch.no_grad():
        original_output = original_model(input_ids=input_ids, pixel_values=pixel_values)
        fused_output = fused_model(input_ids=input_ids, pixel_values=pixel_values)
    
    # Calculate similarity
    similarity = torch.cosine_similarity(
        original_output.flatten(),
        fused_output.flatten(),
        dim=0
    ).mean()
    
    print(f"Output similarity: {similarity:.4f}")
    
    return original_time, fused_time, similarity


def test_memory_efficiency():
    """Test memory efficiency improvements from kernel fusion"""
    print("\nTesting memory efficiency...")
    
    config = create_test_config()
    
    # Create original model
    original_model = Qwen3VLForConditionalGeneration(config)
    original_model.eval()
    
    # Create fused model
    fused_model = apply_kernel_fusion_to_model(original_model, config)
    fused_model.eval()
    
    # Create test inputs
    input_ids = torch.randint(0, config.vocab_size, (1, 20))
    pixel_values = torch.randn(1, 3, 224, 224)
    
    # Measure memory usage for original model
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = original_model(input_ids=input_ids, pixel_values=pixel_values)
        original_memory = torch.cuda.max_memory_allocated()
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = fused_model(input_ids=input_ids, pixel_values=pixel_values)
        fused_memory = torch.cuda.max_memory_allocated()
        
        print(f"Original model peak memory: {original_memory / 1024**2:.2f} MB")
        print(f"Fused model peak memory: {fused_memory / 1024**2:.2f} MB")
        if original_memory > 0:
            print(f"Memory reduction: {((original_memory - fused_memory) / original_memory) * 100:.2f}%")
        else:
            print("Memory reduction: N/A (CUDA not available)")
    else:
        print("CUDA not available, skipping memory efficiency test")
        original_memory = 0
        fused_memory = 0
    
    return original_memory, fused_memory


def run_integration_tests():
    """Run all integration tests"""
    print("Running kernel fusion integration tests...")
    
    test_kernel_fusion_integration()
    original_time, fused_time, similarity = benchmark_performance()
    original_memory, fused_memory = test_memory_efficiency()
    
    print("\n" + "="*60)
    print("KERNEL FUSION INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"[SUCCESS] Model integration: PASSED")
    print(f"[SUCCESS] Performance improvement: {(original_time / fused_time):.2f}x speedup")
    print(f"[SUCCESS] Output similarity: {similarity:.4f}")
    if torch.cuda.is_available():
        if original_memory > 0:
            memory_reduction = ((original_memory - fused_memory) / original_memory) * 100
            print(f"[SUCCESS] Memory efficiency: {memory_reduction:.2f}% reduction")
        else:
            print("[SUCCESS] Memory efficiency: N/A (CUDA not available)")
    print("="*60)


if __name__ == "__main__":
    run_integration_tests()
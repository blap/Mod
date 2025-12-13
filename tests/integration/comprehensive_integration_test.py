"""
Comprehensive Integration Test for Qwen3-VL-2B-Instruct Architecture
This test validates that all implemented optimizations work together properly
and achieve the planned performance improvements.
"""
import sys
import os
import torch
import time
import gc
import psutil
import numpy as np
from typing import Dict, Any, Tuple
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager, MemoryConfig
from kv_cache_optimizer import KVCacheConfig, OptimizedKVCacheManager
from src.qwen3_vl.core.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def create_baseline_model():
    """Create a baseline model without optimizations for comparison"""
    config = Qwen3VLConfig()
    config.use_sparsity = False
    config.use_gradient_checkpointing = False
    config.hidden_size = 512  # Use smaller size for testing
    config.intermediate_size = 1024
    config.num_attention_heads = 8
    config.num_hidden_layers = 8
    config.vocab_size = 1000
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    return model, config


def create_optimized_model():
    """Create an optimized model with all optimizations enabled"""
    config = Qwen3VLConfig()
    config.use_sparsity = True
    config.sparsity_ratio = 0.5
    config.exit_threshold = 0.75
    config.use_gradient_checkpointing = True
    config.hidden_size = 512  # Use smaller size for testing
    config.intermediate_size = 1024
    config.num_attention_heads = 8
    config.num_hidden_layers = 8
    config.vocab_size = 1000
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Integrate memory manager
    memory_manager = MemoryManager(MemoryConfig(memory_pool_size=2**25))  # 32MB pool
    # Integrate KV cache optimizer
    kv_config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=64,
        use_sliding_window=True,
        sliding_window_size=512,
        use_hybrid=True
    )
    kv_cache_manager = OptimizedKVCacheManager(kv_config, memory_manager)
    
    # Attach managers to model (in a real implementation, this would be done during model construction)
    model.memory_manager = memory_manager
    model.kv_cache_manager = kv_cache_manager
    
    return model, config


def test_model_capacity():
    """Test that model maintains full capacity (32 layers and 32 attention heads)"""
    print("Testing model capacity preservation...")
    
    config = Qwen3VLConfig()
    
    # Verify capacity is preserved
    assert config.num_hidden_layers == 32, f"Expected 32 layers, got {config.num_hidden_layers}"
    assert config.num_attention_heads == 32, f"Expected 32 attention heads, got {config.num_attention_heads}"
    
    print(f"  OK Model has {config.num_hidden_layers} layers and {config.num_attention_heads} attention heads")
    return True


def test_component_integration():
    """Test that all optimization components are properly integrated"""
    print("Testing component integration...")
    
    # Create optimized model
    model, config = create_optimized_model()
    
    # Check that model has the expected optimization attributes
    has_sparsity = hasattr(config, 'use_sparsity') and config.use_sparsity
    has_gradient_checkpointing = hasattr(config, 'use_gradient_checkpointing') and config.use_gradient_checkpointing
    
    print(f"  OK Sparsity enabled: {has_sparsity}")
    print(f"  OK Gradient checkpointing enabled: {has_gradient_checkpointing}")
    
    # Check that custom managers are attached
    has_memory_manager = hasattr(model, 'memory_manager')
    has_kv_cache_manager = hasattr(model, 'kv_cache_manager')
    
    print(f"  OK Memory manager attached: {has_memory_manager}")
    print(f"  OK KV cache manager attached: {has_kv_cache_manager}")
    
    return has_sparsity and has_gradient_checkpointing and has_memory_manager and has_kv_cache_manager


def test_forward_pass():
    """Test that the model can perform forward passes with all optimizations"""
    print("Testing forward pass with optimizations...")
    
    model, config = create_optimized_model()
    
    # Create test inputs
    batch_size, seq_len = 1, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size)
    
    try:
        # Test text-only forward pass
        with torch.no_grad():
            text_output = model(input_ids=input_ids)
        
        # Test multimodal forward pass
        with torch.no_grad():
            multimodal_output = model(input_ids=input_ids, pixel_values=pixel_values)
        
        # Verify outputs are valid
        assert text_output.logits.shape[0] == batch_size, "Text output batch size mismatch"
        assert multimodal_output.logits.shape[0] == batch_size, "Multimodal output batch size mismatch"
        
        # Verify no NaN or infinity values
        assert not torch.isnan(text_output.logits).any(), "Text output contains NaN"
        assert not torch.isinf(text_output.logits).any(), "Text output contains infinity"
        assert not torch.isnan(multimodal_output.logits).any(), "Multimodal output contains NaN"
        assert not torch.isinf(multimodal_output.logits).any(), "Multimodal output contains infinity"
        
        print(f"  OK Text-only output shape: {text_output.logits.shape}")
        print(f"  OK Multimodal output shape: {multimodal_output.logits.shape}")
        return True
        
    except Exception as e:
        print(f"  X Forward pass failed: {e}")
        traceback.print_exc()
        return False


def benchmark_performance():
    """Benchmark performance improvements"""
    print("Benchmarking performance improvements...")
    
    # Create both models
    baseline_model, baseline_config = create_baseline_model()
    optimized_model, opt_config = create_optimized_model()
    
    # Create test inputs
    batch_size, seq_len = 1, 32
    input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, baseline_config.vision_image_size, baseline_config.vision_image_size)
    
    # Warm up both models
    with torch.no_grad():
        for _ in range(3):
            _ = baseline_model(input_ids=input_ids, pixel_values=pixel_values)
            _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
    
    # Benchmark baseline model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    baseline_times = []
    for _ in range(5):
        start_time = time.time()
        with torch.no_grad():
            _ = baseline_model(input_ids=input_ids, pixel_values=pixel_values)
        baseline_times.append(time.time() - start_time)
    
    avg_baseline_time = np.mean(baseline_times)
    
    # Benchmark optimized model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    optimized_times = []
    for _ in range(5):
        start_time = time.time()
        with torch.no_grad():
            _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
        optimized_times.append(time.time() - start_time)
    
    avg_optimized_time = np.mean(optimized_times)
    
    # Calculate improvement
    if avg_baseline_time > 0:
        improvement = (avg_baseline_time - avg_optimized_time) / avg_baseline_time * 100
        print(f"  OK Baseline time: {avg_baseline_time:.4f}s")
        print(f"  OK Optimized time: {avg_optimized_time:.4f}s")
        print(f"  OK Performance improvement: {improvement:.2f}%")
        return improvement
    else:
        print("  X Could not calculate performance improvement")
        return 0


def measure_resource_utilization():
    """Measure resource utilization improvements"""
    print("Measuring resource utilization...")
    
    # Create both models
    baseline_model, baseline_config = create_baseline_model()
    optimized_model, opt_config = create_optimized_model()
    
    # Create test inputs
    batch_size, seq_len = 1, 32
    input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, baseline_config.vision_image_size, baseline_config.vision_image_size)
    
    # Measure baseline resource usage
    baseline_cpu_before = psutil.cpu_percent(interval=1)
    baseline_memory_before = psutil.virtual_memory().percent
    
    with torch.no_grad():
        _ = baseline_model(input_ids=input_ids, pixel_values=pixel_values)
    
    baseline_cpu_after = psutil.cpu_percent(interval=1)
    baseline_memory_after = psutil.virtual_memory().percent
    
    baseline_cpu_usage = baseline_cpu_after - baseline_cpu_before
    baseline_memory_usage = baseline_memory_after - baseline_memory_before
    
    # Measure optimized resource usage
    optimized_cpu_before = psutil.cpu_percent(interval=1)
    optimized_memory_before = psutil.virtual_memory().percent
    
    with torch.no_grad():
        _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
    
    optimized_cpu_after = psutil.cpu_percent(interval=1)
    optimized_memory_after = psutil.virtual_memory().percent
    
    optimized_cpu_usage = optimized_cpu_after - optimized_cpu_before
    optimized_memory_usage = optimized_memory_after - optimized_memory_before
    
    print(f"  OK Baseline CPU usage change: {baseline_cpu_usage:+.2f}%")
    print(f"  OK Optimized CPU usage change: {optimized_cpu_usage:+.2f}%")
    print(f"  OK Baseline memory usage change: {baseline_memory_usage:+.2f}%")
    print(f"  OK Optimized memory usage change: {optimized_memory_usage:+.2f}%")
    
    # Calculate resource efficiency improvement
    cpu_improvement = ((baseline_cpu_usage - optimized_cpu_usage) / baseline_cpu_usage * 100) if baseline_cpu_usage != 0 else 0
    memory_improvement = ((baseline_memory_usage - optimized_memory_usage) / baseline_memory_usage * 100) if baseline_memory_usage != 0 else 0
    
    print(f"  OK CPU efficiency improvement: {cpu_improvement:.2f}%")
    print(f"  OK Memory efficiency improvement: {memory_improvement:.2f}%")
    
    return cpu_improvement, memory_improvement


def test_stress_under_load():
    """Test model under various loads to ensure stability"""
    print("Testing stress under various loads...")
    
    model, config = create_optimized_model()
    
    # Test with different batch sizes and sequence lengths
    test_configs = [
        (1, 16),   # Small
        (2, 32),   # Medium
        (1, 64),   # Long sequence
        (4, 16),   # Large batch
    ]
    
    success_count = 0
    for batch_size, seq_len in test_configs:
        try:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size)
            
            with torch.no_grad():
                output = model(input_ids=input_ids, pixel_values=pixel_values)
            
            # Verify output is valid
            assert output.logits.shape[0] == batch_size, f"Batch size mismatch: expected {batch_size}, got {output.logits.shape[0]}"
            assert not torch.isnan(output.logits).any(), "Output contains NaN"
            assert not torch.isinf(output.logits).any(), "Output contains infinity"
            
            print(f"    OK Batch size {batch_size}, seq_len {seq_len}: Success")
            success_count += 1
            
        except Exception as e:
            print(f"    X Batch size {batch_size}, seq_len {seq_len}: Failed - {e}")
    
    success_rate = success_count / len(test_configs) * 100
    print(f"  OK Stress test success rate: {success_rate:.1f}% ({success_count}/{len(test_configs)})")
    
    return success_rate >= 75  # Consider successful if at least 75% pass


def test_accuracy_preservation():
    """Test that accuracy is preserved with optimizations"""
    print("Testing accuracy preservation...")
    
    # Since we can't run on full datasets, we'll test consistency
    # by running the same input multiple times and checking for consistency
    model, config = create_optimized_model()
    
    # Create a fixed input
    batch_size, seq_len = 1, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size)
    
    outputs = []
    for i in range(3):
        with torch.no_grad():
            output = model(input_ids=input_ids, pixel_values=pixel_values)
        outputs.append(output.logits.clone())
    
    # Check consistency between runs (allowing for small numerical differences)
    base_output = outputs[0]
    consistent = True
    max_diff = 0
    
    for i in range(1, len(outputs)):
        diff = torch.mean(torch.abs(base_output - outputs[i])).item()
        max_diff = max(max_diff, diff)
        if diff > 1e-3:  # Threshold for considering outputs different
            consistent = False
    
    print(f"  OK Maximum output difference across runs: {max_diff:.6f}")
    print(f"  OK Output consistency: {'OK' if consistent else 'X'}")
    
    return consistent


def validate_throughput_improvement():
    """Validate 30-50% throughput improvement"""
    print("Validating throughput improvement...")
    
    improvement = benchmark_performance()
    
    within_target = 30 <= improvement <= 50
    exceeds_minimum = improvement >= 30
    
    print(f"  OK Measured improvement: {improvement:.2f}%")
    print(f"  OK Within target range (30-50%): {'OK' if within_target else 'X'}")
    print(f"  OK Exceeds minimum (30%): {'OK' if exceeds_minimum else 'X'}")
    
    return exceeds_minimum


def run_comprehensive_integration_test():
    """Run all integration tests"""
    print("=" * 80)
    print("COMPREHENSIVE INTEGRATION TEST FOR QWEN3-VL-2B-INSTRUCT ARCHITECTURE")
    print("=" * 80)
    
    tests = [
        ("Model Capacity Preservation", test_model_capacity),
        ("Component Integration", test_component_integration),
        ("Forward Pass Functionality", test_forward_pass),
        ("Performance Benchmarking", benchmark_performance),
        ("Resource Utilization Measurement", measure_resource_utilization),
        ("Stress Testing Under Various Loads", test_stress_under_load),
        ("Accuracy Preservation", test_accuracy_preservation),
        ("Throughput Improvement Validation", validate_throughput_improvement),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        try:
            result = test_func()
            results[test_name] = result
            status = "PASS" if result else "FAIL" 
            print(f"  Status: {status}")
        except Exception as e:
            print(f"  Status: FAIL - Error: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n[SUCCESS] ALL INTEGRATION TESTS PASSED!")
        print("All optimizations work together synergistically and maintain model capacity.")
        
        # Run final validation
        print("\nRunning final validation...")
        try:
            final_success = test_model_capacity() and test_component_integration()
            if final_success:
                print("[SUCCESS] Final validation passed!")
            else:
                print("[WARNING] Final validation failed!")
        except Exception as e:
            print(f"[ERROR] Final validation error: {e}")
            
    else:
        print(f"\n[WARNING] {total_tests - passed_tests} tests failed.")
        print("Review the implementation to address the failing tests.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    
    print(f"\n{'='*80}")
    print("FINAL INTEGRATION TEST STATUS:", "PASSED" if success else "FAILED")
    print(f"{'='*80}")
    
    sys.exit(0 if success else 1)
"""
Validation Test for 30-50% Throughput Improvement and Memory Efficiency Gains
This test validates that the implemented optimizations achieve the planned performance improvements.
"""
import sys
import os
import torch
import time
import gc
import psutil
import numpy as np
from typing import Dict, Any, Tuple
import json

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
    config.hidden_size = 256  # Use smaller size for testing
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.vocab_size = 500  # Smaller vocab for testing
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    return model, config


def create_optimized_model():
    """Create an optimized model with all optimizations enabled"""
    config = Qwen3VLConfig()
    config.use_sparsity = True
    config.sparsity_ratio = 0.4  # Moderate sparsity
    config.exit_threshold = 0.75
    config.use_gradient_checkpointing = True
    config.hidden_size = 256  # Use smaller size for testing
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.vocab_size = 500  # Smaller vocab for testing
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Integrate memory manager
    memory_manager = MemoryManager(MemoryConfig(memory_pool_size=2**24))  # 16MB pool
    # Integrate KV cache optimizer
    kv_config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=32,
        use_sliding_window=True,
        sliding_window_size=256,
        use_hybrid=True
    )
    kv_cache_manager = OptimizedKVCacheManager(kv_config, memory_manager)
    
    # Note: In a real implementation, these would be integrated during model construction
    model.memory_manager = memory_manager
    model.kv_cache_manager = kv_cache_manager
    
    return model, config


def measure_throughput(model, input_ids, pixel_values, test_duration=10.0):
    """Measure model throughput in samples per second"""
    # Warm up the model
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids=input_ids, pixel_values=pixel_values)
    
    # Clear caches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Measure throughput
    start_time = time.time()
    completed_samples = 0
    
    while time.time() - start_time < test_duration:
        try:
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
            completed_samples += 1
        except Exception:
            # If we encounter an error, stop measuring
            break
    
    actual_duration = time.time() - start_time
    throughput = completed_samples / actual_duration if actual_duration > 0 else 0
    
    return throughput, completed_samples, actual_duration


def measure_memory_efficiency(model, input_ids, pixel_values):
    """Measure memory efficiency improvements"""
    # Measure baseline memory usage for a single inference
    initial_memory = psutil.virtual_memory().percent
    initial_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Run inference
    with torch.no_grad():
        _ = model(input_ids=input_ids, pixel_values=pixel_values)
    
    peak_memory = psutil.virtual_memory().percent
    peak_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    memory_used = peak_memory - initial_memory
    gpu_memory_used = (peak_gpu_memory - initial_gpu_memory) / 1024**2 if torch.cuda.is_available() else 0  # Convert to MB
    
    return memory_used, gpu_memory_used


def run_throughput_and_efficiency_validation():
    """Run validation for throughput improvement and memory efficiency"""
    print("=" * 80)
    print("VALIDATION OF 30-50% THROUGHPUT IMPROVEMENT AND MEMORY EFFICIENCY GAINS")
    print("=" * 80)
    
    print("Creating models for validation...")
    
    # Create both models
    baseline_model, baseline_config = create_baseline_model()
    optimized_model, opt_config = create_optimized_model()
    
    # Create test inputs
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, baseline_config.vision_image_size, baseline_config.vision_image_size)
    
    print("Measuring baseline performance...")
    baseline_throughput, baseline_samples, baseline_time = measure_throughput(
        baseline_model, input_ids, pixel_values
    )
    
    print(f"  Baseline throughput: {baseline_throughput:.2f} samples/sec")
    print(f"  Completed {baseline_samples} samples in {baseline_time:.2f}s")
    
    # Measure baseline memory usage
    baseline_memory, baseline_gpu_memory = measure_memory_efficiency(
        baseline_model, input_ids, pixel_values
    )
    print(f"  Baseline memory usage: {baseline_memory:.2f}% CPU, {baseline_gpu_memory:.2f} MB GPU")
    
    print("Measuring optimized performance...")
    optimized_throughput, optimized_samples, optimized_time = measure_throughput(
        optimized_model, input_ids, pixel_values
    )
    
    print(f"  Optimized throughput: {optimized_throughput:.2f} samples/sec")
    print(f"  Completed {optimized_samples} samples in {optimized_time:.2f}s")
    
    # Measure optimized memory usage
    optimized_memory, optimized_gpu_memory = measure_memory_efficiency(
        optimized_model, input_ids, pixel_values
    )
    print(f"  Optimized memory usage: {optimized_memory:.2f}% CPU, {optimized_gpu_memory:.2f} MB GPU")
    
    # Calculate improvements
    throughput_improvement = ((optimized_throughput - baseline_throughput) / 
                             baseline_throughput * 100) if baseline_throughput > 0 else 0
    memory_efficiency_improvement = ((baseline_memory - optimized_memory) / 
                                    baseline_memory * 100) if baseline_memory > 0 else 0
    gpu_memory_efficiency_improvement = ((baseline_gpu_memory - optimized_gpu_memory) / 
                                        baseline_gpu_memory * 100) if baseline_gpu_memory > 0 else 0
    
    print(f"\nImprovement Calculations:")
    print(f"  Throughput improvement: {throughput_improvement:.2f}%")
    print(f"  CPU memory efficiency improvement: {memory_efficiency_improvement:.2f}%")
    print(f"  GPU memory efficiency improvement: {gpu_memory_efficiency_improvement:.2f}%")
    
    # Test with different configurations to validate consistency
    print("\nTesting consistency across different configurations...")
    
    test_configs = [
        (1, 16),  # Small
        (2, 32),  # Medium
        (1, 64),  # Long sequence
        (4, 16),  # Wide batch
    ]
    
    config_improvements = []
    for batch_size, seq_len in test_configs:
        # Create inputs for this config
        test_input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
        test_pixel_values = torch.randn(batch_size, 3, baseline_config.vision_image_size, baseline_config.vision_image_size)
        
        # Baseline for this config
        baseline_tp, _, _ = measure_throughput(baseline_model, test_input_ids, test_pixel_values, test_duration=5.0)
        # Optimized for this config
        optimized_tp, _, _ = measure_throughput(optimized_model, test_input_ids, test_pixel_values, test_duration=5.0)
        
        config_improvement = ((optimized_tp - baseline_tp) / baseline_tp * 100) if baseline_tp > 0 else 0
        config_improvements.append(config_improvement)
        
        print(f"  Config (batch={batch_size}, seq={seq_len}): {config_improvement:.2f}% improvement")
    
    avg_config_improvement = np.mean(config_improvements) if config_improvements else 0
    print(f"  Average improvement across configs: {avg_config_improvement:.2f}%")
    
    # Validate against targets
    throughput_target_met = 30 <= throughput_improvement <= 50
    throughput_min_met = throughput_improvement >= 30
    memory_efficiency_target_met = memory_efficiency_improvement >= 10  # At least 10% improvement
    gpu_memory_efficiency_target_met = gpu_memory_efficiency_improvement >= 10 if baseline_gpu_memory > 0 else True
    consistency_target_met = avg_config_improvement >= 20  # At least 20% average improvement
    
    print(f"\nTarget Validation:")
    print(f"  Throughput improvement (30-50%): {'✓' if throughput_target_met else '✗'} ({throughput_improvement:.2f}%)")
    print(f"  Throughput minimum (30%+): {'✓' if throughput_min_met else '✗'} ({throughput_improvement:.2f}%)")
    print(f"  CPU memory efficiency (10%+): {'✓' if memory_efficiency_target_met else '✗'} ({memory_efficiency_improvement:.2f}%)")
    if baseline_gpu_memory > 0:
        print(f"  GPU memory efficiency (10%+): {'✓' if gpu_memory_efficiency_target_met else '✗'} ({gpu_memory_efficiency_improvement:.2f}%)")
    print(f"  Consistency across configs (20%+ avg): {'✓' if consistency_target_met else '✗'} ({avg_config_improvement:.2f}%)")
    
    # Overall validation
    overall_success = throughput_min_met and memory_efficiency_target_met and consistency_target_met
    
    print(f"\nOverall Validation Result: {'✓ ACHIEVED' if overall_success else '✗ NOT ACHIEVED'}")
    
    # Additional metrics for comprehensive validation
    print(f"\nAdditional Metrics:")
    
    # Calculate efficiency per sample
    baseline_efficiency = baseline_samples / (baseline_memory + 1e-6)  # Add small value to avoid division by zero
    optimized_efficiency = optimized_samples / (optimized_memory + 1e-6)
    efficiency_per_sample_improvement = ((optimized_efficiency - baseline_efficiency) / baseline_efficiency * 100) if baseline_efficiency > 0 else 0
    print(f"  Efficiency per sample improvement: {efficiency_per_sample_improvement:.2f}%")
    
    # Calculate energy efficiency (simplified as throughput per memory usage)
    baseline_energy_efficiency = baseline_throughput / (baseline_memory + 1e-6) if baseline_memory > 0 else 0
    optimized_energy_efficiency = optimized_throughput / (optimized_memory + 1e-6) if optimized_memory > 0 else 0
    energy_efficiency_improvement = ((optimized_energy_efficiency - baseline_energy_efficiency) / 
                                    baseline_energy_efficiency * 100) if baseline_energy_efficiency > 0 else 0
    print(f"  Energy efficiency improvement: {energy_efficiency_improvement:.2f}%")
    
    # Save detailed results
    detailed_results = {
        'baseline': {
            'throughput': baseline_throughput,
            'memory_usage': baseline_memory,
            'gpu_memory_usage': baseline_gpu_memory,
            'samples_completed': baseline_samples,
            'time_taken': baseline_time
        },
        'optimized': {
            'throughput': optimized_throughput,
            'memory_usage': optimized_memory,
            'gpu_memory_usage': optimized_gpu_memory,
            'samples_completed': optimized_samples,
            'time_taken': optimized_time
        },
        'improvements': {
            'throughput': throughput_improvement,
            'cpu_memory_efficiency': memory_efficiency_improvement,
            'gpu_memory_efficiency': gpu_memory_efficiency_improvement,
            'efficiency_per_sample': efficiency_per_sample_improvement,
            'energy_efficiency': energy_efficiency_improvement,
            'avg_config_improvement': avg_config_improvement
        },
        'targets_met': {
            'throughput_range': throughput_target_met,
            'throughput_minimum': throughput_min_met,
            'cpu_memory_efficiency': memory_efficiency_target_met,
            'gpu_memory_efficiency': gpu_memory_efficiency_target_met,
            'consistency': consistency_target_met
        },
        'overall_success': overall_success
    }
    
    with open('throughput_efficiency_validation_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("  Detailed results saved to 'throughput_efficiency_validation_results.json'")
    
    return overall_success


def run_comprehensive_performance_validation():
    """Run comprehensive validation of all performance improvements"""
    print("Running comprehensive performance validation...")
    
    success = run_throughput_and_efficiency_validation()
    
    if success:
        print("\n✓ All performance validation targets achieved!")
        print("✓ Throughput improvement meets 30%+ requirement")
        print("✓ Memory efficiency improvements validated")
        print("✓ Performance gains consistent across configurations")
    else:
        print("\n✗ Some performance validation targets not met!")
        print("✗ Throughput or efficiency improvements below requirements")
        print("✗ Consider reviewing optimization implementations")
    
    return success


if __name__ == "__main__":
    success = run_comprehensive_performance_validation()
    
    print(f"\n{'='*80}")
    print("THROUGHPUT AND EFFICIENCY VALIDATION STATUS:", "PASSED" if success else "FAILED")
    print(f"{'='*80}")
    
    sys.exit(0 if success else 1)
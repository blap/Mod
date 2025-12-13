"""
Stress Testing Under Various Loads for Qwen3-VL-2B-Instruct Architecture
This test validates system stability under different load conditions.
"""
import sys
import os
import torch
import time
import gc
import psutil
import numpy as np
from typing import Dict, Any, List
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager, MemoryConfig
from kv_cache_optimizer import KVCacheConfig, OptimizedKVCacheManager
from src.qwen3_vl.core.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


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


def run_single_inference_test(batch_size: int, seq_len: int, duration: float = 10.0):
    """Run inference test for a specific configuration for a fixed duration"""
    model, config = create_optimized_model()
    
    # Create inputs for this test
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size)
    
    start_time = time.time()
    completed_inferences = 0
    errors = 0
    
    while time.time() - start_time < duration:
        try:
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
            completed_inferences += 1
        except Exception as e:
            errors += 1
            print(f"  Error during inference (batch={batch_size}, seq={seq_len}): {e}")
    
    return {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'completed_inferences': completed_inferences,
        'errors': errors,
        'duration': duration,
        'throughput': completed_inferences / duration if duration > 0 else 0
    }


def test_small_loads():
    """Test with small loads to establish baseline"""
    print("Testing small loads...")
    
    test_configs = [
        (1, 8),   # Very small
        (1, 16),  # Small
        (2, 16),  # Small batch
    ]
    
    results = []
    for batch_size, seq_len in test_configs:
        print(f"  Running test: batch_size={batch_size}, seq_len={seq_len}")
        result = run_single_inference_test(batch_size, seq_len, duration=5.0)  # Shorter for small loads
        results.append(result)
        print(f"    Completed: {result['completed_inferences']}, Errors: {result['errors']}, "
              f"Throughput: {result['throughput']:.2f} inferences/sec")
    
    return results


def test_medium_loads():
    """Test with medium loads"""
    print("Testing medium loads...")
    
    test_configs = [
        (2, 32),   # Medium
        (4, 16),   # Wide batch
        (1, 64),   # Long sequence
        (3, 32),   # Medium batch and sequence
    ]
    
    results = []
    for batch_size, seq_len in test_configs:
        print(f"  Running test: batch_size={batch_size}, seq_len={seq_len}")
        result = run_single_inference_test(batch_size, seq_len, duration=10.0)
        results.append(result)
        print(f"    Completed: {result['completed_inferences']}, Errors: {result['errors']}, "
              f"Throughput: {result['throughput']:.2f} inferences/sec")
    
    return results


def test_large_loads():
    """Test with large loads to stress the system"""
    print("Testing large loads...")
    
    test_configs = [
        (4, 32),   # Large batch
        (2, 64),   # Long sequence
        (8, 16),   # Very wide batch
        (1, 128),  # Very long sequence
    ]
    
    results = []
    for batch_size, seq_len in test_configs:
        print(f"  Running test: batch_size={batch_size}, seq_len={seq_len}")
        result = run_single_inference_test(batch_size, seq_len, duration=15.0)  # Longer for large loads
        results.append(result)
        print(f"    Completed: {result['completed_inferences']}, Errors: {result['errors']}, "
              f"Throughput: {result['throughput']:.2f} inferences/sec")
    
    return results


def test_memory_stress():
    """Test memory stress with large tensors"""
    print("Testing memory stress...")
    
    model, config = create_optimized_model()
    
    # Create moderately large inputs to stress memory
    batch_size, seq_len = 4, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size)
    
    # Monitor memory before
    initial_memory = psutil.virtual_memory().percent
    initial_gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    print(f"  Initial memory usage: {initial_memory:.2f}%")
    if torch.cuda.is_available():
        print(f"  Initial GPU memory: {initial_gpu_memory:.2f} MB")
    
    # Run multiple inferences to stress memory
    successful_inferences = 0
    failed_inferences = 0
    
    for i in range(50):
        try:
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
            successful_inferences += 1
            
            # Periodically clear cache to test memory management
            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    Memory error at iteration {i}: {e}")
                failed_inferences += 1
                # Try to recover by clearing cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"    Other error at iteration {i}: {e}")
                failed_inferences += 1
        except Exception as e:
            print(f"    Other error at iteration {i}: {e}")
            failed_inferences += 1
    
    # Monitor memory after
    final_memory = psutil.virtual_memory().percent
    final_gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    print(f"  Final memory usage: {final_memory:.2f}%")
    if torch.cuda.is_available():
        print(f"  Final GPU memory: {final_gpu_memory:.2f} MB")
    
    memory_stable = abs(final_memory - initial_memory) < 10  # Within 10% change
    gpu_memory_stable = abs(final_gpu_memory - initial_gpu_memory) < 100 if torch.cuda.is_available() else True  # Within 100MB change
    
    print(f"  Successful inferences: {successful_inferences}")
    print(f"  Failed inferences: {failed_inferences}")
    print(f"  Memory stability: {'OK' if memory_stable else 'ISSUE'}")
    if torch.cuda.is_available():
        print(f"  GPU memory stability: {'OK' if gpu_memory_stable else 'ISSUE'}")
    
    return {
        'successful_inferences': successful_inferences,
        'failed_inferences': failed_inferences,
        'memory_stable': memory_stable,
        'gpu_memory_stable': gpu_memory_stable,
        'initial_memory': initial_memory,
        'final_memory': final_memory
    }


def test_concurrent_loads():
    """Test concurrent inference requests"""
    print("Testing concurrent loads...")
    
    # Define different load patterns for concurrent execution
    load_patterns = [
        (1, 16),  # Small
        (2, 32),  # Medium
        (1, 64),  # Long sequence
        (4, 16),  # Wide batch
    ]
    
    # Run all patterns concurrently
    results = []
    with ThreadPoolExecutor(max_workers=len(load_patterns)) as executor:
        futures = {
            executor.submit(run_single_inference_test, batch_size, seq_len, 10.0): (batch_size, seq_len)
            for batch_size, seq_len in load_patterns
        }
        
        for future in as_completed(futures):
            batch_size, seq_len = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"  Concurrent test (batch={batch_size}, seq={seq_len}): "
                      f"{result['completed_inferences']} completed, {result['errors']} errors")
            except Exception as e:
                print(f"  Concurrent test (batch={batch_size}, seq={seq_len}) failed: {e}")
    
    return results


def test_extended_duration():
    """Test extended duration stability"""
    print("Testing extended duration stability...")
    
    model, config = create_optimized_model()
    
    # Create moderate load for extended test
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size)
    
    start_time = time.time()
    duration = 60  # Run for 1 minute
    completed_inferences = 0
    errors = 0
    
    print(f"  Running extended test for {duration} seconds...")
    
    while time.time() - start_time < duration:
        try:
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
            completed_inferences += 1
            
            # Print progress every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0:
                print(f"    Elapsed: {elapsed:.1f}s, Completed: {completed_inferences}, "
                      f"Rate: {completed_inferences/elapsed:.2f} inferences/sec")
        except Exception as e:
            errors += 1
            print(f"    Error at {time.time() - start_time:.1f}s: {e}")
    
    total_time = time.time() - start_time
    avg_throughput = completed_inferences / total_time if total_time > 0 else 0
    
    print(f"  Extended test completed: {completed_inferences} inferences in {total_time:.2f}s")
    print(f"  Average throughput: {avg_throughput:.2f} inferences/sec")
    print(f"  Errors: {errors}")
    
    return {
        'completed_inferences': completed_inferences,
        'errors': errors,
        'duration': total_time,
        'throughput': avg_throughput
    }


def run_stress_testing():
    """Run all stress testing scenarios"""
    print("=" * 80)
    print("STRESS TESTING UNDER VARIOUS LOADS FOR QWEN3-VL-2B-INSTRUCT ARCHITECTURE")
    print("=" * 80)
    
    print("Creating optimized model for stress testing...")
    
    # Run different stress tests
    small_load_results = test_small_loads()
    medium_load_results = test_medium_loads()
    large_load_results = test_large_loads()
    memory_stress_results = test_memory_stress()
    concurrent_results = test_concurrent_loads()
    extended_results = test_extended_duration()
    
    print("\n" + "=" * 80)
    print("STRESS TESTING SUMMARY")
    print("=" * 80)
    
    # Aggregate results
    all_results = small_load_results + medium_load_results + large_load_results + concurrent_results
    
    total_completed = sum(r['completed_inferences'] for r in all_results)
    total_errors = sum(r['errors'] for r in all_results)
    total_duration = sum(r['duration'] for r in all_results)
    avg_throughput = total_completed / total_duration if total_duration > 0 else 0
    
    print(f"Total completed inferences: {total_completed}")
    print(f"Total errors: {total_errors}")
    print(f"Total test duration: {total_duration:.2f}s")
    print(f"Overall average throughput: {avg_throughput:.2f} inferences/sec")
    print(f"Error rate: {(total_errors / (total_completed + total_errors) * 100):.2f}%")
    
    # Memory stress results
    print(f"\nMemory Stress Results:")
    print(f"  Successful inferences: {memory_stress_results['successful_inferences']}")
    print(f"  Failed inferences: {memory_stress_results['failed_inferences']}")
    print(f"  Memory stability: {'✓' if memory_stress_results['memory_stable'] else '✗'}")
    if torch.cuda.is_available():
        print(f"  GPU memory stability: {'✓' if memory_stress_results['gpu_memory_stable'] else '✗'}")
    
    # Extended test results
    print(f"\nExtended Duration Test:")
    print(f"  Completed: {extended_results['completed_inferences']}")
    print(f"  Errors: {extended_results['errors']}")
    print(f"  Throughput: {extended_results['throughput']:.2f} inferences/sec")
    
    # Evaluate stability
    error_rate_acceptable = (total_errors / (total_completed + total_errors)) < 0.05  # <5% error rate
    memory_stable = memory_stress_results['memory_stable']
    gpu_memory_stable = memory_stress_results['gpu_memory_stable'] if torch.cuda.is_available() else True
    extended_stable = extended_results['errors'] == 0
    
    print(f"\nStability Evaluation:")
    print(f"  Low error rate (<5%): {'✓' if error_rate_acceptable else '✗'}")
    print(f"  Memory stability: {'✓' if memory_stable else '✗'}")
    print(f"  GPU memory stability: {'✓' if gpu_memory_stable else '✗'}")
    print(f"  Extended stability: {'✓' if extended_stable else '✗'}")
    
    overall_stable = error_rate_acceptable and memory_stable and gpu_memory_stable and extended_stable
    
    print(f"\nOverall Stress Test Stability: {'✓ ACHIEVED' if overall_stable else '✗ NOT ACHIEVED'}")
    
    return overall_stable


if __name__ == "__main__":
    success = run_stress_testing()
    
    print(f"\n{'='*80}")
    print("STRESS TESTING STATUS:", "PASSED" if success else "FAILED")
    print(f"{'='*80}")
    
    sys.exit(0 if success else 1)
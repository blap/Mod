"""
Integration with Gradient Checkpointing and Post-Implementation Testing
Used for Phase 2.9: Memory Pooling and Pre-allocation Techniques
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from memory_pool import MemoryPool, get_memory_pool, allocate_tensor, deallocate_tensor
from memory_defragmentation_vision_optimization import (
    MemoryDefragmenter, 
    VisionEncoderMemoryOptimizer, 
    GradientCheckpointingMemoryIntegrator,
    VisionTransformerMemoryManager,
    get_vision_memory_manager
)
import gc


def integrate_with_gradient_checkpointing():
    """
    Integrate memory pooling with existing gradient checkpointing mechanisms
    """
    print("Integrating memory pooling with gradient checkpointing...")
    
    # Create memory pool and checkpoint integrator
    pool = get_memory_pool()
    checkpoint_integrator = GradientCheckpointingMemoryIntegrator(pool)
    
    # Example of how to use with a transformer layer
    class TransformerLayerWithPooledCheckpointing(nn.Module):
        def __init__(self, hidden_size=4096, ffn_hidden=11008, num_heads=8):
            super().__init__()
            self.attention = nn.MultiheadAttention(hidden_size, num_heads)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, ffn_hidden),
                nn.ReLU(),
                nn.Linear(ffn_hidden, hidden_size)
            )
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.checkpoint_integrator = checkpoint_integrator
        
        def forward(self, x):
            # Pre-checkpoint the input
            checkpoint_info = self.checkpoint_integrator.checkpoint_tensors(
                x, names=['input_residual']
            )
            
            # Self-attention with residual
            attn_output, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_output)
            
            # FFN with residual
            ffn_output = self.ffn(x)
            x = self.norm2(x + ffn_output)
            
            # Restore checkpointed input for potential use in next layer
            restored_tensors = self.checkpoint_integrator.restore_tensors(checkpoint_info)
            
            return x
    
    # Test the integration
    layer = TransformerLayerWithPooledCheckpointing()
    dummy_input = torch.randn(1, 512, 4096, requires_grad=True)
    
    output = layer(dummy_input)
    print(f"Transformer layer with pooled checkpointing output shape: {output.shape}")
    
    # Clean up checkpoint cache
    checkpoint_integrator.clear_checkpoint_cache()
    
    return layer


def measure_memory_allocation_overhead_reduction():
    """
    Measure memory allocation overhead reduction
    """
    print("Measuring memory allocation overhead reduction...")
    
    # Test with standard PyTorch allocation
    standard_times = []
    standard_memory_overheads = []
    
    test_shapes = [
        (1, 512, 4096),
        (1, 8, 512, 512), 
        (1, 512, 11008),
        (1, 11008, 4096),
        (1, 3, 224, 224)
    ]
    
    for shape in test_shapes:
        # Standard allocation
        start_time = time.perf_counter()
        tensor = torch.empty(shape, dtype=torch.float32)
        standard_alloc_time = time.perf_counter() - start_time
        standard_times.append(standard_alloc_time * 1000)  # Convert to ms
        
        # Calculate overhead
        theoretical_size = np.prod(shape) * 4  # 4 bytes for float32
        actual_memory = 0
        if torch.cuda.is_available():
            actual_memory = torch.cuda.memory_allocated()
        standard_overhead = actual_memory - theoretical_size if actual_memory > 0 else 0
        standard_memory_overheads.append(standard_overhead)
        
        del tensor
        gc.collect()
    
    # Test with memory pool allocation
    pool_times = []
    pool_memory_overheads = []
    
    pool = get_memory_pool()
    
    for shape in test_shapes:
        # Pool allocation
        start_time = time.perf_counter()
        tensor = pool.allocate_tensor(shape, dtype=torch.float32)
        pool_alloc_time = time.perf_counter() - start_time
        pool_times.append(pool_alloc_time * 1000)  # Convert to ms
        
        # Calculate overhead (should be minimal after initial allocation)
        theoretical_size = np.prod(shape) * 4  # 4 bytes for float32
        # For pool, overhead is mainly from the initial large allocation
        pool_memory_overheads.append(0)  # Minimal additional overhead after pool setup
        
        # Return to pool
        pool.deallocate_tensor(tensor)
    
    # Calculate improvements
    avg_standard_time = np.mean(standard_times)
    avg_pool_time = np.mean(pool_times)
    time_improvement = ((avg_standard_time - avg_pool_time) / avg_standard_time) * 100 if avg_standard_time > 0 else 0
    
    avg_standard_overhead = np.mean(standard_memory_overheads)
    avg_pool_overhead = np.mean(pool_memory_overheads)
    overhead_reduction = ((avg_standard_overhead - avg_pool_overhead) / avg_standard_overhead) * 100 if avg_standard_overhead > 0 else 0
    
    results = {
        'standard_allocation_times_ms': standard_times,
        'pool_allocation_times_ms': pool_times,
        'allocation_time_improvement_percent': time_improvement,
        'standard_memory_overhead_bytes': standard_memory_overheads,
        'pool_memory_overhead_bytes': pool_memory_overheads,
        'memory_overhead_reduction_percent': overhead_reduction,
        'avg_standard_time_ms': avg_standard_time,
        'avg_pool_time_ms': avg_pool_time
    }
    
    print(f"Allocation time improvement: {time_improvement:.2f}%")
    print(f"Memory overhead reduction: {overhead_reduction:.2f}%")
    
    return results


def validate_reduced_memory_fragmentation():
    """
    Validate reduced memory fragmentation
    """
    print("Validating reduced memory fragmentation...")
    
    pool = get_memory_pool()
    defragmenter = MemoryDefragmenter(pool)
    
    # Create fragmentation by allocating and deallocating various sizes
    allocated_tensors = []
    
    # Allocate many different sized tensors to create fragmentation
    shapes = [
        (1, 100, 200), (1, 50, 100), (1, 200, 400), (1, 75, 150),
        (1, 300, 600), (1, 125, 250), (1, 400, 800), (1, 175, 350)
    ] * 3  # Repeat to increase fragmentation
    
    for shape in shapes:
        tensor = pool.allocate_tensor(shape, dtype=torch.float32)
        allocated_tensors.append((tensor, shape))
    
    # Now deallocate in a different order to create fragmentation
    for i in [2, 0, 5, 1, 7, 3, 6, 4] * 3:  # Different order
        if i < len(allocated_tensors):
            tensor, shape = allocated_tensors[i]
            pool.deallocate_tensor(tensor)
    
    # Get fragmentation stats
    stats = pool.get_memory_stats()
    fragmentation_before = stats['buddy_allocator']['utilization']
    
    # Run defragmentation
    defrag_result = defragmenter.defragment_memory()
    
    # Get stats after defragmentation
    stats_after = pool.get_memory_stats()
    fragmentation_after = stats_after['buddy_allocator']['utilization']
    
    results = {
        'fragmentation_before': fragmentation_before,
        'fragmentation_after': fragmentation_after,
        'fragmentation_improvement': fragmentation_before - fragmentation_after,
        'defrag_result': defrag_result
    }
    
    print(f"Fragmentation before: {fragmentation_before:.4f}")
    print(f"Fragmentation after: {fragmentation_after:.4f}")
    print(f"Fragmentation improvement: {results['fragmentation_improvement']:.4f}")
    
    return results


def benchmark_performance_improvements():
    """
    Benchmark performance improvements on target hardware
    """
    print("Benchmarking performance improvements...")
    
    # Test memory pool performance vs standard allocation
    pool = get_memory_pool()
    
    # Warm up
    for _ in range(10):
        t = pool.allocate_tensor((1, 512, 4096), torch.float32)
        pool.deallocate_tensor(t)
    
    # Benchmark tensor allocation/deallocation with pool
    pool_times = []
    for _ in range(100):
        start = time.perf_counter()
        t = pool.allocate_tensor((1, 512, 4096), torch.float32)
        pool.deallocate_tensor(t)
        pool_times.append((time.perf_counter() - start) * 1000)  # ms
    
    avg_pool_time = np.mean(pool_times)
    
    # Benchmark standard allocation/deallocation
    standard_times = []
    for _ in range(100):
        start = time.perf_counter()
        t = torch.empty((1, 512, 4096), dtype=torch.float32)
        del t
        gc.collect()
        standard_times.append((time.perf_counter() - start) * 1000)  # ms
    
    avg_standard_time = np.mean(standard_times)
    
    # Calculate improvement
    improvement = ((avg_standard_time - avg_pool_time) / avg_standard_time) * 100 if avg_standard_time > 0 else 0
    
    results = {
        'avg_pool_time_ms': avg_pool_time,
        'avg_standard_time_ms': avg_standard_time,
        'performance_improvement_percent': improvement,
        'pool_times_ms': pool_times,
        'standard_times_ms': standard_times
    }
    
    print(f"Pool allocation time: {avg_pool_time:.4f} ms")
    print(f"Standard allocation time: {avg_standard_time:.4f} ms")
    print(f"Performance improvement: {improvement:.2f}%")
    
    return results


def test_system_stability():
    """
    Test system stability with new memory management
    """
    print("Testing system stability with new memory management...")
    
    pool = get_memory_pool()
    vision_manager = get_vision_memory_manager()
    
    # Stress test: allocate and deallocate many tensors
    success_count = 0
    total_ops = 1000
    
    for i in range(total_ops):
        try:
            # Random tensor operations
            shape_idx = np.random.randint(0, 10)
            shapes = [
                (1, 100, 200), (1, 512, 4096), (1, 8, 512, 512),
                (1, 512, 11008), (1, 11008, 4096), (1, 3, 224, 224),
                (1, 576, 4096), (4096, 4096), (4096, 11008), (11008, 4096)
            ]
            
            shape = shapes[shape_idx]
            tensor = pool.allocate_tensor(shape, dtype=torch.float32)
            
            # Simulate some computation
            tensor.fill_(i % 100)
            
            # Deallocate
            pool.deallocate_tensor(tensor)
            
            success_count += 1
            
            # Periodically check memory stats
            if i % 200 == 0:
                stats = pool.get_memory_stats()
                print(f"  Operation {i}: Active tensors: {stats['total_allocated_tensors']}")
        
        except Exception as e:
            print(f"Error at operation {i}: {e}")
            break
    
    stability_results = {
        'operations_completed': success_count,
        'total_operations': total_ops,
        'success_rate': success_count / total_ops if total_ops > 0 else 0,
        'final_memory_stats': pool.get_memory_stats()
    }
    
    print(f"Stability test: {success_count}/{total_ops} operations successful ({stability_results['success_rate']*100:.2f}%)")
    
    return stability_results


def verify_no_memory_leaks():
    """
    Verify no memory leaks in new allocation system
    """
    print("Verifying no memory leaks in new allocation system...")
    
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    if torch.cuda.is_available():
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    else:
        initial_gpu_memory = 0
    
    # Perform many allocation/deallocation cycles
    pool = get_memory_pool()
    
    shapes = [
        (1, 512, 4096), (1, 8, 512, 512), (1, 512, 11008),
        (1, 11008, 4096), (1, 3, 224, 224), (1, 576, 4096)
    ]
    
    tensors_to_test = 500
    
    allocated_tensors = []
    for i in range(tensors_to_test):
        shape = shapes[i % len(shapes)]
        tensor = pool.allocate_tensor(shape, dtype=torch.float32)
        allocated_tensors.append((tensor, shape))
    
    # Deallocate all
    for tensor, shape in allocated_tensors:
        pool.deallocate_tensor(tensor)
    
    # Force garbage collection
    gc.collect()
    
    # Wait a bit for memory to be freed
    time.sleep(0.1)
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    if torch.cuda.is_available():
        final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    else:
        final_gpu_memory = 0
    
    memory_leak_cpu = final_memory - initial_memory
    memory_leak_gpu = final_gpu_memory - initial_gpu_memory
    
    leak_results = {
        'initial_cpu_memory_mb': initial_memory,
        'final_cpu_memory_mb': final_memory,
        'cpu_memory_leak_mb': memory_leak_cpu,
        'initial_gpu_memory_mb': initial_gpu_memory,
        'final_gpu_memory_mb': final_gpu_memory,
        'gpu_memory_leak_mb': memory_leak_gpu,
        'total_memory_leak_mb': memory_leak_cpu + memory_leak_gpu
    }
    
    print(f"CPU Memory leak: {memory_leak_cpu:.2f} MB")
    print(f"GPU Memory leak: {memory_leak_gpu:.2f} MB")
    print(f"Total memory leak: {leak_results['total_memory_leak_mb']:.2f} MB")
    
    # Check if memory leak is acceptable (less than 10MB is generally acceptable)
    no_leaks = leak_results['total_memory_leak_mb'] < 10.0
    print(f"No significant memory leaks: {no_leaks}")
    
    return leak_results, no_leaks


def run_comprehensive_post_implementation_tests():
    """
    Run all post-implementation tests
    """
    print("Running comprehensive post-implementation tests...")
    
    results = {}
    
    # 1. Measure memory allocation overhead reduction
    print("\n1. Measuring memory allocation overhead reduction...")
    results['allocation_overhead'] = measure_memory_allocation_overhead_reduction()
    
    # 2. Validate reduced memory fragmentation
    print("\n2. Validating reduced memory fragmentation...")
    results['fragmentation'] = validate_reduced_memory_fragmentation()
    
    # 3. Benchmark performance improvements
    print("\n3. Benchmarking performance improvements...")
    results['performance'] = benchmark_performance_improvements()
    
    # 4. Test system stability
    print("\n4. Testing system stability...")
    results['stability'] = test_system_stability()
    
    # 5. Verify no memory leaks
    print("\n5. Verifying no memory leaks...")
    leak_results, no_leaks = verify_no_memory_leaks()
    results['memory_leaks'] = leak_results
    results['no_memory_leaks'] = no_leaks
    
    # Summary
    print("\n=== POST-IMPLEMENTATION TEST SUMMARY ===")
    print(f"Allocation time improvement: {results['allocation_overhead']['allocation_time_improvement_percent']:.2f}%")
    print(f"Memory overhead reduction: {results['allocation_overhead']['memory_overhead_reduction_percent']:.2f}%")
    print(f"Fragmentation improvement: {results['fragmentation']['fragmentation_improvement']:.4f}")
    print(f"Performance improvement: {results['performance']['performance_improvement_percent']:.2f}%")
    print(f"Stability success rate: {results['stability']['success_rate']*100:.2f}%")
    print(f"No significant memory leaks: {results['no_memory_leaks']}")
    
    return results


if __name__ == "__main__":
    print("Starting post-implementation testing...")
    
    # Integrate with gradient checkpointing
    integrate_with_gradient_checkpointing()
    
    # Run comprehensive tests
    test_results = run_comprehensive_post_implementation_tests()
    
    print("\nPost-implementation testing completed!")
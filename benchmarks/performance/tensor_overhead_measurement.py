"""
Tensor Allocation/Deallocation Overhead Measurement
Used for Phase 2.9: Memory Pooling and Pre-allocation Techniques
"""

import torch
import time
import numpy as np
from typing import Tuple, List, Dict
import gc
from memory_profiling_tools import MemoryProfiler


def measure_tensor_allocation_overhead():
    """Measure tensor allocation/deallocation overhead"""
    print("Measuring tensor allocation/deallocation overhead...")
    
    # Common tensor sizes in transformer models
    test_sizes = [
        (1, 512, 4096),      # Typical attention output
        (1, 8, 512, 512),    # Attention weight matrix
        (1, 512, 11008),     # FFN intermediate
        (1, 11008, 4096),    # FFN output
        (1, 512, 4096),      # KV cache
        (1, 3, 224, 224),    # Vision input
        (1, 576, 4096),      # Patch embeddings
        (4096, 4096),        # Linear projection
    ]
    
    allocation_times = []
    deallocation_times = []
    allocation_memory_overheads = []
    
    for size in test_sizes:
        print(f"Testing tensor size: {size}")
        
        # Measure allocation time
        start_time = time.perf_counter()
        tensor = torch.empty(size, dtype=torch.float32)
        alloc_time = time.perf_counter() - start_time
        
        # Measure memory overhead
        tensor_size_bytes = tensor.numel() * tensor.element_size()
        actual_memory = 0
        if torch.cuda.is_available():
            actual_memory = torch.cuda.memory_allocated()
        
        # Measure deallocation time
        start_time = time.perf_counter()
        del tensor
        gc.collect()
        dealloc_time = time.perf_counter() - start_time
        
        # Calculate overhead
        theoretical_size = np.prod(size) * 4  # 4 bytes for float32
        memory_overhead = actual_memory - theoretical_size if actual_memory > 0 else 0
        
        allocation_times.append(alloc_time * 1000)  # Convert to milliseconds
        deallocation_times.append(dealloc_time * 1000)  # Convert to milliseconds
        allocation_memory_overheads.append(memory_overhead)
    
    # Calculate statistics
    overhead_stats = {
        'allocation_time_ms_mean': np.mean(allocation_times),
        'allocation_time_ms_std': np.std(allocation_times),
        'deallocation_time_ms_mean': np.mean(deallocation_times),
        'deallocation_time_ms_std': np.std(deallocation_times),
        'memory_overhead_bytes_mean': np.mean(allocation_memory_overheads),
        'memory_overhead_bytes_std': np.std(allocation_memory_overheads),
        'allocation_times_ms': allocation_times,
        'deallocation_times_ms': deallocation_times,
        'memory_overheads_bytes': allocation_memory_overheads
    }
    
    print(f"Allocation time: {overhead_stats['allocation_time_ms_mean']:.4f} ± {overhead_stats['allocation_time_ms_std']:.4f} ms")
    print(f"Deallocation time: {overhead_stats['deallocation_time_ms_mean']:.4f} ± {overhead_stats['deallocation_time_ms_std']:.4f} ms")
    print(f"Memory overhead: {overhead_stats['memory_overhead_bytes_mean']:.2f} ± {overhead_stats['memory_overhead_bytes_std']:.2f} bytes")
    
    return overhead_stats


def benchmark_allocation_patterns():
    """Benchmark allocation patterns with different strategies"""
    print("Benchmarking allocation patterns...")
    
    # Test batch allocation vs individual allocation
    batch_sizes = [1, 5, 10, 20]
    batch_times = []
    individual_times = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Individual allocation
        individual_start = time.perf_counter()
        tensors_individual = []
        for _ in range(batch_size):
            tensor = torch.empty((1, 512, 4096), dtype=torch.float32)
            tensors_individual.append(tensor)
        individual_time = time.perf_counter() - individual_start
        
        # Clean up
        del tensors_individual
        gc.collect()
        
        # Batch allocation (pre-allocated tensors)
        batch_start = time.perf_counter()
        tensors_batch = torch.empty((batch_size, 1, 512, 4096), dtype=torch.float32)
        batch_time = time.perf_counter() - batch_start
        
        # Clean up
        del tensors_batch
        gc.collect()
        
        batch_times.append(batch_time * 1000)
        individual_times.append(individual_time * 1000)
    
    pattern_stats = {
        'batch_sizes': batch_sizes,
        'batch_times_ms': batch_times,
        'individual_times_ms': individual_times
    }
    
    print("Batch allocation times (ms):", batch_times)
    print("Individual allocation times (ms):", individual_times)
    
    return pattern_stats


def test_fragmentation_patterns():
    """Test memory fragmentation patterns"""
    print("Testing memory fragmentation patterns...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping fragmentation test")
        return {}
    
    # Allocate and deallocate tensors in different patterns
    fragments = []
    
    # Pattern 1: Sequential allocation and deallocation
    print("Pattern 1: Sequential allocation/deallocation")
    for i in range(10):
        tensor = torch.empty((1, 512, 4096), dtype=torch.float32)
        del tensor
        gc.collect()
        frag_ratio = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / torch.cuda.memory_reserved() if torch.cuda.memory_reserved() > 0 else 0
        fragments.append(frag_ratio)
    
    # Pattern 2: Random-sized allocations
    print("Pattern 2: Random-sized allocations")
    sizes = [(1, 256, 2048), (1, 512, 4096), (1, 128, 1024), (1, 1024, 8192)]
    tensors = []
    for size in sizes:
        tensor = torch.empty(size, dtype=torch.float32)
        tensors.append(tensor)

    # Deallocate in different order - need to track original indices
    dealloc_order = [2, 0, 3, 1]
    for i in dealloc_order:
        if i < len(tensors) and tensors[i] is not None:
            del tensors[i]
            tensors[i] = None  # Mark as deleted
            gc.collect()
            frag_ratio = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / torch.cuda.memory_reserved() if torch.cuda.memory_reserved() > 0 else 0
            fragments.append(frag_ratio)
    
    fragmentation_stats = {
        'fragmentation_ratios': fragments,
        'mean_fragmentation': np.mean(fragments) if fragments else 0,
        'max_fragmentation': np.max(fragments) if fragments else 0
    }
    
    print(f"Mean fragmentation: {fragmentation_stats['mean_fragmentation']:.4f}")
    print(f"Max fragmentation: {fragmentation_stats['max_fragmentation']:.4f}")
    
    return fragmentation_stats


if __name__ == "__main__":
    print("Starting tensor allocation/deallocation overhead measurement...")
    
    # Measure basic allocation overhead
    overhead_stats = measure_tensor_allocation_overhead()
    
    # Benchmark allocation patterns
    pattern_stats = benchmark_allocation_patterns()
    
    # Test fragmentation patterns
    fragmentation_stats = test_fragmentation_patterns()
    
    print("\nTensor allocation/deallocation overhead measurement completed!")
"""
Comprehensive test for the advanced memory management system
Testing all implemented features and optimizations
"""

import torch
import numpy as np
from typing import Tuple
import time
import gc

from advanced_memory_management_optimizations import (
    AdvancedMemoryPool, CacheAwareMemoryManager, GPUCPUMemoryOptimizer,
    StreamOrderedMemoryPool, VisionLanguageMemoryOptimizer, 
    HardwareSpecificMemoryOptimizer, MemoryPressureMonitor,
    create_memory_optimized_model_context, MemoryPoolType
)


def test_advanced_memory_pool():
    """Test the advanced memory pool functionality"""
    print("Testing Advanced Memory Pool...")
    
    # Create a memory pool
    pool = AdvancedMemoryPool(initial_size=16 * 1024 * 1024)  # 16MB
    
    # Test allocation and deallocation
    ptr1, size1 = pool.allocate(1024, MemoryPoolType.TENSOR_DATA)  # 1KB
    ptr2, size2 = pool.allocate(2048, MemoryPoolType.ACTIVATION_BUFFER)  # 2KB
    ptr3, size3 = pool.allocate(512, MemoryPoolType.TEMPORARY)  # 512B
    
    assert ptr1 is not None, "Allocation failed"
    assert ptr2 is not None, "Allocation failed"
    assert ptr3 is not None, "Allocation failed"
    
    # Verify addresses are different
    assert ptr1 != ptr2 != ptr3, "Addresses should be different"
    
    # Deallocate
    success1 = pool.deallocate(ptr1)
    success2 = pool.deallocate(ptr2)
    success3 = pool.deallocate(ptr3)
    
    assert success1 and success2 and success3, "Deallocation failed"
    
    # Check statistics
    stats = pool.get_stats()
    print(f"  Pool stats: {stats}")
    
    pool.cleanup()
    print("  OK Advanced memory pool test passed")


def test_cache_aware_manager():
    """Test the cache-aware memory manager"""
    print("Testing Cache-Aware Memory Manager...")
    
    manager = CacheAwareMemoryManager()
    
    # Create a tensor and optimize its layout
    data = torch.randn(100, 200)
    optimized = manager.optimize_memory_layout(data, "cache_friendly")
    
    # Check that the tensor is contiguous (cache-friendly)
    assert optimized.is_contiguous(), "Tensor should be contiguous for cache-friendly layout"
    
    print(f"  Original tensor contiguous: {data.is_contiguous()}")
    print(f"  Optimized tensor contiguous: {optimized.is_contiguous()}")
    
    print("  OK Cache-aware memory manager test passed")


def test_gpu_cpu_optimizer():
    """Test GPU-CPU memory optimizer"""
    print("Testing GPU-CPU Memory Optimizer...")
    
    optimizer = GPUCPUMemoryOptimizer()
    
    # Create a tensor
    tensor = torch.randn(100, 100)
    
    # Optimize placement
    if torch.cuda.is_available():
        optimized_tensor = optimizer.optimize_tensor_placement(tensor, "auto")
        print(f"  Tensor placed on: {optimized_tensor.device}")
    else:
        # If CUDA not available, just test CPU path
        optimized_tensor = optimizer.optimize_tensor_placement(tensor, "cpu")
        print(f"  Tensor placed on: {optimized_tensor.device}")
    
    print("  OK GPU-CPU optimizer test passed")


def test_stream_ordered_pool():
    """Test stream-ordered memory pool"""
    print("Testing Stream-Ordered Memory Pool...")
    
    pool = StreamOrderedMemoryPool(pool_size=8 * 1024 * 1024, num_streams=2)  # 8MB, 2 streams
    
    # Allocate tensors with different streams
    tensor1 = pool.allocate((100, 100), torch.float32, 0)
    tensor2 = pool.allocate((50, 50), torch.float32, 1)
    
    assert tensor1 is not None, "Stream allocation failed"
    assert tensor2 is not None, "Stream allocation failed"
    
    # Deallocate
    pool.deallocate(tensor1)
    pool.deallocate(tensor2)
    
    # Test synchronization
    pool.synchronize_stream(0)
    pool.synchronize_all_streams()
    
    print("  OK Stream-ordered memory pool test passed")


def test_vision_language_optimizer():
    """Test the vision-language memory optimizer"""
    print("Testing Vision-Language Memory Optimizer...")
    
    optimizer = VisionLanguageMemoryOptimizer(
        memory_pool_size=32 * 1024 * 1024,  # 32MB
        enable_stream_ordering=True
    )
    
    # Test general tensor allocation
    tensor = optimizer.allocate_tensor_memory((200, 300), torch.float32, "general")
    assert tensor.shape == (200, 300), f"Expected shape (200, 300), got {tensor.shape}"
    
    # Test specialized allocations
    kv_tensor = optimizer.allocate_tensor_memory((4, 128, 768), torch.float32, "kv_cache")
    vision_tensor = optimizer.allocate_tensor_memory((1, 576, 768), torch.float32, "vision_features")
    text_tensor = optimizer.allocate_tensor_memory((1, 128, 768), torch.float32, "text_embeddings")
    
    assert kv_tensor is not None, "KV cache allocation failed"
    assert vision_tensor is not None, "Vision features allocation failed"
    assert text_tensor is not None, "Text embeddings allocation failed"
    
    # Test attention optimization
    attention_result = optimizer.optimize_attention_memory(2, 256, 512, 8)
    assert 'query' in attention_result, "Attention optimization failed"
    assert attention_result['query'].shape == (2, 256, 512), "Wrong query shape"
    
    # Test image processing optimization
    image_batch = torch.randn(2, 224, 224, 3)
    optimized_image = optimizer.optimize_image_processing_memory(image_batch)
    assert optimized_image.shape == image_batch.shape, "Image optimization changed shape"
    
    # Get statistics
    stats = optimizer.get_memory_stats()
    print(f"  Memory stats keys: {list(stats.keys())}")
    
    optimizer.cleanup()
    print("  OK Vision-language optimizer test passed")


def test_hardware_specific_optimizer():
    """Test hardware-specific optimizations"""
    print("Testing Hardware-Specific Optimizer...")
    
    optimizer = HardwareSpecificMemoryOptimizer((6, 1))  # SM61
    
    # Test optimal tile size calculation
    tile_64 = optimizer.get_optimal_tile_size(64)
    tile_256 = optimizer.get_optimal_tile_size(256)
    tile_512 = optimizer.get_optimal_tile_size(512)
    
    print(f"  Optimal tile sizes - 64: {tile_64}, 256: {tile_256}, 512: {tile_512}")
    
    # Test optimal batch size calculation
    batch_size = optimizer.get_optimal_batch_size(512, 768)
    print(f"  Optimal batch size for seq_len=512, hidden=768: {batch_size}")
    
    assert tile_64 >= tile_256 >= tile_512, "Tile sizes should decrease with larger head dimensions"
    assert batch_size > 0, "Batch size should be positive"

    print("  OK Hardware-specific optimizer test passed")


def test_memory_pressure_monitor():
    """Test memory pressure monitoring"""
    print("Testing Memory Pressure Monitor...")
    
    monitor = MemoryPressureMonitor()
    
    # Get initial pressure
    pressure = monitor.get_memory_pressure()
    print(f"  Initial memory pressure: {pressure:.3f}")
    
    # Get advice
    advice = monitor.get_advice()
    print(f"  Allocation advice: {advice}")
    
    # Check pressure levels
    is_high = monitor.is_high_pressure()
    is_low = monitor.is_low_pressure()
    
    print(f"  Is high pressure: {is_high}, Is low pressure: {is_low}")

    print("  OK Memory pressure monitor test passed")


def test_performance_comparison():
    """Test performance of optimized vs standard allocation"""
    print("Testing Performance Comparison...")
    
    # Create optimizers
    opt_optimizer = create_memory_optimized_model_context()
    
    # Measure standard allocation time
    start_time = time.time()
    standard_tensors = []
    for _ in range(100):
        tensor = torch.empty(64, 64, dtype=torch.float32)
        standard_tensors.append(tensor)
    standard_time = time.time() - start_time
    
    # Measure optimized allocation time
    start_time = time.time()
    optimized_tensors = []
    for _ in range(100):
        tensor = opt_optimizer.allocate_tensor_memory((64, 64), torch.float32, "general")
        optimized_tensors.append(tensor)
    optimized_time = time.time() - start_time
    
    print(f"  Standard allocation time: {standard_time:.4f}s for 100 tensors")
    print(f"  Optimized allocation time: {optimized_time:.4f}s for 100 tensors")
    
    # Clean up
    for tensor in optimized_tensors:
        opt_optimizer.free_tensor_memory(tensor)

    print("  OK Performance comparison test passed")


def test_memory_efficiency():
    """Test memory efficiency features"""
    print("Testing Memory Efficiency...")
    
    # Create optimizer with memory pooling enabled
    optimizer = VisionLanguageMemoryOptimizer(
        memory_pool_size=16 * 1024 * 1024,  # 16MB
        enable_memory_pool=True,
        enable_cache_optimization=True,
        enable_gpu_optimization=True
    )
    
    # Allocate many small tensors to test pooling efficiency
    tensors = []
    for i in range(50):
        shape = (16, 16)  # Small tensor
        tensor = optimizer.allocate_tensor_memory(shape, torch.float32, "general")
        tensors.append(tensor)
    
    # Free half of them to create fragmentation
    for i in range(0, 50, 2):
        optimizer.free_tensor_memory(tensors[i])
    
    # Get statistics to see memory usage
    stats = optimizer.get_memory_stats()
    print(f"  Memory pool utilization: {stats.get('general_pool', {}).get('pool_utilization', 0):.3f}")
    print(f"  Memory fragmentation: {stats.get('general_pool', {}).get('fragmentation', 0):.3f}")
    
    # Free remaining tensors
    for i in range(1, 50, 2):
        optimizer.free_tensor_memory(tensors[i])

    optimizer.cleanup()
    print("  OK Memory efficiency test passed")


def run_all_tests():
    """Run all tests"""
    print("Running Advanced Memory Management System Tests")
    print("=" * 50)
    
    test_advanced_memory_pool()
    test_cache_aware_manager()
    test_gpu_cpu_optimizer()
    test_stream_ordered_pool()
    test_vision_language_optimizer()
    test_hardware_specific_optimizer()
    test_memory_pressure_monitor()
    test_performance_comparison()
    test_memory_efficiency()
    
    print("=" * 50)
    print("All tests passed! OK")


if __name__ == "__main__":
    run_all_tests()
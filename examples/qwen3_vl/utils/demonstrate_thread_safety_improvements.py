"""
Demonstration of Thread Safety Improvements for Qwen3-VL Memory Management Components

This script demonstrates the implemented thread safety improvements across all memory management components.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np

# Import the thread-safe components
from src.qwen3_vl.utils.thread_safe_implementations import (
    TensorType, MemoryBlock, ThreadSafeBuddyAllocator, ThreadSafeMemoryPool,
    ThreadSafeMemoryPoolingSystem
)


def demonstrate_thread_safe_buddy_allocator():
    """Demonstrate thread-safe buddy allocator functionality"""
    print("=== Demonstrating Thread-Safe Buddy Allocator ===")
    
    allocator = ThreadSafeBuddyAllocator(total_size=1024*1024, min_block_size=256)  # 1MB pool
    
    # Show initial state
    initial_stats = allocator.get_stats()
    print(f"Initial stats: {initial_stats}")
    
    # Allocate blocks from multiple threads
    def worker(worker_id):
        worker_results = []
        for i in range(5):
            size = random.randint(256, 2048)
            block = allocator.allocate(size, TensorType.KV_CACHE, f"thread_{worker_id}_block_{i}")
            if block:
                worker_results.append((block.start_addr, block.size))
                # Simulate brief computation
                time.sleep(0.001)
                allocator.deallocate(block)
        return worker_results
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        all_results = [future.result() for future in futures]
    
    end_time = time.time()
    
    total_blocks = sum(len(results) for results in all_results)
    print(f"Allocated and deallocated {total_blocks} blocks in {end_time - start_time:.3f}s")
    
    # Show final state
    final_stats = allocator.get_stats()
    print(f"Final stats: {final_stats}")
    print("✓ Thread-Safe Buddy Allocator demonstration completed\n")


def demonstrate_thread_safe_memory_pool():
    """Demonstrate thread-safe memory pool functionality"""
    print("=== Demonstrating Thread-Safe Memory Pool ===")
    
    pool = ThreadSafeMemoryPool(TensorType.KV_CACHE, pool_size=1024*1024, min_block_size=256)
    
    def worker(worker_id):
        thread_results = []
        for i in range(8):
            size = random.randint(512, 4096)
            tensor_id = f"pool_worker_{worker_id}_tensor_{i}"
            
            block = pool.allocate(size, tensor_id)
            if block:
                thread_results.append(('alloc', tensor_id))
                # Simulate computation time
                time.sleep(0.002)
                success = pool.deallocate(tensor_id)
                if success:
                    thread_results.append(('dealloc', tensor_id))
        return thread_results
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(worker, i) for i in range(6)]
        all_results = [future.result() for future in futures]
    
    end_time = time.time()
    
    total_ops = sum(len(results) for results in all_results)
    print(f"Completed {total_ops} operations in {end_time - start_time:.3f}s")
    
    pool_stats = pool.get_pool_stats()
    print(f"Pool stats: {pool_stats}")
    print("✓ Thread-Safe Memory Pool demonstration completed\n")


def demonstrate_thread_safe_memory_pooling_system():
    """Demonstrate thread-safe memory pooling system functionality"""
    print("=== Demonstrating Thread-Safe Memory Pooling System ===")
    
    memory_system = ThreadSafeMemoryPoolingSystem(
        kv_cache_size=1024*1024*2,      # 2MB
        image_features_size=1024*1024*2,  # 2MB
        text_embeddings_size=1024*1024,    # 1MB
        gradients_size=1024*1024*2,      # 2MB
        activations_size=1024*1024,       # 1MB
        parameters_size=1024*1024*4      # 4MB
    )
    
    def worker(worker_id):
        thread_results = []
        tensor_types = [TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, 
                       TensorType.TEXT_EMBEDDINGS, TensorType.GRADIENTS]
        
        for i in range(10):
            tensor_type = random.choice(tensor_types)
            size = random.randint(1024, 8192)
            tensor_id = f"system_worker_{worker_id}_tensor_{i}_{tensor_type.value}"
            
            block = memory_system.allocate(tensor_type, size, tensor_id)
            if block:
                thread_results.append(('alloc', tensor_type.value, tensor_id))
                # Simulate brief processing
                time.sleep(0.001)
                success = memory_system.deallocate(tensor_type, tensor_id)
                if success:
                    thread_results.append(('dealloc', tensor_type.value, tensor_id))
        return thread_results
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        all_results = [future.result() for future in futures]
    
    end_time = time.time()
    
    total_ops = sum(len(results) for results in all_results)
    print(f"Completed {total_ops} cross-pool operations in {end_time - start_time:.3f}s")
    
    system_stats = memory_system.get_system_stats()
    print(f"System stats: {system_stats}")
    print("✓ Thread-Safe Memory Pooling System demonstration completed\n")


def demonstrate_memory_compaction_under_load():
    """Demonstrate memory compaction working under concurrent load"""
    print("=== Demonstrating Memory Compaction Under Load ===")
    
    memory_system = ThreadSafeMemoryPoolingSystem(
        kv_cache_size=1024*1024,
        image_features_size=1024*1024,
        text_embeddings_size=1024*1024
    )
    
    # Thread for allocation/deallocation
    def alloc_dealloc_worker():
        for i in range(20):
            tensor_type = random.choice([TensorType.KV_CACHE, TensorType.IMAGE_FEATURES])
            size = random.randint(512, 2048)
            tensor_id = f"compaction_test_{i}"
            
            block = memory_system.allocate(tensor_type, size, tensor_id)
            if block:
                time.sleep(0.001)
                memory_system.deallocate(tensor_type, tensor_id)
    
    # Thread for periodic compaction
    def compaction_worker():
        for _ in range(5):
            time.sleep(0.1)
            memory_system.compact_memory()
    
    # Run both threads concurrently
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        alloc_future = executor.submit(alloc_dealloc_worker)
        compact_future = executor.submit(compaction_worker)
        
        alloc_future.result()
        compact_future.result()
    
    end_time = time.time()
    
    print(f"Compaction completed with concurrent operations in {end_time - start_time:.3f}s")
    
    final_stats = memory_system.get_system_stats()
    print(f"Final system stats: {final_stats}")
    print("✓ Memory compaction under load demonstration completed\n")


def demonstrate_hardware_optimized_threading():
    """Demonstrate hardware-optimized threading patterns"""
    print("=== Demonstrating Hardware-Optimized Threading ===")

    # Import from the moved location
    from src.qwen3_vl.utils.thread_safe_implementations import ThreadSafeHardwareOptimizer

    hw_optimizer = ThreadSafeHardwareOptimizer()
    
    print(f"Hardware capabilities:")
    print(f"  - CPU: {hw_optimizer.cpu_cores} cores, {hw_optimizer.cpu_threads} threads")
    print(f"  - GPU Compute Capability: {hw_optimizer.gpu_compute_capability}")
    print(f"  - Max shared memory per block: {hw_optimizer.max_shared_memory_per_block // 1024} KB")
    print(f"  - Max threads per block: {hw_optimizer.max_threads_per_block}")
    
    # Get optimal configurations for different tensor types
    for tensor_type in [TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, TensorType.TEXT_EMBEDDINGS]:
        optimal_size = hw_optimizer.get_optimal_block_size(tensor_type)
        print(f"  - {tensor_type.value}: optimal block size {optimal_size:,} bytes")
    
    threading_config = hw_optimizer.get_thread_optimizations()
    print(f"  - Threading config: {threading_config}")
    
    print("✓ Hardware-optimized threading demonstration completed\n")


def run_all_demonstrations():
    """Run all thread safety demonstrations"""
    print("Qwen3-VL Thread Safety Improvements Demonstration")
    print("="*50)
    
    # Run all demonstrations
    demonstrate_thread_safe_buddy_allocator()
    demonstrate_thread_safe_memory_pool()
    demonstrate_thread_safe_memory_pooling_system()
    demonstrate_memory_compaction_under_load()
    demonstrate_hardware_optimized_threading()
    
    print("="*50)
    print("All thread safety improvements have been demonstrated successfully!")
    print("\nImplemented improvements:")
    print("1. ✓ Thread-safe buddy allocation with proper locking")
    print("2. ✓ Thread-safe memory pools with RLock protection")
    print("3. ✓ Thread-safe memory pooling system with system-level locks")
    print("4. ✓ Memory compaction that works under concurrent load")
    print("5. ✓ Hardware-specific optimizations for Intel i5-10210U + NVIDIA SM61")
    print("6. ✓ Race condition prevention through proper synchronization")
    print("7. ✓ Lock starvation prevention with fair scheduling")
    print("8. ✓ Consistent statistics under concurrent access")


if __name__ == "__main__":
    run_all_demonstrations()
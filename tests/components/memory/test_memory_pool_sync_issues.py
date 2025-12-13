"""
Test script to identify synchronization issues in CUDA memory pool implementation
"""
import threading
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add the src directory to the path to import modules
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import memory pooling module directly
import importlib.util
spec = importlib.util.spec_from_file_location("memory_pooling",
    os.path.join(os.path.dirname(__file__), "src", "qwen3_vl", "models", "memory_pooling.py"))
memory_pooling_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_pooling_module)
MemoryPool = memory_pooling_module.MemoryPool


def test_thread_safety():
    """Test thread safety of memory pool operations"""
    print("Testing thread safety of memory pool...")

    # Test with Python-based memory pool
    pool = MemoryPool(pool_size=32 * 1024 * 1024)  # 32MB pool

    results = []
    errors = []

    def worker(worker_id):
        try:
            # Allocate multiple tensors from different threads
            tensors = []
            for i in range(10):
                tensor = pool.allocate_tensor((100, 100), dtype=torch.float32)
                tensors.append(tensor)
                time.sleep(0.001)  # Small delay to increase chance of race conditions

            # Deallocate tensors
            for tensor in tensors:
                pool.free_tensor(tensor)

            # Get stats to check for consistency
            stats = pool.get_memory_stats()
            results.append((worker_id, stats))
        except Exception as e:
            errors.append((worker_id, str(e)))

    # Create multiple threads to test concurrent access
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        for future in futures:
            future.result()

    print(f"Completed {len(results)} operations, {len(errors)} errors")
    if errors:
        print(f"Errors occurred: {errors}")

    # Check for consistency in stats
    print(f"Final pool stats: {pool.get_memory_stats()}")


def test_cpu_gpu_synchronization():
    """Test CPU-GPU synchronization issues"""
    print("\nTesting CPU-GPU synchronization...")

    # Test Python-based memory pool
    pool = MemoryPool(pool_size=16 * 1024 * 1024)  # 16MB pool

    def gpu_compute():
        # Create tensors using memory pool
        tensor1 = pool.allocate_tensor((1000, 1000), dtype=torch.float32)
        tensor2 = pool.allocate_tensor((1000, 1000), dtype=torch.float32)

        # Perform GPU computation
        if torch.cuda.is_available():
            tensor1 = tensor1.cuda()
            tensor2 = tensor2.cuda()

            # Simulate computation that should be synchronized
            result = torch.matmul(tensor1, tensor2.t())

            # Synchronize to ensure completion
            torch.cuda.synchronize()

            # Free tensors back to pool
            pool.free_tensor(tensor1.cpu())
            pool.free_tensor(tensor2.cpu())
            pool.free_tensor(result.cpu())
        else:
            # CPU fallback
            result = torch.matmul(tensor1, tensor2.t())
            pool.free_tensor(tensor1)
            pool.free_tensor(tensor2)
            pool.free_tensor(result)

    # Run multiple concurrent GPU computations
    threads = []
    for i in range(4):
        t = threading.Thread(target=gpu_compute)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Final memory stats after GPU operations: {pool.get_memory_stats()}")


def test_cuda_kernel_synchronization():
    """Test synchronization between CUDA kernels and host code"""
    print("\nTesting CUDA kernel synchronization...")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA kernel sync test")
        return

    # Create tensors and run operations
    def run_kernel_with_sync():
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')

        # Perform computation
        c = torch.matmul(a, b)

        # Without synchronization, the operation might not be complete
        # when we try to access the result
        torch.cuda.synchronize()  # Explicit synchronization

        return c

    # Run multiple concurrent operations
    results = []
    for i in range(5):
        result = run_kernel_with_sync()
        results.append(result)

    print(f"Completed {len(results)} CUDA operations with synchronization")


def test_memory_fragmentation_under_load():
    """Test memory fragmentation under concurrent load"""
    print("\nTesting memory fragmentation under concurrent load...")

    pool = MemoryPool(pool_size=64 * 1024 * 1024)  # 64MB pool

    def allocate_deallocate_pattern(thread_id):
        patterns = [
            [(100, 100), (200, 200), (50, 50)],
            [(500, 500), (100, 100), (300, 300)],
            [(1000, 1000), (50, 50), (250, 250)],
            [(750, 750), (150, 150), (400, 400)]
        ]

        pattern = patterns[thread_id % len(patterns)]

        tensors = []
        for shape in pattern:
            tensor = pool.allocate_tensor(shape, dtype=torch.float32)
            tensors.append(tensor)
            time.sleep(0.001)

        # Deallocate in reverse order
        for tensor in reversed(tensors):
            pool.free_tensor(tensor)
            time.sleep(0.001)

    # Run concurrent allocation/deallocation patterns
    threads = []
    for i in range(6):
        t = threading.Thread(target=allocate_deallocate_pattern, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Check fragmentation after concurrent operations
    stats = pool.get_memory_stats()
    print(f"Memory stats after concurrent operations: {stats}")

    # Defragment and check again
    pool.defragment()
    stats_after = pool.get_memory_stats()
    print(f"Memory stats after defragmentation: {stats_after}")


if __name__ == "__main__":
    print("Testing CUDA Memory Pool Synchronization Issues")
    print("=" * 50)

    test_thread_safety()
    test_cpu_gpu_synchronization()
    test_cuda_kernel_synchronization()
    test_memory_fragmentation_under_load()

    print("\nSynchronization testing completed.")
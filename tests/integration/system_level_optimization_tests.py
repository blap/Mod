"""Comprehensive test suite for system-level optimizations including:
1. CPU-GPU communication pipeline optimization with pinned memory and asynchronous transfers
2. NVMe SSD caching for model components with multi-tier caching and LRU eviction
3. Batch processing strategies with dynamic batching and adaptive scheduling
4. Data loading and preprocessing optimization with multi-threading and prefetching
5. Intelligent resource allocation with dynamic memory management
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import tempfile
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path


class MockModel(nn.Module):
    """Mock model for testing purposes."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


# Global dataset classes to avoid pickling issues in multiprocessing
class GlobalTestDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Simulate some preprocessing time
        time.sleep(0.001)  # Small delay to simulate processing
        return self.data[idx], self.labels[idx]


class GlobalPrefetchDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Simulate preprocessing that takes time
        time.sleep(0.001)
        return self.data[idx], self.labels[idx]


class TestCPUGPUCommunicationOptimization(unittest.TestCase):
    """Test CPU-GPU communication pipeline optimization with pinned memory and asynchronous transfers."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MockModel().to(self.device)
        
    def test_pinned_memory_transfer(self):
        """Test pinned memory transfer optimization."""
        # Create pinned memory tensors
        if torch.cuda.is_available():
            # Test pinned memory allocation
            pinned_tensor = torch.zeros(100, 128, pin_memory=True)
            self.assertTrue(pinned_tensor.is_pinned())
            
            # Test transfer speed comparison
            regular_tensor = torch.randn(1000, 512)
            pinned_tensor = torch.randn(1000, 512, pin_memory=True)
            
            # Transfer to GPU with pinned memory (should be faster)
            start_time = time.time()
            gpu_tensor_pinned = pinned_tensor.to(self.device, non_blocking=True)
            pinned_transfer_time = time.time() - start_time
            
            # Transfer to GPU without pinned memory
            start_time = time.time()
            gpu_tensor_regular = regular_tensor.to(self.device, non_blocking=False)
            regular_transfer_time = time.time() - start_time
            
            # Synchronize to ensure completion
            torch.cuda.synchronize()
            
            # Pinned memory should generally be faster for transfers
            # (Note: This might not always be true due to system variations)
            print(f"Pinned transfer time: {pinned_transfer_time:.6f}s")
            print(f"Regular transfer time: {regular_transfer_time:.6f}s")
            
        else:
            # On CPU, just test that pin_memory attribute exists
            pinned_tensor = torch.zeros(10, 10, pin_memory=True)
            self.assertFalse(pinned_tensor.is_pinned())  # CPU tensors can't be pinned
    
    def test_asynchronous_transfers(self):
        """Test asynchronous data transfers."""
        if torch.cuda.is_available():
            # Create a CUDA stream for asynchronous operations
            stream = torch.cuda.Stream()
            
            # Create tensors
            cpu_tensor = torch.randn(100, 512, pin_memory=True)
            cuda_tensor = torch.zeros_like(cpu_tensor).to(self.device)
            
            # Start timer
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.cuda.stream(stream):
                # Asynchronous copy
                cuda_tensor.copy_(cpu_tensor, non_blocking=True)
            end_event.record()
            
            # Synchronize and measure time
            end_event.synchronize()
            async_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            
            print(f"Asynchronous transfer time: {async_time:.6f}s")
            
            # Test that tensor was properly transferred
            self.assertEqual(cuda_tensor.shape, cpu_tensor.shape)
        else:
            # On CPU, just ensure the code path works
            cpu_tensor = torch.randn(10, 10)
            self.assertEqual(cpu_tensor.shape, (10, 10))


class TestNVMeSSDCaching(unittest.TestCase):
    """Test NVMe SSD caching for model components with multi-tier caching and LRU eviction."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_lru_cache_eviction(self):
        """Test LRU cache eviction policy."""
        from collections import OrderedDict
        
        class LRUCache:
            def __init__(self, capacity: int):
                self.capacity = capacity
                self.cache = OrderedDict()
                
            def get(self, key: str) -> Optional[bytes]:
                if key in self.cache:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    return self.cache[key]
                return None
                
            def put(self, key: str, value: bytes):
                if key in self.cache:
                    # Update existing key
                    self.cache.move_to_end(key)
                elif len(self.cache) >= self.capacity:
                    # Evict least recently used item
                    self.cache.popitem(last=False)
                
                self.cache[key] = value
                
            def size(self) -> int:
                return len(self.cache)
        
        # Test LRU cache
        lru_cache = LRUCache(capacity=3)
        
        # Add items
        lru_cache.put("model_1", b"data1")
        lru_cache.put("model_2", b"data2")
        lru_cache.put("model_3", b"data3")
        
        self.assertEqual(lru_cache.size(), 3)
        self.assertIsNotNone(lru_cache.get("model_1"))
        
        # Access model_1 to make it most recently used
        lru_cache.get("model_1")
        
        # Add new item, should evict model_2 (least recently used)
        lru_cache.put("model_4", b"data4")
        
        self.assertEqual(lru_cache.size(), 3)
        self.assertIsNone(lru_cache.get("model_2"))  # Should be evicted
        self.assertIsNotNone(lru_cache.get("model_1"))  # Should still be there
        self.assertIsNotNone(lru_cache.get("model_4"))  # New item should be there
    
    def test_model_component_caching(self):
        """Test caching of model components to NVMe SSD."""
        # Create a mock model
        model = MockModel()
        
        # Serialize model to bytes
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        model_bytes = buffer.getvalue()
        
        # Test caching mechanism
        cache_path = os.path.join(self.cache_dir, "mock_model.pt")
        
        # Save to cache
        with open(cache_path, 'wb') as f:
            f.write(model_bytes)
        
        # Load from cache
        with open(cache_path, 'rb') as f:
            loaded_bytes = f.read()
        
        # Verify integrity
        self.assertEqual(len(model_bytes), len(loaded_bytes))
        self.assertEqual(model_bytes, loaded_bytes)
        
        # Load into new model
        new_buffer = io.BytesIO(loaded_bytes)
        new_state_dict = torch.load(new_buffer)
        
        new_model = MockModel()
        new_model.load_state_dict(new_state_dict)
        
        # Verify models are equivalent
        for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
            self.assertEqual(name1, name2)
            self.assertTrue(torch.equal(param1, param2))


class TestBatchProcessingStrategies(unittest.TestCase):
    """Test batch processing strategies with dynamic batching and adaptive scheduling."""
    
    def setUp(self):
        self.model = MockModel()
        
    def test_dynamic_batching(self):
        """Test dynamic batching based on input characteristics."""
        class DynamicBatchScheduler:
            def __init__(self, max_batch_size: int = 32, target_batch_time: float = 0.1):
                self.max_batch_size = max_batch_size
                self.target_batch_time = target_batch_time
                self.current_batch_size = 8  # Start with smaller batch
                self.performance_history = []
                
            def estimate_batch_size(self, input_lengths: list) -> int:
                """Estimate optimal batch size based on input characteristics."""
                avg_length = sum(input_lengths) / len(input_lengths) if input_lengths else 100
                
                # Adjust batch size based on sequence length
                if avg_length > 512:
                    estimated_size = max(1, self.current_batch_size // 2)
                elif avg_length > 256:
                    estimated_size = max(2, self.current_batch_size // 1.5)
                else:
                    estimated_size = min(self.max_batch_size, self.current_batch_size * 2)
                
                return int(min(estimated_size, self.max_batch_size))
                
            def update_batch_size(self, actual_time: float, batch_size: int):
                """Update batch size based on performance feedback."""
                if actual_time > self.target_batch_time * 1.2:
                    # Too slow, decrease batch size
                    self.current_batch_size = max(1, int(batch_size * 0.8))
                elif actual_time < self.target_batch_time * 0.8:
                    # Too fast, increase batch size
                    self.current_batch_size = min(self.max_batch_size, int(batch_size * 1.2))
                
                self.performance_history.append({
                    'batch_size': batch_size,
                    'time': actual_time,
                    'target_time': self.target_batch_time
                })
        
        # Test the scheduler
        scheduler = DynamicBatchScheduler(max_batch_size=16, target_batch_time=0.05)
        
        # Simulate different input lengths
        input_lengths = [100, 200, 50, 300, 150]
        estimated_size = scheduler.estimate_batch_size(input_lengths)
        
        self.assertGreaterEqual(estimated_size, 1)
        self.assertLessEqual(estimated_size, 16)
        
        # Simulate performance feedback
        scheduler.update_batch_size(actual_time=0.08, batch_size=8)  # Too slow, should decrease
        self.assertLessEqual(scheduler.current_batch_size, 8)
        
        scheduler.update_batch_size(actual_time=0.02, batch_size=4)  # Too fast, should increase
        self.assertGreaterEqual(scheduler.current_batch_size, 4)
    
    def test_adaptive_scheduling(self):
        """Test adaptive scheduling of batches."""
        class AdaptiveBatchScheduler:
            def __init__(self, batch_sizes: list = [4, 8, 16, 32]):
                self.batch_sizes = batch_sizes
                self.performance_scores = {size: 0.0 for size in batch_sizes}
                self.current_best = batch_sizes[0]
                
            def schedule_batch(self, input_data: list) -> int:
                """Schedule batch size based on input characteristics and historical performance."""
                # Simple heuristic: longer sequences need smaller batches
                avg_length = sum(len(item) for item in input_data) / len(input_data) if input_data else 100
                
                # Select batch size based on sequence length
                if avg_length > 500:
                    return 4
                elif avg_length > 250:
                    return 8
                elif avg_length > 100:
                    return 16
                else:
                    return 32
        
        scheduler = AdaptiveBatchScheduler()
        
        # Test with different input characteristics
        short_inputs = [[1, 2, 3] for _ in range(10)]
        long_inputs = [[i for i in range(1000)] for _ in range(5)]
        
        short_batch_size = scheduler.schedule_batch(short_inputs)
        long_batch_size = scheduler.schedule_batch(long_inputs)
        
        # Short sequences should allow larger batches
        self.assertGreater(short_batch_size, long_batch_size)


class TestDataLoadingOptimization(unittest.TestCase):
    """Test data loading and preprocessing optimization with multi-threading and prefetching."""
    
    def setUp(self):
        self.dataset_size = 100
        self.data = [torch.randn(128) for _ in range(self.dataset_size)]
        self.labels = [torch.randint(0, 10, (1,)) for _ in range(self.dataset_size)]
    
    def test_multi_threaded_data_loading(self):
        """Test multi-threaded data loading."""
        from torch.utils.data import DataLoader
        import threading

        # Use the global dataset class
        dataset = GlobalTestDataset(self.data, self.labels)

        # Test with multiple workers
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=min(2, os.cpu_count()),  # Use fewer workers to avoid pickling issues
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True  # Keep workers alive
        )

        start_time = time.time()
        count = 0
        for batch_data, batch_labels in dataloader:
            count += 1
            if count >= 10:  # Only process a few batches for test speed
                break

        multi_thread_time = time.time() - start_time
        print(f"Multi-threaded loading time: {multi_thread_time:.4f}s for {count} batches")

        # Test with single worker for comparison
        dataloader_single = DataLoader(
            dataset,
            batch_size=8,
            num_workers=1,  # Single worker
            pin_memory=True if torch.cuda.is_available() else False
        )

        start_time = time.time()
        count_single = 0
        for batch_data, batch_labels in dataloader_single:
            count_single += 1
            if count_single >= 10:  # Same number of batches
                break

        single_thread_time = time.time() - start_time
        print(f"Single-threaded loading time: {single_thread_time:.4f}s for {count_single} batches")
    
    def test_prefetching_optimization(self):
        """Test data prefetching optimization."""
        from torch.utils.data import DataLoader
        import queue
        import threading

        # Create prefetching dataset
        dataset = GlobalPrefetchDataset(self.data, self.labels)

        # Use DataLoader with prefetching
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=min(2, os.cpu_count()),
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Process data
        start_time = time.time()
        processed_batches = 0
        for batch_data, batch_labels in dataloader:
            processed_batches += 1
            if processed_batches >= 5:  # Process a few batches
                break

        prefetch_time = time.time() - start_time
        print(f"Prefetching optimization time: {prefetch_time:.4f}s for {processed_batches} batches")


class TestResourceAllocation(unittest.TestCase):
    """Test intelligent resource allocation with dynamic memory management."""
    
    def test_dynamic_memory_management(self):
        """Test dynamic memory management system."""
        class MemoryPool:
            def __init__(self, initial_size: int = 1024 * 1024 * 100):  # 100MB
                self.total_size = initial_size
                self.used_size = 0
                self.free_blocks = [(0, initial_size)]  # (start, size) tuples
                self.allocated_tensors = {}  # id -> (start, size, tensor)
                
            def allocate(self, size: int, dtype=torch.float32):
                """Allocate a tensor of given size."""
                # Find first fit
                for i, (start, block_size) in enumerate(self.free_blocks):
                    if block_size >= size:
                        # Allocate from this block
                        self.free_blocks.pop(i)
                        
                        if block_size > size:
                            # Split the block
                            self.free_blocks.insert(i, (start + size, block_size - size))
                        
                        # Create tensor in the allocated space simulation
                        tensor = torch.zeros(size // 4, dtype=dtype)  # Assuming float32 = 4 bytes
                        
                        tensor_id = id(tensor)
                        self.allocated_tensors[tensor_id] = (start, size, tensor)
                        self.used_size += size
                        
                        return tensor
                
                # If no suitable block found, raise error (in real implementation, we'd expand)
                raise MemoryError(f"Not enough memory to allocate {size} bytes")
                
            def deallocate(self, tensor):
                """Deallocate a tensor."""
                tensor_id = id(tensor)
                if tensor_id in self.allocated_tensors:
                    start, size, _ = self.allocated_tensors[tensor_id]
                    
                    # Add back to free blocks (simple implementation, no coalescing)
                    self.free_blocks.append((start, size))
                    self.free_blocks.sort()  # Keep sorted for easier management
                    
                    del self.allocated_tensors[tensor_id]
                    self.used_size -= size
                    
                    # Coalesce adjacent free blocks
                    self._coalesce_blocks()
                    
            def _coalesce_blocks(self):
                """Coalesce adjacent free blocks."""
                if not self.free_blocks:
                    return
                    
                coalesced = []
                current_start, current_size = self.free_blocks[0]
                
                for start, size in self.free_blocks[1:]:
                    if current_start + current_size == start:
                        # Adjacent blocks, merge them
                        current_size += size
                    else:
                        # Non-adjacent, add previous block and start new
                        coalesced.append((current_start, current_size))
                        current_start, current_size = start, size
                
                coalesced.append((current_start, current_size))
                self.free_blocks = coalesced
                
            def get_memory_stats(self) -> Dict[str, int]:
                """Get memory usage statistics."""
                return {
                    'total_size': self.total_size,
                    'used_size': self.used_size,
                    'free_size': self.total_size - self.used_size,
                    'fragmentation': len(self.free_blocks),
                    'utilization': self.used_size / self.total_size if self.total_size > 0 else 0
                }
        
        # Test memory pool
        pool = MemoryPool(initial_size=1024 * 1024)  # 1MB
        
        # Allocate some tensors
        tensor1 = pool.allocate(1024 * 4, torch.float32)  # 4KB
        tensor2 = pool.allocate(1024 * 8, torch.float32)  # 8KB
        tensor3 = pool.allocate(1024 * 16, torch.float32)  # 16KB
        
        stats = pool.get_memory_stats()
        self.assertGreater(stats['used_size'], 0)
        self.assertLessEqual(stats['used_size'], stats['total_size'])
        
        # Deallocate some tensors
        pool.deallocate(tensor2)
        
        stats_after_dealloc = pool.get_memory_stats()
        self.assertLess(stats_after_dealloc['used_size'], stats['used_size'])
        
        # Allocate again to test reuse
        tensor4 = pool.allocate(1024 * 6, torch.float32)  # 6KB, should fit in freed space
        
        final_stats = pool.get_memory_stats()
        self.assertGreater(final_stats['used_size'], stats_after_dealloc['used_size'])
    
    def test_power_aware_resource_allocation(self):
        """Test power-aware resource allocation."""
        class PowerAwareAllocator:
            def __init__(self, max_power_budget: float = 100.0):
                self.max_power_budget = max_power_budget
                self.current_power_usage = 0.0
                self.resource_allocations = {}
                
            def allocate_resources(self, required_power: float, resource_type: str = "gpu"):
                """Allocate resources considering power constraints."""
                if self.current_power_usage + required_power > self.max_power_budget:
                    # Need to scale down or defer allocation
                    available_power = self.max_power_budget - self.current_power_usage
                    if available_power <= 0:
                        return False, "No power budget available"
                    
                    # Scale down the request
                    scaled_power = min(required_power, available_power)
                    self.current_power_usage += scaled_power
                    self.resource_allocations[id(required_power)] = {
                        'power': scaled_power,
                        'type': resource_type,
                        'scaled_down': True
                    }
                    return True, f"Allocated {scaled_power}/{required_power}W (scaled down)"
                else:
                    # Normal allocation
                    self.current_power_usage += required_power
                    self.resource_allocations[id(required_power)] = {
                        'power': required_power,
                        'type': resource_type,
                        'scaled_down': False
                    }
                    return True, f"Allocated {required_power}W"
                    
            def get_power_stats(self) -> Dict[str, float]:
                """Get power usage statistics."""
                return {
                    'current_usage': self.current_power_usage,
                    'max_budget': self.max_power_budget,
                    'utilization': self.current_power_usage / self.max_power_budget if self.max_power_budget > 0 else 0,
                    'available': self.max_power_budget - self.current_power_usage
                }
        
        allocator = PowerAwareAllocator(max_power_budget=50.0)
        
        # Allocate resources
        success1, msg1 = allocator.allocate_resources(20.0, "gpu")
        self.assertTrue(success1)
        
        success2, msg2 = allocator.allocate_resources(15.0, "cpu")
        self.assertTrue(success2)
        
        # Try to allocate more than available
        success3, msg3 = allocator.allocate_resources(30.0, "gpu")
        self.assertTrue(success3)  # Should succeed but be scaled down
        self.assertIn("scaled down", msg3)
        
        stats = allocator.get_power_stats()
        self.assertLessEqual(stats['current_usage'], stats['max_budget'])


def run_all_tests():
    """Run all system-level optimization tests."""
    print("Running System-Level Optimization Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCPUGPUCommunicationOptimization))
    suite.addTests(loader.loadTestsFromTestCase(TestNVMeSSDCaching))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchProcessingStrategies))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoadingOptimization))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceAllocation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
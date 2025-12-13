"""
Tests for HierarchicalMemoryManager
This module contains comprehensive tests for the HierarchicalMemoryManager class,
validating all functionality including multi-layer caching, tensor movement,
prediction algorithms, and hardware-specific optimizations.
"""

import unittest
import torch
import tempfile
import shutil
import os
import time
from unittest.mock import patch, MagicMock
import sys

# Import the main class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hierarchical_memory_manager import (
    HierarchicalMemoryManager,
    MemoryLevel,
    TensorMetadata,
    LRUCache,
    AccessPredictor
)


class TestLRUCache(unittest.TestCase):
    """Test the LRUCache implementation"""
    
    def setUp(self):
        self.cache = LRUCache(max_size_bytes=1024, memory_level=MemoryLevel.CPU_RAM)
    
    def test_put_and_get_tensor(self):
        """Test putting and getting a tensor from cache"""
        tensor = torch.randn(10, 10)
        metadata = TensorMetadata(
            name="test_tensor",
            shape=(10, 10),
            dtype=torch.float32,
            size_bytes=tensor.element_size() * tensor.nelement(),
            level=MemoryLevel.CPU_RAM,
            last_access_time=time.time(),
            access_count=0
        )
        
        key = "test_tensor"
        success = self.cache.put(key, tensor, metadata)
        self.assertTrue(success)
        
        retrieved_tensor = self.cache.get(key)
        self.assertIsNotNone(retrieved_tensor)
        self.assertTrue(torch.equal(tensor, retrieved_tensor))
    
    def test_cache_eviction_lru_policy(self):
        """Test that cache evicts according to LRU policy"""
        # Create tensors that together exceed cache size
        tensor1 = torch.randn(8, 8)  # 256 floats = 1024 bytes
        tensor2 = torch.randn(5, 5)  # 25 floats = 100 bytes
        tensor3 = torch.randn(6, 6)  # 36 floats = 144 bytes
        
        metadata1 = TensorMetadata(
            name="tensor1",
            shape=(8, 8),
            dtype=torch.float32,
            size_bytes=tensor1.element_size() * tensor1.nelement(),
            level=MemoryLevel.CPU_RAM,
            last_access_time=time.time(),
            access_count=0
        )
        
        metadata2 = TensorMetadata(
            name="tensor2",
            shape=(5, 5),
            dtype=torch.float32,
            size_bytes=tensor2.element_size() * tensor2.nelement(),
            level=MemoryLevel.CPU_RAM,
            last_access_time=time.time(),
            access_count=0
        )
        
        metadata3 = TensorMetadata(
            name="tensor3",
            shape=(6, 6),
            dtype=torch.float32,
            size_bytes=tensor3.element_size() * tensor3.nelement(),
            level=MemoryLevel.CPU_RAM,
            last_access_time=time.time(),
            access_count=0
        )
        
        # Add first two tensors
        self.cache.put("tensor1", tensor1, metadata1)
        self.cache.put("tensor2", tensor2, metadata2)
        
        # Access tensor1 to make it most recently used
        self.cache.get("tensor1")
        
        # Add third tensor - should evict tensor2 (least recently used)
        self.cache.put("tensor3", tensor3, metadata3)
        
        # tensor1 should still be there
        self.assertIsNotNone(self.cache.get("tensor1"))
        # tensor3 should be there
        self.assertIsNotNone(self.cache.get("tensor3"))
        # tensor2 should be evicted
        self.assertIsNone(self.cache.get("tensor2"))
    
    def test_cache_full_detection(self):
        """Test cache full detection"""
        tensor = torch.randn(16, 16)  # 1024 floats = 4096 bytes (too large)
        
        metadata = TensorMetadata(
            name="large_tensor",
            shape=(16, 16),
            dtype=torch.float32,
            size_bytes=tensor.element_size() * tensor.nelement(),
            level=MemoryLevel.CPU_RAM,
            last_access_time=time.time(),
            access_count=0
        )
        
        success = self.cache.put("large_tensor", tensor, metadata)
        self.assertFalse(success)  # Should fail because tensor is too large
    
    def test_remove_tensor(self):
        """Test removing a tensor from cache"""
        tensor = torch.randn(5, 5)
        metadata = TensorMetadata(
            name="removable_tensor",
            shape=(5, 5),
            dtype=torch.float32,
            size_bytes=tensor.element_size() * tensor.nelement(),
            level=MemoryLevel.CPU_RAM,
            last_access_time=time.time(),
            access_count=0
        )
        
        key = "removable_tensor"
        self.cache.put(key, tensor, metadata)
        
        # Verify it's there
        self.assertIsNotNone(self.cache.get(key))
        
        # Remove it
        success = self.cache.remove(key)
        self.assertTrue(success)
        
        # Verify it's gone
        self.assertIsNone(self.cache.get(key))


class TestAccessPredictor(unittest.TestCase):
    """Test the AccessPredictor implementation"""
    
    def setUp(self):
        self.predictor = AccessPredictor()
    
    def test_record_access(self):
        """Test recording access events"""
        key = "test_tensor"
        initial_count = len(self.predictor.access_history.get(key, []))
        
        self.predictor.record_access(key)
        time.sleep(0.01)  # Small delay to ensure different timestamp
        self.predictor.record_access(key)
        
        final_count = len(self.predictor.access_history.get(key, []))
        self.assertEqual(final_count, initial_count + 2)
    
    def test_predict_next_access_single_access(self):
        """Test prediction with only one access recorded"""
        key = "single_access_tensor"
        self.predictor.record_access(key)
        
        prediction = self.predictor.predict_next_access(key)
        self.assertIsNone(prediction)  # Need at least 2 accesses for prediction
    
    def test_predict_next_access_multiple_accesses(self):
        """Test prediction with multiple accesses"""
        key = "multi_access_tensor"
        
        # Record accesses with consistent interval
        start_time = time.time()
        self.predictor.access_history[key] = [
            start_time,
            start_time + 1.0,  # 1 second interval
            start_time + 2.0   # Another 1 second interval
        ]
        
        prediction = self.predictor.predict_next_access(key)
        expected_prediction = start_time + 3.0
        
        # Allow some tolerance for timing variations
        self.assertAlmostEqual(prediction, expected_prediction, delta=0.1)
    
    def test_access_frequency_calculation(self):
        """Test calculation of access frequency"""
        key = "freq_test_tensor"
        
        # Record accesses over 2 seconds
        start_time = time.time()
        self.predictor.access_history[key] = [
            start_time,
            start_time + 0.5,
            start_time + 1.0,
            start_time + 1.5,
            start_time + 2.0
        ]
        
        freq = self.predictor.get_access_frequency(key)
        # 4 intervals in 2 seconds = 2 Hz
        self.assertAlmostEqual(freq, 2.0, delta=0.1)


class TestHierarchicalMemoryManager(unittest.TestCase):
    """Test the main HierarchicalMemoryManager class"""
    
    def setUp(self):
        # Use a temporary directory for disk cache
        self.temp_dir = tempfile.mkdtemp()
        self.manager = HierarchicalMemoryManager(
            cpu_ram_limit_bytes=1024*1024,  # 1MB
            gpu_vram_limit_bytes=2*1024*1024 if torch.cuda.is_available() else 0,  # 2MB or 0 if no GPU
            disk_cache_path=self.temp_dir,
            enable_prediction=True
        )
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
        self.manager.shutdown()
    
    def test_allocate_tensor_cpu(self):
        """Test allocating a tensor in CPU memory"""
        shape = (100, 100)
        tensor = self.manager.allocate_tensor(
            key="cpu_tensor",
            shape=shape,
            dtype=torch.float32,
            preferred_level=MemoryLevel.CPU_RAM
        )
        
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device.type, 'cpu')
        
        # Verify tensor is registered
        self.assertIn("cpu_tensor", self.manager.tensor_registry)
        self.assertEqual(self.manager.tensor_locations["cpu_tensor"], MemoryLevel.CPU_RAM)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_allocate_tensor_gpu(self):
        """Test allocating a tensor in GPU memory"""
        shape = (100, 100)
        tensor = self.manager.allocate_tensor(
            key="gpu_tensor",
            shape=shape,
            dtype=torch.float32,
            preferred_level=MemoryLevel.GPU_VRAM
        )
        
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device.type, 'cuda')
        
        # Verify tensor is registered
        self.assertIn("gpu_tensor", self.manager.tensor_registry)
        self.assertEqual(self.manager.tensor_locations["gpu_tensor"], MemoryLevel.GPU_VRAM)
    
    def test_get_tensor(self):
        """Test retrieving a tensor from the hierarchy"""
        # Allocate a tensor
        original_tensor = self.manager.allocate_tensor(
            key="retrievable_tensor",
            shape=(50, 50),
            dtype=torch.float32,
            preferred_level=MemoryLevel.CPU_RAM
        )
        
        # Modify the tensor to have specific values
        original_tensor.fill_(42.0)
        
        # Retrieve the tensor
        retrieved_tensor = self.manager.get_tensor("retrievable_tensor")
        
        self.assertIsNotNone(retrieved_tensor)
        self.assertTrue(torch.equal(original_tensor, retrieved_tensor))
        self.assertEqual(retrieved_tensor[0, 0].item(), 42.0)
    
    def test_put_tensor(self):
        """Test storing a tensor in the hierarchy"""
        tensor = torch.randn(20, 20)
        
        success = self.manager.put_tensor(
            key="stored_tensor",
            tensor=tensor,
            level=MemoryLevel.CPU_RAM
        )
        
        self.assertTrue(success)
        
        # Verify tensor is accessible
        retrieved_tensor = self.manager.get_tensor("stored_tensor")
        self.assertIsNotNone(retrieved_tensor)
        self.assertTrue(torch.equal(tensor, retrieved_tensor))
    
    def test_tensor_movement_between_levels(self):
        """Test moving a tensor between different memory levels"""
        # Create a small tensor that fits in any level
        tensor = torch.randn(10, 10)
        
        # Store initially in CPU
        self.manager.put_tensor("movable_tensor", tensor, MemoryLevel.CPU_RAM)
        self.assertEqual(self.manager.get_tensor_location("movable_tensor"), MemoryLevel.CPU_RAM)
        
        # Move to GPU if available
        if self.manager.gpu_cache:
            moved_successfully = self.manager._move_tensor("movable_tensor", MemoryLevel.GPU_VRAM)
            self.assertTrue(moved_successfully)
            self.assertEqual(self.manager.get_tensor_location("movable_tensor"), MemoryLevel.GPU_VRAM)
    
    def test_memory_stats(self):
        """Test getting memory statistics"""
        # Allocate a few tensors
        self.manager.allocate_tensor("stat_tensor1", (50, 50), torch.float32, MemoryLevel.CPU_RAM)
        self.manager.allocate_tensor("stat_tensor2", (30, 30), torch.float32, MemoryLevel.CPU_RAM)
        
        stats = self.manager.get_memory_stats()
        
        self.assertIn('cpu_cache', stats)
        self.assertGreater(stats['cpu_cache']['usage_bytes'], 0)
        self.assertGreater(stats['cpu_cache']['num_tensors'], 0)
        
        if self.manager.gpu_cache:
            self.assertIn('gpu_cache', stats)
    
    def test_pin_tensor(self):
        """Test pinning a tensor in CPU memory"""
        # Allocate tensor in CPU
        tensor = self.manager.allocate_tensor(
            "pinnable_tensor",
            (20, 20),
            torch.float32,
            preferred_level=MemoryLevel.CPU_RAM
        )
        
        # Pin the tensor
        success = self.manager.pin_tensor("pinnable_tensor")
        self.assertTrue(success)
        
        # Verify the tensor is now pinned
        cached_tensor = self.manager.get_tensor("pinnable_tensor")
        self.assertTrue(cached_tensor.is_pinned())
    
    def test_evict_tensor(self):
        """Test evicting a tensor"""
        # Allocate tensor
        tensor = self.manager.allocate_tensor(
            "evictable_tensor",
            (15, 15),
            torch.float32,
            preferred_level=MemoryLevel.CPU_RAM
        )
        
        # Verify it exists
        self.assertIsNotNone(self.manager.get_tensor("evictable_tensor"))
        
        # Evict the tensor
        success = self.manager.evict_tensor("evictable_tensor")
        self.assertTrue(success)
        
        # Verify it's no longer in memory (but might be on disk)
        # The tensor should be removed from active memory but preserved on disk
        # So we check that it's not in the CPU cache anymore
        self.assertIsNone(self.manager.cpu_cache.get("evictable_tensor"))
    
    def test_prediction_based_movement(self):
        """Test tensor movement based on access predictions"""
        # Record multiple accesses for a tensor to establish a pattern
        key = "predicted_tensor"
        
        # Allocate tensor
        self.manager.allocate_tensor(key, (10, 10), torch.float32, MemoryLevel.CPU_RAM)
        
        # Simulate access pattern by calling get_tensor multiple times
        for _ in range(5):
            tensor = self.manager.get_tensor(key)
            time.sleep(0.1)  # Small delay to simulate time between accesses
        
        # Check that the predictor has recorded the accesses
        if self.manager.predictor:
            access_freq = self.manager.predictor.get_access_frequency(key)
            self.assertGreater(access_freq, 0.0)


class TestHardwareOptimizations(unittest.TestCase):
    """Test hardware-specific optimizations"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = HierarchicalMemoryManager(
            cpu_ram_limit_bytes=512*1024,  # 512KB
            gpu_vram_limit_bytes=1024*1024 if torch.cuda.is_available() else 0,  # 1MB or 0 if no GPU
            disk_cache_path=self.temp_dir,
            enable_prediction=True
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        self.manager.shutdown()
    
    def test_memory_pressure_monitoring(self):
        """Test that memory pressure monitoring works"""
        # Allocate tensors to increase memory usage
        for i in range(10):
            self.manager.allocate_tensor(
                f"pressure_tensor_{i}",
                (50, 50),
                torch.float32,
                MemoryLevel.CPU_RAM
            )
        
        # Check memory stats
        stats = self.manager.get_memory_stats()
        cpu_usage_percent = stats['cpu_cache']['usage_percent']
        
        # CPU cache should have some usage
        self.assertGreater(cpu_usage_percent, 0)
    
    def test_clear_cache(self):
        """Test clearing cache functionality"""
        # Add some tensors
        for i in range(5):
            self.manager.allocate_tensor(f"clear_tensor_{i}", (10, 10), torch.float32, MemoryLevel.CPU_RAM)
        
        # Verify cache has items
        stats_before = self.manager.get_memory_stats()
        self.assertGreater(stats_before['cpu_cache']['num_tensors'], 0)
        
        # Clear cache
        self.manager.clear_cache(MemoryLevel.CPU_RAM)
        
        # Verify cache is empty
        stats_after = self.manager.get_memory_stats()
        self.assertEqual(stats_after['cpu_cache']['num_tensors'], 0)
        self.assertEqual(stats_after['cpu_cache']['usage_bytes'], 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
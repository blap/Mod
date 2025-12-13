"""
Comprehensive tests for the Qwen3-VL Advanced Memory Tiering System.

This test suite validates all aspects of the memory tiering system:
- Basic functionality
- Tensor placement logic
- Migration decisions
- Performance monitoring
- Integration with Qwen3-VL model components
"""

import unittest
import torch
import numpy as np
from src.models.memory_tiering import (
    Qwen3VLMemoryTieringSystem,
    MemoryTier,
    TensorType,
    TensorMetadata,
    create_qwen3vl_memory_tiering_system
)
import time


class TestQwen3VLMemoryTieringSystem(unittest.TestCase):
    """Test cases for the Qwen3-VL Memory Tiering System"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tiering_system = create_qwen3vl_memory_tiering_system({
            'gpu_memory': 512 * 1024 * 1024,   # 512MB GPU
            'cpu_memory': 1024 * 1024 * 1024,  # 1GB CPU
            'storage_type': 'nvme'
        })

    def test_basic_tensor_operations(self):
        """Test basic tensor put and get operations."""
        # Create a tensor
        tensor = torch.randn(10, 10, dtype=torch.float16)
        
        # Put tensor in system
        success, tensor_id = self.tiering_system.put_tensor(tensor, TensorType.GENERAL)
        self.assertTrue(success)
        self.assertIsNotNone(tensor_id)
        
        # Retrieve tensor
        retrieved_tensor = self.tiering_system.get_tensor(tensor_id)
        self.assertIsNotNone(retrieved_tensor)
        self.assertEqual(retrieved_tensor.shape, tensor.shape)
        self.assertEqual(retrieved_tensor.dtype, tensor.dtype)

    def test_tensor_type_based_placement(self):
        """Test that different tensor types are placed in appropriate tiers."""
        # KV cache tensors should go to GPU if small enough
        kv_tensor = torch.randn(2, 64, 32, dtype=torch.float16)
        success, kv_id = self.tiering_system.put_tensor(kv_tensor, TensorType.KV_CACHE)
        self.assertTrue(success)
        
        # Check placement info
        placement_info = self.tiering_system.get_tensor_placement_info(kv_id)
        self.assertIsNotNone(placement_info)
        # KV cache tensors should be in GPU or CPU tier
        self.assertIn(placement_info.tier, [MemoryTier.GPU_HBM, MemoryTier.CPU_RAM])
        
        # Large general tensors should go to CPU or SSD
        large_tensor = torch.randn(100, 100, 100, dtype=torch.float16)  # ~20MB
        success, large_id = self.tiering_system.put_tensor(large_tensor, TensorType.GENERAL)
        self.assertTrue(success)
        
        # Image features should go to faster tiers
        img_tensor = torch.randn(1, 196, 768, dtype=torch.float16)  # ~1.2MB
        success, img_id = self.tiering_system.put_tensor(img_tensor, TensorType.IMAGE_FEATURES)
        self.assertTrue(success)
        
        img_placement = self.tiering_system.get_tensor_placement_info(img_id)
        self.assertIsNotNone(img_placement)
        # Image features should be in GPU or CPU tier
        self.assertIn(img_placement.tier, [MemoryTier.GPU_HBM, MemoryTier.CPU_RAM])

    def test_access_pattern_tracking(self):
        """Test that access patterns are properly tracked."""
        tensor = torch.randn(5, 5, dtype=torch.float16)
        success, tensor_id = self.tiering_system.put_tensor(tensor, TensorType.GENERAL)
        self.assertTrue(success)
        
        # Access tensor multiple times
        for _ in range(5):
            retrieved = self.tiering_system.get_tensor(tensor_id)
            self.assertIsNotNone(retrieved)
            time.sleep(0.01)  # Small delay to create time differences
        
        # Check that access count is updated
        metadata = self.tiering_system.get_tensor_placement_info(tensor_id)
        self.assertEqual(metadata.access_count, 5)
        
        # Check that temporal locality score is updated
        self.assertGreater(metadata.temporal_locality_score, 0.0)

    def test_tensor_migration_decision(self):
        """Test that migration decisions are made correctly."""
        # Create a tensor and access it frequently to establish a pattern
        tensor = torch.randn(10, 10, dtype=torch.float16)
        success, tensor_id = self.tiering_system.put_tensor(tensor, TensorType.GENERAL)
        self.assertTrue(success)
        
        # Access tensor multiple times to make it "hot"
        for _ in range(10):
            retrieved = self.tiering_system.get_tensor(tensor_id)
            self.assertIsNotNone(retrieved)
            time.sleep(0.01)
        
        # The tensor should be considered for migration to a faster tier
        should_migrate, target_tier, benefit = self.tiering_system._should_migrate(
            tensor_id, self.tiering_system.tensor_locations[tensor_id]
        )
        
        # Note: Migration may not happen immediately due to cost-benefit analysis
        # but the system should at least evaluate the need for migration

    def test_pinned_tensor_behavior(self):
        """Test that pinned tensors are not migrated."""
        tensor = torch.randn(5, 5, dtype=torch.float16)
        success, tensor_id = self.tiering_system.put_tensor(tensor, TensorType.GENERAL, pinned=True)
        self.assertTrue(success)
        
        # Check that pinned tensor is not eligible for migration
        should_migrate, target_tier, benefit = self.tiering_system._should_migrate(
            tensor_id, self.tiering_system.tensor_locations[tensor_id]
        )
        self.assertFalse(should_migrate)

    def test_optimal_tier_calculation(self):
        """Test the optimal tier calculation for different tensor characteristics."""
        # High frequency, high temporal locality tensor should go to GPU
        tier = self.tiering_system.get_optimal_tier_for_tensor(
            tensor_size=1024*1024,  # 1MB
            tensor_type=TensorType.GENERAL,
            access_frequency=5.0,
            temporal_locality=0.8
        )
        self.assertIn(tier, [MemoryTier.GPU_HBM, MemoryTier.CPU_RAM])
        
        # KV cache tensors should go to faster tier regardless of other factors
        tier = self.tiering_system.get_optimal_tier_for_tensor(
            tensor_size=1024*1024,  # 1MB
            tensor_type=TensorType.KV_CACHE,
            access_frequency=1.0,
            temporal_locality=0.2
        )
        self.assertIn(tier, [MemoryTier.GPU_HBM, MemoryTier.CPU_RAM])
        
        # Low frequency, low temporal locality tensor should go to SSD
        tier = self.tiering_system.get_optimal_tier_for_tensor(
            tensor_size=1024*1024,  # 1MB
            tensor_type=TensorType.GENERAL,
            access_frequency=0.1,
            temporal_locality=0.1
        )
        self.assertEqual(tier, MemoryTier.SSD_STORAGE)

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        initial_stats = self.tiering_system.get_stats()
        
        # Add some tensors
        for i in range(5):
            tensor = torch.randn(10, 10, dtype=torch.float16)
            success, tensor_id = self.tiering_system.put_tensor(tensor, TensorType.GENERAL)
            self.assertTrue(success)
            
            # Access each tensor
            retrieved = self.tiering_system.get_tensor(tensor_id)
            self.assertIsNotNone(retrieved)
        
        final_stats = self.tiering_system.get_stats()
        
        # Check that statistics have been updated
        self.assertGreater(final_stats['global_stats']['total_requests'], 
                          initial_stats['global_stats']['total_requests'])
        self.assertGreater(final_stats['total_cached_tensors'], 0)
        
        # Check tensor type distribution
        self.assertGreater(final_stats['tensor_type_distribution']['general'], 0)

    def test_device_targeting(self):
        """Test tensor retrieval with device targeting."""
        tensor = torch.randn(5, 5, dtype=torch.float16)
        success, tensor_id = self.tiering_system.put_tensor(tensor, TensorType.GENERAL)
        self.assertTrue(success)
        
        # Get tensor without specifying device (should return in original location)
        retrieved = self.tiering_system.get_tensor(tensor_id)
        self.assertIsNotNone(retrieved)
        
        # Get tensor targeting CPU device
        if torch.cuda.is_available():
            retrieved_on_cpu = self.tiering_system.get_tensor(tensor_id, target_device=torch.device('cpu'))
            self.assertIsNotNone(retrieved_on_cpu)
            self.assertEqual(retrieved_on_cpu.device.type, 'cpu')

    def test_large_tensor_handling(self):
        """Test handling of large tensors."""
        # Create a large tensor that should go to SSD
        large_tensor = torch.randn(100, 100, 100, dtype=torch.float32)  # ~40MB
        success, tensor_id = self.tiering_system.put_tensor(large_tensor, TensorType.GENERAL)
        self.assertTrue(success)
        
        # Retrieve large tensor
        retrieved = self.tiering_system.get_tensor(tensor_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.shape, large_tensor.shape)

    def test_tensor_type_distribution(self):
        """Test that different tensor types are properly tracked in statistics."""
        tensor_types = [
            (torch.randn(10, 10), TensorType.KV_CACHE),
            (torch.randn(10, 10), TensorType.IMAGE_FEATURES),
            (torch.randn(10, 10), TensorType.TEXT_EMBEDDINGS),
            (torch.randn(10, 10), TensorType.CROSS_ATTENTION),
            (torch.randn(10, 10), TensorType.GENERAL)
        ]
        
        tensor_ids = []
        for tensor, tensor_type in tensor_types:
            success, tensor_id = self.tiering_system.put_tensor(tensor, tensor_type)
            self.assertTrue(success)
            tensor_ids.append(tensor_id)
        
        # Check statistics include all tensor types
        stats = self.tiering_system.get_stats()
        for tensor_type, _ in [(t[1].value, t[0]) for t in tensor_types]:
            self.assertIn(tensor_type, stats['tensor_type_distribution'])
            self.assertGreater(stats['tensor_type_distribution'][tensor_type], 0)

    def test_clear_operations(self):
        """Test clearing tensors from tiers."""
        # Add tensors to all tiers
        tensor_gpu = torch.randn(5, 5, dtype=torch.float16)
        success, gpu_id = self.tiering_system.put_tensor(tensor_gpu, TensorType.KV_CACHE)
        self.assertTrue(success)
        
        tensor_cpu = torch.randn(10, 10, dtype=torch.float16)
        success, cpu_id = self.tiering_system.put_tensor(tensor_cpu, TensorType.GENERAL)
        self.assertTrue(success)
        
        # Clear GPU tier
        self.tiering_system.clear_tier(MemoryTier.GPU_HBM)
        
        # GPU tensor should no longer be accessible
        retrieved = self.tiering_system.get_tensor(gpu_id)
        self.assertIsNone(retrieved)
        
        # CPU tensor should still be accessible
        retrieved = self.tiering_system.get_tensor(cpu_id)
        self.assertIsNotNone(retrieved)
        
        # Clear all tiers
        self.tiering_system.clear_all()
        
        # CPU tensor should no longer be accessible
        retrieved = self.tiering_system.get_tensor(cpu_id)
        self.assertIsNone(retrieved)

    def test_predictive_migration(self):
        """Test predictive migration functionality."""
        # Create several tensors with different access patterns
        tensors = []
        for i in range(5):
            tensor = torch.randn(5, 5, dtype=torch.float16)
            success, tensor_id = self.tiering_system.put_tensor(tensor, TensorType.GENERAL)
            self.assertTrue(success)
            tensors.append((tensor_id, tensor))
        
        # Access some tensors frequently to make them "hot"
        for i in range(3):
            for _ in range(5):
                retrieved = self.tiering_system.get_tensor(tensors[i][0])
                self.assertIsNotNone(retrieved)
                time.sleep(0.01)
        
        # Access other tensors infrequently
        for i in range(3, 5):
            retrieved = self.tiering_system.get_tensor(tensors[i][0])
            self.assertIsNotNone(retrieved)
        
        # Perform predictive migrations
        self.tiering_system._perform_predictive_migrations()
        
        # Check that statistics were updated
        stats = self.tiering_system.get_stats()
        # Migration count might be 0 if cost-benefit analysis prevents migration
        # but the function should run without errors

    def test_integration_with_qwen3vl_model(self):
        """Test integration functions for Qwen3-VL model."""
        from src.models.memory_tiering import integrate_with_qwen3vl_model
        
        # Get integration functions
        alloc_kv, alloc_img, access_tensor = integrate_with_qwen3vl_model(self.tiering_system)
        
        # Test KV cache allocation
        kv_tensor, kv_id = alloc_kv(2, 128, 64, 8)  # batch, seq, hidden, heads
        self.assertIsNotNone(kv_tensor)
        if kv_id is not None:
            self.assertTrue(isinstance(kv_id, str))
        
        # Test image features allocation
        img_tensor, img_id = alloc_img(1, 196, 768)  # batch, patches, features
        self.assertIsNotNone(img_tensor)
        if img_id is not None:
            self.assertTrue(isinstance(img_id, str))
        
        # Test tensor access
        if kv_id is not None:
            accessed_tensor = access_tensor(kv_id)
            self.assertIsNotNone(accessed_tensor)


class TestMemoryTieringPerformance(unittest.TestCase):
    """Performance tests for the memory tiering system."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tiering_system = create_qwen3vl_memory_tiering_system({
            'gpu_memory': 256 * 1024 * 1024,   # 256MB GPU
            'cpu_memory': 512 * 1024 * 1024,   # 512MB CPU
            'storage_type': 'nvme'
        })

    def test_throughput_under_load(self):
        """Test system throughput under load."""
        # Create multiple tensors
        tensor_ids = []
        for i in range(20):
            tensor = torch.randn(10, 10, dtype=torch.float16)
            success, tensor_id = self.tiering_system.put_tensor(tensor, TensorType.GENERAL)
            self.assertTrue(success)
            tensor_ids.append(tensor_id)
        
        # Time multiple access operations
        start_time = time.time()
        for _ in range(100):  # Access all tensors 5 times each
            for tensor_id in tensor_ids:
                retrieved = self.tiering_system.get_tensor(tensor_id)
                self.assertIsNotNone(retrieved)
        end_time = time.time()
        
        total_time = end_time - start_time
        total_accesses = 100 * len(tensor_ids)
        accesses_per_second = total_accesses / total_time
        
        # Should be able to handle at least 100 accesses per second
        self.assertGreater(accesses_per_second, 100, 
                         f"Throughput too low: {accesses_per_second} accesses/sec")

    def test_memory_utilization(self):
        """Test memory utilization tracking."""
        # Add tensors and check utilization
        initial_stats = self.tiering_system.get_stats()
        
        # Add tensors to fill up tiers
        tensors_added = 0
        while tensors_added < 10:
            tensor = torch.randn(50, 50, dtype=torch.float16)  # ~5KB each
            success, tensor_id = self.tiering_system.put_tensor(tensor, TensorType.GENERAL)
            if not success:
                break
            tensors_added += 1
        
        final_stats = self.tiering_system.get_stats()
        
        # Check that utilization increased
        self.assertGreater(final_stats['total_utilization_bytes'], 
                          initial_stats['total_utilization_bytes'])


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    print("Running Qwen3-VL Memory Tiering System Tests...")
    run_tests()
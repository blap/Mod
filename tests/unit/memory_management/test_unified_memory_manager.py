
import unittest
import torch
import numpy as np
import sys
import os

# Adjust path to find the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.qwen3_vl.memory_management.unified_memory_manager import (
    UnifiedMemoryManager, UnifiedTensorType, create_unified_memory_manager
)

class TestUnifiedMemoryManager(unittest.TestCase):
    def setUp(self):
        self.unified_manager = create_unified_memory_manager({
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
            'storage_type': 'nvme'
        })

    def test_allocation(self):
        print("\n1. Testing unified allocation...")
        tensor_block = self.unified_manager.allocate(
            UnifiedTensorType.KV_CACHE,
            10 * 1024 * 1024,  # 10MB
            tensor_id="test_kv_tensor",
            use_compression=True,
            use_tiering=True,
            use_swapping=True
        )

        self.assertIsNotNone(tensor_block)
        self.assertEqual(tensor_block.tensor_id, "test_kv_tensor")
        self.assertEqual(tensor_block.size_bytes, 10 * 1024 * 1024)
        print(f"  Successfully allocated tensor: {tensor_block.tensor_id}")

    def test_tensor_access(self):
        print("\n2. Testing tensor access...")
        # First allocate
        self.unified_manager.allocate(
            UnifiedTensorType.KV_CACHE,
            10 * 1024 * 1024,
            tensor_id="test_access_tensor"
        )

        tensor = self.unified_manager.access_tensor("test_access_tensor")
        self.assertIsNotNone(tensor)
        print(f"  Successfully accessed tensor, shape: {tensor.shape}")

    def test_conflict_resolution(self):
        print("\n3. Testing conflict resolution...")
        self.unified_manager.allocate(
            UnifiedTensorType.KV_CACHE,
            10 * 1024 * 1024,
            tensor_id="test_conflict_tensor"
        )

        resolution = self.unified_manager.resolve_conflicts("test_conflict_tensor")
        self.assertIn('conflict_resolution', resolution)
        print(f"  Resolution: {resolution['conflict_resolution']}")

    def test_system_stats(self):
        print("\n4. Testing system statistics...")
        stats = self.unified_manager.get_system_stats()
        self.assertIn('unified_stats', stats)
        print(f"  Total allocations: {stats['unified_stats']['total_allocations']}")

    def test_tensor_stats(self):
        print("\n5. Testing tensor-specific statistics...")
        self.unified_manager.allocate(
            UnifiedTensorType.KV_CACHE,
            1024,
            tensor_id="test_stats_tensor"
        )

        tensor_stats = self.unified_manager.get_tensor_stats("test_stats_tensor")
        self.assertIsNotNone(tensor_stats)
        print(f"  Tensor type: {tensor_stats['unified_block']['tensor_type']}")

    def test_deallocation(self):
        print("\n6. Testing deallocation...")
        self.unified_manager.allocate(
            UnifiedTensorType.KV_CACHE,
            1024,
            tensor_id="test_dealloc_tensor"
        )

        success = self.unified_manager.deallocate("test_dealloc_tensor")
        self.assertTrue(success)
        print(f"  Deallocation successful: {success}")

if __name__ == '__main__':
    unittest.main()

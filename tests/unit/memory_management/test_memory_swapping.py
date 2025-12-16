
import unittest
import torch
import numpy as np
import sys
import os

# Adjust path to find the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.qwen3_vl.memory_management.memory_swapping import (
    AdvancedMemorySwapper, create_advanced_memory_swapper, MemoryRegionType
)

class TestMemorySwapper(unittest.TestCase):
    def setUp(self):
        self.swapper = create_advanced_memory_swapper({
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
            'storage_type': 'nvme'
        })

    def tearDown(self):
        self.swapper.cleanup()

    def test_registration(self):
        print("\n1. Registering memory blocks...")
        for i in range(5):
            block_id = f"block_{i}"
            size = (i + 1) * 10 * 1024 * 1024
            self.swapper.register_memory_block(
                block_id,
                size,
                MemoryRegionType.TENSOR_DATA,
                pinned=(i == 0)
            )

        status = self.swapper.get_status()
        self.assertEqual(status['total_registered_blocks'], 5)
        print(f"  Registered 5 blocks successfully")

    def test_access(self):
        print("\n2. Simulating block accesses...")
        # Register a block first
        self.swapper.register_memory_block(
            "test_block",
            1024,
            MemoryRegionType.TENSOR_DATA
        )

        block = self.swapper.access_memory_block("test_block")
        self.assertIsNotNone(block)
        print(f"  Accessed test_block")

    def test_swapping_trigger(self):
        print("\n3. Checking swapping trigger...")
        should_swap = self.swapper.should_swap()
        # Might be false depending on system state, but just checking it runs
        print(f"  Should swap: {should_swap}")

    def test_efficiency_metrics(self):
        print("\n6. Efficiency metrics...")
        efficiency = self.swapper.get_swapping_efficiency()
        self.assertIn('hit_rate', efficiency)
        print(f"  Cache hit rate: {efficiency['hit_rate']:.3f}")

if __name__ == '__main__':
    unittest.main()

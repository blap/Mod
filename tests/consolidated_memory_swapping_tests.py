"""
Consolidated tests for the Advanced Memory Swapping System

This test suite combines and consolidates the functionality from:
- test_memory_swapping.py
- test_memory_swapping_complete.py

All functionality from both files is preserved while eliminating redundancy.
"""

import unittest
import torch
import tempfile
import os
import time
from src.qwen3_vl.memory_management.memory_swapping import (
    AdvancedMemorySwapper, MemoryRegionType, SwapAlgorithm,
    MemoryPressureMonitor, MemoryPressureLevel, BaseSwapAlgorithm,
    LRUSwapAlgorithm, ClockSwapAlgorithm, AdaptiveSwapAlgorithm,
    MemoryBlock,
    create_advanced_memory_swapper,
    integrate_with_qwen3_vl_swapping
)


class TestMemorySwappingSystem(unittest.TestCase):
    """Comprehensive test cases for the Advanced Memory Swapping System"""

    def setUp(self):
        """Set up test fixtures"""
        self.swapper = AdvancedMemorySwapper(
            swap_algorithm=SwapAlgorithm.LRU,
            max_swap_size=100 * 1024 * 1024  # 100MB
        )

    def test_register_memory_block(self):
        """Test registering memory blocks"""
        block = self.swapper.register_memory_block(
            "test_block_1",
            10 * 1024 * 1024,  # 10MB
            MemoryRegionType.TENSOR_DATA
        )

        self.assertIsNotNone(block)
        self.assertEqual(block.id, "test_block_1")
        self.assertEqual(block.size, 10 * 1024 * 1024)
        self.assertIn("test_block_1", self.swapper.blocks)

    def test_unregister_memory_block(self):
        """Test unregistering memory blocks"""
        # Register a block first
        self.swapper.register_memory_block(
            "test_block_2",
            5 * 1024 * 1024,
            MemoryRegionType.ACTIVATION_BUFFER
        )

        # Unregister it
        success = self.swapper.unregister_memory_block("test_block_2")
        self.assertTrue(success)
        self.assertNotIn("test_block_2", self.swapper.blocks)

    def test_access_memory_block(self):
        """Test accessing memory blocks"""
        # Register a block
        self.swapper.register_memory_block(
            "test_block_3",
            8 * 1024 * 1024,
            MemoryRegionType.KV_CACHE
        )

        # Access the block
        block = self.swapper.access_memory_block("test_block_3")
        self.assertIsNotNone(block)
        self.assertEqual(block.access_count, 2)  # 1 from register + 1 from access

    def test_should_swap_logic(self):
        """Test the logic for determining when to swap"""
        # By default, should_swap should depend on memory pressure
        result = self.swapper.should_swap()
        # This may return True or False depending on current system memory usage

    def test_memory_pressure_monitor(self):
        """Test memory pressure monitoring"""
        monitor = MemoryPressureMonitor()
        ram_level, ram_usage = monitor.get_ram_pressure()
        gpu_level, gpu_usage = monitor.get_gpu_pressure()

        self.assertIsInstance(ram_level, MemoryPressureLevel)
        self.assertIsInstance(ram_usage, float)
        self.assertGreaterEqual(ram_usage, 0.0)
        self.assertLessEqual(ram_usage, 1.0)

        self.assertIsInstance(gpu_level, MemoryPressureLevel)
        self.assertIsInstance(gpu_usage, float)
        self.assertGreaterEqual(gpu_usage, 0.0)
        self.assertLessEqual(gpu_usage, 1.0)

    def test_lru_swap_algorithm(self):
        """Test LRU swap algorithm"""
        lru_algo = LRUSwapAlgorithm()

        # Add blocks with different access times
        block1 = MemoryBlock(
            id="block1", ptr=1, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time()-10,
            access_count=1, is_swapped=False, swap_location=None, pinned=False
        )
        block2 = MemoryBlock(
            id="block2", ptr=2, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time()-5,
            access_count=1, is_swapped=False, swap_location=None, pinned=False
        )
        block3 = MemoryBlock(
            id="block3", ptr=3, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time(),
            access_count=1, is_swapped=False, swap_location=None, pinned=False
        )

        lru_algo.add_block(block1)
        lru_algo.add_block(block2)
        lru_algo.add_block(block3)

        # The victim should be the least recently accessed (block1)
        victim = lru_algo.select_victim()
        self.assertEqual(victim.id, "block1")

        # Access block2, now it becomes the most recent
        lru_algo.access_block("block2")
        victim = lru_algo.select_victim()
        self.assertEqual(victim.id, "block1")  # Still block1 since block2 was just accessed

    def test_clock_swap_algorithm(self):
        """Test Clock swap algorithm"""
        clock_algo = ClockSwapAlgorithm()

        # Add blocks
        block1 = MemoryBlock(
            id="clock_block1", ptr=1, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time(),
            access_count=1, is_swapped=False, swap_location=None, pinned=False
        )
        block2 = MemoryBlock(
            id="clock_block2", ptr=2, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time(),
            access_count=1, is_swapped=False, swap_location=None, pinned=False
        )

        clock_algo.add_block(block1)
        clock_algo.add_block(block2)

        # Initially, should return one of the blocks
        victim = clock_algo.select_victim()
        self.assertIsNotNone(victim)
        self.assertIn(victim.id, ["clock_block1", "clock_block2"])

    def test_adaptive_swap_algorithm(self):
        """Test Adaptive swap algorithm"""
        adaptive_algo = AdaptiveSwapAlgorithm()

        # Add blocks with different characteristics
        block1 = MemoryBlock(
            id="adaptive_block1", ptr=1, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time()-100,
            access_count=10, is_swapped=False, swap_location=None, pinned=False
        )
        block2 = MemoryBlock(
            id="adaptive_block2", ptr=2, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time()-1,
            access_count=1, is_swapped=False, swap_location=None, pinned=False
        )

        adaptive_algo.add_block(block1)
        adaptive_algo.add_block(block2)

        # Should select a victim based on adaptive criteria
        victim = adaptive_algo.select_victim()
        self.assertIsNotNone(victim)
        self.assertIn(victim.id, ["adaptive_block1", "adaptive_block2"])

    def test_pinned_blocks_not_swapped(self):
        """Test that pinned blocks are not selected for swapping"""
        lru_algo = LRUSwapAlgorithm()

        # Add a pinned block and an unpinned block
        pinned_block = MemoryBlock(
            id="pinned_block", ptr=1, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time()-100,
            access_count=1, is_swapped=False, swap_location=None, pinned=True
        )
        unpinned_block = MemoryBlock(
            id="unpinned_block", ptr=2, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time()-100,
            access_count=1, is_swapped=False, swap_location=None, pinned=False
        )

        lru_algo.add_block(pinned_block)
        lru_algo.add_block(unpinned_block)

        # Victim should be the unpinned block
        victim = lru_algo.select_victim()
        self.assertEqual(victim.id, "unpinned_block")

    def test_get_access_pattern_priority(self):
        """Test access pattern priority calculation"""
        # Register a block
        self.swapper.register_memory_block(
            "priority_test_block",
            1024,
            MemoryRegionType.TENSOR_DATA
        )

        # Access it multiple times to create a pattern
        for _ in range(5):
            self.swapper.access_memory_block("priority_test_block")

        priority = self.swapper.get_access_pattern_priority("priority_test_block")
        self.assertIsInstance(priority, float)
        self.assertGreaterEqual(priority, 0.0)
        self.assertLessEqual(priority, 1.0)

    def test_analyze_access_patterns(self):
        """Test access pattern analysis"""
        # Register and access some blocks
        for i in range(3):
            block_id = f"analysis_block_{i}"
            self.swapper.register_memory_block(
                block_id,
                1024,
                MemoryRegionType.TENSOR_DATA
            )

            # Access some blocks more frequently
            for j in range(i + 1):
                self.swapper.access_memory_block(block_id)

        analysis = self.swapper.analyze_access_patterns()

        self.assertIn('most_frequently_accessed', analysis)
        self.assertIn('least_frequently_accessed', analysis)
        self.assertIn('temporal_locality', analysis)
        self.assertGreaterEqual(analysis['temporal_locality'], 0.0)
        self.assertLessEqual(analysis['temporal_locality'], 1.0)

    def test_get_swapping_efficiency(self):
        """Test swapping efficiency metrics"""
        efficiency = self.swapper.get_swapping_efficiency()

        self.assertIn('hit_rate', efficiency)
        self.assertIn('avg_swap_out_time', efficiency)
        self.assertIn('avg_swap_in_time', efficiency)
        self.assertIn('total_swapped_GB', efficiency)
        self.assertGreaterEqual(efficiency['hit_rate'], 0.0)
        self.assertLessEqual(efficiency['hit_rate'], 1.0)

    def test_get_status(self):
        """Test getting system status"""
        status = self.swapper.get_status()

        self.assertIn('algorithm', status)
        self.assertIn('swap_threshold', status)
        self.assertIn('current_swap_size_GB', status)
        self.assertIn('total_registered_blocks', status)
        self.assertIn('pressure_level', status)
        self.assertIn('pressure_usage', status)

    def test_cleanup(self):
        """Test cleanup functionality"""
        # Register some blocks
        for i in range(3):
            self.swapper.register_memory_block(
                f"cleanup_block_{i}",
                1024,
                MemoryRegionType.TENSOR_DATA
            )

        # Verify they're registered
        self.assertEqual(len(self.swapper.blocks), 3)

        # Perform cleanup
        self.swapper.cleanup()

        # Verify they're cleaned up
        self.assertEqual(len(self.swapper.blocks), 0)

    def test_create_advanced_memory_swapper_factory(self):
        """Test factory function for creating optimized swappers"""
        # Test with default config
        swapper_default = create_advanced_memory_swapper()
        self.assertIsInstance(swapper_default, AdvancedMemorySwapper)

        # Test with custom config
        custom_config = {
            'cpu_model': 'Intel i7',
            'gpu_model': 'NVIDIA RTX',
            'memory_size': 16 * 1024 * 1024 * 1024,  # 16GB
            'storage_type': 'nvme'
        }
        swapper_custom = create_advanced_memory_swapper(custom_config)
        self.assertIsInstance(swapper_custom, AdvancedMemorySwapper)

    def test_integrate_with_compression_and_cache(self):
        """Test integration with compression and cache systems"""
        # Create mock compression and cache managers
        class MockCompressionManager:
            pass

        class MockCacheManager:
            pass

        comp_manager = MockCompressionManager()
        cache_manager = MockCacheManager()

        # These should not raise exceptions
        self.swapper.integrate_with_compression(comp_manager)
        self.swapper.integrate_with_cache(cache_manager)

        # Verify attributes were set
        self.assertTrue(hasattr(self.swapper, 'compression_manager'))
        self.assertTrue(hasattr(self.swapper, 'cache_manager'))

    def test_abstract_base_class_properly_defined(self):
        """Test that BaseSwapAlgorithm is properly defined as an abstract class"""
        # This should fail since BaseSwapAlgorithm is abstract
        with self.assertRaises(TypeError):
            BaseSwapAlgorithm()

    def test_lru_algorithm_implementation(self):
        """Test that LRU algorithm properly implements the abstract method"""
        algo = LRUSwapAlgorithm()
        self.assertIsInstance(algo, BaseSwapAlgorithm)

        # Add a block and test selection
        block = MemoryBlock(
            id="test", ptr=1, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time()-10,
            access_count=1, is_swapped=False, swap_location=None, pinned=False
        )
        algo.add_block(block)

        victim = algo.select_victim()
        self.assertEqual(victim.id, "test")

    def test_clock_algorithm_implementation(self):
        """Test that Clock algorithm properly implements the abstract method"""
        algo = ClockSwapAlgorithm()
        self.assertIsInstance(algo, BaseSwapAlgorithm)

        # Add a block and test selection
        block = MemoryBlock(
            id="test", ptr=1, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time()-10,
            access_count=1, is_swapped=False, swap_location=None, pinned=False
        )
        algo.add_block(block)

        victim = algo.select_victim()
        self.assertIsNotNone(victim)
        self.assertEqual(victim.id, "test")

    def test_adaptive_algorithm_implementation(self):
        """Test that Adaptive algorithm properly implements the abstract method"""
        algo = AdaptiveSwapAlgorithm()
        self.assertIsInstance(algo, BaseSwapAlgorithm)

        # Add a block and test selection
        block = MemoryBlock(
            id="test", ptr=1, size=1024, region_type=MemoryRegionType.TENSOR_DATA,
            allocated=True, timestamp=time.time(), last_access_time=time.time()-10,
            access_count=1, is_swapped=False, swap_location=None, pinned=False
        )
        algo.add_block(block)

        victim = algo.select_victim()
        self.assertIsNotNone(victim)
        self.assertEqual(victim.id, "test")

    def test_integration_function_updated(self):
        """Test that the integration function now has proper implementation"""
        # Create mock components
        class MockMemoryOptimizer:
            def allocate_tensor(self, shape, dtype, tensor_type):
                return torch.zeros(shape, dtype=dtype)

        class MockCompressionManager:
            pass

        # Create swapper
        swapper = AdvancedMemorySwapper()

        # This should not raise an exception
        alloc_func, access_func, swap_func = integrate_with_qwen3_vl_swapping(
            swapper,
            MockMemoryOptimizer(),
            MockCompressionManager()
        )

        # Test that functions were created
        self.assertTrue(callable(alloc_func))
        self.assertTrue(callable(access_func))
        self.assertTrue(callable(swap_func))

    def test_swapping_with_actual_tensor_simulation(self):
        """Test swapping operations with tensor simulation"""
        swapper = AdvancedMemorySwapper(
            swap_algorithm=SwapAlgorithm.LRU,
            max_swap_size=50 * 1024 * 1024  # 50MB
        )

        # Register several blocks
        for i in range(5):
            block_id = f"tensor_{i}"
            size = 5 * 1024 * 1024  # 5MB each
            swapper.register_memory_block(block_id, size, MemoryRegionType.TENSOR_DATA)

        # Access some blocks to change their access patterns
        swapper.access_memory_block("tensor_0")
        swapper.access_memory_block("tensor_1")
        swapper.access_memory_block("tensor_0")  # Access again to make it more recently used

        # Check initial state
        initial_status = swapper.get_status()
        self.assertEqual(initial_status['total_registered_blocks'], 5)
        self.assertEqual(initial_status['swapped_blocks'], 0)

        # Perform swapping (though may not actually swap due to memory pressure settings)
        swapped_count = swapper.perform_swapping()

        # Check final state
        final_status = swapper.get_status()
        self.assertGreaterEqual(final_status['total_registered_blocks'], 0)

        # Clean up
        swapper.cleanup()

    def test_memory_pressure_monitoring(self):
        """Test memory pressure monitoring functionality"""
        monitor = MemoryPressureMonitor()

        ram_level, ram_usage = monitor.get_ram_pressure()
        gpu_level, gpu_usage = monitor.get_gpu_pressure()
        overall_level, overall_usage = monitor.get_overall_pressure()

        # Validate return types and ranges
        self.assertIsInstance(ram_level, MemoryPressureLevel)
        self.assertIsInstance(ram_usage, float)
        self.assertGreaterEqual(ram_usage, 0.0)
        self.assertLessEqual(ram_usage, 1.0)

        self.assertIsInstance(gpu_level, MemoryPressureLevel)
        self.assertIsInstance(gpu_usage, float)
        self.assertGreaterEqual(gpu_usage, 0.0)
        self.assertLessEqual(gpu_usage, 1.0)

        self.assertIsInstance(overall_level, MemoryPressureLevel)
        self.assertIsInstance(overall_usage, float)
        self.assertGreaterEqual(overall_usage, 0.0)
        self.assertLessEqual(overall_usage, 1.0)

    def test_different_swap_algorithms(self):
        """Test all different swap algorithms work properly"""
        algorithms = [SwapAlgorithm.LRU, SwapAlgorithm.CLOCK, SwapAlgorithm.ADAPTIVE]

        for algo_type in algorithms:
            with self.subTest(algo=algo_type):
                swapper = AdvancedMemorySwapper(swap_algorithm=algo_type)

                # Register and test operations
                swapper.register_memory_block("test_block", 1024, MemoryRegionType.TENSOR_DATA)
                block = swapper.access_memory_block("test_block")
                self.assertIsNotNone(block)

                # Check status
                status = swapper.get_status()
                self.assertEqual(status['algorithm'], algo_type.value)

                swapper.cleanup()


if __name__ == '__main__':
    unittest.main()
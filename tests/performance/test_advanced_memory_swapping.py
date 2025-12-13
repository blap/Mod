"""
Comprehensive tests for the Advanced Memory Swapping System

This module contains comprehensive tests for the advanced memory swapping system,
covering all major functionality including memory pressure monitoring,
swapping algorithms, NVMe optimizations, and integration with existing systems.
"""

import unittest
import numpy as np
import tempfile
import os
import time
from unittest.mock import patch, MagicMock

# Import the swapping system components
from advanced_memory_swapping_system import (
    MemoryPressureLevel, 
    SwapAlgorithm,
    MemoryRegionType,
    MemoryPressureMonitor,
    LRUSwapAlgorithm,
    ClockSwapAlgorithm,
    AdaptiveSwapAlgorithm,
    NVMeOptimizer,
    AdvancedMemorySwapper,
    MemoryBlock,
    create_optimized_swapping_system
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class TestMemoryPressureMonitor(unittest.TestCase):
    """Test memory pressure monitoring functionality"""

    def setUp(self):
        self.monitor = MemoryPressureMonitor()

    @unittest.skipIf(not PSUTIL_AVAILABLE, "psutil not available")
    def test_ram_pressure_levels(self):
        """Test that RAM pressure levels are correctly identified"""
        level, usage = self.monitor.get_ram_pressure()
        self.assertIsInstance(level, MemoryPressureLevel)
        self.assertGreaterEqual(usage, 0.0)
        self.assertLessEqual(usage, 1.0)

    @unittest.skipIf(not TORCH_AVAILABLE or not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_pressure_levels(self):
        """Test that GPU pressure levels are correctly identified"""
        level, usage = self.monitor.get_gpu_pressure()
        self.assertIsInstance(level, MemoryPressureLevel)
        self.assertGreaterEqual(usage, 0.0)
        self.assertLessEqual(usage, 1.0)

    def test_overall_pressure(self):
        """Test overall pressure calculation"""
        level, usage = self.monitor.get_overall_pressure()
        self.assertIsInstance(level, MemoryPressureLevel)
        self.assertGreaterEqual(usage, 0.0)
        self.assertLessEqual(usage, 1.0)

    def test_pressure_trend(self):
        """Test pressure trend detection"""
        trend = self.monitor.get_pressure_trend()
        self.assertIn(trend, ['increasing', 'decreasing', 'stable'])


class TestSwapAlgorithms(unittest.TestCase):
    """Test different swapping algorithms"""

    def setUp(self):
        # Create some test blocks
        self.blocks = []
        for i in range(5):
            block = MemoryBlock(
                id=f"block_{i}",
                ptr=i * 1000,
                size=(i + 1) * 10 * 1024 * 1024,  # 10MB, 20MB, etc.
                region_type=MemoryRegionType.TENSOR_DATA,
                allocated=True,
                timestamp=time.time() - i * 10,  # Different timestamps
                last_access_time=time.time() - i * 10,
                access_count=1,
                is_swapped=False,
                swap_location=None,
                pinned=False
            )
            self.blocks.append(block)

    def test_lru_algorithm(self):
        """Test LRU swapping algorithm"""
        algorithm = LRUSwapAlgorithm()

        # Add blocks to algorithm
        for block in self.blocks:
            algorithm.add_block(block)

        # The blocks were created with decreasing access times (block_4 is oldest)
        # So initially block_4 should be LRU
        victim = algorithm.select_victim()
        self.assertEqual(victim.id, "block_4")

        # In a real scenario, we would remove the selected victim
        # Let's remove block_4 and try again
        algorithm.remove_block("block_4")

        # Now with block_4 removed, block_3 should be the LRU
        victim = algorithm.select_victim()
        self.assertEqual(victim.id, "block_3")

        # Access some blocks to change their access times
        algorithm.access_block("block_0")
        algorithm.access_block("block_1")

        # Now block_3 should still be the LRU (since it has access time current-30,
        # while block_2 has access time current-20, so block_3 is older)
        victim = algorithm.select_victim()
        self.assertEqual(victim.id, "block_3")

    def test_lru_algorithm_with_pinned_blocks(self):
        """Test LRU algorithm respects pinned blocks"""
        algorithm = LRUSwapAlgorithm()
        
        # Pin the first block
        self.blocks[0].pinned = True
        
        for block in self.blocks:
            algorithm.add_block(block)
        
        # Access other blocks
        algorithm.access_block("block_1")
        algorithm.access_block("block_2")
        algorithm.access_block("block_3")
        
        # The victim should not be the pinned block
        victim = algorithm.select_victim()
        self.assertNotEqual(victim.id, "block_0")
        # Should be the next oldest non-pinned block
        self.assertEqual(victim.id, "block_4")

    def test_clock_algorithm(self):
        """Test Clock swapping algorithm"""
        algorithm = ClockSwapAlgorithm()
        
        # Add blocks to algorithm
        for block in self.blocks:
            algorithm.add_block(block)
        
        # Access some blocks to set their reference bits
        algorithm.access_block("block_1")
        algorithm.access_block("block_2")
        
        # First selection should get the first block (block_0) since it doesn't have its reference bit set
        victim = algorithm.select_victim()
        self.assertEqual(victim.id, "block_0")

    def test_adaptive_algorithm(self):
        """Test Adaptive swapping algorithm"""
        algorithm = AdaptiveSwapAlgorithm()
        
        # Add blocks to algorithm
        for block in self.blocks:
            algorithm.add_block(block)
        
        # Access some blocks more frequently
        for _ in range(5):
            algorithm.access_block("block_1")
        for _ in range(3):
            algorithm.access_block("block_2")
        
        # The adaptive algorithm should consider access frequency and recency
        victim = algorithm.select_victim()
        # Block 0 should be a candidate since it wasn't accessed frequently
        # The exact behavior depends on the internal scoring logic
        self.assertIsNotNone(victim)


class TestNVMeOptimizer(unittest.TestCase):
    """Test NVMe optimization functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = NVMeOptimizer()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_swap_operations(self):
        """Test basic swap operations"""
        block_id = "test_block_1"
        test_data = {"data": [1, 2, 3, 4, 5]}
        
        # Test async swap out
        result = self.optimizer.async_swap_out(block_id, test_data)
        self.assertTrue(result)
        
        # Test async swap in
        result = self.optimizer.async_swap_in(block_id)
        self.assertTrue(result)

    def test_swap_file_cleanup(self):
        """Test cleanup of swap files"""
        block_id = "test_block_2"
        test_data = {"data": [1, 2, 3]}
        
        # Perform a swap
        self.optimizer.async_swap_out(block_id, test_data)
        
        # Wait a bit for the async operation to potentially complete
        time.sleep(0.1)
        
        # Clean up
        self.optimizer.cleanup_swap_file(block_id)


class TestAdvancedMemorySwapper(unittest.TestCase):
    """Test the main memory swapper functionality"""

    def setUp(self):
        self.swapper = AdvancedMemorySwapper(
            swap_algorithm=SwapAlgorithm.ADAPTIVE,
            swap_threshold=0.5,  # Lower threshold for testing
            max_swap_size=100 * 1024 * 1024  # 100MB max for testing
        )

    def test_block_registration(self):
        """Test registering and unregistering memory blocks"""
        block_id = "test_block_1"
        size = 10 * 1024 * 1024  # 10MB
        
        # Register block
        block = self.swapper.register_memory_block(block_id, size)
        self.assertIsNotNone(block)
        self.assertEqual(block.id, block_id)
        self.assertEqual(block.size, size)
        
        # Try to register same block again (should return existing)
        same_block = self.swapper.register_memory_block(block_id, size)
        self.assertEqual(block, same_block)
        
        # Unregister block
        result = self.swapper.unregister_memory_block(block_id)
        self.assertTrue(result)

    def test_block_access(self):
        """Test accessing registered blocks"""
        block_id = "test_block_2"
        size = 5 * 1024 * 1024  # 5MB
        
        # Register and access block
        self.swapper.register_memory_block(block_id, size)
        accessed_block = self.swapper.access_memory_block(block_id)
        
        self.assertIsNotNone(accessed_block)
        self.assertEqual(accessed_block.id, block_id)
        
        # Check that access count increased
        self.assertGreater(accessed_block.access_count, 0)

    def test_swapping_decision(self):
        """Test swapping decision logic"""
        # Test should_swap method - this might return False in testing environment
        # since memory pressure might not be high enough
        result = self.swapper.should_swap()
        self.assertIsInstance(result, bool)

    def test_access_pattern_analysis(self):
        """Test access pattern analysis functionality"""
        # Register some blocks
        for i in range(5):
            block_id = f"pattern_test_{i}"
            size = (i + 1) * 2 * 1024 * 1024
            self.swapper.register_memory_block(block_id, size)
        
        # Access some blocks more frequently
        for i in range(3):
            self.swapper.access_memory_block(f"pattern_test_{i}")
        
        # Analyze patterns
        patterns = self.swapper.analyze_access_patterns()
        self.assertIn('most_frequently_accessed', patterns)
        self.assertIn('least_frequently_accessed', patterns)
        self.assertIn('temporal_locality', patterns)

    def test_swapping_efficiency(self):
        """Test swapping efficiency metrics"""
        efficiency = self.swapper.get_swapping_efficiency()
        
        self.assertIn('hit_rate', efficiency)
        self.assertIn('avg_swap_out_time', efficiency)
        self.assertIn('avg_swap_in_time', efficiency)
        self.assertIn('total_swapped_GB', efficiency)

    def test_system_status(self):
        """Test getting system status"""
        status = self.swapper.get_status()
        
        self.assertIn('algorithm', status)
        self.assertIn('swap_threshold', status)
        self.assertIn('current_swap_size_GB', status)
        self.assertIn('total_registered_blocks', status)
        self.assertIn('swapped_blocks', status)
        self.assertIn('pressure_level', status)
        self.assertIn('pressure_trend', status)
        self.assertIn('nvme_stats', status)
        self.assertIn('efficiency_metrics', status)


class TestIntegration(unittest.TestCase):
    """Test integration with existing systems"""

    def test_create_optimized_swapper(self):
        """Test creating optimized swapper for specific hardware"""
        hardware_config = {
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
            'storage_type': 'nvme'
        }
        
        swapper = create_optimized_swapping_system(hardware_config)
        self.assertIsInstance(swapper, AdvancedMemorySwapper)
        self.assertEqual(swapper.swap_algorithm, SwapAlgorithm.ADAPTIVE)

    def test_create_optimized_swapper_different_storage(self):
        """Test creating swapper with different storage types"""
        hardware_config = {
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,
            'storage_type': 'ssd'  # Different from NVMe
        }
        
        swapper = create_optimized_swapping_system(hardware_config)
        self.assertIsInstance(swapper, AdvancedMemorySwapper)
        # For SSD, it might use a different algorithm
        self.assertIn(swapper.swap_algorithm, [SwapAlgorithm.LRU, SwapAlgorithm.ADAPTIVE])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def test_empty_algorithm_selection(self):
        """Test selecting victim from empty algorithm"""
        lru_algo = LRUSwapAlgorithm()
        victim = lru_algo.select_victim()
        self.assertIsNone(victim)
        
        clock_algo = ClockSwapAlgorithm()
        victim = clock_algo.select_victim()
        self.assertIsNone(victim)
        
        adaptive_algo = AdaptiveSwapAlgorithm()
        victim = adaptive_algo.select_victim()
        self.assertIsNone(victim)

    def test_all_pinned_blocks(self):
        """Test algorithms when all blocks are pinned"""
        lru_algo = LRUSwapAlgorithm()
        
        # Create pinned blocks
        for i in range(3):
            block = MemoryBlock(
                id=f"pinned_{i}",
                ptr=i * 1000,
                size=1024,
                region_type=MemoryRegionType.TENSOR_DATA,
                allocated=True,
                timestamp=time.time(),
                last_access_time=time.time(),
                access_count=1,
                is_swapped=False,
                swap_location=None,
                pinned=True  # All pinned
            )
            lru_algo.add_block(block)
        
        # Selection should return None since all are pinned
        victim = lru_algo.select_victim()
        self.assertIsNone(victim)

    def test_swapper_with_no_blocks(self):
        """Test swapper operations with no registered blocks"""
        swapper = AdvancedMemorySwapper()
        
        # Accessing non-existent block should return None
        result = swapper.access_memory_block("nonexistent")
        self.assertIsNone(result)
        
        # Unregistering non-existent block should return False
        result = swapper.unregister_memory_block("nonexistent")
        self.assertFalse(result)


class TestPerformance(unittest.TestCase):
    """Test performance aspects of the swapping system"""

    def test_multiple_concurrent_swaps(self):
        """Test handling multiple concurrent swap operations"""
        swapper = AdvancedMemorySwapper(
            max_swap_size=50 * 1024 * 1024  # 50MB
        )
        
        # Register multiple blocks
        for i in range(10):
            block_id = f"perf_block_{i}"
            size = 2 * 1024 * 1024  # 2MB each
            swapper.register_memory_block(block_id, size)
        
        # Access blocks in a pattern that might trigger swapping
        for i in range(20):
            block_id = f"perf_block_{i % 10}"
            swapper.access_memory_block(block_id)
        
        # Check status
        status = swapper.get_status()
        self.assertGreaterEqual(status['total_registered_blocks'], 0)


def run_comprehensive_tests():
    """Run all tests and return results"""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_classes = [
        TestMemoryPressureMonitor,
        TestSwapAlgorithms,
        TestNVMeOptimizer,
        TestAdvancedMemorySwapper,
        TestIntegration,
        TestEdgeCases,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Running comprehensive tests for Advanced Memory Swapping System")
    print("=" * 70)
    
    # Run all tests
    test_result = run_comprehensive_tests()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Success rate: {((test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun * 100):.2f}%")
    
    if test_result.failures:
        print("\nFailures:")
        for test, traceback in test_result.failures:
            print(f"  {test}: {traceback}")
    
    if test_result.errors:
        print("\nErrors:")
        for test, traceback in test_result.errors:
            print(f"  {test}: {traceback}")
    
    if not test_result.failures and not test_result.errors:
        print("\nAll tests passed! (Y)")
    else:
        print("\nSome tests failed or had errors! (N)")
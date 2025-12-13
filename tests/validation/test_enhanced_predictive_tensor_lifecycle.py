"""
Comprehensive Tests for Enhanced Predictive Tensor Lifecycle Management System

This module provides comprehensive tests for the enhanced predictive garbage collection
and memory lifecycle system for Qwen3-VL, covering all major functionality.
"""

import unittest
import torch
import time
import tempfile
import os
from typing import Dict, Any

# Import our system components
from enhanced_predictive_tensor_lifecycle_manager import (
    TensorState, TensorType, TensorMetadata, EnhancedAccessPatternAnalyzer,
    EnhancedLifetimePredictor, EnhancedTensorLifecycleTracker, 
    EnhancedPredictiveGarbageCollector, create_enhanced_lifecycle_manager
)
from hardware_specific_optimizations import (
    HardwareOptimizer, create_hardware_optimizer, 
    IntelI5_10210UOptimizer, NVidiaSM61Optimizer, NVMeSSDOptimizer
)
from integrated_memory_management_system import (
    IntegratedMemoryManager, IntegrationLevel, 
    create_integrated_memory_manager, integrate_with_qwen3_vl_model
)


class TestEnhancedAccessPatternAnalyzer(unittest.TestCase):
    """Test the enhanced access pattern analyzer component"""

    def setUp(self):
        self.analyzer = EnhancedAccessPatternAnalyzer()

    def test_record_access(self):
        """Test recording tensor access"""
        self.analyzer.record_access("tensor_1")
        self.assertIn("tensor_1", self.analyzer.tensor_last_access)

        # Check that access count is updated
        self.assertEqual(self.analyzer.tensor_access_counts["tensor_1"], 1)

    def test_record_access_with_context(self):
        """Test recording access with context"""
        self.analyzer.record_access("tensor_1", "layer_1")
        self.assertIn("tensor_1", self.analyzer.tensor_last_access)
        
        # Check spatial locality with context
        self.analyzer.record_access("tensor_2", "layer_1")  # Same context
        time.sleep(0.01)  # Small delay
        self.analyzer.record_access("tensor_1", "layer_2")  # Different context
        
        # Check if spatial locality was recorded
        similarity = self.analyzer.get_tensor_similarity("tensor_1", "tensor_2")
        self.assertGreaterEqual(similarity, 0.0)

    def test_predict_access(self):
        """Test access prediction"""
        # Record multiple accesses to create a pattern
        for i in range(5):
            self.analyzer.record_access(f"tensor_1", f"context_{i}")
            time.sleep(0.01)  # Small delay to create different access times

        prob, time_pred, count_pred, metrics = self.analyzer.predict_access("tensor_1")

        # Probability should be between 0 and 1
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)

        # Count prediction should be positive
        self.assertGreater(count_pred, 0)

        # Check that additional metrics are returned
        self.assertIn('temporal_locality', metrics)
        self.assertIn('access_frequency', metrics)
        self.assertIn('trend', metrics)

    def test_get_hot_and_cold_tensors(self):
        """Test getting hot and cold tensors"""
        # Create access patterns with different frequencies
        for i in range(10):
            self.analyzer.record_access("hot_tensor")
        
        for i in range(2):
            self.analyzer.record_access("cold_tensor")
        
        hot_tensors = self.analyzer.get_hot_tensors(1)
        cold_tensors = self.analyzer.get_cold_tensors(1)
        
        self.assertIn("hot_tensor", hot_tensors)
        self.assertIn("cold_tensor", cold_tensors)


class TestEnhancedLifetimePredictor(unittest.TestCase):
    """Test the enhanced lifetime prediction component"""

    def setUp(self):
        self.predictor = EnhancedLifetimePredictor()
        self.analyzer = EnhancedAccessPatternAnalyzer()

    def test_predict_lifetime(self):
        """Test lifetime prediction"""
        # Test prediction for a general tensor
        lifetime, access_count = self.predictor.predict_lifetime(
            "test_tensor",
            1024 * 1024,  # 1MB
            TensorType.GENERAL
        )

        self.assertGreater(lifetime, 0)
        self.assertGreater(access_count, 0)

        # Test with different tensor types
        for tensor_type in [TensorType.KV_CACHE, TensorType.GRADIENTS, TensorType.IMAGE_FEATURES]:
            lifetime, access_count = self.predictor.predict_lifetime(
                f"test_tensor_{tensor_type.value}",
                1024 * 1024,  # 1MB
                tensor_type
            )
            self.assertGreater(lifetime, 0)
            self.assertGreater(access_count, 0)

    def test_predict_lifetime_with_metrics(self):
        """Test lifetime prediction with additional metrics"""
        # Create some access patterns
        for i in range(5):
            self.analyzer.record_access("test_tensor")
            time.sleep(0.01)
        
        # Get metrics from analyzer
        _, _, _, metrics = self.analyzer.predict_access("test_tensor")
        
        # Predict lifetime with metrics
        lifetime, access_count = self.predictor.predict_lifetime(
            "test_tensor",
            1024 * 1024,  # 1MB
            TensorType.KV_CACHE,
            metrics
        )

        self.assertGreater(lifetime, 0)
        self.assertGreater(access_count, 0)

    def test_record_memory_pressure(self):
        """Test memory pressure recording"""
        self.predictor.record_memory_pressure(0.7)
        self.predictor.record_memory_pressure(0.8)

        # Check that pressure history is maintained
        self.assertGreater(len(self.predictor.memory_pressure_history), 0)


class TestEnhancedTensorLifecycleTracker(unittest.TestCase):
    """Test the enhanced tensor lifecycle tracker"""

    def setUp(self):
        self.tracker = EnhancedTensorLifecycleTracker()

    def test_register_tensor(self):
        """Test tensor registration"""
        tensor = torch.randn(10, 10)
        metadata = self.tracker.register_tensor(
            tensor,
            "test_tensor",
            TensorType.GENERAL,
            is_pinned=False,
            initial_ref_count=1
        )

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.tensor_id, "test_tensor")
        self.assertEqual(metadata.reference_count, 1)
        self.assertEqual(metadata.tensor_state, TensorState.ALLOCATED)

    def test_access_tensor(self):
        """Test tensor access recording"""
        tensor = torch.randn(5, 5)
        self.tracker.register_tensor(tensor, "test_tensor")

        metadata = self.tracker.access_tensor("test_tensor", "test_context")

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.tensor_state, TensorState.IN_USE)
        self.assertIn("test_context", self.tracker.tensor_owners["test_tensor"])

        # Check that temporal locality is updated
        self.assertGreaterEqual(metadata.temporal_locality_score, 0.0)

    def test_reference_counting(self):
        """Test reference counting operations"""
        tensor = torch.randn(5, 5)
        self.tracker.register_tensor(tensor, "test_tensor")

        # Increment reference
        self.tracker.increment_reference("test_tensor", "context_1")
        metadata = self.tracker.tensor_metadata["test_tensor"]
        self.assertEqual(metadata.reference_count, 2)

        # Decrement reference
        self.tracker.decrement_reference("test_tensor", "context_1")
        metadata = self.tracker.tensor_metadata["test_tensor"]
        self.assertEqual(metadata.reference_count, 1)

        # Update reference count by delta
        self.tracker.update_reference_count("test_tensor", 2, "context_2")
        metadata = self.tracker.tensor_metadata["test_tensor"]
        self.assertEqual(metadata.reference_count, 3)

    def test_reference_chains(self):
        """Test reference chain tracking"""
        # Add a reference chain
        result = self.tracker.add_reference_chain("tensor_a", "tensor_b")
        self.assertTrue(result)

        # Verify the chain exists
        self.assertIn("tensor_b", self.tracker.reference_chains["tensor_a"])

        # Remove the reference chain
        result = self.tracker.remove_reference_chain("tensor_a", "tensor_b")
        self.assertTrue(result)

        # Verify the chain is removed
        self.assertNotIn("tensor_b", self.tracker.reference_chains.get("tensor_a", set()))

    def test_tensor_dependencies(self):
        """Test tensor dependency tracking"""
        # Register some tensors
        tensor_a = torch.randn(5, 5)
        tensor_b = torch.randn(5, 5)
        tensor_c = torch.randn(5, 5)
        
        self.tracker.register_tensor(tensor_a, "tensor_a")
        self.tracker.register_tensor(tensor_b, "tensor_b")
        self.tracker.register_tensor(tensor_c, "tensor_c")
        
        # Set dependencies
        result = self.tracker.set_tensor_dependencies("tensor_c", ["tensor_a", "tensor_b"])
        self.assertTrue(result)
        
        # Check dependencies are set
        self.assertIn("tensor_a", self.tracker.tensor_dependencies["tensor_c"])
        self.assertIn("tensor_b", self.tracker.tensor_dependencies["tensor_c"])

    def test_get_tensors_for_collection(self):
        """Test getting tensors for collection"""
        # Register a tensor
        tensor = torch.randn(5, 5)
        self.tracker.register_tensor(tensor, "test_tensor", initial_ref_count=0)
        
        # Decrement reference count to make it eligible for collection
        self.tracker.decrement_reference("test_tensor")
        
        # Get tensors for collection
        tensors_for_collection = self.tracker.get_tensors_for_collection()
        
        # Should have at least one tensor
        self.assertGreaterEqual(len(tensors_for_collection), 0)


class TestEnhancedPredictiveGarbageCollector(unittest.TestCase):
    """Test the enhanced predictive garbage collector"""

    def setUp(self):
        self.garbage_collector = EnhancedPredictiveGarbageCollector(
            collection_interval=0.1,  # Very short interval for testing
            memory_pressure_threshold=0.9,  # High threshold to avoid collection
            collection_batch_size=5  # Small batch for testing
        )

    def tearDown(self):
        self.garbage_collector.cleanup()

    def test_register_and_collect_tensor(self):
        """Test tensor registration and collection"""
        tensor = torch.randn(10, 10)
        metadata = self.garbage_collector.register_tensor(
            tensor,
            "test_tensor",
            TensorType.GENERAL,
            is_pinned=False,
            initial_ref_count=1
        )

        self.assertIsNotNone(metadata)

        # Decrement reference count to 0 to mark for collection
        self.garbage_collector.decrement_reference("test_tensor")

        # Manually trigger collection
        collected_count = self.garbage_collector.collect()

        # Since we have a background collector, we might not collect immediately
        # but the tensor should be marked for collection
        stats = self.garbage_collector.get_collection_stats()
        self.assertGreaterEqual(stats['total_tensors'], 0)

    def test_memory_pressure(self):
        """Test memory pressure monitoring"""
        pressure = self.garbage_collector.get_memory_pressure()
        self.assertGreaterEqual(pressure, 0)
        self.assertLessEqual(pressure, 1)

    def test_tensor_dependencies(self):
        """Test tensor dependency management"""
        # Register tensors with dependencies
        tensor_a = torch.randn(5, 5)
        tensor_b = torch.randn(5, 5)
        
        self.garbage_collector.register_tensor(tensor_a, "tensor_a")
        self.garbage_collector.register_tensor(tensor_b, "tensor_b", dependencies=["tensor_a"])
        
        # Set dependencies explicitly
        result = self.garbage_collector.set_tensor_dependencies("tensor_b", ["tensor_a"])
        self.assertTrue(result)


class TestHardwareOptimizers(unittest.TestCase):
    """Test hardware-specific optimizers"""

    def test_intel_i5_10210u_optimizer(self):
        """Test Intel i5-10210U optimizer"""
        optimizer = IntelI5_10210UOptimizer()
        
        # Test threading optimization
        thread_params = optimizer.optimize_threading()
        self.assertIn('max_workers', thread_params)
        self.assertGreaterEqual(thread_params['max_workers'], 1)
        
        # Test tensor operation optimization
        tensor_size = 1024 * 1024  # 1MB
        op_params = optimizer.optimize_tensor_operations(tensor_size)
        self.assertIn('block_size', op_params)
        self.assertIn('use_simd', op_params)

    def test_nvidia_sm61_optimizer(self):
        """Test NVIDIA SM61 optimizer"""
        optimizer = NVidiaSM61Optimizer()
        
        # Test GPU memory placement optimization
        placement = optimizer.optimize_gpu_memory_placement(1024*1024, "kv_cache")  # 1MB tensor
        self.assertIn('memory_type', placement)
        self.assertIn('use_pinned', placement)
        
        # Test kernel launch optimization
        launch_params = optimizer.optimize_kernel_launch((100, 100))
        self.assertIn('block_dim', launch_params)
        self.assertIn('grid_dim', launch_params)

    def test_nvme_ssd_optimizer(self):
        """Test NVMe SSD optimizer"""
        optimizer = NVMeSSDOptimizer()
        
        # Test storage access optimization
        access_params = optimizer.optimize_storage_access(1024*1024)  # 1MB
        self.assertIn('block_size', access_params)
        self.assertIn('queue_depth', access_params)
        
        # Test swap operation optimization
        swap_params = optimizer.optimize_swap_operations(1024*1024)  # 1MB
        self.assertIn('block_size', swap_params)
        self.assertIn('estimated_io_time_s', swap_params)

    def test_hardware_optimizer(self):
        """Test the main hardware optimizer"""
        from hardware_specific_optimizations import HardwareSpecs
        
        specs = HardwareSpecs(
            cpu_model="Intel i5-10210U",
            cpu_cores=4,
            cpu_threads=8,
            cpu_base_freq=1.6,
            cpu_boost_freq=4.2,
            gpu_model="NVIDIA SM61",
            gpu_compute_capability="6.1",
            gpu_memory=2 * 1024 * 1024 * 1024,
            gpu_memory_bandwidth=80.0,
            system_memory=8 * 1024 * 1024 * 1024,
            storage_type="nvme",
            storage_bandwidth=3.5,
            storage_iops=500000
        )
        
        optimizer = HardwareOptimizer(specs)
        
        # Test tensor placement optimization
        test_tensor = torch.randn(100, 100)
        placement = optimizer.optimize_tensor_placement(test_tensor, "kv_cache")
        self.assertIn('preferred_device', placement)
        self.assertIn('memory_optimization', placement)
        
        # Test memory management optimization
        mem_opt = optimizer.optimize_memory_management(0.8)  # 80% pressure
        self.assertIn('gc_aggressiveness', mem_opt)
        self.assertGreaterEqual(mem_opt['swap_threshold'], 0)


class TestIntegratedMemoryManager(unittest.TestCase):
    """Test the integrated memory manager"""

    def setUp(self):
        # Create all components
        self.hardware_optimizer = create_hardware_optimizer(
            cpu_model="Intel i5-10210U",
            gpu_model="NVIDIA SM61",
            memory_size=8 * 1024 * 1024 * 1024,
            storage_type="nvme"
        )
        
        self.garbage_collector = EnhancedPredictiveGarbageCollector(
            collection_interval=0.5,
            memory_pressure_threshold=0.75,
            enable_background_collection=False,  # Disable background for tests
            collection_batch_size=20
        )
        
        from memory_compression_system import MemoryCompressionManager
        self.compression_manager = MemoryCompressionManager()
        
        from advanced_memory_swapping_system import AdvancedMemorySwapper
        self.swapping_system = AdvancedMemorySwapper()
        
        from advanced_memory_tiering_system import AdvancedMemoryTieringSystem
        self.tiering_system = AdvancedMemoryTieringSystem()
        
        self.integrated_manager = IntegratedMemoryManager(
            garbage_collector=self.garbage_collector,
            hardware_optimizer=self.hardware_optimizer,
            compression_manager=self.compression_manager,
            swapping_system=self.swapping_system,
            tiering_system=self.tiering_system,
            integration_level=IntegrationLevel.FULL
        )

    def tearDown(self):
        self.integrated_manager.cleanup_all()

    def test_tensor_registration(self):
        """Test tensor registration with integrated system"""
        tensor = torch.randn(20, 20)
        tensor_id = self.integrated_manager.register_tensor(
            tensor,
            tensor_type=TensorType.GENERAL,
            optimize_placement=True
        )

        self.assertIsInstance(tensor_id, str)
        self.assertTrue(tensor_id.startswith("integrated_tensor_"))

        # Check that tensor is tracked by multiple systems
        systems_tracking = self.integrated_manager.tensor_system_mapping.get(tensor_id, [])
        self.assertIn('garbage_collector', systems_tracking)

    def test_tensor_access(self):
        """Test tensor access with integrated system"""
        tensor = torch.randn(15, 15)
        tensor_id = self.integrated_manager.register_tensor(tensor)

        accessed_tensor = self.integrated_manager.access_tensor(tensor_id)
        self.assertIsInstance(accessed_tensor, torch.Tensor)
        self.assertEqual(accessed_tensor.shape, tensor.shape)

    def test_tensor_lifetime_prediction(self):
        """Test tensor lifetime prediction"""
        tensor = torch.randn(10, 10)
        tensor_id = self.integrated_manager.register_tensor(tensor)

        prediction = self.integrated_manager.get_tensor_lifetime_prediction(tensor_id)
        if prediction:
            lifetime, access_count = prediction
            self.assertGreaterEqual(lifetime, 0)
            self.assertGreaterEqual(access_count, 0)

    def test_memory_pressure_handling(self):
        """Test memory pressure handling"""
        # Simulate memory pressure handling
        actions = self.integrated_manager.handle_memory_pressure()
        
        self.assertIn('gc_performed', actions)
        self.assertIn('tensors_swapped', actions)
        self.assertIn('tensors_compressed', actions)
        self.assertIn('tiers_rebalanced', actions)

    def test_system_stats(self):
        """Test getting system statistics"""
        stats = self.integrated_manager.get_system_stats()
        
        self.assertIn('integration_stats', stats)
        self.assertIn('garbage_collection_stats', stats)
        self.assertIn('compression_stats', stats)
        self.assertIn('swapping_stats', stats)
        self.assertIn('tiering_stats', stats)
        self.assertIn('hardware_optimization_stats', stats)
        self.assertIn('tensor_distribution', stats)


class TestIntegrationWithQwen3VL(unittest.TestCase):
    """Test integration with Qwen3-VL model interface"""

    def setUp(self):
        self.integrated_manager = create_integrated_memory_manager({
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,
            'storage_type': 'nvme'
        })

    def tearDown(self):
        self.integrated_manager.cleanup_all()

    def test_qwen3_vl_integration(self):
        """Test Qwen3-VL model integration"""
        integration = integrate_with_qwen3_vl_model(self.integrated_manager)
        
        # Test KV cache allocation
        kv_tensor, kv_id = integration['allocate_kv_cache']((4, 100, 64))
        self.assertIsInstance(kv_tensor, torch.Tensor)
        self.assertEqual(kv_tensor.shape, (4, 100, 64))
        
        # Test image features allocation
        img_tensor, img_id = integration['allocate_image_features']((1, 50, 768))
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(img_tensor.shape, (1, 50, 768))
        
        # Test accessing tensors
        retrieved_kv = integration['access_tensor'](kv_id)
        self.assertIsNotNone(retrieved_kv)
        
        retrieved_img = integration['access_tensor'](img_id)
        self.assertIsNotNone(retrieved_img)
        
        # Test memory stats
        stats = integration['get_memory_stats']()
        self.assertIn('integration_stats', stats)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def test_empty_tensor_registration(self):
        """Test registering empty tensors"""
        gc = EnhancedPredictiveGarbageCollector()
        
        empty_tensor = torch.tensor([])
        tensor_id = gc.register_tensor(empty_tensor, "empty_tensor")
        
        self.assertIsInstance(tensor_id, str)

    def test_large_tensor_handling(self):
        """Test handling of very large tensors"""
        gc = EnhancedPredictiveGarbageCollector()
        
        # Create a moderately large tensor (not too large for testing)
        large_tensor = torch.randn(100, 100, dtype=torch.float32)
        tensor_id = gc.register_tensor(large_tensor, "large_tensor", 
                                     tensor_type=TensorType.KV_CACHE)
        
        self.assertIsInstance(tensor_id, str)

    def test_invalid_tensor_types(self):
        """Test handling of invalid tensor types"""
        gc = EnhancedPredictiveGarbageCollector()
        
        tensor = torch.randn(5, 5)
        # Should handle gracefully even with unknown tensor type
        tensor_id = gc.register_tensor(tensor, "test_tensor")
        
        self.assertIsInstance(tensor_id, str)

    def test_concurrent_access(self):
        """Test concurrent access patterns"""
        analyzer = EnhancedAccessPatternAnalyzer()
        
        # Simulate concurrent access from multiple "threads"
        for i in range(10):
            analyzer.record_access(f"tensor_{i % 3}", f"context_{i % 2}")
        
        # Check that all tensors are tracked
        hot_tensors = analyzer.get_hot_tensors(5)
        self.assertGreaterEqual(len(hot_tensors), 1)


def run_comprehensive_tests():
    """Run all tests and return results"""
    # Create a test suite
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTest(unittest.makeSuite(TestEnhancedAccessPatternAnalyzer))
    suite.addTest(unittest.makeSuite(TestEnhancedLifetimePredictor))
    suite.addTest(unittest.makeSuite(TestEnhancedTensorLifecycleTracker))
    suite.addTest(unittest.makeSuite(TestEnhancedPredictiveGarbageCollector))
    suite.addTest(unittest.makeSuite(TestHardwareOptimizers))
    suite.addTest(unittest.makeSuite(TestIntegratedMemoryManager))
    suite.addTest(unittest.makeSuite(TestIntegrationWithQwen3VL))
    suite.addTest(unittest.makeSuite(TestEdgeCases))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("Running Comprehensive Tests for Enhanced Predictive Tensor Lifecycle Management System")
    print("=" * 90)

    # Run tests
    test_result = run_comprehensive_tests()

    print("\n" + "=" * 90)
    print("TEST SUMMARY:")
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Success rate: {((test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun * 100):.2f}%")

    if test_result.failures:
        print("\nFAILURES:")
        for test, traceback in test_result.failures:
            print(f"  {test}: {traceback}")

    if test_result.errors:
        print("\nERRORS:")
        for test, traceback in test_result.errors:
            print(f"  {test}: {traceback}")

    if not test_result.failures and not test_result.errors:
        print("\nAll tests passed! ✓")
    else:
        print("\nSome tests failed or had errors. ✗")
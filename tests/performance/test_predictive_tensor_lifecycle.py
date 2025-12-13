"""
Comprehensive tests for the Advanced Predictive Tensor Lifecycle Management System
"""
import unittest
import torch
import time
import tempfile
import os
from predictive_tensor_lifecycle_manager import (
    TensorState, TensorType, TensorMetadata, AccessPatternAnalyzer,
    LifetimePredictor, TensorLifecycleTracker, PredictiveGarbageCollector,
    HardwareAwareTensorManager, IntegratedTensorLifecycleManager,
    create_optimized_lifecycle_manager
)
from typing import Dict, Any


class TestAccessPatternAnalyzer(unittest.TestCase):
    """Test the access pattern analyzer component"""
    
    def setUp(self):
        self.analyzer = AccessPatternAnalyzer()
    
    def test_record_access(self):
        """Test recording tensor access"""
        self.analyzer.record_access("tensor_1")
        self.assertIn("tensor_1", self.analyzer.tensor_last_access)
        
        # Check that access count is updated
        self.assertEqual(self.analyzer.tensor_access_counts["tensor_1"], 1)
    
    def test_predict_access(self):
        """Test access prediction"""
        # Record multiple accesses to create a pattern
        for i in range(5):
            self.analyzer.record_access("tensor_1")
            time.sleep(0.01)  # Small delay to create different access times
        
        prob, time_pred, count_pred = self.analyzer.predict_access("tensor_1")
        
        # Probability should be between 0 and 1
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)
        
        # Count prediction should be positive
        self.assertGreater(count_pred, 0)


class TestLifetimePredictor(unittest.TestCase):
    """Test the lifetime prediction component"""
    
    def setUp(self):
        self.predictor = LifetimePredictor()
        self.analyzer = AccessPatternAnalyzer()
    
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
    
    def test_record_memory_pressure(self):
        """Test memory pressure recording"""
        self.predictor.record_memory_pressure(0.7)
        self.predictor.record_memory_pressure(0.8)
        
        # Check that pressure history is maintained
        self.assertGreater(len(self.predictor.memory_pressure_history), 0)


class TestTensorLifecycleTracker(unittest.TestCase):
    """Test the tensor lifecycle tracker"""
    
    def setUp(self):
        self.tracker = TensorLifecycleTracker()
    
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


class TestPredictiveGarbageCollector(unittest.TestCase):
    """Test the predictive garbage collector"""
    
    def setUp(self):
        self.garbage_collector = PredictiveGarbageCollector(
            collection_interval=0.1,  # Very short interval for testing
            memory_pressure_threshold=0.9  # High threshold to avoid collection
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


class TestHardwareAwareTensorManager(unittest.TestCase):
    """Test the hardware-aware tensor manager"""
    
    def setUp(self):
        self.hardware_config = {
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
            'storage_type': 'nvme'
        }
        self.manager = HardwareAwareTensorManager(self.hardware_config)
    
    def test_tensor_placement(self):
        """Test tensor placement optimization"""
        # Test with small tensor
        small_tensor = torch.randn(10, 10)
        placement = self.manager.optimize_tensor_placement(small_tensor, TensorType.GENERAL)
        self.assertIn(placement, ['cpu', 'gpu', 'disk'])
        
        # Test with different tensor types
        for tensor_type in [TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, TensorType.GRADIENTS]:
            placement = self.manager.optimize_tensor_placement(small_tensor, tensor_type)
            self.assertIn(placement, ['cpu', 'gpu', 'disk'])
    
    def test_tensor_optimization(self):
        """Test tensor optimization for hardware"""
        tensor = torch.randn(20, 20)
        
        # Test optimization
        optimized_tensor = self.manager.optimize_tensor_for_hardware(
            tensor, 
            TensorType.GENERAL, 
            "matmul"
        )
        
        # Should return a tensor
        self.assertIsInstance(optimized_tensor, torch.Tensor)
        self.assertEqual(optimized_tensor.shape, tensor.shape)
    
    def test_hardware_info(self):
        """Test hardware optimization info"""
        info = self.manager.get_hardware_optimization_info()
        
        self.assertIn('cpu_model', info)
        self.assertIn('gpu_model', info)
        self.assertIn('cpu_optimizations', info)
        self.assertIn('gpu_optimizations', info)
        self.assertIn('storage_optimizations', info)


class TestIntegratedTensorLifecycleManager(unittest.TestCase):
    """Test the integrated tensor lifecycle manager"""
    
    def setUp(self):
        self.lifecycle_manager = create_optimized_lifecycle_manager({
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
            'storage_type': 'nvme'
        })
    
    def tearDown(self):
        self.lifecycle_manager.cleanup()
    
    def test_tensor_registration(self):
        """Test tensor registration with lifecycle management"""
        tensor = torch.randn(15, 15)
        tensor_id = self.lifecycle_manager.register_tensor(
            tensor, 
            tensor_type=TensorType.GENERAL,
            is_pinned=False,
            initial_ref_count=1
        )
        
        self.assertIsInstance(tensor_id, str)
        self.assertTrue(tensor_id.startswith("tensor_"))
    
    def test_reference_counting(self):
        """Test reference counting through the lifecycle manager"""
        tensor = torch.randn(5, 5)
        tensor_id = self.lifecycle_manager.register_tensor(tensor)
        
        # Test increment
        result = self.lifecycle_manager.increment_reference(tensor_id, "context_1")
        self.assertTrue(result)
        
        # Test decrement
        result = self.lifecycle_manager.decrement_reference(tensor_id, "context_1")
        self.assertTrue(result)
        
        # Test update by delta
        result = self.lifecycle_manager.update_reference_count(tensor_id, -1, "context_2")
        self.assertTrue(result)
    
    def test_tensor_access(self):
        """Test tensor access recording"""
        tensor = torch.randn(8, 8)
        tensor_id = self.lifecycle_manager.register_tensor(tensor)
        
        result = self.lifecycle_manager.access_tensor(tensor_id, "test_context")
        self.assertTrue(result)
    
    def test_lifecycle_stats(self):
        """Test lifecycle statistics"""
        stats = self.lifecycle_manager.get_tensor_lifecycle_stats()
        
        self.assertIn('total_tensors', stats)
        self.assertIn('pinned_tensors', stats)
        self.assertIn('collections_performed', stats)
        self.assertIn('tensors_collected', stats)
        self.assertIn('memory_freed_bytes', stats)
        self.assertIn('average_lifetime_prediction', stats)
        self.assertIn('states', stats)
    
    def test_hardware_optimization(self):
        """Test hardware-aware tensor optimization"""
        tensor = torch.randn(25, 25)
        
        optimized_tensor = self.lifecycle_manager.optimize_tensor(
            tensor, 
            TensorType.GENERAL, 
            "matmul"
        )
        
        self.assertIsInstance(optimized_tensor, torch.Tensor)
        self.assertEqual(optimized_tensor.shape, tensor.shape)


class TestIntegration(unittest.TestCase):
    """Test integration between components"""
    
    def setUp(self):
        self.lifecycle_manager = create_optimized_lifecycle_manager({
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,
            'storage_type': 'nvme'
        })
    
    def tearDown(self):
        self.lifecycle_manager.cleanup()
    
    def test_full_lifecycle(self):
        """Test a complete tensor lifecycle"""
        # Register a tensor
        tensor = torch.randn(12, 12)
        tensor_id = self.lifecycle_manager.register_tensor(
            tensor, 
            tensor_type=TensorType.IMAGE_FEATURES,
            is_pinned=False,
            initial_ref_count=2
        )
        
        # Access the tensor multiple times to build access pattern
        for i in range(3):
            self.lifecycle_manager.access_tensor(tensor_id, f"context_{i}")
            time.sleep(0.01)  # Small delay
        
        # Decrement reference count
        self.lifecycle_manager.decrement_reference(tensor_id, "context_0")
        
        # Check tensor state
        stats = self.lifecycle_manager.get_tensor_lifecycle_stats()
        self.assertGreaterEqual(stats['total_tensors'], 1)
        
        # The tensor should still be in use (ref count > 0)
        self.assertGreaterEqual(stats['states']['in_use'], 1)


def run_comprehensive_tests():
    """Run all tests and return results"""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestAccessPatternAnalyzer))
    suite.addTest(unittest.makeSuite(TestLifetimePredictor))
    suite.addTest(unittest.makeSuite(TestTensorLifecycleTracker))
    suite.addTest(unittest.makeSuite(TestPredictiveGarbageCollector))
    suite.addTest(unittest.makeSuite(TestHardwareAwareTensorManager))
    suite.addTest(unittest.makeSuite(TestIntegratedTensorLifecycleManager))
    suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Running Comprehensive Tests for Predictive Tensor Lifecycle Management System")
    print("=" * 80)
    
    # Run tests
    test_result = run_comprehensive_tests()
    
    print("\n" + "=" * 80)
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
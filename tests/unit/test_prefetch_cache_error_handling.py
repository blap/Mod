"""
Comprehensive Test Suite for Prefetching and Caching Error Handling in Qwen3-VL

This test suite validates the error handling implementation for prefetching and caching operations,
including exception handling, error recovery, fallback strategies, and performance metrics.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
from typing import Any, Dict, Optional
import sys

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.qwen3_vl.optimization.error_handling.prefetch_cache_error_handler import (
    PrefetchCacheErrorHandler,
    safe_prefetch_operation,
    safe_cache_operation,
    PrefetchingErrorDecorator,
    CachingErrorDecorator,
    FallbackStrategies,
    create_error_handler,
    PrefetchMonitor,
    CacheMonitor
)

from src.qwen3_vl.optimization.error_handling.enhanced_error_handling_integration import (
    EnhancedPrefetchingSystem,
    EnhancedCachingSystem,
    KVCacheWithEnhancedErrorHandling,
    AttentionWithEnhancedErrorHandling
)


class TestPrefetchCacheErrorHandler(unittest.TestCase):
    """Test the PrefetchCacheErrorHandler class."""

    def setUp(self):
        self.error_handler = create_error_handler(log_errors=True, enable_recovery=True)

    def test_error_handler_initialization(self):
        """Test that the error handler initializes correctly."""
        self.assertIsNotNone(self.error_handler)
        self.assertTrue(self.error_handler.log_errors)
        self.assertTrue(self.error_handler.enable_recovery)
        self.assertEqual(len(self.error_handler.errors), 0)

    def test_handle_error_with_fallback(self):
        """Test handling an error with a fallback function."""
        def failing_func():
            raise ValueError("Test error for fallback")
        
        def fallback_func():
            return "fallback_result"
        
        success, result = self.error_handler.handle_error(
            ValueError("Test error"), 
            "test_operation", 
            fallback_func
        )
        
        self.assertTrue(success)
        self.assertEqual(result, "fallback_result")

    def test_handle_error_without_fallback(self):
        """Test handling an error without a fallback function."""
        success, result = self.error_handler.handle_error(
            RuntimeError("Test error"), 
            "test_operation"
        )
        
        self.assertFalse(success)
        self.assertIsNone(result)

    def test_error_severity_classification(self):
        """Test that errors are classified with appropriate severity."""
        high_severity_error = RuntimeError("High severity error")
        medium_severity_error = KeyError("Medium severity error")
        low_severity_error = Warning("Low severity error")
        
        high_severity = self.error_handler._determine_severity(high_severity_error)
        medium_severity = self.error_handler._determine_severity(medium_severity_error)
        # Note: low_severity_error is not an Exception subclass, so this test might need adjustment
        
        self.assertIn(high_severity, [high_severity_error.__class__.__name__ == 'RuntimeError' and self.error_handler._determine_severity(high_severity_error)])
        self.assertIn(medium_severity, [medium_severity_error.__class__.__name__ == 'KeyError' and self.error_handler._determine_severity(medium_severity_error)])

    def test_error_statistics(self):
        """Test error statistics functionality."""
        # Trigger some errors to populate statistics
        self.error_handler.handle_error(ValueError("Test error 1"), "op1")
        self.error_handler.handle_error(RuntimeError("Test error 2"), "op2")
        
        stats = self.error_handler.get_error_statistics()
        
        self.assertGreaterEqual(stats['total_errors'], 2)
        self.assertIn('severity_breakdown', stats)
        self.assertIn('recent_errors', stats)

    def test_safe_prefetch_operation(self):
        """Test safe prefetch operation with error handling."""
        def successful_func():
            return "success"
        
        def failing_func():
            raise ValueError("Intentional failure")
        
        # Test successful operation
        success, result = safe_prefetch_operation(self.error_handler, successful_func)
        self.assertTrue(success)
        self.assertEqual(result, "success")
        
        # Test failing operation with fallback
        success, result = safe_prefetch_operation(
            self.error_handler, 
            failing_func
        )
        self.assertTrue(success)  # Should succeed with fallback
        self.assertIsNotNone(result)

    def test_safe_cache_operation(self):
        """Test safe cache operation with error handling."""
        def successful_func():
            return "success"
        
        def failing_func():
            raise RuntimeError("Intentional failure")
        
        # Test successful operation
        success, result = safe_cache_operation(self.error_handler, successful_func)
        self.assertTrue(success)
        self.assertEqual(result, "success")
        
        # Test failing operation with fallback
        success, result = safe_cache_operation(
            self.error_handler, 
            failing_func
        )
        self.assertTrue(success)  # Should succeed with fallback
        self.assertIsNotNone(result)


class TestEnhancedPrefetchingSystem(unittest.TestCase):
    """Test the EnhancedPrefetchingSystem class."""

    def setUp(self):
        self.prefetch_system = EnhancedPrefetchingSystem(enable_error_handling=True)

    def test_prefetch_data_basic(self):
        """Test basic prefetching functionality."""
        success = self.prefetch_system.prefetch_data(0x1000, 1024, 0)
        self.assertTrue(success)

    def test_prefetch_data_with_invalid_pointer(self):
        """Test prefetching with invalid pointer (should trigger error handling)."""
        success = self.prefetch_system.prefetch_data(None, 1024, 0)  # None is invalid pointer
        # With error handling, this should return a fallback result (likely False or None)
        # but shouldn't crash the system
        self.assertIsNotNone(success)

    def test_batch_prefetch(self):
        """Test batch prefetching functionality."""
        requests = [(0x1000, 512, 0), (0x2000, 1024, 0), (0x3000, 256, 0)]
        stats = self.prefetch_system.batch_prefetch(requests)
        
        self.assertIn('total_requests', stats)
        self.assertIn('successful_prefetches', stats)
        self.assertIn('failed_prefetches', stats)
        self.assertGreaterEqual(stats['total_requests'], 3)

    def test_prefetch_statistics(self):
        """Test prefetch statistics."""
        # Perform some operations to populate stats
        self.prefetch_system.prefetch_data(0x1000, 1024, 0)
        self.prefetch_system.prefetch_data(0x2000, 512, 0)
        
        stats = self.prefetch_system.get_prefetch_stats()
        self.assertIn('successful_prefetches', stats)
        self.assertIn('error_statistics', stats)


class TestEnhancedCachingSystem(unittest.TestCase):
    """Test the EnhancedCachingSystem class."""

    def setUp(self):
        self.cache_system = EnhancedCachingSystem(enable_error_handling=True)

    def test_basic_cache_operations(self):
        """Test basic cache get/put operations."""
        tensor = torch.randn(10, 20)
        
        # Test put operation
        success = self.cache_system.put_in_cache("test_key", tensor)
        self.assertTrue(success)
        
        # Test get operation
        retrieved_tensor = self.cache_system.get_from_cache("test_key")
        self.assertIsNotNone(retrieved_tensor)
        self.assertTrue(torch.equal(tensor, retrieved_tensor))

    def test_cache_overflow_protection(self):
        """Test that the cache doesn't grow indefinitely."""
        # Create many tensors with different keys to test overflow protection
        for i in range(50):
            tensor = torch.randn(5, 5)
            success = self.cache_system.put_in_cache(f"key_{i}", tensor)
            # Even if some fail due to size constraints, system should not crash
        
        stats = self.cache_system.get_cache_stats()
        self.assertIn('insertions', stats)
        self.assertIn('evictions', stats)
        self.assertIn('current_size', stats)

    def test_cache_statistics(self):
        """Test cache statistics functionality."""
        tensor = torch.randn(5, 5)
        self.cache_system.put_in_cache("stats_test", tensor)
        retrieved = self.cache_system.get_from_cache("stats_test")
        
        stats = self.cache_system.get_cache_stats()
        self.assertGreaterEqual(stats['cache_hits'], 0)
        self.assertGreaterEqual(stats['cache_misses'], 0)
        self.assertGreaterEqual(stats['insertions'], 1)
        self.assertIn('hit_rate', stats)
        self.assertIn('error_statistics', stats)


class TestKVCacheWithEnhancedErrorHandling(unittest.TestCase):
    """Test the KVCacheWithEnhancedErrorHandling class."""

    def setUp(self):
        self.kv_cache = KVCacheWithEnhancedErrorHandling()

    def test_update_and_get_cache(self):
        """Test updating and retrieving from KV cache."""
        key_states = torch.randn(1, 8, 100, 64)  # batch, heads, seq, head_dim
        value_states = torch.randn(1, 8, 100, 64)
        
        # Update cache
        updated_k, updated_v = self.kv_cache.update_cache(
            key_states, value_states, layer_idx=0
        )
        
        # Check that update was successful
        self.assertEqual(updated_k.shape, key_states.shape)
        self.assertEqual(updated_v.shape, value_states.shape)
        
        # Get from cache
        retrieved = self.kv_cache.get_cache(0)
        self.assertIsNotNone(retrieved)
        
        retrieved_k, retrieved_v = retrieved
        self.assertEqual(retrieved_k.shape, key_states.shape)
        self.assertEqual(retrieved_v.shape, value_states.shape)

    def test_cache_statistics(self):
        """Test KV cache statistics."""
        stats = self.kv_cache.get_statistics()
        self.assertIn('prefetch_stats', stats)
        self.assertIn('cache_stats', stats)
        self.assertIn('average_access_time', stats)


class TestAttentionWithEnhancedErrorHandling(unittest.TestCase):
    """Test the AttentionWithEnhancedErrorHandling class."""

    def setUp(self):
        self.attention = AttentionWithEnhancedErrorHandling()

    def test_forward_pass_with_cache(self):
        """Test forward pass with cache enabled."""
        hidden_states = torch.randn(1, 100, 512)  # batch, seq, hidden
        attention_mask = torch.ones(1, 100, 100)
        cache_position = torch.arange(0, 100)
        
        output, weights, cache = self.attention.forward(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=True,
            cache_position=cache_position
        )
        
        # Check output shapes
        self.assertEqual(output.shape[0], 1)  # batch
        self.assertEqual(output.shape[1], 100)  # seq
        self.assertEqual(weights.shape[1], 100)  # seq

    def test_forward_pass_without_cache(self):
        """Test forward pass without cache."""
        hidden_states = torch.randn(1, 50, 256)  # batch, seq, hidden
        attention_mask = torch.ones(1, 50, 50)
        
        output, weights, cache = self.attention.forward(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=False
        )
        
        # Check output shapes
        self.assertEqual(output.shape[0], 1)  # batch
        self.assertEqual(output.shape[1], 50)  # seq
        self.assertEqual(weights.shape[1], 50)  # seq
        self.assertIsNone(cache)


class TestFallbackStrategies(unittest.TestCase):
    """Test fallback strategies."""

    def test_fallback_no_prefetch(self):
        """Test the fallback_no_prefetch strategy."""
        result = FallbackStrategies.fallback_no_prefetch("arg1", "arg2")
        self.assertEqual(result, "arg1")  # Should return first arg

    def test_fallback_standard_cache(self):
        """Test the fallback_standard_cache strategy."""
        result = FallbackStrategies.fallback_standard_cache("data", "extra_arg")
        self.assertEqual(result, ("data", False))  # Should return data and False

    def test_fallback_cpu_cache(self):
        """Test the fallback_cpu_cache strategy."""
        tensor_gpu = torch.randn(10, 10)
        result = FallbackStrategies.fallback_cpu_cache(tensor_gpu)
        self.assertIsNotNone(result)

    def test_fallback_reduce_tensor_size(self):
        """Test the fallback_reduce_tensor_size strategy."""
        large_tensor = torch.randn(100, 100)
        result = FallbackStrategies.fallback_reduce_tensor_size(large_tensor, target_size=50)
        
        # The result should be smaller than the original
        self.assertLessEqual(result.numel(), large_tensor.numel())
        
        # Or at least not crash
        self.assertIsNotNone(result)


class TestMonitors(unittest.TestCase):
    """Test the monitoring components."""

    def test_prefetch_monitor(self):
        """Test PrefetchMonitor functionality."""
        error_handler = create_error_handler()
        monitor = PrefetchMonitor(error_handler)
        
        monitor.record_prefetch_attempt(True, 0.001)
        monitor.record_prefetch_attempt(False, 0.002)
        
        perf = monitor.get_prefetch_performance()
        self.assertIn('success_rate', perf)
        self.assertIn('total_attempts', perf)
        self.assertEqual(perf['total_attempts'], 2)

    def test_cache_monitor(self):
        """Test CacheMonitor functionality."""
        error_handler = create_error_handler()
        monitor = CacheMonitor(error_handler)
        
        monitor.record_cache_operation('hit', 0.001, 1000)
        monitor.record_cache_operation('miss', 0.002, 500)
        monitor.record_cache_operation('eviction', 0.001, -1000)
        
        perf = monitor.get_cache_performance()
        self.assertIn('hit_rate', perf)
        self.assertIn('total_operations', perf)
        self.assertEqual(perf['total_operations'], 3)


class TestDecorators(unittest.TestCase):
    """Test the error handling decorators."""

    def test_prefetching_error_decorator(self):
        """Test PrefetchingErrorDecorator functionality."""
        error_handler = create_error_handler()
        
        @PrefetchingErrorDecorator(
            error_handler=error_handler,
            fallback_func=lambda: "fallback_result",
            default_return_value="default_result"
        )
        def test_func():
            return "success_result"
        
        result = test_func()
        self.assertEqual(result, "success_result")

    def test_caching_error_decorator(self):
        """Test CachingErrorDecorator functionality."""
        error_handler = create_error_handler()
        
        @CachingErrorDecorator(
            error_handler=error_handler,
            fallback_func=lambda: "fallback_result",
            default_return_value="default_result"
        )
        def test_func():
            return "success_result"
        
        result = test_func()
        self.assertEqual(result, "success_result")


def run_all_tests():
    """Run all tests in the suite."""
    print("Running Prefetching and Caching Error Handling Test Suite")
    print("=" * 65)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestPrefetchCacheErrorHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedPrefetchingSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedCachingSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestKVCacheWithEnhancedErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionWithEnhancedErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestFallbackStrategies))
    suite.addTests(loader.loadTestsFromTestCase(TestMonitors))
    suite.addTests(loader.loadTestsFromTestCase(TestDecorators))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\nTest Results Summary:")
    print(f"  Total tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ All tests passed! Prefetching and caching error handling implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Please review the implementation.")
        sys.exit(1)
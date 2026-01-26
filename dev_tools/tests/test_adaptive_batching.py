"""
Test suite for Adaptive Micro-batching functionality in model plugins.

This test verifies that the adaptive batching system works correctly across all model plugins.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import time
import torch
from src.inference_pio.common.adaptive_batch_manager import AdaptiveBatchManager, get_adaptive_batch_manager
from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin

# TestAdaptiveBatching

    """Test cases for adaptive batching functionality."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        plugins = [
            GLM_4_7_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin(),
            Qwen3_VL_2B_Plugin()
        ]

    def adaptive_batch_manager_creation(self)():
        """Test that the adaptive batch manager can be created and accessed."""
        manager = AdaptiveBatchManager(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16,
            memory_threshold_ratio=0.8
        )
        assert_is_instance(manager, AdaptiveBatchManager)
        assert_equal(manager.current_batch_size, 4)
        assert_equal(manager.min_batch_size, 1)
        assert_equal(manager.max_batch_size, 16)

        # Test global instance
        global_manager = get_adaptive_batch_manager()
        assert_is_instance(global_manager, AdaptiveBatchManager)

    def batch_size_adjustment_logic(self)():
        """Test the logic for adjusting batch sizes based on performance."""
        manager = AdaptiveBatchManager(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16,
            memory_threshold_ratio=0.8,
            performance_window_size=5
        )

        # Simulate good performance with low memory pressure
        for i in range(10):
            new_size = manager.get_optimal_batch_size(
                processing_time_ms=100.0,
                tokens_processed=100
            )

        # Check that batch size increased due to good performance
        status = manager.get_status_report()
        assertGreaterEqual(status['current_batch_size'], 4)

        # Reset for next test
        manager.force_batch_size(4)

        # Simulate poor performance with high memory pressure
        for i in range(10):
            # Artificially high processing time and low throughput to simulate poor performance
            new_size = manager.get_optimal_batch_size(
                processing_time_ms=1000.0,  # Slow processing
                tokens_processed=10  # Low throughput
            )

        # Check that batch size decreased due to poor performance
        status = manager.get_status_report()
        assertLessEqual(status['current_batch_size'], 4)

    def plugin_adaptive_batching_setup(self)():
        """Test that all plugins can set up adaptive batching."""
        for plugin in plugins:
            # Initialize the plugin with adaptive batching enabled
            success = plugin.initialize(enable_adaptive_batching=True)
            assert_true(success)

            # Check that adaptive batching methods are available
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))

            # Test that adaptive batching can be set up
            setup_success = plugin.setup_adaptive_batching()
            assert_true(setup_success)

    def plugin_get_optimal_batch_size(self)():
        """Test that plugins can get optimal batch sizes."""
        for plugin in plugins:
            plugin.initialize(enable_adaptive_batching=True)

            # Test getting optimal batch size with sample metrics
            optimal_size = plugin.get_optimal_batch_size(
                processing_time_ms=200.0,
                tokens_processed=50
            )

            assert_is_instance(optimal_size, int)
            assertGreaterEqual(optimal_size, 1)

    def plugin_batch_size_adjustment(self)():
        """Test that plugins can adjust batch sizes."""
        for plugin in plugins:
            plugin.initialize(enable_adaptive_batching=True)

            # Get initial status
            initial_status = plugin.get_batching_status()
            initial_batch_size = initial_status['current_batch_size']

            # Attempt to adjust batch size
            new_size, was_adjusted, reason = plugin.adjust_batch_size()

            # Check return types
            assert_is_instance(new_size, int)
            assert_is_instance(was_adjusted, bool)
            if reason is not None:
                assert_is_instance(reason, str)

    def plugin_batching_status(self)():
        """Test that plugins can report batching status."""
        for plugin in plugins:
            plugin.initialize(enable_adaptive_batching=True)

            # Get batching status
            status = plugin.get_batching_status()

            # Check that required keys are present
            required_keys = [
                'current_batch_size',
                'adaptive_batching_enabled',
                'memory_pressure_ratio',
                'performance_score'
            ]

            for key in required_keys:
                assert_in(key, status)

    def memory_pressure_response(self)():
        """Test that adaptive batching responds to memory pressure."""
        # Create a manager with low memory threshold to trigger adjustments
        manager = AdaptiveBatchManager(
            initial_batch_size=8,
            min_batch_size=1,
            max_batch_size=16,
            memory_threshold_ratio=0.1,  # Very low threshold to trigger memory pressure response
            performance_window_size=3
        )

        # Simulate high memory pressure by artificially inflating memory usage
        # This is difficult to test without actually consuming memory, so we'll test the logic directly
        # by calling the adjustment methods

        # Force a memory pressure scenario by directly manipulating metrics
        for i in range(5):
            # Report high processing time and low throughput to simulate poor performance
            optimal_size = manager.get_optimal_batch_size(
                processing_time_ms=500.0,
                tokens_processed=10
            )

        # Check that the batch size was reduced due to poor performance
        status = manager.get_status_report()
        assertLessEqual(status['current_batch_size'], 8)

    def adaptive_batching_with_memory_optimizations(self)():
        """Test adaptive batching working with memory optimizations."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with adaptive batching and memory optimizations
            success = plugin.initialize(
                enable_adaptive_batching=True,
                enable_memory_management=True,
                enable_tensor_paging=True,
                enable_smart_swap=True
            )
            assert_true(success)

            # Check that both systems are properly set up
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))

            # Test that batching status reflects memory pressure
            status = plugin.get_batching_status()
            assert_in('memory_pressure_ratio', status)

    def adaptive_batching_with_other_optimizations(self)():
        """Test adaptive batching working with other optimizations."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with adaptive batching and multiple optimizations
            success = plugin.initialize(
                enable_adaptive_batching=True,
                enable_kernel_fusion=True,
                enable_tensor_compression=True,
                enable_disk_offloading=True,
                enable_model_surgery=True
            )
            assert_true(success)

            # Test that batching still works with other optimizations
            optimal_size = plugin.get_optimal_batch_size(
                processing_time_ms=150.0,
                tokens_processed=75
            )
            assert_is_instance(optimal_size, int)
            assertGreaterEqual(optimal_size, 1)

    def cleanup_helper():
        """Clean up after each test method."""
        # Clean up any resources used by the managers
        for plugin in plugins:
            if hasattr(plugin, '_adaptive_batch_manager') and plugin._adaptive_batch_manager:
                plugin._adaptive_batch_manager.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)
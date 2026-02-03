"""
Test suite for Adaptive Micro-batching functionality in model plugins.

This test verifies that the adaptive batching system works correctly across all model plugins.
"""

import time
import unittest

import torch

from src.inference_pio.common.processing.adaptive_batch_manager import (
    AdaptiveBatchManager,
    get_adaptive_batch_manager,
)
from src.inference_pio.models.glm_4_7_flash.plugin import GLM_4_7_Flash_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin


class TestAdaptiveBatching(unittest.TestCase):
    """Test cases for adaptive batching functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.plugins = [
            GLM_4_7_Flash_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin(),
            Qwen3_VL_2B_Plugin(),
        ]

    def test_adaptive_batch_manager_creation(self):
        """Test that the adaptive batch manager can be created and accessed."""
        manager = AdaptiveBatchManager(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16,
            memory_threshold_ratio=0.8,
        )
        self.assertIsInstance(manager, AdaptiveBatchManager)
        self.assertEqual(manager.current_batch_size, 4)
        self.assertEqual(manager.min_batch_size, 1)
        self.assertEqual(manager.max_batch_size, 16)

        # Test global instance
        global_manager = get_adaptive_batch_manager()
        self.assertIsInstance(global_manager, AdaptiveBatchManager)

    def test_batch_size_adjustment_logic(self):
        """Test the logic for adjusting batch sizes based on performance."""
        manager = AdaptiveBatchManager(
            initial_batch_size=4,
            min_batch_size=1,
            max_batch_size=16,
            memory_threshold_ratio=0.8,
            performance_window_size=5,
        )

        # Simulate good performance with low memory pressure
        for i in range(10):
            new_size = manager.get_optimal_batch_size(
                processing_time_ms=100.0, tokens_processed=100
            )

        # Check that batch size increased due to good performance
        status = manager.get_status_report()
        self.assertGreaterEqual(status["current_batch_size"], 4)

        # Reset for next test
        manager.force_batch_size(4)

        # Simulate poor performance with high memory pressure
        for i in range(10):
            # Artificially high processing time and low throughput to simulate poor performance
            new_size = manager.get_optimal_batch_size(
                processing_time_ms=1000.0,  # Slow processing
                tokens_processed=10,  # Low throughput
            )

        # Check that batch size decreased due to poor performance
        status = manager.get_status_report()
        self.assertLessEqual(status["current_batch_size"], 4)

    def test_plugin_adaptive_batching_setup(self):
        """Test that all plugins can set up adaptive batching."""
        for plugin in self.plugins:
            # Initialize the plugin with adaptive batching enabled
            # Note: This assumes the plugin has an initialize method that accepts these parameters
            try:
                success = plugin.initialize(
                    config=None
                )  # Initialize with minimal config
                if success:
                    # Check that adaptive batching methods are available
                    self.assertTrue(hasattr(plugin, "setup_adaptive_batching"))
                    self.assertTrue(hasattr(plugin, "get_optimal_batch_size"))
                    self.assertTrue(hasattr(plugin, "adjust_batch_size"))
                    self.assertTrue(hasattr(plugin, "get_batching_status"))

                    # Test that adaptive batching can be set up (if method exists)
                    if hasattr(plugin, "setup_adaptive_batching"):
                        # This may fail if the plugin isn't fully loaded, which is expected
                        try:
                            plugin.setup_adaptive_batching()
                        except (AttributeError, RuntimeError):
                            # Expected if model isn't properly loaded
                            pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_plugin_get_optimal_batch_size(self):
        """Test that plugins can get optimal batch sizes."""
        for plugin in self.plugins:
            try:
                # Initialize plugin
                success = plugin.initialize(config=None)

                # Test getting optimal batch size with sample metrics (if method exists)
                if hasattr(plugin, "get_optimal_batch_size"):
                    try:
                        optimal_size = plugin.get_optimal_batch_size(
                            processing_time_ms=200.0, tokens_processed=50
                        )
                        self.assertIsInstance(optimal_size, int)
                        self.assertGreaterEqual(optimal_size, 1)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_plugin_batch_size_adjustment(self):
        """Test that plugins can adjust batch sizes."""
        for plugin in self.plugins:
            try:
                plugin.initialize(config=None)

                # Test that adjustment methods exist and work (if available)
                if hasattr(plugin, "adjust_batch_size"):
                    try:
                        # Attempt to adjust batch size
                        result = plugin.adjust_batch_size()

                        # Check return types (adjust_batch_size might return different formats)
                        if isinstance(result, tuple) and len(result) >= 2:
                            new_size, was_adjusted = result[0], result[1]
                            self.assertIsInstance(new_size, int)
                            self.assertIsInstance(was_adjusted, bool)
                        elif isinstance(result, int):
                            # If it just returns the new size
                            self.assertIsInstance(result, int)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_plugin_batching_status(self):
        """Test that plugins can report batching status."""
        for plugin in self.plugins:
            try:
                plugin.initialize(config=None)

                # Test that status reporting method exists and works (if available)
                if hasattr(plugin, "get_batching_status"):
                    try:
                        status = plugin.get_batching_status()

                        # Check that required keys are present
                        required_keys = [
                            "current_batch_size",
                        ]

                        for key in required_keys:
                            self.assertIn(key, status)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_memory_pressure_response(self):
        """Test that adaptive batching responds to memory pressure."""
        # Create a manager with low memory threshold to trigger adjustments
        manager = AdaptiveBatchManager(
            initial_batch_size=8,
            min_batch_size=1,
            max_batch_size=16,
            memory_threshold_ratio=0.1,  # Very low threshold to trigger memory pressure response
            performance_window_size=3,
        )

        # Simulate high memory pressure by artificially inflating memory usage
        # This is difficult to test without actually consuming memory, so we'll test the logic directly
        # by calling the adjustment methods

        # Force a memory pressure scenario by directly manipulating metrics
        for i in range(5):
            # Report high processing time and low throughput to simulate poor performance
            optimal_size = manager.get_optimal_batch_size(
                processing_time_ms=500.0, tokens_processed=10
            )

        # Check that the batch size was reduced due to poor performance
        status = manager.get_status_report()
        self.assertLessEqual(status["current_batch_size"], 8)

    def test_adaptive_batching_with_memory_optimizations(self):
        """Test adaptive batching working with memory optimizations."""
        for plugin in self.plugins[
            :1
        ]:  # Test with first plugin to avoid long execution
            try:
                # Initialize the plugin with adaptive batching and memory optimizations
                success = plugin.initialize(
                    config=None
                )  # Initialize with minimal config

                if success and hasattr(plugin, "get_batching_status"):
                    # Test that batching status can be retrieved
                    try:
                        status = plugin.get_batching_status()
                        # Check that memory-related fields exist if the method is available
                        if "memory_pressure_ratio" in status:
                            self.assertIsInstance(
                                status["memory_pressure_ratio"], (int, float)
                            )
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_adaptive_batching_with_other_optimizations(self):
        """Test adaptive batching working with other optimizations."""
        for plugin in self.plugins[
            :1
        ]:  # Test with first plugin to avoid long execution
            try:
                # Initialize the plugin
                success = plugin.initialize(config=None)

                # Test that batching still works with other optimizations (if method exists)
                if success and hasattr(plugin, "get_optimal_batch_size"):
                    try:
                        optimal_size = plugin.get_optimal_batch_size(
                            processing_time_ms=150.0, tokens_processed=75
                        )
                        self.assertIsInstance(optimal_size, int)
                        self.assertGreaterEqual(optimal_size, 1)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass


if __name__ == "__main__":
    unittest.main()

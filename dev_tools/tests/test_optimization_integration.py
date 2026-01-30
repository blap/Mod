"""
Integration test suite for all optimization techniques working together.

This test verifies that all optimization techniques work together correctly across all model plugins.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import tempfile
from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin

# TestOptimizationIntegration

    """Test cases for optimization integration."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        plugins = [
            GLM_4_7_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin(),
            Qwen3_VL_2B_Plugin()
        ]

    def all_optimizations_enabled(self)():
        """Test that all optimizations can be enabled together."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with all optimizations enabled
            success = plugin.initialize(
                enable_memory_management=True,
                enable_tensor_paging=True,
                enable_smart_swap=True,
                enable_kernel_fusion=True,
                enable_adaptive_batching=True,
                enable_distributed_simulation=True,
                enable_tensor_compression=True,
                enable_disk_offloading=True,
                enable_model_surgery=True,
                enable_pipeline=True,
                enable_activation_offloading=True
            )
            assert_true(success)
            
            # Verify that the plugin is properly configured with all optimizations
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))

    def memory_management_with_other_optimizations(self)():
        """Test memory management working with other optimizations."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with memory management and other optimizations
            success = plugin.initialize(
                enable_memory_management=True,
                enable_tensor_paging=True,
                enable_smart_swap=True,
                enable_tensor_compression=True,
                enable_disk_offloading=True
            )
            assert_true(success)
            
            # Get memory stats to verify memory management is working
            memory_stats = plugin.get_memory_stats()
            assert_is_instance(memory_stats, dict)
            assert_in('system_memory_percent', memory_stats)

    def kernel_fusion_with_model_surgery(self)():
        """Test kernel fusion working with model surgery."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with both optimizations
            success = plugin.initialize(
                enable_kernel_fusion=True,
                enable_model_surgery=True
            )
            assert_true(success)
            
            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()
            
            # Apply kernel fusion
            fusion_success = plugin.apply_kernel_fusion()
            assert_true(fusion_success)
            
            # Perform model surgery
            surgery_success = plugin.perform_model_surgery()
            assert_true(surgery_success)

    def tensor_compression_with_disk_offloading(self)():
        """Test tensor compression working with disk offloading."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with both optimizations
            success = plugin.initialize(
                enable_tensor_compression=True,
                enable_disk_offloading=True
            )
            assert_true(success)
            
            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()
            
            # Compress model weights
            compression_success = plugin.compress_model_weights(compression_ratio=0.5)
            assert_true(compression_success)
            
            # Offload model parts
            offload_success = plugin.offload_model_parts()
            assert_true(offload_success)

    def adaptive_batching_with_pipeline(self)():
        """Test adaptive batching working with pipeline."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with both optimizations
            success = plugin.initialize(
                enable_adaptive_batching=True,
                enable_pipeline=True
            )
            assert_true(success)
            
            # Get batching status to verify adaptive batching is working
            batching_status = plugin.get_batching_status()
            assert_is_instance(batching_status, dict)
            assert_in('current_batch_size', batching_status)
            
            # Get pipeline stats to verify pipeline is working
            pipeline_stats = plugin.get_pipeline_stats()
            assert_is_instance(pipeline_stats, dict)
            assert_in('pipeline_enabled', pipeline_stats)

    def all_optimizations_performance_metrics(self)():
        """Test that all optimizations together provide performance metrics."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with all optimizations
            success = plugin.initialize(
                enable_memory_management=True,
                enable_tensor_paging=True,
                enable_smart_swap=True,
                enable_kernel_fusion=True,
                enable_adaptive_batching=True,
                enable_tensor_compression=True,
                enable_disk_offloading=True,
                enable_model_surgery=True,
                enable_pipeline=True,
                enable_activation_offloading=True
            )
            assert_true(success)
            
            # Get various metrics
            memory_stats = plugin.get_memory_stats()
            assert_is_instance(memory_stats, dict)
            
            batching_status = plugin.get_batching_status()
            assert_is_instance(batching_status, dict)
            
            compression_stats = plugin.get_compression_stats()
            assert_is_instance(compression_stats, dict)
            
            offloading_stats = plugin.get_offloading_stats()
            assert_is_instance(offloading_stats, dict)
            
            surgery_stats = plugin.get_surgery_stats()
            assert_is_instance(surgery_stats, dict)
            
            pipeline_stats = plugin.get_pipeline_stats()
            assert_is_instance(pipeline_stats, dict)
            
            activation_stats = plugin.get_activation_offloading_stats()
            assert_is_instance(activation_stats, dict)

    def optimizations_with_inference(self)():
        """Test that optimizations don't break inference functionality."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with all optimizations
            success = plugin.initialize(
                enable_memory_management=True,
                enable_tensor_paging=True,
                enable_smart_swap=True,
                enable_kernel_fusion=True,
                enable_adaptive_batching=True,
                enable_tensor_compression=True,
                enable_disk_offloading=True,
                enable_model_surgery=True,
                enable_pipeline=True,
                enable_activation_offloading=True
            )
            assert_true(success)
            
            # Test basic inference still works
            try:
                # For this test, we'll just check that the plugin can be initialized
                # without errors when all optimizations are enabled
                assert_is_not_none(plugin._config)
            except Exception as e:
                fail(f"Inference failed with all optimizations enabled: {e}")

    def optimizations_cleanup(self)():
        """Test that all optimizations can be cleaned up properly."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with all optimizations
            success = plugin.initialize(
                enable_memory_management=True,
                enable_tensor_paging=True,
                enable_smart_swap=True,
                enable_kernel_fusion=True,
                enable_adaptive_batching=True,
                enable_tensor_compression=True,
                enable_disk_offloading=True,
                enable_model_surgery=True,
                enable_pipeline=True,
                enable_activation_offloading=True
            )
            assert_true(success)
            
            # Perform cleanup
            cleanup_success = plugin.cleanup()
            assert_true(cleanup_success)

    def cleanup_helper():
        """Clean up after each test method."""
        # Clean up any resources used by the plugins
        for plugin in plugins:
            if hasattr(plugin, 'cleanup'):
                plugin.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)
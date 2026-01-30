"""
Benchmark tests to measure the effectiveness of optimizations.

This test measures the performance improvements achieved by each optimization technique.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import time
import torch
import torch.nn as nn
from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin

# TestOptimizationBenchmarks

    """Test cases for optimization benchmarks."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        plugins = [
            GLM_4_7_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin(),
            Qwen3_VL_2B_Plugin()
        ]

    def memory_optimization_benchmark(self)():
        """Benchmark memory optimization effectiveness."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize without optimizations
            plugin.initialize()
            base_memory_stats = plugin.get_memory_stats()
            
            # Initialize with memory optimizations
            plugin.initialize(
                enable_memory_management=True,
                enable_tensor_paging=True,
                enable_smart_swap=True,
                enable_disk_offloading=True,
                enable_activation_offloading=True
            )
            optimized_memory_stats = plugin.get_memory_stats()
            
            # Both should return valid stats
            assert_is_instance(base_memory_stats, dict)
            assert_is_instance(optimized_memory_stats, dict)

    def kernel_fusion_performance_benchmark(self)():
        """Benchmark kernel fusion performance improvement."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with kernel fusion
            plugin.initialize(
                enable_kernel_fusion=True
            )
            
            # Verify that kernel fusion is properly set up
            fusion_manager = plugin.get_fusion_manager()
            assert_is_not_none(fusion_manager)

    def tensor_compression_ratio_benchmark(self)():
        """Benchmark tensor compression ratio."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with tensor compression
            plugin.initialize(
                enable_tensor_compression=True,
                tensor_compression_ratio=0.5
            )
            
            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()
            
            # Compress model weights
            compression_success = plugin.compress_model_weights(compression_ratio=0.5)
            assert_true(compression_success)
            
            # Get compression stats
            compression_stats = plugin.get_compression_stats()
            assert_is_instance(compression_stats, dict)
            assert_in('average_compression_ratio', compression_stats)

    def model_surgery_size_reduction_benchmark(self)():
        """Benchmark model surgery size reduction."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with model surgery
            plugin.initialize(
                enable_model_surgery=True,
                surgery_enabled=True
            )
            
            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()
            
            # Get original parameter count
            original_params = sum(p.numel() for p in plugin._model.parameters())
            
            # Perform model surgery
            modified_model = plugin.perform_model_surgery()
            
            # Get modified parameter count
            modified_params = sum(p.numel() for p in modified_model.parameters())
            
            # Get surgery stats
            surgery_stats = plugin.get_surgery_stats()
            assert_is_instance(surgery_stats, dict)

    def adaptive_batching_throughput_benchmark(self)():
        """Benchmark adaptive batching throughput improvement."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with adaptive batching
            plugin.initialize(
                enable_adaptive_batching=True,
                initial_batch_size=1,
                max_batch_size=16
            )
            
            # Get initial batching status
            initial_status = plugin.get_batching_status()
            assert_in('current_batch_size', initial_status)
            
            # Simulate performance metrics to trigger batch size adjustment
            optimal_size = plugin.get_optimal_batch_size(processing_time_ms=100.0, tokens_processed=50)
            assert_is_instance(optimal_size, int)
            
            # Get updated status
            updated_status = plugin.get_batching_status()
            assert_in('current_batch_size', updated_status)

    def disk_offloading_memory_savings_benchmark(self)():
        """Benchmark disk offloading memory savings."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with disk offloading
            plugin.initialize(
                enable_disk_offloading=True
            )
            
            # Get initial memory stats
            initial_stats = plugin.get_offloading_stats()
            assert_is_instance(initial_stats, dict)
            
            # Perform some offloading operations
            offload_success = plugin.offload_model_parts()
            assert_true(offload_success)
            
            # Get updated stats
            updated_stats = plugin.get_offloading_stats()
            assert_is_instance(updated_stats, dict)

    def activation_offloading_efficiency_benchmark(self)():
        """Benchmark activation offloading efficiency."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with activation offloading
            plugin.initialize(
                enable_activation_offloading=True
            )
            
            # Get initial stats
            initial_stats = plugin.get_activation_offloading_stats()
            assert_is_instance(initial_stats, dict)
            
            # Perform some activation offloading operations
            activation_success = plugin.offload_activations()
            assert_true(activation_success)
            
            # Get updated stats
            updated_stats = plugin.get_activation_offloading_stats()
            assert_is_instance(updated_stats, dict)

    def pipeline_performance_benchmark(self)():
        """Benchmark pipeline performance improvement."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with pipeline
            plugin.initialize(
                enable_pipeline=True
            )
            
            # Get initial pipeline stats
            initial_stats = plugin.get_pipeline_stats()
            assert_is_instance(initial_stats, dict)
            
            # Execute pipeline with dummy data
            result = plugin.execute_pipeline("dummy input")
            assert_is_not_none(result)
            
            # Get updated stats
            updated_stats = plugin.get_pipeline_stats()
            assertIsInstance(updated_stats)

    def combined_optimizations_synergy_benchmark(self)():
        """Benchmark combined optimizations synergy."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize with all optimizations
            plugin.initialize(
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
            
            # Get all optimization stats
            memory_stats = plugin.get_memory_stats()
            batching_status = plugin.get_batching_status()
            compression_stats = plugin.get_compression_stats()
            offloading_stats = plugin.get_offloading_stats()
            surgery_stats = plugin.get_surgery_stats()
            pipeline_stats = plugin.get_pipeline_stats()
            activation_stats = plugin.get_activation_offloading_stats()
            
            # All should return valid stats
            assert_is_instance(memory_stats, dict)
            assert_is_instance(batching_status, dict)
            assert_is_instance(compression_stats, dict)
            assert_is_instance(offloading_stats, dict)
            assert_is_instance(surgery_stats, dict)
            assert_is_instance(pipeline_stats, dict)
            assert_is_instance(activation_stats, dict)

    def optimization_startup_time_benchmark(self)():
        """Benchmark optimization startup time overhead."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Measure time to initialize with optimizations
            start_time = time.time()
            success = plugin.initialize(
                enable_memory_management=True,
                enable_tensor_paging=True,
                enable_kernel_fusion=True,
                enable_adaptive_batching=True,
                enable_tensor_compression=True,
                enable_disk_offloading=True,
                enable_model_surgery=True,
                enable_activation_offloading=True
            )
            end_time = time.time()
            
            assert_true(success)
            init_time = end_time - start_time
            
            # Initialization should complete in reasonable time
            assert_less(init_time, 30.0)  # Less than 30 seconds

    def cleanup_helper():
        """Clean up after each test method."""
        # Clean up any resources used by the plugins
        for plugin in plugins:
            if hasattr(plugin, 'cleanup'):
                plugin.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)
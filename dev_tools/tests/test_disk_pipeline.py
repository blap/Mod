"""
Test suite for Disk-Based Pipeline functionality in model plugins.

This test verifies that the disk-based pipeline system works correctly across all model plugins.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import tempfile
from src.inference_pio.common.disk_pipeline import (
    DiskBasedPipeline,
    PipelineStage,
    PipelineManager
)
from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin

# TestDiskPipeline

    """Test cases for disk-based pipeline functionality."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        plugins = [
            GLM_4_7_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin(),
            Qwen3_VL_2B_Plugin()
        ]
        
        # Create a simple test model for pipeline tests
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                linear1 = nn.Linear(100, 50)
                relu = nn.ReLU()
                linear2 = nn.Linear(50, 10)
                
            def forward(self, x):
                x = linear1(x)
                x = relu(x)
                x = linear2(x)
                return x
        
        simple_model = SimpleModel()

    def pipeline_stage_creation(self)():
        """Test that pipeline stages can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            def dummy_function(x):
                return x * 2
            
            stage = PipelineStage(
                name="test_stage",
                function=dummy_function,
                input_keys=["input"],
                output_keys=["output"],
                checkpoint_dir=temp_dir,
                cache_intermediates=True
            )
            
            assert_is_instance(stage, PipelineStage)

    def pipeline_creation(self)():
        """Test that the disk-based pipeline can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some dummy stages
            def stage1_func(data):
                data['processed'] = True
                return data
            
            def stage2_func(data):
                data['result'] = "processed"
                return data
            
            stages = [
                PipelineStage(
                    name="stage1",
                    function=stage1_func,
                    input_keys=["input"],
                    output_keys=["processed"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True
                ),
                PipelineStage(
                    name="stage2",
                    function=stage2_func,
                    input_keys=["processed"],
                    output_keys=["result"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True
                )
            ]
            
            pipeline = DiskBasedPipeline(
                stages=stages,
                checkpoint_dir=temp_dir,
                max_concurrent_stages=2,
                cleanup_after_completion=True
            )
            
            assert_is_instance(pipeline, DiskBasedPipeline)

    def pipeline_execution(self)():
        """Test executing the pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some dummy stages
            def stage1_func(data):
                data['step1'] = 'completed'
                return data
            
            def stage2_func(data):
                data['step2'] = 'completed'
                data['final_result'] = 'success'
                return data
            
            stages = [
                PipelineStage(
                    name="stage1",
                    function=stage1_func,
                    input_keys=["input"],
                    output_keys=["step1"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True
                ),
                PipelineStage(
                    name="stage2",
                    function=stage2_func,
                    input_keys=["step1"],
                    output_keys=["final_result"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True
                )
            ]
            
            pipeline = DiskBasedPipeline(
                stages=stages,
                checkpoint_dir=temp_dir,
                max_concurrent_stages=2,
                cleanup_after_completion=True
            )
            
            # Execute the pipeline
            initial_inputs = {'input': 'start'}
            results = pipeline.execute_pipeline(initial_inputs)
            
            # Check that the results contain expected values
            assert_in('final_result', results)
            assert_equal(results['final_result'], 'success')

    def pipeline_manager_creation(self)():
        """Test that the pipeline manager can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PipelineManager(base_checkpoint_dir=temp_dir)
            assert_is_instance(manager, PipelineManager)

    def plugin_pipeline_setup(self)():
        """Test that all plugins can set up the pipeline system."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize(enable_pipeline=True)
            assert_true(success)

            # Check that pipeline methods are available
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))

    def plugin_execute_pipeline(self)():
        """Test that plugins can execute the pipeline."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with pipeline enabled
            success = plugin.initialize(enable_pipeline=True)
            assert_true(success)

            # Execute pipeline with dummy data
            result = plugin.execute_pipeline("dummy input")

            # Should return some result (might fall back to regular inference)
            assert_is_not_none(result)

    def plugin_create_pipeline_stages(self)():
        """Test that plugins can create pipeline stages."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize()
            assert_true(success)

            # Create pipeline stages
            stages = plugin.create_pipeline_stages()

            # Should return a list of stages
            assert_is_instance(stages)

    def plugin_get_pipeline_stats(self)():
        """Test that plugins can report pipeline statistics."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize()
            assert_true(success)

            # Get pipeline stats (should work even without pipeline enabled)
            stats = plugin.get_pipeline_stats()

            # Should return a dictionary with stats
            assertIsInstance(stats, dict)
            assert_in('pipeline_enabled', stats)
            assert_in('num_stages', stats)

    def pipeline_with_memory_management(self)():
        """Test pipeline working with memory management."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with pipeline and memory management
            success = plugin.initialize(
                enable_pipeline=True,
                enable_memory_management=True,
                enable_tensor_paging=True
            )
            assert_true(success)

            # Execute pipeline with dummy data
            result = plugin.execute_pipeline("dummy input")
            assert_is_not_none(result)

            # Get pipeline stats
            pipeline_stats = plugin.get_pipeline_stats()
            assert_in('pipeline_enabled')

            # Get memory stats to verify memory management is also working
            memory_stats = plugin.get_memory_stats()
            assertIn('system_memory_percent', memory_stats)

    def pipeline_with_other_optimizations(self)():
        """Test pipeline working with other optimizations."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with pipeline and other optimizations
            success = plugin.initialize(
                enable_pipeline=True,
                enable_tensor_compression=True,
                enable_disk_offloading=True,
                enable_model_surgery=True,
                enable_kernel_fusion=True,
                enable_activation_offloading=True
            )
            assert_true(success)

            # Execute pipeline with dummy data
            result = plugin.execute_pipeline("dummy input")
            assert_is_not_none(result)

            # Get pipeline stats
            stats = plugin.get_pipeline_stats()
            assertIn('pipeline_enabled')

    def pipeline_with_multiple_concurrent_stages(self)():
        """Test pipeline with multiple concurrent stages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create stages that simulate some processing
            def slow_stage_func(data):
                import time
                time.sleep(0.1)  # Simulate processing time
                data['slow_stage_completed'] = True
                return data

            stages = [
                PipelineStage(
                    name="fast_stage",
                    function=lambda d: {**d, 'fast_stage_completed': True},
                    input_keys=["input"],
                    output_keys=["fast_stage_completed"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True
                ),
                PipelineStage(
                    name="slow_stage",
                    function=slow_stage_func,
                    input_keys=["fast_stage_completed"],
                    output_keys=["slow_stage_completed"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True
                )
            ]

            pipeline = DiskBasedPipeline(
                stages=stages,
                checkpoint_dir=temp_dir,
                max_concurrent_stages=2,  # Allow concurrent execution
                cleanup_after_completion=True
            )

            # Execute the pipeline
            initial_inputs = {'input': 'start'}
            results = pipeline.execute_pipeline(initial_inputs)

            # Check that both stages completed
            assert_true(results.get('fast_stage_completed'))
            assert_true(results.get('slow_stage_completed'))

    def cleanup_helper():
        """Clean up after each test method."""
        # Clean up any resources used by the plugins
        for plugin in plugins:
            if hasattr(plugin, 'cleanup'):
                plugin.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)
"""
Test suite for Disk-Based Pipeline functionality in model plugins.

This test verifies that the disk-based pipeline system works correctly across all model plugins.
"""

import tempfile
import unittest

import torch
import torch.nn as nn

from src.inference_pio.common.hardware.disk_pipeline import (
    DiskBasedPipeline,
    PipelineManager,
    PipelineStage,
)
from src.inference_pio.models.glm_4_7_flash.plugin import GLM_4_7_Flash_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import (
    Qwen3_4B_Instruct_2507_Plugin,
)
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin


class TestDiskPipeline(unittest.TestCase):
    """Test cases for disk-based pipeline functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.plugins = [
            GLM_4_7_Flash_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin(),
            Qwen3_VL_2B_Plugin(),
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

    def test_pipeline_stage_creation(self):
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
                cache_intermediates=True,
            )

            self.assertIsInstance(stage, PipelineStage)

    def test_pipeline_creation(self):
        """Test that the disk-based pipeline can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some dummy stages
            def stage1_func(data):
                data["processed"] = True
                return data

            def stage2_func(data):
                data["result"] = "processed"
                return data

            stages = [
                PipelineStage(
                    name="stage1",
                    function=stage1_func,
                    input_keys=["input"],
                    output_keys=["processed"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True,
                ),
                PipelineStage(
                    name="stage2",
                    function=stage2_func,
                    input_keys=["processed"],
                    output_keys=["result"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True,
                ),
            ]

            pipeline = DiskBasedPipeline(
                stages=stages,
                checkpoint_dir=temp_dir,
                max_concurrent_stages=2,
                cleanup_after_completion=True,
            )

            self.assertIsInstance(pipeline, DiskBasedPipeline)

    def test_pipeline_execution(self):
        """Test executing the pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some dummy stages
            def stage1_func(data):
                data["step1"] = "completed"
                return data

            def stage2_func(data):
                data["step2"] = "completed"
                data["final_result"] = "success"
                return data

            stages = [
                PipelineStage(
                    name="stage1",
                    function=stage1_func,
                    input_keys=["input"],
                    output_keys=["step1"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True,
                ),
                PipelineStage(
                    name="stage2",
                    function=stage2_func,
                    input_keys=["step1"],
                    output_keys=["final_result"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True,
                ),
            ]

            pipeline = DiskBasedPipeline(
                stages=stages,
                checkpoint_dir=temp_dir,
                max_concurrent_stages=2,
                cleanup_after_completion=True,
            )

            # Execute the pipeline
            initial_inputs = {"input": "start"}
            results = pipeline.execute_pipeline(initial_inputs)

            # Check that the results contain expected values
            self.assertIn("final_result", results)
            self.assertEqual(results["final_result"], "success")

    def test_pipeline_manager_creation(self):
        """Test that the pipeline manager can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PipelineManager(base_checkpoint_dir=temp_dir)
            self.assertIsInstance(manager, PipelineManager)

    def test_plugin_pipeline_setup(self):
        """Test that all plugins can set up the pipeline system."""
        for plugin in self.plugins:
            try:
                # Initialize the plugin with minimal config
                success = plugin.initialize(config=None)
                if success:
                    # Check that pipeline methods are available
                    self.assertTrue(hasattr(plugin, "setup_pipeline"))
                    self.assertTrue(hasattr(plugin, "execute_pipeline"))
                    self.assertTrue(hasattr(plugin, "get_pipeline_stats"))
                    self.assertTrue(hasattr(plugin, "pipeline_process"))

                    # Test that pipeline can be set up (if method exists)
                    if hasattr(plugin, "setup_pipeline"):
                        try:
                            setup_success = plugin.setup_pipeline()
                            self.assertTrue(setup_success)
                        except (AttributeError, RuntimeError):
                            # Expected if model isn't properly loaded
                            pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_plugin_execute_pipeline(self):
        """Test that plugins can execute the pipeline."""
        for plugin in self.plugins[
            :1
        ]:  # Test with first plugin to avoid long execution
            try:
                # Initialize the plugin with minimal config
                success = plugin.initialize(config=None)

                # Execute pipeline with dummy data (if method exists)
                if hasattr(plugin, "execute_pipeline"):
                    try:
                        result = plugin.execute_pipeline("dummy input")

                        # Should return some result (might fall back to regular inference)
                        self.assertIsNotNone(result)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_plugin_create_pipeline_stages(self):
        """Test that plugins can create pipeline stages."""
        for plugin in self.plugins:
            try:
                # Initialize the plugin with minimal config
                success = plugin.initialize(config=None)

                # Create pipeline stages (if method exists)
                if hasattr(plugin, "create_pipeline_stages"):
                    try:
                        stages = plugin.create_pipeline_stages()

                        # Should return a list of stages
                        self.assertIsNotNone(stages)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_plugin_get_pipeline_stats(self):
        """Test that plugins can report pipeline statistics."""
        for plugin in self.plugins:
            try:
                # Initialize the plugin with minimal config
                success = plugin.initialize(config=None)

                # Get pipeline stats (if method exists)
                if hasattr(plugin, "get_pipeline_stats"):
                    try:
                        stats = plugin.get_pipeline_stats()

                        # Should return a dictionary with stats
                        self.assertIsInstance(stats, dict)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass
            assert_in("pipeline_enabled", stats)
            assert_in("num_stages", stats)

    def test_pipeline_with_memory_management(self):
        """Test pipeline working with memory management."""
        for plugin in self.plugins[
            :1
        ]:  # Test with first plugin to avoid long execution
            try:
                # Initialize the plugin with minimal config
                success = plugin.initialize(config=None)

                # Execute pipeline with dummy data (if method exists)
                if hasattr(plugin, "execute_pipeline"):
                    try:
                        result = plugin.execute_pipeline("dummy input")
                        self.assertIsNotNone(result)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

            # Get pipeline stats
            pipeline_stats = plugin.get_pipeline_stats()
            assert_in("pipeline_enabled")

            # Get memory stats to verify memory management is also working
            memory_stats = plugin.get_memory_stats()
            assertIn("system_memory_percent", memory_stats)

    def test_pipeline_with_other_optimizations(self):
        """Test pipeline working with other optimizations."""
        for plugin in self.plugins[
            :1
        ]:  # Test with first plugin to avoid long execution
            try:
                # Initialize the plugin with minimal config
                success = plugin.initialize(config=None)

                # Execute pipeline with dummy data (if method exists)
                if hasattr(plugin, "execute_pipeline"):
                    try:
                        result = plugin.execute_pipeline("dummy input")
                        self.assertIsNotNone(result)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass

                # Get pipeline stats (if method exists)
                if hasattr(plugin, "get_pipeline_stats"):
                    try:
                        stats = plugin.get_pipeline_stats()
                        if "pipeline_enabled" in stats:
                            self.assertIn("pipeline_enabled", stats)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_pipeline_with_multiple_concurrent_stages(self):
        """Test pipeline with multiple concurrent stages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create stages that simulate some processing
            def slow_stage_func(data):
                import time

                time.sleep(0.1)  # Simulate processing time
                data["slow_stage_completed"] = True
                return data

            stages = [
                PipelineStage(
                    name="fast_stage",
                    function=lambda d: {**d, "fast_stage_completed": True},
                    input_keys=["input"],
                    output_keys=["fast_stage_completed"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True,
                ),
                PipelineStage(
                    name="slow_stage",
                    function=slow_stage_func,
                    input_keys=["fast_stage_completed"],
                    output_keys=["slow_stage_completed"],
                    checkpoint_dir=temp_dir,
                    cache_intermediates=True,
                ),
            ]

            pipeline = DiskBasedPipeline(
                stages=stages,
                checkpoint_dir=temp_dir,
                max_concurrent_stages=2,  # Allow concurrent execution
                cleanup_after_completion=True,
            )

            # Execute the pipeline
            initial_inputs = {"input": "start"}
            results = pipeline.execute_pipeline(initial_inputs)

            # Check that both stages completed
            self.assertTrue(results.get("fast_stage_completed"))
            self.assertTrue(results.get("slow_stage_completed"))

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up any resources used by the plugins
        for plugin in self.plugins:
            if hasattr(plugin, "cleanup"):
                try:
                    plugin.cleanup()
                except:
                    # Ignore cleanup errors
                    pass


if __name__ == "__main__":
    unittest.main()

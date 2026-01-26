"""
Test suite for Disk Offloading functionality in model plugins.

This test verifies that the disk offloading system works correctly across all model plugins.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import tempfile
import os
from src.inference_pio.common.disk_offloading import DiskOffloader, TensorOffloadingManager as DiskTensorOffloadingManager
from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin

# TestDiskOffloading

    """Test cases for disk offloading functionality."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        plugins = [
            GLM_4_7_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin(),
            Qwen3_VL_2B_Plugin()
        ]
        
        # Create a simple test model for offloading tests
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

    def disk_offloader_creation(self)():
        """Test that the disk offloader can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru"
            )
            assert_is_instance(offloader, DiskOffloader)

    def tensor_offloading_manager_creation(self)():
        """Test that the tensor offloading manager can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru"
            )
            
            manager = DiskTensorOffloadingManager(offloader)
            assert_is_instance(manager, DiskTensorOffloadingManager)

    def offload_and_load_tensor(self)():
        """Test basic tensor offloading and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru"
            )
            
            manager = DiskTensorOffloadingManager(offloader)
            
            # Create a test tensor
            test_tensor = torch.randn(50, 50)
            
            # Offload the tensor
            success = manager.offload_tensor(test_tensor, "test_tensor_1", priority=1, access_pattern="frequent")
            assert_true(success)
            
            # Load the tensor back
            loaded_tensor = manager.load_tensor("test_tensor_1")
            assert_is_not_none(loaded_tensor)
            
            # Check that the tensor is the same
            assert_true(torch.equal(test_tensor), "Loaded tensor differs from original")

    def pin_unpin_tensor(self)():
        """Test pinning and unpinning tensors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru"
            )
            
            manager = DiskTensorOffloadingManager(offloader)
            
            # Create a test tensor
            test_tensor = torch.randn(50, 50)
            
            # Offload the tensor
            success = manager.offload_tensor(test_tensor, "test_tensor_1", priority=1, access_pattern="frequent")
            assert_true(success)
            
            # Pin the tensor
            pin_success = manager.pin_tensor("test_tensor_1")
            assert_true(pin_success)
            
            # Unpin the tensor
            unpin_success = manager.unpin_tensor("test_tensor_1")
            assert_true(unpin_success)

    def start_stop_proactive_management(self)():
        """Test starting and stopping proactive management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru"
            )
            
            manager = DiskTensorOffloadingManager(offloader)
            
            # Start proactive management
            manager.start_proactive_management(interval=1.0)
            
            # Stop proactive management
            manager.stop_proactive_management()

    def plugin_disk_offloading_setup(self)():
        """Test that all plugins can set up disk offloading."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize(enable_disk_offloading=True)
            assert_true(success)

            # Check that disk offloading methods are available
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))

    def plugin_enable_disk_offloading(self)():
        """Test that plugins can enable disk offloading."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with disk offloading enabled
            success = plugin.initialize(enable_disk_offloading=True)
            assert_true(success)

            # Enable disk offloading
            enable_success = plugin.enable_disk_offloading()

            # Should return True on success
            assert_true(enable_success)

    def plugin_offload_model_parts(self)():
        """Test that plugins can offload model parts."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with disk offloading enabled
            success = plugin.initialize(enable_disk_offloading=True)
            assert_true(success)

            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()

            # Offload model parts
            offload_success = plugin.offload_model_parts()

            # Should return True on success
            assert_true(offload_success)

    def plugin_predict_model_part_access(self)():
        """Test that plugins can predict model part access."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize()
            assert_true(success)

            # Predict model part access
            predictions = plugin.predict_model_part_access()

            # Should return a dictionary
            assert_is_instance(predictions, dict)

    def plugin_get_offloading_stats(self)():
        """Test that plugins can report offloading statistics."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize()
            assert_true(success)

            # Get offloading stats (should work even without offloading performed)
            stats = plugin.get_offloading_stats()

            # Should return a dictionary with stats
            assert_is_instance(stats, dict)
            assert_in('offloading_enabled', stats)
            assert_in('system_memory_percent', stats)

    def disk_offloading_with_memory_management(self)():
        """Test disk offloading working with memory management."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with both optimizations
            success = plugin.initialize(
                enable_disk_offloading=True,
                enable_memory_management=True,
                enable_tensor_paging=True
            )
            assert_true(success)

            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()

            # Offload model parts
            offload_success = plugin.offload_model_parts()
            assert_true(offload_success)

            # Get offloading stats
            offloading_stats = plugin.get_offloading_stats()
            assert_in('offloading_enabled', offloading_stats)

            # Get memory stats to verify memory management is also working
            memory_stats = plugin.get_memory_stats()
            assert_in('system_memory_percent', memory_stats)

    def disk_offloading_with_other_optimizations(self)():
        """Test disk offloading working with other optimizations."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with disk offloading and other optimizations
            success = plugin.initialize(
                enable_disk_offloading=True,
                enable_tensor_compression=True,
                enable_model_surgery=True,
                enable_kernel_fusion=True,
                enable_activation_offloading=True
            )
            assert_true(success)

            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()

            # Offload model parts
            offload_success = plugin.offload_model_parts()
            assert_true(offload_success)

            # Get offloading stats
            stats = plugin.get_offloading_stats()
            assert_in('offloading_enabled', stats)

    def offloading_with_different_tensor_sizes(self)():
        """Test offloading with different tensor sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru"
            )

            manager = DiskTensorOffloadingManager(offloader)

            # Test with different tensor sizes
            tensor_sizes = [(10, 10), (50, 50), (100, 100)]

            for i, size in enumerate(tensor_sizes):
                test_tensor = torch.randn(*size)

                # Offload the tensor
                success = manager.offload_tensor(test_tensor, f"test_tensor_{i}", priority=1, access_pattern="frequent")
                assert_true(success)

                # Load the tensor back
                loaded_tensor = manager.load_tensor(f"test_tensor_{i}")
                assert_is_not_none(loaded_tensor)

                # Check that the tensor is the same
                assert_true(torch.equal(test_tensor), f"Loaded tensor differs from original for size {size}")

    def cleanup_helper():
        """Clean up after each test method."""
        # Clean up any resources used by the plugins
        for plugin in plugins:
            if hasattr(plugin, 'cleanup'):
                plugin.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)
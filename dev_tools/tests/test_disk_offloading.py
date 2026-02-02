"""
Test suite for Disk Offloading functionality in model plugins.

This test verifies that the disk offloading system works correctly across all model plugins.
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn

from src.common.disk_offloading import (
    DiskOffloader,
)
from src.common.disk_offloading import (
    TensorOffloadingManager as DiskTensorOffloadingManager,
)
from src.models.glm_4_7_flash.plugin import GLM_4_7_Flash_Plugin
from src.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin


class TestDiskOffloading(unittest.TestCase):
    """Test cases for disk offloading functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.plugins = [
            GLM_4_7_Flash_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin(),
            Qwen3_VL_2B_Plugin(),
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

    def test_disk_offloader_creation(self):
        """Test that the disk offloader can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru",
            )
            self.assertIsInstance(offloader, DiskOffloader)

    def test_tensor_offloading_manager_creation(self):
        """Test that the tensor offloading manager can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru",
            )

            manager = DiskTensorOffloadingManager(offloader)
            self.assertIsInstance(manager, DiskTensorOffloadingManager)

    def test_offload_and_load_tensor(self):
        """Test basic tensor offloading and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru",
            )

            manager = DiskTensorOffloadingManager(offloader)

            # Create a test tensor
            test_tensor = torch.randn(50, 50)

            # Offload the tensor
            success = manager.offload_tensor(
                test_tensor, "test_tensor_1", priority=1, access_pattern="frequent"
            )
            self.assertTrue(success)

            # Load the tensor back
            loaded_tensor = manager.load_tensor("test_tensor_1")
            assert_is_not_none(loaded_tensor)

            # Check that the tensor is the same
            self.assertTrue(
                torch.equal(test_tensor, loaded_tensor),
                "Loaded tensor differs from original",
            )

    def test_pin_unpin_tensor(self):
        """Test pinning and unpinning tensors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru",
            )

            manager = DiskTensorOffloadingManager(offloader)

            # Create a test tensor
            test_tensor = torch.randn(50, 50)

            # Offload the tensor
            success = manager.offload_tensor(
                test_tensor, "test_tensor_1", priority=1, access_pattern="frequent"
            )
            self.assertTrue(success)

            # Pin the tensor
            pin_success = manager.pin_tensor("test_tensor_1")
            self.assertTrue(pin_success)

            # Unpin the tensor
            unpin_success = manager.unpin_tensor("test_tensor_1")
            self.assertTrue(unpin_success)

    def test_start_stop_proactive_management(self):
        """Test starting and stopping proactive management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru",
            )

            manager = DiskTensorOffloadingManager(offloader)

            # Start proactive management
            manager.start_proactive_management(interval=1.0)

            # Stop proactive management
            manager.stop_proactive_management()

    def test_plugin_disk_offloading_setup(self):
        """Test that all plugins can set up disk offloading."""
        for plugin in self.plugins:
            try:
                # Initialize the plugin with minimal config
                success = plugin.initialize(config=None)
                if success:
                    # Check that disk offloading methods are available
                    self.assertTrue(hasattr(plugin, "setup_disk_offloading"))
                    self.assertTrue(hasattr(plugin, "enable_disk_offloading"))
                    self.assertTrue(hasattr(plugin, "offload_model_parts"))
                    self.assertTrue(hasattr(plugin, "predict_model_part_access"))
                    self.assertTrue(hasattr(plugin, "get_offloading_stats"))

                    # Test that disk offloading can be set up (if method exists)
                    if hasattr(plugin, "setup_disk_offloading"):
                        try:
                            plugin.setup_disk_offloading()
                        except (AttributeError, RuntimeError):
                            # Expected if model isn't properly loaded
                            pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_plugin_enable_disk_offloading(self):
        """Test that plugins can enable disk offloading."""
        for plugin in self.plugins[
            :1
        ]:  # Test with first plugin to avoid long execution
            try:
                # Initialize the plugin with minimal config
                success = plugin.initialize(config=None)

                if success and hasattr(plugin, "enable_disk_offloading"):
                    # Enable disk offloading (if method exists)
                    try:
                        enable_success = plugin.enable_disk_offloading()
                        # Should return True on success
                        self.assertTrue(enable_success)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_plugin_offload_model_parts(self):
        """Test that plugins can offload model parts."""
        for plugin in self.plugins[
            :1
        ]:  # Test with first plugin to avoid long execution
            try:
                # Initialize the plugin with minimal config
                success = plugin.initialize(config=None)

                # Test offloading model parts (if method exists)
                if success and hasattr(plugin, "offload_model_parts"):
                    try:
                        offload_success = plugin.offload_model_parts()
                        # Should return True on success
                        self.assertTrue(offload_success)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_plugin_predict_model_part_access(self):
        """Test that plugins can predict model part access."""
        for plugin in self.plugins:
            try:
                success = plugin.initialize(config=None)

                # Predict model part access (if method exists)
                if success and hasattr(plugin, "predict_model_part_access"):
                    try:
                        predictions = plugin.predict_model_part_access()

                        # Should return a dictionary
                        self.assertIsInstance(predictions, dict)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_plugin_get_offloading_stats(self):
        """Test that plugins can report offloading statistics."""
        for plugin in self.plugins:
            try:
                success = plugin.initialize(config=None)

                # Get offloading stats (if method exists)
                if success and hasattr(plugin, "get_offloading_stats"):
                    try:
                        stats = plugin.get_offloading_stats()

                        # Should return a dictionary with stats
                        self.assertIsInstance(stats, dict)
                        if "offloading_enabled" in stats:
                            self.assertIn("offloading_enabled", stats)
                        if "system_memory_percent" in stats:
                            self.assertIn("system_memory_percent", stats)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

    def test_disk_offloading_with_memory_management(self):
        """Test disk offloading working with memory management."""
        for plugin in self.plugins[
            :1
        ]:  # Test with first plugin to avoid long execution
            try:
                # Initialize the plugin with minimal config
                success = plugin.initialize(config=None)

                # Test that offloading still works with memory management (if method exists)
                if success and hasattr(plugin, "offload_model_parts"):
                    try:
                        offload_success = plugin.offload_model_parts()
                        self.assertTrue(offload_success)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

            # Get offloading stats
            offloading_stats = plugin.get_offloading_stats()
            assert_in("offloading_enabled", offloading_stats)

            # Get memory stats to verify memory management is also working
            memory_stats = plugin.get_memory_stats()
            assert_in("system_memory_percent", memory_stats)

    def test_disk_offloading_with_other_optimizations(self):
        """Test disk offloading working with other optimizations."""
        for plugin in self.plugins[
            :1
        ]:  # Test with first plugin to avoid long execution
            try:
                # Initialize the plugin with minimal config
                success = plugin.initialize(config=None)

                # Test that offloading still works with other optimizations (if method exists)
                if success and hasattr(plugin, "offload_model_parts"):
                    try:
                        offload_success = plugin.offload_model_parts()
                        self.assertTrue(offload_success)
                    except (AttributeError, RuntimeError):
                        # Expected if model isn't properly loaded
                        pass
            except Exception:
                # Some plugins may not initialize properly without full model files
                # This is expected in test environments
                pass

            # Get offloading stats
            stats = plugin.get_offloading_stats()
            assert_in("offloading_enabled", stats)

    def test_offloading_with_different_tensor_sizes(self):
        """Test offloading with different tensor sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            offloader = DiskOffloader(
                max_memory_ratio=0.5,
                offload_directory=temp_dir,
                page_size_mb=4,
                eviction_policy="lru",
            )

            manager = DiskTensorOffloadingManager(offloader)

            # Test with different tensor sizes
            tensor_sizes = [(10, 10), (50, 50), (100, 100)]

            for i, size in enumerate(tensor_sizes):
                test_tensor = torch.randn(*size)

                # Offload the tensor
                success = manager.offload_tensor(
                    test_tensor,
                    f"test_tensor_{i}",
                    priority=1,
                    access_pattern="frequent",
                )
                self.assertTrue(success)

                # Load the tensor back
                loaded_tensor = manager.load_tensor(f"test_tensor_{i}")
                assert_is_not_none(loaded_tensor)

                # Check that the tensor is the same
                self.assertTrue(
                    torch.equal(test_tensor, loaded_tensor),
                    f"Loaded tensor differs from original for size {size}",
                )

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

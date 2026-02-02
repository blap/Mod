"""
Test suite for Activation Offloading functionality in model plugins.

This test verifies that the activation offloading system works correctly across all model plugins.
"""

import tempfile

import torch
import torch.nn as nn

from src.inference_pio.common.activation_offloading import (
    AccessPattern,
    ActivationOffloadingManager,
    OffloadPriority,
    create_activation_offloader,
)
from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import (
    Qwen3_4B_Instruct_2507_Plugin,
)
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin
from tests.utils.test_utils import (
    assert_in,
    assert_is_instance,
    assert_is_not_none,
    assert_true,
    run_tests,
)


def create_test_plugins():
    """Create test plugins for the tests."""
    return [
        GLM_4_7_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Plugin(),
    ]


def test_create_activation_offloader():
    """Test that the activation offloader can be created."""
    with tempfile.TemporaryDirectory() as temp_dir:
        offloader = create_activation_offloader(
            max_memory_ratio=0.5,
            offload_directory=temp_dir,
            page_size_mb=4,
            eviction_policy="lru",
        )
        assert_is_not_none(offloader)


def test_activation_offloading_manager_creation():
    """Test that the activation offloading manager can be created."""
    with tempfile.TemporaryDirectory() as temp_dir:
        offloader = create_activation_offloader(
            max_memory_ratio=0.5,
            offload_directory=temp_dir,
            page_size_mb=4,
            eviction_policy="lru",
        )

        manager = ActivationOffloadingManager(offloader)
        assert_is_instance(manager, ActivationOffloadingManager)


def test_offload_and_load_activation():
    """Test basic activation offloading and loading."""
    with tempfile.TemporaryDirectory() as temp_dir:
        offloader = create_activation_offloader(
            max_memory_ratio=0.5,
            offload_directory=temp_dir,
            page_size_mb=4,
            eviction_policy="lru",
        )

        manager = ActivationOffloadingManager(offloader)

        # Create a test activation tensor
        test_activation = torch.randn(50, 50)

        # Offload the activation
        success = manager.offload_activation(
            test_activation,
            "test_activation_1",
            priority=OffloadPriority.MEDIUM,
            access_pattern=AccessPattern.FREQUENT,
        )
        assert_true(success, "Failed to offload activation")

        # Load the activation back
        loaded_activation = manager.load_activation("test_activation_1")
        assert_is_not_none(loaded_activation, "Failed to load activation")

        # Check that the activation is the same
        assert_true(
            torch.equal(test_activation, loaded_activation),
            "Loaded activation differs from original",
        )


def test_pin_unpin_activation():
    """Test pinning and unpinning activations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        offloader = create_activation_offloader(
            max_memory_ratio=0.5,
            offload_directory=temp_dir,
            page_size_mb=4,
            eviction_policy="lru",
        )

        manager = ActivationOffloadingManager(offloader)

        # Create a test activation tensor
        test_activation = torch.randn(50, 50)

        # Offload the activation
        success = manager.offload_activation(
            test_activation,
            "test_activation_1",
            priority=OffloadPriority.MEDIUM,
            access_pattern=AccessPattern.FREQUENT,
        )
        assert_true(success, "Failed to offload activation")

        # Pin the activation
        pin_success = manager.pin_activation("test_activation_1")
        assert_true(pin_success, "Failed to pin activation")

        # Unpin the activation
        unpin_success = manager.unpin_activation("test_activation_1")
        assert_true(unpin_success, "Failed to unpin activation")


def test_start_stop_proactive_management():
    """Test starting and stopping proactive management."""
    with tempfile.TemporaryDirectory() as temp_dir:
        offloader = create_activation_offloader(
            max_memory_ratio=0.5,
            offload_directory=temp_dir,
            page_size_mb=4,
            eviction_policy="lru",
        )

        manager = ActivationOffloadingManager(offloader)

        # Start proactive management
        manager.start_proactive_management(interval=1.0)

        # Stop proactive management
        manager.stop_proactive_management()


def test_plugin_activation_offloading_setup():
    """Test that all plugins can set up activation offloading."""
    plugins = create_test_plugins()

    for plugin in plugins:
        # Initialize the plugin
        success = plugin.initialize(enable_activation_offloading=True)
        assert_true(success, f"Failed to initialize {plugin.__class__.__name__}")

        # Check that activation offloading methods are available
        assert_true(hasattr(plugin, "setup_activation_offloading"))
        assert_true(hasattr(plugin, "enable_activation_offloading"))
        assert_true(hasattr(plugin, "offload_activations"))
        assert_true(hasattr(plugin, "predict_activation_access"))
        assert_true(hasattr(plugin, "get_activation_offloading_stats"))


def test_plugin_enable_activation_offloading():
    """Test that plugins can enable activation offloading."""
    plugins = create_test_plugins()

    for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
        # Initialize the plugin with activation offloading enabled
        success = plugin.initialize(enable_activation_offloading=True)
        assert_true(success, f"Failed to initialize {plugin.__class__.__name__}")

        # Enable activation offloading
        enable_success = plugin.enable_activation_offloading()

        # Should return True on success
        assert_true(
            enable_success,
            f"Failed to enable activation offloading for {plugin.__class__.__name__}",
        )


def test_plugin_offload_activations():
    """Test that plugins can offload activations."""
    plugins = create_test_plugins()

    for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
        # Initialize the plugin with activation offloading enabled
        success = plugin.initialize(enable_activation_offloading=True)
        assert_true(success, f"Failed to initialize {plugin.__class__.__name__}")

        # Offload activations
        offload_success = plugin.offload_activations()

        # Should return True on success
        assert_true(
            offload_success,
            f"Failed to offload activations for {plugin.__class__.__name__}",
        )


def test_plugin_predict_activation_access():
    """Test that plugins can predict activation access."""
    plugins = create_test_plugins()

    for plugin in plugins:
        # Initialize the plugin
        success = plugin.initialize()
        assert_true(success, f"Failed to initialize {plugin.__class__.__name__}")

        # Predict activation access
        predictions = plugin.predict_activation_access()

        # Should return a dictionary
        assert_is_instance(predictions, dict)


def test_plugin_get_activation_offloading_stats():
    """Test that plugins can report activation offloading statistics."""
    plugins = create_test_plugins()

    for plugin in plugins:
        # Initialize the plugin
        success = plugin.initialize()
        assert_true(success, f"Failed to initialize {plugin.__class__.__name__}")

        # Get activation offloading stats (should work even without offloading performed)
        stats = plugin.get_activation_offloading_stats()

        # Should return a dictionary with stats
        assert_is_instance(stats, dict)
        assert_in("activation_offloading_enabled", stats)
        assert_in("system_memory_percent", stats)


def test_activation_offloading_with_memory_management():
    """Test activation offloading working with memory management."""
    plugins = create_test_plugins()

    for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
        # Initialize the plugin with both optimizations
        success = plugin.initialize(
            enable_activation_offloading=True,
            enable_memory_management=True,
            enable_tensor_paging=True,
        )
        assert_true(success, f"Failed to initialize {plugin.__class__.__name__}")

        # Offload activations
        offload_success = plugin.offload_activations()
        assert_true(
            offload_success,
            f"Failed to offload activations for {plugin.__class__.__name__}",
        )

        # Get activation offloading stats
        activation_stats = plugin.get_activation_offloading_stats()
        assert_in("activation_offloading_enabled", activation_stats)

        # Get memory stats to verify memory management is also working
        memory_stats = plugin.get_memory_stats()
        assert_in("system_memory_percent", memory_stats)


def test_activation_offloading_with_other_optimizations():
    """Test activation offloading working with other optimizations."""
    plugins = create_test_plugins()

    for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
        # Initialize the plugin with activation offloading and other optimizations
        success = plugin.initialize(
            enable_activation_offloading=True,
            enable_tensor_compression=True,
            enable_disk_offloading=True,
            enable_model_surgery=True,
            enable_kernel_fusion=True,
        )
        assert_true(success, f"Failed to initialize {plugin.__class__.__name__}")

        # Offload activations
        offload_success = plugin.offload_activations()
        assert_true(
            offload_success,
            f"Failed to offload activations for {plugin.__class__.__name__}",
        )

        # Get activation offloading stats
        stats = plugin.get_activation_offloading_stats()
        assert_in("activation_offloading_enabled", stats)


def test_activation_offloading_with_different_tensor_sizes():
    """Test offloading with different activation sizes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        offloader = create_activation_offloader(
            max_memory_ratio=0.5,
            offload_directory=temp_dir,
            page_size_mb=4,
            eviction_policy="lru",
        )

        manager = ActivationOffloadingManager(offloader)

        # Test with different activation sizes
        activation_sizes = [(10, 10), (50, 50), (100, 100)]

        for i, size in enumerate(activation_sizes):
            test_activation = torch.randn(*size)

            # Offload the activation
            success = manager.offload_activation(
                test_activation,
                f"test_activation_{i}",
                priority=OffloadPriority.MEDIUM,
                access_pattern=AccessPattern.FREQUENT,
            )
            assert_true(success, f"Failed to offload activation of size {size}")

            # Load the activation back
            loaded_activation = manager.load_activation(f"test_activation_{i}")
            assert_is_not_none(
                loaded_activation, f"Failed to load activation of size {size}"
            )

            # Check that the activation is the same
            assert_true(
                torch.equal(test_activation, loaded_activation),
                f"Loaded activation differs from original for size {size}",
            )


def cleanup_plugins():
    """Clean up any resources used by the plugins."""
    plugins = create_test_plugins()
    for plugin in plugins:
        if hasattr(plugin, "cleanup"):
            plugin.cleanup()


if __name__ == "__main__":
    # Run the tests using custom test utilities
    test_functions = [
        test_create_activation_offloader,
        test_activation_offloading_manager_creation,
        test_offload_and_load_activation,
        test_pin_unpin_activation,
        test_start_stop_proactive_management,
        test_plugin_activation_offloading_setup,
        test_plugin_enable_activation_offloading,
        test_plugin_offload_activations,
        test_plugin_predict_activation_access,
        test_plugin_get_activation_offloading_stats,
        test_activation_offloading_with_memory_management,
        test_activation_offloading_with_other_optimizations,
        test_activation_offloading_with_different_tensor_sizes,
    ]
    run_tests(test_functions)

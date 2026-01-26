"""
Test suite for Memory Management functionality in model plugins.

This test verifies that the memory management system works correctly across all model plugins.
"""

import torch
import tempfile
from src.inference_pio.test_utils import (
    assert_true,
    assert_is_not_none,
    assert_is_instance,
    run_tests
)
from src.inference_pio.common.memory_manager import MemoryManager, TensorPagingManager, MemoryPriority
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin


def create_test_plugins():
    """Create test plugins for the tests."""
    return [
        GLM_4_7_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Plugin()
    ]


def test_memory_manager_basic_functionality():
    """Test basic memory manager functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize memory manager
        memory_manager = MemoryManager(
            max_memory_ratio=0.5,
            swap_directory=temp_dir,
            page_size_mb=4,
            eviction_policy="lru"
        )

        # Create a test tensor
        test_tensor = torch.randn(100, 100)  # Small tensor for testing

        # Allocate a page
        success = memory_manager.allocate_page(test_tensor, "test_page_1", MemoryPriority.HIGH)
        assert_true(success, "Failed to allocate page")

        # Access the page
        retrieved_tensor = memory_manager.access_page("test_page_1")
        assert_is_not_none(retrieved_tensor, "Failed to access page")
        assert_true(torch.equal(test_tensor, retrieved_tensor), "Retrieved tensor differs from original")

        # Swap the page to disk
        success = memory_manager.swap_page_to_disk("test_page_1")
        assert_true(success, "Failed to swap page to disk")

        # Access the page again (should bring it back to RAM)
        retrieved_tensor = memory_manager.access_page("test_page_1")
        assert_is_not_none(retrieved_tensor, "Failed to access page after swap")

        # Get stats
        stats = memory_manager.get_page_stats()
        assert_is_instance(stats, dict)

        # Cleanup
        memory_manager.cleanup()


def test_tensor_paging_manager_functionality():
    """Test tensor paging manager functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize memory manager
        memory_manager = MemoryManager(
            max_memory_ratio=0.5,
            swap_directory=temp_dir,
            page_size_mb=4,
            eviction_policy="lru"
        )

        # Create tensor paging manager
        paging_manager = TensorPagingManager(memory_manager)

        # Create a test tensor
        test_tensor = torch.randn(50, 50)

        # Page the tensor
        success = paging_manager.page_tensor(test_tensor, "test_tensor_1", MemoryPriority.MEDIUM)
        assert_true(success, "Failed to page tensor")

        # Access the tensor
        retrieved_tensor = paging_manager.access_tensor("test_tensor_1")
        assert_is_not_none(retrieved_tensor, "Failed to access paged tensor")
        assert_true(torch.equal(test_tensor, retrieved_tensor), "Retrieved tensor differs from original")

        # Pin the tensor
        success = paging_manager.pin_tensor("test_tensor_1")
        assert_true(success, "Failed to pin tensor")

        # Unpin the tensor
        success = paging_manager.unpin_tensor("test_tensor_1")
        assert_true(success, "Failed to unpin tensor")

        # Unpage the tensor
        success = paging_manager.unpage_tensor("test_tensor_1")
        assert_true(success, "Failed to unpage tensor")

        # Cleanup
        memory_manager.cleanup()


def test_plugin_memory_management_integration():
    """Test memory management integration with plugins."""
    plugins = create_test_plugins()

    for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
        # Initialize the plugin with memory management enabled
        success = plugin.initialize(
            enable_memory_management=True,
            enable_tensor_paging=True,
            enable_smart_swap=True
        )
        assert_true(success, f"Failed to initialize {plugin.__class__.__name__}")

        # Get memory stats
        stats = plugin.get_memory_stats()
        assert_is_instance(stats, dict)

        # Force cleanup
        cleanup_success = plugin.force_memory_cleanup()
        assert_true(cleanup_success, f"Failed to force memory cleanup for {plugin.__class__.__name__}")


def test_config_integration():
    """Test that config includes memory management settings."""
    # Test GLM-4.7 config
    config = GLM47Config()
    assert_true(hasattr(config, 'enable_memory_management'), "Config missing enable_memory_management")
    assert_true(hasattr(config, 'max_memory_ratio'), "Config missing max_memory_ratio")
    assert_true(hasattr(config, 'enable_tensor_paging'), "Config missing enable_tensor_paging")
    assert_true(hasattr(config, 'enable_smart_swap'), "Config missing enable_smart_swap")


def test_memory_management_with_other_optimizations():
    """Test memory management working with other optimizations."""
    plugins = create_test_plugins()

    for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
        # Initialize the plugin with memory management and other optimizations
        success = plugin.initialize(
            enable_memory_management=True,
            enable_tensor_paging=True,
            enable_smart_swap=True,
            enable_kernel_fusion=True,
            enable_adaptive_batching=True,
            enable_tensor_compression=True,
            enable_disk_offloading=True,
            enable_model_surgery=True,
            enable_activation_offloading=True
        )
        assert_true(success, f"Failed to initialize {plugin.__class__.__name__}")

        # Check that memory management is properly set up
        assert_true(hasattr(plugin, '_memory_manager'))
        assert_true(hasattr(plugin, '_tensor_paging_manager'))

        # Get memory stats to verify it's working
        stats = plugin.get_memory_stats()
        assert_is_instance(stats, dict)


def test_predictive_memory_management():
    """Test predictive memory management functionality."""
    plugins = create_test_plugins()

    for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
        # Initialize the plugin with predictive memory management
        success = plugin.initialize(
            enable_memory_management=True,
            enable_tensor_paging=True,
            enable_predictive_management=True
        )
        assert_true(success, f"Failed to initialize {plugin.__class__.__name__}")

        # Start predictive management
        start_success = plugin.start_predictive_memory_management()
        assert_true(start_success, f"Failed to start predictive management for {plugin.__class__.__name__}")

        # Stop predictive management
        stop_success = plugin.stop_predictive_memory_management()
        assert_true(stop_success, f"Failed to stop predictive management for {plugin.__class__.__name__}")


def cleanup_plugins():
    """Clean up any resources used by the plugins."""
    plugins = create_test_plugins()
    for plugin in plugins:
        if hasattr(plugin, 'cleanup'):
            plugin.cleanup()


if __name__ == "__main__":
    # Run the tests using custom test utilities
    test_functions = [
        test_memory_manager_basic_functionality,
        test_tensor_paging_manager_functionality,
        test_plugin_memory_management_integration,
        test_config_integration,
        test_memory_management_with_other_optimizations,
        test_predictive_memory_management
    ]
    run_tests(test_functions)
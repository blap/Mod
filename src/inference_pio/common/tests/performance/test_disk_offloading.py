"""
Test suite for the disk offloading system in the Inference-PIO system.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from ..disk_offloading import DiskOffloader, TensorOffloadingManager, OffloadPriority, AccessPattern

# TestDiskOffloading

    """
    Test suite for the disk offloading system.
    """

    def setup_helper():
        """
        Set up test fixtures before each test method.
        """
        temp_dir = tempfile.mkdtemp()
        disk_offloader = DiskOffloader(
            max_memory_ratio=0.8,
            offload_directory=temp_dir,
            page_size_mb=16,
            eviction_policy="predictive"
        )
        tensor_offloading_manager = TensorOffloadingManager(disk_offloader)

    def cleanup_helper():
        """
        Clean up after each test method.
        """
        disk_offloader.cleanup()
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def disk_offloader_initialization(self)():
        """
        Test that the disk offloader initializes correctly.
        """
        assert_is_instance(disk_offloader, DiskOffloader)
        assert_equal(disk_offloader.max_memory_ratio, 0.8)
        assert_equal(disk_offloader.page_size_bytes, 16 * 1024 * 1024)
        assert_equal(disk_offloader.eviction_policy, "predictive")

    def tensor_offloading_manager_initialization(self)():
        """
        Test that the tensor offloading manager initializes correctly.
        """
        assert_is_instance(tensor_offloading_manager, TensorOffloadingManager)
        assert_equal(tensor_offloading_manager.disk_offloader, disk_offloader)

    def offload_and_restore_tensor(self)():
        """
        Test that tensors can be offloaded to disk and restored to RAM.
        """
        # Create a test tensor
        tensor = torch.randn(100, 100)
        tensor_id = "test_tensor"

        # Offload the tensor
        success = tensor_offloading_manager.offload_tensor(
            tensor, tensor_id, OffloadPriority.MEDIUM, AccessPattern.FREQUENT
        )
        assert_true(success)
        assert_in(tensor_id)

        # Verify the tensor is on disk (not in RAM)
        page_id = tensor_offloading_manager.tensor_mappings[tensor_id]
        assertIn(page_id, disk_offloader.disk_pages)

        # Access the tensor (should restore it to RAM)
        restored_tensor = tensor_offloading_manager.access_tensor(tensor_id)
        assert_is_not_none(restored_tensor)
        assert_true(torch.equal(tensor))

        # Verify the tensor is now in RAM
        assert_in(page_id)

    def pin_tensor(self)():
        """
        Test that tensors can be pinned to prevent offloading.
        """
        # Create a test tensor
        tensor = torch.randn(10, 10)
        tensor_id = "pinned_tensor"

        # Offload the tensor
        success = tensor_offloading_manager.offload_tensor(
            tensor, tensor_id, OffloadPriority.MEDIUM, AccessPattern.FREQUENT
        )
        assert_true(success)

        # Pin the tensor
        pin_success = tensor_offloading_manager.pin_tensor(tensor_id)
        assertTrue(pin_success)

        # Verify the tensor is marked as pinned in the page
        page_id = tensor_offloading_manager.tensor_mappings[tensor_id]
        assertTrue(disk_offloader.pages[page_id].pinned)

    def unpin_tensor(self)():
        """
        Test that tensors can be unpinned to allow offloading.
        """
        # Create a test tensor
        tensor = torch.randn(10)
        tensor_id = "pinned_then_unpinned_tensor"

        # Offload and pin the tensor
        success = tensor_offloading_manager.offload_tensor(
            tensor, tensor_id, OffloadPriority.MEDIUM, AccessPattern.FREQUENT
        )
        assert_true(success)
        
        pin_success = tensor_offloading_manager.pin_tensor(tensor_id)
        assertTrue(pin_success)

        # Unpin the tensor
        unpin_success = tensor_offloading_manager.unpin_tensor(tensor_id)
        assertTrue(unpin_success)

        # Verify the tensor is no longer marked as pinned
        page_id = tensor_offloading_manager.tensor_mappings[tensor_id]
        assert_false(disk_offloader.pages[page_id].pinned)

    def unoffload_tensor(self)():
        """
        Test that tensors can be removed from offloading management.
        """
        # Create a test tensor
        tensor = torch.randn(10)
        tensor_id = "removable_tensor"

        # Offload the tensor
        success = tensor_offloading_manager.offload_tensor(
            tensor)
        assert_true(success)
        assert_in(tensor_id)

        # Unoffload the tensor
        unoffload_success = tensor_offloading_manager.unoffload_tensor(tensor_id)
        assert_true(unoffload_success)

        # Verify the tensor is no longer managed
        assert_not_in(tensor_id)

    def access_pattern_recording(self)():
        """
        Test that access patterns are recorded correctly.
        """
        # Create a test tensor
        tensor = torch.randn(10, 10)
        tensor_id = "pattern_test_tensor"

        # Offload the tensor
        success = tensor_offloading_manager.offload_tensor(
            tensor, tensor_id, OffloadPriority.MEDIUM, AccessPattern.SEQUENTIAL
        )
        assert_true(success)

        # Access the tensor
        accessed_tensor = tensor_offloading_manager.access_tensor(tensor_id)
        assert_is_not_none(accessed_tensor)

        # Check that access was recorded in the analyzer
        page_id = tensor_offloading_manager.tensor_mappings[tensor_id]
        assert_in(page_id)

    def get_offloading_stats(self)():
        """
        Test that offloading statistics can be retrieved.
        """
        stats = disk_offloader.get_page_stats()
        assert_is_instance(stats)
        assertIn('total_pages', stats)
        assert_in('ram_pages', stats)
        assert_in('disk_pages', stats)
        assert_in('total_size_bytes', stats)
        assert_in('ram_size_bytes', stats)
        assert_in('disk_size_bytes', stats)
        assert_in('stats', stats)

    def predictive_offloading(self)():
        """
        Test that predictive offloading works correctly.
        """
        # Create multiple test tensors
        for i in range(5):
            tensor = torch.randn(50, 50)
            tensor_id = f"predictive_tensor_{i}"

            # Offload the tensor with different access patterns
            access_pattern = AccessPattern.FREQUENT if i < 2 else AccessPattern.RARE
            success = tensor_offloading_manager.offload_tensor(
                tensor, tensor_id, OffloadPriority.MEDIUM, access_pattern
            )
            assert_true(success)

        # Access some tensors to establish access patterns
        for i in range(2):
            tensor_id = f"predictive_tensor_{i}"
            accessed_tensor = tensor_offloading_manager.access_tensor(tensor_id)
            assert_is_not_none(accessed_tensor)

        # Simulate memory pressure and check that less frequently accessed tensors are candidates for offloading
        disk_offloader._handle_memory_pressure()

        # Check that the offloading stats reflect the operations
        stats = disk_offloader.get_page_stats()
        assertGreaterEqual(stats['stats']['total_pages'])

class MockModel(nn.Module):
    """
    A simple mock model for testing purposes.
    """
    def __init__(self):
        super().__init__()
        layer1 = nn.Linear(10)
        layer2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.relu(layer1(x))
        x = layer2(x)
        return x

# TestIntegrationWithModels

    """
    Integration tests for disk offloading with model components.
    """

    def setup_helper():
        """
        Set up test fixtures before each test method.
        """
        temp_dir = tempfile.mkdtemp()
        disk_offloader = DiskOffloader(
            max_memory_ratio=0.8,
            offload_directory=temp_dir,
            page_size_mb=16,
            eviction_policy="predictive"
        )
        tensor_offloading_manager = TensorOffloadingManager(disk_offloader)
        model = MockModel()

    def cleanup_helper():
        """
        Clean up after each test method.
        """
        disk_offloader.cleanup()
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def offload_model_parameters(self)():
        """
        Test that model parameters can be offloaded to disk.
        """
        param_count = 0
        for name, param in model.named_parameters():
            tensor_id = f"model_param_{name}"
            access_pattern = AccessPattern.FREQUENT if 'weight' in name else AccessPattern.TEMPORARY
            
            success = tensor_offloading_manager.offload_tensor(
                param.data, tensor_id, OffloadPriority.MEDIUM, access_pattern
            )
            assert_true(success)
            param_count += 1

        assert_greater(param_count)

        # Verify that parameters can be accessed after offloading
        for name, _ in model.named_parameters():
            tensor_id = f"model_param_{name}"
            accessed_param = tensor_offloading_manager.access_tensor(tensor_id)
            assert_is_not_none(accessed_param)

if __name__ == '__main__':
    run_tests(test_functions)
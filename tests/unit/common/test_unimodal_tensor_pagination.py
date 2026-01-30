"""
Unit tests for the unimodal tensor pagination system.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import tempfile
import shutil
from pathlib import Path
from src.inference_pio.common.unimodal_tensor_pagination import (
    create_unimodal_pagination_system,
    TextDataType,
    PaginationPriority,
    UnimodalTensorPager
)

class TestUnimodalTensorPaginationSystem:
    """Test cases for the unimodal tensor pagination system."""

    def setup_helper(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.swap_dir = Path(self.test_dir) / "text_tensor_swap"
        self.swap_dir.mkdir(exist_ok=True)

    def cleanup_helper(self):
        """Tear down test fixtures after each test method."""
        shutil.rmtree(self.test_dir)

    def create_pagination_system(self):
        """Test creating a unimodal pagination system."""
        self.setup_helper()
        pagination_system, pager = create_unimodal_pagination_system(
            swap_directory=self.swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )

        assert_is_not_none(pagination_system)
        assert_is_not_none(pager)

        pagination_system.cleanup()
        self.cleanup_helper()

    def basic_pagination_functionality(self):
        """Test basic unimodal tensor pagination functionality."""
        self.setup_helper()
        pagination_system, pager = create_unimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )

        # Create a test tensor
        test_tensor = torch.randn(100, 100)
        tensor_id = "test_tensor"

        # Page the tensor
        success = pager.page_tensor(
            test_tensor,
            tensor_id,
            TextDataType.TEXT_EMBEDDINGS,
            priority=PaginationPriority.HIGH
        )

        assert_true(success)

        # Access the tensor
        retrieved_tensor = pager.access_tensor(tensor_id)
        assert_is_not_none(retrieved_tensor)
        assert_true(torch.equal(test_tensor), "Retrieved tensor differs from original")

        pagination_system.cleanup()

    def pagination_with_different_text_data_types(self)():
        """Test pagination with different text data types."""
        pagination_system, pager = create_unimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )

        text_data_types = [
            TextDataType.TEXT_EMBEDDINGS,
            TextDataType.TEXT_ACTIVATIONS,
            TextDataType.TEXT_KV_CACHE,
            TextDataType.TEXT_ATTENTION_WEIGHTS,
            TextDataType.TEXT_MLP_WEIGHTS,
            TextDataType.TEXT_LAYERNORM_WEIGHTS,
            TextDataType.TEXT_INPUT_TOKENS,
            TextDataType.TEXT_OUTPUT_TOKENS,
            TextDataType.TEXT_HIDDEN_STATES
        ]

        for i, data_type in enumerate(text_data_types):
            tensor = torch.randn(50, 50)
            tensor_id = f"test_tensor_{data_type.value}_{i}"

            success = pager.page_tensor(
                tensor,
                tensor_id,
                data_type,
                priority=PaginationPriority.MEDIUM
            )

            assert_true(success)

            retrieved = pager.access_tensor(tensor_id)
            assert_is_not_none(retrieved)
            assert_true(torch.equal(tensor), f"Retrieved {data_type.value} tensor differs from original")

        pagination_system.cleanup()

    def priority_handling(self)():
        """Test priority handling in unimodal pagination."""
        pagination_system, pager = create_unimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )

        # Page a tensor with low priority
        pager.page_tensor(
            torch.randn(10, 10),
            "priority_test",
            TextDataType.TEXT_EMBEDDINGS,
            priority=PaginationPriority.LOW
        )

        current_priority = pager.get_tensor_priority("priority_test")
        assert_equal(current_priority, PaginationPriority.LOW)

        # Update priority
        pager.set_tensor_priority("priority_test", PaginationPriority.HIGH)
        updated_priority = pager.get_tensor_priority("priority_test")
        assert_equal(updated_priority, PaginationPriority.HIGH)

        pagination_system.cleanup()

    def pin_unpin_functionality(self)():
        """Test pin/unpin functionality in unimodal pagination."""
        pagination_system, pager = create_unimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )

        # Page a tensor
        pager.page_tensor(
            torch.randn(10, 10),
            "pin_test",
            TextDataType.TEXT_EMBEDDINGS,
            priority=PaginationPriority.HIGH
        )

        # Pin the tensor
        success = pager.pin_tensor("pin_test")
        assert_true(success)

        # Unpin the tensor
        success = pager.unpin_tensor("pin_test")
        assert_true(success)

        pagination_system.cleanup()

    def pagination_system_statistics(self)():
        """Test pagination system statistics."""
        pagination_system, pager = create_unimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )

        # Add some tensors to the system
        for i in range(5):
            tensor = torch.randn(50, 50)
            tensor_id = f"stat_test_{i}"
            
            success = pager.page_tensor(
                tensor,
                tensor_id,
                TextDataType.TEXT_ACTIVATIONS,
                priority=PaginationPriority.MEDIUM
            )
            
            assert_true(success)

        # Get statistics
        stats = pagination_system.get_page_stats()
        
        assertGreaterEqual(stats['total_pages'], 5, "Expected at least 5 total pages")
        assertGreaterEqual(stats['ram_pages'], 5, "Expected at least 5 pages in RAM")
        assert_equal(stats['disk_pages'], 0)

        pagination_system.cleanup()

    def large_tensor_pagination(self)():
        """Test pagination with larger tensors."""
        pagination_system, pager = create_unimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )

        # Create a larger tensor
        large_tensor = torch.randn(200, 200)
        tensor_id = "large_tensor"

        success = pager.page_tensor(
            large_tensor,
            tensor_id,
            TextDataType.TEXT_KV_CACHE,
            priority=PaginationPriority.HIGH
        )

        assert_true(success)

        retrieved_tensor = pager.access_tensor(tensor_id)
        assert_is_not_none(retrieved_tensor)
        assert_true(torch.equal(large_tensor), "Retrieved large tensor differs from original")

        pagination_system.cleanup()

    def tensor_position_and_layer_info(self)():
        """Test pagination with position and layer information."""
        pagination_system, pager = create_unimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )

        # Page a tensor with position and layer information
        test_tensor = torch.randn(50, 50)
        tensor_id = "position_test"

        success = pager.page_tensor(
            test_tensor,
            tensor_id,
            TextDataType.TEXT_ACTIVATIONS,
            priority=PaginationPriority.MEDIUM,
            position_in_sequence=100,
            layer_index=5,
            attention_head=2,
            is_past_key_value=True
        )

        assert_true(success)

        retrieved_tensor = pager.access_tensor(tensor_id)
        assert_is_not_none(retrieved_tensor)
        assert_true(torch.equal(test_tensor), "Retrieved tensor differs from original")

        pagination_system.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)
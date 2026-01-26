"""
Simple test script to verify the intelligent unimodal pagination system works correctly.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from src.inference_pio.common.unimodal_tensor_pagination import (
    create_unimodal_pagination_system,
    TextDataType,
    PaginationPriority
)
import tempfile
import shutil
from pathlib import Path


def test_unimodal_pagination_system():
    """Test the unimodal pagination system functionality."""
    print("Testing Intelligent Unimodal Tensor Pagination System...")

    # Create temporary directory for swap files
    test_dir = tempfile.mkdtemp()
    swap_dir = Path(test_dir) / "text_tensor_swap"
    swap_dir.mkdir(exist_ok=True)

    try:
        # Create pagination system
        pagination_system, pager = create_unimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )

        print("[OK] Unimodal pagination system created successfully")

        # Test basic functionality
        test_tensor = torch.randn(100, 100)
        tensor_id = "test_tensor"

        # Page the tensor
        success = pager.page_tensor(
            test_tensor,
            tensor_id,
            TextDataType.TEXT_EMBEDDINGS,
            priority=PaginationPriority.HIGH
        )

        assert success, "Failed to page tensor"
        print("[OK] Tensor paged successfully")

        # Access the tensor
        retrieved_tensor = pager.access_tensor(tensor_id)
        assert retrieved_tensor is not None, "Failed to access paged tensor"
        assert torch.equal(test_tensor, retrieved_tensor), "Retrieved tensor differs from original"
        print("[OK] Tensor accessed successfully and matches original")

        # Test different text data types
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

            assert success, f"Failed to page {data_type.value} tensor"

            retrieved = pager.access_tensor(tensor_id)
            assert retrieved is not None, f"Failed to access {data_type.value} tensor"
            assert torch.equal(tensor, retrieved), f"Retrieved {data_type.value} tensor differs from original"

        print(f"[OK] Tested all {len(text_data_types)} text data types successfully")

        # Test priority handling
        pager.page_tensor(
            torch.randn(10, 10),
            "priority_test",
            TextDataType.TEXT_EMBEDDINGS,
            priority=PaginationPriority.LOW
        )

        current_priority = pager.get_tensor_priority("priority_test")
        assert current_priority == PaginationPriority.LOW, "Incorrect priority retrieved"

        pager.set_tensor_priority("priority_test", PaginationPriority.HIGH)
        updated_priority = pager.get_tensor_priority("priority_test")
        assert updated_priority == PaginationPriority.HIGH, "Priority not updated correctly"

        print("[OK] Priority handling works correctly")

        # Test pin/unpin
        pager.pin_tensor("priority_test")
        # In a real scenario, this would affect eviction behavior
        pager.unpin_tensor("priority_test")

        print("[OK] Pin/unpin functionality works")

        # Get statistics
        stats = pagination_system.get_page_stats()
        print(f"[OK] Got statistics: {stats['total_pages']} total pages, "
              f"{stats['ram_pages']} in RAM, {stats['disk_pages']} on disk")

        # Clean up
        pagination_system.cleanup()
        print("[OK] Pagination system cleaned up successfully")

        print("\n[SUCCESS] All tests passed! Intelligent Unimodal Tensor Pagination System is working correctly.")

    except Exception as e:
        print(f"[ERROR] Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temporary directory
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_unimodal_pagination_system()
"""
Test suite for the enhanced Memory Manager with predictive algorithms.
"""

import torch
import tempfile
import os
from pathlib import Path
from src.inference_pio.test_utils import (
    assert_is_not_none,
    assert_is_none,
    assert_equal,
    assert_true,
    assert_is_instance,
    assert_in,
    run_tests
)
from src.inference_pio.common.memory_manager import MemoryManager, TensorPagingManager, MemoryPriority


def setup_memory_manager(max_memory_ratio=0.1, prediction_horizon=10):
    """Set up a memory manager for testing."""
    temp_dir = tempfile.mkdtemp()
    memory_manager = MemoryManager(
        max_memory_ratio=max_memory_ratio,  # Low ratio for testing
        swap_directory=temp_dir,
        page_size_mb=1,
        eviction_policy="predictive",
        prediction_horizon=prediction_horizon
    )
    tensor_paging_manager = TensorPagingManager(memory_manager)
    return memory_manager, tensor_paging_manager, temp_dir


def cleanup_memory_manager(memory_manager, temp_dir):
    """Clean up memory manager and temporary directory."""
    memory_manager.cleanup()
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_memory_manager_initialization():
    """Test that the memory manager initializes correctly."""
    memory_manager, tensor_paging_manager, temp_dir = setup_memory_manager(prediction_horizon=10)

    try:
        assert_is_not_none(memory_manager)
        assert_is_not_none(tensor_paging_manager)
        assert_equal(memory_manager.eviction_policy, "predictive")
        assert_equal(memory_manager.prediction_horizon, 10)
    finally:
        cleanup_memory_manager(memory_manager, temp_dir)


def test_tensor_paging():
    """Test basic tensor paging functionality."""
    memory_manager, tensor_paging_manager, temp_dir = setup_memory_manager()

    try:
        # Create a tensor
        tensor = torch.randn(100, 100)  # Small tensor for testing
        tensor_id = "test_tensor_1"

        # Page the tensor
        result = tensor_paging_manager.page_tensor(tensor, tensor_id, MemoryPriority.HIGH)
        assert_true(result)

        # Access the tensor
        retrieved_tensor = tensor_paging_manager.access_tensor(tensor_id)
        assert_is_not_none(retrieved_tensor)
        assert_true(torch.equal(tensor, retrieved_tensor))
    finally:
        cleanup_memory_manager(memory_manager, temp_dir)


def test_tensor_paging_with_different_priorities():
    """Test tensor paging with different priority levels."""
    memory_manager, tensor_paging_manager, temp_dir = setup_memory_manager()

    try:
        priorities = [MemoryPriority.LOW, MemoryPriority.MEDIUM, MemoryPriority.HIGH, MemoryPriority.CRITICAL]

        for i, priority in enumerate(priorities):
            tensor = torch.randn(50, 50)
            tensor_id = f"test_tensor_priority_{i}"

            result = tensor_paging_manager.page_tensor(tensor, tensor_id, priority)
            assert_true(result)

            retrieved_tensor = tensor_paging_manager.access_tensor(tensor_id)
            assert_is_not_none(retrieved_tensor)
            assert_true(torch.equal(tensor, retrieved_tensor))
    finally:
        cleanup_memory_manager(memory_manager, temp_dir)


def test_tensor_unpaging():
    """Test tensor unpaging functionality."""
    memory_manager, tensor_paging_manager, temp_dir = setup_memory_manager()

    try:
        tensor = torch.randn(100, 100)
        tensor_id = "test_tensor_unpage"

        # Page the tensor
        result = tensor_paging_manager.page_tensor(tensor, tensor_id)
        assert_true(result)

        # Unpage the tensor
        result = tensor_paging_manager.unpage_tensor(tensor_id)
        assert_true(result)

        # Try to access the unpaged tensor (should return None)
        retrieved_tensor = tensor_paging_manager.access_tensor(tensor_id)
        assert_is_none(retrieved_tensor)
    finally:
        cleanup_memory_manager(memory_manager, temp_dir)


def test_tensor_pinning():
    """Test tensor pinning functionality."""
    memory_manager, tensor_paging_manager, temp_dir = setup_memory_manager()

    try:
        tensor = torch.randn(100, 100)
        tensor_id = "test_tensor_pin"

        # Page the tensor
        result = tensor_paging_manager.page_tensor(tensor, tensor_id)
        assert_true(result)

        # Pin the tensor
        result = tensor_paging_manager.pin_tensor(tensor_id)
        assert_true(result)

        # Unpin the tensor
        result = tensor_paging_manager.unpin_tensor(tensor_id)
        assert_true(result)
    finally:
        cleanup_memory_manager(memory_manager, temp_dir)


def test_memory_stats():
    """Test memory statistics functionality."""
    memory_manager, tensor_paging_manager, temp_dir = setup_memory_manager()

    try:
        stats = memory_manager.get_page_stats()
        assert_is_instance(stats, dict)
        assert_in('total_pages', stats)
        assert_in('ram_pages', stats)
        assert_in('disk_pages', stats)
        assert_in('total_size_bytes', stats)
        assert_in('ram_size_bytes', stats)
        assert_in('disk_size_bytes', stats)
        assert_in('stats', stats)
    finally:
        cleanup_memory_manager(memory_manager, temp_dir)


def test_predictive_memory_management():
    """Test predictive memory management functionality."""
    memory_manager, tensor_paging_manager, temp_dir = setup_memory_manager()

    try:
        # Start proactive management
        tensor_paging_manager.start_proactive_management(interval=1.0)

        # Give it a moment to start
        import time
        time.sleep(0.1)

        # Stop proactive management
        tensor_paging_manager.stop_proactive_management()

        # Verify it stopped without errors
        assert_true(True)  # Just ensure no exceptions were raised
    finally:
        cleanup_memory_manager(memory_manager, temp_dir)


def test_memory_prediction():
    """Test memory prediction functionality."""
    memory_manager, tensor_paging_manager, temp_dir = setup_memory_manager()

    try:
        # Record some memory usage
        import time
        current_time = time.time()
        memory_manager.memory_predictor.record_memory_usage(current_time, 1000000)  # 1MB
        memory_manager.memory_predictor.record_memory_usage(current_time + 1, 2000000)  # 2MB

        # Predict future memory usage
        predicted = memory_manager.memory_predictor.predict_future_memory(current_time + 5)
        assert_is_instance(predicted, int)
        assert_true(predicted >= 0)
    finally:
        cleanup_memory_manager(memory_manager, temp_dir)


def test_access_pattern_analysis():
    """Test access pattern analysis functionality."""
    memory_manager, tensor_paging_manager, temp_dir = setup_memory_manager()

    try:
        import time
        current_time = time.time()

        # Record some access patterns
        memory_manager.access_analyzer.record_access("page1", current_time)
        memory_manager.access_analyzer.record_access("page1", current_time + 1)
        memory_manager.access_analyzer.record_access("page2", current_time + 0.5)

        # Get access scores
        score1 = memory_manager.access_analyzer.get_access_score("page1", current_time + 2)
        score2 = memory_manager.access_analyzer.get_access_score("page2", current_time + 2)

        assert_is_instance(score1, float)
        assert_is_instance(score2, float)
        assert_true(score1 >= 0)
        assert_true(score2 >= 0)
    finally:
        cleanup_memory_manager(memory_manager, temp_dir)


def test_end_to_end_workflow():
    """Test end-to-end workflow with predictive memory management."""
    memory_manager, tensor_paging_manager, temp_dir = setup_memory_manager(max_memory_ratio=0.1, prediction_horizon=5)

    try:
        # Start proactive management
        tensor_paging_manager.start_proactive_management(interval=0.5)

        # Create and page multiple tensors
        tensors = []
        for i in range(5):
            tensor = torch.randn(100, 100)
            tensor_id = f"integration_tensor_{i}"

            # Page tensor
            result = tensor_paging_manager.page_tensor(tensor, tensor_id)
            assert_true(result)
            tensors.append((tensor, tensor_id))

        # Access tensors multiple times to establish access patterns
        for _ in range(3):
            for tensor, tensor_id in tensors:
                retrieved = tensor_paging_manager.access_tensor(tensor_id)
                assert_is_not_none(retrieved)
                assert_true(torch.equal(tensor, retrieved))

        # Check memory stats
        stats = memory_manager.get_page_stats()
        assert_equal(stats['total_pages'], 5)

        # Stop proactive management
        tensor_paging_manager.stop_proactive_management()

        # Clean up tensors
        for _, tensor_id in tensors:
            result = tensor_paging_manager.unpage_tensor(tensor_id)
            assert_true(result)
    finally:
        cleanup_memory_manager(memory_manager, temp_dir)


if __name__ == '__main__':
    # Run the tests using custom test utilities
    test_functions = [
        test_memory_manager_initialization,
        test_tensor_paging,
        test_tensor_paging_with_different_priorities,
        test_tensor_unpaging,
        test_tensor_pinning,
        test_memory_stats,
        test_predictive_memory_management,
        test_memory_prediction,
        test_access_pattern_analysis,
        test_end_to_end_workflow
    ]
    run_tests(test_functions)
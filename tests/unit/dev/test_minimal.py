"""
Minimal test script to verify the Smart Swap and Memory Paging system.
"""

import torch
import tempfile
import os
from pathlib import Path
from typing import Any, Optional
from src.inference_pio.common.memory_manager import MemoryManager, TensorPagingManager, MemoryPriority


def test_basic_functionality() -> None:
    """Test basic functionality of the memory manager."""
    print("Testing basic memory manager functionality...")

    # Create a temporary directory for swap files
    temp_dir: str = tempfile.mkdtemp()

    # Initialize memory manager with predictive algorithms
    memory_manager = MemoryManager(
        max_memory_ratio=0.1,  # Low ratio for testing
        swap_directory=temp_dir,
        page_size_mb=1,
        eviction_policy="predictive",  # Use predictive eviction
        prediction_horizon=10
    )

    # Create tensor paging manager
    tensor_paging_manager = TensorPagingManager(memory_manager)

    # Create a test tensor
    tensor = torch.randn(10, 10)  # Very small tensor for quick testing
    tensor_id: str = "test_tensor_1"

    print(f"Created tensor with shape: {tensor.shape}")

    # Page the tensor
    result: bool = tensor_paging_manager.page_tensor(tensor, tensor_id, MemoryPriority.HIGH)
    print(f"Paging tensor result: {result}")

    # Access the tensor
    retrieved_tensor: Optional[torch.Tensor] = tensor_paging_manager.access_tensor(tensor_id)
    print(f"Retrieved tensor: {retrieved_tensor is not None}")

    if retrieved_tensor is not None:
        print(f"Tensors are equal: {torch.equal(tensor, retrieved_tensor)}")

    # Get memory stats
    stats: Any = memory_manager.get_page_stats()
    print(f"Memory stats: {stats}")

    # Test basic prediction functionality
    import time
    current_time: float = time.time()
    memory_manager.memory_predictor.record_memory_usage(current_time, 1000000)  # 1MB
    memory_manager.memory_predictor.record_memory_usage(current_time + 1, 2000000)  # 2MB

    predicted: float = memory_manager.memory_predictor.predict_future_memory(current_time + 5)
    print(f"Predicted memory usage: {predicted}")

    # Test access pattern analysis
    memory_manager.access_analyzer.record_access("page1", current_time)
    memory_manager.access_analyzer.record_access("page1", current_time + 1)

    score: float = memory_manager.access_analyzer.get_access_score("page1", current_time + 2)
    print(f"Access score for page1: {score}")

    # Cleanup
    unpage_result: bool = tensor_paging_manager.unpage_tensor(tensor_id)
    print(f"Unpage tensor result: {unpage_result}")

    memory_manager.cleanup()
    print("Cleanup completed.")

    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("Basic functionality test completed successfully!")


if __name__ == "__main__":
    print("Starting Smart Swap and Memory Paging System tests...\n")

    test_basic_functionality()

    print("\nTest completed successfully!")
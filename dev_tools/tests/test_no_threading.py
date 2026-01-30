"""
Test script to verify the Smart Swap and Memory Paging system without proactive management.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import tempfile
import os
from pathlib import Path
from src.inference_pio.common.memory_manager import MemoryManager, TensorPagingManager, MemoryPriority


def test_basic_functionality():
    """Test basic functionality of the memory manager."""
    print("Testing basic memory manager functionality...")
    
    # Create a temporary directory for swap files
    temp_dir = tempfile.mkdtemp()
    
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
    tensor_id = "test_tensor_1"
    
    print(f"Created tensor with shape: {tensor.shape}")
    
    # Page the tensor
    result = tensor_paging_manager.page_tensor(tensor, tensor_id, MemoryPriority.HIGH)
    print(f"Paging tensor result: {result}")
    
    # Access the tensor
    retrieved_tensor = tensor_paging_manager.access_tensor(tensor_id)
    print(f"Retrieved tensor: {retrieved_tensor is not None}")
    
    if retrieved_tensor is not None:
        print(f"Tensors are equal: {torch.equal(tensor, retrieved_tensor)}")
    
    # Get memory stats
    stats = memory_manager.get_page_stats()
    print(f"Memory stats: {stats}")
    
    # Test basic prediction functionality
    import time
    current_time = time.time()
    memory_manager.memory_predictor.record_memory_usage(current_time, 1000000)  # 1MB
    memory_manager.memory_predictor.record_memory_usage(current_time + 1, 2000000)  # 2MB
    
    predicted = memory_manager.memory_predictor.predict_future_memory(current_time + 5)
    print(f"Predicted memory usage: {predicted}")
    
    # Test access pattern analysis
    memory_manager.access_analyzer.record_access("page1", current_time)
    memory_manager.access_analyzer.record_access("page1", current_time + 1)
    
    score = memory_manager.access_analyzer.get_access_score("page1", current_time + 2)
    print(f"Access score for page1: {score}")
    
    # Test manual memory pressure handling
    print("Testing memory pressure handling...")
    memory_manager._handle_memory_pressure()  # This should not cause issues
    
    # Test different eviction policies
    print("Testing different eviction policies...")
    memory_manager.eviction_policy = "lru"
    memory_manager._handle_memory_pressure()
    
    memory_manager.eviction_policy = "priority"
    memory_manager._handle_memory_pressure()
    
    memory_manager.eviction_policy = "predictive"
    memory_manager._handle_memory_pressure()
    
    # Cleanup
    unpage_result = tensor_paging_manager.unpage_tensor(tensor_id)
    print(f"Unpage tensor result: {unpage_result}")
    
    memory_manager.cleanup()
    print("Cleanup completed.")
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("Basic functionality test completed successfully!")


def test_multiple_tensors():
    """Test with multiple tensors."""
    print("\nTesting with multiple tensors...")
    
    temp_dir = tempfile.mkdtemp()
    
    memory_manager = MemoryManager(
        max_memory_ratio=0.1,
        swap_directory=temp_dir,
        page_size_mb=1,
        eviction_policy="predictive",
        prediction_horizon=5
    )
    
    tensor_paging_manager = TensorPagingManager(memory_manager)
    
    # Create multiple tensors
    tensors_data = []
    for i in range(3):
        tensor = torch.randn(5, 5)
        tensor_id = f"multi_tensor_{i}"
        
        result = tensor_paging_manager.page_tensor(tensor, tensor_id, MemoryPriority.MEDIUM)
        print(f"Page tensor {tensor_id}: {result}")
        
        tensors_data.append((tensor, tensor_id))
    
    # Access tensors
    for tensor, tensor_id in tensors_data:
        retrieved = tensor_paging_manager.access_tensor(tensor_id)
        if retrieved is not None:
            tensors_equal = torch.equal(tensor, retrieved)
            print(f"  {tensor_id}: accessed={retrieved is not None}, equal={tensors_equal}")
    
    # Get stats
    stats = memory_manager.get_page_stats()
    print(f"Final stats: {stats}")
    
    # Cleanup
    for _, tensor_id in tensors_data:
        tensor_paging_manager.unpage_tensor(tensor_id)
    
    memory_manager.cleanup()
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("Multiple tensors test completed successfully!")


if __name__ == "__main__":
    print("Starting Smart Swap and Memory Paging System tests...\n")
    
    test_basic_functionality()
    test_multiple_tensors()
    
    print("\nAll tests completed successfully!")
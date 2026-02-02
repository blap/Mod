import tempfile

import torch

from src.inference_pio.common.memory_manager import (
    MemoryManager,
    MemoryPriority,
    TensorPagingManager,
)

print("Creating memory manager with high memory ratio...")
temp_dir = tempfile.mkdtemp()
memory_manager = MemoryManager(
    max_memory_ratio=0.9,  # High ratio to avoid memory pressure
    swap_directory=temp_dir,
    page_size_mb=1,
    eviction_policy="predictive",
    prediction_horizon=10,
)
print("Memory manager created successfully")

print("Creating tensor paging manager...")
tensor_paging_manager = TensorPagingManager(memory_manager)
print("Tensor paging manager created successfully")

print("Creating test tensor...")
tensor = torch.randn(5, 5)  # Very small tensor
tensor_id = "test_tensor_1"
print(f"Tensor created with shape: {tensor.shape}")

print("Paging tensor...")
result = tensor_paging_manager.page_tensor(tensor, tensor_id, MemoryPriority.HIGH)
print(f"Paging result: {result}")

print("Accessing tensor...")
retrieved_tensor = tensor_paging_manager.access_tensor(tensor_id)
print(f"Retrieved tensor: {retrieved_tensor is not None}")

if retrieved_tensor is not None:
    print(f"Tensors equal: {torch.equal(tensor, retrieved_tensor)}")

print("Getting memory stats...")
stats = memory_manager.get_page_stats()
print(f"Memory stats: {stats}")

print("Unpaging tensor...")
unpage_result = tensor_paging_manager.unpage_tensor(tensor_id)
print(f"Unpage result: {unpage_result}")

print("Cleaning up...")
memory_manager.cleanup()
print("Cleanup completed.")

print("Test completed successfully!")

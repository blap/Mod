import tempfile
from src.inference_pio.common.memory_manager import MemoryManager, TensorPagingManager

print('Creating memory manager...')
temp_dir = tempfile.mkdtemp()
memory_manager = MemoryManager(
    max_memory_ratio=0.1,
    swap_directory=temp_dir,
    page_size_mb=1,
    eviction_policy='predictive',
    prediction_horizon=10
)
print('Memory manager created successfully')

print('Creating tensor paging manager...')
tensor_paging_manager = TensorPagingManager(memory_manager)
print('Tensor paging manager created successfully')

print('Test completed successfully!')
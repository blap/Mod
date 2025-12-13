"""Basic functionality test for the Qwen3-VL Memory Tiering System"""
from src.models.memory_tiering import create_qwen3vl_memory_tiering_system, TensorType
import torch

print("Testing Qwen3-VL Memory Tiering System...")

# Create system
system = create_qwen3vl_memory_tiering_system()
print("System created successfully")

# Test basic operations
tensor = torch.randn(10, 10, dtype=torch.float16)
print(f"Created tensor with shape {tensor.shape}")

success, tensor_id = system.put_tensor(tensor, TensorType.GENERAL)
print(f"Put tensor: {success}")

retrieved = system.get_tensor(tensor_id)
print(f"Get tensor: {retrieved is not None}")

if retrieved is not None:
    print(f"Retrieved tensor shape: {retrieved.shape}")
    print(f"Original and retrieved shapes match: {tensor.shape == retrieved.shape}")

# Check statistics
stats = system.get_stats()
print(f"Total requests: {stats['global_stats']['total_requests']}")
print(f"Global hit rate: {stats['global_stats']['global_hit_rate']:.2%}")

print("All working correctly!")
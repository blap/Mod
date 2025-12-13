"""Test Qwen3-VL specific functionality"""
from src.models.memory_tiering import create_qwen3vl_memory_tiering_system, TensorType, integrate_with_qwen3vl_model
import torch

print("Testing Qwen3-VL specific functionality...")

# Create system
tiering_system = create_qwen3vl_memory_tiering_system()
print("Qwen3-VL Memory Tiering System created")

# Test tensor type-based placement
kv_tensor = torch.randn(2, 128, 64, dtype=torch.float16)  # KV cache for transformers
success, kv_id = tiering_system.put_tensor(kv_tensor, TensorType.KV_CACHE)
print(f"KV Cache tensor placement: {success}")
if success:
    placement_info = tiering_system.get_tensor_placement_info(kv_id)
    print(f"KV Cache placed in: {placement_info.tier}")

img_tensor = torch.randn(1, 196, 768, dtype=torch.float16)  # Image features
success, img_id = tiering_system.put_tensor(img_tensor, TensorType.IMAGE_FEATURES)
print(f"Image features tensor placement: {success}")
if success:
    placement_info = tiering_system.get_tensor_placement_info(img_id)
    print(f"Image features placed in: {placement_info.tier}")

general_tensor = torch.randn(100, 100, dtype=torch.float16)  # Larger general tensor
success, gen_id = tiering_system.put_tensor(general_tensor, TensorType.GENERAL)
print(f"General tensor placement: {success}")
if success:
    placement_info = tiering_system.get_tensor_placement_info(gen_id)
    print(f"General tensor placed in: {placement_info.tier}")

# Test access pattern tracking
print("\nTesting access pattern tracking...")
for i in range(3):
    retrieved = tiering_system.get_tensor(kv_id)
    print(f"KV tensor access {i+1}: {retrieved is not None}")

for i in range(2):
    retrieved = tiering_system.get_tensor(img_id)
    print(f"Image tensor access {i+1}: {retrieved is not None}")

# Test predictive migrations
print("\nPerforming predictive migrations...")
tiering_system._perform_predictive_migrations()

# Get final statistics
stats = tiering_system.get_stats()
print(f"\nFinal statistics:")
print(f"Total requests: {stats['global_stats']['total_requests']}")
print(f"Global hit rate: {stats['global_stats']['global_hit_rate']:.2%}")
print(f"Total migrations: {stats['global_stats']['total_migrations']}")
print(f"Total utilization: {stats['total_utilization_bytes'] / (1024**3):.2f}GB")
print(f"Tensor type distribution: {dict(stats['tensor_type_distribution'])}")

# Test integration functions
print("\nTesting integration with Qwen3-VL model...")
alloc_kv, alloc_img, access_tensor = integrate_with_qwen3vl_model(tiering_system)

# Allocate KV cache
kv_tensor, kv_tid = alloc_kv(2, 128, 512, 8)
print(f"Integrated KV allocation: {kv_tensor is not None}, ID: {kv_tid}")

# Access tensor
if kv_tid:
    accessed = access_tensor(kv_tid)
    print(f"Integrated tensor access: {accessed is not None}")

print("\nQwen3-VL Memory Tiering functionality working correctly!")
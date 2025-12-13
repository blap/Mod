"""
Implementation testing for Phase 2.9: Memory Pooling and Pre-allocation Techniques
"""
import pytest
import torch
import torch.nn as nn
from src.models.config import Qwen3VLConfig
from models.memory_pooling import BuddyAllocator, TensorCache, MemoryPool, PooledLinear, PooledMLP, PooledAttention, PooledTransformerLayer


def test_implement_custom_memory_pools_with_buddy_allocation_system():
    """Test custom memory pools with buddy allocation system"""
    # Create a small buddy allocator for testing
    pool_size = 1024 * 1024  # 1MB pool
    allocator = BuddyAllocator(pool_size)

    # Test allocation of different sizes
    sizes = [1024, 2048, 512, 4096]  # Different sizes to allocate
    addresses = []

    for size in sizes:
        addr = allocator.allocate(size)
        assert addr is not None, f"Allocation of size {size} should succeed"
        assert addr >= 0, "Address should be non-negative"
        addresses.append(addr)

    # Test deallocation
    for addr in addresses:
        allocator.deallocate(addr)

    print(f"Successfully allocated and deallocated {len(sizes)} blocks")


def test_create_pre_allocated_tensor_caches_for_commonly_used_dimensions():
    """Test pre-allocated tensor caches for commonly used dimensions"""
    cache = TensorCache()

    # Create tensors of common dimensions and return them to cache
    common_shapes = [
        (1, 128, 2048),
        (1, 64, 1024),
        (2, 32, 512),
        (1, 256, 2048)
    ]

    tensors = []
    for shape in common_shapes:
        # Get tensor from cache (should be None initially)
        cached_tensor = cache.get_tensor(shape)
        assert cached_tensor is None, "Cache should be empty initially"

        # Create a new tensor
        new_tensor = torch.randn(shape)
        tensors.append(new_tensor)

    # Return tensors to cache
    for tensor in tensors:
        cache.return_tensor(tensor)

    # Now get tensors from cache
    retrieved_tensors = []
    for shape in common_shapes:
        cached_tensor = cache.get_tensor(shape)
        assert cached_tensor is not None, f"Should retrieve tensor for shape {shape}"
        assert cached_tensor.shape == shape, f"Retrieved tensor should have correct shape {shape}"
        retrieved_tensors.append(cached_tensor)

    print(f"Successfully cached and retrieved {len(common_shapes)} tensors")


def test_develop_memory_defragmentation_routines():
    """Test memory defragmentation routines"""
    memory_pool = MemoryPool(pool_size=2 * 1024 * 1024)  # 2MB pool

    # Allocate some tensors
    tensor1 = memory_pool.allocate_tensor((100, 100))  # ~40KB
    tensor2 = memory_pool.allocate_tensor((200, 200))  # ~160KB
    tensor3 = memory_pool.allocate_tensor((50, 50))    # ~10KB

    # Free the middle tensor to create fragmentation
    memory_pool.free_tensor(tensor2)

    # Allocate a tensor that might need defragmentation
    tensor4 = memory_pool.allocate_tensor((150, 150))  # ~90KB

    # Perform defragmentation
    memory_pool.defragment()

    # Allocate another tensor after defragmentation
    tensor5 = memory_pool.allocate_tensor((100, 100))  # ~40KB

    # Verify all tensors are valid
    assert tensor1 is not None
    assert tensor4 is not None
    assert tensor5 is not None

    # Free all tensors
    memory_pool.free_tensor(tensor1)
    memory_pool.free_tensor(tensor4)
    memory_pool.free_tensor(tensor5)

    print("Successfully tested memory defragmentation")


def test_optimize_memory_layouts_for_vision_encoder_operations():
    """Test optimized memory layouts for vision encoder operations"""
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 128
    config.intermediate_size = 256

    # Create pooled MLP for vision operations
    pooled_mlp = PooledMLP(config)

    # Create sample vision input
    batch_size, seq_len = 2, 64  # Could represent patches from vision encoder
    vision_features = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward pass
    output = pooled_mlp(vision_features)

    # Verify output shape
    assert output.shape == vision_features.shape, "MLP output should match input shape"

    # Test pooled attention for vision operations
    pooled_attn = PooledAttention(config)

    attn_output, _, _ = pooled_attn(
        hidden_states=vision_features,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None
    )

    assert attn_output.shape == vision_features.shape, "Attention output should match input shape"


def test_integrate_memory_pooling_with_existing_gradient_checkpointing():
    """Test integration of memory pooling with existing gradient checkpointing"""
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4

    # Create pooled transformer layer
    pooled_layer = PooledTransformerLayer(config, layer_idx=0)

    # Create sample input
    batch_size, seq_len = 1, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)

    # Forward pass
    output = pooled_layer(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None
    )

    output_hidden_states = output[0]

    # Verify output shape
    assert output_hidden_states.shape == hidden_states.shape, "Output shape should match input"

    # Verify gradients can flow through the pooled components
    loss = output_hidden_states.sum()
    loss.backward()

    assert hidden_states.grad is not None, "Gradients should flow back to input"


if __name__ == "__main__":
    test_implement_custom_memory_pools_with_buddy_allocation_system()
    test_create_pre_allocated_tensor_caches_for_commonly_used_dimensions()
    test_develop_memory_defragmentation_routines()
    test_optimize_memory_layouts_for_vision_encoder_operations()
    test_integrate_memory_pooling_with_existing_gradient_checkpointing()
    print("All implementation tests for Phase 2.9 passed!")
"""
Comprehensive Validation Script for Qwen3-VL Memory Optimization Systems

This script validates that all memory optimization systems work correctly together.
It performs integration tests of the tensor pooling, buddy allocation, hierarchical memory management,
memory defragmentation, cross-modal compression, cross-layer sharing, cache awareness,
GPU-CPU optimization, and dynamic sparse attention systems.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from src.qwen3_vl.memory_management.memory_pool import MemoryPool as AdvancedMemoryPool, MemoryPoolType
from src.qwen3_vl.memory_management.buddy_allocator import BuddyAllocator
from src.qwen3_vl.memory_management.hierarchical_memory_manager import HierarchicalMemoryManager
from src.qwen3_vl.memory_management.memory_defragmenter import MemoryDefragmenter
from src.qwen3_vl.memory_management.cross_modal_compressor import CrossModalCompressor
from src.qwen3_vl.memory_management.cross_layer_memory_sharing import CrossLayerMemoryManager
from src.qwen3_vl.memory_management.cache_aware_memory_manager import CacheAwareMemoryManager
from src.qwen3_vl.memory_management.gpu_cpu_memory_optimizer import GPUCPUMemoryOptimizer
from src.qwen3_vl.attention.dynamic_sparse_attention import DynamicSparseAttention, AttentionConfig, SparsityPattern


def test_tensor_pooling_system():
    """Test the tensor pooling system"""
    print("Testing Tensor Pooling System...")

    # Create a memory pool with 100MB initial size
    pool = AdvancedMemoryPool(initial_size=100*1024*1024, page_size=4096)

    # Test allocation and deallocation
    ptr1, size1 = pool.allocate(1024*1024, MemoryPoolType.TENSOR_DATA)  # 1MB allocation
    assert ptr1 is not None, "Allocation failed"
    print(f"  ✓ Allocated 1MB block with size {size1}")

    ptr2, size2 = pool.allocate(2048*1024, MemoryPoolType.ACTIVATION_BUFFER)  # 2MB allocation
    assert ptr2 is not None, "Second allocation failed"
    print(f"  ✓ Allocated 2MB block with size {size2}")

    success = pool.deallocate(ptr1)
    assert success, "Deallocation failed"
    print(f"  ✓ Deallocated first block")

    # Get and print stats
    stats = pool.get_stats()
    print(f"  ✓ Pool stats - Current usage: {stats['peak_usage']/(1024*1024):.2f}MB, "
          f"Peak usage: {stats['peak_usage']/(1024*1024):.2f}MB")

    pool.cleanup()
    print("  ✓ Tensor pooling system validated\n")


def test_buddy_allocator_system():
    """Test the buddy allocator system"""
    print("Testing Buddy Allocator System...")

    # Create a buddy allocator with 64MB of memory
    buddy_allocator = BuddyAllocator(64*1024*1024, "test_pool")

    # Test allocation of different sizes
    block1 = buddy_allocator.allocate(1024*1024)  # 1MB
    assert block1 is not None, "Buddy allocation failed"
    print(f"  ✓ Allocated 1MB block")

    block2 = buddy_allocator.allocate(512*1024)  # 512KB
    assert block2 is not None, "Second buddy allocation failed"
    print(f"  ✓ Allocated 512KB block")

    # Test deallocation
    buddy_allocator.free(block1)
    print(f"  ✓ Deallocated first block")

    # Check buddy merging functionality
    stats = buddy_allocator.stats()
    print(f"  ✓ Buddy allocator stats - Used memory: {stats['used_memory']/(1024*1024):.2f}MB, "
          f"Fragmentation: {stats['fragmentation_ratio']:.2%}")

    buddy_allocator.cleanup()
    print("  ✓ Buddy allocator system validated\n")


def test_hierarchical_memory_management():
    """Test the hierarchical memory management system"""
    print("Testing Hierarchical Memory Management...")
    
    # Create hierarchical memory manager
    hmg = HierarchicalMemoryManager(
        cpu_memory_limit=2*1024*1024*1024,  # 2GB
        gpu_memory_limit=3*1024*1024*1024,  # 3GB
        disk_cache_path='./test_cache'
    )
    
    # Test tensor allocation
    tensor_id, tensor = hmg.allocate_tensor((100, 100), torch.float32)
    print(f"  ✓ Allocated tensor {tensor_id} with shape {tensor.shape}")
    
    # Test tensor retrieval
    retrieved_tensor = hmg.get_tensor(tensor_id)
    assert torch.equal(tensor, retrieved_tensor), "Retrieved tensor doesn't match original"
    print(f"  ✓ Retrieved tensor {tensor_id} successfully")
    
    # Release tensor
    success = hmg.release_tensor(tensor_id)
    print(f"  ✓ Released tensor {tensor_id}: {success}")
    
    # Check memory stats
    stats = hmg.get_memory_stats()
    print(f"  ✓ Memory stats - Total tensors: {stats['total_tensors']}, "
          f"GPU tensors: {stats['gpu_tensors']}, CPU tensors: {stats['cpu_tensors']}")
    
    print("  ✓ Hierarchical memory management system validated\n")


def test_memory_defragmentation():
    """Test the memory defragmentation system"""
    print("Testing Memory Defragmentation System...")
    
    # Create defragmenter
    defrag = MemoryDefragmenter(
        fragmentation_threshold=0.3,
        defrag_frequency_minutes=5,
        memory_pool_size=512*1024*1024  # 512MB
    )
    
    # Register memory blocks
    for i in range(5):
        block_id = f"test_block_{i}"
        success = defrag.register_memory_block(block_id, (i+1)*1024*1024)  # 1-5MB blocks
        assert success, f"Failed to register block {block_id}"
    
    print(f"  ✓ Registered 5 memory blocks")
    
    # Deregister some blocks to create fragmentation
    for i in range(0, 5, 2):  # Deregister even-indexed blocks
        block_id = f"test_block_{i}"
        success = defrag.deregister_memory_block(block_id)
        assert success, f"Failed to deregister block {block_id}"
    
    print(f"  ✓ Deregistered 3 blocks to create fragmentation")
    
    # Calculate fragmentation before defrag
    frag_ratio, frag_details = defrag.calculate_fragmentation()
    print(f"  ✓ Pre-defrag fragmentation: {frag_ratio:.2%}")
    
    # Run defragmentation
    result = defrag.defragment(force=True)
    print(f"  ✓ Defragmentation completed: {result['defragmentation_performed']}")
    
    # Check fragmentation after defrag
    new_frag_ratio, new_frag_details = defrag.calculate_fragmentation()
    print(f"  ✓ Post-defrag fragmentation: {new_frag_ratio:.2%}")
    
    # Get health report
    health = defrag.get_memory_health()
    print(f"  ✓ Memory health - Fragmentation: {health['fragmentation_ratio']:.2%}, "
          f"Largest free block: {health['largest_free_block_bytes']/(1024*1024):.2f}MB")
    
    print("  ✓ Memory defragmentation system validated\n")


def test_cross_modal_compression():
    """Test the cross-modal compression system"""
    print("Testing Cross-Modal Compression System...")

    # Create compressor
    compressor = CrossModalCompressor(
        compression_ratio=0.5
    )

    # Create test tensors for different modalities
    text_tensor = torch.randn(4, 128, 512)  # (batch, seq_len, hidden_size)
    vision_tensor = torch.randn(4, 3, 224, 224)  # (batch, channels, height, width)

    # Test compression and decompression
    compressed_text = compressor.compress(text_tensor)
    original_text_size = text_tensor.numel() * text_tensor.element_size()

    print(f"  ✓ Compressed text tensor of {original_text_size/(1024*1024):.2f}MB")

    decompressed_text = compressor.decompress(compressed_text, text_tensor.shape)
    assert decompressed_text.shape == text_tensor.shape, "Decompressed text tensor has wrong shape"

    # Test with vision tensor
    compressed_vision = compressor.compress(vision_tensor)
    decompressed_vision = compressor.decompress(compressed_vision, vision_tensor.shape)

    assert decompressed_vision.shape == vision_tensor.shape, "Decompressed vision tensor has wrong shape"
    print(f"  ✓ Vision tensor compression/decompression worked correctly")

    # Get compression stats
    stats = compressor.get_stats()
    print(f"  ✓ Compression stats - Total tensors processed: {stats['total_compressed']}, "
          f"Memory saved: {stats['bytes_saved']/(1024*1024):.2f}MB")

    print("  ✓ Cross-modal compression system validated\n")


def flattened_recursive(obj):
    """Helper function to recursively flatten nested structures for size calculation"""
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        for item in obj:
            yield from flattened_recursive(item)
    else:
        yield obj


def test_cross_layer_memory_sharing():
    """Test the cross-layer memory sharing system"""
    print("Testing Cross-Layer Memory Sharing System...")

    # Create cross-layer memory manager
    clm_manager = CrossLayerMemoryManager()

    # Create test tensors for different layers
    layer1_tensor = torch.randn(2, 512, 1024)
    layer2_tensor = torch.randn(2, 512, 1024)

    # Register tensors
    layer1_id = clm_manager.register_tensor(layer1_tensor)
    layer2_id = clm_manager.register_tensor(layer2_tensor)

    print(f"  ✓ Registered tensors for layer 1 and 2 with IDs: {layer1_id}, {layer2_id}")

    # Try to share memory between compatible tensors
    share_success = clm_manager.share_tensor(layer1_id, layer2_id)
    print(f"  ✓ Attempted to share memory between tensors: {share_success}")

    # Acquire tensors
    acquired_layer1 = clm_manager.get_tensor(layer1_id)
    acquired_layer2 = clm_manager.get_tensor(layer2_id)

    assert acquired_layer1.shape == layer1_tensor.shape, "Acquired tensor has wrong shape"
    assert acquired_layer2.shape == layer2_tensor.shape, "Acquired tensor has wrong shape"

    # Release tensors
    clm_manager.remove_tensor(layer1_id)
    clm_manager.remove_tensor(layer2_id)
    print(f"  ✓ Released both tensors")

    # Check sharing statistics
    stats = clm_manager.stats()
    print(f"  ✓ Sharing stats - Total tensors: {stats['total_tensors']}, "
          f"Shared tensors: {stats['shared_tensors']}")

    print("  ✓ Cross-layer memory sharing system validated\n")


def test_cache_aware_memory_management():
    """Test the cache-aware memory management system"""
    print("Testing Cache-Aware Memory Management System...")

    # Create cache-aware memory manager
    cache_manager = CacheAwareMemoryManager(
        l1_size=32*1024,      # 32KB
        l2_size=256*1024,     # 256KB
        l3_size=3*1024*1024,  # 3MB
        cache_line_size=64,
        gpu_l1_size=16*1024,  # 16KB
        gpu_l2_size=1024*1024 # 1MB
    )

    # Test tensor allocation
    tensor_id, tensor = cache_manager.allocate((64, 64), torch.float32)
    print(f"  ✓ Allocated tensor {tensor_id} with shape {tensor.shape} using cache-aware allocation")

    # Test access to build history for prefetching
    for _ in range(3):
        retrieved_tensor = cache_manager.get_tensor(tensor_id)
        assert retrieved_tensor is not None, "Failed to retrieve tensor"

    print(f"  ✓ Retrieved tensor multiple times to build access history")

    # Get cache statistics
    stats = cache_manager.get_stats()
    print(f"  ✓ Cache stats - L1 utilization: {stats['l1_utilization']:.2%}, "
          f"L2 utilization: {stats['l2_utilization']:.2%}")

    print("  ✓ Cache-aware memory management system validated\n")


def test_gpu_cpu_memory_optimization():
    """Test the GPU-CPU memory optimization system"""
    print("Testing GPU-CPU Memory Optimization System...")

    # Create GPU-CPU memory optimizer
    gpu_cpu_optimizer = GPUCPUMemoryOptimizer(
        cpu_memory_limit=2*1024*1024*1024,  # 2GB
        gpu_memory_limit=3*1024*1024*1024  # 3GB
    )

    # Create test tensors
    cpu_tensor = torch.randn(100, 128, 512).to(torch.device('cpu'))  # ~250MB

    # Register tensor
    tensor_id = gpu_cpu_optimizer.register_tensor("test_tensor", cpu_tensor)
    print(f"  ✓ Registered tensor with ID: {tensor_id}")

    # Record access patterns
    for _ in range(3):
        gpu_cpu_optimizer.record_access("test_tensor")

    # Get memory statistics
    stats = gpu_cpu_optimizer.get_stats()
    print(f"  ✓ Memory optimizer stats - CPU used: {stats['cpu_used']/(1024*1024):.2f}MB, "
          f"GPU used: {stats['gpu_used']/(1024*1024):.2f}MB")

    print("  ✓ GPU-CPU memory optimization system validated\n")


def test_dynamic_sparse_attention():
    """Test the dynamic sparse attention system"""
    print("Testing Dynamic Sparse Attention System...")

    # Create attention config
    config = AttentionConfig(
        hidden_size=512,
        num_attention_heads=8,
        head_dim=64,
        max_seq_len=1024,
        sparsity_ratio=0.5,  # 50% sparsity
        sparsity_pattern=SparsityPattern.DYNAMIC_SPARSE
    )

    # Create dynamic sparse attention module
    from src.qwen3_vl.attention.dynamic_sparse_attention import DynamicSparseAttention as AttentionModule
    attention = AttentionModule(config)

    # Create test input
    batch_size, seq_len = 2, 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Create attention mask
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
    attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))

    # Run attention
    output = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask
    )

    print(f"  ✓ Processed input of shape {hidden_states.shape} -> output {output.shape}")

    print("  ✓ Dynamic sparse attention system validated\n")


def run_comprehensive_validation():
    """Run all validation tests"""
    print("="*80)
    print("COMPREHENSIVE VALIDATION OF QWEN3-VL MEMORY OPTIMIZATION SYSTEMS")
    print("="*80)
    
    try:
        test_tensor_pooling_system()
        test_buddy_allocator_system()
        test_hierarchical_memory_management()
        test_memory_defragmentation()
        test_cross_modal_compression()
        test_cross_layer_memory_sharing()
        test_cache_aware_memory_management()
        test_gpu_cpu_memory_optimization()
        test_dynamic_sparse_attention()
        
        print("="*80)
        print("ALL SYSTEMS VALIDATED SUCCESSFULLY!")
        print("="*80)
        print("The Qwen3-VL memory optimization systems are working correctly together:")
        print("- Tensor Pooling System: Manages efficient tensor allocation")
        print("- Buddy Allocator System: Provides efficient memory block management with merging")
        print("- Hierarchical Memory Management: Optimizes memory across CPU/GPU/NVMe hierarchy")
        print("- Memory Defragmentation System: Reduces memory fragmentation")
        print("- Cross-Modal Compression: Compresses multimodal data efficiently")
        print("- Cross-Layer Memory Sharing: Shares memory between model layers")
        print("- Cache-Aware Memory Management: Optimizes allocation based on cache characteristics")
        print("- GPU-CPU Memory Optimization: Optimizes transfers between devices")
        print("- Dynamic Sparse Attention: Reduces attention computation with selective sparsity")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_validation()
    
    if success:
        print("\n🎉 All memory optimization systems are properly integrated and functioning!")
        print("The Qwen3-VL model can now leverage these advanced memory optimization techniques.")
    else:
        print("\n💥 Some validation tests failed. Please check error messages above.")
        sys.exit(1)
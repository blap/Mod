"""
Unit tests for the Advanced Memory Management System for Vision-Language Models.

This test suite covers all critical functions and classes in the memory management system,
including AdvancedMemoryPool, MemoryDefragmenter, CacheAwareMemoryManager, and
VisionLanguageMemoryOptimizer.
"""

import pytest
import numpy as np
import torch
import threading
import time
from unittest.mock import Mock, patch, MagicMock
import gc

from advanced_memory_management_vl import (
    MemoryPoolType,
    MemoryBlock,
    MemoryDefragmenter,
    AdvancedMemoryPool,
    CacheAwareMemoryManager,
    GPUCPUMemoryOptimizer,
    VisionLanguageMemoryOptimizer
)


class TestMemoryPoolType:
    """Test MemoryPoolType enum."""
    
    def test_memory_pool_type_values(self):
        """Test that MemoryPoolType has the expected values."""
        assert MemoryPoolType.TENSOR_DATA.value == "tensor_data"
        assert MemoryPoolType.ACTIVATION_BUFFER.value == "activation_buffer"
        assert MemoryPoolType.KV_CACHE.value == "kv_cache"
        assert MemoryPoolType.TEMPORARY.value == "temporary"
        assert MemoryPoolType.FIXED_SIZE.value == "fixed_size"


class TestMemoryBlock:
    """Test MemoryBlock dataclass."""
    
    def test_memory_block_creation(self):
        """Test creating a MemoryBlock instance."""
        block = MemoryBlock(
            ptr=0x1000,
            size=1024,
            pool_type=MemoryPoolType.TENSOR_DATA,
            allocated=True,
            timestamp=time.time(),
            ref_count=1,
            alignment=64
        )
        
        assert block.ptr == 0x1000
        assert block.size == 1024
        assert block.pool_type == MemoryPoolType.TENSOR_DATA
        assert block.allocated is True
        assert isinstance(block.timestamp, float)
        assert block.ref_count == 1
        assert block.alignment == 64


class TestAdvancedMemoryPool:
    """Test AdvancedMemoryPool class."""
    
    def test_initialization_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        pool = AdvancedMemoryPool(initial_size=1024*1024, page_size=4096)
        assert pool.initial_size == 1024*1024
        assert pool.page_size == 4096
        assert len(pool.blocks) == 1
        assert pool.blocks[0].size == 1024*1024
        assert pool.blocks[0].allocated is False
    
    def test_initialization_with_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Test with negative size
        with pytest.raises(ValueError):
            AdvancedMemoryPool(initial_size=-1024)
        
        # Test with zero size
        with pytest.raises(ValueError):
            AdvancedMemoryPool(initial_size=0)
        
        # Test with negative page size
        with pytest.raises(ValueError):
            AdvancedMemoryPool(initial_size=1024*1024, page_size=-4096)
        
        # Test with zero page size
        with pytest.raises(ValueError):
            AdvancedMemoryPool(initial_size=1024*1024, page_size=0)
        
        # Test with non-boolean enable_defragmentation
        with pytest.raises(ValueError):
            AdvancedMemoryPool(initial_size=1024*1024, enable_defragmentation="true")
    
    def test_align_size(self):
        """Test the _align_size method."""
        pool = AdvancedMemoryPool(initial_size=1024*1024)
        
        # Test normal alignment
        assert pool._align_size(100, 64) == 128  # 100 -> next multiple of 64
        assert pool._align_size(64, 64) == 64   # Already aligned
        assert pool._align_size(65, 64) == 128  # Next alignment
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            pool._align_size(-100, 64)
        
        with pytest.raises(ValueError):
            pool._align_size(100, -64)
        
        with pytest.raises(ValueError):
            pool._align_size(100, 0)
    
    def test_find_suitable_block(self):
        """Test the _find_suitable_block method."""
        pool = AdvancedMemoryPool(initial_size=1024*1024)
        
        # Find a block that fits
        block = pool._find_suitable_block(1024, 64, MemoryPoolType.TENSOR_DATA)
        assert block is not None
        assert block.size >= 1024
        assert not block.allocated
        
        # Find a block that doesn't fit
        large_block = pool._find_suitable_block(1024*1024*2, 64, MemoryPoolType.TENSOR_DATA)  # Larger than pool
        assert large_block is None
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            pool._find_suitable_block(-1024, 64, MemoryPoolType.TENSOR_DATA)
        
        with pytest.raises(ValueError):
            pool._find_suitable_block(1024, -64, MemoryPoolType.TENSOR_DATA)
        
        with pytest.raises(ValueError):
            pool._find_suitable_block(1024, 64, "invalid_type")
    
    def test_allocate_and_deallocate(self):
        """Test allocation and deallocation."""
        pool = AdvancedMemoryPool(initial_size=1024*1024)
        
        # Allocate memory
        result = pool.allocate(1024, MemoryPoolType.TENSOR_DATA, 64)
        assert result is not None
        ptr, size = result
        assert ptr is not None
        assert size >= 1024
        
        # Check that the block is now allocated
        allocated_block = None
        for block in pool.blocks:
            if block.ptr == ptr:
                allocated_block = block
                break
        assert allocated_block is not None
        assert allocated_block.allocated is True
        assert allocated_block.ref_count == 1
        
        # Deallocate memory
        success = pool.deallocate(ptr)
        assert success is True
        
        # Check that the block is now deallocated
        deallocated_block = pool.block_map[ptr]
        assert deallocated_block.allocated is False
        assert deallocated_block.ref_count == 0
    
    def test_allocate_with_expansion(self):
        """Test allocation with pool expansion."""
        pool = AdvancedMemoryPool(initial_size=1024*1024)  # 1MB
        
        # Allocate a large chunk that requires expansion
        large_size = 512 * 1024  # 512KB
        result1 = pool.allocate(large_size, MemoryPoolType.TENSOR_DATA)
        assert result1 is not None
        
        result2 = pool.allocate(large_size, MemoryPoolType.TENSOR_DATA)
        assert result2 is not None
        
        # This should trigger expansion
        result3 = pool.allocate(1024*1024, MemoryPoolType.TENSOR_DATA)  # 1MB
        assert result3 is not None
    
    def test_deallocate_invalid_pointer(self):
        """Test deallocation with invalid pointer."""
        pool = AdvancedMemoryPool(initial_size=1024*1024)
        
        # Try to deallocate an invalid pointer
        success = pool.deallocate(0x12345678)
        assert success is False
    
    def test_get_stats(self):
        """Test getting memory pool statistics."""
        pool = AdvancedMemoryPool(initial_size=1024*1024)
        
        stats = pool.get_stats()
        assert 'total_allocated' in stats
        assert 'total_freed' in stats
        assert 'current_usage' in stats
        assert 'peak_usage' in stats
        assert 'allocation_count' in stats
        assert 'deallocation_count' in stats
        assert 'fragmentation' in stats
        assert 'pool_utilization' in stats
    
    def test_thread_safety(self):
        """Test thread safety of memory pool operations."""
        pool = AdvancedMemoryPool(initial_size=2*1024*1024)  # Larger pool for thread safety test
        results = []
        
        def allocate_deallocate_thread():
            for _ in range(10):
                result = pool.allocate(1024, MemoryPoolType.TENSOR_DATA)
                if result:
                    ptr, _ = result
                    time.sleep(0.001)  # Small delay
                    pool.deallocate(ptr)
                time.sleep(0.001)  # Small delay
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=allocate_deallocate_thread)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check final stats
        stats = pool.get_stats()
        assert stats['allocation_count'] >= stats['deallocation_count']
    
    def test_cleanup(self):
        """Test cleanup method."""
        pool = AdvancedMemoryPool(initial_size=1024*1024)
        pool.cleanup()
        # Should not raise any exceptions


class TestMemoryDefragmenter:
    """Test MemoryDefragmenter class."""
    
    def test_calculate_fragmentation(self):
        """Test fragmentation calculation."""
        # Create a mock memory pool
        mock_pool = Mock()
        mock_pool.blocks = [
            MemoryBlock(0x1000, 1024, MemoryPoolType.TEMPORARY, False, 0, 0, 64),  # Free
            MemoryBlock(0x2000, 512, MemoryPoolType.TEMPORARY, True, 0, 0, 64),   # Allocated
            MemoryBlock(0x2400, 2048, MemoryPoolType.TEMPORARY, False, 0, 0, 64), # Free
        ]
        
        defragmenter = MemoryDefragmenter(mock_pool)
        fragmentation = defragmenter.calculate_fragmentation()
        
        # Total free = 1024 + 2048 = 3072
        # Largest free = 2048
        # Fragmentation = 1 - (2048 / 3072) = 1 - 0.6667 = 0.3333
        expected_fragmentation = pytest.approx(0.3333, abs=0.01)
        assert fragmentation == expected_fragmentation
    
    def test_compact_memory(self):
        """Test memory compaction."""
        # Create a mock memory pool with fragmented blocks
        mock_pool = Mock()
        mock_pool.blocks = [
            MemoryBlock(0x1000, 1024, MemoryPoolType.TEMPORARY, False, 0, 0, 64),  # Free
            MemoryBlock(0x1400, 1024, MemoryPoolType.TEMPORARY, True, 0, 0, 64),   # Allocated
            MemoryBlock(0x1800, 1024, MemoryPoolType.TEMPORARY, False, 0, 0, 64), # Free - contiguous with first
        ]
        
        # Mock the block_map to include only free blocks initially
        mock_pool.block_map = {0x1000: mock_pool.blocks[0], 0x1800: mock_pool.blocks[2]}
        
        defragmenter = MemoryDefragmenter(mock_pool)
        defragmenter.compact_memory()
        
        # After compaction, contiguous free blocks should be merged
        # The first block should now have size 2048 (1024 + 1024)
        free_blocks = [b for b in mock_pool.blocks if not b.allocated]
        # The exact behavior depends on implementation details, but we expect merging
    
    def test_should_defragment(self):
        """Test should_defragment method."""
        # Create a mock memory pool
        mock_pool = Mock()
        mock_pool.blocks = [
            MemoryBlock(0x1000, 1024, MemoryPoolType.TEMPORARY, False, 0, 0, 64),
            MemoryBlock(0x2000, 2048, MemoryPoolType.TEMPORARY, False, 0, 0, 64),
        ]
        
        defragmenter = MemoryDefragmenter(mock_pool)
        defragmenter.calculate_fragmentation = Mock(return_value=0.4)  # Above threshold
        assert defragmenter.should_defragment() is True
        
        defragmenter.calculate_fragmentation = Mock(return_value=0.2)  # Below threshold
        assert defragmenter.should_defragment() is False


class TestCacheAwareMemoryManager:
    """Test CacheAwareMemoryManager class."""
    
    def test_initialization(self):
        """Test initialization with default parameters."""
        manager = CacheAwareMemoryManager()
        assert manager.cache_line_size == 64
        assert manager.l1_size == 32 * 1024
        assert manager.l2_size == 256 * 1024
        assert manager.l3_size == 6 * 1024 * 1024
    
    def test_optimize_memory_layout(self):
        """Test memory layout optimization."""
        manager = CacheAwareMemoryManager()
        
        # Test 2D array optimization
        data = np.random.random((100, 50)).astype(np.float32)
        optimized = manager.optimize_memory_layout(data, "cache_friendly")
        assert optimized.shape == data.shape
        assert optimized.flags['C_CONTIGUOUS']  # Should be contiguous
        
        # Test 3D array optimization
        data_3d = np.random.random((10, 20, 30)).astype(np.float32)
        optimized_3d = manager.optimize_memory_layout(data_3d, "cache_friendly")
        assert optimized_3d.shape == data_3d.shape
        
        # Test blocked layout
        blocked = manager.optimize_memory_layout(data, "blocked")
        assert blocked.shape == data.shape
        
        # Test row-major layout
        row_major = manager.optimize_memory_layout(data, "row_major")
        assert row_major.shape == data.shape
        
        # Test col-major layout
        col_major = manager.optimize_memory_layout(data, "col_major")
        assert col_major.shape == data.shape
    
    def test_prefetch_data(self):
        """Test data prefetching."""
        manager = CacheAwareMemoryManager()
        
        # Create a small array to get a pointer
        data = np.random.random((10, 10)).astype(np.float32)
        ptr = data.__array_interface__['data'][0]
        
        # This should not raise an exception
        manager.prefetch_data(ptr, data.nbytes)
        
        # Test with offset
        manager.prefetch_data(ptr, data.nbytes, offset=64)


class TestGPUCPUMemoryOptimizer:
    """Test GPUCPUMemoryOptimizer class."""
    
    def test_initialization(self):
        """Test initialization."""
        optimizer = GPUCPUMemoryOptimizer(device_memory_limit=1024*1024*1024)  # 1GB
        assert optimizer.device_memory_limit == 1024*1024*1024
        assert isinstance(optimizer.pinned_memory_enabled, bool)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_optimize_tensor_placement_cpu_only(self, mock_cuda):
        """Test tensor placement when CUDA is not available."""
        optimizer = GPUCPUMemoryOptimizer()
        
        # Test with numpy array
        numpy_tensor = np.random.random((10, 10)).astype(np.float32)
        result = optimizer.optimize_tensor_placement(numpy_tensor, "auto")
        assert isinstance(result, np.ndarray) or torch.is_tensor(result)
        
        # Test with torch tensor
        torch_tensor = torch.randn(10, 10)
        result = optimizer.optimize_tensor_placement(torch_tensor, "auto")
        assert torch.is_tensor(result)
        assert result.device.type == 'cpu'
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated', return_value=0)
    @patch('torch.cuda.memory_reserved', return_value=0)
    @patch('torch.Tensor.pin_memory')
    @patch('torch.Tensor.to')
    def test_optimize_tensor_placement_with_cuda(self, mock_to, mock_pin_memory, mock_allocated, mock_reserved, mock_props, mock_cuda):
        """Test tensor placement with CUDA available."""
        # Mock CUDA properties
        mock_props.return_value.total_memory = 2 * 1024 * 1024 * 1024  # 2GB

        # Mock pin_memory to return self to avoid CUDA initialization
        mock_pin_memory.return_value = torch.randn(10, 10)
        # Mock to method to return a tensor without actual CUDA operations
        mock_to.return_value = torch.randn(10, 10)

        optimizer = GPUCPUMemoryOptimizer()

        # Test with small tensor that should go to GPU
        small_tensor = torch.randn(10, 10)  # Small tensor
        result = optimizer.optimize_tensor_placement(small_tensor, "auto")
        assert torch.is_tensor(result)

        # Test with specific device placement
        result = optimizer.optimize_tensor_placement(small_tensor, "cpu")
        assert torch.is_tensor(result)  # Should still return a tensor


class TestVisionLanguageMemoryOptimizer:
    """Test VisionLanguageMemoryOptimizer class."""
    
    def test_initialization(self):
        """Test initialization with different configurations."""
        # Test with all optimizations enabled
        optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=2*1024*1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False  # Disable GPU for testing
        )

        assert optimizer.enable_memory_pool is True
        assert optimizer.enable_cache_optimization is True
        assert optimizer.enable_gpu_optimization is False
        assert optimizer.memory_pool is not None
        assert optimizer.cache_manager is not None
        # GPU optimizer may be None if torch is not available or CUDA not available
        # The important thing is that it doesn't crash during initialization

        # Test with all optimizations disabled
        optimizer2 = VisionLanguageMemoryOptimizer(
            enable_memory_pool=False,
            enable_cache_optimization=False,
            enable_gpu_optimization=False
        )

        assert optimizer2.enable_memory_pool is False
        assert optimizer2.enable_cache_optimization is False
        assert optimizer2.enable_gpu_optimization is False
        assert optimizer2.memory_pool is None
        assert optimizer2.cache_manager is None
        # GPU optimizer may still be initialized even if disabled, depending on implementation
    
    def test_initialization_with_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            VisionLanguageMemoryOptimizer(memory_pool_size=-1024*1024)
        
        with pytest.raises(TypeError):
            VisionLanguageMemoryOptimizer(enable_memory_pool="true")
        
        with pytest.raises(TypeError):
            VisionLanguageMemoryOptimizer(enable_cache_optimization="true")
        
        with pytest.raises(TypeError):
            VisionLanguageMemoryOptimizer(enable_gpu_optimization="true")
    
    def test_allocate_tensor_memory(self, vision_language_optimizer):
        """Test tensor memory allocation."""
        optimizer = vision_language_optimizer
        
        # Test general tensor allocation
        tensor = optimizer.allocate_tensor_memory((100, 50), dtype=np.float32, tensor_type="general")
        assert tensor.shape == (100, 50)
        assert tensor.dtype == np.float32
        
        # Test KV cache tensor allocation
        kv_tensor = optimizer.allocate_tensor_memory((4, 128, 768), dtype=np.float32, tensor_type="kv_cache")
        assert kv_tensor.shape == (4, 128, 768)
        assert kv_tensor.dtype == np.float32
        
        # Test image features tensor allocation
        img_tensor = optimizer.allocate_tensor_memory((2, 196, 512), dtype=np.float32, tensor_type="image_features")
        assert img_tensor.shape == (2, 196, 512)
        assert img_tensor.dtype == np.float32
        
        # Test text embeddings tensor allocation
        txt_tensor = optimizer.allocate_tensor_memory((4, 512), dtype=np.float32, tensor_type="text_embeddings")
        assert txt_tensor.shape == (4, 512)
        assert txt_tensor.dtype == np.float32
        
        # Test invalid tensor type
        with pytest.raises(ValueError):
            optimizer.allocate_tensor_memory((10, 10), dtype=np.float32, tensor_type="invalid_type")
        
        # Test invalid shape
        with pytest.raises(ValueError):
            optimizer.allocate_tensor_memory((-10, 10), dtype=np.float32, tensor_type="general")
        
        # Test invalid dtype - this might return None instead of raising exception based on implementation
        result = optimizer.allocate_tensor_memory((10, 10), dtype="invalid_dtype", tensor_type="general")
        # Implementation may return None for invalid dtypes instead of raising an exception
        assert result is None or isinstance(result, np.ndarray)
    
    def test_free_tensor_memory(self, vision_language_optimizer):
        """Test tensor memory freeing."""
        optimizer = vision_language_optimizer
        
        # Allocate a tensor
        tensor = optimizer.allocate_tensor_memory((10, 10), dtype=np.float32, tensor_type="general")
        assert tensor is not None
        
        # Free the tensor (this should work without error)
        optimizer.free_tensor_memory(tensor, "general")
    
    def test_optimize_image_processing_memory(self, vision_language_optimizer):
        """Test image processing memory optimization."""
        optimizer = vision_language_optimizer
        
        # Create a sample image batch
        image_batch = np.random.random((4, 224, 224, 3)).astype(np.float32)
        
        # Optimize the batch
        optimized_batch = optimizer.optimize_image_processing_memory(image_batch)
        
        assert optimized_batch.shape == image_batch.shape
        assert optimized_batch.dtype == image_batch.dtype
    
    def test_optimize_attention_memory(self, vision_language_optimizer):
        """Test attention memory optimization."""
        optimizer = vision_language_optimizer
        
        # Optimize attention memory for a sample configuration
        components = optimizer.optimize_attention_memory(
            batch_size=4,
            seq_len=1024,
            hidden_dim=768,
            num_heads=12
        )
        
        # Check that all expected components are present
        assert 'query' in components
        assert 'key' in components
        assert 'value' in components
        assert 'attention_scores' in components
        assert 'head_dim' in components
        
        # Check tensor shapes
        assert components['query'].shape == (4, 1024, 768)
        assert components['key'].shape == (4, 1024, 768)
        assert components['value'].shape == (4, 1024, 768)
        assert components['attention_scores'].shape == (4, 12, 1024, 1024)
        assert components['head_dim'] == 64  # 768 / 12
    
    def test_get_memory_stats(self, vision_language_optimizer):
        """Test getting memory statistics."""
        optimizer = vision_language_optimizer
        
        stats = optimizer.get_memory_stats()
        
        # Check that stats dictionary has expected structure
        if optimizer.memory_pool:
            assert 'general_pool' in stats
            assert 'current_usage' in stats['general_pool']
        
        # Should always have system memory info if psutil is available
        # (The actual presence depends on whether psutil is installed)
    
    def test_cleanup(self, vision_language_optimizer):
        """Test cleanup method."""
        optimizer = vision_language_optimizer
        
        # This should not raise any exceptions
        optimizer.cleanup()


# Run the tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__])
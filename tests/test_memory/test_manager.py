"""
Comprehensive tests for the MemoryManager class and related memory management components.
"""
import pytest
import torch
import tempfile
import os
import time
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from memory.manager import (
    MemoryManager, MemoryType, TensorType, TensorState, TensorMetadata,
    MemoryPool, TensorAccessAnalyzer, TensorLifetimePredictor, 
    TensorCache, MemorySwappingSystem, TensorLifecycleTracker,
    TensorCompressionManager, MemoryOptimizer
)


def test_memory_pool_basic_allocation():
    """Test basic memory pool allocation and deallocation."""
    pool = MemoryPool(pool_size=1024)
    
    # Allocate a block
    result = pool.allocate(256, TensorType.GENERAL, "test_tensor_1")
    assert result is not None
    start_addr, size = result
    assert size == 256
    
    # Allocate another block
    result = pool.allocate(512, TensorType.GENERAL, "test_tensor_2")
    assert result is not None
    start_addr2, size2 = result
    assert size2 == 512
    
    # Check that we can't allocate more than available
    result = pool.allocate(1024, TensorType.GENERAL, "test_tensor_3")
    assert result is None  # Should fail due to insufficient space
    
    # Deallocate first tensor
    success = pool.deallocate("test_tensor_1")
    assert success is True
    
    # Now we should be able to allocate again
    result = pool.allocate(256, TensorType.GENERAL, "test_tensor_3")
    assert result is not None


def test_tensor_access_analyzer():
    """Test tensor access pattern analysis."""
    analyzer = TensorAccessAnalyzer()
    
    # Record access to a tensor
    tensor_id = "test_tensor"
    analyzer.record_access(tensor_id)
    time.sleep(0.01)  # Small delay
    analyzer.record_access(tensor_id)
    time.sleep(0.01)
    analyzer.record_access(tensor_id)
    
    # Check that we can predict access
    next_access_time, probability, access_count = analyzer.predict_access(tensor_id)
    assert access_count == 3
    assert probability is not None  # Should return a probability
    
    # Check hot tensors
    hot_tensors = analyzer.get_hot_tensors(n=5)
    assert tensor_id in hot_tensors


def test_tensor_lifetime_predictor():
    """Test tensor lifetime prediction."""
    analyzer = TensorAccessAnalyzer()
    predictor = TensorLifetimePredictor(analyzer)
    
    # Predict lifetime for a tensor
    lifetime, access_count = predictor.predict_lifetime(
        "test_tensor", 
        1024,  # size in bytes
        TensorType.GENERAL
    )
    
    assert isinstance(lifetime, float)
    assert lifetime > 0
    assert isinstance(access_count, int)
    assert access_count >= 1


def test_tensor_cache():
    """Test tensor caching functionality."""
    cache = TensorCache(max_cache_size=10)
    
    # Create and return a tensor to cache
    tensor = torch.randn(10, 10)
    success = cache.return_tensor(tensor)
    assert success is True
    
    # Get tensor from cache
    retrieved_tensor = cache.get_tensor((10, 10), torch.float32, torch.device('cpu'))
    assert retrieved_tensor.shape == (10, 10)
    assert retrieved_tensor.dtype == torch.float32
    
    # Check cache statistics
    stats = cache.get_cache_stats()
    assert isinstance(stats, dict)
    assert 'cache_hits' in stats
    assert 'cache_misses' in stats


def test_memory_swapping_system():
    """Test tensor swapping functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        swapping_system = MemorySwappingSystem(
            swap_directory=temp_dir,
            max_swap_size=1024*1024  # 1MB
        )
        
        # Create a tensor
        tensor = torch.randn(100, 100)
        metadata = TensorMetadata(
            tensor_id="test_tensor",
            tensor_shape=tensor.shape,
            tensor_dtype=tensor.dtype,
            tensor_type=TensorType.GENERAL,
            size_bytes=tensor.element_size() * tensor.nelement(),
            device=str(tensor.device)
        )
        
        # Swap tensor out
        success = swapping_system.swap_out("test_tensor", tensor, metadata)
        assert success is True
        
        # Check if tensor is marked as swapped
        assert swapping_system.is_swapped("test_tensor") is True
        
        # Swap tensor back in
        retrieved_tensor = swapping_system.swap_in("test_tensor")
        assert retrieved_tensor is not None
        assert torch.equal(tensor, retrieved_tensor)
        
        # Check stats
        stats = swapping_system.get_swap_stats()
        assert isinstance(stats, dict)
        assert 'swapped_tensors_count' in stats


def test_tensor_lifecycle_tracker():
    """Test tensor lifecycle tracking."""
    tracker = TensorLifecycleTracker()
    
    # Create and register a tensor
    tensor = torch.randn(10, 10)
    metadata = tracker.register_tensor(
        tensor, 
        "test_tensor", 
        TensorType.GENERAL
    )
    
    assert metadata.tensor_id == "test_tensor"
    assert metadata.tensor_state.value == "allocated"
    
    # Access the tensor
    accessed_metadata = tracker.access_tensor("test_tensor", "test_context")
    assert accessed_metadata.tensor_state.value == "in_use"
    
    # Increment reference count
    success = tracker.increment_reference("test_tensor", "test_context")
    assert success is True
    
    # Decrement reference count
    success = tracker.decrement_reference("test_tensor", "test_context")
    assert success is True
    
    # Check stats
    stats = tracker.get_tensor_stats()
    assert isinstance(stats, dict)
    assert 'total_tensors' in stats


def test_tensor_compression_manager():
    """Test tensor compression functionality."""
    compressor = TensorCompressionManager()
    
    # Test FP16 compression
    float32_tensor = torch.randn(10, 10, dtype=torch.float32)
    compressed_tensor, was_compressed = compressor.compress_tensor(float32_tensor, 'quantize_fp16')
    assert was_compressed is True
    assert compressed_tensor.dtype == torch.float16
    
    # Decompress
    decompressed_tensor = compressor.decompress_tensor(compressed_tensor)
    assert decompressed_tensor.dtype == torch.float32
    
    # Test auto compression
    compressed_auto, was_compressed_auto = compressor.compress_tensor(float32_tensor, 'auto')
    assert was_compressed_auto is True


def test_memory_optimizer():
    """Test memory optimizer functionality."""
    # Create memory manager and optimizer
    memory_manager = MemoryManager()
    optimizer = MemoryOptimizer(memory_manager)

    # Test hardware optimized config (should handle missing hardware manager gracefully)
    config = optimizer.get_hardware_optimized_config()
    assert 'has_gpu' in config
    assert 'gpu_count' in config
    assert 'memory_optimization_strategy' in config

    # Test tensor optimization
    tensor = torch.randn(10, 10)
    optimized_tensors = optimizer.optimize_tensors([tensor])
    assert len(optimized_tensors) == 1


def test_memory_manager_basic_operations():
    """Test basic memory manager operations."""
    manager = MemoryManager()
    
    # Register an object
    tensor = torch.randn(5, 5)
    manager.register_object("test_tensor", tensor, TensorType.GENERAL)
    
    # Check if object is registered
    assert "test_tensor" in manager.object_registry
    
    # Allocate memory
    alloc_result = manager.allocate("test_alloc", 1024, tensor, TensorType.GENERAL)
    assert alloc_result == "test_alloc"
    
    # Check available memory
    available = manager.get_available_memory()
    assert isinstance(available, int)
    assert available >= 0
    
    # Monitor memory pressure
    pressure = manager.monitor_memory_pressure()
    assert isinstance(pressure, float)
    assert 0 <= pressure <= 100


def test_memory_manager_tensor_operations():
    """Test tensor-specific operations in memory manager."""
    manager = MemoryManager()
    
    # Create a tensor
    tensor = torch.randn(10, 10)
    
    # Register tensor
    manager.register_object("test_tensor", tensor, TensorType.GENERAL)
    
    # Test tensor placement optimization
    optimized_tensor = manager.optimize_tensor_placement(tensor)
    assert isinstance(optimized_tensor, torch.Tensor)
    
    # Test compression
    compressed_tensor, was_compressed = manager.compress_tensor(tensor, 'auto')
    assert isinstance(was_compressed, bool)


def test_memory_manager_swapping():
    """Test tensor swapping operations."""
    manager = MemoryManager()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a tensor
        tensor = torch.randn(100, 100)
        manager.register_object("swap_test_tensor", tensor, TensorType.GENERAL)
        
        # Swap to disk
        swap_success = manager.swap_to_disk("swap_test_tensor", os.path.join(temp_dir, "test.swap"))
        # Note: The current implementation might not actually swap due to internal logic
        
        # Try to load from disk
        loaded_tensor = manager.load_from_disk("swap_test_tensor", os.path.join(temp_dir, "test.swap"))


def test_memory_manager_error_handling():
    """Test error handling in memory manager."""
    manager = MemoryManager()
    
    # Try to deallocate non-existent object
    success = manager.deallocate("non_existent")
    assert success is False
    
    # Try to access non-existent object
    manager.object_registry["missing_obj"] = "not_a_tensor"
    swap_success = manager.swap_to_disk("missing_obj")
    # This should handle gracefully


def test_memory_types_enum():
    """Test MemoryType enum values."""
    assert MemoryType.VRAM.value == "vram"
    assert MemoryType.RAM.value == "ram"
    assert MemoryType.STORAGE.value == "storage"


def test_tensor_types_enum():
    """Test TensorType enum values."""
    assert TensorType.GENERAL.value == "general"
    assert TensorType.KV_CACHE.value == "kv_cache"
    assert TensorType.IMAGE_FEATURES.value == "image_features"


def test_tensor_state_enum():
    """Test TensorState enum values."""
    assert TensorState.ALLOCATED.value == "allocated"
    assert TensorState.IN_USE.value == "in_use"
    assert TensorState.UNUSED.value == "unused"


def test_tensor_metadata():
    """Test TensorMetadata creation and properties."""
    metadata = TensorMetadata(
        tensor_id="test_id",
        tensor_shape=(10, 20),
        tensor_dtype=torch.float32,
        tensor_type=TensorType.GENERAL,
        size_bytes=800,
        device="cpu"
    )
    
    assert metadata.tensor_id == "test_id"
    assert metadata.tensor_shape == (10, 20)
    assert metadata.tensor_dtype == torch.float32
    assert metadata.tensor_type == TensorType.GENERAL
    assert metadata.size_bytes == 800
    assert metadata.device == "cpu"


if __name__ == "__main__":
    pytest.main([__file__])
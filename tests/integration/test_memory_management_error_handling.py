"""
Tests for error handling and input validation in Qwen3-VL optimization codebase
"""
import unittest
import sys
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from enum import Enum

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_memory_management_vl import AdvancedMemoryPool, MemoryPoolType, MemoryBlock
from advanced_memory_pooling_system import BuddyAllocator, MemoryPool, TensorType
from advanced_memory_swapping_system import AdvancedMemorySwapper, MemoryPressureMonitor, ClockSwapAlgorithm, MemoryRegionType


class TestAdvancedMemoryPool(unittest.TestCase):
    """Test AdvancedMemoryPool error handling and input validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pool = AdvancedMemoryPool(initial_size=1024*1024)  # 1MB pool
    
    def test_invalid_initial_size(self):
        """Test initialization with invalid size"""
        with self.assertRaises(ValueError):
            AdvancedMemoryPool(initial_size=-1)
        
        with self.assertRaises(ValueError):
            AdvancedMemoryPool(initial_size=0)
    
    def test_invalid_alignment(self):
        """Test allocation with invalid alignment"""
        with self.assertRaises(ValueError):
            self.pool.allocate(100, alignment=-1)
        
        with self.assertRaises(ValueError):
            self.pool.allocate(100, alignment=0)
    
    def test_invalid_size_allocation(self):
        """Test allocation with invalid size"""
        with self.assertRaises(ValueError):
            self.pool.allocate(-1)
        
        with self.assertRaises(ValueError):
            self.pool.allocate(0)
    
    def test_deallocate_invalid_pointer(self):
        """Test deallocation with invalid pointer"""
        result = self.pool.deallocate(999999999)  # Invalid pointer
        self.assertFalse(result)
    
    def test_deallocate_already_freed(self):
        """Test deallocation of already freed block"""
        ptr, size = self.pool.allocate(100)
        self.pool.deallocate(ptr)
        result = self.pool.deallocate(ptr)  # Try to deallocate again
        self.assertFalse(result)
    
    def test_allocate_too_large(self):
        """Test allocation of size larger than pool"""
        # Try to allocate more than pool size - the pool should expand to accommodate
        # So we'll test with a very large size that exceeds realistic expansion
        large_size = 1024 * 1024 * 1024 * 100  # 100GB - should fail to expand
        result = self.pool.allocate(large_size)
        # This should return None as the pool can't expand to that size
        self.assertIsNone(result)
    
    def test_cleanup_multiple_times(self):
        """Test cleanup method called multiple times"""
        self.pool.cleanup()
        self.pool.cleanup()  # Should not raise exception


class TestBuddyAllocator(unittest.TestCase):
    """Test BuddyAllocator error handling and input validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.allocator = BuddyAllocator(total_size=1024*1024)  # 1MB
    
    def test_invalid_total_size(self):
        """Test initialization with invalid total size"""
        with self.assertRaises(ValueError):
            BuddyAllocator(total_size=-1)
        
        with self.assertRaises(ValueError):
            BuddyAllocator(total_size=0)
    
    def test_invalid_min_block_size(self):
        """Test initialization with invalid min block size"""
        with self.assertRaises(ValueError):
            BuddyAllocator(total_size=1024*1024, min_block_size=-1)
        
        with self.assertRaises(ValueError):
            BuddyAllocator(total_size=1024*1024, min_block_size=0)
    
    def test_invalid_allocation_size(self):
        """Test allocation with invalid size"""
        with self.assertRaises(ValueError):
            self.allocator.allocate(-1, TensorType.KV_CACHE, "test_id")
        
        with self.assertRaises(ValueError):
            self.allocator.allocate(0, TensorType.KV_CACHE, "test_id")
    
    def test_invalid_tensor_type(self):
        """Test allocation with invalid tensor type"""
        with self.assertRaises(ValueError):
            self.allocator.allocate(100, "invalid_type", "test_id")
    
    def test_invalid_tensor_id(self):
        """Test allocation with invalid tensor id"""
        with self.assertRaises(ValueError):
            self.allocator.allocate(100, TensorType.KV_CACHE, "")
    
    def test_deallocate_invalid_block(self):
        """Test deallocation of invalid block"""
        # Create a block not in the allocator
        invalid_block = MagicMock()
        invalid_block.start_addr = 999999999
        invalid_block.size = 100
        
        with self.assertRaises(ValueError):
            self.allocator.deallocate(invalid_block)
    
    def test_deallocate_twice(self):
        """Test deallocation of already deallocated block"""
        block = self.allocator.allocate(100, TensorType.KV_CACHE, "test_id")
        if block:
            self.allocator.deallocate(block)
            with self.assertRaises(ValueError):
                self.allocator.deallocate(block)  # Try to deallocate again


class TestMemoryPool(unittest.TestCase):
    """Test MemoryPool error handling and input validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pool = MemoryPool(TensorType.KV_CACHE, pool_size=1024*1024)
    
    def test_invalid_pool_size(self):
        """Test initialization with invalid pool size"""
        with self.assertRaises(ValueError):
            MemoryPool(TensorType.KV_CACHE, pool_size=-1)
        
        with self.assertRaises(ValueError):
            MemoryPool(TensorType.KV_CACHE, pool_size=0)
    
    def test_invalid_min_block_size(self):
        """Test initialization with invalid min block size"""
        with self.assertRaises(ValueError):
            MemoryPool(TensorType.KV_CACHE, pool_size=1024*1024, min_block_size=-1)
        
        with self.assertRaises(ValueError):
            MemoryPool(TensorType.KV_CACHE, pool_size=1024*1024, min_block_size=0)
    
    def test_invalid_allocation_size(self):
        """Test allocation with invalid size"""
        with self.assertRaises(ValueError):
            self.pool.allocate(-1, "test_id")
        
        with self.assertRaises(ValueError):
            self.pool.allocate(0, "test_id")
    
    def test_invalid_tensor_id(self):
        """Test allocation with invalid tensor id"""
        with self.assertRaises(ValueError):
            self.pool.allocate(100, "")
    
    def test_deallocate_invalid_tensor_id(self):
        """Test deallocation with invalid tensor id"""
        result = self.pool.deallocate("nonexistent_id")
        self.assertFalse(result)
    
    def test_deallocate_empty_tensor_id(self):
        """Test deallocation with empty tensor id"""
        result = self.pool.deallocate("")
        self.assertFalse(result)


class TestMemoryPressureMonitor(unittest.TestCase):
    """Test MemoryPressureMonitor error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MemoryPressureMonitor()
    
    def test_invalid_thresholds(self):
        """Test initialization with invalid thresholds"""
        with self.assertRaises(ValueError):
            MemoryPressureMonitor(ram_thresholds=(-1, 0.8, 0.9))
        
        with self.assertRaises(ValueError):
            MemoryPressureMonitor(ram_thresholds=(0.5, 0.3, 0.9))  # Out of order
        
        with self.assertRaises(ValueError):
            MemoryPressureMonitor(ram_thresholds=(0.5, 0.8, 1.5))  # Above 1.0


class TestClockSwapAlgorithm(unittest.TestCase):
    """Test ClockSwapAlgorithm error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = ClockSwapAlgorithm()
    
    def test_invalid_block_operations(self):
        """Test operations with invalid blocks"""
        # Test with None block
        result = self.algorithm.select_victim()
        self.assertIsNone(result)
    
    def test_access_nonexistent_block(self):
        """Test accessing a block that doesn't exist"""
        self.algorithm.access_block("nonexistent_block")
        # Should not raise an exception


class TestAdvancedMemorySwapper(unittest.TestCase):
    """Test AdvancedMemorySwapper error handling and input validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.swapper = AdvancedMemorySwapper()
    
    def test_invalid_initialization_params(self):
        """Test initialization with invalid parameters"""
        with self.assertRaises(ValueError):
            AdvancedMemorySwapper(swap_threshold=-1)
        
        with self.assertRaises(ValueError):
            AdvancedMemorySwapper(swap_threshold=1.5)
        
        with self.assertRaises(ValueError):
            AdvancedMemorySwapper(max_swap_size=-1)
    
    def test_invalid_block_registration(self):
        """Test registration with invalid parameters"""
        with self.assertRaises(ValueError):
            self.swapper.register_memory_block("", 100)  # Empty ID
        
        with self.assertRaises(ValueError):
            self.swapper.register_memory_block("test_id", -1)  # Negative size
        
        with self.assertRaises(ValueError):
            self.swapper.register_memory_block("test_id", 0)  # Zero size
    
    def test_invalid_block_access(self):
        """Test access to invalid block"""
        result = self.swapper.access_memory_block("nonexistent_block")
        self.assertIsNone(result)
    
    def test_invalid_block_unregistration(self):
        """Test unregistration of invalid block"""
        result = self.swapper.unregister_memory_block("nonexistent_block")
        self.assertFalse(result)
    
    def test_empty_block_id_unregistration(self):
        """Test unregistration with empty block ID"""
        result = self.swapper.unregister_memory_block("")
        self.assertFalse(result)


class TestMemoryRegionType(unittest.TestCase):
    """Test MemoryRegionType enum validation"""
    
    def test_valid_region_types(self):
        """Test that all expected region types exist"""
        expected_types = ['TENSOR_DATA', 'ACTIVATION_BUFFER', 'KV_CACHE', 'TEMPORARY']
        for type_name in expected_types:
            self.assertTrue(hasattr(MemoryRegionType, type_name))
    
    def test_region_type_values(self):
        """Test that region type values are valid strings"""
        for region_type in MemoryRegionType:
            self.assertIsInstance(region_type.value, str)
            self.assertTrue(len(region_type.value) > 0)


class TestTensorType(unittest.TestCase):
    """Test TensorType enum validation"""
    
    def test_valid_tensor_types(self):
        """Test that all expected tensor types exist"""
        expected_types = ['KV_CACHE', 'IMAGE_FEATURES', 'TEXT_EMBEDDINGS', 'GRADIENTS', 'ACTIVATIONS', 'PARAMETERS']
        for type_name in expected_types:
            self.assertTrue(hasattr(TensorType, type_name))
    
    def test_tensor_type_values(self):
        """Test that tensor type values are valid strings"""
        for tensor_type in TensorType:
            self.assertIsInstance(tensor_type.value, str)
            self.assertTrue(len(tensor_type.value) > 0)


if __name__ == '__main__':
    unittest.main()
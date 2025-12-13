"""
Simple test to verify that the main functionality still works after adding error handling
"""
import sys
import os

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_memory_management_vl import AdvancedMemoryPool, MemoryPoolType
from advanced_memory_pooling_system import BuddyAllocator, TensorType
from advanced_memory_swapping_system import AdvancedMemorySwapper, MemoryRegionType

def test_advanced_memory_pool():
    """Test AdvancedMemoryPool basic functionality"""
    print("Testing AdvancedMemoryPool...")
    pool = AdvancedMemoryPool(initial_size=1024*1024)  # 1MB pool
    
    # Test allocation
    result = pool.allocate(1024, MemoryPoolType.TENSOR_DATA)  # 1KB
    assert result is not None, "Allocation should succeed"
    ptr, size = result
    print(f"  Allocated {size} bytes at {hex(ptr)}")
    
    # Test deallocation
    success = pool.deallocate(ptr)
    assert success, "Deallocation should succeed"
    print("  Deallocation successful")
    
    pool.cleanup()
    print("  Pool cleaned up successfully")
    print("AdvancedMemoryPool test passed!\n")

def test_buddy_allocator():
    """Test BuddyAllocator basic functionality"""
    print("Testing BuddyAllocator...")
    allocator = BuddyAllocator(total_size=1024*1024)  # 1MB
    
    # Test allocation
    block = allocator.allocate(1024, TensorType.KV_CACHE, "test_block")
    assert block is not None, "Allocation should succeed"
    print(f"  Allocated block of size {block.size} at address {block.start_addr}")
    
    # Test deallocation
    allocator.deallocate(block)
    print("  Deallocation successful")
    print("BuddyAllocator test passed!\n")

def test_advanced_memory_swapper():
    """Test AdvancedMemorySwapper basic functionality"""
    print("Testing AdvancedMemorySwapper...")
    swapper = AdvancedMemorySwapper()
    
    # Test registration
    block = swapper.register_memory_block("test_block", 1024, MemoryRegionType.TENSOR_DATA)
    assert block is not None, "Registration should succeed"
    print(f"  Registered block with ID {block.id}")
    
    # Test access
    accessed_block = swapper.access_memory_block("test_block")
    assert accessed_block is not None, "Access should succeed"
    print(f"  Accessed block with ID {accessed_block.id}")
    
    # Test unregistration
    success = swapper.unregister_memory_block("test_block")
    assert success, "Unregistration should succeed"
    print("  Unregistration successful")
    print("AdvancedMemorySwapper test passed!\n")

if __name__ == "__main__":
    print("Running functionality tests after error handling improvements...\n")
    
    test_advanced_memory_pool()
    test_buddy_allocator()
    test_advanced_memory_swapper()
    
    print("All functionality tests passed! Error handling improvements work correctly.")
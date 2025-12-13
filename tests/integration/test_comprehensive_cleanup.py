"""
Comprehensive test for the AdvancedMemoryPool cleanup fix.
Tests various edge cases and scenarios.
"""

import os
import tempfile
import sys
import gc
from advanced_memory_management_vl import AdvancedMemoryPool, MemoryPoolType

def test_edge_cases():
    """Test edge cases for memory pool cleanup"""
    print("Testing edge cases...")
    
    # Test 1: Create pool and immediately cleanup without any allocations
    print("  Test 1: Immediate cleanup without allocations")
    pool1 = AdvancedMemoryPool(initial_size=256*1024)
    temp_file_1 = getattr(pool1, 'temp_file', None)
    temp_filename_1 = getattr(temp_file_1, 'name', None) if temp_file_1 else None
    pool1.cleanup()
    if temp_filename_1 and os.path.exists(temp_filename_1):
        print(f"    FAIL: Temp file {temp_filename_1} still exists")
        return False
    print("    PASS: Immediate cleanup works")
    
    # Test 2: Create pool, allocate, deallocate, then cleanup
    print("  Test 2: Allocate, deallocate, then cleanup")
    pool2 = AdvancedMemoryPool(initial_size=256*1024)
    temp_file_2 = getattr(pool2, 'temp_file', None)
    temp_filename_2 = getattr(temp_file_2, 'name', None) if temp_file_2 else None
    
    ptr, size = pool2.allocate(1024)
    pool2.deallocate(ptr)
    pool2.cleanup()
    
    if temp_filename_2 and os.path.exists(temp_filename_2):
        print(f"    FAIL: Temp file {temp_filename_2} still exists")
        return False
    print("    PASS: Allocate-deallocate-cleanup works")
    
    # Test 3: Multiple allocations and deallocations before cleanup
    print("  Test 3: Multiple operations before cleanup")
    pool3 = AdvancedMemoryPool(initial_size=512*1024)
    temp_file_3 = getattr(pool3, 'temp_file', None)
    temp_filename_3 = getattr(temp_file_3, 'name', None) if temp_file_3 else None
    
    allocations = []
    for i in range(5):
        ptr, size = pool3.allocate(1024 * (i + 1))
        allocations.append(ptr)
    
    for ptr in allocations:
        pool3.deallocate(ptr)
    
    pool3.cleanup()
    if temp_filename_3 and os.path.exists(temp_filename_3):
        print(f"    FAIL: Temp file {temp_filename_3} still exists")
        return False
    print("    PASS: Multiple operations before cleanup works")
    
    # Test 4: Pool expansion and cleanup
    print("  Test 4: Pool expansion then cleanup")
    pool4 = AdvancedMemoryPool(initial_size=128*1024)  # Small initial size
    temp_file_4 = getattr(pool4, 'temp_file', None)
    temp_filename_4 = getattr(temp_file_4, 'name', None) if temp_file_4 else None
    
    # Request a large allocation to force expansion
    ptr, size = pool4.allocate(1024*1024)  # 1MB, much larger than initial
    pool4.deallocate(ptr)
    pool4.cleanup()
    
    if temp_filename_4 and os.path.exists(temp_filename_4):
        print(f"    FAIL: Temp file {temp_filename_4} still exists")
        return False
    print("    PASS: Pool expansion and cleanup works")
    
    return True

def test_exception_handling():
    """Test that cleanup handles exceptions gracefully"""
    print("\nTesting exception handling during cleanup...")
    
    # The cleanup method should handle cases where mmap or file is already closed
    pool = AdvancedMemoryPool(initial_size=128*1024)
    
    # Manually close the mmap to simulate already-closed state
    try:
        pool.pool_ptr.close()
        # Now call cleanup - should not raise an exception
        pool.cleanup()  # This should handle the already-closed mmap gracefully
        print("  PASS: Cleanup handles already-closed mmap gracefully")
    except Exception as e:
        print(f"  FAIL: Exception during cleanup with already-closed mmap: {e}")
        return False
    
    return True

def test_with_context_manager():
    """Test using the memory pool with context manager pattern (if implemented)"""
    print("\nTesting cleanup with object deletion...")
    
    def create_and_delete_pool():
        pool = AdvancedMemoryPool(initial_size=256*1024)
        temp_file = getattr(pool, 'temp_file', None)
        temp_filename = getattr(temp_file, 'name', None) if temp_file else None
        
        ptr, size = pool.allocate(2048)
        pool.deallocate(ptr)
        
        # Return the temp filename so we can check if it still exists after deletion
        return temp_filename
    
    temp_filename = create_and_delete_pool()
    
    # Force garbage collection
    gc.collect()
    
    if temp_filename and os.path.exists(temp_filename):
        print(f"  FAIL: Temp file {temp_filename} still exists after object deletion")
        return False
    else:
        print("  PASS: Temp file cleaned up after object deletion")
        return True

if __name__ == "__main__":
    print("Running comprehensive tests for AdvancedMemoryPool cleanup...\n")
    
    success = True
    success &= test_edge_cases()
    success &= test_exception_handling()
    success &= test_with_context_manager()
    
    if success:
        print("\nAll comprehensive tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome comprehensive tests FAILED!")
        sys.exit(1)
"""
Test script to verify the memory cleanup fix in AdvancedMemoryPool class.
This test ensures that temporary files are properly closed and cleaned up on Windows systems.
"""

import os
import tempfile
import sys
import gc
from advanced_memory_management_vl import AdvancedMemoryPool, MemoryPoolType

def test_memory_pool_cleanup():
    """Test that AdvancedMemoryPool properly cleans up temporary files on Windows"""
    print("Testing AdvancedMemoryPool cleanup...")
    
    # Create a memory pool
    pool = AdvancedMemoryPool(initial_size=1024*1024)  # 1MB for testing
    
    # Check if temp_file attribute exists (indicating Windows usage)
    has_temp_file = hasattr(pool, 'temp_file')
    print(f"Has temp_file attribute: {has_temp_file}")
    
    if has_temp_file:
        temp_filename = getattr(pool.temp_file, 'name', None)
        print(f"Temporary file name: {temp_filename}")
        
        # Verify the temp file exists before cleanup
        if temp_filename and os.path.exists(temp_filename):
            print(f"Temporary file exists before cleanup: {temp_filename}")
        else:
            print("Temporary file does not exist before cleanup")
    
    # Perform some allocations to ensure the pool is being used
    ptr1, size1 = pool.allocate(1024, MemoryPoolType.TENSOR_DATA)  # 1KB
    ptr2, size2 = pool.allocate(2048, MemoryPoolType.ACTIVATION_BUFFER)  # 2KB
    
    print(f"Allocated two blocks: {size1} bytes at {hex(ptr1)}, {size2} bytes at {hex(ptr2)}")
    
    # Deallocate
    pool.deallocate(ptr1)
    pool.deallocate(ptr2)
    
    # Now cleanup
    pool.cleanup()
    print("Cleanup completed")
    
    # Force garbage collection to ensure file handles are released
    gc.collect()
    
    if has_temp_file and temp_filename:
        # Check if the temp file still exists after cleanup
        if os.path.exists(temp_filename):
            print(f"ERROR: Temporary file still exists after cleanup: {temp_filename}")
            return False
        else:
            print(f"SUCCESS: Temporary file properly cleaned up: {temp_filename}")
            return True
    else:
        print("SUCCESS: No temporary file to clean up (Unix/Linux system)")
        return True

def test_multiple_pools_cleanup():
    """Test cleanup with multiple memory pools"""
    print("\nTesting multiple memory pools cleanup...")
    
    pools = []
    temp_files = []
    
    # Create multiple pools
    for i in range(3):
        pool = AdvancedMemoryPool(initial_size=512*1024)  # 512KB each
        pools.append(pool)
        
        if hasattr(pool, 'temp_file'):
            temp_filename = getattr(pool.temp_file, 'name', None)
            if temp_filename:
                temp_files.append(temp_filename)
                print(f"Pool {i} temp file: {temp_filename}")
    
    # Perform allocations
    allocations = []
    for i, pool in enumerate(pools):
        ptr, size = pool.allocate(512 + i*256)  # Different sizes
        allocations.append((pool, ptr))
        print(f"Allocated {size} bytes in pool {i}")
    
    # Deallocate
    for pool, ptr in allocations:
        pool.deallocate(ptr)
    
    # Cleanup all pools
    for i, pool in enumerate(pools):
        pool.cleanup()
        print(f"Pool {i} cleaned up")
    
    # Force garbage collection
    gc.collect()
    
    # Check if any temp files still exist
    remaining_files = [f for f in temp_files if os.path.exists(f)]
    if remaining_files:
        print(f"ERROR: Some temporary files still exist: {remaining_files}")
        return False
    else:
        print("SUCCESS: All temporary files properly cleaned up")
        return True

def test_cleanup_idempotency():
    """Test that calling cleanup multiple times doesn't cause errors"""
    print("\nTesting cleanup idempotency...")
    
    pool = AdvancedMemoryPool(initial_size=256*1024)  # 256KB
    
    # Call cleanup multiple times
    try:
        pool.cleanup()
        pool.cleanup()  # Should not raise an exception
        pool.cleanup()  # Should not raise an exception
        print("SUCCESS: Multiple cleanup calls handled gracefully")
        return True
    except Exception as e:
        print(f"ERROR: Exception during multiple cleanup calls: {e}")
        return False

if __name__ == "__main__":
    print("Testing AdvancedMemoryPool cleanup functionality\n")
    
    success = True
    
    success &= test_memory_pool_cleanup()
    success &= test_multiple_pools_cleanup()
    success &= test_cleanup_idempotency()
    
    if success:
        print("\nAll tests PASSED! Memory cleanup is working correctly.")
        sys.exit(0)
    else:
        print("\nSome tests FAILED! Memory cleanup needs attention.")
        sys.exit(1)
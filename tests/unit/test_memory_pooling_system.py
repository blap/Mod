#!/usr/bin/env python3
"""
Simple test for the Advanced Memory Pooling System
"""

from advanced_memory_pooling_system import AdvancedMemoryPoolingSystem, TensorType

def test_memory_pooling_system():
    """Test the complete memory pooling system"""
    print("Creating Advanced Memory Pooling System...")
    system = AdvancedMemoryPoolingSystem()
    
    print("Allocating a block...")
    block = system.allocate(TensorType.KV_CACHE, 1024*1024, 'test_tensor')  # 1MB
    print(f"Allocation successful: {block is not None}")
    
    if block:
        print(f"Allocated block size: {block.size} bytes")
        print(f"Allocated block address: {block.start_addr}")
        
        print("Deallocating the block...")
        success = system.deallocate(TensorType.KV_CACHE, 'test_tensor')
        print(f"Deallocation successful: {success}")
    
    print("Getting system stats...")
    stats = system.get_system_stats()
    print(f"Overall utilization: {stats.get('overall_utilization', 0):.2%}")
    print(f"Average fragmentation: {stats.get('average_fragmentation', 0):.2%}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_memory_pooling_system()
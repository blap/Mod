"""
Optimized Locking Strategies for Memory Management Systems

This module implements advanced locking strategies for memory management systems:
- Reader-Writer locks for read-heavy operations
- Lock striping for granular locking
- Thread-safe memory operations with reduced contention
"""

import threading
import time
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import math


class ReaderWriterLock:
    """
    A reader-writer lock implementation that allows multiple readers
    but exclusive writers, with configurable fairness policies.
    """
    def __init__(self, write_priority: bool = False):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
        self._writers = 0
        self._write_priority = write_priority
        self._waiting_writers = 0

    def acquire_read(self):
        """Acquire read lock"""
        with self._read_ready:
            while self._writers > 0 or (self._write_priority and self._waiting_writers > 0):
                self._read_ready.wait()
            self._readers += 1

    def release_read(self):
        """Release read lock"""
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        """Acquire write lock"""
        with self._read_ready:
            self._waiting_writers += 1
            while self._readers > 0 or self._writers > 0:
                self._read_ready.wait()
            self._waiting_writers -= 1
            self._writers += 1

    def release_write(self):
        """Release write lock"""
        with self._read_ready:
            self._writers -= 1
            if self._write_priority:
                self._read_ready.notify_all()  # Wake up all (including writers)
            else:
                self._read_ready.notify_all()  # Wake up all waiters

    def __enter__(self):
        """Context manager entry for write lock"""
        self.acquire_write()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release_write()


class LockStriping:
    """
    Implements lock striping - using multiple locks for different segments
    of a data structure to reduce contention.
    """
    def __init__(self, num_stripes: int = 16):
        if num_stripes <= 0:
            raise ValueError("num_stripes must be positive")
        self._stripes = [threading.RLock() for _ in range(num_stripes)]
        self._num_stripes = num_stripes

    def get_lock(self, key: Any) -> threading.RLock:
        """Get the appropriate lock for a given key"""
        if isinstance(key, int):
            hash_val = key
        else:
            hash_val = hash(key) if key is not None else 0
        return self._stripes[hash_val % self._num_stripes]

    def get_lock_for_index(self, index: int) -> threading.RLock:
        """Get the appropriate lock for a given index"""
        return self._stripes[index % self._num_stripes]

    def __len__(self):
        return self._num_stripes


class OptimizedMemoryBlock:
    """Thread-safe memory block with optimized locking"""
    def __init__(self, ptr: int, size: int, is_free: bool = True, 
                 ref_count: int = 0, timestamp: float = 0.0):
        self.ptr = ptr
        self.size = size
        self.is_free = is_free
        self.ref_count = ref_count
        self.timestamp = timestamp if timestamp > 0 else time.time()
        self._lock = threading.Lock()  # Fine-grained lock for this block

    def acquire_lock(self):
        """Acquire the lock for this block"""
        self._lock.acquire()

    def release_lock(self):
        """Release the lock for this block"""
        self._lock.release()

    def __enter__(self):
        self.acquire_lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_lock()


class ConcurrentMemoryMap:
    """
    Thread-safe memory map with lock striping for improved performance
    """
    def __init__(self, num_stripes: int = 16):
        self._data: Dict[int, OptimizedMemoryBlock] = {}
        self._stripes = LockStriping(num_stripes)
        self._global_lock = threading.Lock()  # For structural changes

    def get(self, key: int) -> Optional['OptimizedMemoryBlock']:
        """Thread-safe get operation"""
        with self._stripes.get_lock(key):
            return self._data.get(key)

    def set(self, key: int, value: OptimizedMemoryBlock) -> None:
        """Thread-safe set operation"""
        with self._stripes.get_lock(key):
            self._data[key] = value

    def delete(self, key: int) -> bool:
        """Thread-safe delete operation"""
        with self._stripes.get_lock(key):
            if key in self._data:
                del self._data[key]
                return True
            return False

    def keys(self) -> List[int]:
        """Thread-safe keys operation - requires global lock"""
        with self._global_lock:
            return list(self._data.keys())

    def values(self) -> List[OptimizedMemoryBlock]:
        """Thread-safe values operation - requires global lock"""
        with self._global_lock:
            return list(self._data.values())

    def items(self) -> List[tuple]:
        """Thread-safe items operation - requires global lock"""
        with self._global_lock:
            return list(self._data.items())

    def __len__(self) -> int:
        """Thread-safe length operation - requires global lock"""
        with self._global_lock:
            return len(self._data)

    def __contains__(self, key: int) -> bool:
        """Thread-safe contains operation"""
        with self._stripes.get_lock(key):
            return key in self._data


class ConcurrentFreeBlockSet:
    """
    Thread-safe set for free blocks with lock striping
    """
    def __init__(self, num_stripes: int = 16):
        self._sets: Dict[int, Set[OptimizedMemoryBlock]] = defaultdict(set)
        self._stripes = LockStriping(num_stripes)
        self._global_lock = threading.RLock()  # For structural changes

    def add_to_level(self, level: int, block: OptimizedMemoryBlock) -> None:
        """Add a block to a specific level"""
        with self._stripes.get_lock_for_index(level):
            self._sets[level].add(block)

    def remove_from_level(self, level: int, block: OptimizedMemoryBlock) -> bool:
        """Remove a block from a specific level"""
        with self._stripes.get_lock_for_index(level):
            if block in self._sets[level]:
                self._sets[level].remove(block)
                return True
            return False

    def get_free_blocks_for_level(self, level: int) -> set:
        """Get all free blocks for a specific level"""
        with self._stripes.get_lock_for_index(level):
            return self._sets[level].copy()

    def pop_free_block_from_level(self, level: int) -> Optional[OptimizedMemoryBlock]:
        """Pop a free block from a specific level"""
        with self._stripes.get_lock_for_index(level):
            block_set = self._sets[level]
            if block_set:
                # Pop an arbitrary element from the set
                block: OptimizedMemoryBlock = block_set.pop()
                return block
            return None

    def is_level_empty(self, level: int) -> bool:
        """Check if a level is empty"""
        with self._stripes.get_lock_for_index(level):
            return len(self._sets[level]) == 0

    def get_level_count(self, level: int) -> int:
        """Get count of blocks in a level"""
        with self._stripes.get_lock_for_index(level):
            return len(self._sets[level])

    def all_levels(self) -> Dict[int, set]:
        """Get all levels - requires global lock"""
        with self._global_lock:
            return {level: blocks.copy() for level, blocks in self._sets.items()}


if __name__ == "__main__":
    # Test the optimized locking strategies
    print("Testing Optimized Locking Strategies")
    
    # Test ReaderWriterLock
    rw_lock = ReaderWriterLock()
    
    def reader_task(name: str):
        print(f"Reader {name} attempting to acquire read lock")
        rw_lock.acquire_read()
        print(f"Reader {name} got read lock")
        time.sleep(0.1)
        rw_lock.release_read()
        print(f"Reader {name} released read lock")
    
    def writer_task(name: str):
        print(f"Writer {name} attempting to acquire write lock")
        rw_lock.acquire_write()
        print(f"Writer {name} got write lock")
        time.sleep(0.2)
        rw_lock.release_write()
        print(f"Writer {name} released write lock")
    
    # Test concurrent readers
    import threading
    threads = []
    
    # Start multiple readers
    for i in range(3):
        t = threading.Thread(target=reader_task, args=(f"R{i}",))
        threads.append(t)
        t.start()
    
    # Start a writer (should wait for readers to finish)
    t = threading.Thread(target=writer_task, args=("W1",))
    threads.append(t)
    t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    print("\nTesting ConcurrentMemoryMap")
    mem_map = ConcurrentMemoryMap(num_stripes=4)
    
    # Add some blocks
    block1 = OptimizedMemoryBlock(0x1000, 1024, is_free=True)
    block2 = OptimizedMemoryBlock(0x2000, 2048, is_free=True)
    
    mem_map.set(0x1000, block1)
    mem_map.set(0x2000, block2)
    
    print(f"Map size: {len(mem_map)}")
    print(f"Block at 0x1000: {mem_map.get(0x1000).size if mem_map.get(0x1000) else None}")
    
    print("\nTesting ConcurrentFreeBlockSet")
    free_set = ConcurrentFreeBlockSet(num_stripes=4)
    
    # Add blocks to different levels
    free_set.add_to_level(0, block1)
    free_set.add_to_level(1, block2)
    
    print(f"Level 0 count: {free_set.get_level_count(0)}")
    print(f"Level 1 count: {free_set.get_level_count(1)}")
    
    popped = free_set.pop_free_block_from_level(0)
    print(f"Popped block size: {popped.size if popped else None}")
    print(f"Level 0 count after pop: {free_set.get_level_count(0)}")
    
    print("All tests completed successfully!")
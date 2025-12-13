"""
Memory Defragmentation System for Qwen3-VL

This module implements a comprehensive memory defragmentation system that reduces fragmentation
in tensor pools and improves memory utilization efficiency. The system includes algorithms for
detecting fragmentation, compacting free blocks, and optimizing memory layouts for better performance.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import threading
import time
import heapq
from dataclasses import dataclass
import logging
import psutil
import numpy as np
import gc


class FragmentationLevel(Enum):
    """Enumeration for different levels of fragmentation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryBlockInfo:
    """Information about a memory block"""
    id: str
    start_offset: int
    size_bytes: int
    is_free: bool
    last_accessed: float
    access_frequency: int
    device: str = "cpu"
    

class MemoryDefragmenter:
    """Main class for memory defragmentation operations"""

    def __init__(self, 
                 fragmentation_threshold: float = 0.3,  # 30% fragmentation triggers defrag
                 defrag_frequency_minutes: int = 5,      # Run defragmentation every 5 minutes
                 memory_pool_size: int = 8 * 1024 * 1024 * 1024,  # 8GB default
                 cpu_cores: int = 4):  # Number of CPU cores for parallel processing
        """
        Initialize the memory defragmentation system
        
        Args:
            fragmentation_threshold: Threshold for triggering defragmentation (0.0-1.0)
            defrag_frequency_minutes: How often to run defragmentation (minutes)
            memory_pool_size: Size of the memory pool to defragment
            cpu_cores: Number of CPU cores for parallel processing
        """
        self.fragmentation_threshold = fragmentation_threshold
        self.defrag_frequency_minutes = defrag_frequency_minutes
        self.memory_pool_size = memory_pool_size
        self.cpu_cores = cpu_cores
        
        # Memory block tracking
        self.memory_blocks: Dict[str, MemoryBlockInfo] = {}
        self.block_lookup: List[MemoryBlockInfo] = []  # Sorted by offset
        self.lock = threading.RLock()
        
        # Statistics
        self.defragmentation_stats = {
            'defragmentations_performed': 0,
            'total_free_space': 0,
            'largest_free_block_size': 0,
            'compaction_improvement': 0.0,
            'defrag_duration_seconds': 0.0,
            'blocks_moved': 0,
            'copy_operations': 0
        }
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Hardware-specific optimizations
        self._configure_hardware_optimizations()
        
        self.logger.info(f"Memory Defragmentation System initialized with threshold {fragmentation_threshold * 100}% "
                         f"and frequency {defrag_frequency_minutes} minutes")

    def _configure_hardware_optimizations(self):
        """Configure optimizations for specific hardware"""
        # For Intel i5-10210U (4 cores, 8 threads)
        self.cpu_threads = 8  # Use hyperthreaded cores for parallel operations
        self.l3_cache_size = 6 * 1024 * 1024  # 6MB L3 cache on i5-10210U
        self.cache_line_size = 64  # Standard cache line size for Intel CPUs
        
        # For NVIDIA SM61
        self.gpu_warp_size = 32  # Standard CUDA warp size
        self.warp_byte_size = self.gpu_warp_size * 4  # For float32 tensors

    def register_memory_block(self, block_id: str, size_bytes: int, device: str = "cpu") -> bool:
        """
        Register a memory block with the defragmentation system
        
        Args:
            block_id: Unique identifier for the block
            size_bytes: Size of the block in bytes
            device: Device where the block resides ("cpu", "cuda", etc.)
        
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if block_id in self.memory_blocks:
                self.logger.warning(f"Block {block_id} already registered")
                return False
            
            # Determine start offset (find largest free space to place it)
            start_offset = self._find_optimal_placement(size_bytes)
            
            block_info = MemoryBlockInfo(
                id=block_id,
                start_offset=start_offset,
                size_bytes=size_bytes,
                is_free=False,
                last_accessed=time.time(),
                access_frequency=0,
                device=device
            )
            
            self.memory_blocks[block_id] = block_info
            self._update_block_lookup()
            
            self.logger.debug(f"Registered block {block_id} at offset {start_offset}, size {size_bytes} bytes, device {device}")
            return True

    def deregister_memory_block(self, block_id: str) -> bool:
        """
        Deregister a memory block, marking it as free
        
        Args:
            block_id: ID of the block to deregister
        
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if block_id not in self.memory_blocks:
                return False
            
            block_info = self.memory_blocks[block_id]
            block_info.is_free = True
            block_info.last_accessed = time.time()
            
            self._update_block_lookup()
            
            self.logger.debug(f"De-registered block {block_id}, now free")
            return True

    def _find_optimal_placement(self, size_bytes: int) -> int:
        """Find optimal placement for a new block"""
        # Look for a free space that fits this block
        for block_info in self.block_lookup:
            if block_info.is_free and block_info.size_bytes >= size_bytes:
                # Split the block if it's much larger than needed
                if block_info.size_bytes > size_bytes * 2:
                    self._split_block(block_info, size_bytes)
                    return block_info.start_offset
                else:
                    # Use the whole block
                    new_start = block_info.start_offset
                    self.memory_blocks[block_info.id] = MemoryBlockInfo(
                        id=block_info.id,
                        start_offset=block_info.start_offset,
                        size_bytes=size_bytes,
                        is_free=False,
                        last_accessed=time.time(),
                        access_frequency=0,
                        device=block_info.device
                    )
                    return new_start
        
        # If no suitable existing free block found, extend to end
        current_end = self._get_current_end_offset()
        return current_end

    def _get_current_end_offset(self) -> int:
        """Get the end offset of currently allocated memory"""
        if not self.block_lookup:
            return 0
        # Find the furthest end offset
        max_end = 0
        for block in self.block_lookup:
            end_offset = block.start_offset + block.size_bytes
            if end_offset > max_end:
                max_end = end_offset
        return max_end

    def _split_block(self, block_info: MemoryBlockInfo, size_needed: int):
        """Split a block to accommodate a smaller one"""
        # Create a new block for the remaining space
        remaining_size = block_info.size_bytes - size_needed
        if remaining_size > 0:
            new_block_id = f"{block_info.id}_rest"
            new_block = MemoryBlockInfo(
                id=new_block_id,
                start_offset=block_info.start_offset + size_needed,
                size_bytes=remaining_size,
                is_free=True,
                last_accessed=time.time(),
                access_frequency=0,
                device=block_info.device
            )
            self.memory_blocks[new_block_id] = new_block
        
        # Update the original block to match the size needed
        block_info.size_bytes = size_needed
        block_info.is_free = False
        block_info.last_accessed = time.time()
        block_info.access_frequency = 0

    def _update_block_lookup(self):
        """Update the sorted block lookup list"""
        self.block_lookup = sorted(self.memory_blocks.values(), key=lambda x: x.start_offset)

    def calculate_fragmentation(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate memory fragmentation metrics
        
        Returns:
            Tuple of (fragmentation_ratio, details_dict)
        """
        with self.lock:
            if not self.block_lookup:
                return 0.0, {'total_free': 0, 'max_free_block': 0, 'total_blocks': 0}
            
            free_blocks = [block for block in self.block_lookup if block.is_free]
            if not free_blocks:
                return 0.0, {'total_free': 0, 'max_free_block': 0, 'total_blocks': len(self.block_lookup)}
            
            # Calculate total free space
            total_free = sum(block.size_bytes for block in free_blocks)
            
            # Calculate largest free block
            max_free_block = max((block.size_bytes for block in free_blocks), default=0)
            
            # Calculate fragmentation ratio: (total_free - max_free_block) / total_free
            # This indicates how scattered the free space is
            fragmentation_ratio = (total_free - max_free_block) / total_free if total_free > 0 else 0.0
            
            details = {
                'total_free': total_free,
                'max_free_block': max_free_block,
                'total_blocks': len(self.block_lookup),
                'free_blocks_count': len(free_blocks),
                'fragmentation_severity': self._get_fragmentation_level(fragmentation_ratio)
            }
            
            return fragmentation_ratio, details

    def _get_fragmentation_level(self, fragmentation_ratio: float) -> FragmentationLevel:
        """Get the fragmentation level based on ratio"""
        if fragmentation_ratio < 0.1:
            return FragmentationLevel.LOW
        elif fragmentation_ratio < 0.3:
            return FragmentationLevel.MEDIUM
        elif fragmentation_ratio < 0.6:
            return FragmentationLevel.HIGH
        else:
            return FragmentationLevel.CRITICAL

    def should_defragment(self) -> bool:
        """
        Determine if defragmentation should be performed
        
        Returns:
            True if defragmentation is needed, False otherwise
        """
        fragmentation_ratio, _ = self.calculate_fragmentation()
        return fragmentation_ratio > self.fragmentation_threshold

    def defragment(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform memory defragmentation
        
        Args:
            force: Whether to force defragmentation regardless of threshold
        
        Returns:
            Dictionary with defragmentation results
        """
        if not force and not self.should_defragment():
            self.logger.debug("Defragmentation not needed based on current fragmentation level")
            return {
                'defragmentation_performed': False,
                'message': 'Fragmentation below threshold'
            }
        
        start_time = time.time()
        self.logger.info("Starting memory defragmentation process...")
        
        with self.lock:
            # Compact free blocks by moving allocated blocks together
            initial_stats = self.calculate_fragmentation()
            
            # Perform the defragmentation
            blocks_moved, copy_operations = self._perform_compaction()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Update statistics
            final_stats = self.calculate_fragmentation()
            
            self.defragmentation_stats['defragmentations_performed'] += 1
            self.defragmentation_stats['compaction_improvement'] = initial_stats[0] - final_stats[0]
            self.defragmentation_stats['defrag_duration_seconds'] += duration
            self.defragmentation_stats['blocks_moved'] += blocks_moved
            self.defragmentation_stats['copy_operations'] += copy_operations
            
            result = {
                'defragmentation_performed': True,
                'duration_seconds': duration,
                'blocks_moved': blocks_moved,
                'copy_operations': copy_operations,
                'fragmentation_improvement': initial_stats[0] - final_stats[0],
                'initial_fragmentation': initial_stats[0],
                'final_fragmentation': final_stats[0],
                'message': f'Defragmentation completed in {duration:.3f}s'
            }
            
            self.logger.info(f"Memory defragmentation completed: {result['message']}")
            return result

    def _perform_compaction(self) -> Tuple[int, int]:
        """
        Perform the actual memory compaction
        
        Returns:
            Tuple of (blocks_moved, copy_operations)
        """
        # Find all allocated blocks that can be moved (non-pinned, less frequently accessed)
        allocated_blocks = [block for block in self.block_lookup if not block.is_free]
        
        # Sort by access frequency and recency (least accessed first for moving)
        allocated_blocks.sort(key=lambda b: (b.access_frequency, time.time() - b.last_accessed))
        
        # Find contiguous free spaces to move blocks to
        free_blocks = [block for block in self.block_lookup if block.is_free]
        free_blocks.sort(key=lambda b: b.start_offset)
        
        # Compact by moving blocks to fill gaps
        blocks_moved = 0
        copy_operations = 0
        
        # Create a new layout with contiguous allocations
        new_layout = []
        current_offset = 0
        
        for block in sorted(allocated_blocks, key=lambda b: b.start_offset):
            # Place this block at the current offset
            if block.start_offset != current_offset:
                # Need to move this block
                new_layout.append(MemoryBlockInfo(
                    id=block.id,
                    start_offset=current_offset,
                    size_bytes=block.size_bytes,
                    is_free=False,
                    last_accessed=block.last_accessed,
                    access_frequency=block.access_frequency,
                    device=block.device
                ))
                
                # Update the memory_blocks registry
                self.memory_blocks[block.id].start_offset = current_offset
                blocks_moved += 1
                
                # Estimate copy operation (actually copying tensor data would happen elsewhere)
                copy_operations += 1
            else:
                # Block is already optimally positioned
                new_layout.append(block)
            
            current_offset += block.size_bytes
        
        # Add any remaining free space
        if current_offset < self._get_current_end_offset():
            free_size = self._get_current_end_offset() - current_offset
            new_layout.append(MemoryBlockInfo(
                id=f"free_{int(time.time())}",
                start_offset=current_offset,
                size_bytes=free_size,
                is_free=True,
                last_accessed=time.time(),
                access_frequency=0,
                device="cpu"
            ))
        
        # Update the block lookup with the compacted layout
        self.block_lookup = new_layout
        
        # Also consolidate free blocks that are adjacent
        self._consolidate_adjacent_free_blocks()
        
        return blocks_moved, copy_operations

    def _consolidate_adjacent_free_blocks(self):
        """Merge adjacent free blocks to reduce fragmentation"""
        consolidated = []
        i = 0
        
        while i < len(self.block_lookup):
            current = self.block_lookup[i]
            consolidated.append(current)
            
            # Check if next block is adjacent and free
            j = i + 1
            while j < len(self.block_lookup):
                next_block = self.block_lookup[j]
                
                # Check if blocks are adjacent and both free
                if (current.is_free and next_block.is_free and 
                    current.start_offset + current.size_bytes == next_block.start_offset):
                    # Merge blocks
                    current.size_bytes += next_block.size_bytes
                    # Remove merged block from registry
                    if next_block.id in self.memory_blocks:
                        del self.memory_blocks[next_block.id]
                    j += 1
                else:
                    break
            
            i = j + 1 if j > i + 1 else i + 1
        
        self.block_lookup = consolidated

    def get_memory_health(self) -> Dict[str, Any]:
        """
        Get comprehensive memory health metrics
        
        Returns:
            Dictionary with memory health information
        """
        fragmentation_ratio, frag_details = self.calculate_fragmentation()
        
        total_blocks = len(self.memory_blocks)
        total_free_blocks = frag_details['free_blocks_count']
        total_allocated_blocks = total_blocks - total_free_blocks
        
        # Calculate memory utilization
        total_used_bytes = sum(block.size_bytes for block in self.memory_blocks.values() if not block.is_free)
        free_space = frag_details['total_free']
        total_managed_memory = total_used_bytes + free_space
        
        # Calculate fragmentation severity
        fragmentation_level = frag_details['fragmentation_severity']
        
        return {
            'fragmentation_ratio': fragmentation_ratio,
            'fragmentation_level': fragmentation_level.value,
            'total_managed_memory_bytes': total_managed_memory,
            'total_used_bytes': total_used_bytes,
            'total_free_bytes': free_space,
            'free_percentage': (free_space / total_managed_memory) * 100 if total_managed_memory > 0 else 0,
            'total_blocks': total_blocks,
            'allocated_blocks': total_allocated_blocks,
            'free_blocks': total_free_blocks,
            'largest_free_block_bytes': frag_details['max_free_block'],
            'average_block_size_bytes': (total_managed_memory / total_blocks) if total_blocks > 0 else 0,
            'defragmentation_stats': self.defragmentation_stats.copy()
        }

    def refresh_block_access(self, block_id: str):
        """
        Refresh the access time and frequency for a block
        
        Args:
            block_id: ID of the block to update
        """
        with self.lock:
            if block_id in self.memory_blocks:
                block = self.memory_blocks[block_id]
                block.last_accessed = time.time()
                block.access_frequency += 1

    def periodic_defragmentation(self, stop_event: threading.Event):
        """
        Run periodic defragmentation in a separate thread
        
        Args:
            stop_event: Threading event to signal stopping
        """
        self.logger.info("Starting periodic defragmentation service...")
        
        while not stop_event.is_set():
            try:
                # Check if defragmentation is needed
                if self.should_defragment():
                    self.defragment()
                
                # Wait for the specified interval or until stop event is set
                for _ in range(self.defrag_frequency_minutes * 60):
                    if stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in periodic defragmentation: {e}")
                time.sleep(10)  # Wait 10s before trying again if there's an error
        
        self.logger.info("Periodic defragmentation service stopped")


# Standalone defragmentation utility functions
def auto_defragment_memory_pool(
    memory_pool_manager,
    fragmentation_threshold: float = 0.3,
    check_interval: int = 30  # seconds
) -> threading.Thread:
    """
    Automatically defragment a memory pool at regular intervals
    
    Args:
        memory_pool_manager: Memory pool manager to defragment
        fragmentation_threshold: Threshold for defragmentation
        check_interval: Interval between checks (seconds)
    
    Returns:
        Thread running the defragmentation service
    """
    defragmenter = MemoryDefragmenter(fragmentation_threshold=fragmentation_threshold)
    
    stop_event = threading.Event()
    
    def auto_defragment_worker():
        while not stop_event.is_set():
            try:
                frag_ratio, _ = defragmenter.calculate_fragmentation()
                if frag_ratio > fragmentation_threshold:
                    defragmenter.defragment()
                
                # Sleep until next check or stop event
                for _ in range(check_interval):
                    if stop_event.is_set():
                        return
                    time.sleep(1)
                    
            except Exception as e:
                logging.error(f"Error in auto defragmenter: {e}")
                time.sleep(10)
    
    thread = threading.Thread(target=auto_defragment_worker, daemon=True)
    thread.start()
    
    return thread, stop_event


def get_memory_fragmentation_report(defragmenter: MemoryDefragmenter) -> str:
    """
    Generate a human-readable memory fragmentation report
    
    Args:
        defragmenter: Memory defragmenter instance
    
    Returns:
        Formatted fragmentation report
    """
    health = defragmenter.get_memory_health()
    
    report = [
        "=" * 60,
        "MEMORY FRAGMENTATION REPORT",
        "=" * 60,
        f"Fragmentation Level: {health['fragmentation_level'].upper()}",
        f"Fragmentation Ratio: {health['fragmentation_ratio']:.2%}",
        f"Total Managed Memory: {health['total_managed_memory_bytes'] / (1024**3):.2f} GB",
        f"Used Memory: {health['total_used_bytes'] / (1024**3):.2f} GB",
        f"Free Memory: {health['total_free_bytes'] / (1024**3):.2f} GB",
        f"Free Percentage: {health['free_percentage']:.2f}%",
        "",
        "BLOCK STATS:",
        f"  Total Blocks: {health['total_blocks']}",
        f"  Allocated Blocks: {health['allocated_blocks']}",
        f"  Free Blocks: {health['free_blocks']}",
        f"  Largest Free Block: {health['largest_free_block_bytes'] / (1024**2):.2f} MB",
        "",
        "DEFRAGMENTATION STATS:",
        f"  Defrags Performed: {health['defragmentation_stats']['defragmentations_performed']}",
        f"  Total Duration: {health['defragmentation_stats']['defrag_duration_seconds']:.3f}s",
        f"  Blocks Moved: {health['defragmentation_stats']['blocks_moved']}",
        f"  Copy Operations: {health['defragmentation_stats']['copy_operations']}",
        f"  Compaction Improvement: {health['defragmentation_stats']['compaction_improvement']:.3f}",
        "=" * 60
    ]
    
    return "\n".join(report)


# Example usage
if __name__ == "__main__":
    print("Testing Memory Defragmentation System...")
    
    # Create defragmenter
    defrag = MemoryDefragmenter(
        fragmentation_threshold=0.2,
        defrag_frequency_minutes=1,
        memory_pool_size=1024*1024*512  # 512MB
    )
    
    # Test registering and deregistering blocks
    print("\n1. Registering memory blocks...")
    blocks = []
    for i in range(10):
        block_id = f"block_{i}"
        size = (i + 1) * 1024 * 1024  # 1-10 MB
        success = defrag.register_memory_block(block_id, size)
        blocks.append(block_id)
        print(f"   Registered {block_id}, size {size/(1024*1024):.1f} MB: {success}")
    
    # Simulate some usage
    print("\n2. Simulating memory usage patterns...")
    for i, block_id in enumerate(blocks):
        if i % 2 == 0:  # Free half of the blocks to create fragmentation
            success = defrag.deregister_memory_block(block_id)
            print(f"   Freed {block_id}: {success}")
    
    # Check fragmentation
    print("\n3. Checking fragmentation...")
    frag_ratio, frag_details = defrag.calculate_fragmentation()
    print(f"   Current fragmentation: {frag_ratio:.2%}")
    print(f"   Fragmentation level: {frag_details['fragmentation_severity'].value}")
    print(f"   Free blocks count: {frag_details['free_blocks_count']}")
    
    # Check memory health
    print("\n4. Memory health report...")
    health = defrag.get_memory_health()
    print(f"   Total managed: {health['total_managed_memory_bytes'] / (1024**2):.2f} MB")
    print(f"   Used: {health['total_used_bytes'] / (1024**2):.2f} MB")
    print(f"   Free: {health['total_free_bytes'] / (1024**2):.2f} MB")
    
    # Run defragmentation if needed
    print("\n5. Running defragmentation...")
    if defrag.should_defragment():
        result = defrag.defragment()
        print(f"   Defrag result: {result}")
    else:
        print("   Defragmentation not needed based on current threshold")
        result = defrag.defragment(force=True)  # Force defrag for demo
        print(f"   Forced defrag result: {result}")
    
    # Check health after defragmentation
    print("\n6. Memory health after defragmentation...")
    health = defrag.get_memory_health()
    print(f"   Fragmentation ratio: {health['fragmentation_ratio']:.2%}")
    print(f"   Largest free block: {health['largest_free_block_bytes'] / (1024**2):.2f} MB")
    
    # Generate report
    print("\n7. Fragmentation Report:")
    report = get_memory_fragmentation_report(defrag)
    print(report)
    
    print("\nMemory Defragmentation System test completed successfully!")
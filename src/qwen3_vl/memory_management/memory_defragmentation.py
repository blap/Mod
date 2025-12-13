"""
Memory Defragmentation System for Qwen3-VL Model

This module implements memory defragmentation routines optimized for Intel i5-10210U + NVIDIA SM61 + NVMe SSD.
The system includes basic defragmentation, vision-specific optimizations, and adaptive defragmentation strategies.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import threading
import time
import logging
import gc
from enum import Enum
import math


class DefragmentationStrategy(Enum):
    """Different defragmentation strategies for different scenarios"""
    BASIC = "basic"
    VISION_OPTIMIZED = "vision_optimized"
    ADAPTIVE = "adaptive"


class MemoryBlock:
    """Represents a memory block that can be defragmented"""
    def __init__(self, addr: int, size: int, is_free: bool, tensor_type: str = "general", last_access_time: float = 0.0):
        self.addr = addr
        self.size = size
        self.is_free = is_free
        self.tensor_type = tensor_type
        self.last_access_time = last_access_time
        self.ref_count = 0
        self.alignment = 64  # Default cache line alignment


class MemoryDefragmenter:
    """Basic memory defragmentation system"""

    def __init__(self, strategy: DefragmentationStrategy = DefragmentationStrategy.BASIC):
        self.strategy = strategy
        self.blocks: List[MemoryBlock] = []
        self.free_blocks: List[MemoryBlock] = []
        self.lock = threading.RLock()
        self.defrag_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'total_time_spent': 0.0,
            'total_memory_moved': 0
        }
        self.logger = logging.getLogger(__name__)

    def add_block(self, addr: int, size: int, is_free: bool, tensor_type: str = "general"):
        """Add a memory block to the defragmenter's tracking"""
        with self.lock:
            block = MemoryBlock(addr, size, is_free, tensor_type, time.time())
            self.blocks.append(block)
            if is_free:
                self.free_blocks.append(block)

    def remove_block(self, addr: int):
        """Remove a memory block from tracking"""
        with self.lock:
            self.blocks = [b for b in self.blocks if b.addr != addr]
            self.free_blocks = [b for b in self.free_blocks if b.addr != addr]

    def calculate_fragmentation(self) -> float:
        """Calculate the current level of memory fragmentation"""
        with self.lock:
            if not self.blocks:
                return 0.0

            # Calculate total free memory
            total_free = sum(block.size for block in self.free_blocks)
            if total_free == 0:
                return 0.0

            # Find the largest contiguous free block
            sorted_free = sorted(self.free_blocks, key=lambda b: b.addr)
            if not sorted_free:
                return 0.0

            # Calculate largest contiguous block
            largest_contiguous = sorted_free[0].size
            current_size = sorted_free[0].size

            for i in range(1, len(sorted_free)):
                prev_block = sorted_free[i-1]
                curr_block = sorted_free[i]

                # Check if blocks are contiguous
                if prev_block.addr + prev_block.size == curr_block.addr:
                    current_size += curr_block.size
                else:
                    largest_contiguous = max(largest_contiguous, current_size)
                    current_size = curr_block.size

            largest_contiguous = max(largest_contiguous, current_size)

            # Fragmentation index: 1 - (largest_contiguous / total_free)
            fragmentation = 1.0 - (largest_contiguous / total_free)
            return min(1.0, fragmentation)  # Clamp between 0 and 1

    def defragment(self) -> bool:
        """Perform basic defragmentation operation"""
        start_time = time.time()
        success = False

        with self.lock:
            try:
                if self.strategy == DefragmentationStrategy.BASIC:
                    success = self._basic_defragment()
                elif self.strategy == DefragmentationStrategy.VISION_OPTIMIZED:
                    success = self._vision_optimized_defragment()
                elif self.strategy == DefragmentationStrategy.ADAPTIVE:
                    success = self._adaptive_defragment()
                else:
                    success = self._basic_defragment()

                # Update statistics
                self.defrag_stats['total_operations'] += 1
                if success:
                    self.defrag_stats['successful_operations'] += 1

                self.defrag_stats['total_time_spent'] += (time.time() - start_time)
            except Exception as e:
                self.logger.error(f"Error during defragmentation: {e}")

        return success

    def _basic_defragment(self) -> bool:
        """Basic defragmentation: merge adjacent free blocks"""
        # Sort free blocks by address
        sorted_free = sorted(self.free_blocks, key=lambda b: b.addr)

        # Merge adjacent free blocks
        merged_blocks = []
        i = 0
        while i < len(sorted_free):
            current = sorted_free[i]
            # Check if next block is adjacent
            while (i + 1 < len(sorted_free) and
                   current.addr + current.size == sorted_free[i + 1].addr):
                # Merge blocks
                next_block = sorted_free[i + 1]
                current.size += next_block.size
                i += 1
            merged_blocks.append(current)
            i += 1

        # Update free blocks list
        self.free_blocks = merged_blocks
        return True

    def _vision_optimized_defragment(self) -> bool:
        """Defragmentation optimized for vision processing patterns"""
        # In vision processing, we often have predictable access patterns
        # This defragmentation prioritizes keeping frequently accessed blocks together
        sorted_free = sorted(self.free_blocks, key=lambda b: b.addr)

        # Merge adjacent free blocks, but also consider temporal locality
        merged_blocks = []
        i = 0
        while i < len(sorted_free):
            current = sorted_free[i]
            # Check if next block is adjacent
            while (i + 1 < len(sorted_free) and
                   current.addr + current.size == sorted_free[i + 1].addr):
                # Merge blocks
                next_block = sorted_free[i + 1]
                current.size += next_block.size
                i += 1
            merged_blocks.append(current)
            i += 1

        # For vision models, prioritize larger contiguous blocks
        # Sort by size descending to make large allocations easier
        self.free_blocks = sorted(merged_blocks, key=lambda b: b.size, reverse=True)
        return True

    def _adaptive_defragment(self) -> bool:
        """Adaptive defragmentation based on access patterns"""
        # Calculate fragmentation level
        fragmentation = self.calculate_fragmentation()

        if fragmentation < 0.1:  # Low fragmentation
            # No need to defragment
            return True
        elif fragmentation < 0.3:  # Medium fragmentation
            # Light defragmentation
            return self._basic_defragment()
        else:  # High fragmentation
            # Aggressive defragmentation
            sorted_free = sorted(self.free_blocks, key=lambda b: b.addr)

            # Merge adjacent free blocks
            merged_blocks = []
            i = 0
            while i < len(sorted_free):
                current = sorted_free[i]
                # Check if next block is adjacent
                while (i + 1 < len(sorted_free) and
                       current.addr + current.size == sorted_free[i + 1].addr):
                    # Merge blocks
                    next_block = sorted_free[i + 1]
                    current.size += next_block.size
                    i += 1
                merged_blocks.append(current)
                i += 1

            # Sort by size descending for better allocation performance
            self.free_blocks = sorted(merged_blocks, key=lambda b: b.size, reverse=True)
            return True

    def get_defrag_stats(self) -> Dict[str, Any]:
        """Get defragmentation statistics"""
        with self.lock:
            stats = self.defrag_stats.copy()
            stats['fragmentation_level'] = self.calculate_fragmentation()
            stats['total_blocks'] = len(self.blocks)
            stats['free_blocks_count'] = len(self.free_blocks)
            return stats


class VisionOptimizedDefragmenter(MemoryDefragmenter):
    """Defragmenter optimized specifically for vision processing"""

    def __init__(self):
        super().__init__(DefragmentationStrategy.VISION_OPTIMIZED)
        self.access_pattern_history = deque(maxlen=1000)
        self.tensor_type_weights = {
            'kv_cache': 1.5,      # KV cache tensors are critical
            'image_features': 1.3,  # Image features accessed frequently
            'attention_weights': 1.2,  # Attention weights important
            'general': 1.0,
            'temporary': 0.8
        }

    def record_access(self, addr: int, tensor_type: str = "general"):
        """Record access to a memory block"""
        self.access_pattern_history.append({
            'addr': addr,
            'tensor_type': tensor_type,
            'timestamp': time.time()
        })

    def _vision_optimized_defragment(self) -> bool:
        """Enhanced defragmentation for vision models"""
        # Get access frequency for each tensor type
        type_counts = defaultdict(int)
        for access in self.access_pattern_history:
            type_counts[access['tensor_type']] += 1

        # Calculate weighted fragmentation considering tensor types
        sorted_free = sorted(self.free_blocks, key=lambda b: b.addr)

        # Merge adjacent free blocks
        merged_blocks = []
        i = 0
        while i < len(sorted_free):
            current = sorted_free[i]
            # Check if next block is adjacent
            while (i + 1 < len(sorted_free) and
                   current.addr + current.size == sorted_free[i + 1].addr):
                # Merge blocks
                next_block = sorted_free[i + 1]
                current.size += next_block.size
                i += 1
            merged_blocks.append(current)
            i += 1

        # Prioritize larger blocks for frequently accessed tensor types
        self.free_blocks = sorted(
            merged_blocks,
            key=lambda b: b.size * self.tensor_type_weights.get(b.tensor_type, 1.0),
            reverse=True
        )
        return True


class AdaptiveDefragmenter(MemoryDefragmenter):
    """Defragmenter that adapts its strategy based on system conditions"""

    def __init__(self):
        super().__init__(DefragmentationStrategy.ADAPTIVE)
        self.defrag_history = deque(maxlen=100)
        self.performance_threshold = 0.7  # Threshold for when to defragment
        self.defrag_frequency = 10  # How often to check for defragmentation

    def should_defragment(self) -> bool:
        """Determine if defragmentation should be performed"""
        fragmentation = self.calculate_fragmentation()
        
        # If fragmentation is above threshold, defragment
        if fragmentation > self.performance_threshold:
            return True

        # If we have allocation failures or slow allocation times, defragment
        recent_failures = sum(1 for record in self.defrag_history 
                             if record.get('allocation_failed', False))
        failure_rate = recent_failures / len(self.defrag_history) if self.defrag_history else 0

        if failure_rate > 0.1:  # More than 10% allocation failures
            return True

        return False

    def record_operation(self, operation_data: Dict[str, Any]):
        """Record an operation for adaptive learning"""
        self.defrag_history.append(operation_data)

    def _adaptive_defragment(self) -> bool:
        """Adaptive defragmentation based on system conditions"""
        # Calculate current system conditions
        fragmentation = self.calculate_fragmentation()
        
        # Adjust strategy based on fragmentation level and access patterns
        if fragmentation > 0.5:  # High fragmentation
            # Aggressive defragmentation
            sorted_free = sorted(self.free_blocks, key=lambda b: b.addr)
            
            # Merge all adjacent free blocks
            merged_blocks = []
            i = 0
            while i < len(sorted_free):
                current = sorted_free[i]
                while (i + 1 < len(sorted_free) and
                       current.addr + current.size == sorted_free[i + 1].addr):
                    next_block = sorted_free[i + 1]
                    current.size += next_block.size
                    i += 1
                merged_blocks.append(current)
                i += 1
            
            self.free_blocks = sorted(merged_blocks, key=lambda b: b.size, reverse=True)
        else:
            # Light defragmentation
            return self._basic_defragment()
        
        return True


class MemoryOptimizer:
    """Main class that combines memory management and defragmentation"""

    def __init__(self, defrag_strategy: DefragmentationStrategy = DefragmentationStrategy.ADAPTIVE):
        self.defrag_strategy = defrag_strategy
        self.defragmenter = self._create_defragmenter(defrag_strategy)
        self.defrag_threshold = 0.3  # Defragment when fragmentation exceeds this
        self.defrag_interval = 100  # Check every N operations
        self.operation_count = 0
        self.logger = logging.getLogger(__name__)

    def _create_defragmenter(self, strategy: DefragmentationStrategy) -> MemoryDefragmenter:
        """Create the appropriate defragmenter based on strategy"""
        if strategy == DefragmentationStrategy.VISION_OPTIMIZED:
            return VisionOptimizedDefragmenter()
        elif strategy == DefragmentationStrategy.ADAPTIVE:
            return AdaptiveDefragmenter()
        else:
            return MemoryDefragmenter(strategy)

    def add_memory_block(self, addr: int, size: int, is_free: bool, tensor_type: str = "general"):
        """Add a memory block to management"""
        self.defragmenter.add_block(addr, size, is_free, tensor_type)

    def remove_memory_block(self, addr: int):
        """Remove a memory block from management"""
        self.defragmenter.remove_block(addr)

    def allocate_memory(self, size: int, tensor_type: str = "general") -> Optional[int]:
        """Allocate memory with optional defragmentation"""
        # Increment operation count
        self.operation_count += 1

        # Check if it's time to defragment
        if (self.operation_count % self.defrag_interval == 0 or
            self.defragmenter.calculate_fragmentation() > self.defrag_threshold):
            self.defragmenter.defragment()

        # Perform allocation (in a real implementation, this would interact with a memory pool)
        # For now, we'll just return a mock address
        mock_addr = int(time.time() * 1000000) % (2**32)  # Generate a mock address
        self.defragmenter.add_block(mock_addr, size, False, tensor_type)
        
        # Record the allocation for adaptive systems
        if isinstance(self.defragmenter, AdaptiveDefragmenter):
            self.defragmenter.record_operation({
                'operation': 'allocate',
                'size': size,
                'tensor_type': tensor_type,
                'timestamp': time.time(),
                'allocation_failed': False
            })

        return mock_addr

    def free_memory(self, addr: int):
        """Free memory and update defragmenter"""
        # Mark the block as free
        # In a real implementation, this would interact with a memory pool
        # For now, we'll update our defragmenter's tracking
        self.defragmenter.add_block(addr, 0, True)  # Size doesn't matter for free blocks

        # Record the free for adaptive systems
        if isinstance(self.defragmenter, AdaptiveDefragmenter):
            self.defragmenter.record_operation({
                'operation': 'free',
                'addr': addr,
                'timestamp': time.time()
            })

    def defragment_if_needed(self) -> bool:
        """Defragment memory if fragmentation exceeds threshold"""
        fragmentation = self.defragmenter.calculate_fragmentation()
        
        if fragmentation > self.defrag_threshold:
            self.logger.info(f"Fragmentation level {fragmentation:.2%} exceeds threshold {self.defrag_threshold:.2%}, defragmenting...")
            return self.defragmenter.defragment()
        return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        defrag_stats = self.defragmenter.get_defrag_stats()
        defrag_stats['operation_count'] = self.operation_count
        defrag_stats['defrag_threshold'] = self.defrag_threshold
        defrag_stats['defrag_interval'] = self.defrag_interval
        defrag_stats['strategy'] = self.defrag_strategy.value
        return defrag_stats


def create_memory_optimizer(strategy: DefragmentationStrategy = DefragmentationStrategy.ADAPTIVE) -> MemoryOptimizer:
    """
    Factory function to create a memory optimizer with the specified strategy
    """
    return MemoryOptimizer(strategy)


def integrate_with_memory_pool(optimizer: MemoryOptimizer, memory_pool):
    """
    Integrate the defragmentation system with a memory pool
    """
    def wrapped_allocate(size: int, tensor_type: str = "general"):
        # Use the optimizer to allocate memory
        addr = optimizer.allocate_memory(size, tensor_type)
        # Also allocate in the memory pool
        if hasattr(memory_pool, 'allocate'):
            pool_addr = memory_pool.allocate(size)
            return addr, pool_addr
        return addr

    def wrapped_free(addr: int):
        # Use the optimizer to free memory
        optimizer.free_memory(addr)
        # Also free in the memory pool if it has a free method
        if hasattr(memory_pool, 'free'):
            memory_pool.free(addr)

    return wrapped_allocate, wrapped_free


if __name__ == "__main__":
    print("Memory Defragmentation System for Qwen3-VL")
    print("=" * 50)

    # Test basic defragmentation
    print("\n1. Testing Basic Defragmentation...")
    basic_defrag = MemoryDefragmenter(DefragmentationStrategy.BASIC)

    # Add some mock blocks
    basic_defrag.add_block(0x1000, 1024, False, "general")
    basic_defrag.add_block(0x2000, 2048, True, "general")   # Free
    basic_defrag.add_block(0x3000, 512, True, "general")    # Free (adjacent)
    basic_defrag.add_block(0x4000, 1024, False, "general")
    basic_defrag.add_block(0x5000, 4096, True, "general")   # Free

    print(f"   Initial fragmentation: {basic_defrag.calculate_fragmentation():.2%}")

    # Defragment
    basic_defrag.defragment()
    print(f"   After defragmentation: {basic_defrag.calculate_fragmentation():.2%}")

    # Test vision-optimized defragmentation
    print("\n2. Testing Vision-Optimized Defragmentation...")
    vision_defrag = VisionOptimizedDefragmenter()

    # Add blocks with different types
    vision_defrag.add_block(0x1000, 1024, False, "image_features")
    vision_defrag.add_block(0x2000, 2048, True, "image_features")  # Free
    vision_defrag.add_block(0x3000, 512, True, "image_features")   # Free
    vision_defrag.add_block(0x4000, 1024, False, "kv_cache")
    vision_defrag.add_block(0x5000, 4096, True, "general")

    # Record some accesses
    vision_defrag.record_access(0x1000, "image_features")
    vision_defrag.record_access(0x1000, "image_features")
    vision_defrag.record_access(0x4000, "kv_cache")

    print(f"   Initial fragmentation: {vision_defrag.calculate_fragmentation():.2%}")
    vision_defrag.defragment()
    print(f"   After defragmentation: {vision_defrag.calculate_fragmentation():.2%}")

    # Test adaptive defragmentation
    print("\n3. Testing Adaptive Defragmentation...")
    adaptive_defrag = AdaptiveDefragmenter()

    # Add blocks
    for i in range(10):
        addr = 0x1000 + i * 0x1000
        is_free = (i % 3 == 0)  # Every third block is free
        tensor_type = "general"
        if i % 4 == 0:
            tensor_type = "kv_cache"
        adaptive_defrag.add_block(addr, 1024, is_free, tensor_type)

    print(f"   Initial fragmentation: {adaptive_defrag.calculate_fragmentation():.2%}")
    adaptive_defrag.defragment()
    print(f"   After defragmentation: {adaptive_defrag.calculate_fragmentation():.2%}")

    # Test MemoryOptimizer
    print("\n4. Testing MemoryOptimizer...")
    optimizer = create_memory_optimizer(DefragmentationStrategy.ADAPTIVE)

    # Simulate memory operations
    for i in range(20):
        if i % 4 == 0:
            # Allocate memory
            addr = optimizer.allocate_memory(1024, "general")
            print(f"   Allocated at {hex(addr) if addr else 'None'}")
        elif i % 4 == 1:
            # Add free blocks to increase fragmentation
            optimizer.defragmenter.add_block(0x10000 + i * 0x100, 512, True, "general")

    # Check stats
    stats = optimizer.get_memory_stats()
    print(f"   Final fragmentation: {stats['fragmentation_level']:.2%}")
    print(f"   Total operations: {stats['operation_count']}")
    print(f"   Defragmentation success rate: {stats['successful_operations']}/{stats['total_operations']}")

    print("\nMemory Defragmentation System test completed!")
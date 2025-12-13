"""
Memory Management Package for Qwen3-VL Model

This package provides comprehensive memory management solutions for the Qwen3-VL model,
including memory pooling, allocation, compression, defragmentation, tiering, and swapping.
"""

# Import main components from consolidated modules
from .memory_pool import (
    BuddyAllocator, TensorCache, MemoryPool, OptimizedBuddyAllocator, 
    get_memory_pool, allocate_tensor, deallocate_tensor, MemoryPoolError
)

from .allocation import (
    BuddyAllocator as AllocationBuddyAllocator, SlabAllocator, SegregatedFreeListAllocator, 
    MemoryAllocatorFactory, HardwareSpecificAllocator, get_hardware_allocator, 
    allocate_with_hardware_optimization, free_tensor_with_hardware_optimization
)

from .memory_management import (
    AdvancedMemoryPool, CacheAwareMemoryManager, GPUCPUMemoryOptimizer, 
    VisionLanguageMemoryOptimizer, create_memory_optimized_model_context
)

from .memory_tiering import (
    Qwen3VLMemoryTieringSystem, MemoryTier, TensorType, create_qwen3vl_memory_tiering_system
)

from .memory_defragmentation import (
    MemoryDefragmenter, VisionOptimizedDefragmenter, AdaptiveDefragmenter, 
    MemoryOptimizer, create_memory_optimizer
)

from .memory_swapping import (
    AdvancedMemorySwapper, MemoryPressureMonitor, NVMeOptimizer, create_advanced_memory_swapper
)

from .memory_compression import (
    MemoryCompressionManager, create_memory_compression_manager
)

from .memory_pooling import (
    MemoryPool as PoolingMemoryPool, BuddyAllocator as PoolingBuddyAllocator, 
    TensorCache as PoolingTensorCache, PooledLinear, PooledMLP, PooledAttention, 
    PooledTransformerLayer, get_global_memory_pool, set_global_memory_pool, get_memory_pool
)

# Define what gets imported with "from qwen3_vl.memory_management import *"
__all__ = [
    # Memory Pool
    'BuddyAllocator',
    'TensorCache', 
    'MemoryPool',
    'OptimizedBuddyAllocator',
    'get_memory_pool',
    'allocate_tensor',
    'deallocate_tensor',
    'MemoryPoolError',
    
    # Allocation
    'AllocationBuddyAllocator',
    'SlabAllocator',
    'SegregatedFreeListAllocator',
    'MemoryAllocatorFactory',
    'HardwareSpecificAllocator',
    'get_hardware_allocator',
    'allocate_with_hardware_optimization',
    'free_tensor_with_hardware_optimization',
    
    # Memory Management
    'AdvancedMemoryPool',
    'CacheAwareMemoryManager',
    'GPUCPUMemoryOptimizer',
    'VisionLanguageMemoryOptimizer',
    'create_memory_optimized_model_context',
    
    # Memory Tiering
    'Qwen3VLMemoryTieringSystem',
    'MemoryTier',
    'TensorType',
    'create_qwen3vl_memory_tiering_system',
    
    # Memory Defragmentation
    'MemoryDefragmenter',
    'VisionOptimizedDefragmenter',
    'AdaptiveDefragmenter',
    'MemoryOptimizer',
    'create_memory_optimizer',
    
    # Memory Swapping
    'AdvancedMemorySwapper',
    'MemoryPressureMonitor',
    'NVMeOptimizer',
    'create_advanced_memory_swapper',
    
    # Memory Compression
    'MemoryCompressionManager',
    'create_memory_compression_manager',
    
    # Memory Pooling
    'PoolingMemoryPool',
    'PoolingBuddyAllocator',
    'PoolingTensorCache',
    'PooledLinear',
    'PooledMLP',
    'PooledAttention',
    'PooledTransformerLayer',
    'get_global_memory_pool',
    'set_global_memory_pool'
]
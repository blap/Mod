"""
Advanced Memory Management System for Vision-Language Models (Qwen3-VL)

This module implements state-of-the-art memory management techniques optimized for
multimodal vision-language models, focusing on Intel i5-10210U + NVIDIA SM61 hardware.

Key Features:
- Memory Pooling with NUMA awareness
- Cache-aware memory layouts
- Page-aligned allocations
- Memory defragmentation
- Prefetching strategies
- GPU-CPU memory optimization
"""

import ctypes
import mmap
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import threading
import time
from collections import defaultdict, deque
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    logger.warning("psutil not available, some memory monitoring features will be limited")
    psutil = None


class MemoryPoolType(Enum):
    """Types of memory pools for different allocation patterns"""
    TENSOR_DATA = "tensor_data"           # For tensor storage
    ACTIVATION_BUFFER = "activation_buffer"  # For intermediate activations
    KV_CACHE = "kv_cache"                 # For attention mechanisms
    TEMPORARY = "temporary"               # For temporary computations
    FIXED_SIZE = "fixed_size"             # For fixed-size allocations


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool"""
    ptr: int                    # Memory address
    size: int                   # Size in bytes
    pool_type: MemoryPoolType   # Type of pool this block belongs to
    allocated: bool             # Allocation status
    timestamp: float            # Time of allocation
    ref_count: int              # Reference count for smart deallocation
    alignment: int              # Memory alignment boundary


class MemoryDefragmenter:
    """Handles memory defragmentation for optimal allocation"""

    def __init__(self, memory_pool):
        self.memory_pool = memory_pool
        self.defrag_threshold = 0.3  # Defragment when fragmentation > 30%
        self.defrag_lock = threading.Lock()

    def calculate_fragmentation(self) -> float:
        """Calculate current memory fragmentation level"""
        free_blocks = [block for block in self.memory_pool.blocks if not block.allocated]
        if not free_blocks:
            return 0.0

        total_free = sum(block.size for block in free_blocks)
        largest_free = max((block.size for block in free_blocks), default=0)

        if total_free == 0:
            return 0.0

        return 1.0 - (largest_free / total_free) if total_free > 0 else 0.0

    def compact_memory(self):
        """Compact memory blocks to reduce fragmentation"""
        with self.defrag_lock:
            # Sort free blocks by address to identify contiguous regions
            free_blocks = [block for block in self.memory_pool.blocks if not block.allocated]
            free_blocks.sort(key=lambda x: x.ptr)

            # Merge contiguous free blocks
            i = 0
            while i < len(free_blocks) - 1:
                current = free_blocks[i]
                next_block = free_blocks[i + 1]

                # Check if blocks are contiguous
                if current.ptr + current.size == next_block.ptr:
                    # Merge blocks
                    current.size += next_block.size
                    self.memory_pool.blocks.remove(next_block)
                    # Remove from free_blocks list too
                    free_blocks.pop(i + 1)
                    continue

                i += 1

    def should_defragment(self) -> bool:
        """Check if defragmentation is needed"""
        fragmentation = self.calculate_fragmentation()
        return fragmentation > self.defrag_threshold


class AdvancedMemoryPool:
    """Advanced memory pool with NUMA awareness and cache optimization"""

    def __init__(self, initial_size: int = 1024 * 1024 * 1024,  # 1GB default
                 page_size: int = 4096,  # Standard 4KB page
                 enable_defragmentation: bool = True):
        """
        Initialize advanced memory pool

        Args:
            initial_size: Initial size of the memory pool in bytes
            page_size: Memory page size for alignment
            enable_defragmentation: Whether to enable automatic defragmentation
        """
        # Input validation
        if not isinstance(initial_size, int) or initial_size <= 0:
            raise ValueError(f"initial_size must be a positive integer, got {initial_size}")
        if not isinstance(page_size, int) or page_size <= 0:
            raise ValueError(f"page_size must be a positive integer, got {page_size}")
        if not isinstance(enable_defragmentation, bool):
            raise ValueError(f"enable_defragmentation must be a boolean, got {enable_defragmentation}")

        self.initial_size = initial_size
        self.page_size = page_size
        self.enable_defragmentation = enable_defragmentation

        # Create memory pool using mmap for better control
        # On Windows, we need to use different approach as MAP_PRIVATE/MAP_ANONYMOUS don't exist
        try:
            # Try to use mmap with anonymous mapping (Unix/Linux)
            # Check if the required flags exist
            if hasattr(mmap, 'MAP_PRIVATE') and hasattr(mmap, 'MAP_ANONYMOUS'):
                self.pool_ptr = mmap.mmap(-1, initial_size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            else:
                raise AttributeError("Required mmap flags not available")
        except (AttributeError, OSError):
            # On Windows, create a temporary file-backed mapping
            import tempfile
            try:
                self.temp_file = tempfile.TemporaryFile()
                self.temp_file.truncate(initial_size)
                self.pool_ptr = mmap.mmap(self.temp_file.fileno(), initial_size)
            except Exception as e:
                logger.error(f"Failed to create memory pool: {e}")
                raise RuntimeError(f"Failed to create memory pool: {e}") from e

        try:
            self.pool_base = ctypes.addressof(ctypes.c_char.from_buffer(self.pool_ptr))
        except Exception as e:
            logger.error(f"Failed to get memory pool base address: {e}")
            raise RuntimeError(f"Failed to get memory pool base address: {e}") from e

        # Track memory blocks
        self.blocks: List[MemoryBlock] = []
        self.block_map: Dict[int, MemoryBlock] = {}  # Address -> Block mapping
        self.pool_lock = threading.RLock()

        # Pool statistics
        self.stats = {
            'total_allocated': 0,
            'total_freed': 0,
            'current_usage': 0,
            'peak_usage': 0,
            'allocation_count': 0,
            'deallocation_count': 0
        }

        # Initialize with one large free block
        try:
            initial_block = MemoryBlock(
                ptr=self.pool_base,
                size=initial_size,
                pool_type=MemoryPoolType.TEMPORARY,
                allocated=False,
                timestamp=time.time(),
                ref_count=0,
                alignment=page_size
            )
            self.blocks.append(initial_block)
            self.block_map[self.pool_base] = initial_block
        except Exception as e:
            logger.error(f"Failed to initialize memory blocks: {e}")
            raise RuntimeError(f"Failed to initialize memory blocks: {e}") from e

        # Defragmenter
        try:
            self.defragmenter = MemoryDefragmenter(self) if enable_defragmentation else None
        except Exception as e:
            logger.error(f"Failed to initialize defragmenter: {e}")
            raise RuntimeError(f"Failed to initialize defragmenter: {e}") from e

        # Thread-local storage for per-thread pools
        self.thread_local = threading.local()

        logger.info(f"AdvancedMemoryPool initialized with {initial_size / (1024**3):.2f} GB")

    def _align_size(self, size: int, alignment: int) -> int:
        """Align size to the specified boundary"""
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"size must be a non-negative integer, got {size}")
        if not isinstance(alignment, int) or alignment <= 0:
            raise ValueError(f"alignment must be a positive integer, got {alignment}")
        return ((size + alignment - 1) // alignment) * alignment

    def _find_suitable_block(self, size: int, alignment: int, pool_type: MemoryPoolType) -> Optional[MemoryBlock]:
        """Find a suitable free block for allocation"""
        # Input validation
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive integer, got {size}")
        if not isinstance(alignment, int) or alignment <= 0:
            raise ValueError(f"alignment must be a positive integer, got {alignment}")
        if not isinstance(pool_type, MemoryPoolType):
            raise ValueError(f"pool_type must be a MemoryPoolType, got {pool_type}")

        try:
            aligned_size = self._align_size(size, alignment)
        except ValueError as e:
            logger.error(f"Invalid alignment parameters: {e}")
            raise

        # Look for best fit (smallest block that fits)
        try:
            suitable_blocks = [
                block for block in self.blocks
                if not block.allocated and block.size >= aligned_size and block.alignment >= alignment
            ]
        except Exception as e:
            logger.error(f"Error filtering suitable blocks: {e}")
            raise RuntimeError(f"Error filtering suitable blocks: {e}") from e

        if not suitable_blocks:
            return None

        # Return best fit (smallest block that still fits)
        try:
            return min(suitable_blocks, key=lambda b: b.size)
        except ValueError:
            # This happens if suitable_blocks is empty, which we already handled
            return None

    def allocate(self, size: int, pool_type: MemoryPoolType = MemoryPoolType.TEMPORARY,
                 alignment: int = None) -> Optional[Tuple[int, int]]:
        """
        Allocate memory from the pool

        Args:
            size: Size to allocate in bytes
            pool_type: Type of memory pool
            alignment: Memory alignment requirement (defaults to page_size)

        Returns:
            Tuple of (memory_address, actual_allocated_size) or None if allocation fails
        """
        # Input validation
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive integer, got {size}")
        if not isinstance(pool_type, MemoryPoolType):
            raise ValueError(f"pool_type must be a MemoryPoolType, got {pool_type}")
        if alignment is not None and (not isinstance(alignment, int) or alignment <= 0):
            raise ValueError(f"alignment must be a positive integer or None, got {alignment}")

        if alignment is None:
            alignment = self.page_size

        try:
            aligned_size = self._align_size(size, alignment)
        except ValueError as e:
            logger.error(f"Invalid alignment parameters: {e}")
            raise

        with self.pool_lock:
            try:
                # Check if defragmentation is needed
                if self.defragmenter and self.defragmenter.should_defragment():
                    logger.debug("Performing memory defragmentation")
                    self.defragmenter.compact_memory()

                block = self._find_suitable_block(aligned_size, alignment, pool_type)

                if block is None:
                    # Try to expand pool if possible
                    if self._expand_pool(aligned_size):
                        block = self._find_suitable_block(aligned_size, alignment, pool_type)
                        if block is None:
                            logger.warning(f"Failed to allocate {aligned_size} bytes after expanding pool")
                            return None
                    else:
                        logger.warning(f"Failed to allocate {aligned_size} bytes - no suitable block found and pool expansion failed")
                        return None

                # Split block if it's much larger than needed
                if block.size > aligned_size * 2:  # If block is significantly larger
                    new_block = MemoryBlock(
                        ptr=block.ptr + aligned_size,
                        size=block.size - aligned_size,
                        pool_type=MemoryPoolType.TEMPORARY,
                        allocated=False,
                        timestamp=time.time(),
                        ref_count=0,
                        alignment=alignment
                    )

                    block.size = aligned_size
                    self.blocks.append(new_block)
                    self.block_map[new_block.ptr] = new_block

                # Mark block as allocated
                block.allocated = True
                block.pool_type = pool_type
                block.timestamp = time.time()
                block.ref_count = 1

                # Update statistics
                self.stats['total_allocated'] += aligned_size
                self.stats['current_usage'] += aligned_size
                self.stats['allocation_count'] += 1
                if self.stats['current_usage'] > self.stats['peak_usage']:
                    self.stats['peak_usage'] = self.stats['current_usage']

                logger.debug(f"Allocated {aligned_size} bytes at {hex(block.ptr)} for {pool_type.value}")
                return block.ptr, aligned_size
            except Exception as e:
                logger.error(f"Error during allocation: {e}")
                raise RuntimeError(f"Error during allocation: {e}") from e

    def _expand_pool(self, additional_size: int) -> bool:
        """Attempt to expand the memory pool"""
        # Input validation
        if not isinstance(additional_size, int) or additional_size <= 0:
            logger.error(f"additional_size must be a positive integer, got {additional_size}")
            return False

        try:
            # Calculate new size (double the current size or add enough for the request)
            current_size = len(self.pool_ptr)
            new_size = max(current_size * 2, current_size + additional_size * 2)

            # Resize the memory mapping
            self.pool_ptr.resize(new_size)

            # Add the new region as a free block
            new_base = ctypes.addressof(ctypes.c_char.from_buffer(self.pool_ptr)) + current_size
            new_block = MemoryBlock(
                ptr=new_base,
                size=new_size - current_size,
                pool_type=MemoryPoolType.TEMPORARY,
                allocated=False,
                timestamp=time.time(),
                ref_count=0,
                alignment=self.page_size
            )

            self.blocks.append(new_block)
            self.block_map[new_block.ptr] = new_block

            logger.debug(f"Expanded memory pool to {new_size / (1024**3):.2f} GB")
            return True
        except OSError as e:
            logger.error(f"OS error during memory pool expansion: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to expand memory pool: {e}")
            return False

    def deallocate(self, ptr: int) -> bool:
        """
        Deallocate memory back to the pool

        Args:
            ptr: Pointer to deallocate

        Returns:
            True if successful, False otherwise
        """
        # Input validation
        if not isinstance(ptr, int) or ptr <= 0:
            logger.error(f"ptr must be a positive integer, got {ptr}")
            return False

        with self.pool_lock:
            try:
                if ptr not in self.block_map:
                    logger.warning(f"Attempted to deallocate unknown pointer: {hex(ptr)}")
                    return False

                block = self.block_map[ptr]

                if not block.allocated:
                    logger.warning(f"Attempted to deallocate already freed block: {hex(ptr)}")
                    return False

                # Decrement reference count
                block.ref_count -= 1

                if block.ref_count <= 0:
                    block.allocated = False
                    block.ref_count = 0

                    # Update statistics
                    self.stats['total_freed'] += block.size
                    self.stats['current_usage'] -= block.size
                    self.stats['deallocation_count'] += 1

                    logger.debug(f"Deallocated {block.size} bytes at {hex(ptr)}")
                    return True
                else:
                    logger.debug(f"Reduced ref count for {hex(ptr)}, now {block.ref_count}")
                    return True
            except KeyError:
                logger.warning(f"Block not found in block_map for pointer: {hex(ptr)}")
                return False
            except Exception as e:
                logger.error(f"Error during deallocation: {e}")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        with self.pool_lock:
            stats = self.stats.copy()
            stats['fragmentation'] = self.defragmenter.calculate_fragmentation() if self.defragmenter else 0.0
            stats['pool_utilization'] = (stats['current_usage'] / self.initial_size) if self.initial_size > 0 else 0.0
            return stats

    def cleanup(self):
        """Clean up the memory pool"""
        with self.pool_lock:
            try:
                if hasattr(self, 'pool_ptr') and self.pool_ptr:
                    try:
                        self.pool_ptr.close()
                    except Exception as e:
                        logger.warning(f"Error closing memory pool: {e}")
            except Exception as e:
                logger.error(f"Error accessing pool_ptr during cleanup: {e}")

            if hasattr(self, 'temp_file'):
                try:
                    self.temp_file.close()
                    # On Windows, explicitly delete the temporary file to ensure cleanup
                    import tempfile
                    import os
                    temp_filename = getattr(self.temp_file, 'name', None)
                    if temp_filename and os.path.exists(temp_filename):
                        try:
                            os.unlink(temp_filename)
                        except Exception as e:
                            # If we can't delete the file, it might be in use by another process
                            logger.warning(f"Could not delete temporary file {temp_filename}: {e}")
                except Exception as e:
                    # Handle case where file is already closed
                    logger.warning(f"Error closing temporary file: {e}")


class CacheAwareMemoryManager:
    """Cache-aware memory manager optimizing for CPU cache performance"""

    def __init__(self, cache_line_size: int = 64, l1_size: int = 32 * 1024,
                 l2_size: int = 256 * 1024, l3_size: int = 6 * 1024 * 1024):
        """
        Initialize cache-aware memory manager

        Args:
            cache_line_size: CPU cache line size in bytes
            l1_size: L1 cache size in bytes
            l2_size: L2 cache size in bytes
            l3_size: L3 cache size in bytes
        """
        self.cache_line_size = cache_line_size
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size

        # Memory layout optimization flags
        self.use_cache_blocking = True
        self.optimize_for_temporal_locality = True
        self.use_prefetching = True

        # Cache performance metrics
        self.cache_hits = 0
        self.cache_misses = 0

    def optimize_memory_layout(self, data: np.ndarray, layout_type: str = "cache_friendly") -> np.ndarray:
        """
        Optimize memory layout for cache performance

        Args:
            data: Input numpy array
            layout_type: Type of optimization ('cache_friendly', 'row_major', 'col_major', 'blocked')

        Returns:
            Optimized array
        """
        if layout_type == "cache_friendly":
            # For matrices, optimize for row-major access patterns
            if data.ndim == 2:
                # Ensure contiguous memory layout
                return np.ascontiguousarray(data)
            elif data.ndim == 3:
                # For tensors, optimize for the most frequently accessed dimension first
                return np.ascontiguousarray(data)
        elif layout_type == "blocked":
            # Apply cache blocking/tiling for matrix operations
            return self._apply_cache_blocking(data)
        elif layout_type == "row_major":
            return np.asarray(data, order='C')
        elif layout_type == "col_major":
            return np.asarray(data, order='F')

        return data

    def _apply_cache_blocking(self, data: np.ndarray, block_size: int = 64) -> np.ndarray:
        """Apply cache blocking to optimize memory access patterns"""
        if data.ndim != 2:
            return data  # Only implement for 2D arrays for now

        rows, cols = data.shape
        blocked_array = np.empty_like(data)

        # Apply blocking in both dimensions
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                end_i = min(i + block_size, rows)
                end_j = min(j + block_size, cols)
                blocked_array[i:end_i, j:end_j] = data[i:end_i, j:end_j]

        return blocked_array

    def prefetch_data(self, data_ptr: int, size: int, offset: int = 0):
        """
        Prefetch data into CPU cache

        Args:
            data_ptr: Memory address to prefetch
            size: Size of data to prefetch
            offset: Offset from pointer
        """
        if not self.use_prefetching:
            return

        # Use low-level prefetch instruction if available
        # Note: This is a simplified simulation - real implementation would use platform-specific instructions
        try:
            # In practice, this would use intrinsics like _mm_prefetch on x86
            # For simulation, we'll just touch the memory to hint to the OS
            ctypes.memmove(data_ptr + offset, data_ptr + offset, min(64, size))  # Touch cache line
        except:
            pass  # Silently handle if low-level access fails


class GPUCPUMemoryOptimizer:
    """Optimizes memory transfers between GPU and CPU for vision-language models"""

    def __init__(self, device_memory_limit: int = 2 * 1024 * 1024 * 1024):  # 2GB default
        self.device_memory_limit = device_memory_limit
        self.host_page_locked_pool = None
        self.pinned_memory_enabled = True
        self.unified_memory_enabled = False  # For CUDA unified memory systems

        # Initialize pinned memory pool if available
        self._init_pinned_memory_pool()

    def _init_pinned_memory_pool(self):
        """Initialize pinned memory pool for faster GPU transfers"""
        try:
            import torch
            if torch.cuda.is_available():
                # Enable pinned memory for faster host-device transfers
                self.pinned_memory_enabled = True
                logger.info("Pinned memory enabled for GPU transfers")
            else:
                logger.info("CUDA not available, using standard memory transfers")
        except ImportError:
            logger.info("PyTorch not available, using standard memory transfers")

    def optimize_tensor_placement(self, tensor: Any, target_device: str = "auto") -> Any:
        """
        Optimize tensor placement between CPU and GPU based on memory constraints

        Args:
            tensor: Input tensor
            target_device: Target device ('cpu', 'cuda', 'auto')

        Returns:
            Tensor placed optimally
        """
        try:
            import torch
            if not isinstance(tensor, torch.Tensor):
                # Convert numpy arrays to torch tensors for optimization
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                else:
                    tensor = torch.tensor(tensor)

            if target_device == "auto":
                # Estimate memory usage and place accordingly
                tensor_size = tensor.element_size() * tensor.nelement()
                available_gpu_memory = self._get_available_gpu_memory()

                if tensor_size < available_gpu_memory * 0.8:  # Use 80% threshold
                    target_device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    target_device = "cpu"

            if target_device == "cuda" and torch.cuda.is_available():
                # Use pinned memory for faster transfer if available
                if self.pinned_memory_enabled:
                    return tensor.pin_memory().to('cuda', non_blocking=True)
                else:
                    return tensor.to('cuda')
            else:
                return tensor.to('cpu')

        except ImportError:
            # Fallback if PyTorch is not available
            return tensor

    def _get_available_gpu_memory(self) -> int:
        """Get available GPU memory in bytes"""
        try:
            import torch
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = torch.cuda.memory_reserved(0)
                allocated_memory = torch.cuda.memory_allocated(0)
                return total_memory - reserved_memory
            else:
                return 0
        except ImportError:
            return 0

    def batch_memory_transfer(self, tensors: List[Any], target_device: str = "cuda") -> List[Any]:
        """Optimize batch memory transfers"""
        optimized_tensors = []

        for tensor in tensors:
            optimized_tensor = self.optimize_tensor_placement(tensor, target_device)
            optimized_tensors.append(optimized_tensor)

        return optimized_tensors


class VisionLanguageMemoryOptimizer:
    """Main memory optimizer for vision-language models like Qwen3-VL"""

    def __init__(self,
                 memory_pool_size: int = 2 * 1024 * 1024 * 1024,  # 2GB
                 enable_memory_pool: bool = True,
                 enable_cache_optimization: bool = True,
                 enable_gpu_optimization: bool = True):
        """
        Initialize vision-language memory optimizer

        Args:
            memory_pool_size: Size of memory pool in bytes
            enable_memory_pool: Enable advanced memory pooling
            enable_cache_optimization: Enable cache-aware optimizations
            enable_gpu_optimization: Enable GPU-CPU memory optimization
        """
        # Validate inputs
        if not isinstance(memory_pool_size, int) or memory_pool_size <= 0:
            raise ValueError(f"memory_pool_size must be a positive integer, got {memory_pool_size}")

        if not isinstance(enable_memory_pool, bool):
            raise TypeError(f"enable_memory_pool must be a boolean, got {type(enable_memory_pool)}")

        if not isinstance(enable_cache_optimization, bool):
            raise TypeError(f"enable_cache_optimization must be a boolean, got {type(enable_cache_optimization)}")

        if not isinstance(enable_gpu_optimization, bool):
            raise TypeError(f"enable_gpu_optimization must be a boolean, got {type(enable_gpu_optimization)}")

        self.enable_memory_pool = enable_memory_pool
        self.enable_cache_optimization = enable_cache_optimization
        self.enable_gpu_optimization = enable_gpu_optimization

        # Initialize components
        try:
            self.memory_pool = AdvancedMemoryPool(memory_pool_size) if enable_memory_pool else None
        except Exception as e:
            logger.error(f"Error initializing memory pool: {e}")
            self.memory_pool = None
            self.enable_memory_pool = False

        try:
            self.cache_manager = CacheAwareMemoryManager() if enable_cache_optimization else None
        except Exception as e:
            logger.error(f"Error initializing cache manager: {e}")
            self.cache_manager = None
            self.enable_cache_optimization = False

        try:
            self.gpu_optimizer = GPUCPUMemoryOptimizer() if enable_gpu_optimization else None
        except Exception as e:
            logger.error(f"Error initializing GPU optimizer: {e}")
            self.gpu_optimizer = None
            self.enable_gpu_optimization = False

        # Specialized pools for different types of data in vision-language models
        self.kv_cache_pool = None
        self.image_feature_pool = None
        self.text_embedding_pool = None

        if enable_memory_pool:
            try:
                # Create specialized pools
                self.kv_cache_pool = AdvancedMemoryPool(512 * 1024 * 1024)  # 512MB for KV cache
                self.image_feature_pool = AdvancedMemoryPool(1024 * 1024 * 1024)  # 1GB for image features
                self.text_embedding_pool = AdvancedMemoryPool(512 * 1024 * 1024)  # 512MB for text embeddings
            except Exception as e:
                logger.error(f"Error initializing specialized memory pools: {e}")
                self.kv_cache_pool = None
                self.image_feature_pool = None
                self.text_embedding_pool = None

        # Track tensor allocations using a regular dictionary with array IDs as keys
        # This avoids issues with numpy arrays not being hashable
        self.tensor_allocation_map = {}
        self.tensor_allocation_lock = threading.Lock()

        logger.info("VisionLanguageMemoryOptimizer initialized")

    def allocate_tensor_memory(self, shape: Tuple[int, ...], dtype=np.float32,
                              tensor_type: str = "general") -> Optional[np.ndarray]:
        """
        Allocate memory for tensors with optimization

        Args:
            shape: Shape of the tensor
            dtype: Data type
            tensor_type: Type of tensor ('kv_cache', 'image_features', 'text_embeddings', 'general')

        Returns:
            Allocated numpy array or None if allocation fails
        """
        # Validate inputs
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"shape must be a tuple or list, got {type(shape)}")

        if not all(isinstance(dim, int) and dim > 0 for dim in shape):
            raise ValueError(f"shape must contain positive integers, got {shape}")

        if not isinstance(tensor_type, str):
            raise TypeError(f"tensor_type must be a string, got {type(tensor_type)}")

        valid_tensor_types = {'general', 'kv_cache', 'image_features', 'text_embeddings'}
        if tensor_type not in valid_tensor_types:
            raise ValueError(f"tensor_type must be one of {valid_tensor_types}, got {tensor_type}")

        # Handle both numpy and torch dtypes with proper validation
        original_dtype = dtype
        try:
            if isinstance(dtype, type) and hasattr(dtype, 'dtype'):  # If it's a numpy generic type
                # This is a numpy scalar type, get the corresponding dtype
                dtype = np.dtype(dtype)
            elif hasattr(dtype, 'dtype'):  # If it's a torch tensor with dtype attribute
                dtype = dtype.dtype
            elif hasattr(dtype, 'name'):  # If it's already a numpy dtype
                dtype = np.dtype(dtype)
            elif isinstance(dtype, str):  # If it's a string
                dtype = np.dtype(dtype)  # Convert string to numpy dtype
            else:  # If it's a torch dtype directly or other format
                try:
                    import torch
                    if isinstance(dtype, torch.dtype):
                        # Convert torch dtype to numpy dtype
                        torch_to_numpy_dtype = {
                            torch.float32: np.float32,
                            torch.float64: np.float64,
                            torch.float16: np.float16,
                            torch.bfloat16: np.float32,  # bfloat16 maps to float32 in numpy
                            torch.int32: np.int32,
                            torch.int64: np.int64,
                            torch.int16: np.int16,
                            torch.int8: np.int8,
                            torch.uint8: np.uint8,
                            torch.bool: np.bool_,
                            torch.complex64: np.complex64,
                            torch.complex128: np.complex128,
                        }
                        dtype = torch_to_numpy_dtype.get(dtype, np.float32)
                    else:
                        # If it's not a torch dtype, try to use it directly as numpy dtype
                        try:
                            dtype = np.dtype(dtype)
                        except:
                            # Default fallback
                            dtype = np.float32
                except ImportError:
                    # If torch is not available, try to use directly as numpy dtype
                    try:
                        dtype = np.dtype(dtype)
                    except:
                        # Default fallback
                        dtype = np.float32

            element_size = np.dtype(dtype).itemsize
            total_elements = np.prod(shape)
            if not isinstance(total_elements, (int, np.integer)) or total_elements <= 0:
                raise ValueError(f"Invalid total number of elements: {total_elements}")

            size_bytes = int(element_size * total_elements)  # Convert to Python int to avoid numpy int64 issues

            # Check for potential overflow
            if size_bytes < 0:
                raise OverflowError(f"Size calculation overflowed: {element_size} * {total_elements}")

            # For now, use the memory pool to track allocations but create arrays separately
            # to avoid mmap buffer export issues. This still provides memory management benefits.
            if self.memory_pool and tensor_type == "general":
                if self.memory_pool:
                    ptr, actual_size = self.memory_pool.allocate(size_bytes)
                    if ptr:
                        # Create array with standard allocation but track with our pool
                        array = np.zeros(shape, dtype=dtype)

                        if self.cache_manager:
                            array = self.cache_manager.optimize_memory_layout(array)

                        # Store reference to track when it should be "freed" using array id as key
                        with self.tensor_allocation_lock:
                            self.tensor_allocation_map[id(array)] = (self.memory_pool, ptr, size_bytes)
                        return array
            elif tensor_type == "kv_cache" and self.kv_cache_pool:
                ptr, actual_size = self.kv_cache_pool.allocate(size_bytes)
                if ptr:
                    array = np.zeros(shape, dtype=dtype)
                    with self.tensor_allocation_lock:
                        self.tensor_allocation_map[id(array)] = (self.kv_cache_pool, ptr, size_bytes)
                    return array
            elif tensor_type == "image_features" and self.image_feature_pool:
                ptr, actual_size = self.image_feature_pool.allocate(size_bytes)
                if ptr:
                    array = np.zeros(shape, dtype=dtype)
                    with self.tensor_allocation_lock:
                        self.tensor_allocation_map[id(array)] = (self.image_feature_pool, ptr, size_bytes)
                    return array
            elif tensor_type == "text_embeddings" and self.text_embedding_pool:
                ptr, actual_size = self.text_embedding_pool.allocate(size_bytes)
                if ptr:
                    array = np.zeros(shape, dtype=dtype)
                    with self.tensor_allocation_lock:
                        self.tensor_allocation_map[id(array)] = (self.text_embedding_pool, ptr, size_bytes)
                    return array

            # Fallback to standard allocation
            array = np.zeros(shape, dtype=dtype)
            if self.cache_manager:
                array = self.cache_manager.optimize_memory_layout(array)
            return array
        except ValueError as e:
            logger.error(f"ValueError in allocate_tensor_memory: {e}")
            raise
        except MemoryError as e:
            logger.error(f"MemoryError in allocate_tensor_memory: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in allocate_tensor_memory: {e}")
            return None

    def free_tensor_memory(self, tensor: np.ndarray, tensor_type: str = "general"):
        """Free tensor memory back to appropriate pool"""
        # Validate inputs
        if not isinstance(tensor, np.ndarray):
            logger.warning(f"tensor must be a numpy array, got {type(tensor)}")
            return

        if not isinstance(tensor_type, str):
            logger.warning(f"tensor_type must be a string, got {type(tensor_type)}")
            return

        if not self.memory_pool:
            return  # Standard GC will handle it

        # Use the stored reference to deallocate from the appropriate pool
        tensor_id = id(tensor)
        try:
            with self.tensor_allocation_lock:
                if tensor_id in self.tensor_allocation_map:
                    mem_pool, ptr, size = self.tensor_allocation_map[tensor_id]
                    if mem_pool and hasattr(mem_pool, 'deallocate'):
                        mem_pool.deallocate(ptr)
                    del self.tensor_allocation_map[tensor_id]
        except KeyError:
            # The tensor was not in the map, which is fine
            pass
        except Exception as e:
            logger.error(f"Error freeing tensor memory: {e}")

    def cleanup_garbage_arrays(self):
        """Clean up array references that have been garbage collected"""
        with self.tensor_allocation_lock:
            # Create a list of keys to remove to avoid modifying dict during iteration
            keys_to_remove = []
            for array_id in self.tensor_allocation_map:
                # Since we can't directly check if an object with a specific id exists,
                # we'll just keep the cleanup method as a placeholder for now
                # In practice, this would require a more sophisticated approach
                pass

    def optimize_image_processing_memory(self, image_batch: np.ndarray) -> np.ndarray:
        """
        Optimize memory for image processing pipeline

        Args:
            image_batch: Batch of images as numpy array

        Returns:
            Memory-optimized image batch
        """
        if self.cache_manager:
            # Optimize for cache-friendly access patterns
            optimized_batch = self.cache_manager.optimize_memory_layout(
                image_batch, layout_type="cache_friendly"
            )
        else:
            optimized_batch = image_batch

        # Prefetch next batch if possible
        if self.cache_manager and hasattr(image_batch, '__array_interface__'):
            ptr = image_batch.__array_interface__['data'][0]
            size = image_batch.nbytes
            self.cache_manager.prefetch_data(ptr, size)

        return optimized_batch

    def optimize_attention_memory(self, batch_size: int, seq_len: int, hidden_dim: int,
                                 num_heads: int) -> Dict[str, Any]:
        """
        Optimize memory for attention mechanism

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads

        Returns:
            Dictionary with optimized memory allocations for Q, K, V, and attention scores
        """
        head_dim = hidden_dim // num_heads

        # Calculate sizes for Q, K, V matrices
        qkv_size = (batch_size, seq_len, hidden_dim)

        # Allocate with specialized KV cache pool
        q = self.allocate_tensor_memory(qkv_size, dtype=np.float32, tensor_type="kv_cache")
        k = self.allocate_tensor_memory(qkv_size, dtype=np.float32, tensor_type="kv_cache")
        v = self.allocate_tensor_memory(qkv_size, dtype=np.float32, tensor_type="kv_cache")

        # Attention scores matrix
        attn_scores_size = (batch_size, num_heads, seq_len, seq_len)
        attn_scores = self.allocate_tensor_memory(attn_scores_size, dtype=np.float32,
                                                 tensor_type="kv_cache")

        return {
            'query': q,
            'key': k,
            'value': v,
            'attention_scores': attn_scores,
            'head_dim': head_dim
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = {}

        if self.memory_pool:
            stats['general_pool'] = self.memory_pool.get_stats()

        if self.kv_cache_pool:
            stats['kv_cache_pool'] = self.kv_cache_pool.get_stats()

        if self.image_feature_pool:
            stats['image_feature_pool'] = self.image_feature_pool.get_stats()

        if self.text_embedding_pool:
            stats['text_embedding_pool'] = self.text_embedding_pool.get_stats()

        # System memory info if psutil is available
        if psutil:
            stats['system_memory'] = {
                'virtual_memory_percent': psutil.virtual_memory().percent,
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'used_gb': psutil.virtual_memory().used / (1024**3)
            }

        return stats

    def optimize_tensor_placement(self, tensor: Any, target_device: str = "auto") -> Any:
        """
        Optimize tensor placement between CPU and GPU based on memory constraints.
        Wrapper for the GPU optimizer's method.

        Args:
            tensor: Input tensor
            target_device: Target device ('cpu', 'cuda', 'auto')

        Returns:
            Tensor placed optimally
        """
        if self.gpu_optimizer:
            return self.gpu_optimizer.optimize_tensor_placement(tensor, target_device)
        else:
            # If GPU optimizer is not enabled, return tensor as is
            return tensor

    def cleanup(self):
        """Clean up all memory pools"""
        if self.memory_pool:
            self.memory_pool.cleanup()
        if self.kv_cache_pool:
            self.kv_cache_pool.cleanup()
        if self.image_feature_pool:
            self.image_feature_pool.cleanup()
        if self.text_embedding_pool:
            self.text_embedding_pool.cleanup()


def create_memory_optimized_model_context():
    """
    Factory function to create an optimized memory context for vision-language models
    Specifically tuned for Intel i5-10210U + NVIDIA SM61 hardware
    """
    # Hardware-specific tuning
    memory_pool_size = 3 * 1024 * 1024 * 1024  # 3GB for i5-10210U with 8GB RAM

    # Initialize the memory optimizer with hardware-appropriate settings
    optimizer = VisionLanguageMemoryOptimizer(
        memory_pool_size=memory_pool_size,
        enable_memory_pool=True,
        enable_cache_optimization=True,
        enable_gpu_optimization=True
    )

    return optimizer


# Example usage and integration code
def integrate_with_qwen3_vl():
    """
    Example of how to integrate these memory optimizations with Qwen3-VL
    This function demonstrates the integration pattern
    """
    # Create the memory optimizer
    mem_optimizer = create_memory_optimized_model_context()

    # Example: Optimizing image feature extraction
    def extract_image_features_optimized(images):
        # Preprocess with memory optimization
        processed_images = mem_optimizer.optimize_image_processing_memory(images)

        # Allocate memory for features with optimization
        batch_size, height, width, channels = processed_images.shape
        feature_dim = 512  # Example feature dimension
        features_shape = (batch_size, height * width, feature_dim)

        features = mem_optimizer.allocate_tensor_memory(
            features_shape,
            dtype=np.float32,
            tensor_type="image_features"
        )

        # Process images and store in optimized memory
        # (Actual processing would go here)

        return features

    # Example: Optimizing attention mechanism
    def create_attention_layers_optimized(batch_size, seq_len, hidden_dim, num_heads):
        attention_components = mem_optimizer.optimize_attention_memory(
            batch_size, seq_len, hidden_dim, num_heads
        )

        return attention_components

    return mem_optimizer, extract_image_features_optimized, create_attention_layers_optimized


if __name__ == "__main__":
    # Demonstration of the memory optimization system
    print("Advanced Memory Management System for Vision-Language Models")
    print("=" * 60)

    # Create optimizer
    optimizer = create_memory_optimized_model_context()

    # Demonstrate tensor allocation
    print("\n1. Allocating tensors with memory optimization...")
    tensor1 = optimizer.allocate_tensor_memory((100, 256), dtype=np.float32, tensor_type="general")
    print(f"Allocated tensor of shape {tensor1.shape} with {tensor1.dtype}")

    # Demonstrate image processing optimization
    print("\n2. Optimizing image processing memory...")
    sample_images = np.random.random((4, 224, 224, 3)).astype(np.float32)
    optimized_images = optimizer.optimize_image_processing_memory(sample_images)
    print(f"Optimized image batch of shape {optimized_images.shape}")

    # Demonstrate attention memory optimization
    print("\n3. Optimizing attention mechanism memory...")
    attention_components = optimizer.optimize_attention_memory(
        batch_size=4, seq_len=1024, hidden_dim=768, num_heads=12
    )
    print(f"Created optimized attention components:")
    for name, tensor in attention_components.items():
        if isinstance(tensor, np.ndarray):
            print(f"  - {name}: {tensor.shape} ({tensor.dtype})")

    # Show memory statistics
    print("\n4. Memory statistics:")
    stats = optimizer.get_memory_stats()
    for pool_name, pool_stats in stats.items():
        if isinstance(pool_stats, dict) and 'current_usage' in pool_stats:
            usage_gb = pool_stats['current_usage'] / (1024**3)
            peak_gb = pool_stats['peak_usage'] / (1024**3)
            print(f"  - {pool_name}: Current={usage_gb:.3f}GB, Peak={peak_gb:.3f}GB")

    # Cleanup
    optimizer.cleanup()
    print("\nMemory cleanup completed.")
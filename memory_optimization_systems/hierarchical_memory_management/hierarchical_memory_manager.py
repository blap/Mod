"""
Hierarchical Memory Management System for Qwen3-VL

This module implements a hierarchical memory management system that manages different levels
of memory (CPU RAM, GPU VRAM, NVMe SSD) with intelligent movement policies and access prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import threading
import time
import os
import pickle
from collections import OrderedDict
import logging
import psutil
from dataclasses import dataclass, field


class MemoryTier(Enum):
    """Enumeration for different memory tiers"""
    CPU_RAM = "cpu_ram"
    GPU_VRAM = "gpu_vram"
    NVME_SSD = "nvme_ssd"


@dataclass
class MemoryBlock:
    """Represents a memory block with its properties"""
    id: str
    size_bytes: int
    tensor_shape: Tuple[int, ...]
    tensor_dtype: torch.dtype
    tier: MemoryTier
    last_accessed: float
    access_count: int = 0
    predicted_ttl: float = float('inf')  # Time-to-live prediction
    pinned: bool = False  # Whether this block should not be moved
    tensor_ref: Optional[torch.Tensor] = None  # Reference to actual tensor if in memory


class HierarchicalMemoryManager:
    """Main class for hierarchical memory management"""

    def __init__(self, 
                 cpu_memory_limit: int = 4 * 1024 * 1024 * 1024,  # 4GB
                 gpu_memory_limit: int = 6 * 1024 * 1024 * 1024,  # 6GB (appropriate for SM61)
                 disk_cache_path: str = './disk_cache',
                 lru_cache_size: int = 1000):
        """
        Initialize hierarchical memory manager
        
        Args:
            cpu_memory_limit: Maximum memory to use on CPU
            gpu_memory_limit: Maximum memory to use on GPU
            disk_cache_path: Path for disk-based caching
            lru_cache_size: Size of LRU cache
        """
        self.cpu_memory_limit = cpu_memory_limit
        self.gpu_memory_limit = gpu_memory_limit
        self.disk_cache_path = disk_cache_path
        self.lru_cache_size = lru_cache_size
        
        # Create disk cache directory
        os.makedirs(disk_cache_path, exist_ok=True)
        
        # Memory block tracking
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        self.tier_usage: Dict[MemoryTier, int] = {
            MemoryTier.CPU_RAM: 0,
            MemoryTier.GPU_VRAM: 0,
            MemoryTier.NVME_SSD: 0
        }
        
        # LRU caches for each tier
        self.lru_caches: Dict[MemoryTier, OrderedDict] = {
            MemoryTier.CPU_RAM: OrderedDict(),
            MemoryTier.GPU_VRAM: OrderedDict(),
            MemoryTier.NVME_SSD: OrderedDict()
        }
        
        # Access prediction model
        self.access_prediction_model = AccessPredictionModel()
        
        # Thread lock for thread safety
        self.lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Hardware optimization settings
        self._configure_hardware_optimizations()

    def _configure_hardware_optimizations(self):
        """Configure optimizations based on hardware capabilities"""
        # For Intel i5-10210U (4 cores, 8 threads)
        self.cpu_cores = 8  # Use logical cores
        self.l3_cache_size = 6 * 1024 * 1024  # 6MB L3 cache on i5-10210U
        
        # For NVIDIA SM61
        self.gpu_warp_size = 32  # Standard CUDA warp size
        self.sm_count = 3  # Approximate for SM61-class GPU
        self.warp_allocation_unit = self.gpu_warp_size * 4  # 4 bytes per float

    def allocate_tensor(self, 
                       shape: Tuple[int, ...], 
                       dtype: torch.dtype,
                       device: Optional[Union[str, torch.device]] = None,
                       pinned: bool = False) -> Tuple[str, torch.Tensor]:
        """
        Allocate a tensor and assign it to an optimal memory tier
        
        Args:
            shape: Shape of the tensor
            dtype: Data type of the tensor
            device: Preferred device (if None, system decides)
            pinned: Whether this tensor should be pinned (not moved between tiers)
        
        Returns:
            Tuple of (block_id, tensor)
        """
        with self.lock:
            block_id = f"tensor_{int(time.time() * 1000000)}"
            size_bytes = self._calculate_tensor_size(shape, dtype)
            
            # Determine optimal tier based on tensor characteristics
            optimal_tier = self._determine_optimal_tier(shape, dtype, device)
            
            # Create the tensor in the optimal tier
            tensor = self._create_tensor_in_tier(shape, dtype, optimal_tier)
            
            # Track the memory block
            memory_block = MemoryBlock(
                id=block_id,
                size_bytes=size_bytes,
                tensor_shape=shape,
                tensor_dtype=dtype,
                tier=optimal_tier,
                last_accessed=time.time(),
                access_count=1,
                pinned=pinned,
                tensor_ref=tensor
            )
            
            self.memory_blocks[block_id] = memory_block
            self.tier_usage[optimal_tier] += size_bytes
            self._update_lru_cache(optimal_tier, block_id)
            
            self.logger.info(f"Allocated tensor {block_id} on {optimal_tier.value} ({size_bytes / 1024**2:.2f} MB)")
            
            return block_id, tensor

    def get_tensor(self, block_id: str) -> torch.Tensor:
        """
        Retrieve a tensor, moving it to fast tier if needed
        
        Args:
            block_id: ID of the tensor block to retrieve
        
        Returns:
            The tensor
        """
        with self.lock:
            if block_id not in self.memory_blocks:
                raise KeyError(f"Tensor block {block_id} not found")
            
            block = self.memory_blocks[block_id]
            block.access_count += 1
            block.last_accessed = time.time()
            
            # If tensor is not in memory, load it from storage
            if block.tensor_ref is None:
                block.tensor_ref = self._load_tensor_from_tier(block)
            
            # Update LRU cache
            self._update_lru_cache(block.tier, block_id)
            
            # Move to faster tier if it's frequently accessed and not pinned
            if not block.pinned:
                self._consider_tier_upgrade(block_id)
            
            return block.tensor_ref

    def release_tensor(self, block_id: str) -> bool:
        """
        Release a tensor and allow it to be moved or deallocated
        
        Args:
            block_id: ID of the tensor block to release
        
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if block_id not in self.memory_blocks:
                return False
            
            block = self.memory_blocks[block_id]
            
            # Only release if not pinned
            if not block.pinned:
                # If in GPU/CPU memory, consider moving to disk for later reuse
                if block.tier in [MemoryTier.CPU_RAM, MemoryTier.GPU_VRAM]:
                    self._consider_tier_downgrade(block_id)
                
                # Remove from LRU cache
                if block_id in self.lru_caches[block.tier]:
                    del self.lru_caches[block.tier][block_id]
                
                # Decrement usage counter
                self.tier_usage[block.tier] -= block.size_bytes
                
                # Clear tensor reference to free memory
                block.tensor_ref = None
                
                self.logger.info(f"Released tensor {block_id} from {block.tier.value}")
                return True
            
            return False

    def move_tensor_to_tier(self, block_id: str, target_tier: MemoryTier) -> bool:
        """
        Move a tensor to a specific memory tier
        
        Args:
            block_id: ID of the tensor block to move
            target_tier: Target memory tier
        
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if block_id not in self.memory_blocks:
                return False
            
            block = self.memory_blocks[block_id]
            if block.pinned:
                return False  # Cannot move pinned blocks
            
            # Get tensor from current tier
            current_tensor = self.get_tensor(block_id)
            
            # Remove from current tier
            self.tier_usage[block.tier] -= block.size_bytes
            if block_id in self.lru_caches[block.tier]:
                del self.lru_caches[block.tier][block_id]
            
            # Move to target tier
            if target_tier != MemoryTier.NVME_SSD:
                # For CPU/GPU, create a new tensor in target device
                target_device = torch.device('cuda' if target_tier == MemoryTier.GPU_VRAM else 'cpu')
                new_tensor = current_tensor.to(target_device)
                block.tensor_ref = new_tensor
            else:
                # For disk, serialize tensor to file
                file_path = os.path.join(self.disk_cache_path, f"{block_id}.pkl")
                with open(file_path, 'wb') as f:
                    pickle.dump(current_tensor.cpu().numpy(), f)
                block.tensor_ref = None
                # Remove from memory
                del current_tensor
            
            # Update block info
            prev_tier = block.tier
            block.tier = target_tier
            self.tier_usage[target_tier] += block.size_bytes
            self._update_lru_cache(target_tier, block_id)
            
            self.logger.info(f"Moved tensor {block_id} from {prev_tier.value} to {target_tier.value}")
            return True

    def _determine_optimal_tier(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                                preferred_device: Optional[Union[str, torch.device]]) -> MemoryTier:
        """Determine the optimal memory tier for a tensor"""
        size_bytes = self._calculate_tensor_size(shape, dtype)
        
        # Check for preferred device
        if preferred_device is not None:
            device_str = str(preferred_device)
            if 'cuda' in device_str or 'gpu' in device_str.lower():
                if self._check_memory_availability(MemoryTier.GPU_VRAM, size_bytes):
                    return MemoryTier.GPU_VRAM
            elif 'cpu' in device_str.lower():
                if self._check_memory_availability(MemoryTier.CPU_RAM, size_bytes):
                    return MemoryTier.CPU_RAM
        
        # For small, frequently accessed tensors (like attention weights), prefer GPU
        if len(shape) == 2 and shape[0] < 1024 and shape[1] < 1024:
            if self._check_memory_availability(MemoryTier.GPU_VRAM, size_bytes):
                return MemoryTier.GPU_VRAM
            elif self._check_memory_availability(MemoryTier.CPU_RAM, size_bytes):
                return MemoryTier.CPU_RAM
        
        # For large tensors, consider tier based on size
        if size_bytes > 1024 * 1024 * 500:  # 500MB+
            # Too large for GPU, put on CPU or disk
            if self._check_memory_availability(MemoryTier.CPU_RAM, size_bytes):
                return MemoryTier.CPU_RAM
            else:
                return MemoryTier.NVME_SSD
        elif size_bytes > 1024 * 1024:  # 1MB+
            # Could go on GPU if available, else CPU
            if self._check_memory_availability(MemoryTier.GPU_VRAM, size_bytes):
                return MemoryTier.GPU_VRAM
            elif self._check_memory_availability(MemoryTier.CPU_RAM, size_bytes):
                return MemoryTier.CPU_RAM
            else:
                return MemoryTier.NVME_SSD
        else:
            # Small tensors - prefer GPU for performance
            if self._check_memory_availability(MemoryTier.GPU_VRAM, size_bytes):
                return MemoryTier.GPU_VRAM
            else:
                return MemoryTier.CPU_RAM

    def _check_memory_availability(self, tier: MemoryTier, size_bytes: int) -> bool:
        """Check if there's enough space in a memory tier"""
        current_usage = self.tier_usage[tier]
        
        if tier == MemoryTier.CPU_RAM:
            return current_usage + size_bytes < self.cpu_memory_limit
        elif tier == MemoryTier.GPU_VRAM:
            if torch.cuda.is_available():
                max_allowed = min(
                    self.gpu_memory_limit,
                    torch.cuda.get_device_properties(0).total_memory * 0.9  # Leave 10% buffer
                )
                return current_usage + size_bytes < max_allowed
            else:
                return False
        else:  # NVME_SSD - assume plenty of space on disk
            return True

    def _create_tensor_in_tier(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                              tier: MemoryTier) -> torch.Tensor:
        """Create a tensor in the specified memory tier"""
        if tier == MemoryTier.GPU_VRAM and torch.cuda.is_available():
            device = torch.device('cuda')
        elif tier == MemoryTier.CPU_RAM:
            device = torch.device('cpu')
        else:  # NVME_SSD - create in CPU memory, will be moved to disk later
            device = torch.device('cpu')
        
        tensor = torch.empty(shape, dtype=dtype, device=device)
        return tensor

    def _load_tensor_from_tier(self, block: MemoryBlock) -> torch.Tensor:
        """Load a tensor from its current tier (including from disk)"""
        if block.tier == MemoryTier.NVME_SSD:
            # Load from disk
            file_path = os.path.join(self.disk_cache_path, f"{block.id}.pkl")
            with open(file_path, 'rb') as f:
                numpy_data = pickle.load(f)
            tensor = torch.from_numpy(numpy_data).to(block.tensor_dtype)
            
            # Decide where to load it based on size
            if block.size_bytes > 1024 * 1024 * 100:  # 100MB+, prefer CPU
                return tensor.to('cpu')
            else:
                if torch.cuda.is_available() and self._check_memory_availability(MemoryTier.GPU_VRAM, block.size_bytes):
                    return tensor.to('cuda')
                else:
                    return tensor.to('cpu')
        else:
            # Tensor should already be in memory, this shouldn't happen under normal operation
            if block.tensor_ref is not None:
                return block.tensor_ref
            else:
                # Recreate the tensor if somehow lost
                if block.tier == MemoryTier.GPU_VRAM and torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')
                return torch.empty(block.tensor_shape, dtype=block.tensor_dtype, device=device)

    def _update_lru_cache(self, tier: MemoryTier, block_id: str):
        """Update the LRU cache for a memory tier"""
        if block_id in self.lru_caches[tier]:
            # Move to end (most recently used)
            self.lru_caches[tier].move_to_end(block_id)
        else:
            # Add to end
            self.lru_caches[tier][block_id] = True
        
        # Trim if cache is too large
        while len(self.lru_caches[tier]) > self.lru_cache_size:
            oldest_id, _ = self.lru_caches[tier].popitem(last=False)
            # Consider moving the oldest item to a lower tier
            if not self.memory_blocks[oldest_id].pinned:
                self._consider_tier_downgrade(oldest_id)

    def _consider_tier_upgrade(self, block_id: str):
        """Consider upgrading a tensor to a faster memory tier"""
        with self.lock:
            if block_id not in self.memory_blocks:
                return
            
            block = self.memory_blocks[block_id]
            
            # Only upgrade if not already at fastest tier and not pinned
            if block.tier == MemoryTier.GPU_VRAM or block.pinned:
                return
            
            # Upgrade if frequently accessed and small enough
            if (block.access_count > 5 and 
                block.size_bytes < 100 * 1024 * 1024 and  # Less than 100MB
                self._check_memory_availability(MemoryTier.GPU_VRAM, block.size_bytes)):
                
                # Check if the tensor is likely to be accessed again soon
                time_since_access = time.time() - block.last_accessed
                predicted_access = self.access_prediction_model.predict_access_time(block_id)
                
                if (time_since_access < 5.0 or  # Accessed recently
                    predicted_access is not None and predicted_access < 5.0):  # Predicted soon
                    
                    self.move_tensor_to_tier(block_id, MemoryTier.GPU_VRAM)

    def _consider_tier_downgrade(self, block_id: str):
        """Consider downgrading a tensor to a slower memory tier"""
        with self.lock:
            if block_id not in self.memory_blocks:
                return
            
            block = self.memory_blocks[block_id]
            
            # Don't downgrade GPU tensors if they're being actively used in GPU computation
            if block.tier == MemoryTier.CPU_RAM:
                # Consider moving to disk if infrequently accessed
                if (block.access_count < 2 and 
                    time.time() - block.last_accessed > 30.0):  # Not accessed in 30 seconds
                    self.move_tensor_to_tier(block_id, MemoryTier.NVME_SSD)
                    
            elif block.tier == MemoryTier.GPU_VRAM:
                # Move to CPU if infrequently accessed
                if (block.access_count < 3 and 
                    time.time() - block.last_accessed > 10.0):  # Not accessed in 10 seconds
                    self.move_tensor_to_tier(block_id, MemoryTier.CPU_RAM)

    def _calculate_tensor_size(self, shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Calculate the size of a tensor in bytes"""
        elements = 1
        for dim in shape:
            elements *= dim
        
        if dtype == torch.float32:
            return elements * 4
        elif dtype == torch.float16:
            return elements * 2
        elif dtype == torch.bfloat16:
            return elements * 2
        elif dtype == torch.int64:
            return elements * 8
        elif dtype == torch.int32:
            return elements * 4
        elif dtype == torch.int16:
            return elements * 2
        elif dtype == torch.int8:
            return elements * 1
        elif dtype == torch.bool:
            return elements * 1  # In PyTorch, bool tensors use 1 byte per element
        else:
            # Default to 4 bytes per element
            return elements * 4

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        with self.lock:
            total_tensors = len(self.memory_blocks)
            gpu_tensors = len([b for b in self.memory_blocks.values() if b.tier == MemoryTier.GPU_VRAM])
            cpu_tensors = len([b for b in self.memory_blocks.values() if b.tier == MemoryTier.CPU_RAM])
            disk_tensors = len([b for b in self.memory_blocks.values() if b.tier == MemoryTier.NVME_SSD])
            
            total_memory_used = sum(self.tier_usage.values())
            
            return {
                'total_tensors': total_tensors,
                'gpu_tensors': gpu_tensors,
                'cpu_tensors': cpu_tensors,
                'disk_tensors': disk_tensors,
                'tier_usage_bytes': {
                    tier.value: usage for tier, usage in self.tier_usage.items()
                },
                'total_memory_used_bytes': total_memory_used,
                'cpu_memory_utilization': self.tier_usage[MemoryTier.CPU_RAM] / self.cpu_memory_limit if self.cpu_memory_limit > 0 else 0,
                'gpu_memory_utilization': self.tier_usage[MemoryTier.GPU_VRAM] / self.gpu_memory_limit if self.gpu_memory_limit > 0 else 0
            }


class AccessPredictionModel:
    """Simple model to predict tensor access patterns"""

    def __init__(self):
        self.access_history: Dict[str, List[float]] = {}  # block_id -> list of access times

    def record_access(self, block_id: str):
        """Record an access to a tensor block"""
        current_time = time.time()
        if block_id not in self.access_history:
            self.access_history[block_id] = []
        self.access_history[block_id].append(current_time)
        
        # Keep only recent history (last 100 accesses)
        if len(self.access_history[block_id]) > 100:
            self.access_history[block_id] = self.access_history[block_id][-100:]

    def predict_access_time(self, block_id: str) -> Optional[float]:
        """Predict when a tensor will be accessed next (in seconds from now)"""
        if block_id not in self.access_history or len(self.access_history[block_id]) < 2:
            return None  # Not enough history to predict
        
        # Calculate average interval between accesses
        access_times = self.access_history[block_id]
        intervals = []
        for i in range(1, len(access_times)):
            intervals.append(access_times[i] - access_times[i-1])
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            return avg_interval
        else:
            return None


# Example usage and testing
if __name__ == "__main__":
    # Create memory manager
    manager = HierarchicalMemoryManager(
        cpu_memory_limit=2 * 1024 * 1024 * 1024,  # 2GB
        gpu_memory_limit=4 * 1024 * 1024 * 1024,  # 4GB
        disk_cache_path='./test_disk_cache'
    )
    
    print("Testing Hierarchical Memory Management System...")
    
    # Test allocation
    print("\n1. Allocating tensors...")
    block_ids = []
    for i in range(5):
        shape = (100, 100) if i < 3 else (500, 500)  # Mix of small and large tensors
        dtype = torch.float32
        block_id, tensor = manager.allocate_tensor(shape, dtype)
        block_ids.append(block_id)
        print(f"   Allocated tensor {block_id} with shape {shape}")
    
    # Test access patterns
    print("\n2. Accessing tensors...")
    for i in range(2):
        for block_id in block_ids[:3]:  # Frequently access first 3 tensors
            tensor = manager.get_tensor(block_id)
            print(f"   Accessed tensor {block_id}, shape: {tensor.shape}")
    
    # Test memory stats
    print("\n3. Memory stats:")
    stats = manager.get_memory_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test tensor release
    print("\n4. Releasing tensors...")
    for block_id in block_ids:
        success = manager.release_tensor(block_id)
        print(f"   Released tensor {block_id}: {success}")
    
    print("\nHierarchical Memory Management System test completed successfully!")
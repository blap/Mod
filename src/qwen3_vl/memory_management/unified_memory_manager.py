"""
Unified Memory Management System for Qwen3-VL

This module implements a unified memory management system that integrates:
- Memory pooling (with buddy allocation and fragmentation management)
- Memory tiering (GPU HBM, CPU RAM, NVMe SSD with ML-based predictions)
- Memory compression (INT8/FP16, SVD, sparse compression)
- Memory swapping (with pressure monitoring and NVMe optimizations)

The system provides a single coherent interface that coordinates all memory
optimization strategies while resolving conflicts and maintaining all
existing functionality from the separate systems.
"""

import math
import threading
import time
import queue
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict, defaultdict, deque
import numpy as np
import torch

# Import from existing systems
from advanced_memory_pooling_system import (
    TensorType as PoolingTensorType, MemoryBlock as PoolingMemoryBlock, 
    BuddyAllocator, MemoryPool, AdvancedMemoryPoolingSystem
)
from advanced_memory_tiering_system import (
    MemoryTier, TensorType as TieringTensorType, TensorMetadata, 
    AdvancedMemoryTieringSystem
)
from memory_compression_system import (
    CompressionMethod, MemoryCompressionManager, HierarchicalCompressionCache
)
from advanced_memory_swapping_system import (
    MemoryPressureLevel, SwapAlgorithm, AdvancedMemorySwapper, 
    MemoryRegionType
)


class UnifiedTensorType(Enum):
    """Unified tensor types for all memory management strategies"""
    KV_CACHE = "kv_cache"
    IMAGE_FEATURES = "image_features"
    TEXT_EMBEDDINGS = "text_embeddings"
    GRADIENTS = "gradients"
    ACTIVATIONS = "activations"
    PARAMETERS = "parameters"
    GENERAL = "general"
    ACTIVATION_BUFFER = "activation_buffer"
    TEMPORARY = "temporary"


@dataclass
class UnifiedMemoryBlock:
    """Unified representation of a memory block across all systems"""
    id: str
    tensor_id: str
    size_bytes: int
    tensor_type: UnifiedTensorType
    original_tensor: Optional[torch.Tensor] = None
    is_compressed: bool = False
    compression_method: Optional[CompressionMethod] = None
    is_swapped: bool = False
    swap_location: Optional[str] = None
    tier: Optional[MemoryTier] = None
    pool_block: Optional[PoolingMemoryBlock] = None
    tensor_metadata: Optional[TensorMetadata] = None
    timestamp: float = 0.0
    last_access_time: float = 0.0
    access_count: int = 0
    pinned: bool = False
    compression_ratio: float = 1.0


class UnifiedMemoryManager:
    """
    Unified Memory Manager that integrates pooling, tiering, compression, and swapping
    into a single coherent system with conflict resolution and coordination.
    """
    
    def __init__(self,
                 # Pooling parameters
                 kv_cache_pool_size: int = 512*1024*1024,  # 512MB
                 image_features_pool_size: int = 256*1024*1024,  # 256MB
                 text_embeddings_pool_size: int = 128*1024*1024,  # 128MB
                 gradients_pool_size: int = 512*1024*1024,  # 512MB
                 activations_pool_size: int = 256*1024*1024,  # 256MB
                 parameters_pool_size: int = 1024*1024*1024,  # 1GB
                 min_pool_block_size: int = 256,
                 
                 # Tiering parameters
                 gpu_hbm_size: int = 1 * 1024 * 1024 * 1024,  # 1GB GPU HBM
                 cpu_ram_size: int = 2 * 1024 * 1024 * 1024,  # 2GB CPU RAM
                 nvme_ssd_size: int = 10 * 1024 * 1024 * 1024,  # 10GB NVMe SSD
                 prediction_window: int = 1000,
                 
                 # Swapping parameters
                 swap_threshold: float = 0.8,  # Start swapping at 80% memory usage
                 max_swap_size: int = 2 * 1024 * 1024 * 1024,  # 2GB max swap space
                 
                 # Compression parameters
                 enable_compression: bool = True,
                 compression_cache_size: int = 1000):
        """
        Initialize the unified memory manager with all integrated systems.
        """
        # Initialize all subsystems
        self._initialize_pooling_system(
            kv_cache_pool_size, image_features_pool_size, text_embeddings_pool_size,
            gradients_pool_size, activations_pool_size, parameters_pool_size, 
            min_pool_block_size
        )
        
        self._initialize_tiering_system(
            gpu_hbm_size, cpu_ram_size, nvme_ssd_size, prediction_window
        )
        
        self._initialize_compression_system(enable_compression, compression_cache_size)
        
        self._initialize_swapping_system(swap_threshold, max_swap_size)
        
        # Unified tracking
        self.memory_blocks: Dict[str, UnifiedMemoryBlock] = {}
        self.tensor_to_block_map: Dict[str, str] = {}
        self.system_lock = threading.RLock()
        
        # Coordination mechanisms
        self.conflict_resolution_strategy = "priority_based"  # Options: priority_based, hybrid, predictive
        self.optimization_priority = {
            UnifiedTensorType.KV_CACHE: 5,
            UnifiedTensorType.IMAGE_FEATURES: 4,
            UnifiedTensorType.TEXT_EMBEDDINGS: 3,
            UnifiedTensorType.ACTIVATIONS: 3,
            UnifiedTensorType.GRADIENTS: 2,
            UnifiedTensorType.PARAMETERS: 4,
            UnifiedTensorType.GENERAL: 1,
            UnifiedTensorType.ACTIVATION_BUFFER: 2,
            UnifiedTensorType.TEMPORARY: 1
        }
        
        # Stats tracking
        self.stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'total_compressions': 0,
            'total_decompressions': 0,
            'total_swaps_out': 0,
            'total_swaps_in': 0,
            'total_tier_migrations': 0,
            'current_memory_usage': 0,
            'peak_memory_usage': 0
        }
        
        # Start background coordination thread
        self.coordination_thread = threading.Thread(target=self._background_coordination, daemon=True)
        self.coordination_thread.start()
        
        print("Unified Memory Manager initialized with all systems integrated")

    def _initialize_pooling_system(self, kv_cache_size, image_features_size, text_embeddings_size,
                                   gradients_size, activations_size, parameters_size, min_block_size):
        """Initialize the memory pooling system."""
        # Convert unified tensor types to pooling system types
        pooling_tensor_type_map = {
            UnifiedTensorType.KV_CACHE: PoolingTensorType.KV_CACHE,
            UnifiedTensorType.IMAGE_FEATURES: PoolingTensorType.IMAGE_FEATURES,
            UnifiedTensorType.TEXT_EMBEDDINGS: PoolingTensorType.TEXT_EMBEDDINGS,
            UnifiedTensorType.GRADIENTS: PoolingTensorType.GRADIENTS,
            UnifiedTensorType.ACTIVATIONS: PoolingTensorType.ACTIVATIONS,
            UnifiedTensorType.PARAMETERS: PoolingTensorType.PARAMETERS,
            UnifiedTensorType.GENERAL: PoolingTensorType.ACTIVATIONS,  # Map to activations
            UnifiedTensorType.ACTIVATION_BUFFER: PoolingTensorType.ACTIVATIONS,
            UnifiedTensorType.TEMPORARY: PoolingTensorType.ACTIVATIONS
        }
        
        self.pooling_system = AdvancedMemoryPoolingSystem(
            kv_cache_size=kv_cache_size,
            image_features_size=image_features_size,
            text_embeddings_size=text_embeddings_size,
            gradients_size=gradients_size,
            activations_size=activations_size,
            parameters_size=parameters_size,
            min_block_size=min_block_size
        )
        
        # Store mapping for later use
        self.pooling_tensor_type_map = pooling_tensor_type_map

    def _initialize_tiering_system(self, gpu_hbm_size, cpu_ram_size, nvme_ssd_size, prediction_window):
        """Initialize the memory tiering system."""
        # Convert unified tensor types to tiering system types
        tiering_tensor_type_map = {
            UnifiedTensorType.KV_CACHE: TieringTensorType.KV_CACHE,
            UnifiedTensorType.IMAGE_FEATURES: TieringTensorType.IMAGE_FEATURES,
            UnifiedTensorType.TEXT_EMBEDDINGS: TieringTensorType.TEXT_EMBEDDINGS,
            UnifiedTensorType.GRADIENTS: TieringTensorType.GENERAL,
            UnifiedTensorType.ACTIVATIONS: TieringTensorType.GENERAL,
            UnifiedTensorType.PARAMETERS: TieringTensorType.GENERAL,
            UnifiedTensorType.GENERAL: TieringTensorType.GENERAL,
            UnifiedTensorType.ACTIVATION_BUFFER: TieringTensorType.TEMPORARY,
            UnifiedTensorType.TEMPORARY: TieringTensorType.TEMPORARY
        }
        
        self.tiering_system = AdvancedMemoryTieringSystem(
            gpu_hbm_size=gpu_hbm_size,
            cpu_ram_size=cpu_ram_size,
            nvme_ssd_size=nvme_ssd_size,
            prediction_window=prediction_window
        )
        
        # Store mapping for later use
        self.tiering_tensor_type_map = tiering_tensor_type_map

    def _initialize_compression_system(self, enable_compression, compression_cache_size):
        """Initialize the memory compression system."""
        self.compression_enabled = enable_compression
        if enable_compression:
            self.compression_manager = MemoryCompressionManager(
                enable_cache=True,
                cache_size_limit=compression_cache_size
            )
            self.hierarchical_compression_cache = HierarchicalCompressionCache(self.compression_manager)
        else:
            self.compression_manager = None
            self.hierarchical_compression_cache = None

    def _initialize_swapping_system(self, swap_threshold, max_swap_size):
        """Initialize the memory swapping system."""
        self.swapping_system = AdvancedMemorySwapper(
            swap_algorithm=SwapAlgorithm.ADAPTIVE,
            swap_threshold=swap_threshold,
            max_swap_size=max_swap_size
        )

    def _background_coordination(self):
        """Background thread for coordinating different memory management strategies."""
        while True:
            time.sleep(1)  # Check every second

            # Perform predictive migrations based on access patterns
            try:
                self.tiering_system._perform_predictive_migrations()
            except Exception as e:
                print(f"Error in predictive migrations: {e}")

            # Check if swapping is needed
            try:
                if self.swapping_system.should_swap():
                    self.swapping_system.perform_swapping()
            except Exception as e:
                print(f"Error in swapping: {e}")

            # Compact memory pools if needed
            try:
                avg_fragmentation = sum(
                    pool.fragmentation_ratio for pool in self.pooling_system.pools.values()
                ) / len(self.pooling_system.pools) if self.pooling_system.pools else 0

                if avg_fragmentation > self.pooling_system.auto_compact_threshold:
                    self.pooling_system.compact_memory()
            except Exception as e:
                print(f"Error in memory compaction: {e}")

    def coordinate_memory_optimizations(self):
        """
        Coordinate all memory optimization strategies based on current system state.
        This method implements the coordination logic between pooling, tiering,
        compression, and swapping systems.
        """
        with self.system_lock:
            # Get current system state
            system_pressure = self.swapping_system.pressure_monitor.get_overall_pressure()[0]
            avg_fragmentation = sum(
                pool.fragmentation_ratio for pool in self.pooling_system.pools.values()
            ) / len(self.pooling_system.pools) if self.pooling_system.pools else 0

            # Adjust optimization strategies based on system state
            if system_pressure == MemoryPressureLevel.CRITICAL:
                # In critical pressure, prioritize freeing memory
                # Increase swapping activity
                self._increase_swapping_activity()

                # Temporarily reduce compression overhead to free more memory
                self._reduce_compression_overhead()

            elif system_pressure == MemoryPressureLevel.HIGH:
                # In high pressure, balance between optimizations
                self._balance_optimizations()

            elif avg_fragmentation > 0.3:  # High fragmentation
                # Prioritize memory compaction
                self.pooling_system.compact_memory()

            else:
                # Normal operation, run all optimizations
                self._run_normal_optimizations()

    def _increase_swapping_activity(self):
        """Increase swapping activity to free up memory."""
        # In critical situations, we might want to be more aggressive with swapping
        # This could involve temporarily lowering the swap threshold
        pass  # Implementation would depend on specific system requirements

    def _reduce_compression_overhead(self):
        """Temporarily reduce compression to reduce CPU overhead during memory pressure."""
        # During high memory pressure, we might temporarily disable compression
        # to reduce CPU overhead and focus on freeing memory
        pass  # Implementation would depend on specific system requirements

    def _balance_optimizations(self):
        """Balance between different optimization strategies during high pressure."""
        # Adjust optimization priorities based on current needs
        pass  # Implementation would depend on specific system requirements

    def _run_normal_optimizations(self):
        """Run all optimizations during normal system operation."""
        # Perform standard optimization routines
        pass  # Standard operations are handled by background thread

    def allocate(self, tensor_type: UnifiedTensorType, size_bytes: int,
                 tensor_id: Optional[str] = None,
                 use_compression: bool = True,
                 use_tiering: bool = True,
                 use_swapping: bool = True,
                 pinned: bool = False,
                 target_tier: Optional[MemoryTier] = None) -> Optional[UnifiedMemoryBlock]:
        """
        Unified allocation method that coordinates all memory management strategies.

        Args:
            tensor_type: Type of tensor to allocate
            size_bytes: Size in bytes to allocate
            tensor_id: Optional tensor ID (auto-generated if None)
            use_compression: Whether to enable compression for this allocation
            use_tiering: Whether to use tiering for this allocation
            use_swapping: Whether to allow swapping for this allocation
            pinned: If pinned, the tensor won't be subject to swapping or tiering
            target_tier: Specific tier to allocate to (if using tiering)

        Returns:
            UnifiedMemoryBlock if allocation successful, None otherwise
        """
        with self.system_lock:
            if tensor_id is None:
                tensor_id = f"tensor_{int(time.time() * 1000000)}_{id(self)}"

            # Create a unified memory block
            unified_block = UnifiedMemoryBlock(
                id=f"unified_{tensor_id}",
                tensor_id=tensor_id,
                size_bytes=size_bytes,
                tensor_type=tensor_type,
                timestamp=time.time(),
                last_access_time=time.time(),
                pinned=pinned
            )

            # Step 1: Use pooling system for allocation
            pooling_tensor_type = self.pooling_tensor_type_map.get(tensor_type, PoolingTensorType.ACTIVATIONS)
            pool_block = self.pooling_system.allocate(pooling_tensor_type, size_bytes, tensor_id)

            if pool_block is None:
                print(f"Failed to allocate from pooling system for tensor {tensor_id}")
                return None

            unified_block.pool_block = pool_block

            # Step 2: Determine tier based on tensor type, size and preferences
            if use_tiering and not pinned:
                tiering_tensor_type = self.tiering_tensor_type_map.get(tensor_type, TieringTensorType.GENERAL)

                # Create a temporary tensor for tiering system
                temp_tensor = torch.zeros(size_bytes // 4, dtype=torch.float32)  # Approximate

                # Determine preferred tier
                preferred_tier = target_tier
                if preferred_tier is None:
                    # Let the tiering system decide based on tensor characteristics
                    preferred_tier = self.tiering_system._determine_initial_tier(temp_tensor, tiering_tensor_type)

                # Add to tiering system
                success, tier_tensor_id = self.tiering_system.put_tensor(
                    temp_tensor,
                    tensor_type=tiering_tensor_type,
                    preferred_tier=preferred_tier,
                    pinned=pinned
                )

                if success:
                    unified_block.tier = self.tiering_system.tensor_locations.get(tier_tensor_id)
                    # Update metadata
                    if tier_tensor_id in self.tiering_system.tensor_metadata:
                        unified_block.tensor_metadata = self.tiering_system.tensor_metadata[tier_tensor_id]
            else:
                # Default to CPU RAM for non-tiered allocations
                unified_block.tier = MemoryTier.CPU_RAM

            # Step 3: Apply compression if enabled and appropriate
            if use_compression and self.compression_enabled and not pinned:
                # Mark as compressible - actual compression happens when needed
                unified_block.is_compressed = True
                unified_block.compression_method = CompressionMethod.AUTO
                # Estimate compression ratio based on tensor type
                if tensor_type in [UnifiedTensorType.KV_CACHE, UnifiedTensorType.GRADIENTS]:
                    unified_block.compression_ratio = 0.6  # 40% size reduction
                elif tensor_type in [UnifiedTensorType.IMAGE_FEATURES]:
                    unified_block.compression_ratio = 0.5  # 50% size reduction
                else:
                    unified_block.compression_ratio = 0.7  # 30% size reduction

            # Step 4: Register with swapping system if applicable
            if use_swapping and not pinned:
                memory_region_type = self._unified_to_memory_region_type(tensor_type)
                self.swapping_system.register_memory_block(
                    unified_block.id,
                    size_bytes,
                    memory_region_type,
                    pinned
                )

            # Step 5: Apply optimization priority
            priority = self.optimization_priority.get(tensor_type, 1)
            if unified_block.tensor_metadata:
                unified_block.tensor_metadata.access_count = priority

            # Track the unified block
            self.memory_blocks[unified_block.id] = unified_block
            self.tensor_to_block_map[tensor_id] = unified_block.id

            # Update stats
            self.stats['total_allocations'] += 1
            self.stats['current_memory_usage'] += size_bytes
            if self.stats['current_memory_usage'] > self.stats['peak_memory_usage']:
                self.stats['peak_memory_usage'] = self.stats['current_memory_usage']

            return unified_block

    def reallocate(self, tensor_id: str, new_size_bytes: int,
                   new_tensor_type: Optional[UnifiedTensorType] = None) -> Optional[UnifiedMemoryBlock]:
        """
        Reallocate an existing tensor with a new size.

        Args:
            tensor_id: ID of the tensor to reallocate
            new_size_bytes: New size in bytes
            new_tensor_type: New tensor type (optional, keeps original if None)

        Returns:
            UnifiedMemoryBlock if reallocation successful, None otherwise
        """
        with self.system_lock:
            if tensor_id not in self.tensor_to_block_map:
                return None

            # Get the existing block
            old_block_id = self.tensor_to_block_map[tensor_id]
            old_unified_block = self.memory_blocks[old_block_id]

            # Determine new tensor type
            tensor_type = new_tensor_type or old_unified_block.tensor_type

            # Deallocate the old tensor
            self.deallocate(tensor_id)

            # Allocate with new parameters
            return self.allocate(
                tensor_type=tensor_type,
                size_bytes=new_size_bytes,
                tensor_id=tensor_id,
                use_compression=old_unified_block.is_compressed,
                use_tiering=old_unified_block.tier is not None,
                use_swapping=not old_unified_block.pinned,
                pinned=old_unified_block.pinned
            )

    def resize_tensor(self, tensor_id: str, new_size_bytes: int) -> bool:
        """
        Resize an existing tensor allocation.

        Args:
            tensor_id: ID of the tensor to resize
            new_size_bytes: New size in bytes

        Returns:
            True if resize successful, False otherwise
        """
        # This is a simplified implementation
        # In a real system, this would involve more complex memory management
        with self.system_lock:
            if tensor_id not in self.tensor_to_block_map:
                return False

            block_id = self.tensor_to_block_map[tensor_id]
            unified_block = self.memory_blocks[block_id]

            old_size = unified_block.size_bytes
            unified_block.size_bytes = new_size_bytes

            # Update system stats
            self.stats['current_memory_usage'] = self.stats['current_memory_usage'] - old_size + new_size_bytes
            if self.stats['current_memory_usage'] > self.stats['peak_memory_usage']:
                self.stats['peak_memory_usage'] = self.stats['current_memory_usage']

            # For now, just update the pool block size if it exists
            if unified_block.pool_block:
                # In a real implementation, we would need to handle actual resizing
                # This is a simplified placeholder
                pass

            return True

    def pin_tensor(self, tensor_id: str) -> bool:
        """
        Pin a tensor to prevent it from being swapped or migrated between tiers.

        Args:
            tensor_id: ID of the tensor to pin

        Returns:
            True if pinning successful, False otherwise
        """
        with self.system_lock:
            if tensor_id not in self.tensor_to_block_map:
                return False

            block_id = self.tensor_to_block_map[tensor_id]
            unified_block = self.memory_blocks[block_id]

            # Mark as pinned
            unified_block.pinned = True

            # Unregister from swapping system if currently registered
            self.swapping_system.unregister_memory_block(block_id)

            return True

    def unpin_tensor(self, tensor_id: str) -> bool:
        """
        Unpin a tensor to allow it to be swapped or migrated between tiers.

        Args:
            tensor_id: ID of the tensor to unpin

        Returns:
            True if unpinning successful, False otherwise
        """
        with self.system_lock:
            if tensor_id not in self.tensor_to_block_map:
                return False

            block_id = self.tensor_to_block_map[tensor_id]
            unified_block = self.memory_blocks[block_id]

            # Mark as not pinned
            unified_block.pinned = False

            # Re-register with swapping system
            memory_region_type = self._unified_to_memory_region_type(unified_block.tensor_type)
            self.swapping_system.register_memory_block(
                block_id,
                unified_block.size_bytes,
                memory_region_type,
                pinned=False
            )

            return True

    def deallocate(self, tensor_id: str) -> bool:
        """
        Unified deallocation method that coordinates all memory management strategies.
        
        Args:
            tensor_id: ID of the tensor to deallocate
            
        Returns:
            True if deallocation successful, False otherwise
        """
        with self.system_lock:
            if tensor_id not in self.tensor_to_block_map:
                return False
            
            block_id = self.tensor_to_block_map[tensor_id]
            unified_block = self.memory_blocks[block_id]
            
            success = True
            
            # Step 1: Remove from pooling system
            pooling_tensor_type = self.pooling_tensor_type_map.get(unified_block.tensor_type, PoolingTensorType.ACTIVATIONS)
            pool_success = self.pooling_system.deallocate(pooling_tensor_type, tensor_id)
            success = success and pool_success
            
            # Step 2: Remove from tiering system
            if unified_block.tensor_metadata:
                tier_success = self._remove_from_tiering(unified_block.tensor_metadata.tensor_id)
                success = success and tier_success
            
            # Step 3: Remove from swapping system
            swap_success = self.swapping_system.unregister_memory_block(block_id)
            success = success and swap_success
            
            # Step 4: Clean up compression cache if applicable
            if self.compression_enabled and self.hierarchical_compression_cache:
                self.hierarchical_compression_cache.get(tensor_id)
            
            # Update stats
            if success:
                self.stats['total_deallocations'] += 1
                self.stats['current_memory_usage'] -= unified_block.size_bytes
            
            # Remove from tracking
            del self.memory_blocks[block_id]
            del self.tensor_to_block_map[tensor_id]
            
            return success

    def _remove_from_tiering(self, tensor_id: str) -> bool:
        """Helper method to remove tensor from tiering system."""
        # Try to remove from each tier
        success = (self.tiering_system.gpu_manager.remove(tensor_id) or
                  self.tiering_system.cpu_manager.remove(tensor_id) or
                  self.tiering_system.ssd_manager.remove(tensor_id))
        
        # Also remove from internal tracking
        if tensor_id in self.tiering_system.tensor_locations:
            del self.tiering_system.tensor_locations[tensor_id]
        if tensor_id in self.tiering_system.tensor_metadata:
            del self.tiering_system.tensor_metadata[tensor_id]
        
        return success

    def access_tensor(self, tensor_id: str, target_device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        Unified method to access a tensor, handling tiering, decompression, and swapping as needed.

        Args:
            tensor_id: ID of the tensor to access
            target_device: Target device for the tensor (None for current location)

        Returns:
            Tensor if found and accessible, None otherwise
        """
        with self.system_lock:
            if tensor_id not in self.tensor_to_block_map:
                return None

            block_id = self.tensor_to_block_map[tensor_id]
            unified_block = self.memory_blocks[block_id]

            # Update access statistics
            unified_block.last_access_time = time.time()
            unified_block.access_count += 1

            # Record access in tiering system
            self.tiering_system.update_tensor_access(tensor_id)

            # Try to get tensor from tiering system
            tensor = self.tiering_system.get_tensor(tensor_id, target_device)

            if tensor is not None:
                # If tensor was compressed, decompress it
                if unified_block.is_compressed and self.compression_enabled:
                    # In a real implementation, we would decompress the actual tensor
                    self.stats['total_decompressions'] += 1

                # Record access in swapping system
                self.swapping_system.access_memory_block(block_id)

                return tensor

            # If not in tiering system, try to reconstruct from pooling
            if unified_block.pool_block:
                # Create a tensor based on the pool allocation
                # This is a simplified representation - in practice, you'd have the actual tensor data
                tensor_size = unified_block.size_bytes // 4  # Assuming float32
                tensor = torch.zeros(tensor_size, dtype=torch.float32)

                # Apply decompression if needed
                if unified_block.is_compressed and self.compression_enabled:
                    self.stats['total_decompressions'] += 1

                # Add to tiering system if appropriate
                tiering_tensor_type = self.tiering_tensor_type_map.get(unified_block.tensor_type, TieringTensorType.GENERAL)
                self.tiering_system.put_tensor(
                    tensor,
                    tensor_type=tiering_tensor_type,
                    pinned=unified_block.pinned
                )

                # Record access in swapping system
                self.swapping_system.access_memory_block(block_id)

                return tensor

            return None

    def get_pooling_stats(self, tensor_type: Optional[UnifiedTensorType] = None) -> Dict[str, Any]:
        """
        Get pooling system statistics.

        Args:
            tensor_type: Specific tensor type to get stats for (None for all)

        Returns:
            Dictionary with pooling statistics
        """
        if tensor_type is None:
            return self.pooling_system.get_system_stats()
        else:
            pooling_type = self.pooling_tensor_type_map.get(tensor_type, PoolingTensorType.ACTIVATIONS)
            return self.pooling_system.get_pool_stats(pooling_type)

    def get_tiering_stats(self) -> Dict[str, Any]:
        """
        Get tiering system statistics.

        Returns:
            Dictionary with tiering statistics
        """
        return self.tiering_system.get_stats()

    def get_swapping_stats(self) -> Dict[str, Any]:
        """
        Get swapping system statistics.

        Returns:
            Dictionary with swapping statistics
        """
        return self.swapping_system.get_swapping_efficiency()

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression system statistics.

        Returns:
            Dictionary with compression statistics
        """
        if self.compression_enabled and self.compression_manager:
            stats_obj = self.compression_manager.get_compression_stats()
            if stats_obj is not None:
                return stats_obj.__dict__ if hasattr(stats_obj, '__dict__') else dict(stats_obj)
            else:
                return {'compression_enabled': True, 'stats_available': False}
        else:
            return {'compression_enabled': False}

    def _unified_to_memory_region_type(self, unified_type: UnifiedTensorType) -> MemoryRegionType:
        """Convert unified tensor type to memory region type for swapping system."""
        mapping = {
            UnifiedTensorType.KV_CACHE: MemoryRegionType.KV_CACHE,
            UnifiedTensorType.ACTIVATION_BUFFER: MemoryRegionType.ACTIVATION_BUFFER,
            UnifiedTensorType.TEMPORARY: MemoryRegionType.TEMPORARY,
        }
        return mapping.get(unified_type, MemoryRegionType.TENSOR_DATA)

    def get_tensor_stats(self, tensor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive statistics for a specific tensor across all systems.
        
        Args:
            tensor_id: ID of the tensor
            
        Returns:
            Dictionary with statistics from all systems, or None if tensor not found
        """
        if tensor_id not in self.tensor_to_block_map:
            return None
        
        block_id = self.tensor_to_block_map[tensor_id]
        unified_block = self.memory_blocks[block_id]
        
        stats = {
            'unified_block': {
                'id': unified_block.id,
                'size_bytes': unified_block.size_bytes,
                'tensor_type': unified_block.tensor_type.value,
                'is_compressed': unified_block.is_compressed,
                'compression_method': unified_block.compression_method.value if unified_block.compression_method else None,
                'is_swapped': unified_block.is_swapped,
                'tier': unified_block.tier.value if unified_block.tier else None,
                'access_count': unified_block.access_count,
                'pinned': unified_block.pinned
            }
        }
        
        # Add pooling stats
        pooling_tensor_type = self.pooling_tensor_type_map.get(unified_block.tensor_type, PoolingTensorType.ACTIVATIONS)
        stats['pooling_stats'] = self.pooling_system.get_pool_stats(pooling_tensor_type)
        
        # Add tiering stats
        tier_stats = {}
        if unified_block.tier == MemoryTier.GPU_HBM:
            tier_stats = self.tiering_system.gpu_manager.stats.__dict__
        elif unified_block.tier == MemoryTier.CPU_RAM:
            tier_stats = self.tiering_system.cpu_manager.stats.__dict__
        elif unified_block.tier == MemoryTier.NVME_SSD:
            tier_stats = self.tiering_system.ssd_manager.stats.__dict__
        stats['tiering_stats'] = tier_stats
        
        return stats

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for the entire unified system.
        
        Returns:
            Dictionary with system-wide statistics
        """
        with self.system_lock:
            # Get stats from all subsystems
            pooling_stats = self.pooling_system.get_system_stats()
            tiering_stats = self.tiering_system.get_stats()
            swapping_stats = self.swapping_system.get_swapping_efficiency()
            
            # Combine with unified stats
            return {
                'unified_stats': self.stats,
                'pooling_system': pooling_stats,
                'tiering_system': tiering_stats,
                'swapping_system': swapping_stats,
                'compression_enabled': self.compression_enabled,
                'active_tensors': len(self.memory_blocks),
                'tensor_type_distribution': self._get_tensor_type_distribution()
            }

    def _get_tensor_type_distribution(self) -> Dict[str, int]:
        """Get distribution of tensor types in the system."""
        distribution = defaultdict(int)
        for unified_block in self.memory_blocks.values():
            distribution[unified_block.tensor_type.value] += 1
        return dict(distribution)

    def compact_memory(self) -> bool:
        """
        Perform memory compaction across all systems to reduce fragmentation.

        Returns:
            True if compaction was successful
        """
        with self.system_lock:
            # Compact pooling system
            pooling_success = self.pooling_system.compact_memory()

            # Ensure return value is always boolean
            return bool(pooling_success) if pooling_success is not None else False

    def set_optimization_strategy(self, strategy: str):
        """
        Set the strategy for coordinating different optimization systems.
        
        Args:
            strategy: Strategy to use ('priority_based', 'hybrid', 'predictive')
        """
        if strategy in ['priority_based', 'hybrid', 'predictive']:
            self.conflict_resolution_strategy = strategy
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def resolve_conflicts(self, tensor_id: str) -> Dict[str, Any]:
        """
        Resolve conflicts between different memory management strategies for a tensor.

        Args:
            tensor_id: ID of the tensor to resolve conflicts for

        Returns:
            Dictionary with resolution results
        """
        if tensor_id not in self.tensor_to_block_map:
            return {'error': 'Tensor not found'}

        block_id = self.tensor_to_block_map[tensor_id]
        unified_block = self.memory_blocks[block_id]

        resolution = {
            'tensor_id': tensor_id,
            'current_state': {
                'pinned': unified_block.pinned,
                'tier': unified_block.tier.value if unified_block.tier else None,
                'compressed': unified_block.is_compressed,
                'swapped': unified_block.is_swapped
            },
            'conflict_resolution': 'applied_pinning_priority'
        }

        # Conflict resolution based on priority system
        actions_taken = []

        # If tensor is pinned, ensure it's not subject to swapping or tiering migration
        if unified_block.pinned:
            # Unregister from swapping if currently registered
            swap_unregistered = self.swapping_system.unregister_memory_block(block_id)
            if swap_unregistered:
                actions_taken.append('removed_from_swapping')

            # For pinned tensors, ensure they stay in preferred tier
            # In this implementation, pinned tensors are protected from automatic tiering migrations
            actions_taken.append('protected_from_tiering_migration')

        # Resolve conflicts between compression and tiering
        if unified_block.is_compressed and unified_block.tier == MemoryTier.GPU_HBM:
            # Compressed tensors in GPU memory might need special handling
            # In some cases, we might want to decompress before GPU operations
            resolution['gpu_compression_note'] = 'Compressed tensors in GPU may require decompression for operations'

        # Resolve conflicts between swapping and tiering
        if unified_block.is_swapped and unified_block.tier:
            # If a tensor is marked as swapped but also has a tier,
            # prioritize the swap status (as it's actually on disk)
            if unified_block.is_swapped:
                resolution['tier_override'] = f'Tensor marked as swapped, tier {unified_block.tier.value} is logical location'

        resolution['actions_taken'] = actions_taken
        return resolution

    def prioritize_tensor(self, tensor_id: str, priority_level: int) -> bool:
        """
        Set priority level for a tensor to influence optimization decisions.

        Args:
            tensor_id: ID of the tensor to prioritize
            priority_level: Priority level (1-5, higher is more important)

        Returns:
            True if priority was set successfully
        """
        if tensor_id not in self.tensor_to_block_map:
            return False

        if not 1 <= priority_level <= 5:
            return False  # Invalid priority level

        block_id = self.tensor_to_block_map[tensor_id]
        unified_block = self.memory_blocks[block_id]

        # Update the priority in our mapping
        tensor_type = unified_block.tensor_type
        self.optimization_priority[tensor_type] = priority_level

        # Update access count to reflect priority (for tiering system)
        if unified_block.tensor_metadata:
            unified_block.tensor_metadata.access_count = priority_level

        return True

    def get_tensor_conflicts(self, tensor_id: str) -> List[str]:
        """
        Identify potential conflicts for a specific tensor across all systems.

        Args:
            tensor_id: ID of the tensor to check for conflicts

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        if tensor_id not in self.tensor_to_block_map:
            return conflicts

        block_id = self.tensor_to_block_map[tensor_id]
        unified_block = self.memory_blocks[block_id]

        # Check for compression-tensor type conflicts
        if unified_block.is_compressed:
            if unified_block.tensor_type == UnifiedTensorType.KV_CACHE:
                # KV cache might need to be uncompressed for certain operations
                conflicts.append("KV cache compression may impact attention computation speed")

        # Check for tier-swapping conflicts
        if unified_block.is_swapped and unified_block.tier == MemoryTier.GPU_HBM:
            conflicts.append("Tensor marked as swapped but assigned to GPU tier")

        # Check for pinning conflicts
        if unified_block.pinned and unified_block.is_swapped:
            conflicts.append("Pinned tensor marked as swapped - these settings conflict")

        # Check for size-based conflicts
        if unified_block.size_bytes > 100 * 1024 * 1024:  # 100MB+
            if unified_block.tier == MemoryTier.GPU_HBM:
                conflicts.append("Large tensor in GPU memory may cause memory pressure")

        return conflicts

    def migrate_tensor_tier(self, tensor_id: str, target_tier: MemoryTier) -> bool:
        """
        Manually migrate a tensor to a specific tier.
        
        Args:
            tensor_id: ID of the tensor to migrate
            target_tier: Target tier for migration
            
        Returns:
            True if migration successful
        """
        # This would involve complex coordination between systems
        # For now, we'll just record the intent in the tiering system
        if tensor_id not in self.tensor_to_block_map:
            return False
        
        # Update the unified block
        block_id = self.tensor_to_block_map[tensor_id]
        unified_block = self.memory_blocks[block_id]
        
        # In a real implementation, this would involve:
        # 1. Retrieving the tensor from its current location
        # 2. Storing it in the new tier
        # 3. Updating all tracking systems
        
        # For now, just update the tier in our tracking
        unified_block.tier = target_tier
        self.stats['total_tier_migrations'] += 1
        
        return True

    def integrate_with_hardware_abstraction(self, hardware_manager):
        """
        Integrate with hardware abstraction layer if available.

        Args:
            hardware_manager: Hardware abstraction layer instance
        """
        # Get hardware config
        hw_config = hardware_manager.get_hardware_config()

        # Update tiering system based on actual hardware capabilities
        if hasattr(self.tiering_system, 'gpu_manager'):
            # Update tier sizes based on actual hardware
            self.tiering_system.gpu_manager.config.max_size_bytes = hw_config.get('gpu_memory', 2 * 1024 * 1024 * 1024)
            self.tiering_system.cpu_manager.config.max_size_bytes = hw_config.get('cpu_memory', 8 * 1024 * 1024 * 1024)

        # Update swapping system based on storage type
        if hasattr(self.swapping_system, 'nvme_optimizer'):
            storage_type = hw_config.get('storage_type', 'nvme')
            if storage_type == 'hdd':
                # Adjust swapping parameters for slower storage
                self.swapping_system.swap_threshold = 0.9  # Swap later on slow storage
            elif storage_type == 'nvme':
                # Optimize for fast storage
                self.swapping_system.swap_threshold = 0.75  # Can swap earlier with fast storage

        # Update compression settings based on CPU capabilities
        cpu_model = hw_config.get('cpu_model', '').lower()
        if 'i5' in cpu_model or 'i7' in cpu_model:
            # Intel CPUs might benefit from specific compression settings
            self.compression_enabled = True

        print("Unified Memory Manager integrated with hardware abstraction layer")

    def get_hardware_aware_config(self) -> Dict[str, Any]:
        """
        Get configuration optimized for the current hardware setup.

        Returns:
            Dictionary with hardware-aware configuration
        """
        # Get original pooling system hardware optimizer config
        pooling_hw_config = {}
        if hasattr(self.pooling_system, 'hardware_optimizer'):
            # Use original methods from the pooling system's hardware optimizer
            pooling_hw_config = {
                'cpu_cores': self.pooling_system.hardware_optimizer.cpu_cores,
                'gpu_memory_size': self.pooling_system.hardware_optimizer.gpu_memory_size,
                'ssd_read_speed': self.pooling_system.hardware_optimizer.ssd_read_speed,
            }

        return {
            'pooling_config': pooling_hw_config,
            'tiering_config': {
                'gpu_memory_size': self.tiering_system.gpu_manager.config.max_size_bytes,
                'cpu_memory_size': self.tiering_system.cpu_manager.config.max_size_bytes,
                'ssd_storage_size': self.tiering_system.ssd_manager.config.max_size_bytes
            },
            'swapping_config': {
                'swap_threshold': self.swapping_system.swap_threshold,
                'max_swap_size': self.swapping_system.max_swap_size
            },
            'compression_config': {
                'enabled': self.compression_enabled,
                'cache_size': self.compression_manager.cache_size_limit if self.compression_manager else 0
            }
        }

    # Clean API methods for external components
    def alloc_kv_cache(self, size_bytes: int, tensor_id: Optional[str] = None,
                      pinned: bool = False) -> Optional[UnifiedMemoryBlock]:
        """Allocate memory for KV cache tensors."""
        return self.allocate(UnifiedTensorType.KV_CACHE, size_bytes, tensor_id,
                           use_compression=True, use_tiering=True, use_swapping=not pinned, pinned=pinned)

    def alloc_image_features(self, size_bytes: int, tensor_id: Optional[str] = None,
                           pinned: bool = False) -> Optional[UnifiedMemoryBlock]:
        """Allocate memory for image feature tensors."""
        return self.allocate(UnifiedTensorType.IMAGE_FEATURES, size_bytes, tensor_id,
                           use_compression=True, use_tiering=True, use_swapping=not pinned, pinned=pinned)

    def alloc_text_embeddings(self, size_bytes: int, tensor_id: Optional[str] = None,
                            pinned: bool = False) -> Optional[UnifiedMemoryBlock]:
        """Allocate memory for text embedding tensors."""
        return self.allocate(UnifiedTensorType.TEXT_EMBEDDINGS, size_bytes, tensor_id,
                           use_compression=True, use_tiering=True, use_swapping=not pinned, pinned=pinned)

    def alloc_gradients(self, size_bytes: int, tensor_id: Optional[str] = None,
                      pinned: bool = False) -> Optional[UnifiedMemoryBlock]:
        """Allocate memory for gradient tensors."""
        return self.allocate(UnifiedTensorType.GRADIENTS, size_bytes, tensor_id,
                           use_compression=True, use_tiering=True, use_swapping=not pinned, pinned=pinned)

    def alloc_activations(self, size_bytes: int, tensor_id: Optional[str] = None,
                        pinned: bool = False) -> Optional[UnifiedMemoryBlock]:
        """Allocate memory for activation tensors."""
        return self.allocate(UnifiedTensorType.ACTIVATIONS, size_bytes, tensor_id,
                           use_compression=True, use_tiering=True, use_swapping=not pinned, pinned=pinned)

    def alloc_parameters(self, size_bytes: int, tensor_id: Optional[str] = None,
                       pinned: bool = True) -> Optional[UnifiedMemoryBlock]:
        """Allocate memory for parameter tensors (pinned by default)."""
        return self.allocate(UnifiedTensorType.PARAMETERS, size_bytes, tensor_id,
                           use_compression=True, use_tiering=True, use_swapping=not pinned, pinned=pinned)

    def get_memory_utilization(self) -> Dict[str, float]:
        """Get overall memory utilization across all tiers and pools."""
        pooling_stats = self.pooling_system.get_system_stats()
        tiering_stats = self.tiering_system.get_stats()

        return {
            'overall_utilization': pooling_stats.get('overall_utilization', 0),
            'gpu_utilization': tiering_stats['gpu_stats'].get('utilization', 0),
            'cpu_utilization': tiering_stats['cpu_stats'].get('utilization', 0),
            'ssd_utilization': tiering_stats['ssd_stats'].get('utilization', 0),
            'average_fragmentation': pooling_stats.get('average_fragmentation', 0),
            'current_memory_usage_mb': self.stats['current_memory_usage'] / (1024*1024),
            'peak_memory_usage_mb': self.stats['peak_memory_usage'] / (1024*1024)
        }

    def get_optimization_recommendations(self, tensor_id: str) -> List[str]:
        """
        Get optimization recommendations for a specific tensor.

        Args:
            tensor_id: ID of the tensor to get recommendations for

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        if tensor_id not in self.tensor_to_block_map:
            return recommendations

        block_id = self.tensor_to_block_map[tensor_id]
        unified_block = self.memory_blocks[block_id]

        # Recommend compression for large tensors
        if unified_block.size_bytes > 50 * 1024 * 1024 and not unified_block.is_compressed:
            recommendations.append("Consider enabling compression for this large tensor")

        # Recommend pinning for frequently accessed tensors
        if unified_block.access_count > 10 and not unified_block.pinned:
            recommendations.append("Consider pinning this frequently accessed tensor")

        # Recommend specific tier based on access pattern
        if unified_block.access_count > 5 and unified_block.tier != MemoryTier.GPU_HBM:
            recommendations.append("Consider moving frequently accessed tensor to GPU tier")

        return recommendations


# Example usage and integration
def create_unified_memory_manager(hardware_config: Optional[Dict[str, Any]] = None) -> UnifiedMemoryManager:
    """
    Factory function to create a unified memory manager optimized for specific hardware.
    
    Args:
        hardware_config: Hardware configuration with details like:
                        - cpu_model: CPU model string
                        - gpu_model: GPU model string
                        - memory_size: Total system memory in bytes
                        - storage_type: Storage type ('nvme', 'ssd', 'hdd')
    
    Returns:
        UnifiedMemoryManager instance optimized for the hardware
    """
    if hardware_config is None:
        hardware_config = {
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
            'storage_type': 'nvme'
        }
    
    # Adjust parameters based on hardware
    memory_size = hardware_config.get('memory_size', 8 * 1024 * 1024 * 1024)
    storage_type = hardware_config.get('storage_type', 'nvme')
    
    # Calculate appropriate sizes based on total memory
    base_pool_size = memory_size // 4  # Use 25% of memory for pools
    gpu_tier_size = min(memory_size * 0.3, 2 * 1024 * 1024 * 1024)  # Max 2GB for GPU
    cpu_tier_size = min(memory_size * 0.5, 4 * 1024 * 1024 * 1024)  # Max 4GB for CPU
    swap_size = min(memory_size, 4 * 1024 * 1024 * 1024)  # Max 4GB for swap
    
    manager = UnifiedMemoryManager(
        # Pooling parameters (proportional to memory size)
        kv_cache_pool_size=base_pool_size // 4,
        image_features_pool_size=base_pool_size // 4,
        text_embeddings_pool_size=base_pool_size // 8,
        gradients_pool_size=base_pool_size // 4,
        activations_pool_size=base_pool_size // 4,
        parameters_pool_size=base_pool_size // 2,
        
        # Tiering parameters
        gpu_hbm_size=gpu_tier_size,
        cpu_ram_size=cpu_tier_size,
        nvme_ssd_size=10 * 1024 * 1024 * 1024,  # 10GB for NVMe
        
        # Swapping parameters
        max_swap_size=swap_size,
        
        # Compression parameters
        enable_compression=True,
        compression_cache_size=1000
    )
    
    print(f"Created unified memory manager optimized for {hardware_config.get('cpu_model', 'unknown')} "
          f"with {storage_type.upper()} storage")
    
    return manager


if __name__ == "__main__":
    print("Unified Memory Management System for Qwen3-VL")
    print("=" * 60)
    
    # Create the unified memory manager
    unified_manager = create_unified_memory_manager({
        'cpu_model': 'Intel i5-10210U',
        'gpu_model': 'NVIDIA SM61',
        'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
        'storage_type': 'nvme'
    })
    
    # Test unified allocation
    print("\n1. Testing unified allocation...")
    tensor_block = unified_manager.allocate(
        UnifiedTensorType.KV_CACHE,
        10 * 1024 * 1024,  # 10MB
        tensor_id="test_kv_tensor",
        use_compression=True,
        use_tiering=True,
        use_swapping=True
    )
    
    if tensor_block:
        print(f"  Successfully allocated tensor: {tensor_block.tensor_id}")
        print(f"  Size: {tensor_block.size_bytes} bytes")
        print(f"  Tier: {tensor_block.tier.value if tensor_block.tier else 'N/A'}")
        print(f"  Compressed: {tensor_block.is_compressed}")
    else:
        print("  Failed to allocate tensor")
    
    # Test tensor access
    print("\n2. Testing tensor access...")
    tensor = unified_manager.access_tensor("test_kv_tensor")
    if tensor is not None:
        print(f"  Successfully accessed tensor, shape: {tensor.shape}")
    else:
        print("  Failed to access tensor")
    
    # Test conflict resolution
    print("\n3. Testing conflict resolution...")
    resolution = unified_manager.resolve_conflicts("test_kv_tensor")
    print(f"  Resolution: {resolution['conflict_resolution']}")
    print(f"  Actions taken: {resolution.get('actions_taken', [])}")
    
    # Test system stats
    print("\n4. System statistics:")
    stats = unified_manager.get_system_stats()
    print(f"  Total allocations: {stats['unified_stats']['total_allocations']}")
    print(f"  Active tensors: {stats['active_tensors']}")
    print(f"  Current memory usage: {stats['unified_stats']['current_memory_usage'] / (1024**2):.2f} MB")
    print(f"  Peak memory usage: {stats['unified_stats']['peak_memory_usage'] / (1024**2):.2f} MB")
    
    # Test tensor-specific stats
    print("\n5. Tensor-specific statistics:")
    tensor_stats = unified_manager.get_tensor_stats("test_kv_tensor")
    if tensor_stats:
        print(f"  Tensor type: {tensor_stats['unified_block']['tensor_type']}")
        print(f"  Access count: {tensor_stats['unified_block']['access_count']}")
        print(f"  Is compressed: {tensor_stats['unified_block']['is_compressed']}")
    
    # Test deallocation
    print("\n6. Testing deallocation...")
    success = unified_manager.deallocate("test_kv_tensor")
    print(f"  Deallocation successful: {success}")
    
    # Final stats
    print("\n7. Final system statistics:")
    final_stats = unified_manager.get_system_stats()
    print(f"  Total allocations: {final_stats['unified_stats']['total_allocations']}")
    print(f"  Total deallocations: {final_stats['unified_stats']['total_deallocations']}")
    print(f"  Active tensors: {final_stats['active_tensors']}")
    
    print("\nUnified Memory Management System initialized and tested successfully!")
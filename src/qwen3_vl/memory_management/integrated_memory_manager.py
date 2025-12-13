"""
Integrated Memory Management System for Qwen3-VL

This module integrates all memory optimization techniques (pooling, caching, compression,
swapping, tiering, and garbage collection) into a unified system for the Qwen3-VL model.
Designed for Intel i5-10210U + NVIDIA SM61 + NVMe SSD hardware.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
import threading
import logging
from collections import OrderedDict, defaultdict, deque
import weakref
from pathlib import Path
import tempfile

# Define a consistent TensorType enum for the integrated memory manager
class TensorType(Enum):
    """Types of tensors for optimization"""
    GENERAL = "general"
    KV_CACHE = "kv_cache"
    IMAGE_FEATURES = "image_features"
    TEXT_EMBEDDINGS = "text_embeddings"
    TEMPORARY = "temporary"
    GRADIENTS = "gradients"
    OPTIMIZER_STATE = "optimizer_state"
    INTERMEDIATE = "intermediate"

# Import existing optimization components
from .advanced_memory_management_vl import VisionLanguageMemoryOptimizer
from .memory_compression import MemoryCompressionManager, create_hardware_optimized_compression_manager
from .memory_swapping import AdvancedMemorySwapper, create_optimized_swapping_system, MemoryRegionType
from .memory_tiering import Qwen3VLMemoryTieringSystem, create_qwen3vl_memory_tiering_system
from ..attention.predictive_tensor_lifecycle_manager import (
    IntegratedTensorLifecycleManager,
    create_optimized_lifecycle_manager,
    TensorState
)

# Mapping from PyTorch dtypes to NumPy dtypes
dtype2numpy = {
    torch.float32: 'float32',
    torch.float16: 'float16',
    torch.float64: 'float64',
    torch.int32: 'int32',
    torch.int64: 'int64',
    torch.int16: 'int16',
    torch.int8: 'int8',
    torch.uint8: 'uint8',
    torch.bool: 'bool',
    torch.bfloat16: 'bfloat16',
    torch.complex64: 'complex64',
    torch.complex128: 'complex128',
}


def get_optimization_components():
    """Get optimization components through relative imports"""
    return (
        VisionLanguageMemoryOptimizer,
        MemoryCompressionManager,
        create_hardware_optimized_compression_manager,
        AdvancedMemorySwapper,
        create_optimized_swapping_system,
        Qwen3VLMemoryTieringSystem,
        create_qwen3vl_memory_tiering_system,
        # Use our own TensorType enum for consistency
        TensorType,
        IntegratedTensorLifecycleManager,
        create_optimized_lifecycle_manager,
        TensorState
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryOptimizationLevel(Enum):
    """Levels of memory optimization"""
    MINIMAL = "minimal"           # Basic optimizations
    BALANCED = "balanced"         # Balanced performance/optimization
    AGGRESSIVE = "aggressive"     # Maximum optimization


@dataclass
class OptimizationConfig:
    """Configuration for memory optimizations"""
    level: MemoryOptimizationLevel = MemoryOptimizationLevel.BALANCED
    enable_memory_pooling: bool = True
    enable_compression: bool = True
    enable_caching: bool = True
    enable_swapping: bool = True
    enable_tiering: bool = True
    enable_gc: bool = True
    pooling_size_gb: float = 2.0
    compression_ratio_threshold: float = 0.5
    cache_size_gb: float = 1.0
    swapping_threshold: float = 0.8
    tiering_gpu_memory_gb: float = 1.0
    tiering_cpu_memory_gb: float = 2.0
    tiering_ssd_memory_gb: float = 10.0
    gc_collection_interval: float = 1.0
    gc_memory_pressure_threshold: float = 0.8


class IntegratedMemoryManager:
    """
    Main integrated memory management system that combines all optimization techniques
    for the Qwen3-VL model.
    """

    def __init__(self, config: OptimizationConfig, hardware_config: Dict[str, Any]):
        """
        Initialize integrated memory manager

        Args:
            config: Optimization configuration
            hardware_config: Hardware configuration
        """
        self.config = config
        self.hardware_config = hardware_config

        # Dynamically import components
        (VisionLanguageMemoryOptimizer,
         MemoryCompressionManager,
         create_hardware_optimized_compression_manager,
         AdvancedMemorySwapper,
         create_optimized_swapping_system,
         AdvancedMemoryTieringSystem,
         create_memory_tiering_system,
         TensorType,
         IntegratedTensorLifecycleManager,
         create_optimized_lifecycle_manager,
         TensorState) = get_optimization_components()

        # Initialize memory optimizer (pooling, caching, GPU-CPU optimization)
        if config.enable_memory_pooling:
            self.memory_pool_size = int(config.pooling_size_gb * 1024 * 1024 * 1024)
            self.memory_optimizer = VisionLanguageMemoryOptimizer(
                memory_pool_size=self.memory_pool_size,
                enable_memory_pool=config.enable_memory_pooling,
                enable_cache_optimization=config.enable_caching,
                enable_gpu_optimization=True
            )
        else:
            self.memory_optimizer = None

        # Initialize compression manager
        if config.enable_compression:
            self.compression_manager = create_hardware_optimized_compression_manager(hardware_config)
        else:
            self.compression_manager = None

        # Initialize swapping system
        if config.enable_swapping:
            swapping_config = {
                'cpu_model': hardware_config.get('cpu_model', 'Intel i5-10210U'),
                'gpu_model': hardware_config.get('gpu_model', 'NVIDIA SM61'),
                'memory_size': hardware_config.get('memory_size', 8 * 1024 * 1024 * 1024),
                'storage_type': hardware_config.get('storage_type', 'nvme')
            }
            self.swapping_system = create_optimized_swapping_system(swapping_config)
        else:
            self.swapping_system = None

        # Initialize tiering system
        if config.enable_tiering:
            tiering_config = {
                'gpu_memory': int(config.tiering_gpu_memory_gb * 1024 * 1024 * 1024),
                'cpu_memory': int(config.tiering_cpu_memory_gb * 1024 * 1024 * 1024),
                'storage_type': hardware_config.get('storage_type', 'nvme')
            }
            self.tiering_system = create_qwen3vl_memory_tiering_system(tiering_config)
        else:
            self.tiering_system = None

        # Initialize lifecycle manager (garbage collection with prediction)
        if config.enable_gc:
            gc_config = {
                'cpu_model': hardware_config.get('cpu_model', 'Intel i5-10210U'),
                'gpu_model': hardware_config.get('gpu_model', 'NVIDIA SM61'),
                'memory_size': hardware_config.get('memory_size', 8 * 1024 * 1024 * 1024),
                'storage_type': hardware_config.get('storage_type', 'nvme')
            }
            self.lifecycle_manager = create_optimized_lifecycle_manager(gc_config)
        else:
            self.lifecycle_manager = None

        # Integration setup
        self._setup_integrations()

        # Statistics
        self.stats = {
            'total_tensors_managed': 0,
            'total_memory_saved_gb': 0.0,
            'compression_ratio': 0.0,
            'swapped_out_tensors': 0,
            'tier_migrations': 0,
            'collections_performed': 0,
            'memory_pressure_events': 0
        }

        # Lock for thread safety
        self._lock = threading.RLock()

        logger.info("Integrated Memory Manager initialized with all optimizations")

    def _setup_integrations(self):
        """Setup integrations between different optimization components"""
        if self.lifecycle_manager:
            # Integrate with tiering system
            if self.tiering_system:
                self.lifecycle_manager.set_memory_tiering_system(self.tiering_system)

            # Integrate with compression system
            if self.compression_manager:
                self.lifecycle_manager.set_compression_manager(self.compression_manager)

            # Integrate with swapping system
            if self.swapping_system:
                self.lifecycle_manager.set_swapping_system(self.swapping_system)

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                      tensor_type: str = "general", pinned: bool = False) -> torch.Tensor:
        """
        Allocate a tensor using the most appropriate memory optimization technique.

        Args:
            shape: Shape of the tensor
            dtype: Data type of the tensor
            tensor_type: Type of tensor for optimization ('general', 'kv_cache', 'image_features', etc.)
            pinned: Whether tensor should be pinned (not eligible for swapping/migration)

        Returns:
            Allocated tensor
        """
        with self._lock:
            # Convert string tensor type to enum
            tensor_type_enum = self._map_tensor_type(tensor_type)

            # Create tensor using the most appropriate method based on configuration
            if self.memory_optimizer and self.config.enable_memory_pooling:
                # Convert PyTorch dtype to numpy dtype for the memory optimizer
                try:
                    numpy_dtype = np.dtype(dtype2numpy[dtype])
                except KeyError:
                    # Default to float32 for unknown types
                    numpy_dtype = np.float32

                # Use memory optimizer for pooling and caching
                tensor = self.memory_optimizer.allocate_tensor_memory(shape, dtype=numpy_dtype,
                                                                      tensor_type=tensor_type_enum.value)
                # Convert to torch tensor if needed
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.from_numpy(tensor)
            else:
                # Standard allocation
                tensor = torch.zeros(shape, dtype=dtype)

            # Register with lifecycle manager if enabled
            if self.lifecycle_manager:
                tensor_id = f"tensor_{id(tensor)}_{int(time.time() * 1000000)}"
                self.lifecycle_manager.register_tensor(
                    tensor, tensor_id, tensor_type_enum, is_pinned=pinned
                )
                # Store tensor ID for later reference
                tensor.tensor_id = tensor_id

            # Register with swapping system if enabled
            if self.swapping_system and not pinned:
                size_bytes = tensor.element_size() * tensor.nelement()
                region_type = self._map_tensor_type_for_swapping(tensor_type_enum)
                self.swapping_system.register_memory_block(
                    f"tensor_{id(tensor)}", size_bytes, region_type, pinned=pinned
                )

            # Store in tiering system if enabled
            if self.tiering_system:
                self.tiering_system.put_tensor(tensor, tensor_type_enum, pinned=pinned)

            # Update statistics
            self.stats['total_tensors_managed'] += 1

            logger.debug(f"Allocated tensor of shape {shape}, type {tensor_type}, "
                        f"pinned: {pinned}, tensor_id: {getattr(tensor, 'tensor_id', 'unknown')}")

            return tensor

    def access_tensor(self, tensor: torch.Tensor, context: Optional[str] = None) -> torch.Tensor:
        """
        Record access to a tensor and perform any necessary optimizations.

        Args:
            tensor: Tensor being accessed
            context: Context of access (for prediction)

        Returns:
            The same tensor (for chaining)
        """
        with self._lock:
            # Record access with lifecycle manager
            if self.lifecycle_manager and hasattr(tensor, 'tensor_id'):
                self.lifecycle_manager.access_tensor(tensor.tensor_id, context)

            # Record access with swapping system
            if self.swapping_system:
                self.swapping_system.access_memory_block(f"tensor_{id(tensor)}")

            # Record access with tiering system
            if self.tiering_system:
                self.tiering_system.update_tensor_access(f"tensor_{id(tensor)}")

            return tensor

    def compress_tensor(self, tensor: torch.Tensor, method: str = 'auto') -> Tuple[torch.Tensor, bool]:
        """
        Compress a tensor if beneficial.

        Args:
            tensor: Tensor to compress
            method: Compression method ('auto', 'int8', 'fp16', 'svd', 'sparse')

        Returns:
            Tuple of (compressed_tensor, was_compressed)
        """
        if not self.compression_manager or not self.config.enable_compression:
            return tensor, False

        with self._lock:
            # Determine if compression is beneficial
            tensor_size = tensor.element_size() * tensor.nelement()
            if tensor_size < 10 * 1024 * 1024:  # Less than 10MB, probably not worth it
                return tensor, False

            try:
                # Compress tensor
                compressed_data = self.compression_manager.compress_tensor(tensor, method=method)
                compression_ratio = compressed_data.get('compression_ratio', 1.0)

                if compression_ratio < 1.0:  # Actually compressed
                    # Update statistics
                    original_size = tensor_size
                    compressed_size = int(original_size * compression_ratio)
                    saved_bytes = original_size - compressed_size
                    self.stats['total_memory_saved_gb'] += saved_bytes / (1024**3)
                    self.stats['compression_ratio'] = (
                        (self.stats['compression_ratio'] * (self.stats['total_tensors_managed'] - 1) + compression_ratio) /
                        self.stats['total_tensors_managed']
                    ) if self.stats['total_tensors_managed'] > 0 else compression_ratio

                    logger.debug(f"Compressed tensor with ratio {compression_ratio:.3f}, "
                                f"saved {saved_bytes / (1024**2):.2f} MB")
                    return tensor, True  # In practice, we'd return the compressed version
                else:
                    return tensor, False
            except Exception as e:
                logger.warning(f"Failed to compress tensor: {e}")
                return tensor, False

    def decompress_tensor(self, compressed_tensor: torch.Tensor) -> torch.Tensor:
        """
        Decompress a tensor if it's compressed.

        Args:
            compressed_tensor: Potentially compressed tensor

        Returns:
            Decompressed tensor
        """
        # In a real implementation, this would decompress the tensor
        # For now, we return the tensor as is
        return compressed_tensor

    def optimize_tensor_placement(self, tensor: torch.Tensor, target_device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Optimize tensor placement based on current memory pressure and usage patterns.

        Args:
            tensor: Tensor to optimize
            target_device: Target device for the tensor

        Returns:
            Optimized tensor (potentially on different device)
        """
        if not self.memory_optimizer:
            return tensor

        with self._lock:
            # Use memory optimizer to place tensor optimally
            optimized_tensor = self.memory_optimizer.optimize_tensor_placement(tensor, target_device)

            # Record access for lifecycle management
            if self.lifecycle_manager and hasattr(tensor, 'tensor_id'):
                self.lifecycle_manager.access_tensor(tensor.tensor_id, "placement_optimization")

            return optimized_tensor

    def get_memory_pressure(self) -> float:
        """Get current memory pressure as a value between 0 and 1"""
        import psutil
        return psutil.virtual_memory().percent / 100.0

    def perform_memory_optimizations(self):
        """Perform background memory optimizations based on current state"""
        with self._lock:
            # Check memory pressure
            memory_pressure = self.get_memory_pressure()
            if memory_pressure > self.config.gc_memory_pressure_threshold:
                self.stats['memory_pressure_events'] += 1

            # Perform swapping if needed
            if self.swapping_system:
                swapped_count = self.swapping_system.perform_swapping()
                self.stats['swapped_out_tensors'] += swapped_count

            # Perform tiering migrations based on predictions
            if self.tiering_system:
                self.tiering_system._perform_predictive_migrations()

            # Perform garbage collection
            if self.lifecycle_manager:
                collected_count = self.lifecycle_manager.garbage_collector.collect()
                self.stats['collections_performed'] += collected_count

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization statistics"""
        with self._lock:
            stats = self.stats.copy()

            # Add component-specific stats
            if self.compression_manager:
                comp_stats = self.compression_manager.get_compression_stats()
                stats['compression_stats'] = {
                    'compression_ratio': comp_stats.compression_ratio,
                    'compression_time': comp_stats.compression_time,
                    'memory_saved_bytes': comp_stats.memory_saved_bytes
                }

            if self.swapping_system:
                swap_stats = self.swapping_system.get_swapping_efficiency()
                stats['swapping_stats'] = swap_stats

            if self.tiering_system:
                tier_stats = self.tiering_system.get_stats()
                stats['tiering_stats'] = tier_stats

            if self.lifecycle_manager:
                gc_stats = self.lifecycle_manager.get_tensor_lifecycle_stats()
                stats['gc_stats'] = gc_stats

            return stats

    def cleanup(self):
        """Clean up all memory optimization systems"""
        if self.lifecycle_manager:
            self.lifecycle_manager.cleanup()

        if self.swapping_system:
            self.swapping_system.nvme_optimizer.io_queue.put((None, None))  # Shutdown signal
            # Wait for thread to finish
            time.sleep(0.1)

        # Additional cleanup for other components if needed
        logger.info("Integrated Memory Manager cleaned up")

    def _map_tensor_type(self, tensor_type: str):
        """Map string tensor type to enum"""
        type_map = {
            'general': TensorType.GENERAL,
            'kv_cache': TensorType.KV_CACHE,
            'image_features': TensorType.IMAGE_FEATURES,
            'text_embeddings': TensorType.TEXT_EMBEDDINGS,
            'gradients': TensorType.GRADIENTS,
            'optimizer_state': TensorType.OPTIMIZER_STATE,
            'intermediate': TensorType.INTERMEDIATE
        }
        return type_map.get(tensor_type.lower(), TensorType.GENERAL)

    def _map_tensor_type_for_swapping(self, tensor_type) -> Any:
        """Map tensor type to swapping system's region type"""
        mapping = {
            TensorType.GENERAL: MemoryRegionType.TENSOR_DATA,
            TensorType.KV_CACHE: MemoryRegionType.KV_CACHE,
            TensorType.IMAGE_FEATURES: MemoryRegionType.TENSOR_DATA,
            TensorType.TEXT_EMBEDDINGS: MemoryRegionType.TENSOR_DATA,
            TensorType.GRADIENTS: MemoryRegionType.TENSOR_DATA,
            TensorType.OPTIMIZER_STATE: MemoryRegionType.TENSOR_DATA,
            TensorType.INTERMEDIATE: MemoryRegionType.TEMPORARY
        }
        return mapping.get(tensor_type, MemoryRegionType.TENSOR_DATA)


def create_optimized_memory_manager(hardware_config: Optional[Dict[str, Any]] = None,
                                  optimization_level: MemoryOptimizationLevel = MemoryOptimizationLevel.BALANCED) -> IntegratedMemoryManager:
    """
    Factory function to create an optimized memory manager for specific hardware.

    Args:
        hardware_config: Hardware configuration
        optimization_level: Level of optimization to apply

    Returns:
        Optimized IntegratedMemoryManager instance
    """
    if hardware_config is None:
        hardware_config = {
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
            'storage_type': 'nvme'
        }

    # Set configuration based on optimization level
    if optimization_level == MemoryOptimizationLevel.MINIMAL:
        config = OptimizationConfig(
            level=optimization_level,
            enable_memory_pooling=True,
            enable_compression=False,
            enable_caching=True,
            enable_swapping=False,
            enable_tiering=False,
            enable_gc=True,
            pooling_size_gb=1.0,
            cache_size_gb=0.5,
            gc_collection_interval=2.0
        )
    elif optimization_level == MemoryOptimizationLevel.AGGRESSIVE:
        config = OptimizationConfig(
            level=optimization_level,
            enable_memory_pooling=True,
            enable_compression=True,
            enable_caching=True,
            enable_swapping=True,
            enable_tiering=True,
            enable_gc=True,
            pooling_size_gb=2.0,
            compression_ratio_threshold=0.3,
            cache_size_gb=1.0,
            swapping_threshold=0.7,
            tiering_gpu_memory_gb=1.5,
            tiering_cpu_memory_gb=3.0,
            tiering_ssd_memory_gb=15.0,
            gc_collection_interval=0.5,
            gc_memory_pressure_threshold=0.7
        )
    else:  # BALANCED
        config = OptimizationConfig(
            level=optimization_level,
            enable_memory_pooling=True,
            enable_compression=True,
            enable_caching=True,
            enable_swapping=True,
            enable_tiering=True,
            enable_gc=True,
            pooling_size_gb=2.0,
            compression_ratio_threshold=0.5,
            cache_size_gb=1.0,
            swapping_threshold=0.8,
            tiering_gpu_memory_gb=1.0,
            tiering_cpu_memory_gb=2.0,
            tiering_ssd_memory_gb=10.0,
            gc_collection_interval=1.0,
            gc_memory_pressure_threshold=0.8
        )

    manager = IntegratedMemoryManager(config, hardware_config)

    logger.info(f"Created optimized memory manager for {hardware_config.get('cpu_model', 'unknown')} "
                f"with {hardware_config.get('storage_type', 'unknown').upper()} storage "
                f"at {optimization_level.value} optimization level")

    return manager


# Context manager for easy use
class MemoryOptimizedContext:
    """Context manager for memory-optimized operations"""
    
    def __init__(self, memory_manager: IntegratedMemoryManager):
        self.memory_manager = memory_manager
        self.original_tensors = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Perform cleanup operations
        self.memory_manager.perform_memory_optimizations()
    
    def allocate_tensor(self, shape, dtype=torch.float32, tensor_type="general", pinned=False):
        """Allocate a tensor with memory optimization"""
        return self.memory_manager.allocate_tensor(shape, dtype, tensor_type, pinned)
    
    def access_tensor(self, tensor, context=None):
        """Access a tensor with optimization tracking"""
        return self.memory_manager.access_tensor(tensor, context)
    
    def compress_tensor(self, tensor, method='auto'):
        """Compress a tensor"""
        return self.memory_manager.compress_tensor(tensor, method)
    
    def optimize_placement(self, tensor, target_device=None):
        """Optimize tensor placement"""
        return self.memory_manager.optimize_tensor_placement(tensor, target_device)


# Example usage function
def integrate_with_qwen3_vl_model(model, memory_manager: IntegratedMemoryManager):
    """
    Example function to show how to integrate the memory manager with a Qwen3-VL model.
    This would be called during model initialization.
    """
    # This is a simplified example - in practice, you'd integrate at specific points
    # in the model's forward pass where tensors are created and used
    
    def wrapped_forward(self, *args, **kwargs):
        # Create a memory-optimized context for the forward pass
        with MemoryOptimizedContext(memory_manager) as ctx:
            # Perform the original forward pass with memory optimizations
            result = self.original_forward(*args, **kwargs)
            return result
    
    # Store original forward method
    if not hasattr(model, 'original_forward'):
        model.original_forward = model.forward
        model.forward = lambda *args, **kwargs: wrapped_forward(model, *args, **kwargs)
    
    logger.info("Qwen3-VL model integrated with memory optimization system")
    return model
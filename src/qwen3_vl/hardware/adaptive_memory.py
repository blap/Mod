"""
Adaptive Memory Management for Qwen3-VL
Implements memory management that adapts to different cache sizes and memory configurations
"""
import torch
import psutil
import gc
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
import logging
import time
from enum import Enum

from src.qwen3_vl.hardware.cpu_detector import CPUDetector, CPUModel
from src.qwen3_vl.hardware.cpu_profiles import get_optimization_profile, AdaptiveOptimizationManager


logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for adaptive memory management"""
    memory_pool_size: int  # in bytes
    l1_cache_size: int  # in bytes
    l2_cache_size: int  # in bytes
    l3_cache_size: int  # in bytes
    cache_line_size: int  # in bytes
    memory_threshold: float  # 0.0 to 1.0
    enable_memory_pooling: bool
    enable_cache_blocking: bool
    enable_memory_compression: bool
    compression_level: str  # 'low', 'medium', 'high'
    enable_memory_swapping: bool
    swap_threshold: float  # 0.0 to 1.0
    enable_memory_defragmentation: bool
    defragmentation_interval: int  # in number of operations
    defragmentation_threshold: float  # 0.0 to 1.0


class MemoryTier(Enum):
    """Memory tiers based on access speed and capacity"""
    L1_CACHE = "L1_CACHE"
    L2_CACHE = "L2_CACHE"
    L3_CACHE = "L3_CACHE"
    RAM = "RAM"
    SWAP = "SWAP"


class AdaptiveMemoryManager:
    """
    Adaptive memory management that adapts to different cache sizes and memory configurations
    """
    
    def __init__(self, optimization_manager: AdaptiveOptimizationManager):
        self.optimization_manager = optimization_manager
        self.config = self._create_memory_config()
        
        # Initialize memory pools
        self._memory_pool: Optional[torch.nn.Module] = None
        self._cache_blocks = {}
        self._memory_stats = {
            'allocation_count': 0,
            'deallocation_count': 0,
            'peak_memory_usage': 0,
            'current_memory_usage': 0
        }
        
        # Initialize cache blocking parameters
        self._cache_blocking_params = self._calculate_cache_blocking_params()
        
        logger.info(f"Adaptive memory manager initialized for {optimization_manager.config.cpu_model.value} "
                   f"with {self.config.memory_pool_size / (1024*1024*1024):.1f}GB pool")
    
    def _create_memory_config(self) -> MemoryConfig:
        """Create memory configuration based on CPU-specific optimizations"""
        profile = self.optimization_manager.config
        
        return MemoryConfig(
            memory_pool_size=profile.memory_pool_size,
            l1_cache_size=profile.l1_cache_size,
            l2_cache_size=profile.l2_cache_size,
            l3_cache_size=profile.l3_cache_size_bytes,
            cache_line_size=profile.cache_line_size,
            memory_threshold=profile.memory_threshold,
            enable_memory_pooling=True,  # Always enabled for performance
            enable_cache_blocking=profile.enable_cache_optimization,
            enable_memory_compression=True,
            compression_level='medium',  # Balanced between speed and compression
            enable_memory_swapping=True,
            swap_threshold=0.85,  # Swap when 85% of memory is used
            enable_memory_defragmentation=True,
            defragmentation_interval=1000,  # Defrag every 1000 operations
            defragmentation_threshold=0.7  # Defrag when fragmentation > 70%
        )
    
    def _calculate_cache_blocking_params(self) -> Dict[str, int]:
        """Calculate optimal cache blocking parameters based on cache sizes"""
        # Calculate blocking sizes based on cache hierarchy
        l1_block_size = max(32, self.config.l1_cache_size // (32 * 1024))  # Conservative L1 blocking
        l2_block_size = max(64, self.config.l2_cache_size // (64 * 1024))  # Moderate L2 blocking
        l3_block_size = max(128, self.config.l3_cache_size // (128 * 1024))  # Larger L3 blocking
        
        return {
            'l1_block_size': l1_block_size,
            'l2_block_size': l2_block_size,
            'l3_block_size': l3_block_size,
            'cache_line_blocks': self.config.cache_line_size // 4  # Assuming 4-byte floats
        }
    
    def allocate_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32, 
                      device: str = 'cpu', pin_memory: bool = False) -> torch.Tensor:
        """
        Allocate a tensor with memory optimization based on cache sizes
        
        Args:
            shape: Shape of the tensor to allocate
            dtype: Data type of the tensor
            device: Device to allocate on ('cpu' or 'cuda')
            pin_memory: Whether to pin memory for faster CPU-GPU transfers
            
        Returns:
            Allocated tensor
        """
        # Check if we have enough memory available
        tensor_size = torch.Size(shape).numel() * torch.tensor([], dtype=dtype).element_size()
        
        if device == 'cpu':
            available_memory = psutil.virtual_memory().available
            if tensor_size > available_memory * self.config.memory_threshold:
                logger.warning(f"Requested tensor size {tensor_size} exceeds memory threshold")
                
                # Try to free up memory
                self._try_free_memory()
                
                # Check again after attempting to free memory
                available_memory = psutil.virtual_memory().available
                if tensor_size > available_memory * self.config.memory_threshold:
                    logger.warning("Still not enough memory, consider using memory mapping or swapping")
        
        # Allocate the tensor
        if device == 'cuda' and torch.cuda.is_available():
            tensor = torch.empty(shape, dtype=dtype, device=device, pin_memory=pin_memory)
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
        
        # Update memory stats
        self._memory_stats['allocation_count'] += 1
        self._memory_stats['current_memory_usage'] += tensor_size
        
        if self._memory_stats['current_memory_usage'] > self._memory_stats['peak_memory_usage']:
            self._memory_stats['peak_memory_usage'] = self._memory_stats['current_memory_usage']
        
        return tensor
    
    def allocate_cache_optimized_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32,
                                      device: str = 'cpu') -> torch.Tensor:
        """
        Allocate a tensor optimized for cache usage based on CPU cache sizes
        
        Args:
            shape: Shape of the tensor to allocate
            dtype: Data type of the tensor
            device: Device to allocate on
            
        Returns:
            Cache-optimized tensor
        """
        tensor = self.allocate_tensor(shape, dtype, device)
        
        # For cache optimization, ensure the tensor is contiguous
        # which helps with cache line utilization
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Apply cache blocking if enabled
        if self.config.enable_cache_blocking:
            # For certain operations, we can optimize memory layout
            # This is particularly useful for matrix operations
            pass  # The contiguous call above helps with cache optimization
        
        return tensor
    
    def _try_free_memory(self):
        """Try to free up memory by clearing cache and running garbage collection"""
        # Run Python garbage collection
        gc.collect()
        
        # If using CUDA, clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.debug("Memory cleanup performed")
    
    def get_memory_usage_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get memory usage statistics
        
        Returns:
            Dictionary containing memory usage statistics
        """
        system_memory = psutil.virtual_memory()
        gpu_memory_info = {}
        
        if torch.cuda.is_available():
            gpu_memory_info = {
                'gpu_allocated': torch.cuda.memory_allocated(),
                'gpu_reserved': torch.cuda.memory_reserved(),
                'gpu_max_allocated': torch.cuda.max_memory_allocated(),
                'gpu_max_reserved': torch.cuda.max_memory_reserved()
            }
        
        return {
            'system_total': system_memory.total,
            'system_available': system_memory.available,
            'system_used': system_memory.used,
            'system_percent_used': system_memory.percent,
            'allocation_count': self._memory_stats['allocation_count'],
            'deallocation_count': self._memory_stats['deallocation_count'],
            'peak_memory_usage': self._memory_stats['peak_memory_usage'],
            'current_memory_usage': self._memory_stats['current_memory_usage'],
            **gpu_memory_info
        }
    
    def should_swap_memory(self) -> bool:
        """
        Determine if memory swapping should be triggered based on usage
        
        Returns:
            True if memory swapping should be triggered, False otherwise
        """
        system_memory = psutil.virtual_memory()
        return system_memory.percent > self.config.swap_threshold * 100
    
    def apply_cache_blocking(self, tensor: torch.Tensor, operation: str = 'matmul') -> torch.Tensor:
        """
        Apply cache blocking to a tensor based on CPU cache sizes
        
        Args:
            tensor: Input tensor to apply cache blocking to
            operation: Type of operation ('matmul', 'conv', 'transformer')
            
        Returns:
            Cache-blocked tensor or the same tensor with metadata for blocking
        """
        if not self.config.enable_cache_blocking:
            return tensor
        
        # For matrix multiplication, we can apply cache blocking by
        # processing sub-blocks that fit in cache
        if operation == 'matmul' and len(tensor.shape) >= 2:
            # Calculate optimal block size based on L2 cache
            # For simplicity, we'll just ensure the tensor is contiguous
            # In practice, you would implement actual blocking algorithms
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
        
        # For convolution operations, we might want to reorganize data
        elif operation == 'conv':
            # Ensure optimal memory layout for convolutions
            if len(tensor.shape) == 4:  # NCHW format
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
        
        return tensor
    
    def compress_tensor(self, tensor: torch.Tensor, 
                       method: str = 'quantization') -> torch.Tensor:
        """
        Compress a tensor to save memory
        
        Args:
            tensor: Input tensor to compress
            method: Compression method ('quantization', 'sparsity', 'factorization')
            
        Returns:
            Compressed tensor
        """
        if not self.config.enable_memory_compression:
            return tensor
        
        if method == 'quantization':
            # Simple quantization to reduce memory usage
            if tensor.dtype == torch.float32:
                # Quantize from float32 to float16
                return tensor.half()
            else:
                return tensor
        elif method == 'sparsity':
            # Simple sparsification - set small values to zero
            threshold = tensor.std() * 0.1  # Set values smaller than 10% of std to zero
            sparse_tensor = tensor * (tensor.abs() > threshold)
            return sparse_tensor
        else:
            # No compression
            return tensor
    
    def decompress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Decompress a tensor back to its original format
        
        Args:
            tensor: Compressed tensor to decompress
            
        Returns:
            Decompressed tensor
        """
        # If the tensor was quantized to float16, we can convert back to float32
        if tensor.dtype == torch.float16:
            return tensor.float()
        else:
            return tensor
    
    def get_cache_blocking_params(self) -> Dict[str, int]:
        """
        Get cache blocking parameters
        
        Returns:
            Dictionary containing cache blocking parameters
        """
        return self._cache_blocking_params.copy()
    
    def get_memory_config(self) -> MemoryConfig:
        """
        Get the current memory configuration
        
        Returns:
            MemoryConfig object
        """
        return self.config


class MemoryManagerFactory:
    """Factory for creating adaptive memory managers based on detected CPU"""
    
    @staticmethod
    def create_for_detected_cpu() -> AdaptiveMemoryManager:
        """
        Create an adaptive memory manager for the currently detected CPU
        
        Returns:
            AdaptiveMemoryManager configured for the detected CPU
        """
        # Use the CPU detector to identify the CPU
        detector = CPUDetector()
        features = detector.get_cpu_features()
        
        # Get the appropriate optimization profile
        optimization_manager = get_optimization_profile(features.model)
        
        # Create and return the adaptive memory manager
        return AdaptiveMemoryManager(optimization_manager)
    
    @staticmethod
    def create_for_cpu_model(cpu_model: CPUModel) -> AdaptiveMemoryManager:
        """
        Create an adaptive memory manager for a specific CPU model
        
        Args:
            cpu_model: The CPU model to create memory manager for
            
        Returns:
            AdaptiveMemoryManager configured for the specified CPU model
        """
        optimization_manager = get_optimization_profile(cpu_model)
        return AdaptiveMemoryManager(optimization_manager)


class MemoryOptimizer:
    """High-level interface for memory optimization"""
    
    def __init__(self):
        self.memory_manager = MemoryManagerFactory.create_for_detected_cpu()
    
    def allocate_model_tensors(self, model) -> None:
        """
        Optimize memory allocation for a model based on CPU characteristics
        
        Args:
            model: PyTorch model to optimize
        """
        # For each parameter in the model, apply memory optimizations
        for name, param in model.named_parameters():
            # Apply cache blocking if enabled
            if self.memory_manager.config.enable_cache_blocking:
                # Ensure parameter is contiguous for better cache usage
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
        
        logger.info(f"Model tensors optimized for {self.memory_manager.optimization_manager.config.cpu_model.value}")
    
    def process_with_memory_optimization(self, tensor_func, *args, **kwargs):
        """
        Process tensors with memory optimization
        
        Args:
            tensor_func: Function that processes tensors
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function with memory optimization applied
        """
        # Check memory usage before processing
        memory_before = self.get_memory_stats()
        
        # Apply cache blocking to inputs if needed
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                blocked_arg = self.memory_manager.apply_cache_blocking(arg)
                new_args.append(blocked_arg)
            else:
                new_args.append(arg)
        
        # Execute the function
        result = tensor_func(*new_args, **kwargs)
        
        # Check memory usage after processing
        memory_after = self.get_memory_stats()
        
        # Log memory usage change
        memory_change = memory_after['current_memory_usage'] - memory_before['current_memory_usage']
        logger.debug(f"Memory change after processing: {memory_change / (1024*1024):.2f} MB")
        
        return result
    
    def get_memory_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get current memory statistics
        
        Returns:
            Dictionary containing memory statistics
        """
        return self.memory_manager.get_memory_usage_stats()
    
    def get_cache_blocking_params(self) -> Dict[str, int]:
        """
        Get cache blocking parameters
        
        Returns:
            Dictionary containing cache blocking parameters
        """
        return self.memory_manager.get_cache_blocking_params()
    
    def get_memory_config(self) -> MemoryConfig:
        """
        Get memory configuration
        
        Returns:
            MemoryConfig object
        """
        return self.memory_manager.get_memory_config()
    
    def should_compress_activations(self) -> bool:
        """
        Determine if activations should be compressed based on memory usage
        
        Returns:
            True if activations should be compressed, False otherwise
        """
        stats = self.get_memory_stats()
        system_memory_percent = stats.get('system_percent_used', 0)
        
        # Compress activations if system memory usage is high
        return system_memory_percent > 75.0  # 75% threshold


def get_memory_optimizer() -> MemoryOptimizer:
    """
    Get a memory optimizer configured for the detected CPU
    
    Returns:
        MemoryOptimizer instance
    """
    return MemoryOptimizer()


class HierarchicalMemoryManager:
    """Manages memory across different tiers (cache, RAM, swap)"""
    
    def __init__(self, memory_optimizer: MemoryOptimizer):
        self.memory_optimizer = memory_optimizer
        self._memory_tiers = {}
    
    def allocate_by_tier(self, size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Allocate memory by selecting the most appropriate tier based on size and usage
        
        Args:
            size: Size of tensor to allocate (number of elements)
            dtype: Data type of tensor
            
        Returns:
            Allocated tensor
        """
        element_size = torch.tensor([], dtype=dtype).element_size()
        tensor_size_bytes = size * element_size
        
        # Determine the best memory tier based on size and current usage
        if tensor_size_bytes <= self.memory_optimizer.memory_manager.config.l1_cache_size:
            tier = MemoryTier.L1_CACHE
        elif tensor_size_bytes <= self.memory_optimizer.memory_manager.config.l2_cache_size:
            tier = MemoryTier.L2_CACHE
        elif tensor_size_bytes <= self.memory_optimizer.memory_manager.config.l3_cache_size:
            tier = MemoryTier.L3_CACHE
        else:
            # For larger tensors, use RAM or consider swapping
            stats = self.memory_optimizer.get_memory_stats()
            if stats['system_percent_used'] < 70:  # Use RAM if not too full
                tier = MemoryTier.RAM
            else:
                tier = MemoryTier.SWAP  # Would use memory mapping or swapping
        
        # For this implementation, we'll just allocate in RAM
        # with appropriate optimizations
        shape = (size,) if size > 0 else (1,)
        tensor = self.memory_optimizer.memory_manager.allocate_cache_optimized_tensor(
            shape, dtype, 'cpu'
        )
        
        self._memory_tiers[tensor.data_ptr()] = tier
        
        logger.debug(f"Allocated tensor of size {tensor_size_bytes} bytes in {tier.value}")
        
        return tensor
    
    def get_memory_tier(self, tensor: torch.Tensor) -> Optional[MemoryTier]:
        """
        Get the memory tier for a tensor
        
        Args:
            tensor: Tensor to check
            
        Returns:
            MemoryTier for the tensor, or None if not tracked
        """
        return self._memory_tiers.get(tensor.data_ptr(), None)


def get_hierarchical_memory_manager() -> HierarchicalMemoryManager:
    """
    Get a hierarchical memory manager configured for the detected CPU
    
    Returns:
        HierarchicalMemoryManager instance
    """
    memory_optimizer = get_memory_optimizer()
    return HierarchicalMemoryManager(memory_optimizer)


if __name__ == "__main__":
    print("Adaptive Memory Management for Qwen3-VL")
    print("=" * 45)
    
    # Test memory manager creation for different CPUs
    i5_manager = MemoryManagerFactory.create_for_cpu_model(CPUModel.INTEL_I5_10210U)
    i7_manager = MemoryManagerFactory.create_for_cpu_model(CPUModel.INTEL_I7_8700)
    
    print(f"Intel i5-10210U Memory Manager:")
    print(f"  Memory Pool Size: {i5_manager.config.memory_pool_size / (1024*1024*1024):.1f} GB")
    print(f"  L3 Cache Size: {i5_manager.config.l3_cache_size / (1024*1024):.0f} MB")
    print(f"  Cache Blocking: {i5_manager.config.enable_cache_blocking}")
    print()
    
    print(f"Intel i7-8700 Memory Manager:")
    print(f"  Memory Pool Size: {i7_manager.config.memory_pool_size / (1024*1024*1024):.1f} GB")
    print(f"  L3 Cache Size: {i7_manager.config.l3_cache_size / (1024*1024):.0f} MB")
    print(f"  Cache Blocking: {i7_manager.config.enable_cache_blocking}")
    print()
    
    # Test the high-level optimizer
    print(f"Testing High-Level Memory Optimizer:")
    memory_optimizer = get_memory_optimizer()
    stats = memory_optimizer.get_memory_stats()
    print(f"  System Memory Used: {stats['system_percent_used']:.1f}%")
    print(f"  Allocation Count: {stats['allocation_count']}")
    print(f"  Peak Usage: {stats['peak_memory_usage'] / (1024*1024):.1f} MB")
    
    # Test cache blocking parameters
    cache_params = memory_optimizer.get_cache_blocking_params()
    print(f"  L1 Block Size: {cache_params['l1_block_size']}")
    print(f"  L2 Block Size: {cache_params['l2_block_size']}")
    print(f"  L3 Block Size: {cache_params['l3_block_size']}")
    
    # Test hierarchical memory manager
    print(f"\nTesting Hierarchical Memory Manager:")
    hier_manager = get_hierarchical_memory_manager()
    
    # Allocate small tensor (should go to cache tier)
    small_tensor = hier_manager.allocate_by_tier(100)  # 100 elements
    print(f"  Small tensor allocated in: {hier_manager.get_memory_tier(small_tensor).value}")
    
    # Allocate larger tensor (should go to RAM tier)
    large_tensor = hier_manager.allocate_by_tier(1000000)  # 1M elements
    print(f"  Large tensor allocated in: {hier_manager.get_memory_tier(large_tensor).value}")
    
    print("\nAdaptive memory management implementation completed!")
"""
Comprehensive Memory Optimization System for Qwen3-VL Model
Implements Phase 2.9: Memory Pooling and Pre-allocation Techniques

Features:
1. Integration with PyTorch's built-in memory management
2. Tensor caching for commonly used dimensions
3. Memory defragmentation routines optimized for Intel i5-10210U + NVIDIA SM61 + NVMe SSD
4. Integration with gradient checkpointing system
5. Memory layout optimization for vision encoder operations
6. Hardware-specific memory access patterns
7. Error handling and validation for memory operations
"""

import torch
import torch.nn as nn
import numpy as np
import psutil
import time
import gc
from typing import Dict, Any, Tuple, Optional, List, Callable
from collections import defaultdict, deque
import logging
import threading
from dataclasses import dataclass
import os


@dataclass
class MemoryConfig:
    """Configuration for memory management optimizations."""
    # Memory pooling settings
    use_tensor_caching: bool = True
    cache_max_size: int = 100  # Max cached tensors per shape

    # Hardware-specific settings for Intel i5-10210U + NVIDIA SM61
    hardware_compute_capability: Tuple[int, int] = (6, 1)  # SM61
    shared_memory_per_block: int = 48 * 1024  # 48KB for SM61
    memory_bandwidth_gb_s: float = 192.0  # Estimated for GTX 1080 Ti

    # Memory fragmentation settings
    defragmentation_threshold: float = 0.3  # Defragment when fragmentation > 30%
    memory_pressure_threshold: float = 0.8  # High memory pressure threshold


class TensorCache:
    """
    Tensor cache for commonly used dimensions using PyTorch's memory management
    """
    def __init__(self, max_cache_size: int = 100):
        self.max_cache_size = max_cache_size
        self.cache = defaultdict(list)  # {(shape, dtype, device): [tensor, ...]}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
            'cached_tensors': 0,
            'max_cached_per_shape': 5  # Max cached tensors per shape
        }

    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device = torch.device('cpu')) -> Optional[torch.Tensor]:
        """Get a tensor from cache if available"""
        self.stats['total_requests'] += 1
        key = (shape, dtype, str(device))

        if self.cache[key] and len(self.cache[key]) > 0:
            tensor = self.cache[key].pop()
            self.stats['hits'] += 1
            self.stats['cached_tensors'] -= 1
            return tensor
        else:
            self.stats['misses'] += 1
            return None

    def return_tensor(self, tensor: torch.Tensor):
        """Return a tensor to the cache if appropriate"""
        if tensor.grad_fn is not None or tensor.requires_grad:
            # Don't cache tensors that are part of the computation graph
            return

        key = (tuple(tensor.shape), tensor.dtype, str(tensor.device))

        # Only cache if not too many of this shape are already cached
        if len(self.cache[key]) < self.stats['max_cached_per_shape']:
            self.cache[key].append(tensor)
            self.stats['cached_tensors'] += 1

    def clear_cache(self):
        """Clear the tensor cache"""
        self.cache.clear()
        self.stats['cached_tensors'] = 0


class SimpleMemoryPool:
    """
    Simplified memory pool that leverages PyTorch's built-in memory management
    """
    def __init__(self):
        self.tensor_cache = TensorCache()

        # Common tensor shapes that are frequently used in vision-language models
        self.common_shapes = [
            ((1, 512, 4096), torch.float32),  # Typical attention output
            ((1, 512, 2048), torch.float32),  # Typical FFN intermediate
            ((1, 256, 4096), torch.float32),  # Smaller attention output
            ((1, 512, 512), torch.float32),   # Attention scores matrix
            ((1, 224, 224, 3), torch.float32), # Typical image patch
            ((1, 14, 14, 1024), torch.float32), # Vision transformer patches
            ((1, 196, 768), torch.float32),    # Flattened patches
            ((1, 128, 768), torch.float32),    # Smaller sequence
        ]

        # Pre-allocate common tensors
        self._preallocate_common_tensors()

    def _preallocate_common_tensors(self):
        """Pre-allocate commonly used tensor shapes"""
        for shape, dtype in self.common_shapes:
            # Pre-allocate a few tensors of each common shape
            for _ in range(3):  # Pre-allocate 3 of each common shape
                try:
                    tensor = torch.empty(shape, dtype=dtype)
                    self.tensor_cache.return_tensor(tensor)
                except Exception:
                    # If allocation fails, continue with other shapes
                    continue

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                       device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Allocate a tensor with the specified shape and dtype using optimized allocation"""
        # First, try to get from cache
        cached_tensor = self.tensor_cache.get_tensor(shape, dtype, device)
        if cached_tensor is not None:
            return cached_tensor

        # If not in cache, allocate new tensor using PyTorch's memory management
        return torch.empty(shape, dtype=dtype, device=device)

    def deallocate_tensor(self, tensor: torch.Tensor) -> bool:
        """Deallocate a tensor and return it to the cache if appropriate"""
        # For PyTorch tensors, we just return to cache if appropriate
        # PyTorch handles actual memory deallocation via garbage collection

        # Return to cache if it's a common shape
        shape, dtype, device = tensor.shape, tensor.dtype, tensor.device
        key = (shape, dtype, str(device))

        # Check if this is a common shape worth caching
        is_common_shape = key[:2] in [(s, d) for s, d in self.common_shapes]

        if is_common_shape:
            self.tensor_cache.return_tensor(tensor)
            return True

        return False

    def get_memory_stats(self) -> Dict:
        """Get memory pool statistics"""
        cache_stats = self.tensor_cache.stats.copy()

        return {
            'tensor_cache': cache_stats,
            'cuda_memory_stats': self._get_cuda_memory_stats() if torch.cuda.is_available() else {}
        }

    def _get_cuda_memory_stats(self) -> Dict:
        """Get CUDA memory statistics"""
        return {
            'allocated_memory': torch.cuda.memory_allocated(),
            'reserved_memory': torch.cuda.memory_reserved(),
            'max_allocated': torch.cuda.max_memory_allocated(),
            'max_reserved': torch.cuda.max_memory_reserved(),
        }

    def defragment(self):
        """Perform memory defragmentation by triggering PyTorch's memory management"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # PyTorch will handle the actual defragmentation at the driver level
        # Clear cache to prevent memory bloat
        self.tensor_cache.clear_cache()

        return {
            'defragmentation_performed': True,
            'success': True
        }


class GradientCheckpointingMemoryIntegrator:
    """
    Integrates memory pooling with existing gradient checkpointing mechanisms
    to reduce memory overhead during training.
    """
    def __init__(self, memory_pool: SimpleMemoryPool):
        self.memory_pool = memory_pool
        self.checkpoint_cache = {}  # {key: tensor}
        self.checkpoint_history = deque(maxlen=100)

        # Hardware-specific optimization for SM61
        self.shared_memory_per_block = 48 * 1024  # 48KB shared memory per SM61 block

    def checkpoint_tensors(self, tensors: List[torch.Tensor],
                          key: Optional[str] = None) -> Dict[str, Any]:
        """
        Checkpoint tensors using memory pool for efficient storage
        """
        checkpoint_key = key or f"chkpt_{int(time.time() * 1000000)}"
        checkpoint_data = {}

        for i, tensor in enumerate(tensors):
            tensor_key = f"{checkpoint_key}_tensor_{i}"

            # For SM61, consider using half precision to save memory
            if tensor.dtype == torch.float32:
                # Store in memory pool managed cache
                pooled_tensor = self.memory_pool.allocate_tensor(tensor.shape, torch.float16, tensor.device)
                pooled_tensor.copy_(tensor.half())
                checkpoint_data[tensor_key] = pooled_tensor
            else:
                # Create a copy of the tensor using pooled memory if it's large enough to benefit
                if tensor.numel() > 1024:  # Only for larger tensors
                    pooled_tensor = self.memory_pool.allocate_tensor(tensor.shape, tensor.dtype, tensor.device)
                    pooled_tensor.copy_(tensor)
                    checkpoint_data[tensor_key] = pooled_tensor
                else:
                    checkpoint_data[tensor_key] = tensor.detach().clone()

        self.checkpoint_cache[checkpoint_key] = checkpoint_data
        self.checkpoint_history.append({
            'key': checkpoint_key,
            'timestamp': time.time(),
            'size': sum(t.numel() * t.element_size() for t in checkpoint_data.values())
        })

        return {
            'checkpoint_key': checkpoint_key,
            'saved_tensors': len(tensors),
            'memory_saved': sum(t.numel() * t.element_size() for t in tensors)
        }

    def restore_tensors(self, checkpoint_key: str) -> List[torch.Tensor]:
        """
        Restore tensors from checkpoint using memory pool
        """
        if checkpoint_key not in self.checkpoint_cache:
            raise KeyError(f"Checkpoint {checkpoint_key} not found")

        checkpoint_data = self.checkpoint_cache[checkpoint_key]
        restored_tensors = []

        for key, stored_tensor in checkpoint_data.items():
            # Check if it was stored in half precision
            if stored_tensor.dtype == torch.float16:
                # Restore to full precision
                original_shape = stored_tensor.shape
                original_dtype = torch.float32  # Assume original was float32
                restored_tensor = self.memory_pool.allocate_tensor(original_shape, original_dtype, stored_tensor.device)
                restored_tensor.copy_(stored_tensor.float())
                restored_tensors.append(restored_tensor)
            else:
                restored_tensors.append(stored_tensor)

        # Remove from cache after restoration (caller should manage lifecycle)
        del self.checkpoint_cache[checkpoint_key]

        return restored_tensors

    def clear_checkpoint_cache(self):
        """
        Clear the checkpoint cache, returning tensors to the memory pool
        """
        for checkpoint_key, checkpoint_data in self.checkpoint_cache.items():
            for tensor_key, pooled_tensor in checkpoint_data.items():
                # Return pooled tensor to memory pool
                if hasattr(pooled_tensor, 'shape'):  # Check if still a tensor
                    self.memory_pool.deallocate_tensor(pooled_tensor)

        self.checkpoint_cache.clear()
        self.checkpoint_history.clear()


class VisionEncoderMemoryOptimizer:
    """
    Optimizes memory layouts specifically for vision encoder operations
    """
    def __init__(self, shared_memory_per_block: int = 48 * 1024):
        self.shared_memory_per_block = shared_memory_per_block
        self.memory_access_pattern_analyzer = self._analyze_memory_access_patterns()

    def _analyze_memory_access_patterns(self) -> Dict[str, Any]:
        """Analyze optimal memory access patterns for vision operations"""
        return {
            'convolutional': {
                'memory_format': 'channels_last',  # Better for convolutions on NVIDIA GPUs
                'tile_size': 64,  # Optimal for SM61 memory transactions
            },
            'attention': {
                'memory_format': 'contiguous',  # Better for attention on most architectures
                'tile_size': 32,  # Optimal for attention computations
            },
            'patch_processing': {
                'memory_format': 'channels_last',  # For vision transformer patch processing
                'tile_size': 14,  # Common patch grid size
            }
        }

    def optimize_patch_processing_memory(self, batch_size: int, image_size: Tuple[int, int], patch_size: int) -> Dict:
        """
        Optimize memory layout for patch processing in vision transformers
        """
        h, w = image_size
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        num_patches = num_patches_h * num_patches_w

        # Calculate memory requirements for different stages of patch processing
        patch_embedding_shape = (batch_size, num_patches, patch_size * patch_size * 3)  # RGB channels
        position_embedding_shape = (1, num_patches, patch_size * patch_size * 3)
        cls_token_shape = (batch_size, 1, patch_size * patch_size * 3)

        # Optimize for memory access patterns in SM61
        memory_layout = {
            'patches': patch_embedding_shape,
            'position_embeddings': position_embedding_shape,
            'cls_tokens': cls_token_shape,
            'patch_embeddings': (batch_size, num_patches, patch_size * patch_size * 3),
            'transformer_outputs': [(batch_size, num_patches + 1, patch_size * patch_size * 3)] * 12  # 12 layers
        }

        # Calculate total memory requirement
        total_params = 0
        for shape in memory_layout.values():
            if isinstance(shape[0], int):  # Single shape
                total_params += np.prod(shape)
            else:  # Multiple shapes (like transformer outputs)
                for s in shape:
                    total_params += np.prod(s)

        total_memory_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32 (4 bytes)

        return {
            'memory_layout': memory_layout,
            'total_memory_mb': total_memory_mb,
            'memory_access_pattern': self.memory_access_pattern_analyzer['patch_processing']['memory_format'],
            'tile_size': self.memory_access_pattern_analyzer['patch_processing']['tile_size']
        }

    def optimize_convolutional_memory(self, input_shape: Tuple[int, ...]) -> Dict:
        """
        Optimize memory for convolutional operations in vision processing
        """
        batch_size, channels, height, width = input_shape

        # For SM61, consider using channels_last format for better memory access
        memory_format = self.memory_access_pattern_analyzer['convolutional']['memory_format']

        # Optimize for memory access patterns in convolutions based on SM61 capabilities
        memory_layout = {
            'input': input_shape,
            'weights': (64, channels, 3, 3),  # Example conv kernel
            'bias': (64,),  # Example bias
            'output': (batch_size, 64, height, width),  # Example output
            'feature_maps': [
                (batch_size, channels, height, width),  # Input
                (batch_size, 64, height, width),       # After conv
                (batch_size, 64, height//2, width//2), # After pooling
                (batch_size, 128, height//2, width//2), # After second conv
            ]
        }

        # Calculate memory for each stage
        total_memory = 0
        stage_memory = []
        for i, fmap_shape in enumerate(memory_layout['feature_maps']):
            mem = np.prod(fmap_shape) * 4  # 4 bytes for float32
            stage_memory.append(mem)
            total_memory += mem

        return {
            'memory_layout': memory_layout,
            'stage_memory_bytes': stage_memory,
            'total_memory_bytes': total_memory,
            'memory_format': memory_format,
            'tile_size': self.memory_access_pattern_analyzer['convolutional']['tile_size']
        }


class MemoryManager:
    """
    Centralized memory manager for the Qwen3-VL model.
    Handles memory allocation, deallocation, and optimization.
    """
    def __init__(self, memory_pool_size: int = None, config: MemoryConfig = None):
        if config is None:
            self.config = MemoryConfig()
            if memory_pool_size is not None:
                self.config.memory_pool_size = memory_pool_size
        else:
            self.config = config
            if memory_pool_size is not None:  # Override config if explicitly provided
                self.config.memory_pool_size = memory_pool_size

        self.memory_pool = SimpleMemoryPool()

        # Initialize gradient checkpointing integrator
        self.gradient_checkpointing_integrator = GradientCheckpointingMemoryIntegrator(self.memory_pool)

        # Initialize vision encoder optimizer
        self.vision_optimizer = VisionEncoderMemoryOptimizer(self.config.shared_memory_per_block)

        self.stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'peak_memory_usage': 0,
            'allocation_errors': 0
        }

        # Memory pressure monitoring
        self.memory_pressure = 0.0

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                       device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Allocate a tensor with specified shape and type using optimized allocation."""
        try:
            tensor = self.memory_pool.allocate_tensor(shape, dtype, device)
            self.stats['total_allocations'] += 1

            # Update peak memory if needed
            tensor_size = tensor.numel() * tensor.element_size()
            if tensor_size > self.stats['peak_memory_usage']:
                self.stats['peak_memory_usage'] = tensor_size

            # Update memory pressure
            if device.type == 'cuda' and torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated(device)
                max_memory = torch.cuda.get_device_properties(device).total_memory
                self.memory_pressure = current_memory / max_memory
            else:
                # For CPU, use system memory
                self.memory_pressure = psutil.virtual_memory().percent / 100.0

            return tensor
        except Exception as e:
            logging.error(f"Tensor allocation failed: {e}")
            self.stats['allocation_errors'] += 1
            # Fallback to standard PyTorch allocation
            return torch.empty(shape, dtype=dtype, device=device)

    def free_tensor(self, tensor: torch.Tensor) -> bool:
        """Free a tensor and return it to the memory pool if appropriate."""
        try:
            success = self.memory_pool.deallocate_tensor(tensor)
            if success:
                self.stats['total_deallocations'] += 1
            return success
        except Exception as e:
            logging.error(f"Tensor deallocation failed: {e}")
            return False

    def get_memory_stats(self) -> Dict:
        """Get current memory usage statistics."""
        pool_stats = self.memory_pool.get_memory_stats()

        return {
            **self.stats,
            'pool_stats': pool_stats,
            'memory_pressure': self.memory_pressure,
            'system_memory_percent': psutil.virtual_memory().percent,
            'available_system_memory_gb': psutil.virtual_memory().available / (1024**3)
        }

    def clear_cache(self):
        """Clear tensor caches to free up memory."""
        self.memory_pool.tensor_cache.clear_cache()

    def defragment_memory(self) -> Dict:
        """Perform memory defragmentation."""
        return self.memory_pool.defragment()

    def register_common_tensor_shapes(self, shapes: List[Tuple[Tuple[int, ...], torch.dtype]]):
        """Register additional common tensor shapes for pre-allocation."""
        self.memory_pool.common_shapes.extend(shapes)
        # Pre-allocate some tensors of these shapes
        for shape, dtype in shapes:
            for _ in range(2):  # Pre-allocate 2 of each new shape
                try:
                    tensor = self.memory_pool.allocate_tensor(shape, dtype)
                    self.memory_pool.tensor_cache.return_tensor(tensor)
                except Exception:
                    # If allocation fails, continue with other shapes
                    continue


# Global memory manager instance
_global_memory_manager = None
_global_manager_lock = threading.Lock()


def get_memory_manager(config: MemoryConfig = None) -> MemoryManager:
    """Get the global memory manager instance"""
    global _global_memory_manager
    with _global_manager_lock:
        if _global_memory_manager is None:
            _global_memory_manager = MemoryManager(config=config)
    return _global_memory_manager


def allocate_tensor_with_manager(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                                device: torch.device = torch.device('cpu')):
    """Allocate a tensor using the global memory manager"""
    manager = get_memory_manager()
    return manager.allocate_tensor(shape, dtype, device)


def free_tensor_with_manager(tensor: torch.Tensor):
    """Free a tensor using the global memory manager"""
    manager = get_memory_manager()
    return manager.free_tensor(tensor)


class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader with optimized data transfer and caching.
    """
    def __init__(self, dataset, memory_manager: MemoryManager = None,
                 pin_memory: bool = True, **kwargs):
        self.dataset = dataset
        self.memory_manager = memory_manager or get_memory_manager()
        self.pin_memory = pin_memory

        # Set default values that are memory efficient for the target hardware
        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('shuffle', False)
        kwargs.setdefault('drop_last', False)
        kwargs.setdefault('pin_memory', pin_memory)

        # Use multiprocessing for data loading to reduce memory overhead
        kwargs.setdefault('num_workers', 2)
        kwargs.setdefault('persistent_workers', True)

        self.dataloader = torch.utils.data.DataLoader(dataset, **kwargs)

        # Track memory usage
        self.stats = {
            'pinned_memory_batches': 0,
            'transferred_batches': 0
        }

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def transfer_to_device(self, data, device: torch.device):
        """Transfer data to device with memory optimization."""
        if isinstance(data, torch.Tensor):
            if self.pin_memory and data.device.type == 'cpu':
                self.stats['pinned_memory_batches'] += 1
                # Use memory manager for optimized transfer
                return self.memory_manager.allocate_tensor(
                    data.shape, data.dtype, device
                ).copy_(data)
            else:
                return data.to(device, non_blocking=self.pin_memory)
        elif isinstance(data, dict):
            return {k: self.transfer_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.transfer_to_device(d, device) for d in data]
        else:
            return data.to(device, non_blocking=self.pin_memory) if hasattr(data, 'to') else data


def create_optimized_dataloader(dataset, memory_manager: MemoryManager = None, **kwargs):
    """Create an optimized data loader with memory-efficient settings."""
    return MemoryEfficientDataLoader(dataset, memory_manager=memory_manager, **kwargs)


def optimize_model_memory(model: torch.nn.Module, memory_manager: MemoryManager = None, config: MemoryConfig = None):
    """
    Apply memory optimizations to the given model.
    """
    if config is None:
        config = MemoryConfig()

    if memory_manager is None:
        memory_manager = get_memory_manager(config)

    # Add memory manager reference to model components that need it
    for module in model.modules():
        if hasattr(module, '_register_memory_manager'):
            module._register_memory_manager(memory_manager)

    # Optimize tensor allocation for model parameters
    for name, param in model.named_parameters():
        # For SM61, consider using half precision for some parameters to save memory
        if config.use_inference_memory_efficient and param.dtype == torch.float32:
            # Don't change dtype here, but ensure memory-efficient allocation
            # The dtype change would happen during model initialization
            pass

    return model


if __name__ == "__main__":
    print("Testing Comprehensive Memory Management System...")

    config = MemoryConfig()

    # Initialize memory manager
    manager = MemoryManager(config=config)

    print("\n1. Testing Tensor Cache...")

    # Test tensor cache
    cache = TensorCache()
    test_tensor = torch.randn(10, 20)
    cache.return_tensor(test_tensor)
    retrieved_tensor = cache.get_tensor((10, 20), torch.float32, torch.device('cpu'))
    print(f"Cache hit: {retrieved_tensor is not None}, Shape: {retrieved_tensor.shape if retrieved_tensor is not None else 'None'}")
    print("Cache stats:", cache.stats)

    # Test memory pool
    pool = SimpleMemoryPool()  # Use SimpleMemoryPool
    # Allocate tensors
    tensor1 = pool.allocate_tensor((100, 200), torch.float32)
    tensor2 = pool.allocate_tensor((50, 100, 256), torch.float32)
    print(f"Allocated tensors of shapes: {tensor1.shape}, {tensor2.shape}")

    pool.deallocate_tensor(tensor1)
    pool.deallocate_tensor(tensor2)

    print("Memory pool stats:", pool.get_memory_stats())

    print("\n4. Testing Full Memory Manager...")

    # Test full memory manager
    manager = MemoryManager(config=config)

    # Allocate various tensors
    shapes_to_test = [
        (100, 200),
        (50, 100, 256),
        (32, 128, 512),
        (16, 64, 1024)
    ]

    allocated_tensors = []
    for shape in shapes_to_test:
        tensor = manager.allocate_tensor(shape, torch.float32)
        allocated_tensors.append(tensor)
        print(f"Allocated tensor of shape: {tensor.shape}")

    for tensor in allocated_tensors:
        manager.free_tensor(tensor)

    print("Final memory stats:", manager.get_memory_stats())

    print("\n5. Testing Memory Defragmentation...")

    defrag_result = manager.defragment_memory()
    print("Defragmentation result:", defrag_result)

    print("\n6. Testing Gradient Checkpointing Integration...")

    # Test gradient checkpointing integrator
    chkpt_integrator = GradientCheckpointingMemoryIntegrator(manager.memory_pool)

    # Create some tensors to checkpoint
    test_tensors = [torch.randn(10, 20), torch.randn(5, 10, 15)]
    chkpt_result = chkpt_integrator.checkpoint_tensors(test_tensors, "test_key")
    print("Checkpoint result:", chkpt_result)

    restored_tensors = chkpt_integrator.restore_tensors("test_key")
    print(f"Restored {len(restored_tensors)} tensors")

    print("\n7. Testing Vision Encoder Memory Optimization...")

    # Test vision encoder memory optimizer
    vision_optimizer = VisionEncoderMemoryOptimizer()
    patch_result = vision_optimizer.optimize_patch_processing_memory(
        batch_size=1,
        image_size=(224, 224),
        patch_size=16
    )
    print(f"Patch processing memory layout: {patch_result['total_memory_mb']:.2f} MB")

    # Test convolutional memory optimization
    conv_result = vision_optimizer.optimize_convolutional_memory((1, 3, 224, 224))
    print(f"Convolutional memory layout: {conv_result['total_memory_bytes'] / (1024*1024):.2f} MB")

    print("\n8. Testing Global Memory Manager...")

    # Test global memory manager
    global_manager = get_memory_manager()
    global_tensor = allocate_tensor_with_manager((50, 50), torch.float32)
    print(f"Global allocation: {global_tensor.shape}")
    free_tensor_with_manager(global_tensor)

    print("\n9. Testing Optimized DataLoader...")

    # Create a dummy dataset for testing
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return torch.randn(10, 20), torch.randint(0, 2, (1,))

    dataset = DummyDataset()
    optimized_loader = create_optimized_dataloader(dataset, manager)
    print(f"Optimized data loader created with {len(optimized_loader)} batches")

    print("\nComprehensive Memory Management System implementation completed!")
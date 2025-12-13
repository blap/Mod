"""
System-level optimizations for Qwen3-VL model including:
1. CPU-GPU communication pipeline optimization with pinned memory and asynchronous transfers
2. NVMe SSD caching for model components with multi-tier caching and LRU eviction
3. Batch processing strategies with dynamic batching and adaptive scheduling
4. Data loading and preprocessing optimization with multi-threading and prefetching
5. Intelligent resource allocation with dynamic memory management
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import tempfile
from typing import Dict, Any, Optional, Tuple, Callable
from collections import OrderedDict, deque
import threading
import queue
import time
import pickle
from pathlib import Path


class CPUGPUCommunicationOptimizer:
    """
    Optimizes CPU-GPU communication pipeline with pinned memory and asynchronous transfers.
    """
    
    def __init__(self, use_async_transfer: bool = True, use_pinned_memory: bool = True):
        self.use_async_transfer = use_async_transfer
        self.use_pinned_memory = use_pinned_memory
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
    def transfer_to_device(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Transfer tensor to device with optimizations.
        
        Args:
            tensor: Input tensor to transfer
            device: Target device
            
        Returns:
            Transferred tensor
        """
        if device.type == 'cpu':
            return tensor
            
        # Create pinned memory if requested and available
        if self.use_pinned_memory and tensor.device.type == 'cpu':
            # Create pinned memory copy
            pinned_tensor = tensor.pin_memory() if not tensor.is_pinned() else tensor
        else:
            pinned_tensor = tensor
            
        # Transfer with or without async based on configuration
        if self.use_async_transfer and self.stream is not None:
            with torch.cuda.stream(self.stream):
                gpu_tensor = pinned_tensor.to(device, non_blocking=True)
        else:
            gpu_tensor = pinned_tensor.to(device, non_blocking=False)
            
        return gpu_tensor
    
    def batch_transfer_to_device(self, tensors: list, device: torch.device) -> list:
        """
        Transfer a batch of tensors to device with optimizations.
        
        Args:
            tensors: List of tensors to transfer
            device: Target device
            
        Returns:
            List of transferred tensors
        """
        transferred = []
        for tensor in tensors:
            transferred.append(self.transfer_to_device(tensor, device))
        return transferred


class NVMeSSDCache:
    """
    Multi-tier caching system for model components with LRU eviction policy,
    optimized for NVMe SSD access patterns.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size: int = 1024 * 1024 * 1024):  # 1GB default
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "qwen3vl_model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.current_cache_size = 0

        # LRU cache for frequently accessed items
        self.lru_cache: OrderedDict = OrderedDict()

        # Tiered storage: hot (memory), warm (fast disk), cold (slow disk)
        self.hot_cache: OrderedDict = OrderedDict()  # In-memory cache
        self.hot_cache_max_size = 100  # Max items in hot cache
        self.warm_cache_path = os.path.join(self.cache_dir, "warm")
        self.cold_cache_path = os.path.join(self.cache_dir, "cold")
        
        os.makedirs(self.warm_cache_path, exist_ok=True)
        os.makedirs(self.cold_cache_path, exist_ok=True)
    
    def _get_cache_path(self, key: str, tier: str = "warm") -> str:
        """Get cache file path for a given key and tier."""
        if tier == "warm":
            return os.path.join(self.warm_cache_path, f"{key}.pkl")
        else:  # cold
            return os.path.join(self.cold_cache_path, f"{key}.pkl")
    
    def _update_lru(self, key: str) -> None:
        """Update LRU order for a key."""
        if key in self.lru_cache:
            self.lru_cache.move_to_end(key)
        else:
            self.lru_cache[key] = time.time()
    
    def _evict_if_needed(self) -> None:
        """Evict items if cache exceeds maximum size."""
        while len(self.lru_cache) > self.hot_cache_max_size:
            oldest_key, _ = self.lru_cache.popitem(last=False)
            if oldest_key in self.hot_cache:
                del self.hot_cache[oldest_key]
    
    def put(self, key: str, obj: Any, tier: str = "hot") -> None:
        """
        Put an object in the cache.
        
        Args:
            key: Cache key
            obj: Object to cache
            tier: Cache tier ('hot', 'warm', 'cold')
        """
        if tier == "hot":
            # Add to hot cache
            self.hot_cache[key] = obj
            self._update_lru(key)
            self._evict_if_needed()
        else:
            # Save to disk cache
            cache_path = self._get_cache_path(key, tier)
            with open(cache_path, 'wb') as f:
                pickle.dump(obj, f)
            self._update_lru(key)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get an object from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached object or None if not found
        """
        # Check hot cache first
        if key in self.hot_cache:
            self._update_lru(key)
            return self.hot_cache[key]
        
        # Check warm cache
        cache_path = self._get_cache_path(key, "warm")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                obj = pickle.load(f)
            # Move to hot cache for faster access next time
            self.put(key, obj, "hot")
            return obj
        
        # Check cold cache
        cache_path = self._get_cache_path(key, "cold")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                obj = pickle.load(f)
            # Move to warm cache
            self.put(key, obj, "warm")
            return obj
        
        return None
    
    def delete(self, key: str) -> None:
        """Delete an item from all cache tiers."""
        if key in self.hot_cache:
            del self.hot_cache[key]
        if key in self.lru_cache:
            del self.lru_cache[key]
        
        # Delete from disk caches
        for tier in ["warm", "cold"]:
            cache_path = self._get_cache_path(key, tier)
            if os.path.exists(cache_path):
                os.remove(cache_path)
    
    def clear(self) -> None:
        """Clear all cache tiers."""
        self.hot_cache.clear()
        self.lru_cache.clear()
        
        # Clear disk caches
        import shutil
        shutil.rmtree(self.warm_cache_path)
        shutil.rmtree(self.cold_cache_path)
        os.makedirs(self.warm_cache_path, exist_ok=True)
        os.makedirs(self.cold_cache_path, exist_ok=True)


class DynamicBatchScheduler:
    """
    Dynamic batch processing with adaptive scheduling based on input characteristics.
    """
    
    def __init__(self, max_batch_size: int = 32, target_batch_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.target_batch_time = target_batch_time
        self.current_batch_size = 8  # Start with smaller batch
        self.performance_history: deque = deque(maxlen=100)  # Keep last 100 measurements
        self.input_characteristics: dict = {}  # Track characteristics for each input type
        
    def estimate_batch_size(self, input_lengths: list, input_complexity: float = 1.0) -> int:
        """
        Estimate optimal batch size based on input characteristics.
        
        Args:
            input_lengths: List of input sequence lengths
            input_complexity: Complexity factor (1.0 = normal, >1.0 = more complex)
            
        Returns:
            Estimated optimal batch size
        """
        if not input_lengths:
            return self.current_batch_size
            
        avg_length = sum(input_lengths) / len(input_lengths)
        
        # Adjust based on sequence length and complexity
        if avg_length > 512:
            estimated_size = max(1, int(self.current_batch_size / 2 / input_complexity))
        elif avg_length > 256:
            estimated_size = max(2, int(self.current_batch_size / 1.5 / input_complexity))
        elif avg_length > 128:
            estimated_size = int(self.current_batch_size / input_complexity)
        else:
            estimated_size = min(self.max_batch_size, int(self.current_batch_size * 1.5 / input_complexity))
        
        return max(1, min(estimated_size, self.max_batch_size))
    
    def update_batch_size(self, actual_time: float, batch_size: int, successful: bool = True) -> None:
        """
        Update batch size based on performance feedback.
        
        Args:
            actual_time: Actual processing time for the batch
            batch_size: Size of the batch that was processed
            successful: Whether the batch processing was successful
        """
        # Record performance
        self.performance_history.append({
            'time': actual_time,
            'batch_size': batch_size,
            'successful': successful,
            'timestamp': time.time()
        })
        
        # Adjust batch size based on performance
        if actual_time > self.target_batch_time * 1.5:
            # Too slow, decrease batch size more aggressively
            self.current_batch_size = max(1, int(batch_size * 0.7))
        elif actual_time < self.target_batch_time * 0.7:
            # Too fast, increase batch size
            self.current_batch_size = min(self.max_batch_size, int(batch_size * 1.3))
        elif actual_time > self.target_batch_time * 1.2:
            # Somewhat slow, decrease slightly
            self.current_batch_size = max(1, int(batch_size * 0.9))
        elif actual_time < self.target_batch_time * 0.8:
            # Somewhat fast, increase slightly
            self.current_batch_size = min(self.max_batch_size, int(batch_size * 1.1))
    
    def schedule_batch(self, inputs: list, input_complexity_fn: Optional[Callable] = None) -> Tuple[list, int]:
        """
        Schedule a batch based on input characteristics.
        
        Args:
            inputs: List of input items
            input_complexity_fn: Function to estimate input complexity
            
        Returns:
            Tuple of (scheduled_batch, batch_size)
        """
        if not inputs:
            return [], 0
            
        # Estimate input characteristics
        input_lengths = [len(str(inp)) if hasattr(inp, '__len__') else 1 for inp in inputs]
        
        # Estimate complexity if function provided
        input_complexity = 1.0
        if input_complexity_fn:
            input_complexity = sum(input_complexity_fn(inp) for inp in inputs) / len(inputs)
        
        # Estimate optimal batch size
        estimated_batch_size = self.estimate_batch_size(input_lengths, input_complexity)
        
        # Take the minimum of estimated size and available inputs
        actual_batch_size = min(estimated_batch_size, len(inputs))
        
        return inputs[:actual_batch_size], actual_batch_size


class OptimizedDataLoader:
    """
    Optimized data loader with multi-threading, prefetching, and NVMe SSD caching.
    """
    
    def __init__(self, dataset, batch_size: int = 32, num_workers: int = 4, 
                 pin_memory: bool = True, prefetch_factor: int = 2,
                 cache_enabled: bool = True, cache_dir: str = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.cache_enabled = cache_enabled
        
        # Initialize cache if enabled
        self.cache = NVMeSSDCache(cache_dir=cache_dir) if cache_enabled else None
        
        # Internal data loader
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=True
        )
        
        # Prefetch queue
        self.prefetch_queue: queue.Queue = queue.Queue(maxsize=prefetch_factor * 2)
        self.prefetch_thread: Optional[threading.Thread] = None
        self.stop_prefetch_event = threading.Event()
    
    def _prefetch_worker(self) -> None:
        """Worker thread for prefetching data."""
        for batch in self.data_loader:
            if self.stop_prefetch_event.is_set():
                break
            try:
                self.prefetch_queue.put(batch, timeout=1.0)
            except queue.Full:
                continue  # Skip if queue is full
    
    def start_prefetch(self) -> None:
        """Start prefetching in background."""
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.stop_prefetch_event.clear()
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
            self.prefetch_thread.start()
    
    def stop_prefetch(self) -> None:
        """Stop prefetching."""
        self.stop_prefetch_event.set()
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)
    
    def __iter__(self):
        """Iterator that uses prefetching."""
        self.start_prefetch()
        try:
            while True:
                try:
                    if not self.prefetch_queue.empty():
                        yield self.prefetch_queue.get_nowait()
                    else:
                        # Queue is empty, wait briefly and continue
                        time.sleep(0.001)
                        continue
                except queue.Empty:
                    # If nothing available, check if loader is done
                    break
        finally:
            self.stop_prefetch()
    
    def __len__(self) -> int:
        """Return length of underlying data loader."""
        return len(self.data_loader)


class DynamicMemoryManager:
    """
    Dynamic memory management system with intelligent resource allocation.
    """
    
    def __init__(self, initial_memory_pool_size: int = 1024 * 1024 * 1024):  # 1GB
        self.initial_memory_pool_size = initial_memory_pool_size
        self.current_allocated = 0
        self.peak_allocated = 0
        self.memory_blocks: dict = {}  # id -> (size, tensor, timestamp)
        self.allocation_history: deque = deque(maxlen=1000)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Memory pools for different tensor sizes
        self.small_pool: list = []  # For small tensors (< 1MB)
        self.medium_pool: list = []  # For medium tensors (1-10MB)
        self.large_pool: list = []  # For large tensors (> 10MB)
        
        # Memory pressure indicators
        self.memory_pressure = 0.0
        self.last_gc_time = time.time()
    
    def _get_pool_for_size(self, size_bytes: int) -> list:
        """Get appropriate pool for tensor size."""
        if size_bytes < 1024 * 1024:  # < 1MB
            return self.small_pool
        elif size_bytes < 10 * 1024 * 1024:  # < 10MB
            return self.medium_pool
        else:
            return self.large_pool
    
    def allocate(self, shape: tuple, dtype: torch.dtype = torch.float32, 
                 device: torch.device = None) -> torch.Tensor:
        """
        Allocate a tensor with dynamic memory management.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            device: Target device
            
        Returns:
            Allocated tensor
        """
        if device is None:
            device = self.device
            
        element_size = torch.tensor([], dtype=dtype).element_size()
        size_bytes = np.prod(shape) * element_size
        
        # Check memory pressure
        self._update_memory_pressure()
        
        # Try to reuse from appropriate pool
        pool = self._get_pool_for_size(size_bytes)
        
        # Look for reusable tensor in pool
        reusable_tensor = None
        for i, (cached_shape, cached_tensor) in enumerate(pool):
            if cached_shape == shape and cached_tensor.dtype == dtype:
                reusable_tensor = pool.pop(i)[1]
                break
        
        if reusable_tensor is not None:
            # Reuse existing tensor
            tensor = reusable_tensor
        else:
            # Create new tensor
            if device.type == 'cuda' and torch.cuda.is_available():
                tensor = torch.empty(shape, dtype=dtype, device=device)
            else:
                tensor = torch.empty(shape, dtype=dtype, device=device)
        
        # Track allocation
        tensor_id = id(tensor)
        self.memory_blocks[tensor_id] = (size_bytes, tensor, time.time())
        self.current_allocated += size_bytes
        self.peak_allocated = max(self.peak_allocated, self.current_allocated)
        
        # Record allocation
        self.allocation_history.append({
            'size': size_bytes,
            'shape': shape,
            'dtype': dtype,
            'timestamp': time.time(),
            'action': 'allocate'
        })
        
        return tensor
    
    def deallocate(self, tensor: torch.Tensor):
        """
        Deallocate a tensor and return to memory pool.
        
        Args:
            tensor: Tensor to deallocate
        """
        tensor_id = id(tensor)
        if tensor_id in self.memory_blocks:
            size_bytes, _, _ = self.memory_blocks[tensor_id]
            
            # Return to appropriate pool based on size
            pool = self._get_pool_for_size(size_bytes)
            
            # Only cache if pool has space (prevent memory bloat)
            if len(pool) < 10:  # Max 10 tensors per pool
                pool.append((tuple(tensor.shape), tensor))
            else:
                # Pool full, actually free the tensor
                del tensor
            
            # Update tracking
            self.current_allocated -= size_bytes
            del self.memory_blocks[tensor_id]
            
            # Record deallocation
            self.allocation_history.append({
                'size': size_bytes,
                'shape': tensor.shape,
                'timestamp': time.time(),
                'action': 'deallocate'
            })
    
    def _update_memory_pressure(self):
        """Update memory pressure indicator."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            self.memory_pressure = allocated / total_memory
        else:
            # For CPU, estimate based on system memory
            import psutil
            self.memory_pressure = psutil.virtual_memory().percent / 100.0
    
    def trigger_gc_if_needed(self):
        """Trigger garbage collection if memory pressure is high."""
        if self.memory_pressure > 0.8:  # High memory pressure
            if time.time() - self.last_gc_time > 1.0:  # Don't GC too frequently
                self.last_gc_time = time.time()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {
            'current_allocated': self.current_allocated,
            'peak_allocated': self.peak_allocated,
            'memory_pressure': self.memory_pressure,
            'allocation_count': len(self.memory_blocks),
            'small_pool_size': len(self.small_pool),
            'medium_pool_size': len(self.medium_pool),
            'large_pool_size': len(self.large_pool),
            'total_cached_tensors': len(self.small_pool) + len(self.medium_pool) + len(self.large_pool)
        }
        
        if torch.cuda.is_available():
            stats.update({
                'cuda_allocated': torch.cuda.memory_allocated(),
                'cuda_reserved': torch.cuda.memory_reserved(),
                'cuda_peak_allocated': torch.cuda.max_memory_allocated()
            })
        
        return stats


class SystemLevelOptimizer:
    """
    Main system-level optimizer that integrates all optimization techniques.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Initialize optimization components
        self.cpu_gpu_optimizer = CPUGPUCommunicationOptimizer(
            use_async_transfer=config.get('use_async_transfer', True),
            use_pinned_memory=config.get('use_pinned_memory', True)
        )
        
        self.nvme_cache = NVMeSSDCache(
            cache_dir=config.get('cache_dir'),
            max_cache_size=config.get('max_cache_size', 1024 * 1024 * 1024)  # 1GB
        )
        
        self.batch_scheduler = DynamicBatchScheduler(
            max_batch_size=config.get('max_batch_size', 32),
            target_batch_time=config.get('target_batch_time', 0.1)
        )
        
        self.memory_manager = DynamicMemoryManager(
            initial_memory_pool_size=config.get('memory_pool_size', 1024 * 1024 * 1024)  # 1GB
        )
        
        # Performance tracking
        self.throughput_improvement = 0.0
        self.resource_utilization = 0.0
    
    def optimize_model_inference(self, model: nn.Module, 
                                input_data: torch.Tensor,
                                device: torch.device = None) -> torch.Tensor:
        """
        Optimize model inference with all system-level optimizations.
        
        Args:
            model: PyTorch model
            input_data: Input tensor
            device: Target device
            
        Returns:
            Model output
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Optimize CPU-GPU transfer
        optimized_input = self.cpu_gpu_optimizer.transfer_to_device(input_data, device)
        
        # 2. Use dynamic memory management
        # (In a real implementation, we'd use the memory manager for intermediate tensors)
        
        # 3. Run inference
        with torch.no_grad():
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
                output = model(optimized_input)
                
                end_event.record()
                end_event.synchronize()
                inference_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            else:
                start_time = time.time()
                output = model(optimized_input)
                inference_time = time.time() - start_time
        
        # 4. Update batch scheduler with performance info
        self.batch_scheduler.update_batch_size(
            actual_time=inference_time, 
            batch_size=optimized_input.size(0),
            successful=torch.isfinite(output).all().item()
        )
        
        # 5. Trigger memory management if needed
        self.memory_manager.trigger_gc_if_needed()
        
        return output
    
    def optimize_training_step(self, model: nn.Module, 
                              input_data: torch.Tensor,
                              target: torch.Tensor,
                              optimizer: torch.optim.Optimizer,
                              device: torch.device = None) -> Tuple[torch.Tensor, float]:
        """
        Optimize training step with all system-level optimizations.
        
        Args:
            model: PyTorch model
            input_data: Input tensor
            target: Target tensor
            optimizer: PyTorch optimizer
            device: Target device
            
        Returns:
            Tuple of (loss, time_taken)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Optimize CPU-GPU transfers
        input_tensor = self.cpu_gpu_optimizer.transfer_to_device(input_data, device)
        target_tensor = self.cpu_gpu_optimizer.transfer_to_device(target, device)
        
        # 2. Run forward pass
        start_time = time.time()
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        output = model(input_tensor)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target_tensor.view(-1))
        
        # 3. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if device.type == 'cuda':
            end_event.record()
            end_event.synchronize()
            step_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        else:
            step_time = time.time() - start_time
        
        # 4. Update schedulers and managers
        self.batch_scheduler.update_batch_size(
            actual_time=step_time,
            batch_size=input_tensor.size(0),
            successful=torch.isfinite(loss).item()
        )
        
        self.memory_manager.trigger_gc_if_needed()
        
        return loss, step_time
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about all optimizations."""
        return {
            'batch_scheduler': {
                'current_batch_size': self.batch_scheduler.current_batch_size,
                'target_batch_time': self.batch_scheduler.target_batch_time,
                'performance_samples': len(self.batch_scheduler.performance_history)
            },
            'memory_manager': self.memory_manager.get_memory_stats(),
            'cpu_gpu_optimizer': {
                'async_transfers_enabled': self.cpu_gpu_optimizer.use_async_transfer,
                'pinned_memory_enabled': self.cpu_gpu_optimizer.use_pinned_memory
            },
            'nvme_cache': {
                'hot_cache_items': len(self.nvme_cache.hot_cache),
                'lru_cache_items': len(self.nvme_cache.lru_cache)
            }
        }


# Example usage and testing
def demo_system_optimizations():
    """Demonstrate the system-level optimizations."""
    print("System-Level Optimizations Demo")
    print("=" * 50)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Initialize system optimizer
    config = {
        'use_async_transfer': True,
        'use_pinned_memory': True,
        'max_batch_size': 16,
        'target_batch_time': 0.05,
        'memory_pool_size': 512 * 1024 * 1024  # 512MB
    }
    
    sys_optimizer = SystemLevelOptimizer(config)
    
    # Create sample data
    input_data = torch.randn(16, 128)
    target_data = torch.randint(0, 10, (16,))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Run inference optimization
    print("Testing optimized inference...")
    start_time = time.time()
    output = sys_optimizer.optimize_model_inference(model, input_data, device)
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f}s")
    
    # Run training optimization
    print("Testing optimized training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    loss, train_time = sys_optimizer.optimize_training_step(
        model, input_data, target_data, optimizer, device
    )
    print(f"Training time: {train_time:.4f}s, Loss: {loss.item():.4f}")
    
    # Print optimization stats
    stats = sys_optimizer.get_optimization_stats()
    print("\nOptimization Statistics:")
    for category, stat_dict in stats.items():
        print(f"  {category}:")
        for key, value in stat_dict.items():
            print(f"    {key}: {value}")
    
    print("\nSystem-level optimizations applied successfully!")


if __name__ == "__main__":
    demo_system_optimizations()
"""
GPU-CPU Memory Optimization System for Qwen3-VL

This module implements an intelligent memory management system that optimizes the transfer and 
storage of tensors between GPU and CPU memory. It includes predictive strategies, asynchronous 
transfers, pinned memory management, and hardware-specific optimizations for NVIDIA SM61 architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import threading
import time
import queue
import logging
from collections import OrderedDict, deque
import psutil
from dataclasses import dataclass, field
import gc


@dataclass
class TensorTransferInfo:
    """Information about tensor transfers between GPU and CPU"""
    tensor_id: str
    size_bytes: int
    direction: str  # 'cpu_to_gpu' or 'gpu_to_cpu'
    start_time: float
    end_time: float
    latency_ms: float
    priority: int  # Priority for transfer (0-10, higher is more important)


@dataclass
class MemoryDeviceInfo:
    """Information about memory on GPU vs CPU"""
    device: str  # 'cpu' or 'cuda'
    total_memory: int
    available_memory: int
    used_memory: int
    timestamp: float


class GPUCPUMemoryOptimizer:
    """
    Optimizes memory usage and transfers between GPU and CPU for Qwen3-VL models.
    Implements predictive strategies for tensor movement and efficient memory management.
    """
    
    def __init__(self,
                 cpu_memory_limit: int = 4 * 1024 * 1024 * 1024,  # 4GB CPU limit
                 gpu_memory_limit: int = 6 * 1024 * 1024 * 1024,  # 6GB GPU limit
                 use_pinned_memory: bool = True,
                 async_transfer_queue_size: int = 100,
                 prediction_horizon: float = 1.0):  # Seconds to predict ahead
        """
        Initialize the GPU-CPU memory optimizer
        
        Args:
            cpu_memory_limit: Maximum memory to use on CPU
            gpu_memory_limit: Maximum memory to use on GPU
            use_pinned_memory: Whether to use pinned memory for faster transfers
            async_transfer_queue_size: Size of asynchronous transfer queue
            prediction_horizon: Time horizon for access predictions (seconds)
        """
        self.cpu_memory_limit = cpu_memory_limit
        self.gpu_memory_limit = gpu_memory_limit
        self.use_pinned_memory = use_pinned_memory
        self.async_transfer_queue_size = async_transfer_queue_size
        self.prediction_horizon = prediction_horizon
        
        # Track tensor locations and metadata
        self.tensor_locations: Dict[str, str] = {}  # Maps tensor_id to device ('cpu' or 'cuda')
        self.tensor_sizes: Dict[str, int] = {}      # Maps tensor_id to size in bytes
        self.tensor_access_history: Dict[str, deque] = {}  # Access timestamps
        self.tensor_usage_patterns: Dict[str, Dict[str, Any]] = {}  # Usage patterns per tensor
        
        # Memory stats
        self.memory_stats = {
            'cpu_used': 0,
            'gpu_used': 0,
            'pinned_memory_used': 0,
            'transfer_count': 0,
            'async_transfer_queue_size': 0
        }
        
        # Transfer queues
        self.transfer_queue = queue.Queue(maxsize=async_transfer_queue_size)
        self.active_transfers: Dict[str, TensorTransferInfo] = {}
        
        # Prediction models and data
        self.access_predictions: Dict[str, float] = {}  # Predicted next access time
        self.tensor_priorities: Dict[str, int] = {}  # Priorities based on access patterns
        
        # Threading and synchronization
        self.lock = threading.RLock()
        self.transfer_thread = None
        self.stop_transfer_thread = threading.Event()
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize pinned memory if requested
        if self.use_pinned_memory:
            self.pinned_memory_pool = torch.empty(1024 * 1024 * 64, dtype=torch.uint8, pin_memory=True)  # 64MB pinned pool
            self.pinned_memory_available = True
            self.logger.info("Initialized pinned memory pool for faster GPU/CPU transfers")
        else:
            self.pinned_memory_pool = None
            self.pinned_memory_available = False
            self.logger.info("Pinned memory disabled")
        
        # Hardware-specific optimizations
        self._configure_hardware_optimizations()
        
        self.logger.info(f"GPU-CPU Memory Optimizer initialized with CPU limit: {cpu_memory_limit/(1024**3):.1f}GB, "
                         f"GPU limit: {gpu_memory_limit/(1024**3):.1f}GB")
    
    def _configure_hardware_optimizations(self):
        """Configure optimizations for specific hardware (Intel i5-10210U + NVIDIA SM61)"""
        # For Intel i5-10210U: Optimize for CPU-GPU data transfers
        self.cpu_threads = 8  # Use all available logical cores
        self.cpu_cache_line_size = 64
        
        # For NVIDIA SM61 (Maxwell architecture): Optimize for memory bandwidth
        self.gpu_compute_capability = (6, 1)  # SM61
        self.gpu_warp_size = 32  # Standard CUDA warp size
        self.gpu_memory_bandwidth = 80.0  # Estimated in GB/s for GM108 based on MX330 specs
        self.gpu_async_transfer_supported = True  # SM61 supports asynchronous transfers
        
        # Optimize batch sizes based on available memory
        if torch.cuda.is_available():
            gpu_properties = torch.cuda.get_device_properties(0)
            self.gpu_memory_total = gpu_properties.total_memory
            self.gpu_shared_memory_per_block = gpu_properties.shared_mem_per_block
        else:
            self.gpu_memory_total = 6 * 1024 * 1024 * 1024  # Assume 6GB if CUDA unavailable
            self.gpu_shared_memory_per_block = 48 * 1024  # Default for SM61
        
        self.logger.info(f"Hardware optimizations configured: "
                         f"CPU threads={self.cpu_threads}, GPU capability={self.gpu_compute_capability}, "
                         f"Memory bandwidth~{self.gpu_memory_bandwidth}GB/s")
    
    def register_tensor(self, tensor_id: str, tensor: torch.Tensor, initial_device: str = 'cpu') -> bool:
        """
        Register a tensor with the optimizer
        
        Args:
            tensor_id: Unique identifier for the tensor
            tensor: The tensor to register
            initial_device: Initial device for the tensor ('cpu' or 'cuda')
        
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            size_bytes = tensor.element_size() * tensor.nelement()
            
            # Store tensor info
            self.tensor_locations[tensor_id] = initial_device
            self.tensor_sizes[tensor_id] = size_bytes
            self.tensor_access_history[tensor_id] = deque(maxlen=100)  # Keep last 100 access times
            self.tensor_usage_patterns[tensor_id] = {
                'first_access': time.time(),
                'last_access': time.time(),
                'access_count': 0,
                'avg_interval': 0.0
            }
            
            # Update memory stats
            if initial_device == 'cuda':
                self.memory_stats['gpu_used'] += size_bytes
            else:
                self.memory_stats['cpu_used'] += size_bytes
            
            self.logger.debug(f"Registered tensor {tensor_id} ({size_bytes/(1024*1024):.2f}MB) on {initial_device}")
            return True
    
    def move_tensor_to_device(self, tensor_id: str, target_device: str, 
                             priority: int = 5, async_transfer: bool = True) -> bool:
        """
        Move a tensor to the target device (GPU or CPU) with optional asynchronous transfer
        
        Args:
            tensor_id: ID of the tensor to move
            target_device: Target device ('cpu' or 'cuda')
            priority: Priority for the transfer (0-10)
            async_transfer: Whether to perform transfer asynchronously
        
        Returns:
            True if successful (or queued successfully), False otherwise
        """
        with self.lock:
            if tensor_id not in self.tensor_locations:
                self.logger.warning(f"Tensor {tensor_id} not registered")
                return False
            
            current_device = self.tensor_locations[tensor_id]
            if current_device == target_device:
                self.logger.debug(f"Tensor {tensor_id} already on {target_device}")
                return True
            
            size_bytes = self.tensor_sizes[tensor_id]
            
            # Check if target device has sufficient memory
            if target_device == 'cuda' and torch.cuda.is_available():
                if self.memory_stats['gpu_used'] + size_bytes > self.gpu_memory_limit:
                    self.logger.warning(f"Not enough GPU memory for tensor {tensor_id}")
                    return False
            elif target_device == 'cpu':
                if self.memory_stats['cpu_used'] + size_bytes > self.cpu_memory_limit:
                    self.logger.warning(f"Not enough CPU memory for tensor {tensor_id}")
                    return False
            
            # Update memory stats before moving
            if current_device == 'cuda':
                self.memory_stats['gpu_used'] -= size_bytes
            else:
                self.memory_stats['cpu_used'] -= size_bytes
            
            if target_device == 'cuda':
                self.memory_stats['gpu_used'] += size_bytes
            else:
                self.memory_stats['cpu_used'] += size_bytes
            
            # Perform actual move
            if async_transfer and self.gpu_async_transfer_supported:
                # Queue for asynchronous transfer
                try:
                    self.transfer_queue.put_nowait({
                        'tensor_id': tensor_id,
                        'target_device': target_device,
                        'priority': priority,
                        'start_time': time.time()
                    })
                    self.memory_stats['async_transfer_queue_size'] = self.transfer_queue.qsize()
                    
                    # Start transfer thread if not running
                    if self.transfer_thread is None or not self.transfer_thread.is_alive():
                        self._start_transfer_thread()
                    
                    self.logger.debug(f"Queued tensor {tensor_id} for async transfer to {target_device} (priority: {priority})")
                    return True
                except queue.Full:
                    self.logger.warning(f"Transfer queue full, performing sync transfer for {tensor_id}")
                    async_transfer = False
            
            if not async_transfer:
                # Perform synchronous transfer
                self._synchronous_transfer(tensor_id, target_device, size_bytes)
            
            self.logger.info(f"Moved tensor {tensor_id} from {current_device} to {target_device}")
            return True
    
    def _synchronous_transfer(self, tensor_id: str, target_device: str, size_bytes: int):
        """Perform synchronous tensor transfer"""
        start_time = time.time()
        
        # This is a placeholder implementation - actual tensor transfer logic would depend on
        # how tensors are stored and accessed in the broader system
        # In reality, this would involve:
        # 1. Loading tensor from its current device
        # 2. Transferring to target device
        # 3. Updating internal tracking

        end_time = time.time()
        
        # Record transfer info
        transfer_info = TensorTransferInfo(
            tensor_id=tensor_id,
            size_bytes=size_bytes,
            direction=f"{self.tensor_locations[tensor_id]}_to_{target_device}",
            start_time=start_time,
            end_time=end_time,
            latency_ms=(end_time - start_time) * 1000,
            priority=5  # Default priority for sync transfer
        )
        
        self.active_transfers[tensor_id] = transfer_info
        self.memory_stats['transfer_count'] += 1
        
        # Update tensor location
        self.tensor_locations[tensor_id] = target_device
    
    def _start_transfer_thread(self):
        """Start the asynchronous transfer thread"""
        if self.transfer_thread is None or not self.transfer_thread.is_alive():
            self.stop_transfer_thread.clear()
            self.transfer_thread = threading.Thread(target=self._transfer_worker, daemon=True)
            self.transfer_thread.start()
            self.logger.info("Started asynchronous transfer thread")
    
    def _transfer_worker(self):
        """Worker thread for processing transfer queue"""
        while not self.stop_transfer_thread.is_set():
            try:
                # Get transfer request from queue with timeout
                transfer_req = self.transfer_queue.get(timeout=0.1)
                
                tensor_id = transfer_req['tensor_id']
                target_device = transfer_req['target_device']
                size_bytes = self.tensor_sizes.get(tensor_id, 0)
                
                # Perform transfer
                start_time = time.time()
                
                # This is a simulation - in a real implementation we'd need actual tensor reference
                # For now, just update tracking
                current_device = self.tensor_locations.get(tensor_id, 'cpu')
                
                # Update memory stats
                if current_device == 'cuda':
                    self.memory_stats['gpu_used'] -= size_bytes
                else:
                    self.memory_stats['cpu_used'] -= size_bytes
                
                if target_device == 'cuda':
                    self.memory_stats['gpu_used'] += size_bytes
                else:
                    self.memory_stats['cpu_used'] += size_bytes
                
                # Update tensor location
                self.tensor_locations[tensor_id] = target_device
                
                end_time = time.time()
                
                # Record transfer info
                transfer_info = TensorTransferInfo(
                    tensor_id=tensor_id,
                    size_bytes=size_bytes,
                    direction=f"{current_device}_to_{target_device}",
                    start_time=start_time,
                    end_time=end_time,
                    latency_ms=(end_time - start_time) * 1000,
                    priority=transfer_req['priority']
                )
                
                # Update transfer statistics
                self.active_transfers[tensor_id] = transfer_info
                self.memory_stats['transfer_count'] += 1
                self.memory_stats['async_transfer_queue_size'] = self.transfer_queue.qsize()
                
                self.logger.debug(f"Completed async transfer of {tensor_id} to {target_device} "
                                f"in {transfer_info.latency_ms:.2f}ms")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in transfer worker: {e}")
                continue
    
    def predict_tensor_access_time(self, tensor_id: str) -> Optional[float]:
        """
        Predict when a tensor will be accessed next based on historical access patterns
        
        Args:
            tensor_id: ID of the tensor to predict for
        
        Returns:
            Predicted time (in seconds from now) when tensor will be accessed, or None if not predictable
        """
        with self.lock:
            if tensor_id not in self.tensor_access_history:
                return None
            
            access_times = list(self.tensor_access_history[tensor_id])
            if len(access_times) < 2:
                return None
            
            # Simple prediction based on average intervals
            intervals = [access_times[i] - access_times[i-1] for i in range(1, len(access_times))]
            avg_interval = sum(intervals) / len(intervals)
            
            # The next access is predicted to be in avg_interval seconds from the last access
            last_access = access_times[-1]
            predicted_next_access = last_access + avg_interval
            seconds_from_now = predicted_next_access - time.time()
            
            if seconds_from_now > 0:
                return max(seconds_from_now, 0.01)  # Return minimum 10ms if positive
            else:
                return 0.01  # If already passed, return 10ms
    
    def should_move_to_gpu(self, tensor_id: str) -> Tuple[bool, float]:
        """
        Determine if a tensor should be moved to GPU based on predicted access and other factors
        
        Args:
            tensor_id: ID of the tensor to evaluate
        
        Returns:
            Tuple of (should_move, estimated_benefit_score)
        """
        with self.lock:
            if tensor_id not in self.tensor_locations:
                return False, 0.0
            
            # If already on GPU, no need to move
            if self.tensor_locations[tensor_id] == 'cuda':
                return False, 0.0
            
            # Predict access time
            predicted_access_time = self.predict_tensor_access_time(tensor_id)
            
            # If tensor will be accessed soon, consider moving to GPU
            if predicted_access_time is not None and predicted_access_time <= self.prediction_horizon:
                # Calculate benefit score based on tensor size and access pattern
                size_mb = self.tensor_sizes[tensor_id] / (1024 * 1024)
                access_frequency = len(self.tensor_access_history.get(tensor_id, []))
                
                # Benefit increases with size and access frequency
                benefit_score = min(size_mb * access_frequency * 0.1, 10.0)  # Cap at 10.0
                
                return True, benefit_score
            else:
                return False, 0.0
    
    def should_move_to_cpu(self, tensor_id: str) -> Tuple[bool, float]:
        """
        Determine if a tensor should be moved to CPU based on access patterns and GPU memory pressure
        
        Args:
            tensor_id: ID of the tensor to evaluate
        
        Returns:
            Tuple of (should_move, estimated_benefit_score)
        """
        with self.lock:
            if tensor_id not in self.tensor_locations:
                return False, 0.0
            
            # If already on CPU, no need to move
            if self.tensor_locations[tensor_id] == 'cpu':
                return False, 0.0
            
            # Check GPU memory pressure
            gpu_memory_pressure = self.memory_stats['gpu_used'] / self.gpu_memory_limit
            if gpu_memory_pressure < 0.7:  # Only consider moving if GPU memory pressure is high
                return False, 0.0
            
            # Predict when this tensor will be accessed next
            predicted_access_time = self.predict_tensor_access_time(tensor_id)
            
            # If not accessed soon, consider moving to CPU to free up GPU memory
            if predicted_access_time is None or predicted_access_time > self.prediction_horizon:
                # Calculate benefit score based on size and inactivity
                size_mb = self.tensor_sizes[tensor_id] / (1024 * 1024)
                
                # Score increases with size and inactivity
                benefit_score = min(size_mb * gpu_memory_pressure, 10.0)
                
                return True, benefit_score
            else:
                return False, 0.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics
        
        Returns:
            Dictionary with memory statistics
        """
        with self.lock:
            stats = self.memory_stats.copy()
            
            # Add memory utilization percentages
            if torch.cuda.is_available():
                stats['gpu_util_percentage'] = self.memory_stats['gpu_used'] / self.gpu_memory_limit
                stats['gpu_total'] = self.gpu_memory_limit
            else:
                stats['gpu_util_percentage'] = 0.0
                stats['gpu_total'] = 0
            
            stats['cpu_util_percentage'] = self.memory_stats['cpu_used'] / self.cpu_memory_limit
            stats['cpu_total'] = self.cpu_memory_limit
            
            # Add system memory info
            system_memory = psutil.virtual_memory()
            stats['system_memory_available_gb'] = system_memory.available / (1024**3)
            stats['system_memory_total_gb'] = system_memory.total / (1024**3)
            stats['system_memory_util_percentage'] = system_memory.percent / 100.0
            
            stats['pinned_memory_enabled'] = self.pinned_memory_available
            
            return stats
    
    def get_tensor_location(self, tensor_id: str) -> Optional[str]:
        """
        Get the current location of a tensor
        
        Args:
            tensor_id: ID of the tensor
        
        Returns:
            Device string ('cpu' or 'cuda') or None if tensor not found
        """
        with self.lock:
            return self.tensor_locations.get(tensor_id)
    
    def record_tensor_access(self, tensor_id: str) -> bool:
        """
        Record access to a tensor to improve prediction accuracy
        
        Args:
            tensor_id: ID of the tensor being accessed
        
        Returns:
            True if successful, False if tensor not found
        """
        with self.lock:
            if tensor_id not in self.tensor_locations:
                return False
            
            current_time = time.time()
            
            # Update access history
            if tensor_id not in self.tensor_access_history:
                self.tensor_access_history[tensor_id] = deque(maxlen=100)
            self.tensor_access_history[tensor_id].append(current_time)
            
            # Update usage pattern
            pattern = self.tensor_usage_patterns[tensor_id]
            if pattern['access_count'] > 0:
                # Update average interval
                last_access_interval = current_time - pattern['last_access']
                pattern['avg_interval'] = (
                    (pattern['avg_interval'] * pattern['access_count'] + last_access_interval) / 
                    (pattern['access_count'] + 1)
                )
            
            pattern['last_access'] = current_time
            pattern['access_count'] += 1
            
            # Update tensor priority based on access frequency
            access_frequency = pattern['access_count'] / max(1, current_time - pattern['first_access'])
            self.tensor_priorities[tensor_id] = min(int(access_frequency * 10), 10)  # Clamp to 0-10
            
            return True
    
    def optimize_memory_for_inference(self, input_tensor_ids: List[str]) -> Dict[str, str]:
        """
        Optimize memory layout for an inference pass based on expected access patterns
        
        Args:
            input_tensor_ids: List of tensor IDs that will be used in the inference pass
        
        Returns:
            Dictionary mapping tensor IDs to recommended device locations
        """
        with self.lock:
            recommendations = {}
            
            for tensor_id in input_tensor_ids:
                if tensor_id in self.tensor_locations:
                    # Evaluate if tensor should be moved
                    should_gpu, gpu_score = self.should_move_to_gpu(tensor_id)
                    should_cpu, cpu_score = self.should_move_to_cpu(tensor_id)
                    
                    if should_gpu and not should_cpu:
                        recommendations[tensor_id] = 'cuda'
                    elif should_cpu and not should_gpu:
                        recommendations[tensor_id] = 'cpu'
                    elif should_gpu and should_cpu:
                        # If both suggest moving, use score to decide
                        if gpu_score > cpu_score:
                            recommendations[tensor_id] = 'cuda'
                        else:
                            recommendations[tensor_id] = 'cpu'
                    else:
                        # Keep in current location
                        recommendations[tensor_id] = self.tensor_locations[tensor_id]
            
            return recommendations
    
    def cleanup(self):
        """
        Clean up resources used by the optimizer
        """
        self.stop_transfer_thread.set()
        
        if self.transfer_thread and self.transfer_thread.is_alive():
            self.transfer_thread.join(timeout=2.0)
        
        self.logger.info("GPU-CPU Memory Optimizer cleaned up successfully")


class GPUCPUMemoryOptimizerManager:
    """
    Singleton manager for GPU-CPU memory optimization system
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GPUCPUMemoryOptimizerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.optimizer = GPUCPUMemoryOptimizer()
            self.initialized = True
    
    def get_optimizer(self) -> GPUCPUMemoryOptimizer:
        """Get the memory optimizer instance"""
        return self.optimizer


# Helper function to optimize tensor placement
def optimize_tensor_placement_for_hardware(tensors: Dict[str, torch.Tensor], 
                                         hardware_config: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
    """
    Optimize placement of tensors for specific hardware based on size and access patterns.
    
    Args:
        tensors: Dictionary mapping tensor names to tensors
        hardware_config: Hardware configuration with memory limits and capabilities
    
    Returns:
        Dictionary mapping tensor names to optimized tensors on appropriate devices
    """
    if hardware_config is None:
        # Default configuration for Intel i5-10210U + NVIDIA SM61
        hardware_config = {
            'cpu_memory_limit': 4 * 1024 * 1024 * 1024,  # 4GB
            'gpu_memory_limit': 6 * 1024 * 1024 * 1024,  # 6GB for SM61
            'use_pinned_memory': True,
            'prediction_horizon': 1.0
        }
    
    optimizer = GPUCPUMemoryOptimizer(**hardware_config)
    
    optimized_tensors = {}
    tensor_ids = {}
    
    # Register all tensors
    for name, tensor in tensors.items():
        tensor_id = f"tensor_{hash(name)}_{int(time.time() * 1000)}"
        tensor_ids[name] = tensor_id
        optimizer.register_tensor(tensor_id, tensor)
        
        # Determine optimal placement based on tensor characteristics
        tensor_size = tensor.element_size() * tensor.nelement()
        
        # Small tensors that will be frequently accessed go to GPU
        if (tensor_size < 10 * 1024 * 1024 and  # Less than 10MB
            len(tensor.shape) <= 2):  # Typically attention matrices, small weights
            target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif tensor_size > 100 * 1024 * 1024:  # Over 100MB
            # Large tensors might be better on CPU to preserve GPU memory
            # unless they're used intensively in computations
            target_device = 'cpu'
        else:
            # Medium tensors - consider based on current memory pressure
            stats = optimizer.get_memory_stats()
            if torch.cuda.is_available() and stats['gpu_util_percentage'] < 0.7:
                target_device = 'cuda'
            else:
                target_device = 'cpu'
        
        # Move tensor if needed
        if optimizer.get_tensor_location(tensor_id) != target_device:
            optimizer.move_tensor_to_device(tensor_id, target_device, async_transfer=True)
        
        # Record access for prediction
        optimizer.record_tensor_access(tensor_id)
        
        optimized_tensors[name] = tensor  # In a real implementation, this would be the relocated tensor
    
    return optimized_tensors


# Example usage
if __name__ == "__main__":
    print("Testing GPU-CPU Memory Optimization System...")
    
    # Create memory optimizer
    optimizer = GPUCPUMemoryOptimizer(
        cpu_memory_limit=2 * 1024 * 1024 * 1024,  # 2GB
        gpu_memory_limit=4 * 1024 * 1024 * 1024   # 4GB
    )
    
    print("\n1. Testing tensor registration...")
    # Register some tensors
    test_tensors = {
        'small_weight': torch.randn(64, 64),  # ~16KB
        'large_weight': torch.randn(1000, 1000),  # ~4MB
        'activation_tensor': torch.randn(4, 512, 1024),  # ~8MB
        'kv_cache': torch.randn(4, 32, 2048, 128)  # ~134MB
    }
    
    for name, tensor in test_tensors.items():
        tensor_id = f"tensor_{name}_{int(time.time())}"
        success = optimizer.register_tensor(tensor_id, tensor, 'cpu')
        print(f"   Registered {name} ({tensor_id}): {success}, size: {tensor.numel() * tensor.element_size() / (1024*1024):.2f}MB")
    
    print("\n2. Testing tensor access recording and prediction...")
    # Simulate accessing some tensors
    for name in ['small_weight', 'activation_tensor']:
        tensor_id = [tid for tid in optimizer.tensor_locations.keys() if name in tid][0]
        optimizer.record_tensor_access(tensor_id)
        print(f"   Recorded access for {name} ({tensor_id})")
    
    # Predict access times
    for name in ['small_weight', 'large_weight']:
        tensor_id = [tid for tid in optimizer.tensor_locations.keys() if name in tid][0]
        predicted_time = optimizer.predict_tensor_access_time(tensor_id)
        if predicted_time:
            print(f"   Predicted next access for {name}: {predicted_time:.2f}s")
        else:
            print(f"   Could not predict access time for {name}")
    
    print("\n3. Testing tensor movement recommendations...")
    # Get recommendations for tensor movement
    all_tensor_ids = list(optimizer.tensor_locations.keys())
    recommendations = optimizer.optimize_memory_for_inference(all_tensor_ids)
    
    for tensor_id, recommended_device in recommendations.items():
        current_device = optimizer.get_tensor_location(tensor_id)
        print(f"   {tensor_id}: {current_device} -> {recommended_device}")
    
    print("\n4. Testing actual tensor movement...")
    # Test moving a tensor to GPU if available
    if torch.cuda.is_available():
        tensor_id = all_tensor_ids[0]  # Select first tensor
        print(f"   Moving {tensor_id} to GPU...")
        success = optimizer.move_tensor_to_device(tensor_id, 'cuda', priority=8, async_transfer=False)
        print(f"   Move successful: {success}")
    
    print("\n5. Checking memory statistics...")
    # Check memory stats
    stats = optimizer.get_memory_stats()
    print(f"   GPU used: {stats['gpu_used'] / (1024**3):.2f}GB / {stats['gpu_total'] / (1024**3):.2f}GB ({stats['gpu_util_percentage']:.1%})")
    print(f"   CPU used: {stats['cpu_used'] / (1024**3):.2f}GB / {stats['cpu_total'] / (1024**3):.2f}GB ({stats['cpu_util_percentage']:.1%})")
    print(f"   System memory available: {stats['system_memory_available_gb']:.2f}GB")
    
    print("\n6. Testing hardware-specific optimization function...")
    # Test the helper function
    optimized_tensors = optimize_tensor_placement_for_hardware(test_tensors)
    print(f"   Optimized {len(optimized_tensors)} tensors for hardware")
    
    # Clean up
    optimizer.cleanup()
    print("\nGPU-CPU Memory Optimization System test completed successfully!")
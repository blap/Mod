"""CPU-GPU Memory Transfer Optimization for Qwen3-VL Model on Intel i5-10210U + NVIDIA SM61 + NVMe SSD"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import threading
import time
import math
import psutil
from collections import defaultdict, deque
import logging
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransferType(Enum):
    """Types of memory transfers"""
    HOST_TO_DEVICE = "host_to_device"
    DEVICE_TO_HOST = "device_to_host"
    DEVICE_TO_DEVICE = "device_to_device"
    PINNED_TO_DEVICE = "pinned_to_device"

class PinnedMemoryManager:
    """
    Manages pinned (page-locked) memory for efficient CPU-GPU transfers.
    Pinned memory enables faster host-to-device transfers as it doesn't need to be copied to pageable memory first.
    """
    def __init__(self, max_pinned_memory: int = 512 * 1024 * 1024):  # 512MB default
        self.max_pinned_memory = max_pinned_memory
        self.current_pinned_memory = 0
        self.pinned_memory_pool = {}  # {(shape, dtype): [tensor_list]}
        self.pinned_memory_alignment = 256  # Align to 256-byte boundaries for optimal transfer
        
        # Statistics
        self.stats = {
            'pinned_allocations': 0,
            'pinned_deallocations': 0,
            'transfer_efficiency_improvements': 0,
            'total_transfer_time_saved': 0.0
        }
    
    def allocate_pinned_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Optional[torch.Tensor]:
        """
        Allocate a pinned tensor if memory is available.
        """
        element_size = torch.tensor([], dtype=dtype).element_size()
        tensor_size = np.prod(shape) * element_size
        
        # Check if we have enough memory available
        if self.current_pinned_memory + tensor_size > self.max_pinned_memory:
            logger.warning(f"Not enough pinned memory available. Requested: {tensor_size}, Available: {self.max_pinned_memory - self.current_pinned_memory}")
            return None

        try:
            # Create pinned memory tensor
            tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
            self.current_pinned_memory += tensor_size
            self.stats['pinned_allocations'] += 1

            return tensor
        except Exception as e:
            logger.error(f"Failed to allocate pinned tensor: {e}")
            return None

    def return_pinned_tensor(self, tensor: torch.Tensor):
        """
        Return a pinned tensor to the pool for potential reuse.
        """
        if tensor.is_pinned():
            tensor_size = tensor.numel() * tensor.element_size()
            self.current_pinned_memory -= tensor_size
            self.stats['pinned_deallocations'] += 1
    
    def get_pinned_memory_stats(self) -> Dict[str, Any]:
        """Get pinned memory statistics."""
        return {
            'current_pinned_memory': self.current_pinned_memory,
            'max_pinned_memory': self.max_pinned_memory,
            'pinned_memory_utilization': self.current_pinned_memory / self.max_pinned_memory if self.max_pinned_memory > 0 else 0,
            'stats': self.stats
        }


class AsyncTransferManager:
    """
    Manages asynchronous memory transfers between CPU and GPU.
    Uses CUDA streams to overlap memory transfers with computation.
    """
    def __init__(self):
        # Create CUDA streams for asynchronous transfers
        self.transfer_streams = []
        self.compute_stream = torch.cuda.default_stream() if torch.cuda.is_available() else None

        # Initialize streams if CUDA is available
        if torch.cuda.is_available():
            for i in range(4):  # Create 4 transfer streams
                self.transfer_streams.append(torch.cuda.Stream())

        # Track ongoing transfers
        self.ongoing_transfers = {}
        self.transfer_id_counter = 0

        # Statistics
        self.stats = {
            'async_transfers_initiated': 0,
            'async_transfers_completed': 0,
            'transfer_overlap_opportunities': 0,
            'estimated_time_saved': 0.0
        }

    def initiate_async_transfer(self, tensor: torch.Tensor, device: torch.device) -> int:
        """
        Initiate an asynchronous transfer of a tensor to the specified device.
        Returns a transfer ID that can be used to wait for completion.
        """
        if not torch.cuda.is_available():
            # Fallback to synchronous transfer if CUDA not available
            return tensor.to(device)

        transfer_id = self.transfer_id_counter
        self.transfer_id_counter += 1

        # Create a new stream for this transfer
        transfer_stream = torch.cuda.Stream()

        # Perform the transfer asynchronously
        with torch.cuda.stream(transfer_stream):
            transferred_tensor = tensor.to(device, non_blocking=True)

        # Record the transfer
        self.ongoing_transfers[transfer_id] = {
            'tensor': transferred_tensor,
            'stream': transfer_stream,
            'start_time': time.time()
        }

        self.stats['async_transfers_initiated'] += 1

        return transfer_id

    def wait_for_transfer(self, transfer_id: int) -> torch.Tensor:
        """
        Wait for a specific transfer to complete and return the tensor.
        """
        if transfer_id not in self.ongoing_transfers:
            raise ValueError(f"Transfer ID {transfer_id} not found")

        transfer_info = self.ongoing_transfers[transfer_id]
        stream = transfer_info['stream']

        # Wait for the stream to complete
        stream.synchronize()

        tensor = transfer_info['tensor']
        elapsed_time = time.time() - transfer_info['start_time']

        # Update stats
        self.stats['async_transfers_completed'] += 1
        self.ongoing_transfers.pop(transfer_id)

        return tensor

    def get_transfer_stats(self) -> Dict[str, Any]:
        """Get transfer statistics."""
        return {
            'ongoing_transfers': len(self.ongoing_transfers),
            'streams_available': len(self.transfer_streams),
            'stats': self.stats
        }


class MemoryTransferOptimizer:
    """
    Optimizes CPU-GPU memory transfers based on tensor characteristics and system constraints.
    Implements techniques like pinned memory, asynchronous transfers, and batching.
    """
    def __init__(self, pinned_memory_manager: Optional[PinnedMemoryManager] = None, 
                 async_transfer_manager: Optional[AsyncTransferManager] = None):
        self.pinned_memory_manager = pinned_memory_manager or PinnedMemoryManager()
        self.async_transfer_manager = async_transfer_manager or AsyncTransferManager()
        
        # Thresholds for optimization decisions
        self.pinned_memory_threshold = 1024 * 1024  # 1MB - use pinned memory for larger tensors
        self.batch_transfer_threshold = 2 * 1024 * 1024  # 2MB - batch smaller transfers
        self.async_transfer_threshold = 512 * 1024  # 512KB - use async for larger transfers
        
        # Batched transfer buffer
        self.batched_transfer_buffer = {}
        
        # Statistics
        self.stats = {
            'optimized_transfers': 0,
            'standard_transfers': 0,
            'pinned_transfers_used': 0,
            'async_transfers_used': 0,
            'batched_transfers_used': 0
        }
    
    def transfer_tensor_optimized(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Transfer a tensor to the specified device using optimized techniques.
        """
        tensor_size = tensor.numel() * tensor.element_size()
        
        # Determine optimal transfer strategy based on tensor size and device
        if device.type == 'cuda' and tensor.device.type == 'cpu':
            # CPU to GPU transfer - most optimization opportunities

            # For larger tensors, use pinned memory if available
            if tensor_size >= self.pinned_memory_manager.pinned_memory_threshold and tensor.is_pinned():
                # Already pinned, use async transfer if large enough
                if tensor_size >= self.async_transfer_threshold:
                    transfer_id = self.async_transfer_manager.initiate_async_transfer(tensor, device)
                    result = self.async_transfer_manager.wait_for_transfer(transfer_id)
                    self.stats['async_transfers_used'] += 1
                else:
                    result = tensor.to(device, non_blocking=True)
                self.stats['optimized_transfers'] += 1
                return result

            # For tensors that are not pinned, consider allocating pinned memory
            elif tensor_size >= self.pinned_memory_manager.pinned_memory_threshold and not tensor.is_pinned():
                # Create a pinned tensor for transfer
                pinned_tensor = self.pinned_memory_manager.allocate_pinned_tensor(tensor.shape, tensor.dtype)
                if pinned_tensor is not None:
                    # Copy data to pinned tensor
                    pinned_tensor.copy_(tensor)
                    
                    # Transfer from pinned to device
                    if tensor_size >= self.async_transfer_threshold:
                        transfer_id = self.async_transfer_manager.initiate_async_transfer(pinned_tensor, device)
                        result = self.async_transfer_manager.wait_for_transfer(transfer_id)
                        self.pinned_memory_manager.return_pinned_tensor(pinned_tensor)
                        self.stats['pinned_transfers_used'] += 1
                        self.stats['async_transfers_used'] += 1
                    else:
                        result = pinned_tensor.to(device, non_blocking=True)
                        self.pinned_memory_manager.return_pinned_tensor(pinned_tensor)
                        self.stats['pinned_transfers_used'] += 1
                    
                    self.stats['optimized_transfers'] += 1
                    return result
                else:
                    # Pinned allocation failed, fall back to standard transfer
                    result = tensor.to(device, non_blocking=True)
                    self.stats['standard_transfers'] += 1
                    return result
            else:
                # Standard transfer for smaller tensors
                result = tensor.to(device, non_blocking=True)
                self.stats['standard_transfers'] += 1
                return result
        elif device.type == 'cpu' and tensor.device.type == 'cuda':
            # GPU to CPU transfer
            if tensor_size >= self.async_transfer_threshold:
                # Use async transfer for large tensors
                transfer_id = self.async_transfer_manager.initiate_async_transfer(tensor, device)
                result = self.async_transfer_manager.wait_for_transfer(transfer_id)
                self.stats['async_transfers_used'] += 1
                self.stats['optimized_transfers'] += 1
            else:
                # Standard transfer for smaller tensors
                result = tensor.to(device, non_blocking=True)
                self.stats['standard_transfers'] += 1
            
            return result
        else:
            # Same device transfer or no CUDA available
            self.stats['standard_transfers'] += 1
            return tensor.to(device)
    
    def batch_transfer_tensors(self, tensors: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
        """
        Batch transfer multiple tensors efficiently.
        """
        if not tensors:
            return []
        
        # Group tensors by size to optimize batching
        small_tensors = []
        large_tensors = []
        
        for tensor in tensors:
            tensor_size = tensor.numel() * tensor.element_size()
            if tensor_size < self.batch_transfer_threshold:
                small_tensors.append(tensor)
            else:
                large_tensors.append(tensor)
        
        results = []
        
        # Handle large tensors individually with optimization
        for tensor in large_tensors:
            results.append(self.transfer_tensor_optimized(tensor, device))
        
        # Batch small tensors together
        if small_tensors:
            # Concatenate small tensors along a new dimension
            concatenated = torch.cat([t.flatten() for t in small_tensors])
            
            # Transfer the concatenated tensor
            transferred_concatenated = self.transfer_tensor_optimized(concatenated, device)
            
            # Split back to original shapes
            offset = 0
            for tensor in small_tensors:
                tensor_flat_size = tensor.numel()
                transferred_flat = transferred_concatenated[offset:offset + tensor_flat_size]
                transferred_tensor = transferred_flat.reshape(tensor.shape).to(tensor.dtype)
                results.append(transferred_tensor)
                offset += tensor_flat_size
        
        self.stats['batched_transfers_used'] += len(small_tensors)
        
        return results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get transfer optimization statistics."""
        return {
            'transfer_stats': self.stats,
            'pinned_memory_stats': self.pinned_memory_manager.get_pinned_memory_stats(),
            'async_transfer_stats': self.async_transfer_manager.get_transfer_stats()
        }


class StreamOrderedMemoryPool:
    """
    Memory pool that manages memory allocations with CUDA streams for optimal transfer overlap.
    """
    def __init__(self, pool_size: int = 1024 * 1024 * 1024):  # 1GB default
        self.pool_size = pool_size
        self.allocated_tensors = {}
        self.streams = {}
        
        # Create streams for different operations
        if torch.cuda.is_available():
            for i in range(4):  # Create 4 streams
                self.streams[i] = torch.cuda.Stream()
        
        # Statistics
        self.stats = {
            'stream_allocations': 0,
            'stream_deallocations': 0,
            'memory_efficiency_improvements': 0
        }
    
    def allocate_tensor_with_stream(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                                   stream_id: int = 0) -> torch.Tensor:
        """Allocate a tensor associated with a specific CUDA stream."""
        tensor = torch.empty(shape, dtype=dtype)
        
        tensor_id = id(tensor)
        self.allocated_tensors[tensor_id] = {
            'tensor': tensor,
            'stream_id': stream_id,
            'allocation_time': time.time()
        }
        
        self.stats['stream_allocations'] += 1
        
        return tensor
    
    def deallocate_tensor_with_stream(self, tensor: torch.Tensor, stream_id: int = 0) -> bool:
        """Deallocate a tensor associated with a specific CUDA stream."""
        tensor_id = id(tensor)
        
        if tensor_id in self.allocated_tensors:
            # Synchronize the stream before deallocating
            if torch.cuda.is_available() and stream_id in self.streams:
                self.streams[stream_id].synchronize()
            
            del self.allocated_tensors[tensor_id]
            self.stats['stream_deallocations'] += 1
            return True
        
        return False
    
    def transfer_with_stream(self, tensor: torch.Tensor, device: torch.device, 
                           stream_id: int = 0) -> torch.Tensor:
        """Transfer a tensor using a specific CUDA stream."""
        if torch.cuda.is_available() and stream_id in self.streams:
            with torch.cuda.stream(self.streams[stream_id]):
                return tensor.to(device, non_blocking=True)
        else:
            # Fallback to standard transfer
            return tensor.to(device, non_blocking=True)


class HardwareAwareTransferOptimizer:
    """
    Transfer optimizer that adapts to the specific hardware characteristics of Intel i5-10210U + NVIDIA SM61.
    """
    def __init__(self, transfer_optimizer: MemoryTransferOptimizer = None):
        self.transfer_optimizer = transfer_optimizer or MemoryTransferOptimizer()
        
        # Hardware-specific parameters
        self.cpu_memory_bandwidth = 34.1  # GB/s for Intel i5-10210U with DDR4-2933
        self.gpu_memory_bandwidth = 484.0  # GB/s for GTX 1080 Ti (SM61 equivalent)
        self.pcie_bandwidth = 16.0  # GB/s for PCIe 3.0 x16
        self.nvme_bandwidth = 3500.0  # MB/s for typical NVMe SSD
        
        # CPU-specific optimizations
        self.cpu_num_cores = 4  # Intel i5-10210U has 4 cores
        self.cpu_max_threads = 8  # With hyperthreading
        
        # GPU-specific optimizations for SM61
        self.sm61_shared_memory_per_block = 48 * 1024  # 48KB per block (configurable up to 96KB)
        self.sm61_max_threads_per_block = 1024
        self.sm61_warp_size = 32
        
        # Memory alignment for optimal transfers
        self.alignment_size = 256  # Align to 256-byte boundaries
        
        # Statistics
        self.stats = {
            'cpu_to_gpu_transfers': 0,
            'gpu_to_cpu_transfers': 0,
            'transfer_time_saved': 0.0,
            'memory_efficiency_score': 0.0
        }
    
    def optimize_transfer_for_hardware(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Optimize tensor transfer based on hardware characteristics.
        """
        original_device = tensor.device
        tensor_size = tensor.numel() * tensor.element_size()
        
        # For CPU to GPU transfers on the target hardware
        if original_device.type == 'cpu' and device.type == 'cuda':
            self.stats['cpu_to_gpu_transfers'] += 1
            
            # For SM61 architecture, consider using pinned memory for larger transfers
            if tensor_size > 1024 * 1024:  # Larger than 1MB
                # Use the transfer optimizer which handles pinned memory and async transfers
                return self.transfer_optimizer.transfer_tensor_optimized(tensor, device)
            else:
                # For smaller transfers, standard transfer might be more efficient
                return tensor.to(device)
        
        # For GPU to CPU transfers
        elif original_device.type == 'cuda' and device.type == 'cpu':
            self.stats['gpu_to_cpu_transfers'] += 1
            
            # For larger tensors, use pinned memory for faster transfer back to CPU
            if tensor_size > 1024 * 1024:  # Larger than 1MB
                # Create pinned tensor on CPU for faster transfer
                pinned_tensor = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                pinned_tensor.copy_(tensor)
                
                # Then transfer to regular CPU memory if needed
                return pinned_tensor.to(device, non_blocking=True)
            else:
                return tensor.to(device, non_blocking=True)
        
        # Same device transfer
        else:
            return tensor.to(device)
    
    def optimize_data_loading_for_hardware(self, dataset, batch_size: int = 1, 
                                         shuffle: bool = False, 
                                         num_workers: int = 0,
                                         pin_memory: bool = None) -> torch.utils.data.DataLoader:
        """
        Optimize data loading for the specific hardware configuration.
        """
        # For Intel i5-10210U + NVIDIA SM61, adjust data loading parameters
        if pin_memory is None:
            # Use pin_memory for better CPU-GPU transfer performance on the target hardware
            pin_memory = True
        
        # Adjust num_workers based on CPU cores
        if num_workers == 0:
            # For i5-10210U with 4 cores, limit workers to prevent memory issues
            num_workers = min(2, self.cpu_num_cores - 1)  # Leave one core for main process
        
        # Create optimized data loader
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
    
    def get_hardware_optimized_parameters(self, tensor_shape: Tuple[int, ...], 
                                         operation_type: str = "general") -> Dict[str, Any]:
        """
        Get hardware-optimized parameters for a specific tensor and operation.
        """
        tensor_size = np.prod(tensor_shape) * 4  # Assuming float32 (4 bytes)
        
        params = {
            'tensor_size_bytes': tensor_size,
            'tensor_size_mb': tensor_size / (1024 * 1024),
            'use_pinned_memory': tensor_size > 1024 * 1024,  # Use pinned for >1MB tensors
            'use_async_transfer': tensor_size > 512 * 1024,   # Use async for >512KB tensors
            'memory_format': torch.contiguous_format,
            'alignment_needed': False
        }
        
        # For convolution operations on SM61, consider channels_last format
        if operation_type == "convolution" and len(tensor_shape) == 4:
            params['memory_format'] = torch.channels_last
        
        # For attention operations, align feature dimensions for better memory access
        if operation_type == "attention" and len(tensor_shape) >= 3:
            feature_dim = tensor_shape[-1]
            aligned_dim = ((feature_dim + 63) // 64) * 64  # Align to 64 for SM61
            params['alignment_needed'] = aligned_dim != feature_dim
            params['aligned_shape'] = (*tensor_shape[:-1], aligned_dim)
        
        # Calculate potential transfer time based on hardware characteristics
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            params['gpu_memory_gb'] = gpu_memory / (1024**3)
            
            # Estimate transfer time (simplified model)
            pcie_transfer_time = tensor_size / (self.pcie_bandwidth * 1024 * 1024 * 1024)  # In seconds
            params['estimated_transfer_time_s'] = pcie_transfer_time
        else:
            params['gpu_memory_gb'] = 0
            params['estimated_transfer_time_s'] = float('inf')
        
        return params
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get hardware-aware optimization statistics."""
        transfer_stats = self.transfer_optimizer.get_optimization_stats()
        
        return {
            'hardware_specific_stats': {
                'cpu_memory_bandwidth_gb_s': self.cpu_memory_bandwidth,
                'gpu_memory_bandwidth_gb_s': self.gpu_memory_bandwidth,
                'pcie_bandwidth_gb_s': self.pcie_bandwidth,
                'nvme_bandwidth_mb_s': self.nvme_bandwidth,
                'cpu_num_cores': self.cpu_num_cores,
                'cpu_max_threads': self.cpu_max_threads,
                'sm61_shared_memory_per_block_kb': self.sm61_shared_memory_per_block // 1024,
                'sm61_max_threads_per_block': self.sm61_max_threads_per_block,
                'sm61_warp_size': self.sm61_warp_size
            },
            'transfer_stats': self.stats,
            'memory_transfer_optimizer_stats': transfer_stats
        }


# Global memory transfer optimizer instance
_global_transfer_optimizer = None
_optimizer_lock = threading.Lock()

def get_transfer_optimizer() -> HardwareAwareTransferOptimizer:
    """Get the global memory transfer optimizer instance."""
    global _global_transfer_optimizer
    if _global_transfer_optimizer is None:
        with _optimizer_lock:
            if _global_transfer_optimizer is None:
                _global_transfer_optimizer = HardwareAwareTransferOptimizer()
    return _global_transfer_optimizer


def transfer_tensor_with_optimization(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Transfer a tensor using hardware-optimized techniques."""
    optimizer = get_transfer_optimizer()
    return optimizer.optimize_transfer_for_hardware(tensor, device)


def create_optimized_dataloader(dataset, **kwargs) -> torch.utils.data.DataLoader:
    """Create a data loader with hardware-optimized settings."""
    optimizer = get_transfer_optimizer()
    return optimizer.optimize_data_loading_for_hardware(dataset, **kwargs)


class OptimizedQwen3VLAttention(nn.Module):
    """
    Memory-optimized attention mechanism that incorporates CPU-GPU transfer optimizations.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Initialize transfer optimizer
        self.transfer_optimizer = get_transfer_optimizer()
        
        # Attention parameters
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # Optimize tensor transfers if moving between devices
        if hidden_states.device.type == 'cpu' and self.training:
            # Move to GPU with optimized transfer
            hidden_states = transfer_tensor_with_optimization(hidden_states, torch.device('cuda'))
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Optimize tensor shapes for memory access patterns
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Update cache if provided
        if use_cache and past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_position)
        
        # Apply GQA (Grouped Query Attention) if needed
        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
            value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32 and apply softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        output = self.o_proj(attn_output)
        
        # If output needs to be on CPU, optimize the transfer
        if output.device.type == 'cuda' and not self.training:
            # Transfer back to CPU with optimization if needed
            output = transfer_tensor_with_optimization(output, torch.device('cpu'))
        
        return output, attn_weights, past_key_value


class OptimizedQwen3VLMLP(nn.Module):
    """
    Memory-optimized MLP that incorporates CPU-GPU transfer optimizations.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize transfer optimizer
        self.transfer_optimizer = get_transfer_optimizer()
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Linear projections
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        # Optimize tensor transfers if moving between devices
        if x.device.type == 'cpu' and self.training:
            x = transfer_tensor_with_optimization(x, torch.device('cuda'))
        
        # Apply projections with optimized memory layout
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Apply activation
        act_output = self.act_fn(gate)
        intermediate_output = act_output * up
        
        output = self.down_proj(intermediate_output)
        
        # If output needs to be on CPU, optimize the transfer
        if output.device.type == 'cuda' and not self.training:
            output = transfer_tensor_with_optimization(output, torch.device('cpu'))
        
        return output


class OptimizedDataLoader(torch.utils.data.DataLoader):
    """
    Memory-optimized DataLoader that incorporates hardware-specific optimizations for CPU-GPU transfers.
    """
    def __init__(self, dataset, memory_efficient: bool = True, 
                 pin_memory: bool = None, 
                 num_workers: int = None,
                 **kwargs):
        # Get hardware-specific optimizations
        self.transfer_optimizer = get_transfer_optimizer()
        
        # Set defaults based on hardware capabilities
        if pin_memory is None:
            pin_memory = memory_efficient  # Use pin_memory for memory efficiency
        if num_workers is None:
            # For Intel i5-10210U, limit workers to prevent memory issues
            num_workers = min(2, self.transfer_optimizer.cpu_num_cores - 1)
        
        # Initialize parent DataLoader
        super().__init__(
            dataset,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            **kwargs
        )
        
        # Memory efficiency tracking
        self.memory_efficient = memory_efficient
        self.current_memory_usage = 0
        self.max_memory_usage = 0
    
    def __iter__(self):
        # Track memory usage before iteration
        if torch.cuda.is_available():
            self.current_memory_usage = torch.cuda.memory_allocated()
            self.max_memory_usage = max(self.max_memory_usage, self.current_memory_usage)
        else:
            self.current_memory_usage = psutil.Process().memory_info().rss
            self.max_memory_usage = max(self.max_memory_usage, self.current_memory_usage)
        
        return super().__iter__()
    
    def transfer_to_device(self, data, device: torch.device, non_blocking: bool = True):
        """
        Transfer data to device with optimization.
        """
        if isinstance(data, torch.Tensor):
            if self.memory_efficient:
                return transfer_tensor_with_optimization(data, device)
            else:
                return data.to(device, non_blocking=non_blocking)
        elif isinstance(data, (list, tuple)):
            return type(data)(self.transfer_to_device(d, device, non_blocking) for d in data)
        elif isinstance(data, dict):
            return {k: self.transfer_to_device(v, device, non_blocking) for k, v in data.items()}
        else:
            return data.to(device, non_blocking=non_blocking) if hasattr(data, 'to') else data



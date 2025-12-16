"""Memory Access Pattern Optimization for Cache Efficiency in Qwen3-VL Model"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import math
import time
from collections import defaultdict, deque
import threading
import logging
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryAccessType(Enum):
    """Types of memory access patterns"""
    SEQUENTIAL = "sequential"
    STRIDED = "strided"
    RANDOM = "random"
    COALESCED = "coalesced"

class CacheLineOptimizer:
    """
    Optimizes memory access patterns for cache efficiency on target hardware.
    Specifically optimized for Intel i5-10210U (CPU cache lines) + NVIDIA SM61 (GPU memory transactions).
    """
    def __init__(self, cache_line_size: int = 64, shared_memory_size: int = 48 * 1024):
        self.cache_line_size = cache_line_size  # 64 bytes for most modern CPUs
        self.shared_memory_size = shared_memory_size  # 48KB for SM61
        self.alignment_size = 256  # Align to 256-byte boundaries for optimal access
        
        # Track access patterns for optimization
        self.access_patterns = defaultdict(deque)
        self.pattern_history_size = 100
        
        # For SM61 architecture, optimize for warp size (32 threads)
        self.warp_size = 32
        self.sm61_optimized = True
        
        # Statistics
        self.stats = {
            'optimized_accesses': 0,
            'total_accesses': 0,
            'cache_hit_predictions': 0,
            'cache_miss_predictions': 0
        }
    
    def align_tensor_size(self, tensor_size: int) -> int:
        """Align tensor size to cache line boundaries for optimal memory access."""
        return ((tensor_size + self.alignment_size - 1) // self.alignment_size) * self.alignment_size
    
    def optimize_for_memory_access(self, tensor: torch.Tensor, access_type: MemoryAccessType = MemoryAccessType.SEQUENTIAL) -> torch.Tensor:
        """
        Optimize tensor memory layout for the specified access pattern.
        """
        self.stats['total_accesses'] += 1
        
        # For different access patterns, use different memory layouts
        if access_type == MemoryAccessType.COALESCED:
            # For GPU coalesced access, ensure memory layout supports efficient access
            if tensor.dim() == 4:  # Convolutional layers
                # Use channels_last format for better memory bandwidth utilization
                try:
                    return tensor.to(memory_format=torch.channels_last)
                except:
                    return tensor  # Return original if conversion fails
            elif tensor.dim() == 3:  # Attention matrices [batch, seq, features]
                # Ensure features dimension is aligned for coalesced access
                batch, seq, features = tensor.shape
                aligned_features = self.align_for_gpu_access(features)
                if aligned_features != features:
                    # Pad tensor if needed
                    padded_tensor = torch.zeros(batch, seq, aligned_features, dtype=tensor.dtype, device=tensor.device)
                    padded_tensor[:, :, :features] = tensor
                    return padded_tensor
                else:
                    return tensor
        elif access_type == MemoryAccessType.SEQUENTIAL:
            # For sequential access, use contiguous format
            return tensor.contiguous()
        elif access_type == MemoryAccessType.STRIDED:
            # For strided access, consider interleaving patterns
            if tensor.dim() >= 2:
                # Transpose to optimize for strided access if beneficial
                transposed_tensor = tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
                return transposed_tensor
            else:
                # For tensors with less than 2 dimensions, return as is
                return tensor
        else:
            # For random access, no specific optimization
            return tensor
    
    def align_for_gpu_access(self, dimension_size: int) -> int:
        """
        Align dimension size for optimal GPU memory access on SM61 architecture.
        This considers warp size and memory transaction efficiency.
        """
        # For SM61, align to multiples that work well with 32-thread warps
        # and memory transactions (typically 32 or 64 bytes)
        if dimension_size <= 64:
            # For small dimensions, align to warp size
            return ((dimension_size + self.warp_size - 1) // self.warp_size) * self.warp_size
        elif dimension_size <= 256:
            # For medium dimensions, align to 64 for better memory transactions
            return ((dimension_size + 63) // 64) * 64
        else:
            # For larger dimensions, align to 256 for optimal cache utilization
            return ((dimension_size + 255) // 256) * 256
    
    def predict_cache_performance(self, tensor_shape: Tuple[int, ...], access_pattern: MemoryAccessType) -> Dict[str, float]:
        """
        Predict cache performance for a given tensor shape and access pattern.
        """
        # Calculate tensor size in bytes
        total_elements = np.prod(tensor_shape)
        element_size = 4  # Assuming float32
        tensor_size_bytes = total_elements * element_size
        
        # Estimate cache performance based on access pattern
        if access_pattern == MemoryAccessType.COALESCED:
            # Coalesced access typically has better performance
            predicted_hit_rate = min(0.95, 1.0 - (tensor_size_bytes / (2 * 1024 * 1024)))  # Higher for smaller tensors
        elif access_pattern == MemoryAccessType.SEQUENTIAL:
            # Sequential access also has good performance
            predicted_hit_rate = min(0.90, 1.0 - (tensor_size_bytes / (4 * 1024 * 1024)))
        elif access_pattern == MemoryAccessType.STRIDED:
            # Strided access has moderate performance
            predicted_hit_rate = min(0.75, 0.85 - (tensor_size_bytes / (8 * 1024 * 1024)))
        else:
            # Random access has poor cache performance
            predicted_hit_rate = min(0.60, 0.70 - (tensor_size_bytes / (16 * 1024 * 1024)))
        
        # Adjust for SM61 architecture characteristics
        if self.sm61_optimized:
            # SM61 has good memory bandwidth but limited shared memory
            predicted_hit_rate *= 1.05  # Slight boost for GPU-optimized access
        
        predicted_latency = 1.0 / max(predicted_hit_rate, 0.1)  # Inverse relationship
        
        self.stats['cache_hit_predictions'] += 1
        
        return {
            'predicted_cache_hit_rate': predicted_hit_rate,
            'predicted_latency': predicted_latency,
            'tensor_size_mb': tensor_size_bytes / (1024 * 1024)
        }
    
    def get_access_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about memory access patterns."""
        return {
            'stats': self.stats.copy(),
            'cache_line_size': self.cache_line_size,
            'shared_memory_size': self.shared_memory_size,
            'alignment_size': self.alignment_size
        }


class MemoryLayoutOptimizer:
    """
    Optimizes memory layouts for different operations in the Qwen3-VL model.
    Focuses on cache efficiency for both CPU and GPU operations.
    """
    def __init__(self, cache_optimizer: CacheLineOptimizer = None):
        self.cache_optimizer = cache_optimizer or CacheLineOptimizer()
        
        # Operation-specific memory layout configurations
        self.operation_configs = {
            'attention': {
                'memory_format': torch.contiguous_format,
                'alignment': 64,  # Align to 64 for attention operations
                'access_pattern': MemoryAccessType.COALESCED
            },
            'mlp': {
                'memory_format': torch.contiguous_format,
                'alignment': 256,  # Align to 256 for MLP operations
                'access_pattern': MemoryAccessType.SEQUENTIAL
            },
            'convolution': {
                'memory_format': torch.channels_last,
                'alignment': 32,   # Align to 32 for convolution operations
                'access_pattern': MemoryAccessType.COALESCED
            },
            'embedding': {
                'memory_format': torch.contiguous_format,
                'alignment': 128,  # Align to 128 for embedding operations
                'access_pattern': MemoryAccessType.RANDOM
            },
            'norm': {
                'memory_format': torch.contiguous_format,
                'alignment': 64,   # Align to 64 for normalization operations
                'access_pattern': MemoryAccessType.SEQUENTIAL
            }
        }

    def optimize_tensor_layout(self, tensor: torch.Tensor, operation_type: str = 'general') -> torch.Tensor:
        """
        Optimize tensor memory layout based on operation type.
        """
        if operation_type not in self.operation_configs:
            operation_type = 'general'

        config = self.operation_configs.get(operation_type, self.operation_configs['general'])
        
        # Apply memory format optimization
        if tensor.dim() == 4 and config['memory_format'] == torch.channels_last:
            try:
                tensor = tensor.to(memory_format=torch.channels_last)
            except:
                # Fallback to contiguous if channels_last fails
                tensor = tensor.contiguous()
        else:
            tensor = tensor.contiguous()

        # Apply alignment optimization
        aligned_tensor = self._align_tensor_for_operation(tensor, config['alignment'], config['access_pattern'])

        return aligned_tensor
    
    def _align_tensor_for_operation(self, tensor: torch.Tensor, alignment: int, access_pattern: MemoryAccessType) -> torch.Tensor:
        """
        Align tensor dimensions for optimal memory access in the specified operation.
        """
        original_shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype
        
        # For tensors with feature dimensions, align the last dimension
        if len(tensor.shape) > 0:
            last_dim = tensor.shape[-1]
            aligned_last_dim = ((last_dim + alignment - 1) // alignment) * alignment
            
            if aligned_last_dim != last_dim:
                # Create a new tensor with aligned dimensions
                new_shape = list(tensor.shape)
                new_shape[-1] = aligned_last_dim
                
                # Create aligned tensor
                aligned_tensor = torch.zeros(new_shape, dtype=dtype, device=device)
                
                # Copy original data to aligned tensor
                slice_indices = [slice(None)] * len(new_shape)
                slice_indices[-1] = slice(0, last_dim)
                aligned_tensor[tuple(slice_indices)] = tensor
                
                # Optimize for access pattern
                optimized_tensor = self.cache_optimizer.optimize_for_memory_access(aligned_tensor, access_pattern)
                return optimized_tensor
            else:
                # Apply access pattern optimization directly
                return self.cache_optimizer.optimize_for_memory_access(tensor, access_pattern)
        else:
            return tensor
    
    def optimize_attention_memory_layout(self, 
                                       query: torch.Tensor, 
                                       key: torch.Tensor, 
                                       value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimize memory layout specifically for attention operations.
        """
        # Optimize for coalesced memory access on GPU
        optimized_query = self.optimize_tensor_layout(query, 'attention')
        optimized_key = self.optimize_tensor_layout(key, 'attention')
        optimized_value = self.optimize_tensor_layout(value, 'attention')
        
        return optimized_query, optimized_key, optimized_value
    
    def optimize_mlp_memory_layout(self, 
                                 intermediate_tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize memory layout specifically for MLP operations.
        """
        return self.optimize_tensor_layout(intermediate_tensor, 'mlp')
    
    def optimize_conv_memory_layout(self, 
                                  input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize memory layout specifically for convolution operations.
        """
        return self.optimize_tensor_layout(input_tensor, 'convolution')
    
    def get_optimal_tile_sizes(self, 
                             tensor_shape: Tuple[int, ...], 
                             device_type: str = 'cuda') -> Tuple[int, int, int]:
        """
        Get optimal tile sizes for matrix operations based on tensor shape and device.
        """
        if len(tensor_shape) < 2:
            return 1, 1, 1  # Default tile sizes for scalar operations
            
        # For SM61 architecture, optimize tile sizes based on shared memory and warp size
        if device_type == 'cuda':
            # SM61 has 48KB shared memory per block and 32 threads per warp
            max_shared_memory = self.cache_optimizer.shared_memory_size
            
            # Calculate optimal tile sizes to fit in shared memory while maximizing occupancy
            # For attention operations: tiles need to store Q, K, V, and attention scores
            seq_len = tensor_shape[-2] if len(tensor_shape) >= 2 else tensor_shape[0]
            feature_dim = tensor_shape[-1] if len(tensor_shape) >= 1 else tensor_shape[0]
            
            # For attention, we need to consider both sequence and feature dimensions
            if len(tensor_shape) >= 3:  # Likely attention: [batch, seq, features]
                # Optimize for sequence length first
                tile_size_seq = min(128, seq_len)  # Limit to 128 for attention
                tile_size_feature = min(64, feature_dim)  # Limit to 64 for features
                
                # Ensure tile sizes are multiples of warp size for coalesced access
                tile_size_seq = ((tile_size_seq + self.cache_optimizer.warp_size - 1) // 
                                self.cache_optimizer.warp_size) * self.cache_optimizer.warp_size
                tile_size_feature = ((tile_size_feature + 31) // 32) * 32  # Align to 32
            else:
                # For other operations, use general optimization
                tile_size_seq = min(64, int(math.sqrt(max_shared_memory / (4 * 4))))  # Max tile size for 4-byte elements
                tile_size_feature = min(64, feature_dim)
                
                # Align to warp size
                tile_size_seq = ((tile_size_seq + self.cache_optimizer.warp_size - 1) // 
                                self.cache_optimizer.warp_size) * self.cache_optimizer.warp_size
                tile_size_feature = ((tile_size_feature + 31) // 32) * 32
                
            # Third dimension is typically batch or heads
            tile_size_batch = min(8, tensor_shape[0] if len(tensor_shape) > 0 else 1)
            
            return tile_size_seq, tile_size_feature, tile_size_batch
        else:
            # For CPU, optimize for cache lines (64 bytes)
            # Typical tile sizes for CPU cache efficiency
            tile_size_seq = min(32, tensor_shape[-2] if len(tensor_shape) >= 2 else tensor_shape[0])
            tile_size_feature = min(16, tensor_shape[-1] if len(tensor_shape) >= 1 else tensor_shape[0])
            tile_size_batch = min(4, tensor_shape[0] if len(tensor_shape) > 0 else 1)
            
            return tile_size_seq, tile_size_feature, tile_size_batch
    
    def get_memory_access_prediction(self, tensor_shape: Tuple[int, ...], operation_type: str) -> Dict[str, float]:
        """
        Get memory access performance prediction for a tensor in a specific operation.
        """
        access_pattern = self.operation_configs.get(operation_type, self.operation_configs['general'])['access_pattern']
        return self.cache_optimizer.predict_cache_performance(tensor_shape, access_pattern)


class MemoryAccessPatternTracker:
    """
    Tracks memory access patterns during model execution to identify optimization opportunities.
    """
    def __init__(self):
        self.access_pattern_history = deque(maxlen=1000)
        self.tensor_access_stats = defaultdict(lambda: {'count': 0, 'total_size': 0, 'access_pattern': []})
        
        # Thread safety
        self.lock = threading.RLock()
    
    def record_tensor_access(self, tensor: torch.Tensor, operation_type: str, layer_idx: int = None):
        """
        Record a tensor access event for pattern analysis.
        """
        with self.lock:
            tensor_shape = tuple(tensor.shape)
            tensor_size = tensor.numel() * tensor.element_size()
            device_type = tensor.device.type
            
            key = (tensor_shape, tensor.dtype, device_type, operation_type, layer_idx)
            
            # Record access
            self.access_pattern_history.append({
                'timestamp': time.time(),
                'shape': tensor_shape,
                'dtype': tensor.dtype,
                'device': device_type,
                'operation_type': operation_type,
                'layer_idx': layer_idx,
                'size_bytes': tensor_size,
                'element_count': tensor.numel()
            })
            
            # Update tensor-specific stats
            self.tensor_access_stats[key]['count'] += 1
            self.tensor_access_stats[key]['total_size'] += tensor_size
            self.tensor_access_stats[key]['access_pattern'].append(operation_type)
    
    def get_access_patterns_for_optimization(self) -> List[Dict[str, Any]]:
        """
        Get tensor access patterns that offer optimization opportunities.
        """
        with self.lock:
            optimization_opportunities = []
            
            for key, stats in self.tensor_access_stats.items():
                shape, dtype, device, operation_type, layer_idx = key
                access_count = stats['count']
                total_size = stats['total_size']
                
                # Identify tensors that are accessed frequently and would benefit from layout optimization
                if access_count >= 5:  # Accessed at least 5 times
                    avg_size = total_size / access_count
                    
                    # Calculate potential optimization benefit
                    # Larger tensors and frequently accessed tensors have higher optimization potential
                    optimization_potential = avg_size * access_count
                    
                    if optimization_potential > 1024 * 1024:  # More than 1MB of access potential
                        opportunity = {
                            'shape': shape,
                            'dtype': dtype,
                            'device': device,
                            'operation_type': operation_type,
                            'layer_idx': layer_idx,
                            'access_count': access_count,
                            'total_size_mb': total_size / (1024 * 1024),
                            'avg_size_mb': avg_size / (1024 * 1024),
                            'optimization_potential_score': optimization_potential,
                            'recommended_optimization': self._recommend_optimization(shape, operation_type, device)
                        }
                        optimization_opportunities.append(opportunity)
            
            # Sort by optimization potential score (descending)
            optimization_opportunities.sort(key=lambda x: x['optimization_potential_score'], reverse=True)
            
            return optimization_opportunities
    
    def _recommend_optimization(self, shape: Tuple[int, ...], operation_type: str, device: str) -> str:
        """
        Recommend specific optimizations based on tensor shape, operation type, and device.
        """
        if device == 'cuda':
            if len(shape) == 4:  # Convolutional layer
                return f"Use channels_last memory format for shape {shape}"
            elif len(shape) == 3 and operation_type == 'attention':  # Attention matrices
                return f"Align feature dimension to multiple of 64 for shape {shape}"
            elif len(shape) == 2:  # Linear layer weights
                return f"Align to optimal tile sizes for matrix multiplication"
            else:
                return f"Optimize for coalesced memory access on GPU"
        else:  # CPU
            return f"Optimize for cache line alignment for shape {shape}"
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """
        Get a summary of memory access patterns.
        """
        with self.lock:
            if not self.access_pattern_history:
                return {'error': 'No access patterns recorded'}
            
            # Calculate statistics
            total_accesses = len(self.access_pattern_history)
            total_size_accessed = sum(item['size_bytes'] for item in self.access_pattern_history)
            unique_tensors = len(self.tensor_access_stats)
            
            # Most common operation types
            operation_counts = defaultdict(int)
            for item in self.access_pattern_history:
                operation_counts[item['operation_type']] += 1
            
            # Most common layer indices (if available)
            layer_counts = defaultdict(int)
            for item in self.access_pattern_history:
                if item['layer_idx'] is not None:
                    layer_counts[item['layer_idx']] += 1
            
            # Device distribution
            device_counts = defaultdict(int)
            for item in self.access_pattern_history:
                device_counts[item['device']] += 1
            
            return {
                'total_accesses': total_accesses,
                'total_size_accessed_mb': total_size_accessed / (1024 * 1024),
                'unique_tensors_accessed': unique_tensors,
                'most_common_operations': dict(operation_counts),
                'layer_access_distribution': dict(layer_counts),
                'device_access_distribution': dict(device_counts),
                'average_tensor_size_mb': (total_size_accessed / total_accesses) / (1024 * 1024) if total_accesses > 0 else 0
            }


class HardwareSpecificMemoryOptimizer:
    """
    Memory optimizer that adapts to specific hardware characteristics (Intel i5-10210U + NVIDIA SM61).
    """
    def __init__(self, cpu_cache_line_size: int = 64, gpu_shared_memory_per_block: int = 48 * 1024, 
                 memory_bandwidth_gb_s: float = 192.0):
        self.cpu_cache_line_size = cpu_cache_line_size
        self.gpu_shared_memory_per_block = gpu_shared_memory_per_block
        self.memory_bandwidth_gb_s = memory_bandwidth_gb_s
        
        # Initialize optimizers
        self.cache_optimizer = CacheLineOptimizer(
            cache_line_size=cpu_cache_line_size, 
            shared_memory_size=gpu_shared_memory_per_block
        )
        self.layout_optimizer = MemoryLayoutOptimizer(self.cache_optimizer)
        self.pattern_tracker = MemoryAccessPatternTracker()
        
        # Hardware-specific configurations
        self.hardware_configs = {
            'cpu': {
                'cache_line_size': cpu_cache_line_size,
                'l1_cache_size': 32 * 1024,  # 32KB per core
                'l2_cache_size': 256 * 1024, # 256KB per core
                'l3_cache_size': 6 * 1024 * 1024,  # 6MB shared
                'memory_alignment': 64
            },
            'gpu': {
                'shared_memory_per_block': gpu_shared_memory_per_block,
                'warp_size': 32,
                'max_threads_per_block': 1024,
                'memory_alignment': 128,
                'memory_bandwidth_gb_s': memory_bandwidth_gb_s
            }
        }
    
    def optimize_tensor_for_hardware(self, tensor: torch.Tensor, operation_type: str, 
                                   layer_idx: int = None) -> torch.Tensor:
        """
        Optimize tensor for specific hardware based on operation type and layer.
        """
        # Record access pattern
        self.pattern_tracker.record_tensor_access(tensor, operation_type, layer_idx)
        
        # Apply layout optimization based on operation type
        optimized_tensor = self.layout_optimizer.optimize_tensor_layout(tensor, operation_type)
        
        # For GPU tensors on SM61, consider additional optimizations
        if tensor.device.type == 'cuda':
            # Optimize for SM61's memory hierarchy
            optimized_tensor = self._optimize_for_sm61(optimized_tensor, operation_type)
        
        return optimized_tensor
    
    def _optimize_for_sm61(self, tensor: torch.Tensor, operation_type: str) -> torch.Tensor:
        """
        Apply SM61-specific optimizations to the tensor.
        """
        if operation_type == 'attention':
            # For attention operations on SM61, optimize for 48KB shared memory
            # and coalesced memory access patterns
            if tensor.dim() == 3:  # [batch, seq, features]
                # Ensure feature dimension is aligned for optimal memory transactions
                batch, seq, features = tensor.shape
                aligned_features = ((features + 63) // 64) * 64  # Align to 64
                if aligned_features != features:
                    # Pad tensor if needed
                    padded_tensor = torch.zeros(batch, seq, aligned_features, dtype=tensor.dtype, device=tensor.device)
                    padded_tensor[:, :, :features] = tensor
                    return padded_tensor
        elif operation_type == 'mlp':
            # For MLP operations, ensure dimensions are multiples that work well with tensor cores
            # SM61 doesn't have tensor cores, but we optimize for 32-byte memory transactions
            if tensor.dim() == 2:  # [input_features, output_features]
                in_features, out_features = tensor.shape
                aligned_in = ((in_features + 31) // 32) * 32  # Align to 32
                aligned_out = ((out_features + 31) // 32) * 32  # Align to 32
                
                if aligned_in != in_features or aligned_out != out_features:
                    # Pad tensor if needed
                    padded_tensor = torch.zeros(aligned_in, aligned_out, dtype=tensor.dtype, device=tensor.device)
                    padded_tensor[:in_features, :out_features] = tensor
                    return padded_tensor
        elif operation_type == 'convolution':
            # For convolution operations on SM61, use channels_last memory format
            if tensor.dim() == 4:  # [batch, channels, height, width]
                try:
                    return tensor.to(memory_format=torch.channels_last)
                except:
                    # Fallback if channels_last is not supported
                    return tensor.contiguous()
        
        return tensor.contiguous()
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get optimization recommendations based on access patterns and hardware characteristics.
        """
        # Get access patterns that need optimization
        opportunities = self.pattern_tracker.get_access_patterns_for_optimization()
        
        recommendations = []
        for opp in opportunities:
            recommendation = {
                'tensor_shape': opp['shape'],
                'operation_type': opp['operation_type'],
                'access_count': opp['access_count'],
                'total_size_mb': opp['total_size_mb'],
                'suggested_optimization': opp['recommended_optimization'],
                'hardware_specific_advice': self._get_hardware_specific_advice(opp)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_hardware_specific_advice(self, opportunity: Dict[str, Any]) -> str:
        """
        Get hardware-specific optimization advice for an opportunity.
        """
        shape = opportunity['shape']
        operation_type = opportunity['operation_type']
        device = opportunity['device']
        
        if device == 'cuda':
            if operation_type == 'attention':
                # SM61 attention optimization advice
                if len(shape) >= 3:
                    seq_len = shape[-2]
                    feature_dim = shape[-1]
                    if seq_len > 1024:
                        return "Consider using sparse attention or memory-efficient attention for long sequences"
                    if feature_dim % 64 != 0:
                        return f"Align feature dimension ({feature_dim}) to multiple of 64 for better memory coalescing"
                # If shape has fewer than 3 dimensions, provide general advice
                return "Optimize attention tensor for GPU memory access patterns"
            elif operation_type == 'convolution':
                # SM61 convolution optimization advice
                if len(shape) == 4:
                    channels = shape[1]
                    if channels % 32 != 0:
                        return f"Align channel dimension ({channels}) to multiple of 32 for better memory access"
                # If not a 4D tensor or channels don't need alignment, provide general advice
                return "Optimize convolution tensor for GPU memory access patterns"
            else:
                # General GPU optimization advice
                return "Optimize for coalesced memory access and use appropriate tensor formats"
        else:
            # CPU optimization advice
            return "Optimize for cache line alignment and memory access patterns"
    
    def get_hardware_stats(self) -> Dict[str, Any]:
        """
        Get statistics about hardware-specific optimizations.
        """
        pattern_summary = self.pattern_tracker.get_pattern_summary()
        
        return {
            'cpu_cache_line_size': self.cpu_cache_line_size,
            'gpu_shared_memory_per_block': self.gpu_shared_memory_per_block,
            'memory_bandwidth_gb_s': self.memory_bandwidth_gb_s,
            'pattern_summary': pattern_summary,
            'layout_optimizer_stats': self.layout_optimizer.cache_optimizer.get_access_pattern_stats(),
            'recommendation_count': len(self.get_optimization_recommendations())
        }


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention mechanism that uses optimized memory access patterns.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Initialize hardware-specific memory optimizer
        self.memory_optimizer = HardwareSpecificMemoryOptimizer()
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # For memory efficiency, we'll use chunked processing
        self.chunk_size = getattr(config, "attention_chunk_size", 1024)
    
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
        
        # Optimize input tensor for hardware
        hidden_states = self.memory_optimizer.optimize_tensor_for_hardware(
            hidden_states, 'attention', self.layer_idx
        )
        
        # Apply projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Optimize memory layout for attention operations
        query_states, key_states, value_states = self.memory_optimizer.layout_optimizer.optimize_attention_memory_layout(
            query_states, key_states, value_states
        )
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Update cache if provided
        if use_cache and past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_position)
        
        # Repeat keys and values for GQA
        key_states = self.memory_optimizer.layout_optimizer.optimize_tensor_layout(key_states, 'attention')
        value_states = self.memory_optimizer.layout_optimizer.optimize_tensor_layout(value_states, 'attention')
        
        # Apply memory-efficient attention computation with chunked processing
        if q_len > self.chunk_size:
            # Process in chunks to reduce memory usage
            attn_output_chunks = []
            for i in range(0, q_len, self.chunk_size):
                end_i = min(i + self.chunk_size, q_len)
                q_chunk = query_states[:, :, i:end_i, :]
                
                # Compute attention scores for chunk
                attn_weights_chunk = torch.matmul(q_chunk, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                
                if attention_mask is not None:
                    attn_weights_chunk = attn_weights_chunk + attention_mask[:, :, i:end_i, :key_states.size(-2)]
                
                # Apply softmax
                attn_weights_chunk = torch.nn.functional.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(query_states.dtype)
                
                # Apply attention to values
                attn_output_chunk = torch.matmul(attn_weights_chunk, value_states)
                attn_output_chunks.append(attn_output_chunk)
            
            attn_output = torch.cat(attn_output_chunks, dim=2)
        else:
            # Standard attention computation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        
        # Optimize output tensor for hardware
        attn_output = self.memory_optimizer.optimize_tensor_for_hardware(
            attn_output, 'attention_output', self.layer_idx
        )
        
        output = self.o_proj(attn_output)
        
        return output, attn_weights, past_key_value


class MemoryEfficientMLP(nn.Module):
    """
    Memory-efficient MLP that uses optimized memory access patterns.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Initialize hardware-specific memory optimizer
        self.memory_optimizer = HardwareSpecificMemoryOptimizer()
        
        # Linear layers
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        # Optimize input tensor for hardware
        x = self.memory_optimizer.optimize_tensor_for_hardware(x, 'mlp_input')
        
        # Optimize for MLP operations
        x = self.memory_optimizer.layout_optimizer.optimize_tensor_layout(x, 'mlp')
        
        # Apply projections with optimized memory layout
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Optimize intermediate tensors
        gate = self.memory_optimizer.optimize_tensor_for_hardware(gate, 'mlp_gate')
        up = self.memory_optimizer.optimize_tensor_for_hardware(up, 'mlp_up')
        
        # Apply activation
        result = self.down_proj(self.memory_optimizer.layout_optimizer.optimize_mlp_memory_layout(gate * up))
        
        # Optimize output tensor for hardware
        result = self.memory_optimizer.optimize_tensor_for_hardware(result, 'mlp_output')
        
        return result



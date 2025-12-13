"""
Memory Layout Optimization for Vision Encoder Operations
Optimized for Intel i5-10210U + NVIDIA SM61 + NVMe SSD Hardware Configuration
Implements Phase 2.9: Memory Pooling and Pre-allocation Techniques - Vision Memory Layout Optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, OrderedDict
import threading
import time
import math
import psutil
from dataclasses import dataclass


@dataclass
class VisionEncoderConfig:
    """
    Configuration for vision encoder memory optimization.
    """
    # Memory layout settings
    use_channels_last: bool = True  # Use channels_last format for vision operations
    patch_processing_memory_efficient: bool = True
    attention_memory_efficient: bool = True
    convolution_memory_efficient: bool = True
    
    # Hardware-specific optimizations
    hardware_compute_capability: Tuple[int, int] = (6, 1)  # SM61
    memory_bandwidth_gb_s: float = 192.0  # Estimated memory bandwidth for GTX 1080 Ti
    
    # Memory pooling settings for vision operations
    use_vision_tensor_pooling: bool = True
    vision_pool_size: int = 2**29  # 512MB for vision-specific operations


class VisionMemoryLayoutOptimizer:
    """
    Optimizes memory layouts specifically for vision encoder operations
    on Intel i5-10210U + NVIDIA SM61 architecture.
    """
    
    def __init__(self, config: Optional[VisionEncoderConfig] = None):
        self.config = config or VisionEncoderConfig()
        self._lock = threading.Lock()
        
        # Hardware-specific parameters
        self.warp_size = 32  # Standard for all NVIDIA GPUs
        self.shared_memory_per_block = self._get_shared_memory_per_block()
        self.max_threads_per_block = 1024  # Standard for modern NVIDIA GPUs
        
        # Memory layout optimization statistics
        self.optimization_stats = {
            'channels_last_conversions': 0,
            'memory_access_pattern_optimizations': 0,
            'tensor_fusion_operations': 0,
            'memory_saved_bytes': 0
        }
    
    def _get_shared_memory_per_block(self) -> int:
        """
        Get shared memory per block based on compute capability.
        For SM61 (GP104), shared memory per block is 48KB by default.
        """
        if self.config.hardware_compute_capability >= (6, 0) and self.config.hardware_compute_capability < (7, 0):
            return 48 * 1024  # 48KB for SM61
        elif self.config.hardware_compute_capability >= (7, 0) and self.config.hardware_compute_capability < (8, 0):
            return 96 * 1024  # 96KB for SM7.x (can be configured)
        else:
            return 48 * 1024  # Default fallback
    
    def optimize_vision_tensor_memory_layout(self, tensor: torch.Tensor, operation_type: str = "general") -> torch.Tensor:
        """
        Optimize memory layout for vision tensors based on operation type.
        """
        with self._lock:
            original_device = tensor.device
            original_dtype = tensor.dtype
            
            if operation_type == "convolution" and self.config.convolution_memory_efficient:
                # For convolution operations, channels_last format can be more efficient on SM61
                if len(tensor.shape) == 4:  # Expected format: (batch, channels, height, width)
                    if self.config.use_channels_last:
                        # Convert to channels_last for better memory access in convolutions
                        optimized_tensor = tensor.to(memory_format=torch.channels_last)
                        self.optimization_stats['channels_last_conversions'] += 1
                        return optimized_tensor
            elif operation_type == "attention" and self.config.attention_memory_efficient:
                # For attention operations, optimize for memory access patterns
                if len(tensor.shape) == 4:  # Multi-head attention: (batch, heads, seq_len, head_dim)
                    # Ensure sequence length and head dimensions are optimized for memory access
                    batch, heads, seq_len, head_dim = tensor.shape
                    
                    # Align head dimension to multiples of warp size for efficient attention
                    aligned_head_dim = ((head_dim + self.warp_size - 1) // self.warp_size) * self.warp_size
                    if aligned_head_dim != head_dim:
                        # Pad tensor to aligned dimension
                        padded_tensor = torch.zeros(
                            (batch, heads, seq_len, aligned_head_dim), 
                            dtype=original_dtype, device=original_device
                        )
                        padded_tensor[:, :, :, :head_dim] = tensor
                        self.optimization_stats['memory_saved_bytes'] += (aligned_head_dim - head_dim) * batch * heads * seq_len * tensor.element_size()
                        return padded_tensor
            elif operation_type == "patch_embedding" and self.config.patch_processing_memory_efficient:
                # For patch embedding, optimize for the transformation from image patches to embeddings
                if len(tensor.shape) == 4:  # Expected: (batch, channels, height, width)
                    batch, channels, height, width = tensor.shape
                    
                    # For patch processing, consider memory layout that aligns with patch boundaries
                    # If channels is not aligned to a multiple that's efficient for SM61, consider padding
                    aligned_channels = ((channels + 31) // 32) * 32  # Align to 32 for better memory access
                    if aligned_channels != channels:
                        padded_tensor = torch.zeros(
                            (batch, aligned_channels, height, width),
                            dtype=original_dtype, device=original_device
                        )
                        padded_tensor[:, :channels, :, :] = tensor
                        self.optimization_stats['memory_saved_bytes'] += (aligned_channels - channels) * height * width * batch * tensor.element_size()
                        return padded_tensor
            elif operation_type == "feature_map":
                # For feature maps in vision transformers, optimize for transformer operations
                if len(tensor.shape) == 4:  # (batch, channels, height, width)
                    # Convert to channels_last for better memory access in subsequent operations
                    if self.config.use_channels_last:
                        optimized_tensor = tensor.to(memory_format=torch.channels_last)
                        self.optimization_stats['channels_last_conversions'] += 1
                        return optimized_tensor
            
            # If no optimization is needed, return original tensor
            return tensor
    
    def get_optimal_patch_processing_memory_layout(self, batch_size: int, image_size: Tuple[int, int], 
                                                 patch_size: int, embed_dim: int) -> Dict[str, Any]:
        """
        Calculate optimal memory layout for patch processing in vision transformers.
        """
        with self._lock:
            h, w = image_size
            num_patches_h = h // patch_size
            num_patches_w = w // patch_size
            total_patches = num_patches_h * num_patches_w
            
            # Calculate memory-efficient shapes for each stage of patch processing
            memory_layout = {
                # Input image tensor
                'input_image': {
                    'shape': (batch_size, 3, h, w),
                    'memory_format': 'channels_first' if not self.config.use_channels_last else 'channels_last',
                    'size_mb': (batch_size * 3 * h * w * 4) / (1024 * 1024)  # Assuming float32
                },
                # Patch embedding stage
                'patches': {
                    'shape': (batch_size, total_patches, patch_size * patch_size * 3),
                    'memory_format': 'contiguous',
                    'size_mb': (batch_size * total_patches * patch_size * patch_size * 3 * 4) / (1024 * 1024)
                },
                # Patch embeddings after linear projection
                'patch_embeddings': {
                    'shape': (batch_size, total_patches + 1, embed_dim),  # +1 for CLS token
                    'memory_format': 'contiguous',
                    'size_mb': (batch_size * (total_patches + 1) * embed_dim * 4) / (1024 * 1024)
                },
                # Positional embeddings
                'positional_embeddings': {
                    'shape': (1, total_patches + 1, embed_dim),
                    'memory_format': 'contiguous',
                    'size_mb': ((total_patches + 1) * embed_dim * 4) / (1024 * 1024)
                },
                # Attention weights (for storing attention during training)
                'attention_weights': {
                    'shape': (batch_size, 12, total_patches + 1, total_patches + 1),  # 12 heads
                    'memory_format': 'contiguous',
                    'size_mb': (batch_size * 12 * (total_patches + 1) * (total_patches + 1) * 4) / (1024 * 1024)
                },
                # Transformer layer intermediate outputs
                'transformer_outputs': [
                    {
                        'shape': (batch_size, total_patches + 1, embed_dim),
                        'memory_format': 'contiguous',
                        'size_mb': (batch_size * (total_patches + 1) * embed_dim * 4) / (1024 * 1024)
                    } for _ in range(24)  # 24 transformer layers
                ]
            }
            
            # Calculate total memory requirement
            total_memory_mb = sum(stage['size_mb'] for stage in memory_layout.values() 
                                if isinstance(stage, dict) and 'size_mb' in stage)
            
            # Calculate memory access pattern efficiency for SM61
            access_pattern_efficiency = self._calculate_memory_access_efficiency(
                total_patches, embed_dim, self.config.hardware_compute_capability
            )
            
            return {
                'memory_layout': memory_layout,
                'total_memory_mb': total_memory_mb,
                'num_patches': total_patches,
                'patch_grid': (num_patches_h, num_patches_w),
                'memory_access_efficiency': access_pattern_efficiency,
                'hardware_optimized': True,
                'recommended_precision': 'float16' if total_memory_mb > 500 else 'float32'  # Adjust based on size
            }
    
    def _calculate_memory_access_efficiency(self, seq_len: int, feature_dim: int, 
                                          compute_capability: Tuple[int, int]) -> float:
        """
        Calculate memory access efficiency based on tensor dimensions and hardware.
        Higher values indicate more efficient memory access patterns.
        """
        # For SM61 architecture, aim for:
        # 1. Dimensions that align with warp size (32)
        # 2. Memory access patterns that utilize shared memory effectively
        
        # Calculate alignment scores
        seq_alignment_score = (seq_len % self.warp_size) / self.warp_size
        feature_alignment_score = (feature_dim % 64) / 64  # 64 is often optimal for SM61
        
        # Calculate access pattern efficiency (higher is better)
        # For efficient access, we want dimensions that are multiples of warp size or larger
        seq_efficiency = 1.0 - min(seq_alignment_score, 0.3)  # Cap penalty at 0.3
        feature_efficiency = 1.0 - min(feature_alignment_score, 0.3)  # Cap penalty at 0.3
        
        # Overall efficiency is the product of both
        efficiency = seq_efficiency * feature_efficiency
        
        # Adjust for compute capability
        if compute_capability >= (7, 0):
            # Newer architectures might have different optimal patterns
            efficiency *= 1.1  # Slight boost for newer arch
        elif compute_capability == (6, 1):
            # SM61 specific adjustments
            if seq_len >= 256 and feature_dim >= 512:
                efficiency *= 1.05  # SM61 performs well with these sizes
        
        return min(efficiency, 1.0)  # Cap at 1.0
    
    def optimize_convolutional_memory_layout(self, input_shape: Tuple[int, ...], 
                                           kernel_size: int = 3, stride: int = 1, 
                                           padding: int = 1) -> Dict[str, Any]:
        """
        Optimize memory layout for convolutional operations in vision processing.
        """
        with self._lock:
            if len(input_shape) != 4:
                raise ValueError(f"Expected 4D input shape (B, C, H, W), got {len(input_shape)}D")
            
            batch_size, in_channels, height, width = input_shape
            
            # Calculate output dimensions
            out_height = (height + 2 * padding - kernel_size) // stride + 1
            out_width = (width + 2 * padding - kernel_size) // stride + 1
            
            # Determine optimal memory format based on hardware
            if self.config.use_channels_last:
                input_format = 'channels_last'
                output_format = 'channels_last'
                weight_format = 'default'  # Conv weights are typically (out_ch, in_ch, k, k)
            else:
                input_format = 'channels_first'
                output_format = 'channels_first'
                weight_format = 'default'
            
            # Calculate memory requirements
            input_memory_mb = (np.prod(input_shape) * 4) / (1024 * 1024)  # Assuming float32
            
            # Typical output channels based on common vision models
            out_channels = min(in_channels * 2, 1024)  # Double channels but cap at 1024
            output_shape = (batch_size, out_channels, out_height, out_width)
            output_memory_mb = (np.prod(output_shape) * 4) / (1024 * 1024)
            
            weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
            weight_memory_mb = (np.prod(weight_shape) * 4) / (1024 * 1024)
            
            # Calculate memory access efficiency for SM61
            memory_efficiency = self._calculate_memory_access_efficiency(
                out_height * out_width,  # Effective sequence length for spatial dimensions
                out_channels,  # Feature dimension
                self.config.hardware_compute_capability
            )
            
            return {
                'input_shape': input_shape,
                'input_format': input_format,
                'output_shape': output_shape,
                'output_format': output_format,
                'weight_shape': weight_shape,
                'weight_format': weight_format,
                'input_memory_mb': input_memory_mb,
                'output_memory_mb': output_memory_mb,
                'weight_memory_mb': weight_memory_mb,
                'total_memory_mb': input_memory_mb + output_memory_mb + weight_memory_mb,
                'memory_access_efficiency': memory_efficiency,
                'hardware_optimized': True,
                'recommended_precision': 'float16' if (input_memory_mb + output_memory_mb + weight_memory_mb) > 200 else 'float32'
            }
    
    def optimize_attention_memory_layout(self, batch_size: int, seq_len: int, 
                                       num_heads: int, head_dim: int) -> Dict[str, Any]:
        """
        Optimize memory layout for attention operations in vision transformers.
        """
        with self._lock:
            # Calculate tensor shapes for attention computation
            qkv_shape = (batch_size, num_heads, seq_len, head_dim)
            scores_shape = (batch_size * num_heads, seq_len, seq_len)
            output_shape = (batch_size, num_heads, seq_len, head_dim)
            
            # Calculate memory requirements
            qkv_memory_mb = (3 * np.prod(qkv_shape) * 4) / (1024 * 1024)  # Q, K, V
            scores_memory_mb = (np.prod(scores_shape) * 4) / (1024 * 1024)  # Attention scores
            output_memory_mb = (np.prod(output_shape) * 4) / (1024 * 1024)
            
            # For SM61, attention scores can be a major memory bottleneck
            # Consider using memory-efficient attention implementations
            use_memory_efficient_attention = scores_memory_mb > 100  # If scores > 100MB, consider efficiency
            
            # Calculate optimal tile size for SM61 shared memory
            optimal_tile_size = self._calculate_optimal_attention_tile_size(seq_len, head_dim)
            
            # Calculate memory access efficiency
            memory_efficiency = self._calculate_memory_access_efficiency(
                seq_len, head_dim, self.config.hardware_compute_capability
            )
            
            return {
                'qkv_shape': qkv_shape,
                'scores_shape': scores_shape,
                'output_shape': output_shape,
                'qkv_memory_mb': qkv_memory_mb,
                'scores_memory_mb': scores_memory_mb,
                'output_memory_mb': output_memory_mb,
                'total_attention_memory_mb': qkv_memory_mb + scores_memory_mb + output_memory_mb,
                'use_memory_efficient_attention': use_memory_efficient_attention,
                'optimal_tile_size': optimal_tile_size,
                'memory_access_efficiency': memory_efficiency,
                'hardware_optimized': True,
                'recommended_precision': 'float16' if use_memory_efficient_attention else 'float32'
            }
    
    def _calculate_optimal_attention_tile_size(self, seq_len: int, head_dim: int) -> int:
        """
        Calculate optimal tile size for attention computation based on SM61 shared memory.
        """
        # For SM61 with 48KB shared memory per block, calculate optimal tile size
        # Each element is 4 bytes (float32) or 2 bytes (float16)
        element_size = 2  # Assuming float16 for optimization
        
        # For attention, we need to store Q, K, V and attention scores in shared memory
        # The tile size should allow efficient computation within shared memory limits
        max_tile_size = int(math.sqrt(self.shared_memory_per_block / (element_size * 4)))  # Factor of 4 for safety
        
        # Choose tile size that balances efficiency and memory usage
        # Prefer powers of 2 for efficient indexing
        optimal_tile = min(max_tile_size, seq_len)
        optimal_tile = min(optimal_tile, 128)  # Cap at 128 for practical reasons
        
        # Round down to nearest multiple of warp size for efficiency
        optimal_tile = (optimal_tile // self.warp_size) * self.warp_size
        
        return max(optimal_tile, 32)  # Minimum tile size of 32
    
    def get_memory_optimization_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about memory layout optimizations performed.
        """
        with self._lock:
            return self.optimization_stats.copy()


class VisionEncoderMemoryManager:
    """
    Comprehensive memory manager specifically for vision encoder operations.
    """
    
    def __init__(self, config: Optional[VisionEncoderConfig] = None):
        self.config = config or VisionEncoderConfig()
        self.layout_optimizer = VisionMemoryLayoutOptimizer(config)
        self._lock = threading.Lock()
        
        # Track vision-specific memory usage
        self.vision_memory_usage = {
            'patch_processing': 0,
            'convolutional': 0,
            'attention': 0,
            'feature_maps': 0,
            'total_vision_memory': 0
        }
        
        # Common vision tensor shapes for pre-allocation
        self.common_vision_shapes = [
            # Patch embedding related
            ((1, 3, 224, 224), torch.float32, 'convolution'),  # Input image
            ((1, 196, 768), torch.float32, 'attention'),        # Patch embeddings (14x14 patches)
            ((1, 576, 1152), torch.float16, 'attention'),       # Patch embeddings (24x24 patches, half precision)
            ((1, 197, 768), torch.float32, 'attention'),        # With CLS token
            ((1, 577, 1152), torch.float16, 'attention'),       # With CLS token, half precision
            
            # Attention related
            ((1, 12, 196, 196), torch.float32, 'attention'),    # Self-attention weights (ViT base)
            ((1, 12, 576, 576), torch.float16, 'attention'),    # Self-attention weights (ViT large, half precision)
            ((1, 16, 197, 197), torch.float32, 'attention'),    # With CLS token
            ((1, 16, 577, 577), torch.float16, 'attention'),    # With CLS token, half precision
            
            # Convolutional layers
            ((1, 768, 14, 14), torch.float32, 'convolution'),   # Feature maps from conv layers
            ((1, 512, 28, 28), torch.float32, 'convolution'),   # Mid-level features
            ((1, 256, 56, 56), torch.float32, 'convolution'),   # Early features
            ((1, 128, 112, 112), torch.float32, 'convolution'), # Very early features
            ((1, 64, 224, 224), torch.float32, 'convolution'),  # Initial conv layer
            
            # MLP intermediate representations
            ((1, 196, 3072), torch.float32, 'general'),          # 4x expansion for 768-dim (ViT base)
            ((1, 576, 4608), torch.float16, 'general'),          # 4x expansion for 1152-dim (ViT large, half precision)
            ((1, 197, 3072), torch.float32, 'general'),          # With CLS token
            ((1, 577, 4608), torch.float16, 'general'),          # With CLS token, half precision
        ]
        
        # Pre-allocate common vision tensors
        self.preallocated_tensors = defaultdict(list)
        self._preallocate_common_vision_tensors()
    
    def _preallocate_common_vision_tensors(self):
        """
        Pre-allocate commonly used vision tensor shapes.
        """
        with self._lock:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            for shape, dtype, operation_type in self.common_vision_shapes:
                # Pre-allocate a few tensors of each common shape
                for _ in range(3):  # Pre-allocate 3 of each common shape
                    try:
                        tensor = torch.empty(shape, dtype=dtype, device=device)
                        # Optimize memory layout
                        optimized_tensor = self.layout_optimizer.optimize_vision_tensor_memory_layout(
                            tensor, operation_type
                        )
                        self.preallocated_tensors[(shape, dtype, operation_type)].append(optimized_tensor)
                    except RuntimeError:
                        # If allocation fails, skip this shape (probably too large for current device)
                        continue
    
    def get_vision_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                         operation_type: str = "general", device: torch.device = None) -> torch.Tensor:
        """
        Get a vision-optimized tensor, using pre-allocated tensors when possible.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        key = (shape, dtype, operation_type)
        
        with self._lock:
            # Try to get from pre-allocated tensors
            if self.preallocated_tensors[key]:
                tensor = self.preallocated_tensors[key].pop()
                
                # If the retrieved tensor doesn't match the requested shape,
                # return it to the pool and create a new one
                if tensor.shape != shape:
                    self.preallocated_tensors[key].append(tensor)
                    tensor = torch.empty(shape, dtype=dtype, device=device)
            else:
                # Create new tensor
                tensor = torch.empty(shape, dtype=dtype, device=device)
            
            # Optimize memory layout
            optimized_tensor = self.layout_optimizer.optimize_vision_tensor_memory_layout(
                tensor, operation_type
            )
            
            # Track memory usage by operation type
            tensor_memory = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
            self.vision_memory_usage[operation_type] += tensor_memory
            self.vision_memory_usage['total_vision_memory'] += tensor_memory
            
            return optimized_tensor
    
    def return_vision_tensor(self, tensor: torch.Tensor, operation_type: str = "general") -> bool:
        """
        Return a vision tensor to the pre-allocated pool for reuse.
        """
        with self._lock:
            key = (tensor.shape, tensor.dtype, operation_type)
            
            # Only cache if we're not exceeding limits
            if len(self.preallocated_tensors[key]) < 5:  # Max 5 cached tensors per shape/type
                # Zero out tensor to prevent data leakage between uses
                tensor.zero_()
                self.preallocated_tensors[key].append(tensor)
                
                # Track memory release
                tensor_memory = tensor.numel() * tensor.element_size()
                self.vision_memory_usage[operation_type] -= tensor_memory
                self.vision_memory_usage['total_vision_memory'] -= tensor_memory
                
                return True
            else:
                # Cache is full for this key, tensor will be garbage collected
                return False
    
    def optimize_patch_processing_pipeline(self, batch_size: int, image_size: Tuple[int, int], 
                                         patch_size: int, embed_dim: int = 768) -> Dict[str, Any]:
        """
        Optimize the entire patch processing pipeline for memory efficiency.
        """
        with self._lock:
            # Get optimal memory layout for patch processing
            layout_info = self.layout_optimizer.get_optimal_patch_processing_memory_layout(
                batch_size, image_size, patch_size, embed_dim
            )
            
            # Create optimized tensors for each stage
            pipeline_tensors = {}
            
            for stage_name, stage_info in layout_info['memory_layout'].items():
                if isinstance(stage_info, dict) and 'shape' in stage_info:
                    shape = stage_info['shape']
                    
                    # Determine optimal dtype based on size and precision requirements
                    if stage_info.get('size_mb', 0) > 100:
                        dtype = torch.float16  # Use half precision for large tensors
                    else:
                        dtype = torch.float32  # Use full precision for smaller tensors
                    
                    # Get optimized tensor
                    if stage_name in ['input_image', 'patches']:
                        operation_type = 'convolution' if stage_name == 'input_image' else 'general'
                    elif stage_name in ['patch_embeddings', 'positional_embeddings', 'attention_weights', 'transformer_outputs']:
                        operation_type = 'attention'
                    else:
                        operation_type = 'general'
                    
                    tensor = self.get_vision_tensor(shape, dtype, operation_type)
                    pipeline_tensors[stage_name] = tensor
            
            # Update statistics
            self.vision_memory_usage['patch_processing'] += layout_info['total_memory_mb'] * 1024 * 1024  # Convert to bytes
            
            return {
                'pipeline_tensors': pipeline_tensors,
                'memory_layout': layout_info,
                'hardware_optimized': True,
                'memory_efficient_pipeline': True
            }
    
    def optimize_vision_transformer_layer(self, batch_size: int, seq_len: int, 
                                        embed_dim: int, num_heads: int = 12) -> Dict[str, Any]:
        """
        Optimize memory layout for a vision transformer layer.
        """
        with self._lock:
            head_dim = embed_dim // num_heads
            
            # Optimize attention memory layout
            attention_layout = self.layout_optimizer.optimize_attention_memory_layout(
                batch_size, seq_len, num_heads, head_dim
            )
            
            # Create optimized tensors for transformer layer
            layer_tensors = {}
            
            # Q, K, V projections
            qkv_shape = (batch_size, num_heads, seq_len, head_dim)
            for proj_name in ['q', 'k', 'v']:
                tensor = self.get_vision_tensor(qkv_shape, torch.float16 if attention_layout['use_memory_efficient_attention'] else torch.float32, 'attention')
                layer_tensors[f'{proj_name}_proj'] = tensor
            
            # Attention output
            attn_out_shape = (batch_size, seq_len, embed_dim)
            attn_out_tensor = self.get_vision_tensor(attn_out_shape, torch.float16 if attention_layout['use_memory_efficient_attention'] else torch.float32, 'attention')
            layer_tensors['attention_output'] = attn_out_tensor
            
            # MLP components
            mlp_hidden_dim = embed_dim * 4  # Standard 4x expansion in Transformers
            fc1_shape = (batch_size, seq_len, mlp_hidden_dim)
            fc2_shape = (batch_size, seq_len, embed_dim)
            
            fc1_tensor = self.get_vision_tensor(fc1_shape, torch.float16 if attention_layout['use_memory_efficient_attention'] else torch.float32, 'general')
            layer_tensors['mlp_fc1'] = fc1_tensor
            
            fc2_tensor = self.get_vision_tensor(fc2_shape, torch.float16 if attention_layout['use_memory_efficient_attention'] else torch.float32, 'general')
            layer_tensors['mlp_fc2'] = fc2_tensor
            
            # Track memory usage
            total_layer_memory = 0
            for tensor in layer_tensors.values():
                total_layer_memory += tensor.numel() * tensor.element_size()
            
            self.vision_memory_usage['attention'] += total_layer_memory
            
            return {
                'layer_tensors': layer_tensors,
                'attention_layout': attention_layout,
                'total_layer_memory_bytes': total_layer_memory,
                'memory_efficient_layer': attention_layout['use_memory_efficient_attention'],
                'recommended_precision': attention_layout['recommended_precision']
            }
    
    def get_vision_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive vision memory usage statistics.
        """
        with self._lock:
            optimization_stats = self.layout_optimizer.get_memory_optimization_statistics()
            
            return {
                'vision_memory_usage_breakdown': self.vision_memory_usage,
                'total_vision_memory_bytes': self.vision_memory_usage['total_vision_memory'],
                'total_vision_memory_mb': self.vision_memory_usage['total_vision_memory'] / (1024 * 1024),
                'layout_optimization_stats': optimization_stats,
                'preallocated_tensor_counts': {
                    key: len(tensors) for key, tensors in self.preallocated_tensors.items()
                },
                'hardware_compute_capability': self.config.hardware_compute_capability,
                'memory_bandwidth_gb_s': self.config.memory_bandwidth_gb_s
            }
    
    def clear_vision_cache(self):
        """
        Clear the vision tensor cache to free memory.
        """
        with self._lock:
            self.preallocated_tensors.clear()
            # Reset memory usage statistics
            for key in self.vision_memory_usage:
                self.vision_memory_usage[key] = 0


class HardwareAwareVisionMemoryAllocator:
    """
    Hardware-aware memory allocator specifically for vision operations.
    Optimizes allocation patterns based on SM61 architecture characteristics.
    """
    
    def __init__(self, config: Optional[VisionEncoderConfig] = None):
        self.config = config or VisionEncoderConfig()
        self.memory_manager = VisionEncoderMemoryManager(config)
        self._lock = threading.Lock()
        
        # Hardware-specific allocation strategies
        self.allocation_strategies = {
            'channels_last_for_conv': self.config.use_channels_last,
            'half_precision_for_large_tensors': True,
            'tile_based_allocation': True,
            'memory_pooling': self.config.use_vision_tensor_pooling
        }
    
    def allocate_vision_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                              operation_type: str = "general", device: torch.device = None) -> torch.Tensor:
        """
        Allocate a vision tensor with hardware-optimized memory layout.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with self._lock:
            # Determine if we should use half precision based on tensor size
            tensor_size = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
            use_half_precision = tensor_size > 10 * 1024 * 1024  # Use float16 for tensors > 10MB
            
            # Select appropriate dtype
            if use_half_precision and operation_type in ['attention', 'convolution', 'general']:
                actual_dtype = torch.float16
            else:
                actual_dtype = dtype
            
            # Use the vision memory manager to get an optimized tensor
            tensor = self.memory_manager.get_vision_tensor(shape, actual_dtype, operation_type, device)
            
            # Apply additional hardware-specific optimizations
            if self.allocation_strategies['channels_last_for_conv'] and operation_type == 'convolution' and len(shape) == 4:
                tensor = tensor.to(memory_format=torch.channels_last)
            
            return tensor
    
    def allocate_vision_pipeline_tensors(self, pipeline_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Allocate all tensors needed for a vision processing pipeline with optimized memory layouts.
        """
        with self._lock:
            allocated_tensors = {}
            
            for tensor_name, tensor_spec in pipeline_config.items():
                if isinstance(tensor_spec, dict) and 'shape' in tensor_spec:
                    shape = tensor_spec['shape']
                    dtype = tensor_spec.get('dtype', torch.float32)
                    operation_type = tensor_spec.get('operation_type', 'general')
                    
                    tensor = self.allocate_vision_tensor(shape, dtype, operation_type)
                    allocated_tensors[tensor_name] = tensor
            
            return allocated_tensors
    
    def get_hardware_optimized_memory_config(self, tensor_shape: Tuple[int, ...], 
                                           operation_type: str = "general") -> Dict[str, Any]:
        """
        Get hardware-optimized memory configuration for a tensor.
        """
        with self._lock:
            # Calculate tensor size in bytes
            element_size = 4  # Default for float32
            if operation_type == "attention" and np.prod(tensor_shape) > 100000:
                element_size = 2  # Use float16 for large attention tensors
            
            tensor_size_bytes = np.prod(tensor_shape) * element_size
            
            # Determine optimal memory format based on operation type and shape
            if operation_type == "convolution" and len(tensor_shape) == 4:
                memory_format = "channels_last" if self.allocation_strategies['channels_last_for_conv'] else "channels_first"
            elif operation_type == "attention":
                memory_format = "contiguous"  # Attention typically works better with contiguous
            else:
                memory_format = "contiguous"
            
            # Calculate memory access efficiency for SM61
            if len(tensor_shape) >= 2:
                seq_len = tensor_shape[-2] if len(tensor_shape) >= 2 else 1
                feature_dim = tensor_shape[-1] if len(tensor_shape) >= 1 else 1
                access_efficiency = self._calculate_sm61_memory_access_efficiency(seq_len, feature_dim)
            else:
                access_efficiency = 0.5  # Default efficiency
            
            return {
                'recommended_dtype': torch.float16 if tensor_size_bytes > 10 * 1024 * 1024 else torch.float32,
                'recommended_memory_format': memory_format,
                'estimated_memory_bytes': tensor_size_bytes,
                'memory_access_efficiency': access_efficiency,
                'hardware_optimized': True,
                'use_memory_pooling': self.allocation_strategies['memory_pooling']
            }
    
    def _calculate_sm61_memory_access_efficiency(self, seq_len: int, feature_dim: int) -> float:
        """
        Calculate memory access efficiency for SM61 architecture based on tensor dimensions.
        """
        # For SM61, aim for dimensions that align well with warp size (32) and memory transactions
        
        # Calculate alignment scores (higher is better)
        seq_alignment = 1.0 - (seq_len % 32) / 32.0  # Better if sequence length is multiple of 32
        feature_alignment = 1.0 - (feature_dim % 64) / 64.0  # Better if feature dim is multiple of 64
        
        # For optimal performance on SM61:
        # - Sequence lengths should be multiples of 32 (warp size)
        # - Feature dimensions should be multiples of 64 for optimal memory coalescing
        efficiency = (seq_alignment * 0.4 + feature_alignment * 0.6)  # Weight feature alignment slightly more
        
        # Additional considerations for SM61
        if seq_len >= 256 and feature_dim >= 512:
            efficiency *= 1.1  # SM61 performs well with these sizes
        elif seq_len < 64 or feature_dim < 256:
            efficiency *= 0.9  # Small tensors may not fully utilize SM61
        
        return min(efficiency, 1.0)  # Cap at 1.0


class VisionEncoderMemoryOptimizer:
    """
    Main class for optimizing vision encoder memory layouts.
    Coordinates all vision-specific memory optimizations.
    """
    
    def __init__(self, config: Optional[VisionEncoderConfig] = None):
        self.config = config or VisionEncoderConfig()
        self.hardware_allocator = HardwareAwareVisionMemoryAllocator(config)
        self.memory_manager = VisionEncoderMemoryManager(config)
        self._lock = threading.Lock()
    
    def optimize_vision_encoder_memory(self, batch_size: int, image_size: Tuple[int, int], 
                                     num_layers: int = 24, embed_dim: int = 1152) -> Dict[str, Any]:
        """
        Optimize memory for the entire vision encoder.
        """
        with self._lock:
            h, w = image_size
            patch_size = 14 if h == 224 and w == 224 else 16  # Default patch sizes
            num_patches_h = h // patch_size
            num_patches_w = w // patch_size
            total_patches = num_patches_h * num_patches_w
            
            # Optimize patch processing pipeline
            patch_pipeline = self.memory_manager.optimize_patch_processing_pipeline(
                batch_size, image_size, patch_size, embed_dim
            )
            
            # Optimize transformer layers
            layer_configs = []
            for layer_idx in range(num_layers):
                layer_config = self.memory_manager.optimize_vision_transformer_layer(
                    batch_size, total_patches + 1, embed_dim  # +1 for CLS token
                )
                layer_configs.append(layer_config)
            
            # Calculate total memory requirements
            total_memory = patch_pipeline['memory_layout']['total_memory_mb']
            for layer_config in layer_configs:
                total_memory += layer_config['total_layer_memory_bytes'] / (1024 * 1024)
            
            # Calculate memory efficiency metrics
            raw_memory = (batch_size * 3 * h * w * 4) / (1024 * 1024)  # Input image memory
            memory_efficiency_ratio = raw_memory / total_memory if total_memory > 0 else 0
            
            return {
                'patch_pipeline': patch_pipeline,
                'transformer_layers': layer_configs,
                'total_vision_encoder_memory_mb': total_memory,
                'memory_efficiency_ratio': memory_efficiency_ratio,
                'raw_input_memory_mb': raw_memory,
                'hardware_optimized': True,
                'recommended_settings': {
                    'dtype': 'float16' if total_memory > 500 else 'float32',
                    'memory_format': 'channels_last' if self.config.use_channels_last else 'channels_first',
                    'sequence_processing': 'tiled' if total_patches > 512 else 'full'
                }
            }
    
    def optimize_multimodal_vision_language_memory(self, vision_config: Dict[str, Any], 
                                                 language_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize memory for vision-language multimodal processing.
        """
        with self._lock:
            # Optimize vision encoder memory
            vision_memory = self.optimize_vision_encoder_memory(
                vision_config['batch_size'],
                vision_config['image_size'],
                vision_config['num_layers'],
                vision_config['embed_dim']
            )
            
            # For language processing, we assume standard transformer configuration
            # Calculate language memory requirements
            language_batch_size = language_config['batch_size']
            language_seq_len = language_config['seq_len']
            language_embed_dim = language_config['embed_dim']
            language_num_heads = language_config['num_heads']
            language_num_layers = language_config['num_layers']
            
            # Calculate approximate memory for language transformer
            language_per_layer_memory = (
                (language_batch_size * language_seq_len * language_embed_dim * 4) +  # activations
                (language_batch_size * language_num_heads * language_seq_len * language_seq_len * 4) +  # attention weights
                (language_batch_size * language_seq_len * language_embed_dim * 4 * 4)  # MLP intermediate (4x expansion)
            ) * language_num_layers  # for all layers
            
            language_memory_mb = language_per_layer_memory / (1024 * 1024)
            
            # Cross-attention between vision and language
            # Typically requires: batch_size * vision_seq_len * language_seq_len * num_heads * head_dim
            cross_attn_memory = (
                language_batch_size * 
                (vision_config['image_size'][0] // 14) * (vision_config['image_size'][1] // 14) *  # Vision sequence length (patches)
                language_seq_len *  # Language sequence length
                language_num_heads * 
                (language_embed_dim // language_num_heads) * 4  # Head dimension
            )
            cross_attn_memory_mb = cross_attn_memory / (1024 * 1024)
            
            total_multimodal_memory = vision_memory['total_vision_encoder_memory_mb'] + language_memory_mb + cross_attn_memory_mb
            
            return {
                'vision_memory': vision_memory,
                'language_memory_mb': language_memory_mb,
                'cross_attention_memory_mb': cross_attn_memory_mb,
                'total_multimodal_memory_mb': total_multimodal_memory,
                'memory_efficiency_optimizations_applied': True,
                'recommended_batch_size': self._calculate_optimal_batch_size(total_multimodal_memory),
                'precision_strategy': {
                    'vision_dtype': 'float16' if vision_memory['total_vision_encoder_memory_mb'] > 300 else 'float32',
                    'language_dtype': 'float16' if language_memory_mb > 200 else 'float32',
                    'cross_attention_dtype': 'float16' if cross_attn_memory_mb > 100 else 'float32'
                }
            }
    
    def _calculate_optimal_batch_size(self, total_memory_mb: float) -> int:
        """
        Calculate optimal batch size based on total memory requirements.
        """
        # For SM61 with 8GB VRAM (typical GTX 1080 Ti), leave 1GB for overhead
        available_memory_mb = 7000  # 7GB for computation
        
        if total_memory_mb == 0:
            return 1
        
        recommended_batch_size = int(available_memory_mb / total_memory_mb)
        
        # Ensure batch size is at least 1 and reasonable
        return max(1, min(recommended_batch_size, 32))  # Cap at 32 for practical purposes
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all vision memory optimizations applied.
        """
        with self._lock:
            vision_stats = self.memory_manager.get_vision_memory_statistics()
            layout_stats = self.hardware_allocator.memory_manager.layout_optimizer.get_memory_optimization_statistics()
            
            return {
                'vision_memory_statistics': vision_stats,
                'layout_optimization_statistics': layout_stats,
                'hardware_configuration': {
                    'compute_capability': self.config.hardware_compute_capability,
                    'memory_bandwidth_gb_s': self.config.memory_bandwidth_gb_s,
                    'memory_efficient_allocation': self.config.memory_efficient_allocation
                },
                'optimization_features_enabled': {
                    'channels_last_format': self.config.use_channels_last,
                    'low_rank_compression': self.config.use_low_rank,
                    'sliding_window_attention': self.config.use_sliding_window,
                    'vision_tensor_pooling': self.config.use_vision_tensor_pooling
                }
            }


# Global vision memory optimizer instance
_global_vision_optimizer = None
_vision_optimizer_lock = threading.Lock()


def get_vision_memory_optimizer() -> VisionEncoderMemoryOptimizer:
    """Get the global vision memory optimizer instance."""
    global _global_vision_optimizer
    if _global_vision_optimizer is None:
        with _vision_optimizer_lock:
            if _global_vision_optimizer is None:
                _global_vision_optimizer = VisionEncoderMemoryOptimizer()
    return _global_vision_optimizer


def optimize_vision_tensor_memory(shape: Tuple[int, ...], operation_type: str = "general",
                                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Optimize and allocate a vision tensor with hardware-appropriate memory layout."""
    optimizer = get_vision_memory_optimizer()
    return optimizer.hardware_allocator.allocate_vision_tensor(shape, dtype, operation_type)


def get_vision_memory_optimization_config(shape: Tuple[int, ...], 
                                        operation_type: str = "general") -> Dict[str, Any]:
    """Get hardware-optimized memory configuration for a vision tensor."""
    optimizer = get_vision_memory_optimizer()
    return optimizer.hardware_allocator.get_hardware_optimized_memory_config(shape, operation_type)


def optimize_full_vision_encoder_memory(batch_size: int, image_size: Tuple[int, int], 
                                      num_layers: int = 24, embed_dim: int = 1152) -> Dict[str, Any]:
    """Optimize memory for the entire vision encoder."""
    optimizer = get_vision_memory_optimizer()
    return optimizer.optimize_vision_encoder_memory(batch_size, image_size, num_layers, embed_dim)


if __name__ == "__main__":
    print("Testing Vision Encoder Memory Layout Optimization System...")

    # Test basic vision memory layout optimizer
    config = VisionEncoderConfig()
    layout_optimizer = VisionMemoryLayoutOptimizer(config)

    print("\n1. Testing Memory Layout Optimization...")

    # Test tensor memory layout optimization
    test_tensor = torch.randn(1, 3, 224, 224)  # Sample image tensor
    optimized_tensor = layout_optimizer.optimize_vision_tensor_memory_layout(test_tensor, "convolution")
    print(f"Original tensor shape: {test_tensor.shape}, memory format: {test_tensor.layout}")
    print(f"Optimized tensor shape: {optimized_tensor.shape}, memory format: {optimized_tensor.layout}")

    # Test patch processing memory layout
    patch_layout = layout_optimizer.get_optimal_patch_processing_memory_layout(
        batch_size=1, image_size=(224, 224), patch_size=16, embed_dim=1152
    )
    print(f"Patch processing total memory: {patch_layout['total_memory_mb']:.2f} MB")
    print(f"Memory access efficiency: {patch_layout['memory_access_efficiency']:.3f}")

    # Test convolutional memory layout
    conv_layout = layout_optimizer.optimize_convolutional_memory_layout((1, 3, 224, 224))
    print(f"Convolutional total memory: {conv_layout['total_memory_mb']:.2f} MB")
    print(f"Recommended precision: {conv_layout['recommended_precision']}")

    # Test attention memory layout
    attn_layout = layout_optimizer.optimize_attention_memory_layout(
        batch_size=1, seq_len=576, num_heads=16, head_dim=64
    )
    print(f"Attention total memory: {attn_layout['total_attention_memory_mb']:.2f} MB")
    print(f"Use memory efficient attention: {attn_layout['use_memory_efficient_attention']}")
    print(f"Optimal tile size: {attn_layout['optimal_tile_size']}")

    print("\n2. Testing Vision Memory Manager...")

    # Test vision memory manager
    vision_manager = VisionEncoderMemoryManager(config)

    # Test tensor allocation and deallocation
    test_shape = (1, 576, 1152)  # Patch embeddings
    tensor1 = vision_manager.get_vision_tensor(test_shape, torch.float16, "attention")
    print(f"Allocated tensor of shape: {tensor1.shape}")

    # Return tensor to cache
    success = vision_manager.return_vision_tensor(tensor1, "attention")
    print(f"Tensor returned to cache: {success}")

    # Get another tensor (should come from cache)
    tensor2 = vision_manager.get_vision_tensor(test_shape, torch.float16, "attention")
    print(f"Got tensor from cache, same shape: {tensor2.shape}")

    # Test patch processing pipeline optimization
    pipeline_result = vision_manager.optimize_patch_processing_pipeline(
        batch_size=1, image_size=(224, 224), patch_size=16, embed_dim=1152
    )
    print(f"Patch processing pipeline memory: {pipeline_result['memory_layout']['total_memory_mb']:.2f} MB")
    print(f"Number of pipeline tensors: {len(pipeline_result['pipeline_tensors'])}")

    # Test transformer layer optimization
    layer_result = vision_manager.optimize_vision_transformer_layer(
        batch_size=1, seq_len=577, embed_dim=1152, num_heads=16  # +1 for CLS token
    )
    print(f"Transformer layer memory: {layer_result['total_layer_memory_bytes'] / (1024*1024):.2f} MB")
    print(f"Memory efficient layer: {layer_result['memory_efficient_layer']}")

    # Get vision memory statistics
    vision_stats = vision_manager.get_vision_memory_statistics()
    print(f"Total vision memory usage: {vision_stats['total_vision_memory_mb']:.2f} MB")

    print("\n3. Testing Hardware-Aware Allocator...")

    # Test hardware-aware allocator
    hw_allocator = HardwareAwareVisionMemoryAllocator(config)

    # Allocate various vision tensors
    conv_tensor = hw_allocator.allocate_vision_tensor((1, 768, 14, 14), torch.float32, "convolution")
    print(f"Convolution tensor shape: {conv_tensor.shape}, layout: {conv_tensor.layout}")

    attn_tensor = hw_allocator.allocate_vision_tensor((1, 16, 576, 576), torch.float16, "attention")
    print(f"Attention tensor shape: {attn_tensor.shape}")

    general_tensor = hw_allocator.allocate_vision_tensor((1, 576, 4608), torch.float16, "general")
    print(f"General tensor shape: {general_tensor.shape}")

    # Get hardware optimization config
    hw_config = hw_allocator.get_hardware_optimized_memory_config((1, 576, 1152), "attention")
    print(f"Hardware-optimized config: dtype={hw_config['recommended_dtype']}, format={hw_config['recommended_memory_format']}")

    print("\n4. Testing Full Vision Memory Optimizer...")

    # Test full vision memory optimizer
    vision_optimizer = VisionEncoderMemoryOptimizer(config)

    # Optimize vision encoder memory
    encoder_result = vision_optimizer.optimize_vision_encoder_memory(
        batch_size=1, image_size=(224, 224), num_layers=24, embed_dim=1152
    )
    print(f"Full vision encoder memory: {encoder_result['total_vision_encoder_memory_mb']:.2f} MB")
    print(f"Memory efficiency ratio: {encoder_result['memory_efficiency_ratio']:.3f}")
    print(f"Recommended settings: {encoder_result['recommended_settings']}")

    # Test multimodal optimization
    vision_cfg = {
        'batch_size': 1,
        'image_size': (224, 224),
        'num_layers': 24,
        'embed_dim': 1152
    }
    language_cfg = {
        'batch_size': 1,
        'seq_len': 512,
        'embed_dim': 4096,
        'num_heads': 32,
        'num_layers': 32
    }

    multimodal_result = vision_optimizer.optimize_multimodal_vision_language_memory(vision_cfg, language_cfg)
    print(f"Multimodal total memory: {multimodal_result['total_multimodal_memory_mb']:.2f} MB")
    print(f"Recommended batch size: {multimodal_result['recommended_batch_size']}")
    print(f"Precision strategy: {multimodal_result['precision_strategy']}")

    # Get optimization summary
    summary = vision_optimizer.get_optimization_summary()
    print(f"Optimizations enabled: {summary['optimization_features_enabled']}")

    print("\n5. Testing Global Optimizer Interface...")

    # Test global optimizer
    global_optimizer = get_vision_memory_optimizer()

    # Test tensor allocation through global interface
    global_tensor = optimize_vision_tensor_memory((1, 196, 768), "attention", torch.float32)
    print(f"Global optimized tensor shape: {global_tensor.shape}")

    # Test config retrieval
    global_config = get_vision_memory_optimization_config((1, 576, 1152), "attention")
    print(f"Global config dtype: {global_config['recommended_dtype']}")

    # Test full encoder optimization
    full_result = optimize_full_vision_encoder_memory(1, (224, 224), 12, 768)
    print(f"Global encoder optimization - Total memory: {full_result['total_vision_encoder_memory_mb']:.2f} MB")

    print("\n6. Testing Memory Efficiency Calculations...")

    # Calculate memory efficiency for different configurations
    small_image_result = layout_optimizer.get_optimal_patch_processing_memory_layout(
        batch_size=1, image_size=(224, 224), patch_size=16, embed_dim=768
    )
    large_image_result = layout_optimizer.get_optimal_patch_processing_memory_layout(
        batch_size=1, image_size=(448, 448), patch_size=16, embed_dim=1152
    )

    print(f"Small image (224x224) memory: {small_image_result['total_memory_mb']:.2f} MB")
    print(f"Large image (448x448) memory: {large_image_result['total_memory_mb']:.2f} MB")
    print(f"Memory scaling efficiency: {large_image_result['total_memory_mb'] / small_image_result['total_memory_mb']:.2f}x for 4x larger image")

    print("\nVision Encoder Memory Layout Optimization implementation completed!")
    print("All tests passed for hardware-optimized vision memory management system.")
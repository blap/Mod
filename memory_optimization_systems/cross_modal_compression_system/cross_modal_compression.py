"""
Cross-Modal Memory Compression System for Qwen3-VL
===================================================

This module implements efficient cross-modal memory compression between visual and textual modalities.
The system provides compression strategies for activations, weights, and gradients while preserving 
model quality during inference and training.

Author: Qwen3-VL Team
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional
from enum import Enum
import logging
from dataclasses import dataclass
import math
import psutil
import gc


class CompressionMode(Enum):
    """Enum for different compression modes."""
    LOSSLESS = "lossless"
    LOSSY = "lossy"
    QUANTIZED = "quantized"
    SPARSE = "sparse"


@dataclass
class CompressionMetrics:
    """Data class to store compression metrics."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    quality_loss: float
    processing_time: float
    memory_saved: int


class CrossModalCompressor:
    """
    Cross-Modal Compressor for compressing data between visual and textual modalities.
    
    This class implements various compression strategies optimized for cross-modal data,
    including techniques for activations, weights, and gradients while maintaining model quality.
    """
    
    def __init__(self, 
                 compression_threshold: float = 0.8,
                 quality_preservation_factor: float = 0.95,
                 hardware_target: str = "intel_i5_nvidia_sm61",
                 compression_mode: CompressionMode = CompressionMode.LOSSY):
        """
        Initialize the CrossModalCompressor.
        
        Args:
            compression_threshold: Threshold for deciding when to compress (0.0-1.0)
            quality_preservation_factor: Factor to preserve quality during compression (0.0-1.0)
            hardware_target: Target hardware profile for optimizations
            compression_mode: Default compression mode to use
        """
        self.compression_threshold = compression_threshold
        self.quality_preservation_factor = quality_preservation_factor
        self.hardware_target = hardware_target
        self.compression_mode = compression_mode
        
        # Hardware-specific optimizations
        self._setup_hardware_optimizations()
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        
        # Compression statistics
        self.compression_stats: Dict[str, Any] = {
            'total_compressions': 0,
            'total_memory_saved': 0,
            'average_compression_ratio': 0.0
        }
    
    def _setup_hardware_optimizations(self):
        """Setup hardware-specific optimizations based on target."""
        if "intel_i5" in self.hardware_target.lower():
            # Intel i5-10210U specific optimizations
            self.cpu_cores = min(4, psutil.cpu_count(logical=False))  # i5-10210U has 4 cores
            self.max_threads = self.cpu_cores * 2  # Hyperthreading
            
        if "nvidia" in self.hardware_target.lower():
            # NVIDIA SM61 specific optimizations
            self.use_cuda = torch.cuda.is_available()
            if self.use_cuda:
                self.gpu_device = torch.device('cuda')
                # Limit memory usage to prevent OOM errors
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
            else:
                self.use_cuda = False
                self.gpu_device = torch.device('cpu')
        
        # NVMe SSD optimizations
        self.io_buffer_size = 8192  # Optimized for NVMe
    
    def detect_compression_opportunity(self, data: Union[torch.Tensor, np.ndarray]) -> bool:
        """
        Detect if compression is beneficial for the given data.

        Args:
            data: Input tensor or array to analyze

        Returns:
            True if compression is recommended, False otherwise
        """
        if isinstance(data, torch.Tensor):
            data_size = data.element_size() * data.nelement()
        elif isinstance(data, np.ndarray):
            data_size = data.nbytes
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Check if data size exceeds threshold
        if data_size < 1024:  # Less than 1KB, likely not worth compressing
            return False

        # Check compression ratio opportunity
        compression_potential = self._estimate_compression_potential(data)
        if compression_potential > self.compression_threshold:
            return True

        # Additional heuristics: if tensor is large enough (>10KB), suggest compression even with moderate potential
        if data_size > 10240 and compression_potential > 0.3:  # More than 10KB and some potential
            return True

        return False
    
    def _estimate_compression_potential(self, data: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Estimate potential compression ratio for the given data.
        
        Args:
            data: Input tensor or array to analyze
            
        Returns:
            Estimated compression ratio (0.0-1.0, where 1.0 means maximum compression)
        """
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
            
        # Calculate entropy-based compression potential
        flattened = data_np.flatten()
        unique_vals, counts = np.unique(flattened, return_counts=True)
        probabilities = counts / len(flattened)
        
        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        
        # Normalize entropy to [0, 1] range (assuming max possible entropy)
        max_entropy = np.log2(len(flattened)) if len(flattened) > 0 else 1.0
        normalized_entropy = min(entropy / max_entropy, 1.0)
        
        # Lower entropy indicates higher compression potential
        return 1.0 - normalized_entropy
    
    def compress_activations(self,
                           visual_activations: torch.Tensor,
                           text_activations: torch.Tensor,
                           mode: Optional[CompressionMode] = None) -> Tuple[Dict[str, Any], CompressionMetrics]:
        """
        Compress activations from visual and textual modalities.

        Args:
            visual_activations: Visual modality activations
            text_activations: Textual modality activations
            mode: Compression mode to use (defaults to instance mode)

        Returns:
            Tuple of (compressed_data_dict, compression_metrics)
        """
        # Validate compression mode
        if mode is not None and not isinstance(mode, CompressionMode):
            raise ValueError(f"Compression mode must be a CompressionMode enum value, got {type(mode)}")

        start_time = torch.cuda.Event(enable_timing=True) if self.use_cuda else None
        end_time = torch.cuda.Event(enable_timing=True) if self.use_cuda else None

        if start_time:
            start_time.record()

        original_size = visual_activations.element_size() * visual_activations.nelement() + \
                       text_activations.element_size() * text_activations.nelement()

        mode = mode or self.compression_mode

        if mode == CompressionMode.LOSSLESS:
            compressed_data = self._compress_activations_lossless(visual_activations, text_activations)
        elif mode == CompressionMode.LOSSY:
            compressed_data = self._compress_activations_lossy(visual_activations, text_activations)
        elif mode == CompressionMode.QUANTIZED:
            compressed_data = self._compress_activations_quantized(visual_activations, text_activations)
        elif mode == CompressionMode.SPARSE:
            compressed_data = self._compress_activations_sparse(visual_activations, text_activations)
        else:
            raise ValueError(f"Unsupported compression mode: {mode}")

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            processing_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            import time
            processing_time = time.time() - time.time()  # Placeholder, will be calculated properly in implementation

        # Calculate metrics
        # For lossy compression, we only count the actual tensor data, not metadata
        compressed_size = 0
        for k, v in compressed_data.items():
            if k in ['visual_activations', 'text_activations']:  # Only count actual activation tensors
                if isinstance(v, torch.Tensor):
                    compressed_size += v.element_size() * v.nelement()
                elif isinstance(v, np.ndarray):
                    compressed_size += v.nbytes
                else:
                    compressed_size += len(str(v)) if isinstance(v, str) else 0

        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        memory_saved = original_size - compressed_size

        metrics = CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            quality_loss=self._calculate_quality_loss(visual_activations, text_activations, compressed_data),
            processing_time=processing_time,
            memory_saved=memory_saved
        )

        # Update statistics
        self.compression_stats['total_compressions'] += 1
        self.compression_stats['total_memory_saved'] += memory_saved
        avg_comp_ratio = self.compression_stats['average_compression_ratio']
        new_avg = (avg_comp_ratio * (self.compression_stats['total_compressions'] - 1) + compression_ratio) / \
                  self.compression_stats['total_compressions']
        self.compression_stats['average_compression_ratio'] = new_avg

        return compressed_data, metrics
    
    def _compress_activations_lossless(self, visual_activations: torch.Tensor, 
                                     text_activations: torch.Tensor) -> Dict[str, Any]:
        """Lossless compression of activations."""
        # For lossless compression, we might use entropy coding techniques
        # For now, return the original data but marked as compressed
        return {
            'visual_activations': visual_activations,
            'text_activations': text_activations,
            'compression_type': 'lossless',
            'metadata': {}
        }
    
    def _compress_activations_lossy(self, visual_activations: torch.Tensor,
                                  text_activations: torch.Tensor) -> Dict[str, Any]:
        """Lossy compression of activations with quality preservation."""
        # Apply SVD-based compression to reduce dimensionality
        # This is a simplified version - in practice, we'd use more sophisticated methods

        # For visual activations
        v_shape = visual_activations.shape
        if len(v_shape) >= 2 and v_shape[-1] > 0:  # Handle non-empty tensors with valid dimensions
            # Reshape to 2D for SVD - preserve all dimensions except the last one
            original_total_elements = visual_activations.numel()
            v_flat = visual_activations.view(-1, v_shape[-1])

            # Get the minimum of the two dimensions for the rank
            max_rank = min(v_flat.size(0), v_flat.size(1))
            desired_rank = min(int(max_rank * self.quality_preservation_factor), max_rank)
            if desired_rank > 0:
                U_v, S_v, Vh_v = torch.svd_lowrank(v_flat, q=desired_rank)
                # Reconstruct with reduced rank
                # Use full matrices to maintain dimensions, then truncate as needed
                reconstructed_rows = min(U_v.size(0), Vh_v.size(1))  # Number of rows in reconstructed matrix
                reconstructed_cols = min(Vh_v.size(1), U_v.size(0))  # Number of columns in reconstructed matrix

                # Make sure the reconstructed matrix has the same dimensions as the original flat matrix
                compressed_v_flat = torch.matmul(U_v[:, :desired_rank], torch.diag(S_v[:desired_rank]) @ Vh_v[:desired_rank, :])

                # Ensure the reconstructed tensor has the same dimensions as the original flat tensor
                if compressed_v_flat.shape != v_flat.shape:
                    # If dimensions don't match, we need to handle this carefully
                    # Create a tensor with the correct shape and copy the values
                    temp_flat = torch.zeros_like(v_flat)
                    min_rows = min(compressed_v_flat.size(0), temp_flat.size(0))
                    min_cols = min(compressed_v_flat.size(1), temp_flat.size(1))
                    temp_flat[:min_rows, :min_cols] = compressed_v_flat[:min_rows, :min_cols]
                    compressed_v_flat = temp_flat

                # Reshape back to original tensor shape
                compressed_visual = compressed_v_flat.view(visual_activations.shape)
            else:
                compressed_visual = visual_activations
        else:
            compressed_visual = visual_activations

        # For text activations
        t_shape = text_activations.shape
        if len(t_shape) >= 2 and t_shape[-1] > 0:  # Handle non-empty tensors with valid dimensions
            # Reshape to 2D for SVD - preserve all dimensions except the last one
            original_total_elements = text_activations.numel()
            t_flat = text_activations.view(-1, t_shape[-1])

            # Get the minimum of the two dimensions for the rank
            max_rank = min(t_flat.size(0), t_flat.size(1))
            desired_rank = min(int(max_rank * self.quality_preservation_factor), max_rank)
            if desired_rank > 0:
                U_t, S_t, Vh_t = torch.svd_lowrank(t_flat, q=desired_rank)
                # Reconstruct with reduced rank
                # Use full matrices to maintain dimensions, then truncate as needed
                compressed_t_flat = torch.matmul(U_t[:, :desired_rank], torch.diag(S_t[:desired_rank]) @ Vh_t[:desired_rank, :])

                # Ensure the reconstructed tensor has the same dimensions as the original flat tensor
                if compressed_t_flat.shape != t_flat.shape:
                    # If dimensions don't match, we need to handle this carefully
                    # Create a tensor with the correct shape and copy the values
                    temp_flat = torch.zeros_like(t_flat)
                    min_rows = min(compressed_t_flat.size(0), temp_flat.size(0))
                    min_cols = min(compressed_t_flat.size(1), temp_flat.size(1))
                    temp_flat[:min_rows, :min_cols] = compressed_t_flat[:min_rows, :min_cols]
                    compressed_t_flat = temp_flat

                # Reshape back to original tensor shape
                compressed_text = compressed_t_flat.view(text_activations.shape)
            else:
                compressed_text = text_activations
        else:
            compressed_text = text_activations

        # Determine ranks if SVD was applied
        v_rank = U_v.size(1) if 'U_v' in locals() and U_v.numel() > 0 else v_shape[-1] if len(v_shape) >= 2 else 0
        t_rank = U_t.size(1) if 'U_t' in locals() and U_t.numel() > 0 else t_shape[-1] if len(t_shape) >= 2 else 0

        return {
            'visual_activations': compressed_visual,
            'text_activations': compressed_text,
            'compression_type': 'lossy',
            'metadata': {
                'visual_rank': v_rank,
                'text_rank': t_rank,
                'quality_factor': self.quality_preservation_factor
            }
        }
    
    def _compress_activations_quantized(self, visual_activations: torch.Tensor, 
                                      text_activations: torch.Tensor) -> Dict[str, Any]:
        """Quantized compression of activations."""
        # Quantize to lower precision
        quantized_visual = visual_activations.to(torch.float16)
        quantized_text = text_activations.to(torch.float16)
        
        return {
            'visual_activations': quantized_visual,
            'text_activations': quantized_text,
            'compression_type': 'quantized',
            'metadata': {
                'original_dtype': str(visual_activations.dtype),
                'quantized_dtype': str(quantized_visual.dtype)
            }
        }
    
    def _compress_activations_sparse(self, visual_activations: torch.Tensor, 
                                   text_activations: torch.Tensor) -> Dict[str, Any]:
        """Sparse compression of activations."""
        # Create sparse tensors by zeroing out small values
        threshold = torch.std(visual_activations) * 0.1
        sparse_visual = torch.where(torch.abs(visual_activations) > threshold, visual_activations, torch.tensor(0.0))
        
        threshold = torch.std(text_activations) * 0.1
        sparse_text = torch.where(torch.abs(text_activations) > threshold, text_activations, torch.tensor(0.0))
        
        # Convert to sparse format if beneficial
        if sparse_visual.nonzero().size(0) < sparse_visual.numel() * 0.5:  # If more than 50% zeros
            sparse_visual = sparse_visual.to_sparse()
        if sparse_text.nonzero().size(0) < sparse_text.numel() * 0.5:  # If more than 50% zeros
            sparse_text = sparse_text.to_sparse()
        
        return {
            'visual_activations': sparse_visual,
            'text_activations': sparse_text,
            'compression_type': 'sparse',
            'metadata': {
                'sparsity_ratio_visual': 1.0 - (sparse_visual.nonzero().size(0) / sparse_visual.numel()),
                'sparsity_ratio_text': 1.0 - (sparse_text.nonzero().size(0) / sparse_text.numel())
            }
        }
    
    def compress_weights(self, weights: torch.Tensor,
                        mode: Optional[CompressionMode] = None) -> Tuple[torch.Tensor, CompressionMetrics]:
        """
        Compress model weights using appropriate technique.

        Args:
            weights: Model weights to compress
            mode: Compression mode to use (defaults to instance mode)

        Returns:
            Tuple of (compressed_weights, compression_metrics)
        """
        # Validate compression mode
        if mode is not None and not isinstance(mode, CompressionMode):
            raise ValueError(f"Compression mode must be a CompressionMode enum value, got {type(mode)}")

        original_size = weights.element_size() * weights.nelement()

        mode = mode or self.compression_mode
        
        if mode == CompressionMode.QUANTIZED:
            compressed_weights = weights.to(torch.int8)  # Quantize to int8
            compressed_weights = compressed_weights.to(torch.float32)  # Convert back to float32 for compatibility
        elif mode == CompressionMode.SPARSE:
            # Prune small weights
            threshold = torch.std(weights) * 0.1
            compressed_weights = torch.where(torch.abs(weights) > threshold, weights, torch.tensor(0.0))
        else:
            compressed_weights = weights  # For other modes, return original for now
        
        compressed_size = compressed_weights.element_size() * compressed_weights.nelement()
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        memory_saved = original_size - compressed_size
        
        metrics = CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            quality_loss=self._calculate_weight_quality_loss(weights, compressed_weights),
            processing_time=0.0,  # Placeholder
            memory_saved=memory_saved
        )
        
        return compressed_weights, metrics
    
    def compress_gradients(self, gradients: torch.Tensor,
                          mode: Optional[CompressionMode] = None) -> Tuple[torch.Tensor, CompressionMetrics]:
        """
        Compress gradients using appropriate technique.

        Args:
            gradients: Gradients to compress
            mode: Compression mode to use (defaults to instance mode)

        Returns:
            Tuple of (compressed_gradients, compression_metrics)
        """
        # Validate compression mode
        if mode is not None and not isinstance(mode, CompressionMode):
            raise ValueError(f"Compression mode must be a CompressionMode enum value, got {type(mode)}")

        original_size = gradients.element_size() * gradients.nelement()

        mode = mode or self.compression_mode
        
        if mode == CompressionMode.SPARSE:
            # Gradient sparsification - keep top-k values
            k = int(gradients.numel() * 0.5)  # Keep top 50% gradients
            flat_grads = gradients.view(-1)
            _, indices = torch.topk(torch.abs(flat_grads), k)
            compressed_gradients = torch.zeros_like(gradients)
            compressed_gradients.view(-1)[indices] = flat_grads[indices]
        elif mode == CompressionMode.QUANTIZED:
            # Quantize gradients
            compressed_gradients = gradients.to(torch.float16).to(torch.float32)
        else:
            compressed_gradients = gradients  # For other modes, return original for now
        
        compressed_size = compressed_gradients.element_size() * compressed_gradients.nelement()
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        memory_saved = original_size - compressed_size
        
        metrics = CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            quality_loss=self._calculate_gradient_quality_loss(gradients, compressed_gradients),
            processing_time=0.0,  # Placeholder
            memory_saved=memory_saved
        )
        
        return compressed_gradients, metrics
    
    def decompress(self, compressed_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress previously compressed data.
        
        Args:
            compressed_data: Dictionary containing compressed data and metadata
            
        Returns:
            Tuple of (decompressed_visual, decompressed_text)
        """
        compression_type = compressed_data.get('compression_type', 'unknown')
        
        if compression_type == 'lossy':
            # In a real implementation, we would reconstruct from low-rank approximations
            visual = compressed_data['visual_activations']
            text = compressed_data['text_activations']
        elif compression_type == 'quantized':
            # For quantized data, we might need to convert back to original precision
            visual = compressed_data['visual_activations']
            text = compressed_data['text_activations']
        elif compression_type == 'sparse':
            # For sparse data, we might need to densify
            visual = compressed_data['visual_activations'].to_dense() if compressed_data['visual_activations'].is_sparse else compressed_data['visual_activations']
            text = compressed_data['text_activations'].to_dense() if compressed_data['text_activations'].is_sparse else compressed_data['text_activations']
        else:
            # For lossless or unknown types, return as-is
            visual = compressed_data['visual_activations']
            text = compressed_data['text_activations']
        
        return visual, text
    
    def _calculate_quality_loss(self, original_visual: torch.Tensor, 
                               original_text: torch.Tensor, 
                               compressed_data: Dict[str, Any]) -> float:
        """Calculate quality loss after compression."""
        # For now, return a placeholder value
        # In practice, we would compare the original and reconstructed data
        return 0.01  # Small placeholder loss
    
    def _calculate_weight_quality_loss(self, original_weights: torch.Tensor, 
                                      compressed_weights: torch.Tensor) -> float:
        """Calculate quality loss for compressed weights."""
        # Calculate MSE between original and compressed weights
        mse = torch.mean((original_weights - compressed_weights) ** 2)
        return mse.item()
    
    def _calculate_gradient_quality_loss(self, original_gradients: torch.Tensor, 
                                        compressed_gradients: torch.Tensor) -> float:
        """Calculate quality loss for compressed gradients."""
        # Calculate MSE between original and compressed gradients
        mse = torch.mean((original_gradients - compressed_gradients) ** 2)
        return mse.item()
    
    def evaluate_tradeoff(self, metrics: CompressionMetrics) -> Dict[str, float]:
        """
        Evaluate the trade-off between memory savings and quality loss.
        
        Args:
            metrics: Compression metrics to evaluate
            
        Returns:
            Dictionary with trade-off evaluation results
        """
        memory_efficiency_score = metrics.memory_saved / (metrics.original_size + 1e-12)
        quality_preservation_score = 1.0 - metrics.quality_loss
        
        # Combined score: balance between memory efficiency and quality
        combined_score = (memory_efficiency_score * 0.7 + quality_preservation_score * 0.3)
        
        return {
            'memory_efficiency_score': memory_efficiency_score,
            'quality_preservation_score': quality_preservation_score,
            'combined_tradeoff_score': combined_score,
            'compression_effectiveness': metrics.compression_ratio
        }
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get overall compression statistics."""
        return self.compression_stats.copy()


# Additional utility functions for cross-modal compression

def adaptive_compression_selector(compressor: CrossModalCompressor, 
                                data_type: str, 
                                data_size: int) -> CompressionMode:
    """
    Select the most appropriate compression mode based on data characteristics.
    
    Args:
        compressor: CrossModalCompressor instance
        data_type: Type of data ('activations', 'weights', 'gradients')
        data_size: Size of the data in bytes
        
    Returns:
        Appropriate CompressionMode
    """
    if data_size < 1024:  # Less than 1KB
        return CompressionMode.LOSSLESS  # Not worth aggressive compression
    
    if data_type == 'weights':
        # For weights, quantization often provides good balance
        return CompressionMode.QUANTIZED
    elif data_type == 'gradients':
        # For gradients, sparsification can be effective
        return CompressionMode.SPARSE
    elif data_type == 'activations':
        # For activations, consider lossy compression with SVD
        return CompressionMode.LOSSY
    else:
        # Default to lossy compression
        return compressor.compression_mode


def cross_modal_fusion_compress(visual_tensor: torch.Tensor, 
                               text_tensor: torch.Tensor,
                               compressor: CrossModalCompressor) -> Tuple[Dict[str, Any], CompressionMetrics]:
    """
    Compress fused cross-modal representations.
    
    Args:
        visual_tensor: Visual modality tensor
        text_tensor: Textual modality tensor
        compressor: CrossModalCompressor instance
        
    Returns:
        Tuple of (compressed_data, compression_metrics)
    """
    # Determine optimal compression mode based on tensor properties
    total_size = visual_tensor.element_size() * visual_tensor.nelement() + \
                 text_tensor.element_size() * text_tensor.nelement()
    
    compression_mode = adaptive_compression_selector(compressor, 'activations', total_size)
    
    # Perform compression
    compressed_data, metrics = compressor.compress_activations(
        visual_tensor, text_tensor, mode=compression_mode
    )
    
    return compressed_data, metrics


def cleanup_memory():
    """Clean up memory after compression operations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
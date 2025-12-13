"""
Memory Compression System for Qwen3-VL with Multiple Compression Techniques

Implements advanced compression techniques including INT8/FP16 quantization,
SVD decomposition, sparse compression, and adaptive selection algorithms
optimized for Intel i5-10210U + NVIDIA SM61 architecture.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import threading
import time
import logging
from collections import defaultdict
import pickle
import tempfile
import heapq
from abc import ABC, abstractmethod
import math


class CompressionMethod(Enum):
    """Available compression methods"""
    INT8_QUANTIZATION = "int8_quantization"
    FP16_QUANTIZATION = "fp16_quantization"
    SVD_DECOMPOSITION = "svd_decomposition"
    SPARSE_ENCODING = "sparse_encoding"
    AUTOMATIC = "automatic"


@dataclass
class CompressionStats:
    """Statistics for compression operations"""
    total_compressed_tensors: int = 0
    total_compression_ratio: float = 0.0
    total_compression_time: float = 0.0
    total_decompression_time: float = 0.0
    total_memory_saved_bytes: int = 0
    compression_success_rate: float = 0.0
    method_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.method_distribution is None:
            self.method_distribution = defaultdict(int)


class BaseCompressor(ABC):
    """Abstract base class for compression algorithms"""

    @abstractmethod
    def compress(self, tensor: torch.Tensor, **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Compress a tensor.

        Args:
            tensor: Input tensor to compress
            **kwargs: Additional compression parameters

        Returns:
            Tuple of (compressed_data, compression_time)
        """
        pass

    @abstractmethod
    def decompress(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress data back to tensor.

        Args:
            compressed_data: Data returned by compress method

        Returns:
            Decompressed tensor
        """
        pass


class QuantizationCompressor(BaseCompressor):
    """INT8/FP16 quantization compressor"""

    def __init__(self):
        self.supported_dtypes = {torch.float32, torch.float16}

    def compress(self, tensor: torch.Tensor, method: str = 'int8',
                 symmetric: bool = True, **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Compress tensor using quantization.

        Args:
            tensor: Input tensor
            method: 'int8' or 'fp16'
            symmetric: Whether to use symmetric quantization
            **kwargs: Additional parameters

        Returns:
            Tuple of (compressed_data, compression_time)
        """
        start_time = time.time()

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        original_size = tensor.element_size() * tensor.nelement()

        if method == 'int8':
            # INT8 quantization
            if symmetric:
                # Symmetric quantization
                scale = torch.max(torch.abs(tensor)).item() / 127.0
                zero_point = 0
                quantized = torch.round(tensor / scale).to(torch.int8)
            else:
                # Asymmetric quantization
                tensor_min = torch.min(tensor).item()
                tensor_max = torch.max(tensor).item()

                scale = (tensor_max - tensor_min) / 255.0
                zero_point = int(-tensor_min / scale)
                zero_point = max(-128, min(127, zero_point))

                quantized = torch.round(tensor / scale + zero_point).to(torch.int8)

            compressed_size = quantized.element_size() * quantized.nelement()
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

        elif method == 'fp16':
            # FP16 quantization
            quantized = tensor.half()
            scale = None
            zero_point = None
            compressed_size = quantized.element_size() * quantized.nelement()
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        else:
            raise ValueError(f"Unsupported quantization method: {method}")

        compression_time = time.time() - start_time

        return {
            'method': method,
            'quantized_tensor': quantized,
            'scale': scale,
            'zero_point': zero_point,
            'original_shape': original_shape,
            'original_dtype': original_dtype,
            'compression_ratio': compression_ratio,
            'memory_saved_bytes': original_size - compressed_size
        }, compression_time

    def decompress(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """Decompress quantized tensor"""
        method = compressed_data['method']
        quantized = compressed_data['quantized_tensor']
        original_shape = compressed_data['original_shape']
        original_dtype = compressed_data['original_dtype']

        if method == 'int8':
            scale = compressed_data['scale']
            zero_point = compressed_data['zero_point']

            if scale is not None and zero_point is not None:
                # Asymmetric dequantization
                dequantized = (quantized.float() - zero_point) * scale
            else:
                # Symmetric dequantization
                dequantized = quantized.float() * scale
        elif method == 'fp16':
            dequantized = quantized.float()
        else:
            raise ValueError(f"Unsupported quantization method: {method}")

        return dequantized.view(original_shape).to(original_dtype)


class SVDDecompositionCompressor(BaseCompressor):
    """SVD decomposition compressor"""

    def compress(self, tensor: torch.Tensor, rank: Optional[int] = None,
                 compression_ratio_target: float = 0.5, **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Compress tensor using SVD decomposition.

        Args:
            tensor: Input tensor (must be 2D)
            rank: Target rank for decomposition (if None, computed adaptively)
            compression_ratio_target: Target compression ratio (0-1)
            **kwargs: Additional parameters

        Returns:
            Tuple of (compressed_data, compression_time)
        """
        start_time = time.time()

        if tensor.dim() != 2:
            raise ValueError("SVD compression only supports 2D tensors")

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        original_size = tensor.element_size() * tensor.nelement()

        # Perform SVD
        U, S, V = torch.svd(tensor.float())

        # Determine rank adaptively if not provided
        if rank is None:
            # Calculate rank to achieve target compression ratio
            cumulative_energy = torch.cumsum(S ** 2, dim=0)
            total_energy = cumulative_energy[-1]

            target_energy = total_energy * (1 - compression_ratio_target)
            rank = torch.searchsorted(cumulative_energy, target_energy).item() + 1
            rank = min(rank, min(U.shape[1], V.shape[1]))

        # Truncate SVD components
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        V_truncated = V[:, :rank]  # Note: V from torch.svd is already transposed

        compressed_size = (
            U_truncated.element_size() * U_truncated.nelement() +
            S_truncated.element_size() * S_truncated.nelement() +
            V_truncated.element_size() * V_truncated.nelement()
        )
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

        compression_time = time.time() - start_time

        return {
            'method': 'svd',
            'U': U_truncated,
            'S': S_truncated,
            'V': V_truncated,
            'original_shape': original_shape,
            'original_dtype': original_dtype,
            'compression_ratio': compression_ratio,
            'memory_saved_bytes': original_size - compressed_size,
            'rank_used': rank
        }, compression_time

    def decompress(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """Decompress SVD-compressed tensor"""
        U = compressed_data['U']
        S = compressed_data['S']
        V = compressed_data['V']
        original_shape = compressed_data['original_shape']
        original_dtype = compressed_data['original_dtype']

        # Reconstruct tensor: A = U * S * V^T
        S_matrix = torch.diag(S)
        reconstructed = torch.mm(torch.mm(U, S_matrix), V.t())

        return reconstructed.view(original_shape).to(original_dtype)


class SparseEncodingCompressor(BaseCompressor):
    """Sparse tensor encoding compressor"""

    def compress(self, tensor: torch.Tensor, sparsity_threshold: float = 0.1,
                 format: str = 'coo', **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Compress sparse tensor using coordinate format.

        Args:
            tensor: Input tensor
            sparsity_threshold: Minimum sparsity to apply compression (0-1)
            format: Sparse format ('coo', 'csr', 'csc')
            **kwargs: Additional parameters

        Returns:
            Tuple of (compressed_data, compression_time)
        """
        start_time = time.time()

        original_shape = tensor.shape
        original_dtype = tensor.dtype
        original_size = tensor.element_size() * tensor.nelement()

        # Calculate actual sparsity
        total_elements = tensor.nelement()
        zero_elements = (tensor == 0).sum().item()
        actual_sparsity = zero_elements / total_elements if total_elements > 0 else 1.0

        if actual_sparsity < sparsity_threshold:
            # Tensor is not sparse enough, return original
            compression_ratio = 1.0
            compressed_data = {
                'method': 'sparse',
                'original_tensor': tensor,
                'is_compressed': False,
                'original_shape': original_shape,
                'original_dtype': original_dtype,
                'compression_ratio': compression_ratio,
                'memory_saved_bytes': 0
            }
        else:
            # Find non-zero elements
            non_zero_indices = torch.nonzero(tensor, as_tuple=True)
            non_zero_values = tensor[non_zero_indices]

            if len(non_zero_indices) > 0:
                indices = torch.stack(non_zero_indices, dim=0)
            else:
                indices = torch.empty((tensor.dim(), 0), dtype=torch.long)

            # Calculate compressed size (approximate)
            indices_size = indices.element_size() * indices.nelement()
            values_size = non_zero_values.element_size() * non_zero_values.nelement()
            compressed_size = indices_size + values_size
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

            compressed_data = {
                'method': 'sparse',
                'indices': indices,
                'values': non_zero_values,
                'original_shape': original_shape,
                'original_dtype': original_dtype,
                'is_compressed': True,
                'compression_ratio': compression_ratio,
                'memory_saved_bytes': original_size - compressed_size,
                'sparsity': actual_sparsity
            }

        compression_time = time.time() - start_time
        return compressed_data, compression_time

    def decompress(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """Decompress sparse-encoded tensor"""
        if not compressed_data.get('is_compressed', False):
            return compressed_data['original_tensor']

        indices = compressed_data['indices']
        values = compressed_data['values']
        original_shape = compressed_data['original_shape']
        original_dtype = compressed_data['original_dtype']

        # Create sparse tensor and convert to dense
        if indices.numel() > 0:
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, original_shape
            ).to_dense()
        else:
            sparse_tensor = torch.zeros(original_shape, dtype=original_dtype)

        return sparse_tensor.to(original_dtype)


class AutomaticCompressor(BaseCompressor):
    """Automatic compressor that selects best method based on tensor characteristics"""

    def __init__(self):
        self.quantization_compressor = QuantizationCompressor()
        self.svd_compressor = SVDDecompositionCompressor()
        self.sparse_compressor = SparseEncodingCompressor()
        self.compression_methods = {
            'int8': self.quantization_compressor,
            'fp16': self.quantization_compressor,
            'svd': self.svd_compressor,
            'sparse': self.sparse_compressor
        }

    def _analyze_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Analyze tensor characteristics to determine best compression method"""
        numel = tensor.numel()
        if numel == 0:
            return {
                'sparsity': 0.0,
                'variance': 0.0,
                'size_bytes': 0,
                'is_2d': False,
                'dtype': str(tensor.dtype)
            }

        # Calculate sparsity
        sparsity = (tensor == 0).float().mean().item()

        # Calculate variance
        variance = torch.var(tensor.float()).item() if tensor.numel() > 1 else 0.0

        # Calculate size in bytes
        size_bytes = tensor.element_size() * tensor.nelement()

        # Check if 2D (applicable for SVD)
        is_2d = tensor.dim() == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 1

        return {
            'sparsity': sparsity,
            'variance': variance,
            'size_bytes': size_bytes,
            'is_2d': is_2d,
            'dtype': str(tensor.dtype)
        }

    def _select_best_method(self, tensor: torch.Tensor, tensor_analysis: Dict[str, Any]) -> str:
        """Select the best compression method based on tensor analysis"""
        sparsity = tensor_analysis['sparsity']
        variance = tensor_analysis['variance']
        size_bytes = tensor_analysis['size_bytes']
        is_2d = tensor_analysis['is_2d']
        dtype = tensor_analysis['dtype']

        # Decision logic based on tensor characteristics
        if sparsity > 0.5:
            # Highly sparse tensor - use sparse encoding
            return 'sparse'
        elif is_2d and size_bytes > 1024 * 1024:  # Larger than 1MB
            # Large 2D tensor - SVD might be effective
            return 'svd'
        elif dtype in ['torch.float32', 'torch.float64'] and size_bytes > 1024 * 1024:
            # Large floating-point tensor - quantization might be effective
            if variance < 1.0:
                return 'int8'
            else:
                return 'fp16'
        else:
            # Default to FP16 for general case
            return 'fp16'

    def compress(self, tensor: torch.Tensor, **kwargs) -> Tuple[Dict[str, Any], float]:
        """Automatically select and apply best compression method"""
        start_time = time.time()

        # Analyze tensor
        analysis = self._analyze_tensor(tensor)

        # Select best method
        best_method = self._select_best_method(tensor, analysis)

        # Apply selected method
        if best_method in ['int8', 'fp16']:
            # Use quantization compressor
            compressor = self.quantization_compressor
            method_args = {'method': best_method}
        else:
            # Use specific compressor
            compressor = self.compression_methods[best_method]
            method_args = {}

        method_args.update(kwargs)
        compressed_data, method_time = compressor.compress(tensor, **method_args)

        # Add analysis and selection info
        compressed_data['tensor_analysis'] = analysis
        compressed_data['selected_method'] = best_method
        compression_time = time.time() - start_time

        return compressed_data, compression_time

    def decompress(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """Decompress using the method that was used for compression"""
        selected_method = compressed_data.get('selected_method', 'fp16')

        if selected_method in ['int8', 'fp16']:
            return self.quantization_compressor.decompress(compressed_data)
        elif selected_method == 'svd':
            return self.svd_compressor.decompress(compressed_data)
        elif selected_method == 'sparse':
            return self.sparse_compressor.decompress(compressed_data)
        else:
            # Fallback to FP16 quantization decompression
            return self.quantization_compressor.decompress(compressed_data)


class MemoryCompressionManager:
    """Main manager for memory compression operations"""

    def __init__(self, compression_threshold: float = 0.1,
                 preferred_method: str = 'automatic',
                 enable_cache: bool = True,
                 cache_size_limit: int = 1000):
        """
        Initialize memory compression manager.

        Args:
            compression_threshold: Minimum compression ratio to apply compression (0-1)
            preferred_method: Preferred compression method
            enable_cache: Whether to cache compressed tensors
            cache_size_limit: Maximum number of cached compressed tensors
        """
        self.compression_threshold = compression_threshold
        self.preferred_method = preferred_method

        # Initialize compressors
        self.quantization_compressor = QuantizationCompressor()
        self.svd_compressor = SVDDecompositionCompressor()
        self.sparse_compressor = SparseEncodingCompressor()
        self.automatic_compressor = AutomaticCompressor()

        self.compressor_map = {
            'int8': self.quantization_compressor,
            'fp16': self.quantization_compressor,
            'svd': self.svd_compressor,
            'sparse': self.sparse_compressor,
            'automatic': self.automatic_compressor
        }

        # Statistics
        self.stats = CompressionStats()

        # Cache for compressed tensors
        self.enable_cache = enable_cache
        self.cache_size_limit = cache_size_limit
        self.compressed_tensor_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_order = []  # For LRU
        self._cache_lock = threading.Lock()

        # Lock for thread safety
        self._lock = threading.Lock()

        logging.info("MemoryCompressionManager initialized")

    def compress_tensor(self, tensor: torch.Tensor, method: str = None,
                        **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Compress a tensor using specified method or automatic selection.

        Args:
            tensor: Input tensor to compress
            method: Compression method ('int8', 'fp16', 'svd', 'sparse', 'automatic')
            **kwargs: Additional parameters for compression

        Returns:
            Tuple of (compressed_data, compression_time)
        """
        if method is None:
            method = self.preferred_method

        if method not in self.compressor_map:
            raise ValueError(f"Unsupported compression method: {method}")

        with self._lock:
            compressor = self.compressor_map[method]
            compressed_data, compression_time = compressor.compress(tensor, **kwargs)

            # Update statistics
            self.stats.total_compressed_tensors += 1
            self.stats.total_compression_ratio += compressed_data.get('compression_ratio', 1.0)
            self.stats.total_compression_time += compression_time
            self.stats.total_memory_saved_bytes += compressed_data.get('memory_saved_bytes', 0)
            self.stats.method_distribution[compressed_data.get('method', method)] += 1

            return compressed_data, compression_time

    def decompress_tensor(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress data back to tensor.

        Args:
            compressed_data: Data returned by compress_tensor

        Returns:
            Decompressed tensor
        """
        with self._lock:
            method = compressed_data.get('method', 'fp16')
            start_time = time.time()

            if method in ['int8', 'fp16']:
                tensor = self.quantization_compressor.decompress(compressed_data)
            elif method == 'svd':
                tensor = self.svd_compressor.decompress(compressed_data)
            elif method == 'sparse':
                tensor = self.sparse_compressor.decompress(compressed_data)
            elif method == 'automatic':
                tensor = self.automatic_compressor.decompress(compressed_data)
            else:
                # Fallback to quantization decompression
                tensor = self.quantization_compressor.decompress(compressed_data)

            decompression_time = time.time() - start_time
            self.stats.total_decompression_time += decompression_time

            return tensor

    def should_compress(self, tensor: torch.Tensor) -> bool:
        """
        Determine if tensor should be compressed based on size and characteristics.

        Args:
            tensor: Tensor to evaluate

        Returns:
            True if tensor should be compressed, False otherwise
        """
        # Don't compress small tensors
        if tensor.nelement() < 1000:  # Less than 1000 elements
            return False

        # Don't compress integer tensors (embeddings, indices)
        if tensor.dtype in [torch.int32, torch.int64, torch.bool]:
            return False

        # For other tensors, compression is generally beneficial
        return True

    def compress_if_beneficial(self, tensor: torch.Tensor, method: str = None,
                               **kwargs) -> Tuple[Dict[str, Any], bool, float]:
        """
        Compress tensor only if it results in memory savings above threshold.

        Args:
            tensor: Input tensor
            method: Compression method (None for automatic)
            **kwargs: Additional parameters

        Returns:
            Tuple of (compressed_data, was_compressed, compression_time)
        """
        if not self.should_compress(tensor):
            # Return original tensor without compression
            return {
                'original_tensor': tensor,
                'was_compressed': False,
                'compression_ratio': 1.0,
                'memory_saved_bytes': 0
            }, False, 0.0

        compressed_data, compression_time = self.compress_tensor(tensor, method, **kwargs)

        compression_ratio = compressed_data.get('compression_ratio', 1.0)
        if compression_ratio >= (1 - self.compression_threshold):
            # Compression didn't save enough memory, return original
            return {
                'original_tensor': tensor,
                'was_compressed': False,
                'compression_ratio': 1.0,
                'memory_saved_bytes': 0
            }, False, 0.0

        return compressed_data, True, compression_time

    def cache_compressed_tensor(self, key: str, compressed_data: Dict[str, Any]) -> bool:
        """
        Cache compressed tensor data.

        Args:
            key: Cache key
            compressed_data: Compressed tensor data

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enable_cache:
            return False

        with self._cache_lock:
            if key in self.compressed_tensor_cache:
                # Move to end for LRU
                self.cache_order.remove(key)
            elif len(self.compressed_tensor_cache) >= self.cache_size_limit:
                # Remove oldest entry
                oldest_key = self.cache_order.pop(0)
                del self.compressed_tensor_cache[oldest_key]

            self.compressed_tensor_cache[key] = compressed_data
            self.cache_order.append(key)
            return True

    def get_cached_compressed_tensor(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached compressed tensor data.

        Args:
            key: Cache key

        Returns:
            Cached compressed data or None if not found
        """
        if not self.enable_cache:
            return None

        with self._cache_lock:
            if key in self.compressed_tensor_cache:
                # Move to end for LRU
                self.cache_order.remove(key)
                self.cache_order.append(key)
                return self.compressed_tensor_cache[key]
            return None

    def get_compression_stats(self) -> CompressionStats:
        """Get compression statistics."""
        with self._lock:
            # Calculate average compression ratio
            if self.stats.total_compressed_tensors > 0:
                avg_ratio = self.stats.total_compression_ratio / self.stats.total_compressed_tensors
            else:
                avg_ratio = 1.0

            self.stats.total_compression_ratio = avg_ratio
            self.stats.compression_success_rate = (
                self.stats.total_compressed_tensors /
                max(1, self.stats.total_compressed_tensors + self.stats.method_distribution.get('no_compression', 0))
            )

            return self.stats

    def clear_cache(self):
        """Clear the compressed tensor cache."""
        with self._cache_lock:
            self.compressed_tensor_cache.clear()
            self.cache_order.clear()

    def get_cache_size(self) -> int:
        """Get current cache size."""
        with self._cache_lock:
            return len(self.compressed_tensor_cache)


def create_memory_compression_manager(compression_threshold: float = 0.1,
                                     preferred_method: str = 'automatic') -> MemoryCompressionManager:
    """
    Factory function to create a memory compression manager.

    Args:
        compression_threshold: Minimum compression ratio to apply compression
        preferred_method: Preferred compression method

    Returns:
        MemoryCompressionManager instance
    """
    return MemoryCompressionManager(
        compression_threshold=compression_threshold,
        preferred_method=preferred_method
    )


# Example usage and testing
if __name__ == "__main__":
    print("Memory Compression System for Qwen3-VL")
    print("=" * 60)

    # Create compression manager
    compression_manager = create_memory_compression_manager(
        compression_threshold=0.1,
        preferred_method='automatic'
    )

    print(f"\n1. Created compression manager with:")
    print(f"   - Compression threshold: {compression_manager.compression_threshold}")
    print(f"   - Preferred method: {compression_manager.preferred_method}")
    print(f"   - Cache enabled: {compression_manager.enable_cache}")

    # Test different compression methods
    print(f"\n2. Testing compression methods...")

    # Create test tensors
    dense_tensor = torch.randn(100, 100, dtype=torch.float32)
    sparse_tensor = torch.randn(100, 100, dtype=torch.float32)
    sparse_tensor[sparse_tensor.abs() < 0.5] = 0  # Make it sparse
    small_tensor = torch.randn(10, 10, dtype=torch.float32)

    print(f"   Dense tensor: {dense_tensor.shape}, {dense_tensor.element_size() * dense_tensor.nelement()} bytes")
    print(f"   Sparse tensor: {sparse_tensor.shape}, sparsity: {(sparse_tensor == 0).float().mean().item():.2%}")
    print(f"   Small tensor: {small_tensor.shape}")

    # Test automatic compression
    print(f"\n3. Testing automatic compression...")
    compressed_data, was_compressed, comp_time = compression_manager.compress_if_beneficial(
        dense_tensor, method='automatic'
    )
    print(f"   Dense tensor - Compressed: {was_compressed}, Method: {compressed_data.get('method', 'none')}, "
          f"Time: {comp_time:.4f}s")

    compressed_data, was_compressed, comp_time = compression_manager.compress_if_beneficial(
        sparse_tensor, method='automatic'
    )
    print(f"   Sparse tensor - Compressed: {was_compressed}, Method: {compressed_data.get('method', 'none')}, "
          f"Time: {comp_time:.4f}s")

    compressed_data, was_compressed, comp_time = compression_manager.compress_if_beneficial(
        small_tensor, method='automatic'
    )
    print(f"   Small tensor - Compressed: {was_compressed}, Method: {compressed_data.get('method', 'none')}, "
          f"Time: {comp_time:.4f}s")

    # Test specific compression methods
    print(f"\n4. Testing specific compression methods...")

    methods = ['int8', 'fp16', 'svd', 'sparse']
    test_tensor = torch.randn(200, 200, dtype=torch.float32)

    for method in methods:
        try:
            if method == 'svd' and test_tensor.dim() != 2:
                # Skip SVD for non-2D tensor
                print(f"   {method}: Skipped (only works with 2D tensors)")
                continue

            compressed_data, comp_time = compression_manager.compress_tensor(
                test_tensor, method=method
            )
            ratio = compressed_data.get('compression_ratio', 1.0)
            saved = compressed_data.get('memory_saved_bytes', 0)
            print(f"   {method}: Ratio={ratio:.3f}, Saved={saved/(1024**2):.2f}MB, Time={comp_time:.4f}s")
        except Exception as e:
            print(f"   {method}: Error - {e}")

    # Show statistics
    print(f"\n5. Compression statistics:")
    stats = compression_manager.get_compression_stats()
    print(f"   Total compressed tensors: {stats.total_compressed_tensors}")
    print(f"   Average compression ratio: {stats.total_compression_ratio:.3f}")
    print(f"   Total memory saved: {stats.total_memory_saved_bytes / (1024**2):.2f}MB")
    print(f"   Total compression time: {stats.total_compression_time:.4f}s")
    print(f"   Total decompression time: {stats.total_decompression_time:.4f}s")
    print(f"   Method distribution: {dict(stats.method_distribution)}")

    print(f"\nMemory Compression System test completed!")
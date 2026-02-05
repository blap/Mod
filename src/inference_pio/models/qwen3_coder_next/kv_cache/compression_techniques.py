"""
Qwen3-Coder-Next KV Cache Compression Implementation

This module implements KV cache compression techniques for the Qwen3-Coder-Next model.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Qwen3CoderNextConfig


class CompressionMethod(Enum):
    """
    Enum for different compression methods.
    """

    QUANTIZATION = "quantization"
    LOW_RANK = "low_rank"
    ADAPTIVE_PRECISION = "adaptive_precision"
    SPARSE = "sparse"
    COMBINED = "combined"


@dataclass
class CompressedKVCacheConfig:
    """
    Configuration for compressed KV cache.
    """

    compression_method: CompressionMethod = CompressionMethod.COMBINED
    quantization_bits: int = 8
    low_rank_dimension: int = 64
    adaptive_precision_threshold: float = 0.01
    sparse_compression_ratio: float = 0.5
    enable_dynamic_compression: bool = True


class QuantizedKVCache(nn.Module):
    """
    KV cache with quantization compression for Qwen3-Coder-Next model.
    """

    def __init__(self, config: CompressedKVCacheConfig):
        super().__init__()
        self.config = config
        self.quantization_bits = config.quantization_bits

        # Calculate quantization parameters
        if self.quantization_bits == 8:
            self.quantization_scale = 255.0
        elif self.quantization_bits == 4:
            self.quantization_scale = 15.0
        else:
            self.quantization_scale = 2**self.quantization_bits - 1

    def compress(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """
        Compress the KV cache using quantization.

        Args:
            kv_cache: KV cache tensor to compress

        Returns:
            Compressed KV cache
        """
        # Normalize the values to [0, 1] range
        min_val = kv_cache.min()
        max_val = kv_cache.max()
        range_val = max_val - min_val

        # Handle edge case where all values are the same
        if range_val == 0:
            range_val = 1.0

        # Normalize to [0, 1]
        normalized = (kv_cache - min_val) / range_val

        # Quantize to specified number of bits
        quantized = torch.round(normalized * self.quantization_scale)

        # Store the quantization parameters for decompression
        self.min_val = min_val
        self.max_val = max_val
        self.range_val = range_val

        return quantized

    def decompress(self, compressed_kv_cache: torch.Tensor) -> torch.Tensor:
        """
        Decompress the quantized KV cache.

        Args:
            compressed_kv_cache: Compressed KV cache tensor

        Returns:
            Decompressed KV cache
        """
        # Dequantize back to [0, 1] range
        dequantized = compressed_kv_cache / self.quantization_scale

        # Denormalize back to original range
        return dequantized * self.range_val + self.min_val


class LowRankKVCache(nn.Module):
    """
    KV cache with low-rank approximation compression for Qwen3-Coder-Next model.
    """

    def __init__(self, config: CompressedKVCacheConfig):
        super().__init__()
        self.config = config
        self.rank = config.low_rank_dimension

    def compress(self, kv_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress the KV cache using low-rank approximation.

        Args:
            kv_cache: KV cache tensor to compress

        Returns:
            Tuple of (left_matrix, right_matrix) representing the low-rank approximation
        """
        # Store original shape for reconstruction
        self.original_shape = kv_cache.shape

        # Reshape for SVD: (batch_size * num_heads, seq_len, head_dim)
        reshaped = kv_cache.view(-1, self.original_shape[-2], self.original_shape[-1])

        # Perform SVD
        U, S, V = torch.svd(reshaped)

        # Take only the top 'rank' singular values/vectors
        U_truncated = U[:, :, : self.rank]
        S_truncated = S[:, : self.rank]
        V_truncated = V[:, :, : self.rank]

        # Scale the singular values to the left matrix
        left_matrix = U_truncated * S_truncated.unsqueeze(-2)
        right_matrix = V_truncated

        return left_matrix, right_matrix

    def decompress(
        self, left_matrix: torch.Tensor, right_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Decompress the low-rank KV cache.

        Args:
            left_matrix: Left matrix from SVD
            right_matrix: Right matrix from SVD

        Returns:
            Decompressed KV cache
        """
        # Reconstruct the original matrix
        reconstructed = torch.matmul(left_matrix, right_matrix.transpose(-2, -1))

        # Reshape back to original dimensions
        return reconstructed.view(self.original_shape)


class AdaptivePrecisionKVCache(nn.Module):
    """
    KV cache with adaptive precision compression for Qwen3-Coder-Next model.
    """

    def __init__(self, config: CompressedKVCacheConfig):
        super().__init__()
        self.config = config
        self.threshold = config.adaptive_precision_threshold

    def compress(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """
        Compress the KV cache using adaptive precision.

        Args:
            kv_cache: KV cache tensor to compress

        Returns:
            Compressed KV cache with adaptive precision
        """
        # Identify values that require higher precision
        abs_values = torch.abs(kv_cache)

        # For values below threshold, we can use lower precision
        # For values above threshold, we maintain higher precision
        mask = abs_values > self.threshold

        # Create compressed version using lower precision for small values
        compressed = kv_cache.clone()

        # Apply different quantization based on magnitude
        small_values = compressed[~mask]
        large_values = compressed[mask]

        # Quantize small values more aggressively
        if small_values.numel() > 0:
            small_min, small_max = small_values.min(), small_values.max()
            if small_max != small_min:
                scale = 15.0  # 4-bit quantization for small values
                quantized_small = torch.round(
                    (small_values - small_min) / (small_max - small_min) * scale
                )
                compressed[~mask] = (quantized_small / scale) * (
                    small_max - small_min
                ) + small_min

        # Quantize large values with higher precision
        if large_values.numel() > 0:
            large_min, large_max = large_values.min(), large_values.max()
            if large_max != large_min:
                scale = 255.0  # 8-bit quantization for large values
                quantized_large = torch.round(
                    (large_values - large_min) / (large_max - large_min) * scale
                )
                compressed[mask] = (quantized_large / scale) * (
                    large_max - large_min
                ) + large_min

        return compressed


class SparseKVCache(nn.Module):
    """
    KV cache with sparse compression for Qwen3-Coder-Next model.
    """

    def __init__(self, config: CompressedKVCacheConfig):
        super().__init__()
        self.config = config
        self.compression_ratio = config.sparse_compression_ratio

    def compress(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """
        Compress the KV cache using sparsification.

        Args:
            kv_cache: KV cache tensor to compress

        Returns:
            Sparsified KV cache
        """
        # Calculate how many elements to keep based on compression ratio
        total_elements = kv_cache.numel()
        elements_to_keep = int(total_elements * (1 - self.compression_ratio))

        # Get the top-k values based on magnitude
        flat_cache = kv_cache.flatten()
        _, top_indices = torch.topk(
            torch.abs(flat_cache), elements_to_keep, largest=True
        )

        # Create a mask for the values to keep
        mask = torch.zeros_like(flat_cache, dtype=torch.bool)
        mask[top_indices] = True

        # Apply the mask to zero out small values
        sparsified = kv_cache * mask.view_as(kv_cache)

        return sparsified


class CombinedKVCacheCompression(nn.Module):
    """
    Combined KV cache compression using multiple techniques for Qwen3-Coder-Next model.
    """

    def __init__(self, config: CompressedKVCacheConfig):
        super().__init__()
        self.config = config

        # Initialize all compression methods
        if config.compression_method in [
            CompressionMethod.QUANTIZATION,
            CompressionMethod.COMBINED,
        ]:
            self.quantized_cache = QuantizedKVCache(config)

        if config.compression_method in [
            CompressionMethod.LOW_RANK,
            CompressionMethod.COMBINED,
        ]:
            self.low_rank_cache = LowRankKVCache(config)

        if config.compression_method in [
            CompressionMethod.ADAPTIVE_PRECISION,
            CompressionMethod.COMBINED,
        ]:
            self.adaptive_cache = AdaptivePrecisionKVCache(config)

        if config.compression_method in [
            CompressionMethod.SPARSE,
            CompressionMethod.COMBINED,
        ]:
            self.sparse_cache = SparseKVCache(config)

    def compress(self, kv_cache: torch.Tensor) -> Dict[str, Any]:
        """
        Compress the KV cache using the configured method.

        Args:
            kv_cache: KV cache tensor to compress

        Returns:
            Dictionary containing compressed cache and metadata
        """
        method = self.config.compression_method

        if method == CompressionMethod.QUANTIZATION:
            compressed = self.quantized_cache.compress(kv_cache)
            return {
                "compressed_data": compressed,
                "method": "quantization",
                "metadata": {
                    "min_val": self.quantized_cache.min_val,
                    "max_val": self.quantized_cache.max_val,
                    "range_val": self.quantized_cache.range_val,
                },
            }
        elif method == CompressionMethod.LOW_RANK:
            left, right = self.low_rank_cache.compress(kv_cache)
            return {
                "compressed_data": (left, right),
                "method": "low_rank",
                "metadata": {},
            }
        elif method == CompressionMethod.ADAPTIVE_PRECISION:
            compressed = self.adaptive_cache.compress(kv_cache)
            return {
                "compressed_data": compressed,
                "method": "adaptive_precision",
                "metadata": {},
            }
        elif method == CompressionMethod.SPARSE:
            compressed = self.sparse_cache.compress(kv_cache)
            return {"compressed_data": compressed, "method": "sparse", "metadata": {}}
        elif method == CompressionMethod.COMBINED:
            # Apply multiple compression techniques
            # For example, quantize then make sparse
            quantized = self.quantized_cache.compress(kv_cache)
            sparsified = self.sparse_cache.compress(quantized)

            return {
                "compressed_data": sparsified,
                "method": "combined",
                "metadata": {
                    "min_val": self.quantized_cache.min_val,
                    "max_val": self.quantized_cache.max_val,
                    "range_val": self.quantized_cache.range_val,
                },
            }
        else:
            # No compression
            return {"compressed_data": kv_cache, "method": "none", "metadata": {}}

    def decompress(self, compressed_result: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress the KV cache.

        Args:
            compressed_result: Dictionary containing compressed data and metadata

        Returns:
            Decompressed KV cache
        """
        method = compressed_result["method"]
        compressed_data = compressed_result["compressed_data"]
        metadata = compressed_result["metadata"]

        if method == "quantization":
            # Restore quantization parameters
            self.quantized_cache.min_val = metadata["min_val"]
            self.quantized_cache.max_val = metadata["max_val"]
            self.quantized_cache.range_val = metadata["range_val"]
            return self.quantized_cache.decompress(compressed_data)
        elif method == "low_rank":
            left, right = compressed_data
            return self.low_rank_cache.decompress(left, right)
        elif method == "adaptive_precision":
            return compressed_data  # Adaptive precision is lossy, return as is
        elif method == "sparse":
            return compressed_data  # Sparse is lossy, return as is
        elif method == "combined":
            # First decompress sparsity, then dequantize
            sparsified = compressed_data

            # Restore quantization parameters
            self.quantized_cache.min_val = metadata["min_val"]
            self.quantized_cache.max_val = metadata["max_val"]
            self.quantized_cache.range_val = metadata["range_val"]

            # Decompress quantization
            return self.quantized_cache.decompress(sparsified)
        else:
            # No compression
            return compressed_data


def apply_compressed_kv_cache_to_model(
    model: nn.Module, config: CompressedKVCacheConfig
) -> nn.Module:
    """
    Apply KV cache compression to the model.

    Args:
        model: The model to apply compression to
        config: KV cache compression configuration

    Returns:
        Model with KV cache compression applied
    """
    # In this implementation, we're preparing the compression system
    # The actual compression would happen during inference when KV caches are created
    model.kv_cache_compressor = CombinedKVCacheCompression(config)

    return model


def get_compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    Calculate the compression ratio.

    Args:
        original_size: Original size of the data
        compressed_size: Compressed size of the data

    Returns:
        Compression ratio (original_size / compressed_size)
    """
    if compressed_size == 0:
        return float("inf")
    return original_size / compressed_size


__all__ = [
    "CompressionMethod",
    "CompressedKVCacheConfig",
    "QuantizedKVCache",
    "LowRankKVCache",
    "AdaptivePrecisionKVCache",
    "SparseKVCache",
    "CombinedKVCacheCompression",
    "apply_compressed_kv_cache_to_model",
    "get_compression_ratio",
]
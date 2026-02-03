"""
Tensor Compression System for Real-Time Model Optimization

This module implements advanced tensor compression techniques including
PCA and SVD incremental compression for maintaining compact representations
of model weights during execution.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TensorCompressor:
    """
    Advanced tensor compression system using PCA and SVD techniques.
    """

    def __init__(
        self,
        compression_method: str = "incremental_pca",
        compression_ratio: float = 0.5,
        max_components: int = 512,
        incremental_update: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize the tensor compressor.

        Args:
            compression_method: Method to use for compression ("incremental_pca", "svd", "auto")
            compression_ratio: Target compression ratio (0.0 to 1.0, where 0.5 means 50% reduction)
            max_components: Maximum number of components to keep
            incremental_update: Whether to use incremental updates for compression models
            device: Device to perform compression on
        """
        self.compression_method = compression_method
        self.compression_ratio = compression_ratio
        self.max_components = max_components
        self.incremental_update = incremental_update
        self.device = device

        # Compression models
        self.pca_models: Dict[str, IncrementalPCA] = {}
        self.svd_models: Dict[str, TruncatedSVD] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Statistics
        self.compression_stats: Dict[str, Dict[str, float]] = {}

        logger.info(
            f"TensorCompressor initialized with method: {compression_method}, "
            f"ratio: {compression_ratio}, max_components: {max_components}"
        )

    def compress_tensor(
        self, tensor: torch.Tensor, tensor_id: str = "default"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress a tensor using the specified method.

        Args:
            tensor: Input tensor to compress
            tensor_id: Unique identifier for the tensor

        Returns:
            Tuple of (compressed_tensor, metadata_dict)
        """
        original_shape = tensor.shape
        original_size = tensor.numel()

        # Move tensor to CPU for compression operations
        tensor_cpu = tensor.cpu().detach().numpy()

        # Determine compression parameters
        n_samples, n_features = self._get_matrix_dimensions(tensor_cpu)
        target_components = min(
            int(n_features * self.compression_ratio),
            self.max_components,
            n_features - 1,  # Ensure we don't exceed feature count
        )

        if target_components <= 0:
            target_components = 1

        # Choose compression method based on tensor properties
        method = self.compression_method
        if method == "auto":
            if n_samples > n_features:
                method = "pca"
            else:
                method = "svd"

        compressed_data = None
        compression_metadata = {
            "original_shape": original_shape,
            "original_size": original_size,
            "target_components": target_components,
            "compression_method": method,
            "tensor_id": tensor_id,
        }

        try:
            if method in ["pca", "incremental_pca"]:
                compressed_data, metadata = self._compress_with_pca(
                    tensor_cpu, tensor_id, target_components
                )
            elif method == "svd":
                compressed_data, metadata = self._compress_with_svd(
                    tensor_cpu, tensor_id, target_components
                )
            else:
                raise ValueError(f"Unsupported compression method: {method}")

            compression_metadata.update(metadata)

            # Calculate compression statistics
            if isinstance(compressed_data, dict):
                compressed_size = 0
                for param in compressed_data.values():
                    if isinstance(param, torch.Tensor):
                        compressed_size += param.numel()
                    elif hasattr(param, "numel"):
                        compressed_size += param.numel()
                    elif hasattr(param, "size"):
                        compressed_size += param.size
                    elif isinstance(param, (list, tuple)):
                        compressed_size += len(param)
                    else:
                        # If it's a dict, sum its elements
                        if isinstance(param, dict):
                            for sub_param in param.values():
                                if isinstance(sub_param, torch.Tensor):
                                    compressed_size += sub_param.numel()
                                elif hasattr(sub_param, "numel"):
                                    compressed_size += sub_param.numel()
                                elif hasattr(sub_param, "size"):
                                    compressed_size += sub_param.size
                                elif isinstance(sub_param, (list, tuple)):
                                    compressed_size += len(sub_param)
                                else:
                                    compressed_size += 1  # Fallback
                        else:
                            compressed_size += 1  # Fallback
            else:
                compressed_size = compressed_data.numel()

            compression_ratio_actual = compressed_size / original_size
            compression_metadata["actual_compression_ratio"] = compression_ratio_actual
            compression_metadata["compression_saved_bytes"] = (
                original_size - compressed_size
            ) * tensor.element_size()

            # Store statistics
            self.compression_stats[tensor_id] = {
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio_actual,
                "saved_bytes": (original_size - compressed_size)
                * tensor.element_size(),
            }

            logger.debug(
                f"Compressed tensor {tensor_id}: {original_size} -> {compressed_size} "
                f"(ratio: {compression_ratio_actual:.3f})"
            )

            return compressed_data, compression_metadata

        except Exception as e:
            logger.error(f"Error compressing tensor {tensor_id}: {e}")
            # Return original tensor if compression fails
            return tensor, {
                "original_shape": original_shape,
                "original_size": original_size,
                "compression_failed": True,
                "error": str(e),
            }

    def decompress_tensor(
        self,
        compressed_data: Union[torch.Tensor, Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Decompress a tensor using stored metadata.

        Args:
            compressed_data: Compressed tensor data
            metadata: Metadata from compression step

        Returns:
            Decompressed tensor
        """
        tensor_id = metadata.get("tensor_id", "default")

        try:
            if metadata.get("compression_failed", False):
                # Return original tensor if compression failed
                return compressed_data

            method = metadata.get("compression_method", "pca")

            if method in ["pca", "incremental_pca"]:
                decompressed_tensor = self._decompress_with_pca(
                    compressed_data, metadata
                )
            elif method == "svd":
                decompressed_tensor = self._decompress_with_svd(
                    compressed_data, metadata
                )
            else:
                raise ValueError(f"Unsupported compression method: {method}")

            # Restore original shape if needed
            original_shape = metadata.get("original_shape")
            if decompressed_tensor.shape != original_shape:
                decompressed_tensor = decompressed_tensor.view(original_shape)

            logger.debug(
                f"Decompressed tensor {tensor_id}, shape: {decompressed_tensor.shape}"
            )
            return decompressed_tensor

        except Exception as e:
            logger.error(f"Error decompressing tensor {tensor_id}: {e}")
            # Return compressed data if decompression fails
            if (
                isinstance(compressed_data, dict)
                and "original_tensor" in compressed_data
            ):
                return compressed_data["original_tensor"]
            else:
                return compressed_data

    def _compress_with_pca(
        self, tensor: np.ndarray, tensor_id: str, n_components: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Compress tensor using PCA.

        Args:
            tensor: Input tensor as numpy array
            tensor_id: Unique identifier for the tensor
            n_components: Number of principal components to keep

        Returns:
            Tuple of (compressed_data_dict, metadata_dict)
        """
        # Reshape tensor to 2D matrix if needed
        reshaped_tensor, original_shape_info = self._reshape_for_compression(tensor)

        # Standardize the data
        if tensor_id not in self.scalers:
            self.scalers[tensor_id] = StandardScaler()

        standardized_tensor = self.scalers[tensor_id].fit_transform(reshaped_tensor)

        # Apply PCA
        if tensor_id not in self.pca_models or not self.incremental_update:
            self.pca_models[tensor_id] = IncrementalPCA(n_components=n_components)

        if self.incremental_update:
            # Use partial_fit for incremental updates
            # First call fit to initialize, then use partial_fit for updates
            if not hasattr(self.pca_models[tensor_id], "components_"):
                # First time - use fit_transform
                transformed = self.pca_models[tensor_id].fit_transform(
                    standardized_tensor
                )
            else:
                # For subsequent updates with IncrementalPCA, we need to refit with all data
                # since partial_fit doesn't update the transform of existing data
                # For this implementation, we'll just use fit_transform each time
                transformed = self.pca_models[tensor_id].fit_transform(
                    standardized_tensor
                )
        else:
            # Fit and transform in one step
            transformed = self.pca_models[tensor_id].fit_transform(standardized_tensor)

        # Convert to torch tensors
        compressed_tensor = torch.from_numpy(transformed).to(self.device)
        components = torch.from_numpy(self.pca_models[tensor_id].components_).to(
            self.device
        )
        mean = torch.from_numpy(self.scalers[tensor_id].mean_).to(self.device)
        scale = torch.from_numpy(self.scalers[tensor_id].scale_).to(self.device)

        compressed_data = {
            "transformed": compressed_tensor,
            "components": components,
            "mean": mean,
            "scale": scale,
            "original_shape_info": original_shape_info,
        }

        metadata = {
            "n_components": n_components,
            "explained_variance_ratio": float(
                np.sum(self.pca_models[tensor_id].explained_variance_ratio_)
            ),
            "compression_type": "pca",
        }

        return compressed_data, metadata

    def _decompress_with_pca(
        self, compressed_data: Dict[str, torch.Tensor], metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Decompress tensor using PCA.

        Args:
            compressed_data: Compressed data dictionary
            metadata: Metadata from compression step

        Returns:
            Decompressed tensor
        """
        transformed = compressed_data["transformed"].cpu().numpy()
        components = compressed_data["components"].cpu().numpy()
        mean = compressed_data["mean"].cpu().numpy()
        scale = compressed_data["scale"].cpu().numpy()

        # Reconstruct using inverse transform
        standardized_reconstructed = transformed @ components
        reconstructed = standardized_reconstructed * scale + mean

        # Reshape back to original dimensions
        original_shape_info = compressed_data.get("original_shape_info", {})
        tensor = self._restore_from_compression(reconstructed, original_shape_info)

        return torch.from_numpy(tensor).to(self.device)

    def _compress_with_svd(
        self, tensor: np.ndarray, tensor_id: str, n_components: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Compress tensor using SVD.

        Args:
            tensor: Input tensor as numpy array
            tensor_id: Unique identifier for the tensor
            n_components: Number of singular values/vectors to keep

        Returns:
            Tuple of (compressed_data_dict, metadata_dict)
        """
        # Reshape tensor to 2D matrix if needed
        reshaped_tensor, original_shape_info = self._reshape_for_compression(tensor)

        # Apply SVD
        if tensor_id not in self.svd_models or not self.incremental_update:
            self.svd_models[tensor_id] = TruncatedSVD(n_components=n_components)

        if self.incremental_update:
            # Note: TruncatedSVD doesn't support incremental fitting like IncrementalPCA
            # So we'll just fit on the current data
            transformed = self.svd_models[tensor_id].fit_transform(reshaped_tensor)
        else:
            transformed = self.svd_models[tensor_id].fit_transform(reshaped_tensor)

        # Convert to torch tensors
        U = torch.from_numpy(self.svd_models[tensor_id].components_).to(
            self.device
        )  # Actually V^T
        S = torch.from_numpy(self.svd_models[tensor_id].singular_values_).to(
            self.device
        )
        # The transformed data is U*S, and we store V^T separately
        compressed_tensor = torch.from_numpy(transformed).to(self.device)

        compressed_data = {
            "transformed": compressed_tensor,  # U*S
            "Vt": U,  # V^T components
            "singular_values": S,
            "original_shape_info": original_shape_info,
        }

        metadata = {"n_components": n_components, "compression_type": "svd"}

        return compressed_data, metadata

    def _decompress_with_svd(
        self, compressed_data: Dict[str, torch.Tensor], metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Decompress tensor using SVD.

        Args:
            compressed_data: Compressed data dictionary
            metadata: Metadata from compression step

        Returns:
            Decompressed tensor
        """
        transformed = compressed_data["transformed"].cpu().numpy()  # U*S
        Vt = compressed_data["Vt"].cpu().numpy()  # V^T
        S = compressed_data["singular_values"].cpu().numpy()

        # Reconstruct: (U*S) @ V
        reconstructed = transformed @ (S[:, np.newaxis] * Vt)

        # Reshape back to original dimensions
        original_shape_info = compressed_data.get("original_shape_info", {})
        tensor = self._restore_from_compression(reconstructed, original_shape_info)

        return torch.from_numpy(tensor).to(self.device)

    def _reshape_for_compression(
        self, tensor: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reshape tensor to 2D matrix for compression algorithms.

        Args:
            tensor: Input tensor

        Returns:
            Tuple of (reshaped_matrix, shape_info_dict)
        """
        original_shape = tensor.shape

        if len(original_shape) == 1:
            # 1D tensor: treat as (1, n) or (n, 1) depending on size
            if original_shape[0] > 1:
                reshaped = tensor.reshape(1, -1)
                shape_info = {
                    "original_shape": original_shape,
                    "reshaped_shape": reshaped.shape,
                    "mode": "1d_to_2d",
                }
            else:
                reshaped = tensor.reshape(-1, 1)
                shape_info = {
                    "original_shape": original_shape,
                    "reshaped_shape": reshaped.shape,
                    "mode": "1d_to_2d",
                }
        elif len(original_shape) == 2:
            # Already 2D
            reshaped = tensor
            shape_info = {
                "original_shape": original_shape,
                "reshaped_shape": original_shape,
                "mode": "2d_preserved",
            }
        else:
            # Multi-dimensional: flatten all but last dimension
            reshaped = tensor.reshape(-1, original_shape[-1])
            shape_info = {
                "original_shape": original_shape,
                "reshaped_shape": reshaped.shape,
                "mode": "nd_flattened",
            }

        return reshaped, shape_info

    def _restore_from_compression(
        self, matrix: np.ndarray, shape_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Restore tensor from 2D matrix back to original shape.

        Args:
            matrix: Compressed 2D matrix
            shape_info: Information about original shape

        Returns:
            Restored tensor with original shape
        """
        original_shape = shape_info["original_shape"]

        if shape_info["mode"] == "1d_to_2d":
            # Restore 1D tensor
            restored = matrix.flatten()
        elif shape_info["mode"] == "2d_preserved":
            # Original was 2D, preserve shape
            restored = matrix
        elif shape_info["mode"] == "nd_flattened":
            # Restore multi-dimensional tensor
            restored = matrix.reshape(original_shape)
        else:
            # Default: reshape to original shape
            restored = matrix.reshape(original_shape)

        return restored

    def _get_matrix_dimensions(self, tensor: np.ndarray) -> Tuple[int, int]:
        """
        Get appropriate matrix dimensions for compression.

        Args:
            tensor: Input tensor

        Returns:
            Tuple of (n_samples, n_features)
        """
        if len(tensor.shape) == 1:
            return 1, tensor.shape[0]
        elif len(tensor.shape) == 2:
            return tensor.shape[0], tensor.shape[1]
        else:
            # For multi-dimensional tensors, treat all but last dim as samples
            n_samples = int(np.prod(tensor.shape[:-1]))
            n_features = tensor.shape[-1]
            return n_samples, n_features

    def get_compression_stats(self, tensor_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get compression statistics.

        Args:
            tensor_id: Specific tensor ID to get stats for, or None for all

        Returns:
            Dictionary with compression statistics
        """
        if tensor_id is None:
            return self.compression_stats
        else:
            return self.compression_stats.get(tensor_id, {})

    def clear_compression_models(self):
        """
        Clear all compression models to free memory.
        """
        self.pca_models.clear()
        self.svd_models.clear()
        self.scalers.clear()
        self.compression_stats.clear()
        logger.info("Cleared all compression models")


class AdaptiveTensorCompressor(TensorCompressor):
    """
    Adaptive tensor compressor that adjusts compression based on available memory.
    """

    def __init__(
        self,
        compression_method: str = "incremental_pca",
        base_compression_ratio: float = 0.5,
        max_components: int = 512,
        incremental_update: bool = True,
        device: str = "cpu",
        memory_threshold_high: float = 0.8,
        memory_threshold_critical: float = 0.9,
    ):
        """
        Initialize the adaptive tensor compressor.

        Args:
            compression_method: Method to use for compression
            base_compression_ratio: Base compression ratio
            max_components: Maximum number of components to keep
            incremental_update: Whether to use incremental updates
            device: Device to perform compression on
            memory_threshold_high: Memory threshold for high compression (0.0 to 1.0)
            memory_threshold_critical: Memory threshold for critical compression (0.0 to 1.0)
        """
        super().__init__(
            compression_method=compression_method,
            compression_ratio=base_compression_ratio,
            max_components=max_components,
            incremental_update=incremental_update,
            device=device,
        )

        self.base_compression_ratio = base_compression_ratio
        self.memory_threshold_high = memory_threshold_high
        self.memory_threshold_critical = memory_threshold_critical

        # Track memory usage over time
        self.memory_history: List[float] = []

    def get_current_memory_usage(self) -> float:
        """
        Get current memory usage ratio.

        Returns:
            Memory usage ratio (0.0 to 1.0)
        """
        if torch.cuda.is_available():
            # Get GPU memory usage
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_memory = torch.cuda.get_device_properties(0).total_memory
            return reserved / max_memory
        else:
            # Get system memory usage
            import psutil

            return psutil.virtual_memory().percent / 100.0

    def adjust_compression_ratio(self) -> float:
        """
        Adjust compression ratio based on current memory usage.

        Returns:
            Adjusted compression ratio
        """
        current_memory = self.get_current_memory_usage()
        self.memory_history.append(current_memory)

        # Keep only last 10 memory readings
        if len(self.memory_history) > 10:
            self.memory_history = self.memory_history[-10:]

        # Calculate average memory usage
        avg_memory = sum(self.memory_history) / len(self.memory_history)

        # Adjust compression ratio based on memory pressure
        if avg_memory >= self.memory_threshold_critical:
            # Critical memory pressure - maximum compression
            adjusted_ratio = min(0.1, self.base_compression_ratio * 0.5)
        elif avg_memory >= self.memory_threshold_high:
            # High memory pressure - increased compression
            adjusted_ratio = min(0.25, self.base_compression_ratio * 0.75)
        else:
            # Normal memory usage - base compression
            adjusted_ratio = self.base_compression_ratio

        logger.debug(
            f"Memory usage: {avg_memory:.3f}, Adjusted compression ratio: {adjusted_ratio:.3f}"
        )
        return adjusted_ratio

    def compress_tensor(
        self, tensor: torch.Tensor, tensor_id: str = "default"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress a tensor with adaptive compression ratio based on memory usage.

        Args:
            tensor: Input tensor to compress
            tensor_id: Unique identifier for the tensor

        Returns:
            Tuple of (compressed_tensor, metadata_dict)
        """
        # Adjust compression ratio based on current memory usage
        self.compression_ratio = self.adjust_compression_ratio()

        return super().compress_tensor(tensor, tensor_id)


# Global tensor compressor instance
_tensor_compressor: Optional[AdaptiveTensorCompressor] = None


def get_tensor_compressor() -> AdaptiveTensorCompressor:
    """
    Get the global tensor compressor instance.

    Returns:
        AdaptiveTensorCompressor instance
    """
    global _tensor_compressor
    if _tensor_compressor is None:
        _tensor_compressor = AdaptiveTensorCompressor()
    return _tensor_compressor


def compress_model_weights(
    model: nn.Module, compression_ratio: float = 0.5, device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Compress all model weights using tensor compression.

    Args:
        model: PyTorch model to compress
        compression_ratio: Compression ratio to use
        device: Device to perform compression on

    Returns:
        Tuple of (compressed_model, compression_metadata)
    """
    compressor = AdaptiveTensorCompressor(
        compression_method="incremental_pca",
        base_compression_ratio=compression_ratio,
        device=device,
    )

    compression_metadata = {}

    # Compress parameters in-place
    for name, param in model.named_parameters():
        if (
            param.requires_grad or len(param.shape) > 1
        ):  # Only compress trainable or multi-dimensional params
            compressed_param, metadata = compressor.compress_tensor(param, name)
            compression_metadata[name] = metadata

            # If compression returned a tensor, we can replace the parameter
            # If it returned a dict (complex compressed format), we need to handle differently
            if isinstance(compressed_param, torch.Tensor):
                # Direct tensor replacement
                with torch.no_grad():
                    if param.shape == compressed_param.shape:
                        param.copy_(compressed_param)
                    else:
                        # If shapes don't match, we can't directly replace
                        # In this case, we'll skip compression for this parameter
                        pass
            else:
                # For complex compressed formats (dict), we can't directly replace
                # We'll need to implement a different approach for model compression
                # For now, we'll skip replacing this parameter
                pass

    return model, compression_metadata


def decompress_model_weights(
    compressed_model: nn.Module, compression_metadata: Dict[str, Any]
) -> nn.Module:
    """
    Decompress model weights back to original form.

    Args:
        compressed_model: Model with compressed weights
        compression_metadata: Metadata from compression step

    Returns:
        Model with decompressed weights
    """
    # For now, this function is a placeholder since our current compression approach
    # doesn't actually modify the model parameters in-place in a way that requires decompression
    # The actual decompression would happen when using the metadata to restore original tensors
    return compressed_model


__all__ = [
    "TensorCompressor",
    "AdaptiveTensorCompressor",
    "get_tensor_compressor",
    "compress_model_weights",
    "decompress_model_weights",
]

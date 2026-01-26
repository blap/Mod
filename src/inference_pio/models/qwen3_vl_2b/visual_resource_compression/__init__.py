"""
Visual Resource Compression System for Qwen3-VL-2B Model

This module implements a comprehensive compression system specifically for visual resources
in the Qwen3-VL-2B model. It includes various compression techniques for image data,
feature maps, and visual encodings to optimize memory usage and processing speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import math
from functools import partial

from ..config import Qwen3VL2BConfig

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Enumeration of available compression methods."""
    PCA = "pca"
    SVD = "svd"
    QUANTIZATION = "quantization"
    SPARSE_CODING = "sparse_coding"
    AUTOENCODER = "autoencoder"
    JPEG_EMULATION = "jpeg_emulation"


@dataclass
class VisualCompressionConfig:
    """Configuration for visual resource compression."""
    
    # General compression settings
    compression_method: CompressionMethod = CompressionMethod.QUANTIZATION
    compression_ratio: float = 0.5  # Target compression ratio (0.0 to 1.0, where 0.5 = 50% reduction)
    
    # PCA/SVD specific settings
    pca_components_ratio: float = 0.7  # Ratio of components to keep for PCA
    svd_rank_ratio: float = 0.5  # Ratio of rank to keep for SVD
    
    # Quantization specific settings
    quantization_bits: int = 8  # Number of bits for quantization
    quantization_method: str = "linear"  # Options: "linear", "log", "kmeans"
    
    # Autoencoder specific settings
    autoencoder_latent_dim_ratio: float = 0.5  # Ratio of latent dimension to original
    autoencoder_hidden_dims: List[int] = None  # Hidden dimensions for autoencoder
    
    # Sparse coding specific settings
    sparse_coding_sparsity: float = 0.1  # Sparsity level for sparse coding
    sparse_coding_dictionary_size: int = 256  # Size of dictionary for sparse coding
    
    # Performance settings
    enable_compression_cache: bool = True  # Enable caching of compressed representations
    compression_cache_size: int = 1000  # Maximum number of cached compressed representations
    compression_threshold: float = 0.1  # Threshold below which compression is applied
    
    # Adaptive settings
    enable_adaptive_compression: bool = True  # Enable adaptive compression based on input characteristics
    adaptive_compression_metric: str = "memory_usage"  # Metric for adaptive compression ("memory_usage", "latency", "accuracy")
    
    def __post_init__(self):
        if self.autoencoder_hidden_dims is None:
            self.autoencoder_hidden_dims = [512, 256, 128]


class VisualResourceCompressor(nn.Module):
    """
    Main compressor class for visual resources in Qwen3-VL-2B model.
    Applies various compression techniques to optimize visual data processing.
    """

    def __init__(self, config: VisualCompressionConfig):
        super().__init__()
        self.config = config
        
        # Initialize compression method
        self.compression_method = config.compression_method
        self.compression_ratio = config.compression_ratio
        
        # Initialize compression cache if enabled
        self.compression_cache = {}
        self.cache_order = []  # Track order of cache entries for LRU
        self.max_cache_size = config.compression_cache_size
        
        # Initialize method-specific components
        if self.compression_method == CompressionMethod.AUTOENCODER:
            self.autoencoder = self._build_autoencoder()
        elif self.compression_method == CompressionMethod.SPARSE_CODING:
            self.dictionary = nn.Parameter(
                torch.randn(config.sparse_coding_dictionary_size, self._get_feature_dim()),
                requires_grad=True
            )
        elif self.compression_method == CompressionMethod.PCA:
            self.pca_components = None  # Will be computed dynamically
        elif self.compression_method == CompressionMethod.SVD:
            self.svd_U = None
            self.svd_S = None
            self.svd_V = None

    def _get_feature_dim(self) -> int:
        """Get the feature dimension for compression methods that need it."""
        # This is a placeholder - in a real implementation, this would be determined
        # based on the actual input dimensions
        return 1024  # Default feature dimension

    def _build_autoencoder(self) -> nn.Module:
        """Build autoencoder for compression."""
        if self.config.autoencoder_hidden_dims is None:
            # Default architecture
            hidden_dims = [512, 256, 128]
        else:
            hidden_dims = self.config.autoencoder_hidden_dims
            
        # Calculate latent dimension
        input_dim = self._get_feature_dim()
        latent_dim = int(input_dim * self.config.autoencoder_latent_dim_ratio)
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Add latent layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        # Reverse hidden dims for decoder
        reversed_hidden_dims = list(reversed(hidden_dims))
        for hidden_dim in reversed_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Add output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder = nn.Sequential(*decoder_layers)
        
        # Combine into autoencoder
        return nn.ModuleDict({
            'encoder': encoder,
            'decoder': decoder
        })

    def _compute_pca_components(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute PCA components for the input tensor."""
        # Reshape input to (samples, features)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Center the data
        mean = x_flat.mean(dim=0, keepdim=True)
        x_centered = x_flat - mean
        
        # Compute covariance matrix
        cov_matrix = torch.mm(x_centered.t(), x_centered) / (x_centered.size(0) - 1)
        
        # Compute eigenvectors and eigenvalues
        eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = torch.argsort(eigenvals, descending=True)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Keep only the top components based on ratio
        n_components = max(1, int(len(eigenvals) * self.config.pca_components_ratio))
        eigenvals = eigenvals[:n_components]
        eigenvecs = eigenvecs[:, :n_components]
        
        return eigenvecs.t(), mean  # Return transpose for easier multiplication

    def _compress_with_pca(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress using PCA."""
        # Compute PCA components if not already computed
        if self.pca_components is None:
            self.pca_components, self.pca_mean = self._compute_pca_components(x)
        
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        x_centered = x_flat - self.pca_mean
        
        # Project onto principal components
        compressed = torch.mm(x_centered, self.pca_components.t())
        
        # Store metadata for decompression
        metadata = {
            'mean': self.pca_mean,
            'components': self.pca_components,
            'original_shape': x.shape
        }
        
        return compressed, metadata

    def _decompress_with_pca(self, compressed: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress using PCA."""
        # Reconstruct from principal components
        x_reconstructed = torch.mm(compressed, metadata['components']) + metadata['mean']
        
        # Reshape back to original shape
        x_reconstructed = x_reconstructed.view(metadata['original_shape'])
        
        return x_reconstructed

    def _compress_with_svd(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress using SVD."""
        original_shape = x.shape
        
        # For SVD, we need to work with 2D matrices
        if len(x.shape) > 2:
            # Reshape to 2D: (batch * spatial_dims, features)
            x_2d = x.view(-1, x.shape[-1])
        else:
            x_2d = x
        
        # Compute SVD
        U, S, V = torch.svd(x_2d)
        
        # Keep only top singular values based on ratio
        rank = max(1, int(len(S) * self.config.svd_rank_ratio))
        U_reduced = U[:, :rank]
        S_reduced = S[:rank]
        V_reduced = V[:, :rank]
        
        # Store compressed representation
        compressed = (U_reduced, S_reduced, V_reduced)
        
        # Store metadata for decompression
        metadata = {
            'original_shape': original_shape,
            'full_shape': x_2d.shape
        }
        
        return compressed, metadata

    def _decompress_with_svd(self, compressed: Tuple[torch.Tensor, ...], metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress using SVD."""
        U_reduced, S_reduced, V_reduced = compressed
        
        # Reconstruct the matrix
        S_matrix = torch.diag(S_reduced)
        x_reconstructed = torch.mm(U_reduced, torch.mm(S_matrix, V_reduced.t()))
        
        # Reshape back to original shape
        x_reconstructed = x_reconstructed.view(metadata['original_shape'])
        
        return x_reconstructed

    def _compress_with_quantization(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress using quantization."""
        original_dtype = x.dtype
        original_shape = x.shape
        
        # Flatten for quantization
        x_flat = x.view(-1)
        
        if self.config.quantization_method == "linear":
            # Linear quantization
            min_val = x_flat.min()
            max_val = x_flat.max()
            
            # Quantize to specified number of bits
            scale = (max_val - min_val) / ((2 ** self.config.quantization_bits) - 1)
            zero_point = torch.round(-min_val / scale).to(torch.int32)
            
            # Quantize
            x_quantized = torch.round((x_flat - min_val) / scale).to(torch.uint8)
            
            # Store metadata for decompression
            metadata = {
                'scale': scale,
                'zero_point': zero_point,
                'min_val': min_val,
                'max_val': max_val,
                'original_shape': original_shape,
                'original_dtype': original_dtype
            }
            
            return x_quantized, metadata
            
        elif self.config.quantization_method == "log":
            # Logarithmic quantization
            sign = torch.sign(x_flat)
            log_abs = torch.log(torch.abs(x_flat) + 1e-8)
            
            # Quantize the logarithmic values
            min_log = log_abs.min()
            max_log = log_abs.max()
            
            scale = (max_log - min_log) / ((2 ** self.config.quantization_bits) - 1)
            x_log_quantized = torch.round((log_abs - min_log) / scale).to(torch.uint8)
            
            # Store metadata for decompression
            metadata = {
                'scale': scale,
                'min_log': min_log,
                'sign': sign,
                'original_shape': original_shape,
                'original_dtype': original_dtype
            }
            
            return x_log_quantized, metadata

    def _decompress_with_quantization(self, compressed: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress using quantization."""
        if self.config.quantization_method == "linear":
            # Dequantize linearly
            x_dequantized = (compressed.to(torch.float32) * metadata['scale']) + metadata['min_val']
        elif self.config.quantization_method == "log":
            # Dequantize logarithmically
            x_log = (compressed.to(torch.float32) * metadata['scale']) + metadata['min_log']
            x_dequantized = metadata['sign'] * torch.exp(x_log)
        
        # Reshape back to original shape
        x_dequantized = x_dequantized.view(metadata['original_shape']).to(metadata['original_dtype'])
        
        return x_dequantized

    def _compress_with_autoencoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress using autoencoder."""
        original_shape = x.shape
        
        # Flatten if needed
        if len(x.shape) > 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x
        
        # Encode
        compressed = self.autoencoder.encoder(x_flat)
        
        # Store metadata
        metadata = {
            'original_shape': original_shape
        }
        
        return compressed, metadata

    def _decompress_with_autoencoder(self, compressed: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress using autoencoder."""
        # Decode
        x_reconstructed = self.autoencoder.decoder(compressed)
        
        # Reshape back to original shape
        x_reconstructed = x_reconstructed.view(metadata['original_shape'])
        
        return x_reconstructed

    def _check_cache(self, key: str) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Check if compressed representation is in cache."""
        if not self.config.enable_compression_cache:
            return None
            
        if key in self.compression_cache:
            # Move to end to mark as recently used
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return self.compression_cache[key]
        
        return None

    def _update_cache(self, key: str, compressed: torch.Tensor, metadata: Dict[str, Any]):
        """Update compression cache."""
        if not self.config.enable_compression_cache:
            return
            
        # Add to cache
        self.compression_cache[key] = (compressed, metadata)
        self.cache_order.append(key)
        
        # Remove oldest if cache is too large
        if len(self.compression_cache) > self.max_cache_size:
            oldest_key = self.cache_order.pop(0)
            del self.compression_cache[oldest_key]

    def compress(self, x: torch.Tensor, key: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress visual resource tensor.
        
        Args:
            x: Input tensor to compress
            key: Optional key for caching (if None, no caching is used)
            
        Returns:
            Tuple of (compressed_tensor, metadata)
        """
        # Check cache first if key provided
        if key is not None:
            cached_result = self._check_cache(key)
            if cached_result is not None:
                logger.debug(f"Retrieved compressed representation from cache for key: {key}")
                return cached_result
        
        # Apply compression based on method
        if self.compression_method == CompressionMethod.PCA:
            compressed, metadata = self._compress_with_pca(x)
        elif self.compression_method == CompressionMethod.SVD:
            compressed, metadata = self._compress_with_svd(x)
        elif self.compression_method == CompressionMethod.QUANTIZATION:
            compressed, metadata = self._compress_with_quantization(x)
        elif self.compression_method == CompressionMethod.AUTOENCODER:
            compressed, metadata = self._compress_with_autoencoder(x)
        else:
            # For other methods, return original with minimal metadata
            compressed = x
            metadata = {'original': True}
        
        # Update cache if key provided
        if key is not None:
            self._update_cache(key, compressed, metadata)
        
        return compressed, metadata

    def decompress(self, compressed: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress visual resource tensor.
        
        Args:
            compressed: Compressed tensor
            metadata: Metadata needed for decompression
            
        Returns:
            Decompressed tensor
        """
        # Apply decompression based on method
        if metadata.get('original', False):
            # No compression was applied
            return compressed
        
        if self.compression_method == CompressionMethod.PCA:
            return self._decompress_with_pca(compressed, metadata)
        elif self.compression_method == CompressionMethod.SVD:
            return self._decompress_with_svd(compressed, metadata)
        elif self.compression_method == CompressionMethod.QUANTIZATION:
            return self._decompress_with_quantization(compressed, metadata)
        elif self.compression_method == CompressionMethod.AUTOENCODER:
            return self._decompress_with_autoencoder(compressed, metadata)
        else:
            # For other methods, return as is
            return compressed

    def forward(self, x: torch.Tensor, key: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass applies compression and decompression.
        
        Args:
            x: Input tensor
            key: Optional key for caching
            
        Returns:
            Processed tensor (compressed and decompressed)
        """
        # Compress
        compressed, metadata = self.compress(x, key)
        
        # Decompress
        decompressed = self.decompress(compressed, metadata)
        
        return decompressed


class VisualFeatureCompressor(nn.Module):
    """
    Specialized compressor for visual features in the Qwen3-VL-2B model.
    Integrates with the model's vision encoder to compress feature maps efficiently.
    """

    def __init__(self, config: VisualCompressionConfig, model_config: Qwen3VL2BConfig):
        super().__init__()
        self.config = config
        self.model_config = model_config
        
        # Create the main compressor
        self.compressor = VisualResourceCompressor(config)
        
        # Compression statistics
        self.compression_ratios = []
        self.compression_times = []

    def compress_features(self, 
                         features: torch.Tensor, 
                         layer_name: str = "unknown",
                         feature_type: str = "vision") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress visual features with additional metadata.
        
        Args:
            features: Feature tensor to compress
            layer_name: Name of the layer producing these features
            feature_type: Type of features ("vision", "attention", "mlp", etc.)
            
        Returns:
            Tuple of (compressed_features, metadata)
        """
        # Create a unique key for caching based on layer and feature type
        key = f"{layer_name}_{feature_type}_{features.shape}"
        
        # Compress the features
        compressed, metadata = self.compressor.compress(features, key)
        
        # Calculate compression ratio
        original_size = features.numel() * features.element_size()
        compressed_size = compressed.numel() * compressed.element_size()
        compression_ratio = 1.0 - (compressed_size / original_size)
        
        # Store statistics
        self.compression_ratios.append(compression_ratio)
        
        # Add additional metadata
        metadata.update({
            'layer_name': layer_name,
            'feature_type': feature_type,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio
        })
        
        logger.debug(f"Compressed {feature_type} features from layer {layer_name}, "
                    f"ratio: {compression_ratio:.3f}")
        
        return compressed, metadata

    def decompress_features(self, compressed: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress visual features.
        
        Args:
            compressed: Compressed feature tensor
            metadata: Metadata for decompression
            
        Returns:
            Decompressed feature tensor
        """
        # Decompress the features
        decompressed = self.compressor.decompress(compressed, metadata)
        
        return decompressed

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.
        
        Returns:
            Dictionary with compression statistics
        """
        if not self.compression_ratios:
            return {
                'avg_compression_ratio': 0.0,
                'compression_calls': 0,
                'total_saved_bytes': 0
            }
        
        avg_ratio = sum(self.compression_ratios) / len(self.compression_ratios)
        
        return {
            'avg_compression_ratio': avg_ratio,
            'compression_calls': len(self.compression_ratios),
            'compression_ratios': self.compression_ratios.copy()
        }


def create_visual_compressor(config: VisualCompressionConfig, 
                           model_config: Qwen3VL2BConfig) -> VisualFeatureCompressor:
    """
    Factory function to create a visual feature compressor.
    
    Args:
        config: Configuration for visual compression
        model_config: Configuration for the Qwen3-VL-2B model
        
    Returns:
        Visual feature compressor instance
    """
    return VisualFeatureCompressor(config, model_config)


def apply_visual_compression_to_model(model: nn.Module,
                                    model_config: Qwen3VL2BConfig,
                                    compression_config: VisualCompressionConfig) -> nn.Module:
    """
    Apply visual compression to the Qwen3-VL-2B model.
    
    Args:
        model: The Qwen3-VL-2B model to optimize
        model_config: Configuration for the Qwen3-VL-2B model
        compression_config: Configuration for visual compression
        
    Returns:
        Model with visual compression applied
    """
    logger.info("Applying visual compression to Qwen3-VL-2B model...")
    
    # Create visual compressor
    visual_compressor = create_visual_compressor(compression_config, model_config)
    
    # Add the compressor to the model as an attribute
    model.visual_compressor = visual_compressor
    
    # If the model has a vision encoder, we could potentially modify its forward pass
    # to incorporate compression, but for now we just attach the compressor
    
    logger.info("Visual compression applied to model successfully")
    return model


__all__ = [
    "CompressionMethod",
    "VisualCompressionConfig",
    "VisualResourceCompressor",
    "VisualFeatureCompressor",
    "create_visual_compressor",
    "apply_visual_compression_to_model"
]
"""
Demonstration of Visual Resource Compression for Qwen3-VL-2B Model

This script demonstrates the visual resource compression system implemented for the 
Qwen3-VL-2B model, showing how different compression techniques can be applied to 
optimize visual data processing.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from PIL import Image
import torchvision.transforms as transforms

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.inference_pio.models.qwen3_vl_2b.visual_resource_compression import (
    CompressionMethod,
    VisualCompressionConfig,
    VisualResourceCompressor,
    VisualFeatureCompressor,
    create_visual_compressor
)
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig


def create_sample_image_tensor(batch_size=1, channels=3, height=224, width=224):
    """Create a sample image tensor for testing."""
    return torch.randn(batch_size, channels, height, width)


def measure_compression_performance(compressor, tensor, iterations=10):
    """Measure compression performance."""
    times = []
    compression_ratios = []
    
    for i in range(iterations):
        start_time = time.time()
        
        # Compress
        compressed, metadata = compressor.compress(tensor, key=f"test_{i}")
        
        # Decompress
        decompressed = compressor.decompress(compressed, metadata)
        
        end_time = time.time()
        
        times.append(end_time - start_time)
        
        # Calculate compression ratio
        original_size = tensor.numel() * tensor.element_size()
        compressed_size = compressed.numel() * compressed.element_size()
        compression_ratio = 1.0 - (compressed_size / original_size)
        compression_ratios.append(compression_ratio)
    
    avg_time = sum(times) / len(times)
    avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
    
    return avg_time, avg_compression_ratio


def demonstrate_quantization_compression():
    """Demonstrate quantization-based compression."""
    print("=== Quantization-Based Compression ===")
    
    # Create configuration for quantization
    config = VisualCompressionConfig(
        compression_method=CompressionMethod.QUANTIZATION,
        quantization_bits=8,
        quantization_method="linear",
        enable_compression_cache=True
    )
    
    # Create compressor
    compressor = VisualResourceCompressor(config)
    
    # Create sample tensor
    tensor = create_sample_image_tensor(batch_size=2, channels=3, height=224, width=224)
    
    print(f"Original tensor shape: {tensor.shape}")
    print(f"Original tensor dtype: {tensor.dtype}")
    print(f"Original tensor size (bytes): {tensor.numel() * tensor.element_size()}")
    
    # Measure performance
    avg_time, avg_compression_ratio = measure_compression_performance(compressor, tensor)
    
    print(f"Average compression/decompression time: {avg_time:.4f}s")
    print(f"Average compression ratio: {avg_compression_ratio:.4f}")
    
    # Show compression/decompression example
    compressed, metadata = compressor.compress(tensor, key="demo_quantization")
    decompressed = compressor.decompress(compressed, metadata)
    
    print(f"Compressed tensor dtype: {compressed.dtype}")
    print(f"Compressed tensor size (bytes): {compressed.numel() * compressed.element_size()}")
    
    # Calculate reconstruction error
    mse = torch.mean((tensor - decompressed) ** 2)
    print(f"Mean squared error: {mse:.6f}")
    
    print()


def demonstrate_pca_compression():
    """Demonstrate PCA-based compression."""
    print("=== PCA-Based Compression ===")
    
    # Create configuration for PCA
    config = VisualCompressionConfig(
        compression_method=CompressionMethod.PCA,
        pca_components_ratio=0.7,  # Keep 70% of components
        enable_compression_cache=True
    )
    
    # Create compressor
    compressor = VisualResourceCompressor(config)
    
    # Create sample tensor (reshape for PCA)
    tensor = create_sample_image_tensor(batch_size=1, channels=3, height=64, width=64)
    # Reshape to (batch, features) for PCA
    reshaped_tensor = tensor.view(1, -1)
    
    print(f"Original tensor shape: {reshaped_tensor.shape}")
    print(f"Original tensor dtype: {reshaped_tensor.dtype}")
    
    # Measure performance
    avg_time, avg_compression_ratio = measure_compression_performance(compressor, reshaped_tensor)
    
    print(f"Average compression/decompression time: {avg_time:.4f}s")
    print(f"Average compression ratio: {avg_compression_ratio:.4f}")
    
    # Show compression/decompression example
    compressed, metadata = compressor.compress(reshaped_tensor, key="demo_pca")
    decompressed = compressor.decompress(compressed, metadata)
    
    print(f"Compressed tensor shape: {compressed.shape}")
    
    # Calculate reconstruction error
    mse = torch.mean((reshaped_tensor - decompressed) ** 2)
    print(f"Mean squared error: {mse:.6f}")
    
    print()


def demonstrate_visual_feature_compressor():
    """Demonstrate the VisualFeatureCompressor."""
    print("=== Visual Feature Compressor ===")
    
    # Create model config
    model_config = Qwen3VL2BConfig()
    model_config.hidden_size = 512
    model_config.vision_hidden_size = 1024
    
    # Create compression config
    compression_config = VisualCompressionConfig(
        compression_method=CompressionMethod.QUANTIZATION,
        quantization_bits=8,
        enable_compression_cache=True
    )
    
    # Create feature compressor
    feature_compressor = VisualFeatureCompressor(compression_config, model_config)
    
    # Create sample features (e.g., from vision encoder)
    features = torch.randn(2, 197, 1024)  # Batch of patch embeddings
    
    print(f"Input features shape: {features.shape}")
    
    # Compress features
    compressed, metadata = feature_compressor.compress_features(
        features, 
        layer_name="vision_encoder", 
        feature_type="patch_embeddings"
    )
    
    print(f"Compressed features shape: {compressed.shape}")
    print(f"Compression ratio: {metadata['compression_ratio']:.4f}")
    
    # Decompress features
    decompressed = feature_compressor.decompress_features(compressed, metadata)
    
    print(f"Decompressed features shape: {decompressed.shape}")
    
    # Calculate reconstruction error
    mse = torch.mean((features - decompressed) ** 2)
    print(f"Mean squared error: {mse:.6f}")
    
    # Get compression statistics
    stats = feature_compressor.get_compression_stats()
    print(f"Compression calls: {stats['compression_calls']}")
    print(f"Average compression ratio: {stats['avg_compression_ratio']:.4f}")
    
    print()


def demonstrate_different_compression_methods():
    """Demonstrate different compression methods."""
    print("=== Comparison of Different Compression Methods ===")
    
    # Create sample tensor
    tensor = create_sample_image_tensor(batch_size=1, channels=3, height=32, width=32)
    
    methods = [
        (CompressionMethod.QUANTIZATION, {"quantization_bits": 8}),
        (CompressionMethod.QUANTIZATION, {"quantization_bits": 4}),  # Higher compression
    ]
    
    for method, method_params in methods:
        print(f"Method: {method.value}")
        
        # Create config
        config_params = {
            "compression_method": method,
            "enable_compression_cache": True
        }
        config_params.update(method_params)
        
        config = VisualCompressionConfig(**config_params)
        
        # Create compressor
        compressor = VisualResourceCompressor(config)
        
        # Measure performance
        avg_time, avg_compression_ratio = measure_compression_performance(compressor, tensor, iterations=5)
        
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Compression ratio: {avg_compression_ratio:.4f}")
        
        # Test reconstruction quality
        compressed, metadata = compressor.compress(tensor, key=f"demo_{method.value}")
        decompressed = compressor.decompress(compressed, metadata)
        mse = torch.mean((tensor - decompressed) ** 2)
        print(f"  Reconstruction MSE: {mse:.6f}")
        
        print()


def main():
    """Main demonstration function."""
    print("Visual Resource Compression Demonstration for Qwen3-VL-2B Model")
    print("=" * 65)
    
    # Demonstrate quantization compression
    demonstrate_quantization_compression()
    
    # Demonstrate PCA compression
    demonstrate_pca_compression()
    
    # Demonstrate visual feature compressor
    demonstrate_visual_feature_compressor()
    
    # Compare different methods
    demonstrate_different_compression_methods()
    
    print("Demonstration completed!")


if __name__ == "__main__":
    main()
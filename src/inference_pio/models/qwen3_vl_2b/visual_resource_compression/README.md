# Visual Resource Compression for Qwen3-VL-2B Model

This module implements a comprehensive compression system specifically for visual resources in the Qwen3-VL-2B model. It includes various compression techniques for image data, feature maps, and visual encodings to optimize memory usage and processing speed.

## Features

- **Multiple Compression Methods**: Support for PCA, SVD, quantization, sparse coding, and autoencoders
- **Configurable Compression Ratios**: Adjustable compression ratios based on requirements
- **Caching Mechanism**: Efficient caching of compressed representations
- **Adaptive Compression**: Dynamic compression based on input characteristics
- **Integration**: Seamless integration with the Qwen3-VL-2B model architecture

## Compression Methods

### 1. Quantization
- Reduces precision of floating-point numbers
- Configurable bit depths (4-bit, 8-bit, etc.)
- Multiple quantization strategies (linear, logarithmic, k-means)

### 2. PCA (Principal Component Analysis)
- Dimensionality reduction technique
- Preserves most important features
- Configurable component ratios

### 3. SVD (Singular Value Decomposition)
- Matrix factorization approach
- Effective for 2D data compression
- Configurable rank ratios

### 4. Sparse Coding
- Represents data as sparse linear combinations
- Efficient for certain types of visual data
- Configurable sparsity levels

### 5. Autoencoder
- Neural network-based compression
- Learned representations
- Configurable architectures

## Configuration Options

The visual resource compression system is highly configurable through the `VisualCompressionConfig` class:

```python
from src.inference_pio.models.qwen3_vl_2b.visual_resource_compression import VisualCompressionConfig, CompressionMethod

config = VisualCompressionConfig(
    compression_method=CompressionMethod.QUANTIZATION,  # Choose compression method
    compression_ratio=0.5,  # Target compression ratio (0.0 to 1.0)
    quantization_bits=8,  # Number of bits for quantization
    quantization_method="linear",  # Quantization method
    enable_compression_cache=True,  # Enable caching of compressed representations
    compression_cache_size=1000,  # Maximum number of cached representations
    enable_adaptive_compression=True,  # Enable adaptive compression
    pca_components_ratio=0.7,  # Ratio of components to keep for PCA
    svd_rank_ratio=0.5,  # Ratio of rank to keep for SVD
    # ... additional parameters
)
```

## Integration with Qwen3-VL-2B Model

The visual compression system integrates seamlessly with the Qwen3-VL-2B model through the following configuration options in `Qwen3VL2BConfig`:

```python
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

config = Qwen3VL2BConfig()
config.enable_visual_resource_compression = True  # Enable visual compression
config.visual_compression_method = 'quantization'  # Compression method
config.visual_compression_ratio = 0.5  # Compression ratio
config.visual_quantization_bits = 8  # Quantization bits
```

## Usage Example

```python
from src.inference_pio.models.qwen3_vl_2b.visual_resource_compression import (
    VisualCompressionConfig, 
    VisualResourceCompressor, 
    CompressionMethod
)
import torch

# Create configuration
config = VisualCompressionConfig(
    compression_method=CompressionMethod.QUANTIZATION,
    quantization_bits=8,
    enable_compression_cache=True
)

# Create compressor
compressor = VisualResourceCompressor(config)

# Create sample visual data
image_tensor = torch.randn(1, 3, 224, 224)

# Compress
compressed, metadata = compressor.compress(image_tensor, key="sample_image")

# Decompress
decompressed = compressor.decompress(compressed, metadata)

# The decompressed tensor should have the same shape as the original
assert decompressed.shape == image_tensor.shape
```

## Performance Benefits

- **Memory Reduction**: Significant reduction in memory footprint for visual data
- **Processing Speed**: Faster processing of compressed visual features
- **Bandwidth Efficiency**: Reduced data transfer requirements
- **Scalability**: Better scalability for large-scale multimodal applications

## Testing

Run the tests to verify the implementation:

```bash
python -m pytest src/inference_pio/models/qwen3_vl_2b/visual_resource_compression/test_visual_compression.py
```

## License

This implementation is part of the Inference-PIO system and is licensed under the same terms as the main project.
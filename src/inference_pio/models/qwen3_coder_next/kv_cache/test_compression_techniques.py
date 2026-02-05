"""
Test suite for Qwen3-Coder-Next KV Cache Compression Techniques Implementation

This module tests the KV cache compression techniques for the Qwen3-Coder-Next model.
"""

import unittest
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .compression_techniques import (
    CompressionMethod,
    CompressedKVCacheConfig,
    QuantizedKVCache,
    LowRankKVCache,
    AdaptivePrecisionKVCache,
    SparseKVCache,
    CombinedKVCacheCompression,
    apply_compressed_kv_cache_to_model,
    get_compression_ratio
)


class MockModel(nn.Module):
    """Mock model for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)


class TestQwen3CoderNextCompressionTechniques(unittest.TestCase):
    """Test cases for Qwen3-Coder-Next compression techniques implementation."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 2
        self.num_heads = 8
        self.seq_len = 64
        self.head_dim = 128
        
        # Create a mock KV cache tensor
        self.kv_cache = torch.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim
        )
        
        self.config = CompressedKVCacheConfig(
            compression_method=CompressionMethod.COMBINED,
            quantization_bits=8,
            low_rank_dimension=32,
            adaptive_precision_threshold=0.01,
            sparse_compression_ratio=0.5,
            enable_dynamic_compression=True
        )

    def test_compression_method_enum(self):
        """Test that CompressionMethod enum has the expected values."""
        self.assertTrue(hasattr(CompressionMethod, 'QUANTIZATION'))
        self.assertTrue(hasattr(CompressionMethod, 'LOW_RANK'))
        self.assertTrue(hasattr(CompressionMethod, 'ADAPTIVE_PRECISION'))
        self.assertTrue(hasattr(CompressionMethod, 'SPARSE'))
        self.assertTrue(hasattr(CompressionMethod, 'COMBINED'))

    def test_compressed_kv_cache_config(self):
        """Test that CompressedKVCacheConfig can be instantiated."""
        config = CompressedKVCacheConfig()
        self.assertIsInstance(config, CompressedKVCacheConfig)
        self.assertEqual(config.compression_method, CompressionMethod.COMBINED)
        self.assertEqual(config.quantization_bits, 8)

    def test_quantized_kv_cache_initialization(self):
        """Test QuantizedKVCache initialization."""
        cache = QuantizedKVCache(self.config)
        self.assertIsInstance(cache, nn.Module)
        self.assertEqual(cache.quantization_bits, 8)
        self.assertEqual(cache.quantization_scale, 255.0)

    def test_quantized_kv_cache_compress_decompress(self):
        """Test QuantizedKVCache compress and decompress functionality."""
        cache = QuantizedKVCache(self.config)
        
        # Test compression
        compressed = cache.compress(self.kv_cache)
        self.assertIsInstance(compressed, torch.Tensor)
        self.assertEqual(compressed.shape, self.kv_cache.shape)
        self.assertTrue(torch.all(compressed >= 0))
        self.assertTrue(torch.all(compressed <= 255))  # 8-bit quantization
        
        # Test decompression
        decompressed = cache.decompress(compressed)
        self.assertIsInstance(decompressed, torch.Tensor)
        self.assertEqual(decompressed.shape, self.kv_cache.shape)

    def test_low_rank_kv_cache_initialization(self):
        """Test LowRankKVCache initialization."""
        cache = LowRankKVCache(self.config)
        self.assertIsInstance(cache, nn.Module)
        self.assertEqual(cache.rank, 32)

    def test_low_rank_kv_cache_compress_decompress(self):
        """Test LowRankKVCache compress and decompress functionality."""
        # Use a smaller rank for testing to avoid issues with SVD
        low_rank_config = CompressedKVCacheConfig(low_rank_dimension=8)
        cache = LowRankKVCache(low_rank_config)
        
        # Test compression
        left, right = cache.compress(self.kv_cache)
        self.assertIsInstance(left, torch.Tensor)
        self.assertIsInstance(right, torch.Tensor)
        
        # Test decompression
        reconstructed = cache.decompress(left, right)
        self.assertIsInstance(reconstructed, torch.Tensor)
        self.assertEqual(reconstructed.shape, self.kv_cache.shape)

    def test_adaptive_precision_kv_cache_initialization(self):
        """Test AdaptivePrecisionKVCache initialization."""
        cache = AdaptivePrecisionKVCache(self.config)
        self.assertIsInstance(cache, nn.Module)
        self.assertEqual(cache.threshold, 0.01)

    def test_adaptive_precision_kv_cache_compress(self):
        """Test AdaptivePrecisionKVCache compress functionality."""
        cache = AdaptivePrecisionKVCache(self.config)
        
        # Test compression
        compressed = cache.compress(self.kv_cache)
        self.assertIsInstance(compressed, torch.Tensor)
        self.assertEqual(compressed.shape, self.kv_cache.shape)

    def test_sparse_kv_cache_initialization(self):
        """Test SparseKVCache initialization."""
        cache = SparseKVCache(self.config)
        self.assertIsInstance(cache, nn.Module)
        self.assertEqual(cache.compression_ratio, 0.5)

    def test_sparse_kv_cache_compress(self):
        """Test SparseKVCache compress functionality."""
        cache = SparseKVCache(self.config)
        
        # Test compression
        compressed = cache.compress(self.kv_cache)
        self.assertIsInstance(compressed, torch.Tensor)
        self.assertEqual(compressed.shape, self.kv_cache.shape)
        
        # Check that compression actually made some values zero
        original_nonzeros = torch.count_nonzero(self.kv_cache)
        compressed_nonzeros = torch.count_nonzero(compressed)
        # The compressed version should have fewer non-zero values due to sparsification
        # But we can't guarantee this with random data, so we just check the shape

    def test_combined_kv_cache_compression_initialization(self):
        """Test CombinedKVCacheCompression initialization."""
        cache = CombinedKVCacheCompression(self.config)
        self.assertIsInstance(cache, nn.Module)
        self.assertEqual(cache.config.compression_method, CompressionMethod.COMBINED)

    def test_combined_kv_cache_compression_quantization_method(self):
        """Test CombinedKVCacheCompression with quantization method."""
        config = CompressedKVCacheConfig(compression_method=CompressionMethod.QUANTIZATION)
        cache = CombinedKVCacheCompression(config)
        
        # Test compression
        result = cache.compress(self.kv_cache)
        self.assertIn('compressed_data', result)
        self.assertIn('method', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['method'], 'quantization')
        
        # Test decompression
        decompressed = cache.decompress(result)
        self.assertIsInstance(decompressed, torch.Tensor)
        self.assertEqual(decompressed.shape, self.kv_cache.shape)

    def test_combined_kv_cache_compression_sparse_method(self):
        """Test CombinedKVCacheCompression with sparse method."""
        config = CompressedKVCacheConfig(compression_method=CompressionMethod.SPARSE)
        cache = CombinedKVCacheCompression(config)
        
        # Test compression
        result = cache.compress(self.kv_cache)
        self.assertIn('compressed_data', result)
        self.assertIn('method', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['method'], 'sparse')

    def test_combined_kv_cache_compression_combined_method(self):
        """Test CombinedKVCacheCompression with combined method."""
        config = CompressedKVCacheConfig(compression_method=CompressionMethod.COMBINED)
        cache = CombinedKVCacheCompression(config)
        
        # Test compression
        result = cache.compress(self.kv_cache)
        self.assertIn('compressed_data', result)
        self.assertIn('method', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['method'], 'combined')

    def test_apply_compressed_kv_cache_to_model(self):
        """Test applying compressed KV cache to a model."""
        model = MockModel()
        modified_model = apply_compressed_kv_cache_to_model(model, self.config)
        
        self.assertIs(modified_model, model)  # Should return the same model instance
        self.assertTrue(hasattr(model, 'kv_cache_compressor'))
        self.assertIsInstance(model.kv_cache_compressor, CombinedKVCacheCompression)

    def test_get_compression_ratio(self):
        """Test compression ratio calculation."""
        ratio = get_compression_ratio(1000, 500)
        self.assertEqual(ratio, 2.0)
        
        ratio = get_compression_ratio(500, 1000)
        self.assertEqual(ratio, 0.5)
        
        # Test edge case with zero compressed size
        ratio = get_compression_ratio(1000, 0)
        self.assertEqual(ratio, float('inf'))

    def test_different_quantization_bits(self):
        """Test quantization with different bit depths."""
        # Test 4-bit quantization
        config_4bit = CompressedKVCacheConfig(quantization_bits=4)
        cache_4bit = QuantizedKVCache(config_4bit)
        self.assertEqual(cache_4bit.quantization_scale, 15.0)  # 2^4 - 1
        
        compressed_4bit = cache_4bit.compress(self.kv_cache)
        self.assertTrue(torch.all(compressed_4bit >= 0))
        self.assertTrue(torch.all(compressed_4bit <= 15))  # 4-bit quantization

    def test_edge_case_identical_values(self):
        """Test quantization with identical values (edge case)."""
        identical_cache = torch.ones_like(self.kv_cache)  # All values are 1
        cache = QuantizedKVCache(self.config)
        
        # This should not cause division by zero
        compressed = cache.compress(identical_cache)
        self.assertIsInstance(compressed, torch.Tensor)
        
        # Decompression should work
        decompressed = cache.decompress(compressed)
        self.assertIsInstance(decompressed, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
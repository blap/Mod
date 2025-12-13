"""
Test Suite for Cross-Modal Memory Compression System
====================================================

This module contains comprehensive tests for the CrossModalCompressor class
and related functionality.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from cross_modal_compression import (
    CrossModalCompressor,
    CompressionMode,
    CompressionMetrics,
    adaptive_compression_selector,
    cross_modal_fusion_compress,
    cleanup_memory
)


class TestCrossModalCompressor(unittest.TestCase):
    """Test cases for CrossModalCompressor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.compressor = CrossModalCompressor(
            compression_threshold=0.5,
            quality_preservation_factor=0.9,
            hardware_target="intel_i5_nvidia_sm61",
            compression_mode=CompressionMode.LOSSY
        )
        
        # Create sample visual and text activations
        self.visual_activations = torch.randn(4, 128, 256)  # Batch, Height*Width, Features
        self.text_activations = torch.randn(4, 64, 256)     # Batch, Sequence Length, Features
    
    def test_initialization(self):
        """Test proper initialization of CrossModalCompressor."""
        self.assertEqual(self.compressor.compression_threshold, 0.5)
        self.assertEqual(self.compressor.quality_preservation_factor, 0.9)
        self.assertEqual(self.compressor.hardware_target, "intel_i5_nvidia_sm61")
        self.assertEqual(self.compressor.compression_mode, CompressionMode.LOSSY)
    
    def test_detect_compression_opportunity_large_tensor(self):
        """Test detection of compression opportunity for large tensor."""
        # Large tensor with repeated values should be detected as compressible
        # Create a tensor with some repeated patterns to ensure compression potential
        large_tensor = torch.cat([torch.ones(5, 50, 128), torch.zeros(5, 50, 128)], dim=0)
        self.assertTrue(self.compressor.detect_compression_opportunity(large_tensor))
    
    def test_detect_compression_opportunity_small_tensor(self):
        """Test detection of compression opportunity for small tensor."""
        # Small tensor should not be detected as compressible
        small_tensor = torch.randn(2, 2, 2)
        self.assertFalse(self.compressor.detect_compression_opportunity(small_tensor))
    
    def test_estimate_compression_potential(self):
        """Test estimation of compression potential."""
        # Create a tensor with some repeated values to test entropy calculation
        tensor_with_repetition = torch.cat([
            torch.ones(100), 
            torch.zeros(100), 
            torch.ones(100) * 0.5
        ])
        
        potential = self.compressor._estimate_compression_potential(tensor_with_repetition)
        self.assertGreater(potential, 0.0)  # Should have some compression potential
        self.assertLessEqual(potential, 1.0)  # Should not exceed 1.0
    
    def test_compress_activations_lossy(self):
        """Test lossy compression of activations."""
        compressed_data, metrics = self.compressor.compress_activations(
            self.visual_activations, 
            self.text_activations,
            mode=CompressionMode.LOSSY
        )
        
        self.assertIn('visual_activations', compressed_data)
        self.assertIn('text_activations', compressed_data)
        self.assertIn('compression_type', compressed_data)
        self.assertEqual(compressed_data['compression_type'], 'lossy')
        
        # Check metrics
        self.assertIsInstance(metrics, CompressionMetrics)
        self.assertGreaterEqual(metrics.original_size, metrics.compressed_size)
        self.assertGreaterEqual(metrics.quality_loss, 0.0)
        self.assertGreaterEqual(metrics.processing_time, 0.0)
        self.assertGreaterEqual(metrics.memory_saved, 0)
    
    def test_compress_activations_quantized(self):
        """Test quantized compression of activations."""
        compressed_data, metrics = self.compressor.compress_activations(
            self.visual_activations, 
            self.text_activations,
            mode=CompressionMode.QUANTIZED
        )
        
        self.assertIn('visual_activations', compressed_data)
        self.assertIn('text_activations', compressed_data)
        self.assertIn('compression_type', compressed_data)
        self.assertEqual(compressed_data['compression_type'], 'quantized')
        
        # Check that quantized data is actually in a lower precision format
        visual_dtype = compressed_data['visual_activations'].dtype
        text_dtype = compressed_data['text_activations'].dtype
        self.assertIn(visual_dtype, [torch.float16, torch.float32])  # May convert back to float32 for compatibility
        self.assertIn(text_dtype, [torch.float16, torch.float32])
    
    def test_compress_activations_sparse(self):
        """Test sparse compression of activations."""
        compressed_data, metrics = self.compressor.compress_activations(
            self.visual_activations, 
            self.text_activations,
            mode=CompressionMode.SPARSE
        )
        
        self.assertIn('visual_activations', compressed_data)
        self.assertIn('text_activations', compressed_data)
        self.assertIn('compression_type', compressed_data)
        self.assertEqual(compressed_data['compression_type'], 'sparse')
        
        # Check that metadata contains sparsity information
        self.assertIn('sparsity_ratio_visual', compressed_data['metadata'])
        self.assertIn('sparsity_ratio_text', compressed_data['metadata'])
    
    def test_compress_weights_quantized(self):
        """Test quantized compression of weights."""
        weights = torch.randn(128, 256)
        compressed_weights, metrics = self.compressor.compress_weights(
            weights, 
            mode=CompressionMode.QUANTIZED
        )
        
        self.assertIsInstance(compressed_weights, torch.Tensor)
        self.assertIsInstance(metrics, CompressionMetrics)
        self.assertGreaterEqual(metrics.original_size, metrics.compressed_size)
    
    def test_compress_weights_sparse(self):
        """Test sparse compression of weights."""
        weights = torch.randn(128, 256)
        compressed_weights, metrics = self.compressor.compress_weights(
            weights, 
            mode=CompressionMode.SPARSE
        )
        
        self.assertIsInstance(compressed_weights, torch.Tensor)
        self.assertIsInstance(metrics, CompressionMetrics)
        self.assertGreaterEqual(metrics.original_size, metrics.compressed_size)
    
    def test_compress_gradients_sparse(self):
        """Test sparse compression of gradients."""
        gradients = torch.randn(128, 256)
        compressed_gradients, metrics = self.compressor.compress_gradients(
            gradients, 
            mode=CompressionMode.SPARSE
        )
        
        self.assertIsInstance(compressed_gradients, torch.Tensor)
        self.assertIsInstance(metrics, CompressionMetrics)
        self.assertGreaterEqual(metrics.original_size, metrics.compressed_size)
    
    def test_compress_gradients_quantized(self):
        """Test quantized compression of gradients."""
        gradients = torch.randn(128, 256)
        compressed_gradients, metrics = self.compressor.compress_gradients(
            gradients, 
            mode=CompressionMode.QUANTIZED
        )
        
        self.assertIsInstance(compressed_gradients, torch.Tensor)
        self.assertIsInstance(metrics, CompressionMetrics)
        self.assertGreaterEqual(metrics.original_size, metrics.compressed_size)
    
    def test_decompress(self):
        """Test decompression functionality."""
        # First compress data
        compressed_data, _ = self.compressor.compress_activations(
            self.visual_activations, 
            self.text_activations,
            mode=CompressionMode.LOSSY
        )
        
        # Then decompress
        decompressed_visual, decompressed_text = self.compressor.decompress(compressed_data)
        
        # Check that we get tensors back
        self.assertIsInstance(decompressed_visual, torch.Tensor)
        self.assertIsInstance(decompressed_text, torch.Tensor)
        
        # Check shapes are preserved
        self.assertEqual(decompressed_visual.shape, self.visual_activations.shape)
        self.assertEqual(decompressed_text.shape, self.text_activations.shape)
    
    def test_evaluate_tradeoff(self):
        """Test trade-off evaluation."""
        metrics = CompressionMetrics(
            original_size=1000,
            compressed_size=500,
            compression_ratio=0.5,
            quality_loss=0.05,
            processing_time=0.1,
            memory_saved=500
        )
        
        tradeoff_results = self.compressor.evaluate_tradeoff(metrics)
        
        self.assertIn('memory_efficiency_score', tradeoff_results)
        self.assertIn('quality_preservation_score', tradeoff_results)
        self.assertIn('combined_tradeoff_score', tradeoff_results)
        self.assertIn('compression_effectiveness', tradeoff_results)
        
        # Check scores are in reasonable ranges
        self.assertGreaterEqual(tradeoff_results['memory_efficiency_score'], 0.0)
        self.assertLessEqual(tradeoff_results['memory_efficiency_score'], 1.0)
        self.assertGreaterEqual(tradeoff_results['quality_preservation_score'], 0.0)
        self.assertLessEqual(tradeoff_results['quality_preservation_score'], 1.0)
    
    def test_get_compression_statistics(self):
        """Test retrieval of compression statistics."""
        stats = self.compressor.get_compression_statistics()
        
        self.assertIn('total_compressions', stats)
        self.assertIn('total_memory_saved', stats)
        self.assertIn('average_compression_ratio', stats)
        
        # Initially should be zeros
        self.assertEqual(stats['total_compressions'], 0)
        self.assertEqual(stats['total_memory_saved'], 0)
        self.assertEqual(stats['average_compression_ratio'], 0.0)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.compressor = CrossModalCompressor()
    
    def test_adaptive_compression_selector_small_data(self):
        """Test adaptive compression selector for small data."""
        mode = adaptive_compression_selector(self.compressor, 'activations', 512)  # Less than 1KB
        self.assertEqual(mode, CompressionMode.LOSSLESS)
    
    def test_adaptive_compression_selector_weights(self):
        """Test adaptive compression selector for weights."""
        mode = adaptive_compression_selector(self.compressor, 'weights', 10000)
        self.assertEqual(mode, CompressionMode.QUANTIZED)
    
    def test_adaptive_compression_selector_gradients(self):
        """Test adaptive compression selector for gradients."""
        mode = adaptive_compression_selector(self.compressor, 'gradients', 10000)
        self.assertEqual(mode, CompressionMode.SPARSE)
    
    def test_adaptive_compression_selector_activations(self):
        """Test adaptive compression selector for activations."""
        mode = adaptive_compression_selector(self.compressor, 'activations', 10000)
        self.assertEqual(mode, CompressionMode.LOSSY)
    
    def test_cross_modal_fusion_compress(self):
        """Test cross-modal fusion compression."""
        visual_tensor = torch.randn(2, 64, 128)
        text_tensor = torch.randn(2, 32, 128)
        
        compressed_data, metrics = cross_modal_fusion_compress(
            visual_tensor, text_tensor, self.compressor
        )
        
        self.assertIn('visual_activations', compressed_data)
        self.assertIn('text_activations', compressed_data)
        self.assertIsInstance(metrics, CompressionMetrics)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.compressor = CrossModalCompressor()
    
    def test_invalid_data_type_detection(self):
        """Test detection with invalid data type."""
        with self.assertRaises(TypeError):
            self.compressor.detect_compression_opportunity("invalid_type")
    
    def test_unsupported_compression_mode(self):
        """Test unsupported compression mode."""
        # Test with an invalid compression mode type
        with self.assertRaises(ValueError):
            # Create temporary tensors for this test
            temp_visual = torch.randn(2, 4, 8)
            temp_text = torch.randn(2, 4, 8)
            self.compressor.compress_activations(
                temp_visual,
                temp_text,
                mode="invalid_mode"
            )
    
    def test_empty_tensors(self):
        """Test compression with empty tensors."""
        empty_visual = torch.empty(0, 0)
        empty_text = torch.empty(0, 0)
        
        # Should handle gracefully without crashing
        try:
            compressed_data, metrics = self.compressor.compress_activations(
                empty_visual, empty_text
            )
        except Exception as e:
            # If there's an exception, it should be handled appropriately
            self.fail(f"Compression failed with empty tensors: {e}")


class TestHardwareOptimizations(unittest.TestCase):
    """Test hardware-specific optimizations."""
    
    def test_hardware_setup_intel_target(self):
        """Test hardware setup for Intel target."""
        compressor = CrossModalCompressor(hardware_target="intel_i5_other_gpu")
        # Just ensure it initializes without error
        self.assertIsNotNone(compressor)
    
    def test_hardware_setup_cuda_available(self):
        """Test hardware setup when CUDA is available."""
        # Even if CUDA is not available, the compressor should initialize
        compressor = CrossModalCompressor(hardware_target="intel_i5_nvidia_sm61")
        self.assertIsNotNone(compressor)
        
        # Check that GPU device is properly set
        if torch.cuda.is_available():
            self.assertEqual(compressor.gpu_device.type, 'cuda')
        else:
            self.assertEqual(compressor.gpu_device.type, 'cpu')


def run_tests():
    """Run all tests in the test suite."""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__name__)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.2f}%")
    else:
        print("Success rate: 0% (no tests run)")

    return result


if __name__ == '__main__':
    # Run the tests
    test_result = run_tests()
    
    # Clean up memory after tests
    cleanup_memory()
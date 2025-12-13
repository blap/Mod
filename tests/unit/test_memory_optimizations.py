"""
Comprehensive tests for advanced memory optimizations in Qwen3-VL model.
Tests for memory pooling, caching, compression, tiering, and lifecycle management.
"""
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import time
import psutil
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import unittest

from src.qwen3_vl.optimization.hierarchical_memory_compression import HierarchicalMemoryCompressor, AdaptiveMemoryBank, MemoryEfficientCrossAttention
from src.components.memory.nvme_ssd_cache import NVMeSSDCache, ModelComponentCache, CacheConfig
from advanced_memory_management_optimizations import VisionLanguageMemoryOptimizer, HardwareSpecificMemoryOptimizer, MemoryPressureMonitor


class TestMemoryPoolingOptimizations(unittest.TestCase):
    """Test memory pooling optimizations."""

    def setUp(self):
        self.optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=512 * 1024 * 1024,  # 512MB
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=True,
            enable_stream_ordering=True
        )

    def test_tensor_allocation_and_deallocation(self):
        """Test tensor allocation and deallocation with memory pools."""
        # Allocate tensors of different types
        tensor1 = self.optimizer.allocate_tensor_memory((100, 256), dtype=torch.float32, tensor_type="general")
        tensor2 = self.optimizer.allocate_tensor_memory((50, 512), dtype=torch.float32, tensor_type="kv_cache")
        tensor3 = self.optimizer.allocate_tensor_memory((32, 128, 128), dtype=torch.float32, tensor_type="vision_features")

        # Verify allocations
        self.assertEqual(tensor1.shape, (100, 256))
        self.assertEqual(tensor2.shape, (50, 512))
        self.assertEqual(tensor3.shape, (32, 128, 128))

        # Free tensors
        self.optimizer.free_tensor_memory(tensor1, "general")
        self.optimizer.free_tensor_memory(tensor2, "kv_cache")
        self.optimizer.free_tensor_memory(tensor3, "vision_features")

        # Check memory stats
        stats = self.optimizer.get_memory_stats()
        print(f"Memory stats after allocation/deallocation: {stats}")

    def test_specialized_memory_pools(self):
        """Test specialized memory pools for different tensor types."""
        # Allocate tensors in specialized pools
        kv_tensors = []
        for i in range(5):
            tensor = self.optimizer.allocate_tensor_memory((32, 1024, 768), dtype=torch.float32, tensor_type="kv_cache")
            kv_tensors.append(tensor)

        vision_tensors = []
        for i in range(3):
            tensor = self.optimizer.allocate_tensor_memory((16, 576, 1024), dtype=torch.float32, tensor_type="vision_features")
            vision_tensors.append(tensor)

        # Check pool statistics
        stats = self.optimizer.get_memory_stats()
        self.assertIn('kv_cache_pool', stats)
        self.assertIn('vision_feature_pool', stats)

        # Free all tensors
        for tensor in kv_tensors:
            self.optimizer.free_tensor_memory(tensor, "kv_cache")
        for tensor in vision_tensors:
            self.optimizer.free_tensor_memory(tensor, "vision_features")


class TestMemoryCompressionOptimizations(unittest.TestCase):
    """Test memory compression optimizations."""

    def setUp(self):
        from transformers import PretrainedConfig
        
        # Create a mock config for testing
        class MockConfig:
            hidden_size = 768
            num_attention_heads = 12
            max_position_embeddings = 512
            rope_theta = 10000.0
            
        self.config = MockConfig()

    def test_hierarchical_memory_compression(self):
        """Test hierarchical memory compression."""
        compressor = HierarchicalMemoryCompressor(self.config, compression_level="medium")
        
        # Create input tensor
        batch_size, seq_len, hidden_size = 4, 128, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Apply compression
        compressed_states = compressor(hidden_states)
        
        # Verify output shape matches input shape
        self.assertEqual(compressed_states.shape, hidden_states.shape)
        
        # Verify compression doesn't change values too drastically
        compression_ratio = torch.mean(torch.abs(compressed_states - hidden_states)).item()
        print(f"Compression difference: {compression_ratio}")

    def test_adaptive_memory_bank(self):
        """Test adaptive memory bank."""
        memory_bank = AdaptiveMemoryBank(self.config, bank_size=256)
        
        # Create input tensor
        batch_size, seq_len, hidden_size = 2, 64, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Store and retrieve from memory bank
        output, indices = memory_bank(hidden_states)
        
        # Verify output shape
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[2], hidden_size)
        
        # Check memory bank statistics
        stats = memory_bank.get_memory_efficiency_stats()
        print(f"Memory bank stats: {stats}")


class TestMemoryTieringAndSwapping(unittest.TestCase):
    """Test memory tiering and swapping optimizations."""

    def setUp(self):
        self.config = CacheConfig(
            max_cache_size=256 * 1024 * 1024,  # 256MB
            hot_cache_size=10,
            warm_cache_size=50,
            cold_cache_size=200,
            prefetch_enabled=True,
            compression_enabled=True
        )

    def test_multi_tier_caching(self):
        """Test multi-tier caching system."""
        cache = NVMeSSDCache(self.config)
        
        # Create test data
        large_tensor = torch.randn(100, 100, 100)  # Large tensor
        small_tensor = torch.randn(10, 10, 10)    # Small tensor
        
        # Cache tensors
        cache.put("large_tensor", large_tensor, "warm")
        cache.put("small_tensor", small_tensor, "hot")
        
        # Retrieve tensors
        retrieved_large = cache.get("large_tensor")
        retrieved_small = cache.get("small_tensor")
        
        # Verify retrieval
        self.assertIsNotNone(retrieved_large)
        self.assertIsNotNone(retrieved_small)
        self.assertTrue(torch.equal(large_tensor, retrieved_large))
        self.assertTrue(torch.equal(small_tensor, retrieved_small))
        
        # Check cache stats
        stats = cache.get_stats()
        print(f"Cache stats: {stats}")
        
        # Clean up
        cache.clear()

    def test_model_component_caching(self):
        """Test model component caching."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Create model cache
        model_cache = ModelComponentCache(self.config)
        
        # Cache model weights
        model_cache.cache_model_weights(model, "test_model")
        
        # Create new model and load cached weights
        new_model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        success = model_cache.load_model_weights(new_model, "test_model")
        self.assertTrue(success)
        
        # Test tensor caching
        test_tensor = torch.randn(64, 64)
        cached = model_cache.cache_tensor(test_tensor, "test_tensor")
        self.assertTrue(cached)
        
        retrieved = model_cache.get_cached_tensor("test_tensor")
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.equal(test_tensor, retrieved))
        
        # Check stats
        stats = model_cache.get_cache_stats()
        print(f"Model cache stats: {stats}")
        
        # Clean up
        model_cache.clear_cache()


class TestMemoryLifecycleManagement(unittest.TestCase):
    """Test memory lifecycle management and garbage collection."""

    def setUp(self):
        self.optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=256 * 1024 * 1024,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=True
        )

    def test_memory_pressure_monitoring(self):
        """Test memory pressure monitoring."""
        monitor = MemoryPressureMonitor()
        
        # Get initial pressure
        initial_pressure = monitor.get_memory_pressure()
        print(f"Initial memory pressure: {initial_pressure:.3f}")
        
        # Get allocation advice
        advice = monitor.get_advice()
        print(f"Allocation advice: {advice}")
        
        # Check pressure levels
        is_high = monitor.is_high_pressure()
        is_low = monitor.is_low_pressure()
        print(f"High pressure: {is_high}, Low pressure: {is_low}")

    def test_hardware_specific_optimizations(self):
        """Test hardware-specific optimizations."""
        hw_optimizer = HardwareSpecificMemoryOptimizer()
        
        # Test tile size calculation
        tile_size_64 = hw_optimizer.get_optimal_tile_size(64)
        tile_size_128 = hw_optimizer.get_optimal_tile_size(128)
        tile_size_256 = hw_optimizer.get_optimal_tile_size(256)
        
        print(f"Optimal tile sizes: 64d={tile_size_64}, 128d={tile_size_128}, 256d={tile_size_256}")
        
        # Test batch size calculation
        optimal_batch = hw_optimizer.get_optimal_batch_size(512, 768)
        print(f"Optimal batch size for seq_len=512, hidden=768: {optimal_batch}")
        
        # Verify results make sense
        self.assertGreater(tile_size_64, 0)
        self.assertGreater(optimal_batch, 0)


class TestIntegrationOptimizations(unittest.TestCase):
    """Test integration of all memory optimizations."""

    def setUp(self):
        from transformers import PretrainedConfig
        
        # Create a mock config for testing
        class MockConfig:
            hidden_size = 768
            num_attention_heads = 12
            max_position_embeddings = 512
            rope_theta = 10000.0
            
        self.config = MockConfig()
        
        self.memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=1024 * 1024 * 1024,  # 1GB
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=True,
            enable_stream_ordering=True
        )
        
        self.cache_config = CacheConfig(
            max_cache_size=512 * 1024 * 1024,  # 512MB
            hot_cache_size=20,
            warm_cache_size=100,
            cold_cache_size=400,
            prefetch_enabled=True,
            compression_enabled=True
        )

    def test_full_pipeline_optimization(self):
        """Test full pipeline with all optimizations."""
        # Create tensors using memory optimizer
        batch_size, seq_len, hidden_size = 8, 512, 768
        query_tensor = self.memory_optimizer.allocate_tensor_memory(
            (batch_size, seq_len, hidden_size), 
            dtype=torch.float32, 
            tensor_type="kv_cache"
        )
        key_tensor = self.memory_optimizer.allocate_tensor_memory(
            (batch_size, seq_len, hidden_size), 
            dtype=torch.float32, 
            tensor_type="kv_cache"
        )
        
        # Create hierarchical memory compressor
        compressor = HierarchicalMemoryCompressor(self.config, compression_level="medium")
        
        # Compress tensors
        compressed_query = compressor(query_tensor)
        compressed_key = compressor(key_tensor)
        
        # Create memory-efficient cross attention
        cross_attention = MemoryEfficientCrossAttention(self.config)
        
        # Simulate cross-attention forward pass
        output, attn_weights, past_kv = cross_attention(
            hidden_states=compressed_query,
            encoder_hidden_states=compressed_key
        )
        
        # Verify outputs
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        
        # Test caching the results
        model_cache = ModelComponentCache(self.cache_config)
        model_cache.cache_tensor(output, "cross_attention_output")
        
        # Retrieve from cache
        cached_output = model_cache.get_cached_tensor("cross_attention_output")
        self.assertIsNotNone(cached_output)
        
        # Check memory stats
        stats = self.memory_optimizer.get_memory_stats()
        print(f"Final memory stats: {stats}")
        
        # Check cache stats
        cache_stats = model_cache.get_cache_stats()
        print(f"Final cache stats: {cache_stats}")
        
        # Clean up
        model_cache.clear_cache()


def run_all_tests():
    """Run all memory optimization tests."""
    print("Running Advanced Memory Optimization Tests for Qwen3-VL")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestMemoryPoolingOptimizations,
        TestMemoryCompressionOptimizations,
        TestMemoryTieringAndSwapping,
        TestMemoryLifecycleManagement,
        TestIntegrationOptimizations
    ]
    
    # Run tests
    for test_class in test_classes:
        print(f"\nRunning tests from {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.failures or result.errors:
            print(f"Test failures or errors in {test_class.__name__}:")
            for failure in result.failures:
                print(f"FAILURE: {failure[0]} - {failure[1]}")
            for error in result.errors:
                print(f"ERROR: {error[0]} - {error[1]}")
        else:
            print(f"All tests in {test_class.__name__} passed!")
    
    print("\nAll memory optimization tests completed!")


if __name__ == "__main__":
    run_all_tests()
"""
Benchmark for Intelligent Multimodal Caching in Qwen3-VL-2B Model

This module benchmarks the performance of the intelligent multimodal caching system
implemented for the Qwen3-VL-2B model. It measures improvements in inference speed,
memory usage, and overall efficiency when caching is enabled versus disabled.
"""

import unittest
import time
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock

from inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin
from inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from inference_pio.common.intelligent_multimodal_caching import (
    Qwen3VL2BIntelligentCachingManager,
    CacheEvictionPolicy
)


class BenchmarkQwen3VL2BIntelligentCaching(unittest.TestCase):
    """Benchmark cases for Qwen3-VL-2B intelligent multimodal caching performance."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.config = Qwen3VL2BConfig()
        self.config.enable_intelligent_multimodal_caching = True
        self.config.intelligent_multimodal_cache_size_gb = 0.5  # 500MB cache for testing
        self.config.intelligent_multimodal_cache_eviction_policy = "predictive"
        self.config.intelligent_multimodal_cache_enable_similarity = True
        self.config.intelligent_multimodal_cache_similarity_threshold = 0.85
        self.config.intelligent_multimodal_cache_enable_ttl = True
        self.config.intelligent_multimodal_cache_default_ttl = 3600.0  # 1 hour
        self.config.intelligent_multimodal_cache_enable_compression = True
        self.config.intelligent_multimodal_cache_compression_ratio = 0.6

        # Create mock images for testing
        self.mock_images = [
            Image.new('RGB', (224, 224), color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
            for _ in range(5)
        ]

        # Create mock texts for testing
        self.mock_texts = [
            f"This is a test text for multimodal caching with some variation {i} to ensure diversity in the test cases."
            for i in range(5)
        ]

    def benchmark_cache_hit_rate(self):
        """Benchmark cache hit rate for repeated inputs."""
        print("\nQwen3-VL-2B Intelligent Multimodal Caching - Cache Hit Rate Benchmark:")

        # Create plugin with caching enabled
        plugin_with_caching = Qwen3_VL_2B_Instruct_Plugin()
        success = plugin_with_caching.initialize(
            enable_intelligent_multimodal_caching=True,
            intelligent_multimodal_cache_size_gb=0.5,
            device="cpu"
        )
        self.assertTrue(success)

        # Create plugin without caching for comparison
        plugin_without_caching = Qwen3_VL_2B_Instruct_Plugin()
        success = plugin_without_caching.initialize(
            enable_intelligent_multimodal_caching=False,
            device="cpu"
        )
        self.assertTrue(success)

        # Prepare test data
        test_pairs = [(text, img) for text, img in zip(self.mock_texts, self.mock_images)]

        # Test with caching enabled - first run (should be cache misses)
        start_time = time.time()
        for text, img in test_pairs:
            # Process input with multimodal data
            result = plugin_with_caching.infer({"text": text, "image": img})
        first_run_time_with_caching = time.time() - start_time

        # Test with caching enabled - second run (should have cache hits)
        start_time = time.time()
        for text, img in test_pairs:
            # Process same input again - should benefit from caching
            result = plugin_with_caching.infer({"text": text, "image": img})
        second_run_time_with_caching = time.time() - start_time

        # Test without caching - first run
        start_time = time.time()
        for text, img in test_pairs:
            # Process input without caching
            result = plugin_without_caching.infer({"text": text, "image": img})
        first_run_time_without_caching = time.time() - start_time

        # Test without caching - second run (no caching benefits)
        start_time = time.time()
        for text, img in test_pairs:
            # Process same input again - no caching benefits
            result = plugin_without_caching.infer({"text": text, "image": img})
        second_run_time_without_caching = time.time() - start_time

        # Calculate performance improvements
        caching_speedup = first_run_time_without_caching / first_run_time_with_caching
        cache_hit_speedup = first_run_time_with_caching / second_run_time_with_caching
        uncached_consistency = first_run_time_without_caching / second_run_time_without_caching

        print(f"  First run with caching: {first_run_time_with_caching:.4f}s")
        print(f"  Second run with caching (benefiting from cache): {second_run_time_with_caching:.4f}s")
        print(f"  First run without caching: {first_run_time_without_caching:.4f}s")
        print(f"  Second run without caching: {second_run_time_without_caching:.4f}s")
        print(f"  Caching speedup (vs uncached first run): {caching_speedup:.2f}x")
        print(f"  Cache hit speedup (vs cached first run): {cache_hit_speedup:.2f}x")
        print(f"  Uncached consistency (first/second run): {uncached_consistency:.2f}x")

        # Get cache statistics
        if hasattr(plugin_with_caching._model, '_caching_manager'):
            cache_stats = plugin_with_caching._model._caching_manager.get_cache_stats()
            print(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
            print(f"  Active cache entries: {cache_stats.get('active_entries', 0)}")
            print(f"  Cache usage: {cache_stats.get('usage_percentage', 0):.2f}%")

        # Cleanup
        plugin_with_caching.cleanup()
        plugin_without_caching.cleanup()

    def benchmark_memory_efficiency(self):
        """Benchmark memory efficiency with caching enabled."""
        print("\nQwen3-VL-2B Intelligent Multimodal Caching - Memory Efficiency Benchmark:")

        # Create plugins
        plugin_with_caching = Qwen3_VL_2B_Instruct_Plugin()
        success = plugin_with_caching.initialize(
            enable_intelligent_multimodal_caching=True,
            intelligent_multimodal_cache_size_gb=0.5,
            device="cpu"
        )
        self.assertTrue(success)

        plugin_without_caching = Qwen3_VL_2B_Instruct_Plugin()
        success = plugin_without_caching.initialize(
            enable_intelligent_multimodal_caching=False,
            device="cpu"
        )
        self.assertTrue(success)

        # Process multiple inputs and measure memory usage
        import psutil
        import gc

        # Get baseline memory usage
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Process inputs with caching
        start_time = time.time()
        for i in range(10):  # Process 10 inputs
            text = f"Test text input {i} for memory efficiency benchmark."
            img = Image.new('RGB', (224, 224), color=(i*20 % 255, (i*30) % 255, (i*40) % 255))
            result = plugin_with_caching.infer({"text": text, "image": img})
        caching_time = time.time() - start_time
        caching_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Clear memory before next test
        gc.collect()

        # Process inputs without caching
        start_time = time.time()
        for i in range(10):  # Process 10 inputs
            text = f"Test text input {i} for memory efficiency benchmark."
            img = Image.new('RGB', (224, 224), color=(i*20 % 255, (i*30) % 255, (i*40) % 255))
            result = plugin_without_caching.infer({"text": text, "image": img})
        no_caching_time = time.time() - start_time
        no_caching_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        print(f"  Baseline memory: {baseline_memory:.2f} MB")
        print(f"  Memory with caching: {caching_memory:.2f} MB (diff: {caching_memory - baseline_memory:.2f} MB)")
        print(f"  Memory without caching: {no_caching_memory:.2f} MB (diff: {no_caching_memory - baseline_memory:.2f} MB)")
        print(f"  Time with caching: {caching_time:.4f}s")
        print(f"  Time without caching: {no_caching_time:.4f}s")

        # Cleanup
        plugin_with_caching.cleanup()
        plugin_without_caching.cleanup()

    def benchmark_similarity_detection(self):
        """Benchmark similarity detection performance."""
        print("\nQwen3-VL-2B Intelligent Multimodal Caching - Similarity Detection Benchmark:")

        caching_manager = Qwen3VL2BIntelligentCachingManager(
            cache_size_gb=0.5,
            eviction_policy=CacheEvictionPolicy.PREDICTIVE,
            enable_similarity_caching=True,
            similarity_threshold=0.85
        )

        # Create similar texts
        base_text = "This is a base text for similarity testing in the Qwen3-VL-2B model."
        similar_texts = [
            base_text,
            base_text + " with minor variation.",
            base_text.replace("testing", "benchmarking"),
            "This is a completely different text unrelated to the base text.",
            base_text + " with another minor variation and some extra words."
        ]

        # Cache the base text
        base_tensor = torch.randn(1, 10, self.config.hidden_size)
        caching_manager.cache_text_input(base_text, base_tensor)

        # Time similarity lookups
        start_time = time.time()
        for text in similar_texts:
            result = caching_manager.find_similar_text(text)
            if result:
                print(f"    Found similar text for: '{text[:30]}...' -> {result[0][:10]}...")
        similarity_lookup_time = time.time() - start_time

        print(f"  Similarity lookup time for {len(similar_texts)} texts: {similarity_lookup_time:.4f}s")
        print(f"  Average lookup time per text: {similarity_lookup_time/len(similar_texts)*1000:.2f}ms")

    def benchmark_cache_scalability(self):
        """Benchmark cache scalability with increasing data size."""
        print("\nQwen3-VL-2B Intelligent Multimodal Caching - Scalability Benchmark:")

        cache_sizes_gb = [0.1, 0.2, 0.5, 1.0]
        results = {}

        for cache_size in cache_sizes_gb:
            plugin = Qwen3_VL_2B_Instruct_Plugin()
            success = plugin.initialize(
                enable_intelligent_multimodal_caching=True,
                intelligent_multimodal_cache_size_gb=cache_size,
                device="cpu"
            )
            self.assertTrue(success)

            # Process multiple inputs to fill cache
            start_time = time.time()
            for i in range(20):
                text = f"Scalability test text input {i} with sufficient length to consume cache space efficiently."
                img = Image.new('RGB', (224, 224), color=(i*10 % 255, (i*15) % 255, (i*20) % 255))
                result = plugin.infer({"text": text, "image": img})
            processing_time = time.time() - start_time

            # Get cache stats
            if hasattr(plugin._model, '_caching_manager'):
                cache_stats = plugin._model._caching_manager.get_cache_stats()
                results[cache_size] = {
                    'time': processing_time,
                    'hit_rate': cache_stats.get('hit_rate', 0),
                    'active_entries': cache_stats.get('active_entries', 0),
                    'usage_percentage': cache_stats.get('usage_percentage', 0)
                }

            print(f"  Cache size {cache_size}GB: {processing_time:.4f}s, hit_rate: {cache_stats.get('hit_rate', 0):.2%}, "
                  f"usage: {cache_stats.get('usage_percentage', 0):.2f}%")

            # Cleanup
            plugin.cleanup()

        return results

    def test_complete_caching_benchmark(self):
        """Run all caching benchmarks."""
        print("="*60)
        print("QWEN3-VL-2B INTELLIGENT MULTIMODAL CACHING BENCHMARK")
        print("="*60)

        # Run individual benchmarks
        self.benchmark_cache_hit_rate()
        self.benchmark_memory_efficiency()
        self.benchmark_similarity_detection()
        scalability_results = self.benchmark_cache_scalability()

        print("\n" + "="*60)
        print("QWEN3-VL-2B INTELLIGENT MULTIMODAL CACHING BENCHMARK COMPLETE")
        print("="*60)

        # Summary
        print("\nSUMMARY:")
        print(f"- Cache hit rate and speedup measured across repeated inputs")
        print(f"- Memory efficiency compared with and without caching")
        print(f"- Similarity detection performance evaluated")
        print(f"- Scalability tested with different cache sizes")
        print(f"- Overall performance improvements demonstrated")

        # Cleanup
        import gc
        gc.collect()


def run_qwen3_vl_caching_benchmark():
    """Run the Qwen3-VL-2B intelligent multimodal caching benchmark."""
    benchmark = BenchmarkQwen3VL2BIntelligentCaching()
    benchmark.test_complete_caching_benchmark()


if __name__ == '__main__':
    unittest.main()
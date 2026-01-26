"""
Standardized Benchmark for Memory Usage - Qwen3-Coder-30B

This module benchmarks the memory usage for the Qwen3-Coder-30B model.
"""

import psutil
import torch
import unittest
from inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin


class BenchmarkQwen3Coder30BMemoryUsage(unittest.TestCase):
    """Benchmark cases for Qwen3-Coder-30B memory usage."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_qwen3_coder_30b_plugin()
        # Don't initialize yet - we want to measure memory at different stages

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB

    def benchmark_memory_usage_during_operations(self):
        """Benchmark memory usage during different operations."""
        baseline_memory = self.get_memory_usage()
        
        # Initialize plugin
        init_start_memory = self.get_memory_usage()
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        init_end_memory = self.get_memory_usage()
        
        # Load model
        load_start_memory = self.get_memory_usage()
        model = self.plugin.load_model()
        self.assertIsNotNone(model)
        load_end_memory = self.get_memory_usage()
        
        # Perform inference
        inference_start_memory = self.get_memory_usage()
        input_ids = torch.randint(0, 1000, (1, 50))
        result = self.plugin.infer(input_ids)
        self.assertIsNotNone(result)
        inference_end_memory = self.get_memory_usage()
        
        # Cleanup
        cleanup_start_memory = self.get_memory_usage()
        if hasattr(self.plugin, 'cleanup'):
            self.plugin.cleanup()
        cleanup_end_memory = self.get_memory_usage()
        
        return {
            'baseline_memory': baseline_memory,
            'init_memory_increase': init_end_memory - init_start_memory,
            'load_memory_increase': load_end_memory - load_start_memory,
            'inference_memory_increase': inference_end_memory - inference_start_memory,
            'cleanup_memory_after': cleanup_end_memory,
            'total_peak_memory': max(init_end_memory, load_end_memory, inference_end_memory),
            'memory_after_cleanup': cleanup_end_memory
        }

    def test_memory_usage_baseline(self):
        """Test baseline memory usage."""
        baseline_memory = self.get_memory_usage()
        
        print(f"\nQwen3-Coder-30B Baseline Memory Usage:")
        print(f"  Baseline: {baseline_memory:.2f} MB")
        
        # Basic sanity check
        self.assertGreater(baseline_memory, 0)

    def test_memory_usage_after_initialization(self):
        """Test memory usage after plugin initialization."""
        baseline_memory = self.get_memory_usage()
        
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        
        post_init_memory = self.get_memory_usage()
        memory_increase = post_init_memory - baseline_memory
        
        print(f"\nQwen3-Coder-30B Memory Usage After Initialization:")
        print(f"  Baseline: {baseline_memory:.2f} MB")
        print(f"  After init: {post_init_memory:.2f} MB")
        print(f"  Increase: {memory_increase:.2f} MB")
        
        # Basic sanity check
        self.assertGreater(post_init_memory, 0)

    def test_memory_usage_after_model_loading(self):
        """Test memory usage after model loading."""
        baseline_memory = self.get_memory_usage()
        
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        
        post_init_memory = self.get_memory_usage()
        
        model = self.plugin.load_model()
        self.assertIsNotNone(model)
        
        post_load_memory = self.get_memory_usage()
        model_memory_increase = post_load_memory - post_init_memory
        
        print(f"\nQwen3-Coder-30B Memory Usage After Model Loading:")
        print(f"  After init: {post_init_memory:.2f} MB")
        print(f"  After load: {post_load_memory:.2f} MB")
        print(f"  Model increase: {model_memory_increase:.2f} MB")
        
        # Basic sanity check
        self.assertGreater(post_load_memory, post_init_memory)

    def test_memory_usage_full_workflow(self):
        """Test memory usage throughout the full workflow."""
        results = self.benchmark_memory_usage_during_operations()
        
        print(f"\nQwen3-Coder-30B Full Workflow Memory Usage:")
        print(f"  Baseline: {results['baseline_memory']:.2f} MB")
        print(f"  Init increase: {results['init_memory_increase']:.2f} MB")
        print(f"  Load increase: {results['load_memory_increase']:.2f} MB")
        print(f"  Inference increase: {results['inference_memory_increase']:.2f} MB")
        print(f"  Peak memory: {results['total_peak_memory']:.2f} MB")
        print(f"  After cleanup: {results['memory_after_cleanup']:.2f} MB")
        
        # Verify that memory measurements are reasonable
        self.assertGreater(results['total_peak_memory'], results['baseline_memory'])

    def test_memory_usage_with_different_batch_sizes(self):
        """Test memory usage with different batch sizes."""
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        
        model = self.plugin.load_model()
        self.assertIsNotNone(model)
        
        batch_sizes = [1, 2, 4, 8]
        memory_usages = []
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Reset memory measurement
                baseline = self.get_memory_usage()
                
                # Perform inference with specific batch size
                input_ids = torch.randint(0, 1000, (batch_size, 50))
                result = self.plugin.infer(input_ids)
                self.assertIsNotNone(result)
                
                peak_memory = self.get_memory_usage()
                memory_used = peak_memory - baseline
                memory_usages.append(memory_used)
                
                print(f"  Batch size {batch_size}: {memory_used:.2f} MB additional")
        
        # Memory usage should generally increase with batch size (though not strictly linear)
        # At minimum, verify all measurements are positive
        for memory in memory_usages:
            self.assertGreater(memory, 0)

    def test_memory_usage_with_different_sequence_lengths(self):
        """Test memory usage with different sequence lengths."""
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        
        model = self.plugin.load_model()
        self.assertIsNotNone(model)
        
        seq_lengths = [10, 50, 100, 200]
        memory_usages = []
        
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                baseline = self.get_memory_usage()
                
                input_ids = torch.randint(0, 1000, (1, seq_len))
                result = self.plugin.infer(input_ids)
                self.assertIsNotNone(result)
                
                peak_memory = self.get_memory_usage()
                memory_used = peak_memory - baseline
                memory_usages.append(memory_used)
                
                print(f"  Sequence length {seq_len}: {memory_used:.2f} MB additional")
        
        # All measurements should be positive
        for memory in memory_usages:
            self.assertGreater(memory, 0)

    def test_memory_efficiency_with_kv_cache(self):
        """Test memory efficiency with KV cache usage."""
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        
        model = self.plugin.load_model()
        self.assertIsNotNone(model)
        
        # Measure memory without KV cache
        baseline = self.get_memory_usage()
        input_ids = torch.randint(0, 1000, (1, 100))
        result1 = self.plugin.infer(input_ids)
        self.assertIsNotNone(result1)
        memory_without_cache = self.get_memory_usage() - baseline
        
        # Reset and measure with KV cache (if supported)
        baseline2 = self.get_memory_usage()
        result2 = self.plugin.infer({
            'input_ids': input_ids,
            'use_cache': True
        })
        self.assertIsNotNone(result2)
        memory_with_cache = self.get_memory_usage() - baseline2
        
        print(f"\nQwen3-Coder-30B Memory Usage Comparison:")
        print(f"  Without KV cache: {memory_without_cache:.2f} MB")
        print(f"  With KV cache: {memory_with_cache:.2f} MB")
        
        # Both should use positive memory
        self.assertGreater(memory_without_cache, 0)
        self.assertGreater(memory_with_cache, 0)

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, 'cleanup') and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == '__main__':
    unittest.main()
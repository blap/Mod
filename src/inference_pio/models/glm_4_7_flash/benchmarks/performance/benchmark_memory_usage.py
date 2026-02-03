"""
Real Performance Benchmark for Memory Usage - GLM-4.7

This module benchmarks the memory usage for the GLM-4.7 model using real performance measurements.
"""

import unittest

import torch

from inference_pio.models.glm_4_7.plugin import create_glm_4_7_plugin
from benchmarks.core.real_performance_monitor import get_real_system_metrics


class BenchmarkGLM47MemoryUsage(unittest.TestCase):
    """Benchmark cases for GLM-4.7 memory usage using real performance measurements."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_glm_4_7_plugin()
        # Don't initialize yet - we want to measure memory at different stages

    def benchmark_memory_usage_during_operations(self):
        """Benchmark memory usage during different operations using real metrics."""
        baseline_metrics = get_real_system_metrics()

        # Initialize plugin
        init_start_metrics = get_real_system_metrics()
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        init_end_metrics = get_real_system_metrics()

        # Load model
        load_start_metrics = get_real_system_metrics()
        model = self.plugin.load_model()
        self.assertIsNotNone(model)
        load_end_metrics = get_real_system_metrics()

        # Perform inference
        inference_start_metrics = get_real_system_metrics()
        input_ids = torch.randint(0, 1000, (1, 50))
        result = self.plugin.infer(input_ids)
        self.assertIsNotNone(result)
        inference_end_metrics = get_real_system_metrics()

        # Cleanup
        cleanup_start_metrics = get_real_system_metrics()
        if hasattr(self.plugin, "cleanup"):
            self.plugin.cleanup()
        cleanup_end_metrics = get_real_system_metrics()

        return {
            "baseline_memory_mb": baseline_metrics.memory_used_mb,
            "init_memory_increase_mb": init_end_metrics.memory_used_mb
            - init_start_metrics.memory_used_mb,
            "load_memory_increase_mb": load_end_metrics.memory_used_mb
            - load_start_metrics.memory_used_mb,
            "inference_memory_increase_mb": inference_end_metrics.memory_used_mb
            - inference_start_metrics.memory_used_mb,
            "cleanup_memory_after_mb": cleanup_end_metrics.memory_used_mb,
            "total_peak_memory_mb": max(
                init_end_metrics.memory_used_mb,
                load_end_metrics.memory_used_mb,
                inference_end_metrics.memory_used_mb,
            ),
            "memory_after_cleanup_mb": cleanup_end_metrics.memory_used_mb,
            "baseline_cpu_percent": baseline_metrics.cpu_percent,
            "peak_cpu_percent": max(
                init_end_metrics.cpu_percent,
                load_end_metrics.cpu_percent,
                inference_end_metrics.cpu_percent,
            ),
            "baseline_memory_percent": baseline_metrics.memory_percent,
            "peak_memory_percent": max(
                init_end_metrics.memory_percent,
                load_end_metrics.memory_percent,
                inference_end_metrics.memory_percent,
            ),
            "gpu_memory_used_mb": (
                inference_end_metrics.gpu_memory_used_mb
                if inference_end_metrics.gpu_memory_used_mb is not None
                else 0
            ),
        }

    def test_memory_usage_baseline(self):
        """Test baseline memory usage using real metrics."""
        baseline_metrics = get_real_system_metrics()

        print(f"\nGLM-4.7 Baseline Memory Usage:")
        print(f"  Memory used: {baseline_metrics.memory_used_mb:.2f} MB")
        print(f"  Memory percent: {baseline_metrics.memory_percent:.2f}%")
        print(f"  CPU percent: {baseline_metrics.cpu_percent:.2f}%")

        # Basic sanity check
        self.assertGreater(baseline_metrics.memory_used_mb, 0)

    def test_memory_usage_after_initialization(self):
        """Test memory usage after plugin initialization using real metrics."""
        baseline_metrics = get_real_system_metrics()

        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)

        post_init_metrics = get_real_system_metrics()
        memory_increase = (
            post_init_metrics.memory_used_mb - baseline_metrics.memory_used_mb
        )

        print(f"\nGLM-4.7 Memory Usage After Initialization:")
        print(f"  Memory used: {post_init_metrics.memory_used_mb:.2f} MB")
        print(f"  Memory percent: {post_init_metrics.memory_percent:.2f}%")
        print(f"  CPU percent: {post_init_metrics.cpu_percent:.2f}%")
        print(f"  Memory increase: {memory_increase:.2f} MB")

        # Basic sanity check
        self.assertGreater(post_init_metrics.memory_used_mb, 0)

    def test_memory_usage_after_model_loading(self):
        """Test memory usage after model loading using real metrics."""
        baseline_metrics = get_real_system_metrics()

        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)

        post_init_metrics = get_real_system_metrics()

        model = self.plugin.load_model()
        self.assertIsNotNone(model)

        post_load_metrics = get_real_system_metrics()
        model_memory_increase = (
            post_load_metrics.memory_used_mb - post_init_metrics.memory_used_mb
        )

        print(f"\nGLM-4.7 Memory Usage After Model Loading:")
        print(
            f"  After init: {post_init_metrics.memory_used_mb:.2f} MB ({post_init_metrics.memory_percent:.2f}%)"
        )
        print(
            f"  After load: {post_load_metrics.memory_used_mb:.2f} MB ({post_load_metrics.memory_percent:.2f}%)"
        )
        print(f"  Model increase: {model_memory_increase:.2f} MB")

        # Basic sanity check
        self.assertGreater(
            post_load_metrics.memory_used_mb, post_init_metrics.memory_used_mb
        )

    def test_memory_usage_full_workflow(self):
        """Test memory usage throughout the full workflow using real metrics."""
        results = self.benchmark_memory_usage_during_operations()

        print(f"\nGLM-4.7 Full Workflow Memory Usage:")
        print(f"  Baseline: {results['baseline_memory_mb']:.2f} MB")
        print(f"  Init increase: {results['init_memory_increase_mb']:.2f} MB")
        print(f"  Load increase: {results['load_memory_increase_mb']:.2f} MB")
        print(f"  Inference increase: {results['inference_memory_increase_mb']:.2f} MB")
        print(f"  Peak memory: {results['total_peak_memory_mb']:.2f} MB")
        print(f"  After cleanup: {results['memory_after_cleanup_mb']:.2f} MB")
        print(f"  Peak CPU usage: {results['peak_cpu_percent']:.2f}%")
        print(f"  Peak memory usage: {results['peak_memory_percent']:.2f}%")
        if results["gpu_memory_used_mb"] > 0:
            print(f"  GPU memory used: {results['gpu_memory_used_mb']:.2f} MB")

        # Verify that memory measurements are reasonable
        self.assertGreater(
            results["total_peak_memory_mb"], results["baseline_memory_mb"]
        )

    def test_memory_usage_with_different_batch_sizes(self):
        """Test memory usage with different batch sizes using real metrics."""
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)

        model = self.plugin.load_model()
        self.assertIsNotNone(model)

        batch_sizes = [1, 2, 4, 8]
        memory_increases = []

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Reset memory measurement
                baseline_metrics = get_real_system_metrics()

                # Perform inference with specific batch size
                input_ids = torch.randint(0, 1000, (batch_size, 50))
                result = self.plugin.infer(input_ids)
                self.assertIsNotNone(result)

                peak_metrics = get_real_system_metrics()
                memory_increase = (
                    peak_metrics.memory_used_mb - baseline_metrics.memory_used_mb
                )
                memory_increases.append(memory_increase)

                print(f"  Batch size {batch_size}: {memory_increase:.2f} MB additional")

        # Memory usage should generally increase with batch size (though not strictly linear)
        # At minimum, verify all measurements are positive
        for memory in memory_increases:
            self.assertGreater(memory, 0)

    def test_memory_usage_with_different_sequence_lengths(self):
        """Test memory usage with different sequence lengths using real metrics."""
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)

        model = self.plugin.load_model()
        self.assertIsNotNone(model)

        seq_lengths = [10, 50, 100, 200]
        memory_increases = []

        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                baseline_metrics = get_real_system_metrics()

                input_ids = torch.randint(0, 1000, (1, seq_len))
                result = self.plugin.infer(input_ids)
                self.assertIsNotNone(result)

                peak_metrics = get_real_system_metrics()
                memory_increase = (
                    peak_metrics.memory_used_mb - baseline_metrics.memory_used_mb
                )
                memory_increases.append(memory_increase)

                print(
                    f"  Sequence length {seq_len}: {memory_increase:.2f} MB additional"
                )

        # All measurements should be positive
        for memory in memory_increases:
            self.assertGreater(memory, 0)

    def test_memory_efficiency_with_kv_cache(self):
        """Test memory efficiency with KV cache usage using real metrics."""
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)

        model = self.plugin.load_model()
        self.assertIsNotNone(model)

        # Measure memory without KV cache
        baseline_metrics = get_real_system_metrics()
        input_ids = torch.randint(0, 1000, (1, 100))
        result1 = self.plugin.infer(input_ids)
        self.assertIsNotNone(result1)
        memory_without_cache = (
            get_real_system_metrics().memory_used_mb - baseline_metrics.memory_used_mb
        )

        # Reset and measure with KV cache (if supported)
        baseline2_metrics = get_real_system_metrics()
        result2 = self.plugin.infer({"input_ids": input_ids, "use_cache": True})
        self.assertIsNotNone(result2)
        memory_with_cache = (
            get_real_system_metrics().memory_used_mb - baseline2_metrics.memory_used_mb
        )

        print(f"\nGLM-4.7 Memory Usage Comparison:")
        print(f"  Without KV cache: {memory_without_cache:.2f} MB")
        print(f"  With KV cache: {memory_with_cache:.2f} MB")

        # Both should use positive memory
        self.assertGreater(memory_without_cache, 0)
        self.assertGreater(memory_with_cache, 0)

    def test_detailed_system_metrics(self):
        """Test detailed system metrics collection during operations."""
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)

        model = self.plugin.load_model()
        self.assertIsNotNone(model)

        # Get detailed system metrics during operation
        metrics = get_real_system_metrics()

        print(f"\nGLM-4.7 Detailed System Metrics:")
        print(f"  Memory Used: {metrics.memory_used_mb:.2f} MB")
        print(f"  Memory Available: {metrics.memory_available_mb:.2f} MB")
        print(f"  Memory Percent: {metrics.memory_percent:.2f}%")
        print(f"  CPU Percent: {metrics.cpu_percent:.2f}%")

        if metrics.gpu_percent is not None:
            print(f"  GPU Percent: {metrics.gpu_percent:.2f}%")
            print(f"  GPU Memory Used: {metrics.gpu_memory_used_mb} MB")
            print(f"  GPU Memory Total: {metrics.gpu_memory_total_mb} MB")

        if metrics.memory_allocated_mb is not None:
            print(f"  CUDA Memory Allocated: {metrics.memory_allocated_mb:.2f} MB")
            print(f"  CUDA Memory Reserved: {metrics.memory_reserved_mb:.2f} MB")

        # Verify that all metrics are properly collected
        self.assertIsNotNone(metrics.timestamp)
        self.assertGreaterEqual(metrics.cpu_percent, 0)
        self.assertGreaterEqual(metrics.memory_used_mb, 0)
        self.assertGreaterEqual(metrics.memory_percent, 0)
        self.assertGreaterEqual(metrics.memory_available_mb, 0)

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()

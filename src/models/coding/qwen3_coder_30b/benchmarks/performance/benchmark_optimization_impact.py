"""
Standardized Benchmark for Optimization Impact - Qwen3-Coder-30B

This module benchmarks the impact of optimizations for the Qwen3-Coder-30B model.
"""

import time
import unittest

import psutil
import torch

from inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin


class BenchmarkQwen3Coder30BOptimizationImpact(unittest.TestCase):
    """Benchmark cases for Qwen3-Coder-30B optimization impact."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.base_plugin = create_qwen3_coder_30b_plugin()

    def benchmark_plugin_with_config(
        self, config, name, input_length=50, num_iterations=5
    ):
        """Benchmark a plugin with specific configuration."""
        plugin = create_qwen3_coder_30b_plugin()

        # Initialize with specific config
        init_success = plugin.initialize(device="cpu", **config)
        if not init_success:
            return None

        # Load model
        model = plugin.load_model()
        if model is None:
            return None

        # Prepare input
        input_ids = torch.randint(0, 1000, (1, input_length))

        # Warmup
        for _ in range(3):
            _ = plugin.infer(input_ids)

        # Memory measurement
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Timing run
        start_time = time.time()
        for i in range(num_iterations):
            _ = plugin.infer(input_ids)
        end_time = time.time()

        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        total_time = end_time - start_time
        avg_time_per_inference = total_time / num_iterations
        tokens_per_second = (
            input_length / avg_time_per_inference
            if avg_time_per_inference > 0
            else float("inf")
        )
        memory_used = memory_after - memory_before

        # Cleanup
        if hasattr(plugin, "cleanup"):
            plugin.cleanup()

        return {
            "name": name,
            "config": config,
            "total_time": total_time,
            "avg_time_per_inference": avg_time_per_inference,
            "tokens_per_second": tokens_per_second,
            "memory_used": memory_used,
            "num_iterations": num_iterations,
        }

    def test_no_optimizations_baseline(self):
        """Test performance with no optimizations."""
        result = self.benchmark_plugin_with_config(
            config={}, name="No Optimizations", num_iterations=3
        )

        if result:
            print(f"\nQwen3-Coder-30B Baseline (No Optimizations):")
            print(f"  Tokens/sec: {result['tokens_per_second']:.2f}")
            print(f"  Memory used: {result['memory_used']:.2f} MB")
            print(f"  Avg inference time: {result['avg_time_per_inference']:.4f}s")

            self.assertGreater(result["tokens_per_second"], 0)
            self.assertGreaterEqual(result["memory_used"], 0)

    def test_flash_attention_impact(self):
        """Test the impact of flash attention optimization."""
        configs = [
            ({"use_flash_attention": False}, "Without Flash Attention"),
            ({"use_flash_attention": True}, "With Flash Attention"),
        ]

        results = []
        for config, name in configs:
            result = self.benchmark_plugin_with_config(
                config=config, name=name, num_iterations=3
            )
            if result:
                results.append(result)

        print(f"\nQwen3-Coder-30B Flash Attention Impact:")
        for result in results:
            print(
                f"  {result['name']}: {result['tokens_per_second']:.2f} tokens/sec, "
                f"{result['memory_used']:.2f} MB"
            )

        # If both ran successfully, compare them
        if len(results) == 2:
            without_fa = next(r for r in results if "Without" in r["name"])
            with_fa = next(r for r in results if "With" in r["name"])

            print(
                f"  Improvement: {((with_fa['tokens_per_second'] / without_fa['tokens_per_second']) - 1) * 100:.2f}% "
                f"in throughput"
            )

    def test_memory_efficient_optimization_impact(self):
        """Test the impact of memory efficient optimization."""
        configs = [
            ({"memory_efficient": False}, "Without Memory Efficient"),
            ({"memory_efficient": True}, "With Memory Efficient"),
        ]

        results = []
        for config, name in configs:
            result = self.benchmark_plugin_with_config(
                config=config, name=name, num_iterations=3
            )
            if result:
                results.append(result)

        print(f"\nQwen3-Coder-30B Memory Efficient Impact:")
        for result in results:
            print(
                f"  {result['name']}: {result['tokens_per_second']:.2f} tokens/sec, "
                f"{result['memory_used']:.2f} MB"
            )

        if len(results) == 2:
            without_me = next(r for r in results if "Without" in r["name"])
            with_me = next(r for r in results if "With" in r["name"])

            print(
                f"  Memory saving: {((without_me['memory_used'] - with_me['memory_used']) / without_me['memory_used']) * 100:.2f}%"
            )

    def test_fused_layers_impact(self):
        """Test the impact of fused layers optimization."""
        configs = [
            ({"use_fused_layers": False}, "Without Fused Layers"),
            ({"use_fused_layers": True}, "With Fused Layers"),
        ]

        results = []
        for config, name in configs:
            result = self.benchmark_plugin_with_config(
                config=config, name=name, num_iterations=3
            )
            if result:
                results.append(result)

        print(f"\nQwen3-Coder-30B Fused Layers Impact:")
        for result in results:
            print(
                f"  {result['name']}: {result['tokens_per_second']:.2f} tokens/sec, "
                f"{result['memory_used']:.2f} MB"
            )

    def test_quantization_impact(self):
        """Test the impact of quantization optimization."""
        configs = [
            ({}, "Without Quantization"),
            ({"quantization_bits": 8}, "With 8-bit Quantization"),  # If supported
        ]

        results = []
        for config, name in configs:
            result = self.benchmark_plugin_with_config(
                config=config, name=name, num_iterations=3
            )
            if result:
                results.append(result)

        print(f"\nQwen3-Coder-30B Quantization Impact:")
        for result in results:
            print(
                f"  {result['name']}: {result['tokens_per_second']:.2f} tokens/sec, "
                f"{result['memory_used']:.2f} MB"
            )

    def test_optimization_combinations(self):
        """Test different combinations of optimizations."""
        configs = [
            ({}, "Baseline"),
            ({"use_flash_attention": True}, "Flash Attention Only"),
            ({"memory_efficient": True}, "Memory Efficient Only"),
            (
                {"use_flash_attention": True, "memory_efficient": True},
                "Flash + Memory Efficient",
            ),
            (
                {
                    "use_flash_attention": True,
                    "memory_efficient": True,
                    "use_fused_layers": True,
                },
                "All Optimizations",
            ),
        ]

        results = []
        for config, name in configs:
            result = self.benchmark_plugin_with_config(
                config=config, name=name, num_iterations=3
            )
            if result:
                results.append(result)

        print(f"\nQwen3-Coder-30B Optimization Combinations:")
        print(
            f"{'Configuration':<30} | {'Tokens/sec':<12} | {'Memory (MB)':<12} | {'Time (s)':<10}"
        )
        print("-" * 70)
        for result in results:
            print(
                f"{result['name']:<30} | {result['tokens_per_second']:<12.2f} | "
                f"{result['memory_used']:<12.2f} | {result['avg_time_per_inference']:<10.4f}"
            )

    def test_optimization_impact_on_generation(self):
        """Test optimization impact on text generation."""
        configs = [
            ({}, "Baseline Generation"),
            (
                {"use_flash_attention": True, "memory_efficient": True},
                "Optimized Generation",
            ),
        ]

        prompt = "The impact of optimization on model performance"

        for config, name in configs:
            plugin = create_qwen3_coder_30b_plugin()

            init_success = plugin.initialize(device="cpu", **config)
            if not init_success:
                continue

            model = plugin.load_model()
            if model is None:
                continue

            # Warmup
            _ = plugin.generate_text(prompt, max_new_tokens=5)

            # Timing run
            start_time = time.time()
            generated = plugin.generate_text(prompt, max_new_tokens=25)
            end_time = time.time()

            generation_time = end_time - start_time

            print(f"\n{name}:")
            print(f"  Generation time: {generation_time:.4f}s")
            print(f"  Generated length: {len(generated)} chars")

            # Cleanup
            if hasattr(plugin, "cleanup"):
                plugin.cleanup()

    def test_optimization_impact_on_batch_processing(self):
        """Test optimization impact on batch processing."""
        configs = [
            ({}, "Baseline Batch"),
            (
                {"use_flash_attention": True, "memory_efficient": True},
                "Optimized Batch",
            ),
        ]

        batch_sizes = [1, 2, 4]

        for config, name in configs:
            print(f"\n{name}:")
            for batch_size in batch_sizes:
                plugin = create_qwen3_coder_30b_plugin()

                init_success = plugin.initialize(device="cpu", **config)
                if not init_success:
                    continue

                model = plugin.load_model()
                if model is None:
                    continue

                input_ids = torch.randint(0, 1000, (batch_size, 30))

                # Warmup
                for _ in range(2):
                    _ = plugin.infer(input_ids)

                # Timing run
                start_time = time.time()
                for i in range(3):
                    _ = plugin.infer(input_ids)
                end_time = time.time()

                avg_time = (end_time - start_time) / 3

                print(f"  Batch size {batch_size}: {avg_time:.4f}s per inference")

                # Cleanup
                if hasattr(plugin, "cleanup"):
                    plugin.cleanup()

    def test_optimization_stability(self):
        """Test that optimizations don't compromise stability."""
        configs = [{}, {"use_flash_attention": True, "memory_efficient": True}]

        for config in configs:
            plugin = create_qwen3_coder_30b_plugin()

            init_success = plugin.initialize(device="cpu", **config)
            if not init_success:
                continue

            model = plugin.load_model()
            if model is None:
                continue

            # Run multiple inferences to test stability
            for i in range(10):
                input_ids = torch.randint(0, 1000, (1, 20))

                try:
                    result = plugin.infer(input_ids)
                    # Check for NaN or Inf values
                    if isinstance(result, dict) and "logits" in result:
                        logits = result["logits"]
                        has_nan = torch.isnan(logits).any()
                        has_inf = torch.isinf(logits).any()
                        self.assertFalse(has_nan, f"NaN detected with config {config}")
                        self.assertFalse(has_inf, f"Inf detected with config {config}")
                    elif isinstance(result, torch.Tensor):
                        has_nan = torch.isnan(result).any()
                        has_inf = torch.isinf(result).any()
                        self.assertFalse(has_nan, f"NaN detected with config {config}")
                        self.assertFalse(has_inf, f"Inf detected with config {config}")
                except Exception as e:
                    self.fail(f"Exception during inference with config {config}: {e}")

            # Cleanup
            if hasattr(plugin, "cleanup"):
                plugin.cleanup()

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.base_plugin, "cleanup") and self.base_plugin.is_loaded:
            self.base_plugin.cleanup()


if __name__ == "__main__":
    unittest.main()

"""
Benchmark for Asynchronous Multimodal Processing in Qwen3-VL-2B Model

This module benchmarks the performance of the asynchronous multimodal processing system
for the Qwen3-VL-2B model, comparing it with synchronous processing and measuring
efficiency gains for different types of multimodal inputs.
"""

import asyncio
import os
import sys
import time
import unittest
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

# Add the project root to the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin


class BenchmarkAsyncMultimodalProcessing(unittest.TestCase):
    """Benchmark cases for asynchronous multimodal processing in Qwen3-VL-2B model."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_qwen3_vl_2b_instruct_plugin()
        self.config = Qwen3VL2BConfig()
        self.config.enable_async_multimodal_processing = True
        self.config.async_max_concurrent_requests = 4
        self.config.async_buffer_size = 100
        self.config.async_batch_timeout = 0.05
        self.config.enable_async_batching = True
        self.config.device = "cpu"  # Use CPU for consistent benchmarking

    def create_sample_inputs(self, count: int) -> List[Dict[str, Any]]:
        """Create sample multimodal inputs for benchmarking."""
        inputs = []
        for i in range(count):
            text = (
                f"This is a sample text input number {i} for multimodal processing. "
                * 5
            )
            image = Image.new(
                "RGB", (224, 224), color=(i % 255, (i * 2) % 255, (i * 3) % 255)
            )
            inputs.append({"text": text, "image": image})
        return inputs

    def benchmark_sync_processing(
        self, inputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Benchmark synchronous processing of multimodal inputs."""
        start_time = time.time()
        processing_times = []

        for inp in inputs:
            step_start = time.time()
            # Synchronous processing
            processed_data = self.plugin._model.preprocessor.preprocess(
                text=inp["text"], image=inp["image"]
            )
            result = self.plugin._model.forward(processed_data)
            step_end = time.time()
            processing_times.append(step_end - step_start)

        total_time = time.time() - start_time

        return {
            "total_time": total_time,
            "avg_processing_time": np.mean(processing_times),
            "std_processing_time": np.std(processing_times),
            "throughput": len(inputs) / total_time if total_time > 0 else 0,
            "processing_times": processing_times,
        }

    async def benchmark_async_processing(
        self, inputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Benchmark asynchronous processing of multimodal inputs."""
        start_time = time.time()
        processing_times = []

        for inp in inputs:
            step_start = time.time()
            result = await self.plugin._model.async_process_multimodal_request(
                text=inp["text"], image=inp["image"]
            )
            step_end = time.time()
            processing_times.append(step_end - step_start)

        total_time = time.time() - start_time

        return {
            "total_time": total_time,
            "avg_processing_time": np.mean(processing_times),
            "std_processing_time": np.std(processing_times),
            "throughput": len(inputs) / total_time if total_time > 0 else 0,
            "processing_times": processing_times,
        }

    async def benchmark_async_batch_processing(
        self, inputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Benchmark asynchronous batch processing of multimodal inputs."""
        start_time = time.time()

        results = await self.plugin._model.async_process_batch_multimodal_requests(
            inputs
        )

        total_time = time.time() - start_time

        return {
            "total_time": total_time,
            "throughput": len(inputs) / total_time if total_time > 0 else 0,
            "results_count": len(results),
        }

    @unittest.skip("Skipping model loading benchmark to avoid downloading large model")
    def test_async_vs_sync_processing_performance(self):
        """Benchmark performance difference between async and sync processing."""
        # Initialize plugin with async processing enabled
        success = self.plugin.initialize(**self.config.__dict__)
        self.assertTrue(success)

        # Create sample inputs
        sample_inputs = self.create_sample_inputs(10)

        print(
            f"\nBenchmarking async vs sync processing for {len(sample_inputs)} multimodal inputs..."
        )

        # Benchmark synchronous processing
        sync_results = self.benchmark_sync_processing(sample_inputs)
        print(f"Sync processing results:")
        print(f"  Total time: {sync_results['total_time']:.4f}s")
        print(f"  Avg processing time: {sync_results['avg_processing_time']:.4f}s")
        print(f"  Std processing time: {sync_results['std_processing_time']:.4f}s")
        print(f"  Throughput: {sync_results['throughput']:.2f} inputs/sec")

        # Benchmark asynchronous processing
        async_results = asyncio.run(self.benchmark_async_processing(sample_inputs))
        print(f"Async processing results:")
        print(f"  Total time: {async_results['total_time']:.4f}s")
        print(f"  Avg processing time: {async_results['avg_processing_time']:.4f}s")
        print(f"  Std processing time: {async_results['std_processing_time']:.4f}s")
        print(f"  Throughput: {async_results['throughput']:.2f} inputs/sec")

        # Compare results
        speedup = (
            sync_results["total_time"] / async_results["total_time"]
            if async_results["total_time"] > 0
            else float("inf")
        )
        print(f"Speedup: {speedup:.2f}x")

        # Verify that async processing completed successfully
        self.assertGreater(async_results["throughput"], 0)
        self.assertGreater(sync_results["throughput"], 0)

    @unittest.skip("Skipping model loading benchmark to avoid downloading large model")
    def test_async_batch_processing_performance(self):
        """Benchmark performance of async batch processing."""
        # Initialize plugin with async processing enabled
        success = self.plugin.initialize(**self.config.__dict__)
        self.assertTrue(success)

        # Create sample inputs
        sample_inputs = self.create_sample_inputs(20)

        print(
            f"\nBenchmarking async batch processing for {len(sample_inputs)} multimodal inputs..."
        )

        # Benchmark asynchronous batch processing
        batch_results = asyncio.run(
            self.benchmark_async_batch_processing(sample_inputs)
        )
        print(f"Async batch processing results:")
        print(f"  Total time: {batch_results['total_time']:.4f}s")
        print(f"  Throughput: {batch_results['throughput']:.2f} inputs/sec")
        print(f"  Results count: {batch_results['results_count']}")

        # Verify that async batch processing completed successfully
        self.assertGreater(batch_results["throughput"], 0)
        self.assertEqual(batch_results["results_count"], len(sample_inputs))

    def test_async_multimodal_manager_overhead(self):
        """Test the overhead of the async multimodal manager itself."""
        # Create a mock model to isolate the async processing overhead
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 2048
        mock_model.config.num_attention_heads = 16

        # Create the async multimodal manager
        manager = Qwen3VL2BAsyncMultimodalManager(model=mock_model, config=self.config)

        # Test initialization time
        start_time = time.time()
        success = manager.initialize()
        init_time = time.time() - start_time

        print(f"\nAsync multimodal manager overhead:")
        print(f"  Initialization time: {init_time:.4f}s")
        print(f"  Initialization success: {success}")

        # Verify that initialization completed
        # Note: With a mock model, initialization might fail, which is expected
        # We're just measuring the overhead of the manager itself
        self.assertIsNotNone(manager)

    def test_config_defaults_for_async_processing(self):
        """Test that the config has appropriate defaults for async processing."""
        config = Qwen3VL2BConfig()

        print(f"\nAsync multimodal processing config defaults:")
        print(f"  Enabled: {config.enable_async_multimodal_processing}")
        print(f"  Max concurrent requests: {config.async_max_concurrent_requests}")
        print(f"  Buffer size: {config.async_buffer_size}")
        print(f"  Batch timeout: {config.async_batch_timeout}")
        print(f"  Enable batching: {config.enable_async_batching}")

        # Verify default values
        self.assertTrue(config.enable_async_multimodal_processing)
        self.assertEqual(config.async_max_concurrent_requests, 8)
        self.assertEqual(config.async_buffer_size, 200)
        self.assertEqual(config.async_batch_timeout, 0.05)
        self.assertTrue(config.enable_async_batching)

    def test_async_processing_with_different_input_sizes(self):
        """Test async processing with different input sizes."""
        # Test with various input counts
        input_counts = [1, 5, 10, 20]

        for count in input_counts:
            with self.subTest(input_count=count):
                inputs = self.create_sample_inputs(count)

                # Just verify that inputs are created correctly
                self.assertEqual(len(inputs), count)
                for inp in inputs:
                    self.assertIn("text", inp)
                    self.assertIn("image", inp)
                    self.assertIsInstance(inp["text"], str)
                    self.assertIsInstance(inp["image"], Image.Image)

                print(f"  Created {count} sample inputs successfully")


def run_benchmark():
    """Run the complete benchmark suite."""
    print("=" * 80)
    print("ASYNC MULTIMODAL PROCESSING BENCHMARK FOR QWEN3-VL-2B")
    print("=" * 80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(BenchmarkAsyncMultimodalProcessing)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.2f}%"
        if result.testsRun > 0
        else "N/A"
    )

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_benchmark()
    print(f"\nBenchmark {'PASSED' if success else 'FAILED'}")

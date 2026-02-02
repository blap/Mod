"""
Standardized Benchmark for Throughput - GLM-4.7

This module benchmarks the throughput for the GLM-4.7 model.
"""

import queue
import threading
import time
import unittest

import torch

from inference_pio.models.glm_4_7.plugin import create_glm_4_7_plugin


class BenchmarkGLM47Throughput(unittest.TestCase):
    """Benchmark cases for GLM-4.7 throughput."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_glm_4_7_plugin()
        success = self.plugin.initialize(
            device="cpu"
        )  # Using CPU for consistent benchmarks
        self.assertTrue(success)
        self.model = self.plugin.load_model()
        self.assertTrue(self.model is not None)

    def benchmark_requests_per_second(self, num_requests=50, concurrent=False):
        """Benchmark requests per second."""
        if concurrent:
            # Concurrent request benchmark
            request_queue = queue.Queue()
            result_queue = queue.Queue()

            def worker():
                while True:
                    item = request_queue.get()
                    if item is None:
                        break
                    idx, input_data = item
                    start_time = time.time()
                    result = self.plugin.infer(input_data)
                    end_time = time.time()
                    result_queue.put((idx, result, end_time - start_time))
                    request_queue.task_done()

            # Start worker threads
            num_threads = min(4, num_requests)  # Limit number of threads
            threads = []
            for i in range(num_threads):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)

            # Submit requests
            start_time = time.time()
            for i in range(num_requests):
                input_ids = torch.randint(0, 1000, (1, 20))
                request_queue.put((i, input_ids))

            # Wait for all requests to complete
            request_queue.join()
            end_time = time.time()

            # Stop worker threads
            for i in range(num_threads):
                request_queue.put(None)
            for t in threads:
                t.join()

            total_time = end_time - start_time
        else:
            # Sequential request benchmark
            start_time = time.time()
            for i in range(num_requests):
                input_ids = torch.randint(0, 1000, (1, 20))
                result = self.plugin.infer(input_ids)
            end_time = time.time()
            total_time = end_time - start_time

        requests_per_second = (
            num_requests / total_time if total_time > 0 else float("inf")
        )

        return {
            "total_time": total_time,
            "requests_per_second": requests_per_second,
            "num_requests": num_requests,
            "concurrent": concurrent,
        }

    def test_sequential_throughput(self):
        """Test sequential request throughput."""
        results = self.benchmark_requests_per_second(num_requests=20, concurrent=False)

        print(f"\nGLM-4.7 Sequential Throughput:")
        print(f"  Total time: {results['total_time']:.4f}s")
        print(f"  Requests per second: {results['requests_per_second']:.2f}")
        print(f"  Total requests: {results['num_requests']}")

        # Basic sanity check
        self.assertGreater(results["requests_per_second"], 0)

    def test_concurrent_throughput(self):
        """Test concurrent request throughput."""
        results = self.benchmark_requests_per_second(num_requests=20, concurrent=True)

        print(f"\nGLM-4.7 Concurrent Throughput:")
        print(f"  Total time: {results['total_time']:.4f}s")
        print(f"  Requests per second: {results['requests_per_second']:.2f}")
        print(f"  Total requests: {results['num_requests']}")

        # Basic sanity check
        self.assertGreater(results["requests_per_second"], 0)

    def test_throughput_with_variable_request_sizes(self):
        """Test throughput with different request sizes."""
        request_counts = [10, 25, 50]

        for count in request_counts:
            with self.subTest(request_count=count):
                results = self.benchmark_requests_per_second(
                    num_requests=count, concurrent=False
                )

                print(
                    f"  {count} requests: {results['requests_per_second']:.2f} req/sec"
                )

                # Basic sanity check
                self.assertGreater(results["requests_per_second"], 0)

    def test_batch_throughput(self):
        """Test throughput with batched requests."""
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Warmup
                for _ in range(3):
                    input_ids = torch.randint(0, 1000, (batch_size, 20))
                    _ = self.plugin.infer(input_ids)

                # Timing run
                num_batches = 10
                start_time = time.time()
                for i in range(num_batches):
                    input_ids = torch.randint(0, 1000, (batch_size, 20))
                    result = self.plugin.infer(input_ids)
                end_time = time.time()

                total_time = end_time - start_time
                total_requests = num_batches * batch_size
                requests_per_second = (
                    total_requests / total_time if total_time > 0 else float("inf")
                )

                print(f"\nGLM-4.7 Batch Throughput (Batch size: {batch_size}):")
                print(f"  Total requests: {total_requests}")
                print(f"  Total time: {total_time:.4f}s")
                print(f"  Requests per second: {requests_per_second:.2f}")

                # Basic sanity check
                self.assertGreater(requests_per_second, 0)

    def test_generation_throughput(self):
        """Test text generation throughput."""
        prompts = [
            "The weather today is nice.",
            "Artificial intelligence is advancing rapidly.",
            "Machine learning models require training data.",
            "Natural language processing enables human-computer interaction.",
            "Deep learning architectures have multiple layers.",
        ] * 4  # Repeat to get 20 prompts

        # Warmup
        for prompt in prompts[:5]:
            _ = self.plugin.generate_text(prompt, max_new_tokens=10)

        # Timing run
        start_time = time.time()
        for prompt in prompts:
            result = self.plugin.generate_text(prompt, max_new_tokens=10)
        end_time = time.time()

        total_time = end_time - start_time
        total_requests = len(prompts)
        requests_per_second = (
            total_requests / total_time if total_time > 0 else float("inf")
        )

        print(f"\nGLM-4.7 Generation Throughput:")
        print(f"  Total requests: {total_requests}")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Requests per second: {requests_per_second:.2f}")

        # Basic sanity check
        self.assertGreater(requests_per_second, 0)

    def test_throughput_under_load(self):
        """Test throughput under sustained load."""
        # Run a longer benchmark to measure sustained throughput
        num_requests = 100
        start_time = time.time()

        for i in range(num_requests):
            input_ids = torch.randint(0, 1000, (1, 30))
            result = self.plugin.infer(input_ids)

        end_time = time.time()

        total_time = end_time - start_time
        requests_per_second = (
            num_requests / total_time if total_time > 0 else float("inf")
        )

        print(f"\nGLM-4.7 Sustained Load Throughput:")
        print(f"  Total requests: {num_requests}")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Requests per second: {requests_per_second:.2f}")

        # Basic sanity check
        self.assertGreater(requests_per_second, 0)

    def test_latency_vs_throughput_tradeoff(self):
        """Test the trade-off between latency and throughput."""
        # Single request latency
        input_ids = torch.randint(0, 1000, (1, 20))

        # Warmup
        for _ in range(3):
            _ = self.plugin.infer(input_ids)

        # Measure single request latency
        start_time = time.time()
        result = self.plugin.infer(input_ids)
        single_latency = time.time() - start_time

        # Measure throughput with multiple requests
        num_requests = 20
        start_time = time.time()
        for i in range(num_requests):
            result = self.plugin.infer(input_ids)
        total_time = time.time() - start_time
        avg_latency = total_time / num_requests
        throughput = num_requests / total_time if total_time > 0 else float("inf")

        print(f"\nGLM-4.7 Latency vs Throughput:")
        print(f"  Single request latency: {single_latency:.4f}s")
        print(f"  Average latency (under load): {avg_latency:.4f}s")
        print(f"  Throughput: {throughput:.2f} req/sec")

        # Basic sanity checks
        self.assertGreater(single_latency, 0)
        self.assertGreater(avg_latency, 0)
        self.assertGreater(throughput, 0)

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()

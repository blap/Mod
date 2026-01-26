"""
Template for Standardized Model Benchmarks

This template provides a standardized structure for creating benchmarks for new models.
Each model should implement benchmarks following this structure.
"""

import time
import torch
import unittest
from inference_pio.common.benchmark_interface import (
    InferenceSpeedBenchmark,
    MemoryUsageBenchmark, 
    AccuracyBenchmark,
    BatchProcessingBenchmark,
    ModelLoadingTimeBenchmark,
    BenchmarkRunner
)


class StandardizedModelBenchmarkTemplate(unittest.TestCase):
    """
    Template for standardized model benchmarks.
    Each model should create a subclass of this template.
    """

    def setUp(self):
        """
        Set up benchmark fixtures before each test method.
        Subclasses should override this method to initialize their specific model plugin.
        """
        # This should be overridden by subclasses
        self.plugin = None
        self.model_name = "template_model"
        self.assertTrue(self.plugin is not None, "Plugin must be initialized in subclass")

    def run_standard_performance_benchmarks(self):
        """
        Run standard performance benchmarks for the model.
        """
        runner = BenchmarkRunner()

        # Create standard performance benchmarks
        benchmarks = [
            InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=20),
            InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=50), 
            InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=100),
            MemoryUsageBenchmark(self.plugin, self.model_name),
            BatchProcessingBenchmark(self.plugin, self.model_name),
            ModelLoadingTimeBenchmark(self.plugin, self.model_name)
        ]

        # Run benchmarks
        results = runner.run_multiple_benchmarks(benchmarks)

        # Print results
        for result in results:
            print(f"{self.model_name} - {result.name}: {result.value} {result.unit}")

        return results

    def run_standard_accuracy_benchmarks(self):
        """
        Run standard accuracy benchmarks for the model.
        """
        runner = BenchmarkRunner()

        # Create standard accuracy benchmarks
        benchmarks = [
            AccuracyBenchmark(self.plugin, self.model_name)
        ]

        # Run benchmarks
        results = runner.run_multiple_benchmarks(benchmarks)

        # Print results
        for result in results:
            print(f"{self.model_name} - {result.name}: {result.value} {result.unit}")

        return results

    def test_performance_benchmarks(self):
        """
        Test method to run performance benchmarks.
        """
        results = self.run_standard_performance_benchmarks()

        # Basic assertions - ensure we got results and they are valid
        self.assertGreater(len(results), 0, "Should have at least one benchmark result")

        for result in results:
            # Check that results have expected attributes
            self.assertIsNotNone(result.name)
            self.assertIsNotNone(result.value)
            self.assertIsNotNone(result.unit)
            self.assertIsNotNone(result.model_name)

            # For performance metrics, ensure they are positive (where applicable)
            if result.category == "performance":
                if result.unit in ["tokens/sec", "MB", "seconds"]:
                    self.assertGreaterEqual(result.value, 0, f"Performance value should be non-negative: {result.name}")

    def test_accuracy_benchmarks(self):
        """
        Test method to run accuracy benchmarks.
        """
        results = self.run_standard_accuracy_benchmarks()

        # Basic assertions
        self.assertGreater(len(results), 0, "Should have at least one benchmark result")

        for result in results:
            # Check that results have expected attributes
            self.assertIsNotNone(result.name)
            self.assertIsNotNone(result.value)
            self.assertIsNotNone(result.unit)
            self.assertIsNotNone(result.model_name)

            # For accuracy metrics, ensure they are in valid range
            if "accuracy" in result.name:
                self.assertGreaterEqual(result.value, 0, "Accuracy should be non-negative")
                if result.unit == "ratio":
                    self.assertLessEqual(result.value, 1.0, "Accuracy ratio should be <= 1.0")

    def tearDown(self):
        """
        Clean up after each test method.
        Subclasses should override this method to properly clean up their model plugin.
        """
        if hasattr(self.plugin, 'cleanup') and self.plugin.is_loaded:
            self.plugin.cleanup()


# Example implementation for a specific model
class GLM47FlashBenchmark(StandardizedModelBenchmarkTemplate):
    """
    Specific benchmark implementation for GLM-4.7-Flash model.
    """

    def setUp(self):
        """Set up benchmark fixtures for GLM-4.7-Flash."""
        from inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
        
        self.plugin = create_glm_4_7_flash_plugin()
        success = self.plugin.initialize(device="cpu", use_mock_model=True)  # Using CPU for consistent benchmarks
        self.assertTrue(success)
        self.model_name = "GLM-4.7-Flash"
        
        # Load model for benchmarking
        model = self.plugin.load_model()
        self.assertTrue(model is not None)

    def test_specific_inference_speeds(self):
        """Test specific inference speeds for GLM-4.7-Flash."""
        # Run specific speed tests
        short_result = InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=20).run()
        medium_result = InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=50).run()
        long_result = InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=100).run()

        print(f"\nGLM-4.7-Flash Inference Speed Results:")
        print(f"  Short (20 tokens): {short_result.value:.2f} {short_result.unit}")
        print(f"  Medium (50 tokens): {medium_result.value:.2f} {medium_result.unit}")
        print(f"  Long (100 tokens): {long_result.value:.2f} {long_result.unit}")

        # Basic sanity checks
        self.assertGreater(short_result.value, 0)
        self.assertGreater(medium_result.value, 0)
        self.assertGreater(long_result.value, 0)

    def test_specific_memory_usage(self):
        """Test memory usage for GLM-4.7-Flash."""
        memory_result = MemoryUsageBenchmark(self.plugin, self.model_name).run()

        print(f"\nGLM-4.7-Flash Memory Usage: {memory_result.value:.2f} {memory_result.unit}")

        # Memory usage should be positive
        self.assertGreater(memory_result.value, 0)

    def test_specific_batch_processing(self):
        """Test batch processing for GLM-4.7-Flash."""
        batch_result = BatchProcessingBenchmark(self.plugin, self.model_name).run()

        print(f"\nGLM-4.7-Flash Batch Processing Throughput: {batch_result.value:.2f} {batch_result.unit}")

        # Throughput should be positive
        self.assertGreater(batch_result.value, 0)


# Example implementation for another model
class Qwen34BInstruct2507Benchmark(StandardizedModelBenchmarkTemplate):
    """
    Specific benchmark implementation for Qwen3-4B-Instruct-2507 model.
    """

    def setUp(self):
        """Set up benchmark fixtures for Qwen3-4B-Instruct-2507."""
        from inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
        
        self.plugin = create_qwen3_4b_instruct_2507_plugin()
        success = self.plugin.initialize(device="cpu", use_mock_model=True)  # Using CPU for consistent benchmarks
        self.assertTrue(success)
        self.model_name = "Qwen3-4B-Instruct-2507"
        
        # Load model for benchmarking
        model = self.plugin.load_model()
        self.assertTrue(model is not None)

    def test_specific_inference_speeds(self):
        """Test specific inference speeds for Qwen3-4B-Instruct-2507."""
        # Run specific speed tests
        short_result = InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=20).run()
        medium_result = InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=50).run()
        long_result = InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=100).run()

        print(f"\nQwen3-4B-Instruct-2507 Inference Speed Results:")
        print(f"  Short (20 tokens): {short_result.value:.2f} {short_result.unit}")
        print(f"  Medium (50 tokens): {medium_result.value:.2f} {medium_result.unit}")
        print(f"  Long (100 tokens): {long_result.value:.2f} {long_result.unit}")

        # Basic sanity checks
        self.assertGreater(short_result.value, 0)
        self.assertGreater(medium_result.value, 0)
        self.assertGreater(long_result.value, 0)

    def test_specific_memory_usage(self):
        """Test memory usage for Qwen3-4B-Instruct-2507."""
        memory_result = MemoryUsageBenchmark(self.plugin, self.model_name).run()

        print(f"\nQwen3-4B-Instruct-2507 Memory Usage: {memory_result.value:.2f} {memory_result.unit}")

        # Memory usage should be positive
        self.assertGreater(memory_result.value, 0)

    def test_specific_accuracy(self):
        """Test accuracy for Qwen3-4B-Instruct-2507."""
        accuracy_result = AccuracyBenchmark(self.plugin, self.model_name).run()

        print(f"\nQwen3-4B-Instruct-2507 Accuracy Score: {accuracy_result.value} {accuracy_result.unit}")

        # Accuracy should be between 0 and 1 for ratio
        if accuracy_result.unit == "ratio":
            self.assertGreaterEqual(accuracy_result.value, 0)
            self.assertLessEqual(accuracy_result.value, 1.0)


if __name__ == '__main__':
    unittest.main()
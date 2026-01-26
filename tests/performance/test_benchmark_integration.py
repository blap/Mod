"""
Integration Test for Performance Regression with Benchmark System

This test validates that the performance regression system properly integrates
with the existing benchmark infrastructure.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import time

from src.inference_pio.common.benchmark_interface import (
    InferenceSpeedBenchmark,
    MemoryUsageBenchmark,
    BenchmarkResult
)
from src.inference_pio.common.performance_regression_tracker import (
    PerformanceRegressionTracker,
    record_performance_metric,
    get_regression_alerts
)


class MockModelPlugin:
    """Mock model plugin for testing purposes."""
    
    def __init__(self):
        self.is_loaded = False
        self._model = None
    
    def initialize(self, **kwargs):
        """Initialize the mock plugin."""
        self._config = kwargs
        return True
    
    def load_model(self):
        """Load the mock model."""
        self.is_loaded = True
        self._model = "mock_model"
        return self._model
    
    def infer(self, input_ids):
        """Mock inference method."""
        # Simulate inference by sleeping briefly
        time.sleep(0.01)
        return {"output": "mock_output", "input_ids": input_ids}
    
    def generate_text(self, prompt, max_new_tokens=10):
        """Mock text generation."""
        return f"Generated: {prompt}"
    
    def cleanup(self):
        """Clean up resources."""
        self.is_loaded = False
        self._model = None
        return True


class TestBenchmarkIntegration(unittest.TestCase):
    """Test integration between benchmark system and performance regression tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tracker = PerformanceRegressionTracker(
            storage_dir=self.temp_dir,
            regression_threshold=5.0
        )
        self.model_plugin = MockModelPlugin()
        self.model_name = "mock_model"
        
        # Initialize the mock plugin
        self.model_plugin.initialize()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        self.model_plugin.cleanup()
    
    def test_inference_speed_benchmark_integration(self):
        """Test that InferenceSpeedBenchmark results are properly tracked."""
        # Run the benchmark
        benchmark = InferenceSpeedBenchmark(self.model_plugin, self.model_name, input_length=50, num_iterations=3)
        result = benchmark.run()
        
        # Manually record the result using the regression tracker
        record_performance_metric(
            name=result.name,
            value=result.value,
            unit=result.unit,
            model_name=result.model_name,
            category=result.category,
            metadata=result.metadata
        )
        
        # Verify the metric was recorded
        metric_key = f"{result.model_name}:{result.name}"
        self.assertIn(metric_key, self.tracker.metrics_history)
        self.assertEqual(len(self.tracker.metrics_history[metric_key]), 1)
        recorded_metric = self.tracker.metrics_history[metric_key][0]
        
        # Verify the recorded values match the benchmark result
        self.assertEqual(recorded_metric.name, result.name)
        self.assertEqual(recorded_metric.value, result.value)
        self.assertEqual(recorded_metric.unit, result.unit)
        self.assertEqual(recorded_metric.model_name, result.model_name)
        self.assertEqual(recorded_metric.category, result.category)
        self.assertEqual(recorded_metric.metadata, result.metadata)
    
    def test_memory_usage_benchmark_integration(self):
        """Test that MemoryUsageBenchmark results are properly tracked."""
        # Run the benchmark
        benchmark = MemoryUsageBenchmark(self.model_plugin, self.model_name)
        result = benchmark.run()
        
        # Manually record the result using the regression tracker
        record_performance_metric(
            name=result.name,
            value=result.value,
            unit=result.unit,
            model_name=result.model_name,
            category=result.category,
            metadata=result.metadata
        )
        
        # Verify the metric was recorded
        metric_key = f"{result.model_name}:{result.name}"
        self.assertIn(metric_key, self.tracker.metrics_history)
        self.assertEqual(len(self.tracker.metrics_history[metric_key]), 1)
        recorded_metric = self.tracker.metrics_history[metric_key][0]
        
        # Verify the recorded values match the benchmark result
        self.assertEqual(recorded_metric.name, result.name)
        self.assertEqual(recorded_metric.value, result.value)
        self.assertEqual(recorded_metric.unit, result.unit)
        self.assertEqual(recorded_metric.model_name, result.model_name)
        self.assertEqual(recorded_metric.category, result.category)
        self.assertEqual(recorded_metric.metadata, result.metadata)
    
    def test_regression_detection_with_benchmark_results(self):
        """Test that regressions are properly detected from benchmark results."""
        # Run first benchmark (baseline)
        benchmark1 = InferenceSpeedBenchmark(self.model_plugin, self.model_name, input_length=20, num_iterations=3)
        result1 = benchmark1.run()
        
        record_performance_metric(
            name=result1.name,
            value=result1.value,
            unit=result1.unit,
            model_name=result1.model_name,
            category=result1.category,
            metadata=result1.metadata
        )
        
        # Simulate a regression by artificially reducing the performance value
        # In a real scenario, this would come from a subsequent benchmark run
        degraded_value = result1.value * 0.9  # 10% degradation
        
        record_performance_metric(
            name=result1.name,
            value=degraded_value,
            unit=result1.unit,
            model_name=result1.model_name,
            category=result1.category,
            metadata=result1.metadata
        )
        
        # Check for regression alerts
        alerts = get_regression_alerts()
        
        # Verify a regression was detected
        regression_alerts = [a for a in alerts if a.metric_name == result1.name and a.model_name == result1.model_name]
        self.assertEqual(len(regression_alerts), 1, "Regression should be detected")
        
        alert = regression_alerts[0]
        self.assertGreater(alert.previous_value, alert.current_value, "Previous value should be greater than current (regression)")
        self.assertEqual(alert.severity.value, "warning", "Should be a warning for 10% regression")
    
    def test_multiple_benchmarks_same_model(self):
        """Test tracking multiple benchmarks for the same model."""
        # Run multiple different benchmarks for the same model
        benchmarks = [
            InferenceSpeedBenchmark(self.model_plugin, self.model_name, input_length=20),
            MemoryUsageBenchmark(self.model_plugin, self.model_name),
        ]
        
        results = []
        for benchmark in benchmarks:
            result = benchmark.run()
            results.append(result)
            
            # Record each result
            record_performance_metric(
                name=result.name,
                value=result.value,
                unit=result.unit,
                model_name=result.model_name,
                category=result.category,
                metadata=result.metadata
            )
        
        # Verify all metrics were recorded
        expected_keys = [f"{self.model_name}:{result.name}" for result in results]
        for key in expected_keys:
            self.assertIn(key, self.tracker.metrics_history)
            self.assertEqual(len(self.tracker.metrics_history[key]), 1)
    
    def test_different_models_same_metric(self):
        """Test tracking the same metric for different models."""
        model_names = ["model_a", "model_b"]
        
        for model_name in model_names:
            # Create a mock plugin for each model
            plugin = MockModelPlugin()
            plugin.initialize()
            
            # Run the same benchmark for both models
            benchmark = InferenceSpeedBenchmark(plugin, model_name, input_length=30)
            result = benchmark.run()
            
            # Record the result
            record_performance_metric(
                name=result.name,
                value=result.value,
                unit=result.unit,
                model_name=result.model_name,
                category=result.category,
                metadata=result.metadata
            )
            
            plugin.cleanup()
        
        # Verify metrics were recorded separately for each model
        for model_name in model_names:
            key = f"{model_name}:inference_speed_30tokens"
            self.assertIn(key, self.tracker.metrics_history)
            self.assertEqual(len(self.tracker.metrics_history[key]), 1)


class TestRealisticBenchmarkWorkflow(unittest.TestCase):
    """Test a realistic workflow of running benchmarks and tracking performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tracker = PerformanceRegressionTracker(
            storage_dir=self.temp_dir,
            regression_threshold=10.0  # Higher threshold for this test
        )
        self.model_plugin = MockModelPlugin()
        self.model_name = "workflow_test_model"
        
        # Initialize the mock plugin
        self.model_plugin.initialize()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        self.model_plugin.cleanup()
    
    def test_complete_workflow(self):
        """Test a complete workflow of benchmarking and regression tracking."""
        # Step 1: Establish baseline by running benchmarks
        print("Step 1: Establishing baseline...")
        baseline_results = {}
        
        # Run multiple benchmarks to establish baseline
        benchmarks = [
            InferenceSpeedBenchmark(self.model_plugin, self.model_name, input_length=20, num_iterations=5),
            InferenceSpeedBenchmark(self.model_plugin, self.model_name, input_length=50, num_iterations=5),
            MemoryUsageBenchmark(self.model_plugin, self.model_name),
        ]
        
        for benchmark in benchmarks:
            result = benchmark.run()
            baseline_results[result.name] = result.value
            
            # Record baseline result
            record_performance_metric(
                name=result.name,
                value=result.value,
                unit=result.unit,
                model_name=result.model_name,
                category=result.category,
                metadata=result.metadata
            )
        
        # Verify no regressions in baseline (should be none since it's first run)
        alerts = get_regression_alerts()
        baseline_alerts = [a for a in alerts if a.model_name == self.model_name]
        self.assertEqual(len(baseline_alerts), 0, "No alerts should be generated for baseline")
        
        print(f"Baseline established with {len(baseline_results)} metrics")
        
        # Step 2: Simulate a later run with potentially different results
        print("Step 2: Running follow-up benchmarks...")
        
        # Run the same benchmarks again
        followup_results = {}
        for benchmark in benchmarks:
            result = benchmark.run()
            followup_results[result.name] = result.value
            
            # Record follow-up result
            record_performance_metric(
                name=result.name,
                value=result.value,
                unit=result.unit,
                model_name=result.model_name,
                category=result.category,
                metadata=result.metadata
            )
        
        # Check for regressions (there shouldn't be any since we're using the same mock)
        alerts = get_regression_alerts()
        followup_alerts = [a for a in alerts if a.model_name == self.model_name and a not in baseline_alerts]
        self.assertEqual(len(followup_alerts), 0, "No regressions should be detected with identical results")
        
        print(f"Follow-up run completed, {len(followup_results)} metrics recorded")
        
        # Step 3: Simulate a regression scenario
        print("Step 3: Simulating performance regression...")
        
        # Artificially introduce a regression in one of the metrics
        speed_metric_name = "inference_speed_20tokens"
        baseline_speed = baseline_results[speed_metric_name]
        degraded_speed = baseline_speed * 0.85  # 15% degradation
        
        record_performance_metric(
            name=speed_metric_name,
            value=degraded_speed,
            unit="tokens/sec",
            model_name=self.model_name,
            category="performance",
            metadata={"simulated_regression": True}
        )
        
        # Verify regression was detected
        alerts = get_regression_alerts()
        regression_alerts = [a for a in alerts if 
                           a.model_name == self.model_name and 
                           a.metric_name == speed_metric_name and 
                           a not in baseline_alerts and 
                           a not in followup_alerts]
        
        self.assertEqual(len(regression_alerts), 1, "One regression should be detected")
        
        regression_alert = regression_alerts[0]
        self.assertGreater(regression_alert.previous_value, regression_alert.current_value)
        self.assertEqual(regression_alert.severity.value, "warning", "15% regression should trigger warning")
        
        print(f"Regression successfully detected: {regression_alert.message}")
        
        print("Complete workflow test passed!")


if __name__ == "__main__":
    unittest.main()
"""
Performance Testing Module for Mod Project

This module provides independent functionality for performance testing
of different aspects of the Mod project. Each model/plugin is independent 
with its own configuration, tests and benchmarks.
"""

from typing import Type
import unittest
import time
import statistics
import psutil
import GPUtil
import numpy as np
from typing import Callable, Any, Dict, List, Tuple
import sys
import os
import logging
from contextlib import contextmanager

# Add the src directory to the path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)

@contextmanager
def performance_monitor():
    """
    Context manager to monitor system resources during performance tests.
    """
    # Get initial resource usage
    initial_cpu_percent = psutil.cpu_percent(interval=None)
    initial_memory = psutil.virtual_memory().percent
    initial_gpus = []
    gpus_present = GPUtil.getGPUs()
    for gpu in gpus_present:
        initial_gpus.append({
            'id': gpu.id,
            'load': gpu.load,
            'memory_util': gpu.memoryUtil,
            'memory_used': gpu.memoryUsed,
            'memory_total': gpu.memoryTotal
        })
    
    initial_time = time.time()
    initial_process = psutil.Process()
    initial_rss = initial_process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    # Calculate final resource usage
    final_time = time.time()
    final_process = psutil.Process()
    final_rss = final_process.memory_info().rss / 1024 / 1024  # MB
    
    final_cpu_percent = psutil.cpu_percent(interval=None)
    final_memory = psutil.virtual_memory().percent
    
    # Calculate differences
    time_taken = final_time - initial_time
    memory_delta = final_rss - initial_rss
    
    # Log resource usage
    logger.info(f"Performance metrics: Time={time_taken:.4f}s, Memory Delta={memory_delta:.2f}MB")
    
    # Return metrics
    metrics = {
        'time_taken': time_taken,
        'memory_delta_mb': memory_delta,
        'initial_cpu_percent': initial_cpu_percent,
        'final_cpu_percent': final_cpu_percent,
        'initial_memory_percent': initial_memory,
        'final_memory_percent': final_memory,
        'initial_rss_mb': initial_rss,
        'final_rss_mb': final_rss
    }
    
    # Add GPU metrics if available
    final_gpus = GPUtil.getGPUs()
    for i, gpu in enumerate(final_gpus):
        if i < len(initial_gpus):
            metrics[f'gpu_{gpu.id}_load_delta'] = gpu.load - initial_gpus[i]['load']
            metrics[f'gpu_{gpu.id}_memory_util_delta'] = gpu.memoryUtil - initial_gpus[i]['memory_util']
            metrics[f'gpu_{gpu.id}_memory_used_delta'] = gpu.memoryUsed - initial_gpus[i]['memory_used']
    
    yield metrics


class PerformanceTestBase(unittest.TestCase):
    """
    Base class for performance tests.
    Provides common functionality for measuring performance metrics.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        super().setUp()
        logger.info(f"Setting up performance test: {self._testMethodName}")
        
    def tearDown(self):
        """Clean up after each test method."""
        super().tearDown()
        logger.info(f"Tearing down performance test: {self._testMethodName}")
    
    def measure_execution_time(self, func: Callable, *args, **kwargs) -> float:
        """
        Measure the execution time of a function.
        
        Args:
            func: Function to measure
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Execution time in seconds
        """
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time
    
    def measure_multiple_executions(self, func: Callable, iterations: int, *args, **kwargs) -> List[float]:
        """
        Measure execution times over multiple executions.
        
        Args:
            func: Function to measure
            iterations: Number of iterations to run
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            List of execution times
        """
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        return times
    
    def assert_performance_threshold(self, execution_time: float, threshold: float, 
                                   msg: str = None):
        """
        Assert that execution time is below a threshold.
        
        Args:
            execution_time: Measured execution time
            threshold: Maximum allowed execution time
            msg: Custom failure message
        """
        if msg is None:
            msg = f"Execution time {execution_time:.4f}s exceeded threshold {threshold:.4f}s"
        self.assertLessEqual(execution_time, threshold, msg)
    
    def calculate_performance_stats(self, times: List[float]) -> Dict[str, float]:
        """
        Calculate performance statistics from execution times.
        
        Args:
            times: List of execution times
            
        Returns:
            Dictionary with performance statistics
        """
        if not times:
            return {}
        
        stats = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min': min(times),
            'max': max(times),
            'total': sum(times)
        }
        return stats


class ModelPerformanceTest(PerformanceTestBase):
    """
    Performance test class for model plugins.
    Each model should inherit from this class and implement required methods.
    """
    
    def get_model_plugin_class(self):
        """Override this method to return the model plugin class to test."""
        raise NotImplementedError("Method not implemented")
    
    def setUp(self):
        """Set up the model plugin for performance testing."""
        super().setUp()
        self.model_plugin_class = self.get_model_plugin_class()
        self.model_instance = None
        
        # Initialize the model plugin
        try:
            self.model_instance = self.model_plugin_class()
        except Exception as e:
            self.fail(f"Failed to initialize model plugin: {str(e)}")
    
    def benchmark_model_processing(self, input_data: Any, iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark the model processing performance.
        
        Args:
            input_data: Input data to process
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with performance statistics
        """
        def process_once():
            return self.model_instance.process(input_data)
        
        times = self.measure_multiple_executions(process_once, iterations)
        return self.calculate_performance_stats(times)
    
    def test_model_throughput(self):
        """Test the throughput of the model."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
            
        # Test with a reasonable input
        test_input = "This is a test input for performance measurement."
        
        # Measure performance over multiple iterations
        stats = self.benchmark_model_processing(test_input, iterations=5)
        
        logger.info(f"Model throughput stats: {stats}")
        
        # Assert reasonable performance (adjust thresholds as needed)
        self.assertLess(stats['mean'], 5.0, "Average processing time should be under 5 seconds")
    
    def test_memory_usage(self):
        """Test the memory usage of the model during processing."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
            
        test_input = "This is a test input for memory usage measurement."
        
        # Monitor memory before and after processing
        initial_memory = psutil.virtual_memory().percent
        result = self.model_instance.process(test_input)
        final_memory = psutil.virtual_memory().percent
        
        memory_increase = final_memory - initial_memory
        logger.info(f"Memory usage increase: {memory_increase}%")
        
        # Assert reasonable memory usage (adjust threshold as needed)
        self.assertLess(memory_increase, 20.0, "Memory increase should be under 20%")


class PluginPerformanceTest(PerformanceTestBase):
    """
    Performance test class for plugins.
    Each plugin should inherit from this class and implement required methods.
    """
    
    def get_plugin_class(self):
        """Override this method to return the plugin class to test."""
        raise NotImplementedError("Method not implemented")
    
    def setUp(self):
        """Set up the plugin for performance testing."""
        super().setUp()
        self.plugin_class = self.get_plugin_class()
        self.plugin_instance = None
        
        # Initialize the plugin
        try:
            self.plugin_instance = self.plugin_class()
        except Exception as e:
            self.fail(f"Failed to initialize plugin: {str(e)}")
    
    def benchmark_plugin_execution(self, input_data: Any, iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark the plugin execution performance.
        
        Args:
            input_data: Input data to process
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with performance statistics
        """
        def execute_once():
            return self.plugin_instance.execute(input_data)
        
        times = self.measure_multiple_executions(execute_once, iterations)
        return self.calculate_performance_stats(times)
    
    def test_plugin_execution_speed(self):
        """Test the execution speed of the plugin."""
        if not self.plugin_instance:
            self.skipTest("Plugin instance not available")
            
        # Test with a reasonable input
        test_input = "This is a test input for performance measurement."
        
        # Measure performance over multiple iterations
        stats = self.benchmark_plugin_execution(test_input, iterations=5)
        
        logger.info(f"Plugin execution stats: {stats}")
        
        # Assert reasonable performance (adjust thresholds as needed)
        self.assertLess(stats['mean'], 2.0, "Average execution time should be under 2 seconds")


def run_performance_tests(test_classes: List[Type[unittest.TestCase]], verbosity: int = 2):
    """
    Run performance tests with specified test classes.
    
    Args:
        test_classes: List of test classes to run
        verbosity: Verbosity level for test output
    
    Returns:
        TestResult object with results
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def performance_test_suite():
    """
    Create a test suite for performance tests.
    
    Returns:
        TestSuite object containing all performance tests
    """
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.join(os.path.dirname(__file__), '..', 'tests', 'performance'),
        pattern='test_*.py',
        top_level_dir=os.path.join(os.path.dirname(__file__), '..')
    )
    return suite


# Example usage and test runner
if __name__ == "__main__":
    # This would typically be called from the main test runner
    # For demonstration purposes, we'll show the structure
    print("Performance Testing Module loaded successfully")
    print("Available test classes:")
    print("- PerformanceTestBase")
    print("- ModelPerformanceTest") 
    print("- PluginPerformanceTest")
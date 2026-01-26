"""
Benchmark Fixtures for Standardized Testing

This module provides standardized fixtures for benchmark testing across all models.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Generator, Optional, Dict, Any
import torch
import numpy as np
from unittest.mock import Mock, patch
import pytest


class MockModelPlugin:
    """Mock model plugin for testing purposes."""
    
    def __init__(self):
        self.is_loaded = False
        self.device = "cpu"
        self.model_path = None
        
    def initialize(self, device: str = "cpu", use_mock_model: bool = True, **kwargs) -> bool:
        """Initialize the mock plugin."""
        self.device = device
        self.model_path = kwargs.get("model_path")
        self.is_loaded = True
        return True
        
    def load_model(self):
        """Load the mock model."""
        self.is_loaded = True
        return self
        
    def infer(self, input_ids: torch.Tensor):
        """Mock inference function."""
        # Return a tensor with the same batch size but random outputs
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        return torch.randn(batch_size, seq_len, 1000)  # 1000 vocab size
        
    def generate_text(self, prompt: str, max_new_tokens: int = 10) -> str:
        """Mock text generation."""
        return f"Generated text for: {prompt}"
        
    def cleanup(self):
        """Clean up the mock plugin."""
        self.is_loaded = False


class BenchmarkTestFixture:
    """Base class for benchmark test fixtures."""
    
    def __init__(self):
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.mock_plugin = None
        
    def setup(self) -> None:
        """Setup the test fixture."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock plugin
        self.mock_plugin = MockModelPlugin()
        
    def teardown(self) -> None:
        """Teardown the test fixture."""
        if self.temp_dir:
            self.temp_dir.cleanup()
            
    def get_temp_dir(self) -> Path:
        """Get the temporary directory path."""
        if not self.temp_dir:
            raise RuntimeError("Fixture not initialized. Call setup() first.")
        return Path(self.temp_dir.name)
        
    def get_mock_plugin(self):
        """Get the mock plugin."""
        if not self.mock_plugin:
            raise RuntimeError("Fixture not initialized. Call setup() first.")
        return self.mock_plugin


class PerformanceBenchmarkFixture(BenchmarkTestFixture):
    """Fixture for performance benchmarks."""
    
    def __init__(self, input_lengths: Optional[list] = None, batch_sizes: Optional[list] = None):
        super().__init__()
        self.input_lengths = input_lengths or [20, 50, 100]
        self.batch_sizes = batch_sizes or [1, 2, 4, 8]
        
    def get_test_inputs(self) -> Generator[torch.Tensor, None, None]:
        """Generate test inputs for different configurations."""
        for input_length in self.input_lengths:
            for batch_size in self.batch_sizes:
                yield torch.randint(0, 1000, (batch_size, input_length))


class AccuracyBenchmarkFixture(BenchmarkTestFixture):
    """Fixture for accuracy benchmarks."""
    
    def __init__(self):
        super().__init__()
        self.test_prompts = [
            "The capital of France is",
            "What is 2+2?",
            "Who wrote Romeo and Juliet?",
            "The largest planet in our solar system is",
            "Python is a programming language"
        ]
        self.expected_answers = [
            "Paris",
            "4",
            "Shakespeare",
            "Jupiter",
            "true"
        ]
        
    def get_test_cases(self) -> Generator[tuple, None, None]:
        """Generate test cases for accuracy evaluation."""
        for prompt, expected in zip(self.test_prompts, self.expected_answers):
            yield prompt, expected


class MemoryBenchmarkFixture(BenchmarkTestFixture):
    """Fixture for memory benchmarks."""
    
    def __init__(self):
        super().__init__()
        self.memory_scenarios = [
            {"input_length": 50, "batch_size": 1},
            {"input_length": 100, "batch_size": 2},
            {"input_length": 200, "batch_size": 4}
        ]
        
    def get_memory_test_scenarios(self) -> Generator[Dict[str, Any], None, None]:
        """Generate memory test scenarios."""
        for scenario in self.memory_scenarios:
            yield scenario


# Context managers for easy fixture usage
from contextlib import contextmanager


@contextmanager
def performance_benchmark_fixture(input_lengths=None, batch_sizes=None):
    """Context manager for performance benchmark fixture."""
    fixture = PerformanceBenchmarkFixture(input_lengths, batch_sizes)
    try:
        fixture.setup()
        yield fixture
    finally:
        fixture.teardown()


@contextmanager
def accuracy_benchmark_fixture():
    """Context manager for accuracy benchmark fixture."""
    fixture = AccuracyBenchmarkFixture()
    try:
        fixture.setup()
        yield fixture
    finally:
        fixture.teardown()


@contextmanager
def memory_benchmark_fixture():
    """Context manager for memory benchmark fixture."""
    fixture = MemoryBenchmarkFixture()
    try:
        fixture.setup()
        yield fixture
    finally:
        fixture.teardown()


# Factory function to create appropriate fixtures
def create_benchmark_fixture(fixture_type: str, **kwargs):
    """
    Factory function to create benchmark fixtures.
    
    Args:
        fixture_type: Type of fixture to create ('performance', 'accuracy', 'memory')
        **kwargs: Additional arguments for fixture initialization
    
    Returns:
        Appropriate benchmark fixture instance
    """
    if fixture_type == 'performance':
        return PerformanceBenchmarkFixture(**kwargs)
    elif fixture_type == 'accuracy':
        return AccuracyBenchmarkFixture()
    elif fixture_type == 'memory':
        return MemoryBenchmarkFixture()
    else:
        raise ValueError(f"Unknown fixture type: {fixture_type}")


# Common utility functions for benchmark testing
def simulate_model_loading_time(min_time: float = 0.1, max_time: float = 1.0) -> float:
    """
    Simulate realistic model loading times.
    
    Args:
        min_time: Minimum loading time in seconds
        max_time: Maximum loading time in seconds
    
    Returns:
        Simulated loading time
    """
    return np.random.uniform(min_time, max_time)


def generate_realistic_inference_times(
    input_length: int, 
    batch_size: int, 
    base_time_per_token: float = 0.001
) -> float:
    """
    Generate realistic inference times based on input characteristics.
    
    Args:
        input_length: Length of input sequence
        batch_size: Batch size
        base_time_per_token: Base time per token in seconds
    
    Returns:
        Estimated inference time
    """
    # More complex simulation considering batch effects
    base_time = input_length * batch_size * base_time_per_token
    # Add some variance based on batch size efficiency
    batch_efficiency_factor = 1.0 - (batch_size - 1) * 0.05  # Efficiency improves with batching
    return base_time * batch_efficiency_factor


def measure_memory_usage_mb() -> float:
    """
    Measure current memory usage in MB.
    
    Returns:
        Current memory usage in MB
    """
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024
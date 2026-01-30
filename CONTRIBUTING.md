# Contributing to Inference-PIO

We welcome contributions to the Inference-PIO project! This document provides comprehensive guidelines for contributing to the project, including development setup, coding standards, testing procedures, and more.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Project Structure](#project-structure)
4. [Coding Standards](#coding-standards)
5. [Testing Framework](#testing-framework)
6. [Benchmark System](#benchmark-system)
7. [Performance Regression Testing](#performance-regression-testing)
8. [Test Optimization System](#test-optimization-system)
9. [Unified Test Discovery](#unified-test-discovery)
10. [Submitting Changes](#submitting-changes)
11. [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before contributing to Inference-PIO, ensure you have:

- Python 3.8 or higher
- Git version control system
- pip package manager
- CUDA-compatible GPU (optional, for GPU-specific features)

### Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/inference-pio.git`
3. Navigate to the project directory: `cd inference-pio`
4. Install in development mode: `pip install -e .`
5. Install development dependencies: `pip install -e ".[dev]"`

## Development Setup

### Environment Setup

Create a virtual environment for development:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality:

```bash
pre-commit install
```

## Project Structure

The Inference-PIO project follows a modular architecture with standardized directory structures:

```
src/
└── inference_pio/
    ├── models/
    │   └── {model_name}/
    │       ├── __init__.py
    │       ├── plugin.py
    │       ├── config.py
    │       ├── tests/
    │       │   ├── unit/
    │       │   ├── integration/
    │       │   └── performance/
    │       └── benchmarks/
    │           ├── unit/
    │           ├── integration/
    │           └── performance/
    ├── plugin_system/
    │   ├── __init__.py
    │   ├── base_plugin.py
    │   ├── registry.py
    │   ├── tests/
    │   │   ├── unit/
    │   │   ├── integration/
    │   │   └── performance/
    │   └── benchmarks/
    │       ├── unit/
    │       ├── integration/
    │       └── performance/
    ├── common/
    │   ├── __init__.py
    │   ├── test_utils.py
    │   ├── test_discovery.py
    │   ├── unified_test_discovery.py
    │   ├── performance_regression_tests.py
    │   ├── tests/
    │   │   ├── unit/
    │   │   ├── integration/
    │   │   └── performance/
    │   └── benchmarks/
    │       ├── unit/
    │       ├── integration/
    │       └── performance/
    └── __init__.py
tests/
├── unit/
├── integration/
└── performance/
```

### Model-Specific Directories

Each model plugin has its own self-contained directory with all necessary components:

- `plugin.py`: Main plugin implementation
- `config.py`: Model-specific configuration
- `tests/`: Model-specific tests organized by type
- `benchmarks/`: Model-specific benchmarks organized by type

## Coding Standards

### General Guidelines

We follow PEP 8 guidelines for Python code with additional project-specific standards:

- Use type hints for all function parameters and return values
- Write docstrings for all classes and functions using Google-style format
- Keep functions focused and small (preferably under 50 lines)
- Use meaningful variable and function names
- Follow the DRY (Don't Repeat Yourself) principle
- Use constants for magic numbers and strings
- Prefer composition over inheritance
- Write immutable data structures when possible

### Type Hints

All functions must have type hints:

```python
from typing import List, Dict, Optional, Union, Any
import torch

def process_tensor(data: torch.Tensor, 
                  config: Dict[str, Any], 
                  batch_size: Optional[int] = None) -> List[torch.Tensor]:
    """Process tensor data with the given configuration.
    
    Args:
        data: Input tensor to process
        config: Configuration dictionary
        batch_size: Optional batch size override
        
    Returns:
        List of processed tensors
    """
    # Implementation here
    pass
```

### Error Handling

Use specific exception types and provide meaningful error messages:

```python
def validate_input(input_data: Any) -> None:
    """Validate input data and raise appropriate exceptions."""
    if input_data is None:
        raise ValueError("Input data cannot be None")
    
    if not isinstance(input_data, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(input_data)}")
    
    if input_data.dim() == 0:
        raise ValueError("Input tensor cannot be scalar")
```

## Testing Framework

### Custom Testing Framework

Inference-PIO uses a custom-built testing framework designed for efficiency and consistency. The framework is located in `tests.utils.test_utils.py` and provides comprehensive assertion functions.

### Assertion Functions

The framework includes over 100 assertion functions for various testing scenarios:

#### Basic Assertions
- `assert_true(condition, message)`
- `assert_false(condition, message)`
- `assert_equal(actual, expected, message)`
- `assert_not_equal(actual, expected, message)`

#### Container Assertions
- `assert_in(item, container, message)`
- `assert_not_in(item, container, message)`
- `assert_length(container, expected_length, message)`
- `assert_items_equal(actual, expected, message)`

#### Numeric Assertions
- `assert_greater(value, comparison, message)`
- `assert_less(value, comparison, message)`
- `assert_between(value, lower_bound, upper_bound, message)`
- `assert_close(value, expected, rel_tol=1e-09, abs_tol=0.0, message)`

#### Tensor Assertions
- `assert_tensor_equal(tensor1, tensor2, message)`
- `assert_tensor_close(tensor1, tensor2, rtol=1e-05, atol=1e-08, message)`
- `assert_tensor_shape(tensor, expected_shape, message)`
- `assert_tensor_dtype(tensor, expected_dtype, message)`

#### File System Assertions
- `assert_file_exists(file_path, message)`
- `assert_dir_exists(dir_path, message)`
- `assert_readable(path, message)`
- `assert_writable(path, message)`

### Writing Tests

#### Basic Test Function

```python
from tests.utils.test_utils import assert_equal, assert_true, run_tests

def test_basic_functionality():
    """Test basic arithmetic operations."""
    result = 2 + 2
    assert_equal(result, 4, "Addition should work correctly")
    assert_true(result > 0, "Result should be positive")

if __name__ == '__main__':
    run_tests([test_basic_functionality])
```

#### Class-Based Tests

```python
from tests.utils.test_utils import assert_equal

class TestCalculator:
    def test_addition(self):
        result = 5 + 3
        assert_equal(result, 8, "Addition should work correctly")

    def test_subtraction(self):
        result = 10 - 4
        assert_equal(result, 6, "Subtraction should work correctly")
```

#### Tensor Tests

```python
import torch
from tests.utils.test_utils import assert_tensor_equal, assert_tensor_shape

def test_tensor_operations():
    """Test tensor operations."""
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([1, 2, 3])
    
    assert_tensor_equal(tensor1, tensor2, "Tensors should be equal")
    assert_tensor_shape(tensor1, (3,), "Tensor should have shape (3,)")
```

### Running Tests

#### Individual Test Execution

```python
from tests.utils.test_utils import run_test

def test_example():
    assert 1 + 1 == 2

run_test(test_example, "test_example")
```

#### Multiple Test Execution

```python
from tests.utils.test_utils import run_tests

def test_one():
    assert True

def test_two():
    assert 1 == 1

run_tests([test_one, test_two])
```

## Benchmark System

### Standardized Benchmark Structure

The benchmark system provides a consistent interface for evaluating model performance across multiple dimensions:

- Performance (speed, memory usage, throughput)
- Accuracy (correctness of outputs)
- Resource utilization (CPU, memory, loading time)

### Benchmark Categories

#### Performance Benchmarks
- `InferenceSpeedBenchmark`: Measures tokens per second for various input lengths
- `MemoryUsageBenchmark`: Measures memory consumption during model operations
- `BatchProcessingBenchmark`: Evaluates throughput with different batch sizes
- `ModelLoadingTimeBenchmark`: Times how long it takes to load a model

#### Accuracy Benchmarks
- `AccuracyBenchmark`: Evaluates correctness of model outputs using known facts

### Implementing Benchmarks

To implement benchmarks for a new model:

1. Ensure your model plugin implements the `ModelPluginInterface`
2. Create benchmark files in the standard directory structure:

```
src/inference_pio/models/{model_name}/benchmarks/
├── unit/
│   └── benchmark_accuracy.py
├── integration/
│   └── benchmark_comparison.py
└── performance/
    └── benchmark_inference_speed.py
```

3. Use the standardized benchmark classes from the framework

### Standard Benchmark Class Structure

```python
from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult

class MyModelBenchmark(BaseBenchmark):
    def run(self) -> BenchmarkResult:
        # Implementation here
        return BenchmarkResult(
            name="my_metric",
            value=123.45,
            unit="units",
            metadata={"param_used": "value"},
            model_name=self.model_name,
            category="performance"
        )
```

### Running Benchmarks

#### Single Model
```bash
python -m pytest src/inference_pio/models/{model_name}/benchmarks/ -v
```

#### All Models
```python
from benchmarks.standardized_runner import run_standardized_benchmarks

# Run all benchmarks
results = run_standardized_benchmarks()
```

## Performance Regression Testing

### Overview

The performance regression testing system tracks performance metrics over time, detects performance regressions, and provides alerts when performance degrades. The system integrates with the existing test infrastructure and CI/CD pipeline.

### Components

1. **Performance Regression Tracker**: Manages historical performance data and detects regressions
2. **Performance Regression Tests**: Framework for writing performance regression tests
3. **Pytest Plugin**: Enables performance regression testing within pytest
4. **CI/CD Integration**: Scripts for integrating with CI/CD pipelines

### Usage

#### In Test Cases

```python
from src.inference_pio.common.performance_regression_tests import PerformanceRegressionTestCase

class MyModelPerformanceTest(PerformanceRegressionTestCase):
    def test_inference_speed(self):
        # Your test code here
        self.record_performance_metric(
            name="inference_speed",
            value=150.0,
            unit="tokens/sec",
            model_name="my_model",
            category="performance"
        )

        # Assert no regression
        self.assert_no_performance_regression("my_model", "inference_speed")
```

#### Configuration

The system can be configured using the `performance_regression_config.ini` file:

```ini
[performance_regression]
regression_threshold = 5.0
storage_dir = performance_history
reports_dir = performance_reports
fail_on_regression = true
```

### Metrics Categories

- **Performance**: Metrics where higher values are better (e.g., tokens/sec)
- **Time**: Metrics where lower values are better (e.g., seconds)
- **Memory**: Metrics where lower values are better (e.g., MB)

## Test Optimization System

### Features

- **Parallel Execution**: Run tests concurrently using multiple processes or threads
- **Result Caching**: Cache test results to avoid redundant execution
- **Flexible Configuration**: Control caching and parallelization settings
- **Performance Monitoring**: Track execution times and optimization effectiveness

### Usage

#### Using the Optimized Test Runner

```bash
# Run tests with default optimization (parallel + caching)
python optimized_test_runner.py

# Run tests without parallelization
python optimized_test_runner.py --no-parallel

# Run tests without caching
python optimized_test_runner.py --no-cache

# Specify a custom test directory
python optimized_test_runner.py --directory ./my_tests

# Generate a JSON report
python optimized_test_runner.py --report results.json
```

#### Using the Python API

```python
from test_optimization import run_tests_with_optimization

# Define your test functions
def test_example():
    assert 1 + 1 == 2

def test_another():
    assert "hello".upper() == "HELLO"

# Run with optimization
results = run_tests_with_optimization(
    test_functions=[test_example, test_another],
    test_paths=["test_example", "test_another"],
    cache_enabled=True,
    parallel_enabled=True,
    max_workers=4
)

print(f"Passed: {results['passed']}/{results['total_tests']}")
```

### Configuration Options

#### Parallel Execution
- `max_workers`: Number of parallel workers (defaults to CPU count)
- `use_processes`: Use processes (True) or threads (False) for parallelization

#### Caching
- `cache_enabled`: Enable/disable result caching
- `cache_dir`: Custom directory for cache files (defaults to `.test_cache`)
- Cache expiration: Results are cached for 24 hours

## Unified Test Discovery

### Overview

The Unified Test Discovery System is a comprehensive solution for discovering, organizing, and running both tests and benchmarks across the Inference-PIO project. It consolidates multiple existing discovery mechanisms into a single, efficient system that supports the standardized directory structure and naming conventions.

### Key Features

1. **Comprehensive Discovery**: Discovers both traditional tests and benchmark functions
2. **Flexible Naming Conventions**: Supports multiple naming conventions for tests and benchmarks
3. **Standardized Directory Structure Support**: Works with the project's directory structure

### Naming Conventions

#### Test Function Patterns:
- `test_*` - Standard test functions
- `should_*` - Behavior-driven test naming
- `when_*` - Scenario-based test naming
- `verify_*` - Verification-based test naming
- `validate_*` - Validation-based test naming
- `check_*` - Check-based test naming

#### Benchmark Function Patterns:
- `run_*` - Standard run functions
- `benchmark_*` - Standard benchmark functions
- `perf_*` - Performance-related functions
- `measure_*` - Measurement-related functions
- `profile_*` - Profiling-related functions
- `time_*` - Timing-related functions
- `speed_*` - Speed-related functions
- `stress_*` - Stress testing functions
- `load_*` - Load testing functions

### Usage Examples

#### Basic Discovery
```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

# Create a discovery instance
discovery = UnifiedTestDiscovery()

# Discover all items
items = discovery.discover_all()

print(f"Discovered {len(items)} total items")
print(f"Tests: {len(discovery.test_functions)}")
print(f"Benchmarks: {len(discovery.benchmark_functions)}")
```

#### Model-Specific Discovery
```python
from inference_pio.unified_test_discovery import discover_tests_for_model

# Discover tests for a specific model
model_tests = discover_tests_for_model('qwen3_vl_2b')
print(f"Found {len(model_tests)} tests for qwen3_vl_2b model")
```

#### Running Tests and Benchmarks
```python
from inference_pio.unified_test_discovery import discover_and_run_tests_only

# Discover and run only tests
results = discover_and_run_tests_only()
```

## Submitting Changes

### Pull Request Process

1. Ensure all tests pass before submitting
2. Update documentation as needed
3. Add tests for new functionality
4. Follow the coding standards
5. Write clear, descriptive commit messages
6. Submit a pull request with a detailed description

### Commit Message Format

Use the conventional commits format:

```
feat: Add new model support for Qwen3-VL-2B
fix: Resolve memory leak in tensor processing
docs: Update contribution guidelines
style: Format code according to PEP 8
refactor: Simplify tensor operation implementation
test: Add unit tests for new functionality
chore: Update dependencies
```

### Code Review Process

- At least one maintainer review is required
- Address all feedback before merging
- Ensure CI checks pass
- Maintain backward compatibility when possible

## Community Guidelines

### Code of Conduct

Please follow our Code of Conduct in all interactions:

- Be respectful and inclusive
- Provide constructive feedback
- Welcome newcomers
- Focus on technical merit

### Getting Help

- Open an issue for bugs or feature requests
- Join our community discussions
- Check the documentation first
- Be patient with responses

### Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment information (Python version, OS, etc.)
- Relevant logs or error messages

---

Thank you for contributing to Inference-PIO! Your efforts help make this project better for everyone.
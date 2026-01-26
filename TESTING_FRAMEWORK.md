# Inference-PIO Testing Framework Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Assertion Functions](#assertion-functions)
4. [Test Structure](#test-structure)
5. [Writing Tests](#writing-tests)
6. [Running Tests](#running-tests)
7. [Best Practices](#best-practices)
8. [Test Categories](#test-categories)
9. [Advanced Features](#advanced-features)
10. [Integration with Other Systems](#integration-with-other-systems)

## Overview

The Inference-PIO project utilizes a comprehensive, custom-built testing framework designed to provide a lightweight, efficient, and consistent testing experience across all components. This framework eliminates dependencies on external testing libraries while providing extensive assertion capabilities and advanced testing features.

The framework includes:
- Over 100 specialized assertion functions
- Support for tensor operations testing
- File system assertions
- Performance regression tracking
- Parallel execution and caching
- Unified test discovery system

## Core Components

### Test Utilities (`src/inference_pio/test_utils.py`)

The core of the testing framework consists of custom assertion functions and test runners:

#### Assertion Functions

The framework provides comprehensive assertion functions for various testing scenarios:

**Basic Assertions:**
- `assert_true(condition, message)` - Asserts that a condition evaluates to True
- `assert_false(condition, message)` - Asserts that a condition evaluates to False
- `assert_equal(actual, expected, message)` - Asserts that two values are equal
- `assert_not_equal(actual, expected, message)` - Asserts that two values are not equal
- `assert_is_none(value, message)` - Asserts that a value is None
- `assert_is_not_none(value, message)` - Asserts that a value is not None

**Container Assertions:**
- `assert_in(item, container, message)` - Asserts that an item is in a container
- `assert_not_in(item, container, message)` - Asserts that an item is not in a container
- `assert_length(container, expected_length, message)` - Asserts container length
- `assert_items_equal(actual, expected, message)` - Asserts items in containers are equal (order-independent)

**Numeric Assertions:**
- `assert_greater(value, comparison, message)` - Asserts that value is greater than comparison
- `assert_less(value, comparison, message)` - Asserts that value is less than comparison
- `assert_between(value, lower_bound, upper_bound, message)` - Asserts value is between bounds
- `assert_close(value, expected, rel_tol=1e-09, abs_tol=0.0, message)` - Asserts values are close within tolerance

**Tensor Assertions:**
- `assert_tensor_equal(tensor1, tensor2, message)` - Asserts that two tensors are equal
- `assert_tensor_close(tensor1, tensor2, rtol=1e-05, atol=1e-08, message)` - Asserts that two tensors are close within tolerance
- `assert_tensor_shape(tensor, expected_shape, message)` - Asserts that tensor has the expected shape
- `assert_tensor_dtype(tensor, expected_dtype, message)` - Asserts that tensor has the expected dtype

**File System Assertions:**
- `assert_file_exists(file_path, message)` - Asserts that a file exists
- `assert_dir_exists(dir_path, message)` - Asserts that a directory exists
- `assert_readable(path, message)` - Asserts that path is readable
- `assert_writable(path, message)` - Asserts that path is writable

**Exception Assertions:**
- `assert_raises(exception_type, callable_func, *args, **kwargs)` - Asserts that calling a function raises an exception

#### Test Runner Functions

- `run_test(test_func, test_name)` - Runs a single test function
- `run_tests(test_functions)` - Runs multiple test functions and provides a summary

#### Special Functions

- `skip_test(reason)` - Allows skipping tests with a reason
- `SkipTestException` - Exception raised when a test is intentionally skipped

### Test Discovery System (`src/inference_pio/test_discovery.py`)

The test discovery system automatically finds and executes tests across the project:
- Discovers test functions that start with `test_` in Python files
- Supports both standalone functions and class methods
- Can discover tests for specific models or the entire project
- Provides summary statistics about test coverage

### Unified Test Discovery System (`src/inference_pio/unified_test_discovery.py`)

The unified discovery system extends the basic discovery with:
- Support for both tests and benchmarks
- Flexible naming conventions
- Standardized directory structure support
- Advanced filtering capabilities

## Assertion Functions

The framework includes over 100 assertion functions organized by category.
(See code for full list)

## Test Structure

Tests follow a standardized directory structure organized by type:

```
src/
└── inference_pio/
    ├── models/
    │   └── {model_name}/
    │       └── tests/
    │           ├── unit/
    │           ├── integration/
    │           └── performance/
    ├── plugin_system/
    │   └── tests/
    │       ├── unit/
    │       ├── integration/
    │       └── performance/
    └── tests/
        ├── unit/
        ├── integration/
        └── performance/
```

## Writing Tests

### Basic Test Function

```python
from src.inference_pio.test_utils import assert_equal, assert_true, run_tests

def test_basic_functionality():
    """Test basic arithmetic operations."""
    result = 2 + 2
    assert_equal(result, 4, "Addition should work correctly")
    assert_true(result > 0, "Result should be positive")

if __name__ == '__main__':
    run_tests([test_basic_functionality])
```

## Running Tests

### Using Optimized Test Runner

The primary way to run tests is via the `optimized_test_runner.py` script:

```bash
# Run all tests in 'tests' directory
python optimized_test_runner.py

# List available tests
python optimized_test_runner.py --list
```

### Individual Model Tests

```python
from src.inference_pio.test_discovery import run_model_tests

# Run all tests for a specific model
run_model_tests('qwen3_vl_2b')
```

### Manual Test Execution

```python
from src.inference_pio.test_utils import run_tests

def test_example():
    assert_equal(1 + 1, 2, "Basic addition should work")

if __name__ == '__main__':
    run_tests([test_example])
```

## Best Practices

### 1. Descriptive Test Names
Use clear, descriptive names that indicate what is being tested:
```python
def test_model_initialization_with_valid_config():
    """Test that model initializes correctly with valid configuration."""
    # Implementation here
```

## Advanced Features

### Performance Regression Testing

The framework includes performance regression testing capabilities:

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

### Test Optimization

The framework supports parallel execution and caching:

```python
from test_optimization import run_tests_with_optimization

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
```

## Integration with Other Systems

### CI/CD Integration

The testing framework integrates seamlessly with CI/CD pipelines:

```bash
# Run all tests in CI (using custom runner)
python optimized_test_runner.py --no-cache
```

### Reporting

The framework generates detailed reports:

```python
from inference_pio.unified_test_discovery import get_discovery_summary

# Get a summary of all discovered items
summary = get_discovery_summary()
print(f"Total tests: {summary['total_tests']}")
print(f"Models covered: {list(summary['by_model'].keys())}")
```

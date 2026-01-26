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

The framework includes over 100 assertion functions organized by category:

### Basic Assertions
```python
from src.inference_pio.test_utils import (
    assert_true, assert_false, assert_equal, assert_not_equal,
    assert_is_none, assert_is_not_none
)

def test_basic_assertions():
    assert_true(5 > 3)
    assert_false(5 < 3)
    assert_equal(2 + 2, 4)
    assert_not_equal(2 + 2, 5)
    assert_is_none(None)
    assert_is_not_none("not none")
```

### Container Assertions
```python
from src.inference_pio.test_utils import (
    assert_in, assert_not_in, assert_length, assert_items_equal
)

def test_container_assertions():
    assert_in("a", ["a", "b", "c"])
    assert_not_in("d", ["a", "b", "c"])
    assert_length([1, 2, 3], 3)
    assert_items_equal([1, 2, 3], [3, 1, 2])  # Order-independent
```

### Numeric Assertions
```python
from src.inference_pio.test_utils import (
    assert_greater, assert_less, assert_between, assert_close
)

def test_numeric_assertions():
    assert_greater(10, 5)
    assert_less(3, 7)
    assert_between(5, 1, 10)
    assert_close(1.0000001, 1.0, rel_tol=1e-06)
```

### Tensor Assertions
```python
import torch
from src.inference_pio.test_utils import (
    assert_tensor_equal, assert_tensor_close, assert_tensor_shape, assert_tensor_dtype
)

def test_tensor_assertions():
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([1.0, 2.0, 3.0])
    
    assert_tensor_equal(tensor1, tensor2)
    assert_tensor_shape(tensor1, (3,))
    assert_tensor_dtype(tensor1, torch.float32)
    
    tensor3 = torch.tensor([1.0001, 2.0001, 3.0001])
    assert_tensor_close(tensor1, tensor3, rtol=1e-03)
```

### File System Assertions
```python
from src.inference_pio.test_utils import (
    assert_file_exists, assert_dir_exists, assert_readable, assert_writable
)

def test_file_system_assertions():
    assert_file_exists("/path/to/existing/file.txt")
    assert_dir_exists("/path/to/existing/directory")
    assert_readable("/path/to/readable/file.txt")
    assert_writable("/path/to/writable/file.txt")
```

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

### Class-Based Tests

```python
from src.inference_pio.test_utils import assert_equal

class TestCalculator:
    def test_addition(self):
        result = 5 + 3
        assert_equal(result, 8, "Addition should work correctly")

    def test_subtraction(self):
        result = 10 - 4
        assert_equal(result, 6, "Subtraction should work correctly")

# The discovery system will automatically wrap class methods
```

### Using Different Assertions

```python
from src.inference_pio.test_utils import (
    assert_equal, assert_true, assert_false,
    assert_in, assert_is_none, assert_greater,
    assert_raises, run_tests
)

def test_various_assertions():
    # Equality assertions
    assert_equal(1 + 1, 2, "Basic addition")
    assert_equal("hello", "hello", "String equality")

    # Boolean assertions
    assert_true(5 > 3, "Greater than comparison")
    assert_false(5 < 3, "Less than comparison")

    # Container assertions
    assert_in("a", ["a", "b", "c"], "Item in list")
    assert_in("key", {"key": "value"}, "Key in dictionary")

    # Null assertions
    assert_is_none(None, "None value")
    assert_is_not_none("not none", "Non-none value")

    # Numeric assertions
    assert_greater(10, 5, "Greater than")
    assert_less(3, 7, "Less than")

    # Exception assertions
    def raises_error():
        raise ValueError("Test error")

    assert_raises(ValueError, raises_error)

if __name__ == '__main__':
    run_tests([test_various_assertions])
```

### Tensor-Specific Tests

```python
import torch
from src.inference_pio.test_utils import (
    assert_tensor_equal, assert_tensor_shape, assert_tensor_dtype
)

def test_tensor_operations():
    """Test tensor operations."""
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([1, 2, 3])
    
    # Test tensor equality
    assert_tensor_equal(tensor1, tensor2, "Tensors should be equal")
    
    # Test tensor shape
    assert_tensor_shape(tensor1, (3,), "Tensor should have shape (3,)")
    
    # Test tensor dtype
    assert_tensor_dtype(tensor1, torch.int64, "Tensor should have int64 dtype")

def test_tensor_with_tolerance():
    """Test tensor operations with tolerance."""
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([1.0001, 2.0001, 3.0001])
    
    # Test tensor closeness with tolerance
    from src.inference_pio.test_utils import assert_tensor_close
    assert_tensor_close(tensor1, tensor2, rtol=1e-03, 
                       message="Tensors should be close within tolerance")
```

### Skipping Tests

```python
from src.inference_pio.test_utils import skip_test, run_tests

def test_feature_that_requires_gpu():
    # Check if GPU is available
    import torch
    if not torch.cuda.is_available():
        skip_test("GPU not available for this test")

    # Test code here...
    assert torch.cuda.is_available(), "CUDA should be available"

def test_conditional_skip():
    import sys
    if sys.version_info < (3, 8):
        skip_test("Requires Python 3.8 or higher")
    
    # Test code here...
    assert sys.version_info >= (3, 8), "Python version should be 3.8+"
```

### File System Tests

```python
from src.inference_pio.test_utils import (
    assert_file_exists, assert_dir_exists, assert_readable
)

def test_file_operations():
    """Test file system operations."""
    # Test file existence
    assert_file_exists("requirements.txt", "Requirements file should exist")
    
    # Test directory existence
    assert_dir_exists("src", "Source directory should exist")
    
    # Test file readability
    assert_readable("README.md", "README should be readable")
```

## Running Tests

### Individual Model Tests

```python
from src.inference_pio.test_discovery import run_model_tests

# Run all tests for a specific model
run_model_tests('qwen3_vl_2b')
```

### All Project Tests

```python
from src.inference_pio.test_discovery import run_all_project_tests

# Run all tests in the project
run_all_project_tests()
```

### Manual Test Execution

```python
from src.inference_pio.test_utils import run_tests

def test_example():
    assert_equal(1 + 1, 2, "Basic addition should work")

if __name__ == '__main__':
    run_tests([test_example])
```

### Using the Unified Discovery System

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

# Create a discovery instance
discovery = UnifiedTestDiscovery()

# Discover all items
items = discovery.discover_all()

# Run all discovered tests
results = discovery.run_tests_only()

# Run all discovered benchmarks
benchmark_results = discovery.run_benchmarks_only()
```

### Model-Specific Test Execution

```python
from inference_pio.unified_test_discovery import run_tests_for_model

# Run tests for a specific model
results = run_tests_for_model('qwen3_vl_2b')
```

## Best Practices

### 1. Descriptive Test Names
Use clear, descriptive names that indicate what is being tested:
```python
def test_model_initialization_with_valid_config():
    """Test that model initializes correctly with valid configuration."""
    # Implementation here
```

### 2. Single Responsibility
Each test should focus on a single aspect of functionality:
```python
def test_tensor_addition():
    """Test tensor addition operation."""
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6])
    result = tensor1 + tensor2
    expected = torch.tensor([5, 7, 9])
    assert_tensor_equal(result, expected)

def test_tensor_multiplication():
    """Test tensor multiplication operation."""
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([2, 3, 4])
    result = tensor1 * tensor2
    expected = torch.tensor([2, 6, 12])
    assert_tensor_equal(result, expected)
```

### 3. Meaningful Messages
Provide clear messages in assertion calls:
```python
def test_model_output_shape():
    """Test that model output has expected shape."""
    model = create_model()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    expected_shape = (1, 1000)
    assert_tensor_shape(output, expected_shape, 
                       f"Model output shape should be {expected_shape}, got {output.shape}")
```

### 4. Isolation
Tests should not depend on each other's execution order:
```python
def test_first():
    # This test should not affect test_second
    pass

def test_second():
    # This test should not depend on test_first
    pass
```

### 5. Consistent Structure
Follow the standardized directory structure:
```
tests/
├── unit/
├── integration/
└── performance/
```

### 6. Comprehensive Coverage
Include unit, integration, and performance tests as appropriate:
- Unit tests: Test individual functions, methods, or classes in isolation
- Integration tests: Test interactions between multiple components
- Performance tests: Measure execution time and resource usage

## Test Categories

### Unit Tests
- Test individual functions, methods, or classes in isolation
- Fast execution with minimal external dependencies
- Focus on logic correctness
- Located in `tests/unit/` directories

### Integration Tests
- Test interactions between multiple components
- May involve real external dependencies
- Validate end-to-end workflows
- Located in `tests/integration/` directories

### Performance Tests
- Measure execution time and resource usage
- Test scalability under load
- Monitor performance regressions
- Located in `tests/performance/` directories

### Benchmark Tests
- Evaluate model performance across multiple dimensions
- Compare against baseline metrics
- Track performance over time
- Located in `benchmarks/` directories

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

### Custom Assertion Functions

You can create custom assertion functions that integrate with the framework:

```python
from src.inference_pio.test_utils import assert_true

def assert_tensor_values_in_range(tensor, min_val, max_val, message="Tensor values not in range"):
    """Custom assertion to check if all tensor values are within a range."""
    in_range = (tensor >= min_val) & (tensor <= max_val)
    assert_true(in_range.all().item(), 
               f"{message}: tensor values should be in range [{min_val}, {max_val}], "
               f"but found values outside this range")
```

## Integration with Other Systems

### CI/CD Integration

The testing framework integrates seamlessly with CI/CD pipelines:

```bash
# Run all tests in CI
python -m pytest tests/ --junit-xml=test-results.xml

# Run performance regression tests
python scripts/ci_performance_tests.py --threshold 5.0 --fail-on-regression
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

### Configuration

The system can be configured using configuration files:

```ini
# performance_regression_config.ini
[performance_regression]
regression_threshold = 5.0
storage_dir = performance_history
reports_dir = performance_reports
fail_on_regression = true
```

---

This comprehensive testing framework provides all the tools needed to write robust, reliable tests for the Inference-PIO project. The extensive assertion library, combined with advanced features like performance regression tracking and test optimization, ensures that the project maintains high quality and performance standards.
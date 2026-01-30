# Inference-PIO Testing Framework Guide

## Overview

The Inference-PIO project uses a custom-built testing framework designed to provide a lightweight, efficient, and consistent testing experience across all components. This framework eliminates dependencies on external testing libraries while providing essential assertion and test execution capabilities.

## Core Components

### Test Utilities (`tests.utils.test_utils.py`)

The core of the testing framework consists of custom assertion functions and test runners:

#### Assertion Functions

- `assert_true(condition, message)` - Asserts that a condition evaluates to True
- `assert_false(condition, message)` - Asserts that a condition evaluates to False
- `assert_equal(actual, expected, message)` - Asserts that two values are equal
- `assert_not_equal(actual, expected, message)` - Asserts that two values are not equal
- `assert_is_none(value, message)` - Asserts that a value is None
- `assert_is_not_none(value, message)` - Asserts that a value is not None
- `assert_in(item, container, message)` - Asserts that an item is in a container
- `assert_not_in(item, container, message)` - Asserts that an item is not in a container
- `assert_greater(value, comparison, message)` - Asserts that value is greater than comparison
- `assert_less(value, comparison, message)` - Asserts that value is less than comparison
- `assert_is_instance(obj, expected_class, message)` - Asserts that an object is an instance of a class
- `assert_raises(exception_type, callable_func, *args, **kwargs)` - Asserts that calling a function raises an exception

#### Test Runner Functions

- `run_test(test_func, test_name)` - Runs a single test function
- `run_tests(test_functions)` - Runs multiple test functions and provides a summary

#### Special Functions

- `skip_test(reason)` - Allows skipping tests with a reason
- `SkipTestException` - Exception raised when a test is intentionally skipped

## Standardized Test Structure

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

### Unit Tests
- Test individual functions, methods, or classes in isolation
- Fast execution with minimal external dependencies
- Focus on logic correctness

### Integration Tests
- Test interactions between multiple components
- May involve real external dependencies
- Validate end-to-end workflows

### Performance Tests
- Measure execution time and resource usage
- Test scalability under load
- Monitor performance regressions

## Writing Tests

### Basic Test Function

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

### Class-Based Tests

```python
from tests.utils.test_utils import assert_equal

class TestCalculator:
    def test_addition(self):
        result = 5 + 3
        assert_equal(result, 8)

    def test_subtraction(self):
        result = 10 - 4
        assert_equal(result, 6)

# The discovery system will automatically wrap class methods
```

### Using Different Assertions

```python
from tests.utils.test_utils import (
    assert_equal, assert_true, assert_false,
    assert_in, assert_is_none, assert_greater,
    assert_raises, run_tests
)

def test_various_assertions():
    # Equality assertions
    assert_equal(1 + 1, 2)
    assert_equal("hello", "hello")

    # Boolean assertions
    assert_true(5 > 3)
    assert_false(5 < 3)

    # Container assertions
    assert_in("a", ["a", "b", "c"])
    assert_in("key", {"key": "value"})

    # Null assertions
    assert_is_none(None)
    assert_is_not_none("not none")

    # Numeric assertions
    assert_greater(10, 5)
    assert_less(3, 7)

    # Exception assertions
    def raises_error():
        raise ValueError("Test error")

    assert_raises(ValueError, raises_error)

if __name__ == '__main__':
    run_tests([test_various_assertions])
```

### Skipping Tests

```python
from tests.utils.test_utils import skip_test, run_tests

def test_feature_that_requires_gpu():
    # Check if GPU is available
    if not gpu_available():
        skip_test("GPU not available for this test")

    # Test code here...
    pass

def gpu_available():
    # Implementation to check GPU availability
    return False
```

## Test Discovery Mechanism

The test discovery system automatically finds and executes tests across the project:

- Discovers test functions that start with `test_` in Python files
- Supports both standalone functions and class methods
- Can discover tests for specific models or the entire project
- Provides summary statistics about test coverage

### Test Discovery System (`tests.utils.discovery.py`)

The test discovery system automatically finds and executes tests across the project:

- Discovers test functions that start with `test_` in Python files
- Supports both standalone functions and class methods
- Can discover tests for specific models or the entire project
- Provides summary statistics about test coverage

## Running Tests

### Individual Model Tests

```python
from tests.utils.discovery import run_model_tests

# Run all tests for a specific model
run_model_tests('qwen3_vl_2b')
```

### All Project Tests

```python
from tests.utils.discovery import run_all_project_tests

# Run all tests in the project
run_all_project_tests()
```

### Test Summary

```python
from tests.utils.discovery import get_test_summary

# Get a summary of all tests in the project
summary = get_test_summary()
print(f"Total tests: {summary['total_tests']}")
```

### Manual Test Execution

```python
from tests.utils.test_utils import run_tests

def test_example():
    assert_equal(1 + 1, 2)

if __name__ == '__main__':
    run_tests([test_example])
```

## Best Practices

1. **Descriptive Test Names**: Use clear, descriptive names that indicate what is being tested
2. **Single Responsibility**: Each test should focus on a single aspect of functionality
3. **Meaningful Messages**: Provide clear messages in assertion calls
4. **Isolation**: Tests should not depend on each other's execution order
5. **Consistent Structure**: Follow the standardized directory structure
6. **Comprehensive Coverage**: Include unit, integration, and performance tests as appropriate
7. **Clean Up**: Ensure tests clean up any resources they create
8. **Deterministic**: Tests should produce the same results regardless of execution order
9. **Fast Execution**: Keep tests fast to encourage frequent execution
10. **Documentation**: Comment tests to explain the purpose and expected behavior

## Testing Philosophy

The Inference-PIO project follows these testing principles:

1. **Minimal Dependencies**: Use custom test utilities to avoid external dependencies
2. **Fast Feedback**: Tests should run quickly to provide rapid feedback
3. **Comprehensive Coverage**: Strive for high test coverage across all components
4. **Clear Failures**: Tests should provide clear, actionable failure messages
5. **Maintainable**: Tests should be easy to understand and modify
6. **Automated**: Tests should be easily executable in CI/CD pipelines
7. **Performance Conscious**: Include performance benchmarks to monitor regressions
8. **Integration Focused**: Emphasize integration tests to validate component interactions
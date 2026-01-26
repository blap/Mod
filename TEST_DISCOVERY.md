# Test Discovery Mechanism Documentation

## Overview

The Inference-PIO project includes an automated test discovery system that can find and execute tests across the entire codebase without requiring explicit test suite definitions. This system enables easy test maintenance and execution by automatically detecting test functions based on naming conventions and directory structure.

## Core Discovery Components

### Discovery Functions

The test discovery system provides several functions for different use cases:

- `discover_test_functions_from_file(file_path)` - Discover tests in a single file
- `discover_test_functions_from_directory(directory_path)` - Discover tests in a directory and subdirectories
- `discover_tests_for_model(model_name)` - Discover all tests for a specific model
- `discover_tests_for_plugin_system()` - Discover all tests for the plugin system
- `discover_all_tests()` - Discover all tests in the project
- `run_discovered_tests(test_functions)` - Execute discovered tests
- `get_test_summary()` - Get statistics about available tests

### Test Detection Criteria

The discovery system identifies test functions based on:

1. **Naming Convention**: Functions that start with `test_`
2. **Class Methods**: Methods in classes that start with `test_`
3. **File Location**: Python files in designated test directories

## Directory Structure Recognition

The discovery system understands the standardized test structure:

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

## Using the Discovery System

### Discovering Tests for a Specific Model

```python
from src.inference_pio.test_discovery import discover_tests_for_model

# Discover all tests for a specific model
test_functions = discover_tests_for_model('qwen3_vl_2b')

print(f"Found {len(test_functions)} test functions for qwen3_vl_2b")
```

### Running Model-Specific Tests

```python
from src.inference_pio.test_discovery import run_model_tests

# Run all tests for a specific model
success = run_model_tests('qwen3_vl_2b')

if success:
    print("All tests passed!")
else:
    print("Some tests failed.")
```

### Discovering All Project Tests

```python
from src.inference_pio.test_discovery import discover_all_tests, run_discovered_tests

# Discover all tests in the project
all_test_functions = discover_all_tests()

# Run all discovered tests
success = run_discovered_tests(all_test_functions)
```

### Getting Test Summary

```python
from src.inference_pio.test_discovery import get_test_summary

# Get a summary of all tests in the project
summary = get_test_summary()

print(f"Total tests: {summary['total_tests']}")
print(f"Main tests: {summary['main_tests']}")
print("Model tests:")
for model, count in summary['models'].items():
    print(f"  {model}: {count}")
print(f"Plugin system tests: {summary['plugin_system']}")
```

## Test Function Discovery Process

### File-Level Discovery

The discovery system processes each Python file by:

1. Dynamically importing the module
2. Inspecting all functions that start with `test_`
3. Inspecting class methods that start with `test_`
4. Creating wrappers for class methods to make them executable as functions

Example of how class methods are handled:

```python
class TestExample:
    def test_method(self):
        assert True

# The discovery system creates a wrapper function:
def wrapper_TestExample_test_method():
    instance = TestExample()
    return instance.test_method()
```

### Directory-Level Discovery

For directory discovery, the system:

1. Walks through all subdirectories
2. Identifies Python files (excluding `__init__.py`)
3. Applies file-level discovery to each file
4. Aggregates all discovered test functions

## Running Discovered Tests

### With Custom Test Utilities

The discovery system integrates with the custom test utilities:

```python
from src.inference_pio.test_discovery import discover_all_tests
from src.inference_pio.test_utils import run_tests

# Discover and run all tests
test_functions = discover_all_tests()
success = run_tests(test_functions)
```

### Verbose Output

The discovery system can provide detailed output:

```python
from src.inference_pio.test_discovery import run_all_project_tests

# Run with verbose output
run_all_project_tests(verbose=True)
```

## Integration with Test Utilities

The discovery system works seamlessly with the custom test utilities framework:

```python
from src.inference_pio.test_utils import assert_equal, assert_true
from src.inference_pio.test_discovery import run_discovered_tests

def test_example():
    """Example test function that will be discovered."""
    result = 2 + 2
    assert_equal(result, 4)
    assert_true(result > 0)

# This function will be automatically discovered and executed
```

## Advanced Usage Patterns

### Filtering Tests by Category

While the discovery system doesn't have built-in category filtering, you can implement it:

```python
from src.inference_pio.test_discovery import discover_all_tests

def filter_tests_by_category(test_functions, category):
    """Filter test functions by category based on their file path."""
    filtered = []
    for test_func in test_functions:
        # Get the file path of the test function
        file_path = test_func.__code__.co_filename
        if f"/{category}/" in file_path:
            filtered.append(test_func)
    return filtered

# Discover and filter tests
all_tests = discover_all_tests()
unit_tests = filter_tests_by_category(all_tests, 'unit')
integration_tests = filter_tests_by_category(all_tests, 'integration')
performance_tests = filter_tests_by_category(all_tests, 'performance')
```

### Custom Test Execution

You can customize how discovered tests are executed:

```python
from src.inference_pio.test_discovery import discover_all_tests

def custom_test_runner(test_functions):
    """Custom test runner with additional logging."""
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            print(f"✓ {test_func.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

# Discover and run tests with custom runner
test_functions = discover_all_tests()
success = custom_test_runner(test_functions)
```

## Error Handling

The discovery system handles various error conditions gracefully:

### Import Errors

If a test file cannot be imported, the system logs a warning and continues:

```python
# In case of import errors, the system continues with other files
# Warning: Could not import /path/to/file.py: ImportError message
```

### Test Execution Errors

Individual test failures don't stop the execution of other tests:

```python
# Even if one test fails, others continue to run
# Test summary shows total passed/failed/skipped
```

## Performance Considerations

### Efficient Discovery

The discovery system is designed to be efficient:

- Files are only imported once during discovery
- Test functions are cached for subsequent runs
- Directory walking is optimized

### Memory Usage

- Test functions are stored as references, not copies
- Large test suites are handled efficiently
- Memory usage scales linearly with test count

## Extending the Discovery System

### Adding New Discovery Paths

To add new discovery paths, extend the discovery functions:

```python
def discover_tests_in_custom_path(custom_path):
    """Discover tests in a custom path."""
    from src.inference_pio.test_discovery import discover_test_functions_from_directory
    return discover_test_functions_from_directory(custom_path)

def discover_all_tests_extended():
    """Discover all tests including custom paths."""
    from src.inference_pio.test_discovery import discover_all_tests
    
    all_tests = discover_all_tests()
    custom_tests = discover_tests_in_custom_path('/custom/test/path')
    return all_tests + custom_tests
```

### Custom Naming Conventions

The discovery system can be extended to recognize different naming patterns:

```python
def discover_custom_named_tests(file_path, prefix='spec_'):
    """Discover tests with custom naming convention."""
    import importlib.util
    import inspect
    
    spec = importlib.util.spec_from_file_location("__test_module__", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    test_functions = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith(prefix):
            test_functions.append(obj)
    
    return test_functions
```

## Integration Examples

### With CI/CD Systems

The discovery system works well with continuous integration:

```bash
# Run all tests in CI/CD pipeline
python -c "from src.inference_pio.test_discovery import run_all_project_tests; run_all_project_tests()" || exit 1
```

### Per-Model Testing

Test individual models during development:

```bash
# Test a specific model
python -c "from src.inference_pio.test_discovery import run_model_tests; run_model_tests('qwen3_vl_2b')"
```

### Plugin System Testing

Test the plugin system independently:

```bash
# Test plugin system
python -c "from src.inference_pio.test_discovery import run_plugin_system_tests; run_plugin_system_tests()"
```

## Troubleshooting

### Common Issues

1. **Tests Not Found**: Ensure test functions start with `test_`
2. **Import Errors**: Check that test files have valid Python syntax
3. **Path Issues**: Verify that test files are in recognized directories
4. **Circular Dependencies**: Avoid circular imports in test files

### Debugging Discovery

Enable debugging to see what tests are being discovered:

```python
from src.inference_pio.test_discovery import discover_all_tests

test_functions = discover_all_tests()
for test_func in test_functions:
    print(f"Discovered: {test_func.__name__} in {test_func.__code__.co_filename}")
```

The test discovery system provides a robust, scalable solution for managing and executing tests across the Inference-PIO project, enabling efficient development and quality assurance workflows.
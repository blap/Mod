# Test Discovery System

## Overview

The Inference-PIO project includes an automated test discovery system that automatically finds and executes tests across the standardized project structure. This system eliminates the need to manually specify test files and ensures comprehensive test coverage.

## Discovery Algorithm

The test discovery system operates using the following algorithm:

1. **Directory Traversal**: Scans predefined directories for test files
2. **File Identification**: Identifies Python files containing test functions
3. **Function Extraction**: Finds functions that match the test naming pattern
4. **Import and Validation**: Imports and validates test functions
5. **Execution**: Executes discovered tests with appropriate reporting

## Standardized Structure

The discovery system expects tests to follow the standardized directory structure:

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

## Test Function Recognition

The discovery system identifies test functions based on the following criteria:

- Function names must start with `test_`
- Functions must be callable
- Functions must accept no required parameters (for automatic execution)
- Functions must be defined at the module level (not inside other functions)

## Configuration Options

The discovery system can be configured with the following options:

- **Search Paths**: Directories to search for tests
- **File Patterns**: Patterns to match test files (defaults to `*_test.py` and `test_*.py`)
- **Exclusion Filters**: Patterns to exclude from discovery
- **Recursive Search**: Whether to search subdirectories

## Usage Examples

### Discover and Run All Tests

```python
from src.inference_pio.test_discovery import run_all_project_tests

# Run all tests in the project
run_all_project_tests()
```

### Discover Tests for Specific Model

```python
from src.inference_pio.test_discovery import run_model_tests

# Run all tests for a specific model
run_model_tests('qwen3_vl_2b')
```

### Custom Discovery

```python
from src.inference_pio.test_discovery import TestDiscovery

# Create a custom discovery instance
discovery = TestDiscovery(search_paths=['src/my_module/tests'])

# Discover tests
test_functions = discovery.discover_tests()

# Run discovered tests
from src.inference_pio.test_utils import run_tests
run_tests(test_functions)
```

## Integration with Test Framework

The discovery system integrates seamlessly with the custom test utilities:

- Automatically imports `src.inference_pio.test_utils` for discovered tests
- Ensures discovered tests have access to assertion functions
- Maintains compatibility with `run_tests` function
- Preserves test execution context and reporting

## Performance Considerations

The discovery system is optimized for performance:

- **Caching**: Results are cached to avoid repeated file system scans
- **Filtering**: Unnecessary files are filtered early in the process
- **Parallel Processing**: Supports parallel test execution when possible
- **Memory Efficiency**: Uses generators to minimize memory usage during discovery

## Error Handling

The discovery system handles various error conditions gracefully:

- **Import Errors**: Logs and continues when modules fail to import
- **Syntax Errors**: Reports syntax errors in test files without stopping discovery
- **Permission Issues**: Handles file permission problems appropriately
- **Circular Dependencies**: Detects and reports circular import issues

## Extensibility

The discovery system is designed to be extensible:

- **Custom Matchers**: Support for custom test function identification
- **Plugin Architecture**: Ability to add custom discovery plugins
- **Event Hooks**: Hooks for pre/post discovery actions
- **Reporting Extensions**: Support for custom reporting formats

## Best Practices

When using the test discovery system:

- Follow the standardized directory structure
- Use consistent naming conventions for test files and functions
- Keep test files focused and well-organized
- Use appropriate subdirectories for different test types
- Ensure test functions are self-contained and don't rely on execution order
- Include meaningful docstrings for test functions
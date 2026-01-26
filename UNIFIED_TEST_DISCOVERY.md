# Inference-PIO Unified Test Discovery System

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Core Components](#core-components)
4. [Usage Examples](#usage-examples)
5. [Benefits](#benefits)
6. [Integration Points](#integration-points)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)
9. [Advanced Usage](#advanced-usage)
10. [API Reference](#api-reference)

## Overview

The Unified Test Discovery System is a comprehensive solution for discovering, organizing, and running both tests and benchmarks across the Inference-PIO project. It consolidates multiple existing discovery mechanisms into a single, efficient system that supports the standardized directory structure and naming conventions.

The system provides a unified interface for finding and executing tests and benchmarks, regardless of their location in the project hierarchy or naming convention used.

## Key Features

### 1. Comprehensive Discovery
- Discovers both traditional tests and benchmark functions
- Supports multiple naming conventions for tests and benchmarks
- Compatible with the standardized directory structure
- Handles both function-based and class-based tests

### 2. Flexible Naming Conventions
The system recognizes multiple naming patterns:

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

### 3. Standardized Directory Structure Support
The system understands and navigates the following directory structure:

```
src/
└── inference_pio/
    ├── models/
    │   └── {model_name}/
    │       ├── tests/
    │       │   ├── unit/
    │       │   ├── integration/
    │       │   └── performance/
    │       └── benchmarks/
    │           ├── unit/
    │           ├── integration/
    │           └── performance/
    ├── plugin_system/
    │   ├── tests/
    │   │   ├── unit/
    │   │   ├── integration/
    │   │   └── performance/
    │   └── benchmarks/
    │       ├── unit/
    │       ├── integration/
    │       └── performance/
    └── common/
        ├── tests/
        │   ├── unit/
        │   ├── integration/
        │   └── performance/
        └── benchmarks/
            ├── unit/
            ├── integration/
            └── performance/
```

### 4. Advanced Filtering Capabilities
- Filter by test type (unit, integration, performance)
- Filter by category (tests, benchmarks)
- Filter by model name
- Filter by directory or file

## Core Components

### UnifiedTestDiscovery Class
The main class that provides comprehensive discovery functionality:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

discovery = UnifiedTestDiscovery()
items = discovery.discover_all()
```

### Key Methods

#### Discovery Methods
- `discover_all()` - Discover all tests and benchmarks
- `discover_tests_only()` - Discover only tests
- `discover_benchmarks_only()` - Discover only benchmarks

#### Filtering Methods
- `get_items_by_type(type)` - Filter by item type (UNIT_TEST, INTEGRATION_TEST, etc.)
- `get_items_by_category(category)` - Filter by category (unit, integration, performance)
- `get_items_by_model(model_name)` - Filter by model name
- `get_items_by_file(file_path)` - Filter by specific file

#### Execution Methods
- `run_all_items()` - Run all discovered items
- `run_tests_only()` - Run only tests
- `run_benchmarks_only()` - Run only benchmarks
- `run_item(item_func)` - Run a single item

#### Utility Methods
- `get_discovery_summary()` - Get a summary of all discovered items
- `save_results(results, output_dir)` - Save discovery results to JSON

### Utility Functions
- `discover_and_run_all_items()` - Discover and run everything
- `discover_and_run_tests_only()` - Discover and run only tests
- `discover_and_run_benchmarks_only()` - Discover and run only benchmarks
- `discover_tests_for_model(model_name)` - Discover tests for a specific model
- `discover_benchmarks_for_model(model_name)` - Discover benchmarks for a specific model
- `run_tests_for_model(model_name)` - Run tests for a specific model
- `run_benchmarks_for_model(model_name)` - Run benchmarks for a specific model

### Test Type Enumeration
```python
from enum import Enum

class TestType(Enum):
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    BENCHMARK = "benchmark"
    UNKNOWN = "unknown"
```

## Usage Examples

### Basic Discovery
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

### Model-Specific Discovery
```python
from inference_pio.unified_test_discovery import discover_tests_for_model

# Discover tests for a specific model
model_tests = discover_tests_for_model('qwen3_vl_2b')
print(f"Found {len(model_tests)} tests for qwen3_vl_2b model")

# Discover benchmarks for a specific model
model_benchmarks = discover_benchmarks_for_model('glm_4_7_flash')
print(f"Found {len(model_benchmarks)} benchmarks for glm_4_7_flash model")
```

### Running Tests and Benchmarks
```python
from inference_pio.unified_test_discovery import discover_and_run_tests_only

# Discover and run only tests
results = discover_and_run_tests_only()
```

### Advanced Filtering
```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

discovery = UnifiedTestDiscovery()
discovery.discover_all()

# Get only unit tests
unit_tests = discovery.get_items_by_type(discovery.TestType.UNIT_TEST)

# Get only performance items
performance_items = discovery.get_items_by_category('performance')

# Get items for a specific model
model_items = discovery.get_items_by_model('qwen3_vl_2b')

# Get items from a specific file
file_items = discovery.get_items_by_file('src/inference_pio/models/qwen3_vl_2b/tests/unit/test_basic.py')
```

### Custom Search Paths
```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

# Use custom search paths
custom_paths = [
    "src/my_custom_tests",
    "external_tests",
    "integration_tests"
]

discovery = UnifiedTestDiscovery(search_paths=custom_paths)
items = discovery.discover_all()
```

### Running Model-Specific Items
```python
from inference_pio.unified_test_discovery import run_tests_for_model, run_benchmarks_for_model

# Run tests for a specific model
test_results = run_tests_for_model('qwen3_vl_2b')

# Run benchmarks for a specific model
benchmark_results = run_benchmarks_for_model('glm_4_7_flash')
```

### Getting Discovery Summary
```python
from inference_pio.unified_test_discovery import get_discovery_summary

# Get a comprehensive summary
summary = get_discovery_summary()

print("Discovery Summary:")
print(f"  Total items: {summary['total_items']}")
print(f"  Total tests: {summary['total_tests']}")
print(f"  Total benchmarks: {summary['total_benchmarks']}")
print(f"  By type: {summary['by_type']}")
print(f"  By category: {summary['by_category']}")
print(f"  By model: {summary['by_model']}")
```

## Benefits

### 1. Consolidated Discovery
Single system for both tests and benchmarks, eliminating the need for separate discovery mechanisms.

### 2. Flexible Naming
Supports multiple naming conventions, allowing teams to use their preferred style while maintaining consistency.

### 3. Standardized Structure
Works seamlessly with the project's standardized directory structure, making navigation and maintenance easier.

### 4. Efficient Execution
Optimized for performance with large codebases, using efficient file system traversal and import strategies.

### 5. Comprehensive Coverage
Finds all types of test and benchmark functions, ensuring nothing is missed during execution.

### 6. Easy Integration
Simple API for integration with CI/CD pipelines and other automation tools.

### 7. Class Method Support
Automatically discovers and wraps class-based test methods, supporting both function and class-based testing styles.

## Integration Points

### With Test Utilities Framework
The unified discovery system integrates with the existing test utilities framework:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery
from src.inference_pio.test_utils import run_test

discovery = UnifiedTestDiscovery()
discovery.discover_all()

# Run discovered tests using the utilities framework
for test_func in discovery.test_functions:
    run_test(test_func['function'], test_func['name'])
```

### With Standardized Test and Benchmark Directories
Works with the project's standardized directory structure:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

# Automatically discovers tests in standardized locations
discovery = UnifiedTestDiscovery()
items = discovery.discover_all()  # Finds tests in src/inference_pio/tests/, etc.
```

### With Model-Specific Testing Workflows
Supports model-specific testing workflows:

```python
from inference_pio.unified_test_discovery import run_tests_for_model

# Run tests for a specific model
results = run_tests_for_model('qwen3_vl_2b')
```

### With Plugin System Testing
Integrates with plugin system testing:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

discovery = UnifiedTestDiscovery()
# Automatically finds plugin system tests in standardized locations
items = discovery.discover_all()
```

### With Continuous Integration Pipelines
Designed for CI/CD integration:

```bash
# Run all tests in CI
python -c "from inference_pio.unified_test_discovery import discover_and_run_all_items; discover_and_run_all_items()"
```

## Error Handling

### Import Errors
The system handles import errors in test files gracefully:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

discovery = UnifiedTestDiscovery()
# Will print warnings but continue discovery for other files
items = discovery.discover_all()
```

### Individual Test/Benchmark Failures
Individual failures don't stop the discovery process:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

discovery = UnifiedTestDiscovery()
discovery.discover_all()

# Run items, individual failures handled gracefully
results = discovery.run_all_items()
# Results will contain error information for failed items
```

### Missing Directories
Handles missing directories without crashing:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

# Will warn about missing paths but continue with existing ones
discovery = UnifiedTestDiscovery(search_paths=[
    "existing/path",
    "nonexistent/path"  # This will be skipped with a warning
])
items = discovery.discover_all()
```

### Invalid File Formats
Handles invalid file formats gracefully:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

# Will skip invalid Python files and continue with valid ones
discovery = UnifiedTestDiscovery()
items = discovery.discover_all()
```

## Performance Considerations

### 1. File Import Optimization
- Files are only imported once during discovery
- Imported modules are cached for subsequent runs
- Import errors are logged but don't stop the process

### 2. Directory Walking Efficiency
- Optimized directory traversal algorithms
- Skips unnecessary directories (like __pycache__)
- Respects .gitignore patterns when available

### 3. Memory Usage
- Memory usage scales linearly with the number of discovered items
- Efficient data structures for storing discovered items
- Cleanup of temporary imports after discovery

### 4. Scalability
- Designed to handle large codebases efficiently
- Parallel discovery options for very large projects
- Configurable discovery depth limits

### Performance Tips
```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

# For large projects, use specific search paths
discovery = UnifiedTestDiscovery(search_paths=[
    "src/inference_pio/models/qwen3_vl_2b"  # Target specific model
])

# Or limit discovery to specific categories
items = discovery.discover_tests_only()  # Only discover tests
```

## Advanced Usage

### Custom Discovery Logic
Extend the discovery system for custom needs:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

class CustomDiscovery(UnifiedTestDiscovery):
    def _is_custom_test_function(self, name):
        """Custom logic to identify test functions."""
        return name.startswith('custom_test_')
    
    def discover_custom_items(self):
        """Discover items with custom logic."""
        # Implementation for custom discovery
        pass
```

### Integration with Test Optimization
Combine with the test optimization system:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery
from test_optimization import run_tests_with_optimization

discovery = UnifiedTestDiscovery()
discovery.discover_all()

# Run discovered tests with optimization
test_functions = [item['function'] for item in discovery.test_functions]
test_paths = [item['full_name'] for item in discovery.test_functions]

results = run_tests_with_optimization(
    test_functions=test_functions,
    test_paths=test_paths,
    cache_enabled=True,
    parallel_enabled=True
)
```

### Filtering and Grouping
Advanced filtering and grouping capabilities:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

discovery = UnifiedTestDiscovery()
discovery.discover_all()

# Group by multiple criteria
unit_tests_for_model = [
    item for item in discovery.test_functions
    if item['type'] == discovery.TestType.UNIT_TEST and item['model_name'] == 'qwen3_vl_2b'
]

# Complex filtering
performance_benchmarks = [
    item for item in discovery.benchmark_functions
    if item['category'] == 'performance'
]

# Cross-model comparisons
all_integration_tests = discovery.get_items_by_type(discovery.TestType.INTEGRATION_TEST)
```

### Custom Result Processing
Process results in custom ways:

```python
from inference_pio.unified_test_discovery import UnifiedTestDiscovery

discovery = UnifiedTestDiscovery()
discovery.discover_all()

# Custom result processing
results = discovery.run_all_items()

# Process results
for item_name, result in results['results'].items():
    if 'error' in result:
        print(f"FAILED: {item_name} - {result['error']}")
    else:
        print(f"PASSED: {item_name}")

# Generate custom reports
print(f"Success rate: {results['summary']['success_rate']:.2%}")
```

## API Reference

### UnifiedTestDiscovery Class

#### Constructor
```python
def __init__(self, search_paths: List[str] = None):
    """
    Initialize the unified discovery system.

    Args:
        search_paths: List of paths to search for tests and benchmarks. 
                     If None, uses default paths.
    """
```

#### Discovery Methods
```python
def discover_all(self) -> List[Dict[str, Any]]:
    """
    Discover all test and benchmark functions in the search paths.

    Returns:
        List of dictionaries containing discovered item information
    """

def discover_tests_only(self) -> List[Dict[str, Any]]:
    """
    Discover only test functions.

    Returns:
        List of test function information
    """

def discover_benchmarks_only(self) -> List[Dict[str, Any]]:
    """
    Discover only benchmark functions.

    Returns:
        List of benchmark function information
    """
```

#### Filtering Methods
```python
def get_items_by_type(self, item_type: TestType) -> List[Dict[str, Any]]:
    """
    Get all items of a specific type.

    Args:
        item_type: Type to filter by (TestType.UNIT_TEST, TestType.BENCHMARK, etc.)

    Returns:
        List of items of the specified type
    """

def get_items_by_category(self, category: str) -> List[Dict[str, Any]]:
    """
    Get all items in a specific category.

    Args:
        category: Category to filter by ('unit', 'integration', 'performance', 'other')

    Returns:
        List of items in the category
    """

def get_items_by_model(self, model_name: str) -> List[Dict[str, Any]]:
    """
    Get all items for a specific model.

    Args:
        model_name: Name of the model to filter by

    Returns:
        List of items for the model
    """

def get_items_by_file(self, file_path: str) -> List[Dict[str, Any]]:
    """
    Get all items from a specific file.

    Args:
        file_path: Path to the file to filter by

    Returns:
        List of items from the file
    """
```

#### Execution Methods
```python
def run_item(self, item_func: Callable, *args, **kwargs) -> Any:
    """
    Run a single test or benchmark function safely.

    Args:
        item_func: The function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function
    """

def run_all_items(self, include_types: List[TestType] = None, 
                  include_categories: List[str] = None) -> Dict[str, Any]:
    """
    Run all discovered items.

    Args:
        include_types: List of types to include. If None, runs all types.
        include_categories: List of categories to include. If None, runs all categories.

    Returns:
        Dictionary with results from all items
    """

def run_tests_only(self) -> Dict[str, Any]:
    """
    Run only the discovered tests (not benchmarks).

    Returns:
        Dictionary with results from tests
    """

def run_benchmarks_only(self) -> Dict[str, Any]:
    """
    Run only the discovered benchmarks (not tests).

    Returns:
        Dictionary with results from benchmarks
    """
```

#### Utility Methods
```python
def save_results(self, results: Dict[str, Any], output_dir: str = "discovery_results") -> None:
    """
    Save discovery results to JSON file.

    Args:
        results: Results dictionary to save
        output_dir: Directory to save results to
    """
```

### Standalone Functions

#### Discovery Functions
```python
def discover_and_run_all_items() -> Dict[str, Any]:
    """
    Convenience function to discover and run all tests and benchmarks in the project.

    Returns:
        Dictionary with results from all items
    """

def discover_and_run_tests_only() -> Dict[str, Any]:
    """
    Convenience function to discover and run only tests (not benchmarks).

    Returns:
        Dictionary with results from tests
    """

def discover_and_run_benchmarks_only() -> Dict[str, Any]:
    """
    Convenience function to discover and run only benchmarks (not tests).

    Returns:
        Dictionary with results from benchmarks
    """
```

#### Model-Specific Functions
```python
def discover_tests_for_model(model_name: str) -> List[Dict[str, Any]]:
    """
    Discover all tests for a specific model.

    Args:
        model_name: Name of the model to discover tests for

    Returns:
        List of test functions for the model
    """

def discover_benchmarks_for_model(model_name: str) -> List[Dict[str, Any]]:
    """
    Discover all benchmarks for a specific model.

    Args:
        model_name: Name of the model to discover benchmarks for

    Returns:
        List of benchmark functions for the model
    """

def run_tests_for_model(model_name: str) -> Dict[str, Any]:
    """
    Run all tests for a specific model.

    Args:
        model_name: Name of the model to run tests for

    Returns:
        Dictionary with results from model tests
    """

def run_benchmarks_for_model(model_name: str) -> Dict[str, Any]:
    """
    Run all benchmarks for a specific model.

    Args:
        model_name: Name of the model to run benchmarks for

    Returns:
        Dictionary with results from model benchmarks
    """
```

#### Summary Functions
```python
def get_discovery_summary() -> Dict[str, Any]:
    """
    Get a summary of all discovered items in the project.

    Returns:
        Dictionary with discovery summary information
    """
```

---

The Unified Test Discovery System provides a comprehensive, flexible, and efficient way to discover and run tests and benchmarks across the Inference-PIO project. Its standardized approach ensures consistency while its flexibility allows for various naming conventions and organizational structures.
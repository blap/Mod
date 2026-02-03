# Test Fixtures and Setup Guidelines

This document establishes standards and best practices for creating and using test fixtures and setups in the Inference-PIO project.

## Overview

Fixtures are special functions (often used with pytest) that provide data or objects required for tests. They help eliminate duplicate code and ensure consistency across tests.

## Fundamental Principles

1.  **Reusability**: Fixtures should be designed to be reusable across multiple tests.
2.  **Isolation**: Each test must be independent and not affect others.
3.  **Cleanup**: Every fixture must handle the cleanup of resources it created.
4.  **Clarity**: Fixture names must be descriptive and self-explanatory.
5.  **Efficiency**: Fixtures should be lightweight and fast to execute.

## Types of Fixtures

### Temporary Resource Fixtures
-   **Purpose**: Provide temporary resources like directories, files, or connections.
-   **Example**: `temp_dir` fixture that creates and cleans up a temporary directory.

### Test Data Fixtures
-   **Purpose**: Provide consistent data for tests.
-   **Examples**: `sample_text_data`, `sample_tensor_data`, `sample_config`.

### Mock Object Fixtures
-   **Purpose**: Provide mocked instances of complex objects.
-   **Examples**: `mock_torch_model`, `realistic_test_plugin`.

### Configuration Fixtures
-   **Purpose**: Provide default configurations for tests.
-   **Examples**: `sample_config`, `sample_plugin_manifest`.

## Naming Conventions

-   Use descriptive names in `snake_case`.
-   Prefer short but meaningful names.
-   Avoid unnecessary abbreviations.
-   Use prefixes when appropriate (e.g., `mock_`, `sample_`, `temp_`).

## Recommended Structure

```python
import pytest
from pathlib import Path
from typing import Generator

@pytest.fixture
def fixture_name() -> ReturnType:
    """
    Brief description of what this fixture does.

    Returns:
        Description of the return type.
    """
    # Setup: code to prepare the resource
    resource = create_resource()

    # Yield: returns the resource to the tests
    yield resource

    # Teardown: code to clean up the resource
    cleanup_resource(resource)
```

## Plugin Initialization Utilities

For testing plugins, we provide shared utilities to centralize initialization logic and avoid duplication. These are located in `tests.shared.utils.plugin_init_utils`.

### Available Functions
- `initialize_plugin_for_test()`: Initializes a plugin with test configuration.
- `create_and_initialize_plugin()`: Creates and initializes a plugin in a single operation.
- `cleanup_plugin()`: Safely cleans up a plugin.
- `verify_plugin_interface()`: Verifies if a plugin implements required methods.

### Usage Example

```python
from tests.shared.utils.plugin_init_utils import create_and_initialize_plugin

def test_my_plugin_feature():
    from src.inference_pio.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin

    # Create and initialize the plugin in one call
    plugin = create_and_initialize_plugin(Qwen3_0_6B_Plugin)

    # The plugin is now ready for use in tests
    assert plugin.is_loaded is True
```

## Best Practices

### 1. Use Type Hints
Always specify return types for fixtures for better readability and static checking.

```python
@pytest.fixture
def sample_tensor_data() -> torch.Tensor:
    return torch.randn(4, 10, 128)
```

### 2. Document Properly
Include clear docstrings explaining the fixture's purpose and return value.

### 3. Avoid Side Effects
Fixtures should have minimal and predictable side effects. Avoid modifying global state.

### 4. Use Appropriate Scope
Define the correct scope to optimize performance (e.g., `scope="module"` for expensive resources, `scope="function"` for mutable state).

### 5. Encapsulate Complex Logic
When setup logic is complex, encapsulate it in helper functions.

## Fixture Organization

### Shared Fixtures
-   **Location**: `src/inference_pio/tests/shared/fixtures/`
-   Should be generic enough for multiple contexts.

### Domain-Specific Fixtures
-   **Location**: Inside the corresponding test directory (e.g., `src/inference_pio/models/<model_name>/tests/`).
-   Can have domain-specific dependencies.

## Available Shared Fixtures

The following fixtures are available in `src.inference_pio.tests.shared.fixtures`:

### Basic Fixtures
- `temp_dir`: Temporary directory for tests.
- `temp_file`: Temporary file within the temp directory.
- `sample_text_data`: Sample text data.
- `sample_tensor_data`: Sample tensor data.
- `parametrized_tensor_data`: Parameterized tensor data for sizing tests.

### Configuration Fixtures
- `sample_config`: Sample configuration.
- `plugin_config_with_gpu`: Plugin configuration with GPU settings.
- `sample_plugin_manifest`: Sample plugin manifest.

### Mock Object Fixtures
- `mock_torch_model`: Mock PyTorch model.
- `mock_plugin_dependencies`: Mock dependencies for plugins.
- `realistic_test_plugin`: Realistic plugin instance.
- `mock_plugin_with_error_handling`: Mock plugin with error handling.

### Advanced Fixtures
- `complex_test_environment`: Full test environment with multiple resources.
- `device_and_precision_config`: Parameterized device and precision configs.
- `specialized_plugin_metadata`: Specialized plugin metadata.
- `expensive_resource`: Expensive resource initialized once per session.
- `mocked_network_operations`: Mocks for network operations.
- `performance_test_data`: Large data for performance tests.
- `concurrency_test_setup`: Resources for concurrency tests.
- `validated_test_state`: Test state with validation.
- `plugin_factory`: Factory to create multiple plugin instances.
- `integration_test_components`: Multiple components for integration tests.

## Examples

### Simple Data Fixture
```python
@pytest.fixture
def sample_text_data() -> list:
    """
    Provide sample text data for testing.

    Returns:
        List of sample text strings
    """
    return ["text1", "text2", "text3", "text4", "text5"]
```

### Fixture with Setup and Teardown
```python
@pytest.fixture
def temp_database() -> Generator[DatabaseConnection, None, None]:
    """
    Create a temporary in-memory database for testing.

    Yields:
        Database connection object
    """
    db = create_in_memory_db()
    db.connect()

    yield db

    db.disconnect()
    cleanup_temp_db(db)
```

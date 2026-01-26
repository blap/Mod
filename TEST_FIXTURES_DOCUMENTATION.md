# Test Fixtures System Documentation

## Overview

The Inference-PIO project includes a comprehensive test fixture system designed to provide consistent test data setup and teardown, manage test resources properly, and follow the project's architecture patterns. This system enables reusable fixtures across different test modules while properly handling resource cleanup.

## Key Components

### 1. Fixture Manager
The central component that manages the lifecycle of test fixtures:
- Registers fixtures with different scopes
- Handles fixture instantiation and caching
- Manages cleanup operations

### 2. Fixture Decorators
- `@fixture(scope='function')`: Decorator to register fixture functions
- `@use_fixtures(*fixture_names)`: Decorator to inject fixtures into test functions

### 3. Built-in Fixtures
- `temp_dir`: Temporary directory fixture
- `tensor_fixture`: PyTorch tensor fixture
- `mock_model_fixture`: Mock model components fixture
- `config_fixture`: Configuration fixture
- `db_fixture`: Mock database fixture

### 4. Context Managers
- `fixture_context(*fixture_names)`: Context manager for using fixtures

## Usage Examples

### Basic Fixture Registration and Usage

```python
from src.inference_pio.test_fixtures import fixture, use_fixtures

@fixture(scope='function')
def my_custom_fixture():
    return {"data": [1, 2, 3, 4, 5]}

@use_fixtures('my_custom_fixture')
def test_my_function(my_custom_fixture=None):
    assert len(my_custom_fixture['data']) == 5
```

### Using Context Manager

```python
from src.inference_pio.test_fixtures import fixture_context

def test_with_context():
    with fixture_context('temp_dir', 'tensor_fixture') as fixtures:
        temp_dir = fixtures['temp_dir']
        tensor_fixture = fixtures['tensor_fixture']
        
        # Use the fixtures...
        assert temp_dir is not None
        assert tensor_fixture is not None
```

### Creating Custom Fixtures

```python
from src.inference_pio.test_fixtures import FixtureManager

def create_custom_fixture():
    # Create and return your fixture object
    return MyCustomResource()

# Register with the manager
manager = FixtureManager()
manager.register_fixture("custom_fixture", create_custom_fixture, scope="function")
```

## Fixture Scopes

Fixtures can have different scopes that determine their lifecycle:

- `function`: New instance for each test function
- `class`: Single instance for all tests in a test class
- `module`: Single instance for all tests in a module
- `session`: Single instance for the entire test session

## Available Built-in Fixtures

### Temporary Directory Fixture
Provides a temporary directory that is automatically cleaned up after the test.

```python
@use_fixtures('temp_dir')
def test_file_operations(temp_dir=None):
    # temp_dir is a path to a temporary directory
    test_file = os.path.join(temp_dir, "test.txt")
    # File operations...
```

### Tensor Fixture
Provides pre-created PyTorch tensors for testing.

```python
@use_fixtures('tensor_fixture')
def test_tensor_operations(tensor_fixture=None):
    tensor = tensor_fixture.get_tensor("tensor_0")
    # Tensor operations...
```

### Mock Model Fixture
Provides mock model components for testing.

```python
@use_fixtures('mock_model_fixture')
def test_model_operations(mock_model_fixture=None):
    model = mock_model_fixture.model
    optimizer = mock_model_fixture.optimizer
    # Model operations...
```

### Configuration Fixture
Provides configuration objects for testing.

```python
@use_fixtures('config_fixture')
def test_config_operations(config_fixture=None):
    config = config_fixture.get_config()
    # Config operations...
```

## Best Practices

1. **Use appropriate scopes**: Choose the right scope for your fixture to avoid unnecessary resource creation.

2. **Implement cleanup**: Ensure your fixtures implement proper cleanup methods to free resources.

3. **Keep fixtures focused**: Each fixture should serve a specific purpose rather than being overly complex.

4. **Use context managers**: When possible, use the `fixture_context` manager for explicit fixture management.

5. **Combine fixtures wisely**: Use multiple fixtures together when needed, but avoid excessive dependencies.

## Resource Management

The fixture system automatically handles resource cleanup:
- Temp files and directories are removed
- Memory-intensive objects are deleted
- Database connections are closed
- GPU memory is freed when applicable

## Integration with Existing Test Framework

The fixture system integrates seamlessly with the existing Inference-PIO test utilities:

```python
from src.inference_pio.test_utils import assert_equal, run_tests
from src.inference_pio.test_fixtures import use_fixtures

@use_fixtures('tensor_fixture')
def test_tensor_equality(tensor_fixture=None):
    tensor1 = tensor_fixture.get_tensor("tensor_0")
    tensor2 = tensor_fixture.get_tensor("tensor_0")  # Same name gives same tensor for session-scoped
    assert_equal(tensor1.shape, tensor2.shape)

if __name__ == "__main__":
    run_tests([test_tensor_equality])
```

## Error Handling

The fixture system includes proper error handling:
- Invalid fixture requests raise ValueError
- Fixture creation errors are propagated appropriately
- Cleanup operations are robust to errors

## Advanced Usage

### Creating Custom Fixture Classes

```python
class CustomDataFixture:
    def __init__(self, data_size=100):
        self.data = list(range(data_size))
    
    def get_sample(self, size=10):
        return self.data[:size]
    
    def cleanup(self):
        del self.data

@fixture(scope='session')
def custom_data_fixture():
    return CustomDataFixture(data_size=1000)
```

### Conditional Fixture Usage

```python
def conditional_fixture_test():
    if torch.cuda.is_available():
        with fixture_context('gpu_tensor_fixture') as fixtures:
            tensor = fixtures['gpu_tensor_fixture']
            # GPU-specific tests
    else:
        with fixture_context('cpu_tensor_fixture') as fixtures:
            tensor = fixtures['cpu_tensor_fixture']
            # CPU-specific tests
```

This fixture system provides a robust foundation for managing test data and resources across the Inference-PIO project, ensuring consistent, reliable, and efficient testing.
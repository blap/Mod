# Writing Tests for Inference-PIO

## Table of Contents

1. [Introduction](#introduction)
2. [Test Organization](#test-organization)
3. [Naming Conventions](#naming-conventions)
4. [Test Types](#test-types)
5. [Writing Effective Tests](#writing-effective-tests)
6. [Using Assertion Functions](#using-assertion-functions)
7. [Testing Different Components](#testing-different-components)
8. [Performance Testing](#performance-testing)
9. [Tensor Operations Testing](#tensor-operations-testing)
10. [File System Testing](#file-system-testing)
11. [Error Handling Testing](#error-handling-testing)
12. [Test Data Management](#test-data-management)
13. [Test Dependencies](#test-dependencies)
14. [Best Practices](#best-practices)
15. [Common Pitfalls](#common-pitfalls)

## Introduction

This guide provides comprehensive instructions on how to write effective tests for the Inference-PIO project. The project uses a custom testing framework with over 100 assertion functions designed specifically for AI/ML applications, particularly those involving tensor operations and model inference.

## Test Organization

### Directory Structure

Tests should be organized in the following structure:

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
    │   └── tests/
    │       ├── unit/
    │       ├── integration/
    │       └── performance/
    └── common/
        └── tests/
            ├── unit/
            ├── integration/
            └── performance/
```

### Test Categories

- **Unit tests**: Test individual functions/methods in isolation
- **Integration tests**: Test interactions between components
- **Performance tests**: Measure execution time and resource usage
- **Benchmarks**: Compare performance against baselines

## Naming Conventions

### Test Function Names

Use descriptive names that clearly indicate what is being tested:

```python
# Good
def test_model_initialization_with_valid_config():
    """Test model initialization with valid configuration."""
    pass

def test_tensor_addition_returns_correct_result():
    """Test that tensor addition returns the correct result."""
    pass

def test_invalid_input_raises_value_error():
    """Test that invalid input raises ValueError."""
    pass

# Avoid
def test1():
    pass

def test_func():
    pass
```

### Supported Naming Patterns

The unified discovery system supports multiple naming patterns:

**Test patterns:**
- `test_*` - Standard test functions
- `should_*` - Behavior-driven test naming
- `when_*` - Scenario-based test naming
- `verify_*` - Verification-based test naming
- `validate_*` - Validation-based test naming
- `check_*` - Check-based test naming

**Benchmark patterns:**
- `run_*` - Standard run functions
- `benchmark_*` - Standard benchmark functions
- `perf_*` - Performance-related functions
- `measure_*` - Measurement-related functions
- `profile_*` - Profiling-related functions
- `time_*` - Timing-related functions
- `speed_*` - Speed-related functions
- `stress_*` - Stress testing functions
- `load_*` - Load testing functions

## Test Types

### Unit Tests

Unit tests focus on individual functions or methods in isolation:

```python
from src.inference_pio.test_utils import assert_equal, assert_true

def test_calculate_loss_returns_expected_value():
    """Test that loss calculation returns expected value."""
    predictions = [0.1, 0.2, 0.7]
    targets = [0, 0, 1]
    expected_loss = 0.3567  # Calculated expected value
    
    actual_loss = calculate_loss(predictions, targets)
    assert_equal(actual_loss, expected_loss, 
                "Calculated loss should match expected value")
```

### Integration Tests

Integration tests verify that multiple components work together:

```python
from src.inference_pio.test_utils import assert_tensor_shape

def test_model_pipeline_produces_correct_output_shape():
    """Test that the complete model pipeline produces correct output shape."""
    model = create_model()
    input_data = prepare_input_data()
    
    output = model.process(input_data)
    
    assert_tensor_shape(output, (1, 1000), 
                       "Model output should have shape (1, 1000)")
```

### Performance Tests

Performance tests measure execution time and resource usage:

```python
import time
from src.inference_pio.test_utils import assert_less

def test_model_inference_completes_within_time_limit():
    """Test that model inference completes within acceptable time."""
    model = create_model()
    input_data = prepare_input_data()
    
    start_time = time.time()
    model.infer(input_data)
    execution_time = time.time() - start_time
    
    assert_less(execution_time, 1.0,  # Less than 1 second
               "Model inference should complete in under 1 second")
```

## Writing Effective Tests

### Test Structure

Follow the AAA (Arrange, Act, Assert) pattern:

```python
def test_tensor_concatenation_produces_correct_shape():
    """Test tensor concatenation produces correct shape."""
    # Arrange
    tensor1 = torch.tensor([[1, 2], [3, 4]])
    tensor2 = torch.tensor([[5, 6]])
    expected_shape = (3, 2)
    
    # Act
    result = torch.cat([tensor1, tensor2], dim=0)
    
    # Assert
    assert_tensor_shape(result, expected_shape,
                      "Concatenated tensor should have correct shape")
```

### Test Independence

Each test should be independent and not rely on other tests:

```python
# Good - Independent test
def test_create_model_returns_valid_instance():
    """Test that create_model returns a valid model instance."""
    model = create_model()
    assert_is_not_none(model, "Model should not be None")
    assert_is_instance(model, BaseModel, "Model should be instance of BaseModel")

# Bad - Depends on other test
def test_model_prediction():
    """Test model prediction (depends on test_create_model_returns_valid_instance)."""
    # Assumes model was created by previous test
    result = model.predict(input_data)  # model not defined in this test
    assert_true(result is not None)
```

### Test Specificity

Make tests specific to one behavior:

```python
# Good - Tests one thing
def test_tensor_addition_returns_correct_values():
    """Test that tensor addition returns correct values."""
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6])
    expected = torch.tensor([5, 7, 9])
    
    result = tensor1 + tensor2
    assert_tensor_equal(result, expected)

# Bad - Tests multiple things
def test_tensor_operations():
    """Test multiple tensor operations."""
    # Tests addition, subtraction, multiplication in one test
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6])
    
    # Addition test
    add_result = tensor1 + tensor2
    assert_tensor_equal(add_result, torch.tensor([5, 7, 9]))
    
    # Subtraction test
    sub_result = tensor2 - tensor1
    assert_tensor_equal(sub_result, torch.tensor([3, 3, 3]))
    
    # Multiplication test
    mul_result = tensor1 * tensor2
    assert_tensor_equal(mul_result, torch.tensor([4, 10, 18]))
```

## Using Assertion Functions

### Basic Assertions

```python
from src.inference_pio.test_utils import (
    assert_true, assert_false, assert_equal, assert_not_equal,
    assert_is_none, assert_is_not_none
)

def test_basic_assertions():
    # Boolean assertions
    assert_true(5 > 3, "Five should be greater than three")
    assert_false(5 < 3, "Five should not be less than three")
    
    # Equality assertions
    assert_equal(2 + 2, 4, "Basic addition should work")
    assert_not_equal(2 + 2, 5, "Two plus two should not equal five")
    
    # Null assertions
    assert_is_none(None, "None should be None")
    assert_is_not_none("not none", "String should not be None")
```

### Container Assertions

```python
from src.inference_pio.test_utils import (
    assert_in, assert_not_in, assert_length, assert_items_equal
)

def test_container_assertions():
    # Membership assertions
    assert_in("apple", ["apple", "banana", "cherry"], "Apple should be in list")
    assert_not_in("orange", ["apple", "banana", "cherry"], "Orange should not be in list")
    
    # Length assertions
    assert_length([1, 2, 3, 4], 4, "List should have 4 elements")
    
    # Item equality (order-independent)
    assert_items_equal([1, 2, 3], [3, 1, 2], "Lists should contain same items")
```

### Numeric Assertions

```python
from src.inference_pio.test_utils import (
    assert_greater, assert_less, assert_between, assert_close
)

def test_numeric_assertions():
    # Comparison assertions
    assert_greater(10, 5, "Ten should be greater than five")
    assert_less(3, 7, "Three should be less than seven")
    
    # Range assertions
    assert_between(5, 1, 10, "Five should be between 1 and 10")
    
    # Approximate equality
    assert_close(1.0000001, 1.0, rel_tol=1e-06, 
                "Values should be close within tolerance")
```

### Tensor Assertions

```python
import torch
from src.inference_pio.test_utils import (
    assert_tensor_equal, assert_tensor_close, assert_tensor_shape, 
    assert_tensor_dtype, assert_tensor_all_positive
)

def test_tensor_assertions():
    # Equality assertions
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([1.0, 2.0, 3.0])
    assert_tensor_equal(tensor1, tensor2, "Tensors should be equal")
    
    # Close assertions (with tolerance)
    tensor3 = torch.tensor([1.0001, 2.0001, 3.0001])
    assert_tensor_close(tensor1, tensor3, rtol=1e-03, 
                       "Tensors should be close within tolerance")
    
    # Shape assertions
    assert_tensor_shape(tensor1, (3,), "Tensor should have shape (3,)")
    
    # Dtype assertions
    assert_tensor_dtype(tensor1, torch.float32, "Tensor should have float32 dtype")
    
    # Value property assertions
    positive_tensor = torch.tensor([1.0, 2.0, 3.0])
    assert_tensor_all_positive(positive_tensor, "All values should be positive")
```

### Exception Assertions

```python
from src.inference_pio.test_utils import assert_raises

def test_exception_assertions():
    def raises_value_error():
        raise ValueError("Test error")
    
    def raises_type_error():
        raise TypeError("Test error")
    
    # Test specific exception type
    assert_raises(ValueError, raises_value_error)
    assert_raises(TypeError, raises_type_error)
```

### File System Assertions

```python
from src.inference_pio.test_utils import (
    assert_file_exists, assert_dir_exists, assert_readable, assert_writable
)

def test_file_system_assertions():
    # Existence assertions
    assert_file_exists("requirements.txt", "Requirements file should exist")
    assert_dir_exists("src", "Source directory should exist")
    
    # Permission assertions
    assert_readable("README.md", "README should be readable")
    assert_writable("/tmp", "Temp directory should be writable")
```

## Testing Different Components

### Model Plugin Tests

```python
from src.inference_pio.test_utils import assert_is_instance, assert_true

def test_model_plugin_initialization():
    """Test model plugin initialization."""
    plugin = create_model_plugin()
    
    assert_is_instance(plugin, ModelPluginInterface, 
                      "Plugin should implement ModelPluginInterface")
    assert_true(hasattr(plugin, 'initialize'), 
               "Plugin should have initialize method")
    assert_true(hasattr(plugin, 'infer'), 
               "Plugin should have infer method")
```

### Configuration Tests

```python
from src.inference_pio.test_utils import assert_in, assert_equal

def test_configuration_loading():
    """Test configuration loading."""
    config = load_model_config()
    
    # Check required keys exist
    required_keys = ['model_path', 'device', 'batch_size']
    for key in required_keys:
        assert_in(key, config, f"Configuration should contain {key}")
    
    # Check specific values
    assert_equal(config['batch_size'], 32, "Default batch size should be 32")
```

### Data Processing Tests

```python
import torch
from src.inference_pio.test_utils import assert_tensor_shape, assert_tensor_dtype

def test_data_preprocessing():
    """Test data preprocessing pipeline."""
    raw_data = load_raw_data()
    processed_data = preprocess(raw_data)
    
    # Check output shape
    expected_shape = (3, 224, 224)  # RGB image
    assert_tensor_shape(processed_data, expected_shape,
                      f"Processed data should have shape {expected_shape}")
    
    # Check output type
    assert_tensor_dtype(processed_data, torch.float32,
                       "Processed data should be float32")
```

## Performance Testing

### Timing Tests

```python
import time
from src.inference_pio.test_utils import assert_less

def test_model_inference_timing():
    """Test model inference timing."""
    model = create_model()
    input_data = create_test_input()
    
    start_time = time.perf_counter()
    result = model.infer(input_data)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    # Assert the operation completes within acceptable time
    assert_less(execution_time, 0.5,  # 500ms limit
               f"Inference should complete in under 500ms, took {execution_time:.3f}s")
```

### Memory Usage Tests

```python
import psutil
import gc
from src.inference_pio.test_utils import assert_less

def test_memory_usage_during_processing():
    """Test memory usage during processing."""
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Perform memory-intensive operation
    large_tensor = torch.randn(1000, 1000, 1000)  # 4GB tensor
    processed = process_large_tensor(large_tensor)
    
    # Clean up
    del large_tensor, processed
    gc.collect()
    
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Assert memory usage is within limits
    memory_increase = final_memory - initial_memory
    assert_less(memory_increase, 1000,  # 1GB limit
               f"Memory increase should be under 1GB, was {memory_increase:.2f}MB")
```

### Throughput Tests

```python
import time
from src.inference_pio.test_utils import assert_greater

def test_model_throughput():
    """Test model throughput (requests per second)."""
    model = create_model()
    inputs = [create_test_input() for _ in range(100)]
    
    start_time = time.time()
    for input_data in inputs:
        model.infer(input_data)
    end_time = time.time()
    
    total_time = end_time - start_time
    requests_per_second = len(inputs) / total_time
    
    # Assert minimum throughput
    assert_greater(requests_per_second, 10.0,  # 10 RPS minimum
                  f"Throughput should be at least 10 RPS, got {requests_per_second:.2f} RPS")
```

## Tensor Operations Testing

### Basic Tensor Operations

```python
import torch
from src.inference_pio.test_utils import assert_tensor_equal

def test_tensor_addition():
    """Test tensor addition operation."""
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6])
    expected = torch.tensor([5, 7, 9])
    
    result = tensor1 + tensor2
    assert_tensor_equal(result, expected, "Tensor addition should produce correct result")

def test_tensor_matrix_multiplication():
    """Test matrix multiplication."""
    matrix1 = torch.tensor([[1, 2], [3, 4]])  # 2x2
    matrix2 = torch.tensor([[5, 6], [7, 8]])  # 2x2
    expected = torch.tensor([[19, 22], [43, 50]])  # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    
    result = torch.matmul(matrix1, matrix2)
    assert_tensor_equal(result, expected, "Matrix multiplication should produce correct result")
```

### Tensor Shape Testing

```python
import torch
from src.inference_pio.test_utils import assert_tensor_shape

def test_convolution_output_shape():
    """Test convolution layer output shape."""
    conv_layer = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch=1, Channels=3, Height=32, Width=32
    
    output = conv_layer(input_tensor)
    
    # For a 3x3 kernel with no padding and stride=1 on 32x32 input,
    # output should be 1x16x30x30
    expected_shape = (1, 16, 30, 30)
    assert_tensor_shape(output, expected_shape,
                       f"Convolution output should have shape {expected_shape}, got {output.shape}")
```

### Tensor Dtype Testing

```python
import torch
from src.inference_pio.test_utils import assert_tensor_dtype

def test_tensor_dtype_preservation():
    """Test that operations preserve tensor dtype."""
    input_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
    
    # Operations that should preserve dtype
    result = input_tensor * 2
    
    assert_tensor_dtype(result, torch.int64,
                      "Multiplication should preserve int64 dtype")
```

### Tensor Value Range Testing

```python
import torch
from src.inference_pio.test_utils import (
    assert_tensor_all_between, assert_tensor_all_positive, assert_tensor_all_finite
)

def test_activation_function_output_ranges():
    """Test activation function output ranges."""
    input_tensor = torch.randn(100)  # Random values
    
    # Sigmoid should output values between 0 and 1
    sigmoid_output = torch.sigmoid(input_tensor)
    assert_tensor_all_between(sigmoid_output, 0.0, 1.0,
                             "Sigmoid should output values between 0 and 1")
    
    # ReLU should output non-negative values
    relu_output = torch.relu(input_tensor)
    assert_tensor_all_positive(relu_output,
                              "ReLU should output positive values")
    
    # All outputs should be finite (no NaN or Inf)
    assert_tensor_all_finite(sigmoid_output, "Sigmoid output should be finite")
    assert_tensor_all_finite(relu_output, "ReLU output should be finite")
```

## File System Testing

### File Existence and Permissions

```python
from src.inference_pio.test_utils import (
    assert_file_exists, assert_dir_exists, assert_readable, assert_writable
)

def test_model_files_exist():
    """Test that required model files exist."""
    model_files = [
        "model_weights.bin",
        "config.json",
        "tokenizer.json"
    ]
    
    for file_path in model_files:
        assert_file_exists(file_path, f"Required model file {file_path} should exist")

def test_config_directory_permissions():
    """Test configuration directory permissions."""
    config_dir = "/path/to/config"
    
    assert_dir_exists(config_dir, "Config directory should exist")
    assert_readable(config_dir, "Config directory should be readable")
    assert_writable(config_dir, "Config directory should be writable")
```

### File Content Testing

```python
from src.inference_pio.test_utils import assert_json_valid, assert_json_equal

def test_config_file_content():
    """Test configuration file content."""
    with open("config.json", "r") as f:
        config_content = f.read()
    
    # Validate JSON format
    assert_json_valid(config_content, "Config file should contain valid JSON")
    
    # Check specific configuration values
    expected_config = {
        "model_name": "test_model",
        "input_size": 224,
        "batch_size": 32
    }
    
    assert_json_equal(config_content, expected_config,
                     "Config should match expected values")
```

## Error Handling Testing

### Exception Testing

```python
from src.inference_pio.test_utils import assert_raises

def test_invalid_input_handling():
    """Test handling of invalid input."""
    model = create_model()
    
    # Test with wrong input type
    with assert_raises(TypeError):
        model.infer("invalid_input_type")
    
    # Test with wrong tensor shape
    wrong_shape_tensor = torch.randn(1, 10)  # Wrong shape
    with assert_raises(ValueError):
        model.infer(wrong_shape_tensor)
    
    # Test with empty input
    empty_tensor = torch.empty(0)
    with assert_raises(ValueError):
        model.infer(empty_tensor)
```

### Boundary Condition Testing

```python
from src.inference_pio.test_utils import assert_raises

def test_boundary_conditions():
    """Test boundary conditions."""
    processor = DataProcessor(max_batch_size=32)
    
    # Valid batch sizes
    valid_inputs = [torch.randn(1, 10), torch.randn(16, 10), torch.randn(32, 10)]
    for input_tensor in valid_inputs:
        # Should not raise exception
        result = processor.process(input_tensor)
    
    # Invalid batch size (too large)
    with assert_raises(ValueError):
        large_batch = torch.randn(64, 10)  # Exceeds max_batch_size
        processor.process(large_batch)
    
    # Invalid batch size (zero)
    with assert_raises(ValueError):
        zero_batch = torch.randn(0, 10)
        processor.process(zero_batch)
```

### Resource Limit Testing

```python
from src.inference_pio.test_utils import assert_raises

def test_resource_limits():
    """Test behavior when resource limits are exceeded."""
    model = create_model_with_memory_limit(100)  # 100MB limit
    
    # Small input should work
    small_input = torch.randn(10, 10)
    result = model.process(small_input)
    
    # Large input should raise exception
    with assert_raises(MemoryError):
        huge_input = torch.randn(10000, 10000)  # Would exceed memory limit
        model.process(huge_input)
```

## Test Data Management

### Mock Data Creation

```python
import torch
import numpy as np

def create_mock_image_data(batch_size=4, channels=3, height=224, width=224):
    """Create mock image data for testing."""
    return torch.randn(batch_size, channels, height, width)

def create_mock_text_data(batch_size=4, seq_length=512, vocab_size=1000):
    """Create mock text data for testing."""
    return torch.randint(0, vocab_size, (batch_size, seq_length))

def create_mock_audio_data(batch_size=4, samples=16000, sample_rate=16000):
    """Create mock audio data for testing."""
    return torch.randn(batch_size, samples)
```

### Test Data Validation

```python
from src.inference_pio.test_utils import assert_tensor_shape, assert_tensor_dtype

def validate_test_data(data, expected_shape, expected_dtype):
    """Validate test data properties."""
    assert_tensor_shape(data, expected_shape,
                       f"Test data should have shape {expected_shape}")
    assert_tensor_dtype(data, expected_dtype,
                       f"Test data should have dtype {expected_dtype}")
    
    # Additional validations
    assert_tensor_all_finite(data, "Test data should not contain NaN or Inf values")
    assert_not_empty(data, "Test data should not be empty")

def test_data_creation():
    """Test that test data creation functions work correctly."""
    # Create test data
    image_data = create_mock_image_data()
    
    # Validate properties
    validate_test_data(image_data, (4, 3, 224, 224), torch.float32)
```

### Parameterized Testing

```python
def test_model_with_different_input_sizes():
    """Test model with different input sizes."""
    test_cases = [
        {"shape": (1, 3, 224, 224), "name": "small_image"},
        {"shape": (4, 3, 512, 512), "name": "large_image"},
        {"shape": (8, 3, 224, 224), "name": "batched_images"},
    ]
    
    for case in test_cases:
        input_tensor = torch.randn(case["shape"])
        
        # Test that model can handle this input size
        model = create_model()
        output = model(input_tensor)
        
        # Validate output properties
        assert_is_not_none(output, f"Model should produce output for {case['name']}")
        assert_tensor_shape(output, output.shape, 
                           f"Output should have valid shape for {case['name']}")
```

## Test Dependencies

### Conditional Test Execution

```python
import torch
from src.inference_pio.test_utils import skip_test

def test_gpu_functionality():
    """Test GPU-specific functionality."""
    if not torch.cuda.is_available():
        skip_test("CUDA not available, skipping GPU tests")
    
    # GPU-specific tests here
    device = torch.device("cuda")
    tensor = torch.randn(10, 10).to(device)
    result = tensor @ tensor.T
    assert_tensor_shape(result, (10, 10), "GPU computation should produce correct result")

def test_specific_pytorch_version():
    """Test functionality that depends on specific PyTorch version."""
    import torch
    
    if torch.__version__ < "2.0.0":
        skip_test(f"Requires PyTorch 2.0+, got {torch.__version__}")
    
    # Tests that require PyTorch 2.0+ features
    compiled_model = torch.compile(create_model())
    result = compiled_model(torch.randn(1, 3, 224, 224))
    assert_is_not_none(result, "Compiled model should produce output")
```

### External Service Testing

```python
import requests
from src.inference_pio.test_utils import skip_test

def test_external_api_integration():
    """Test integration with external API."""
    try:
        response = requests.get("https://api.example.com/health", timeout=5)
        if response.status_code != 200:
            skip_test("External API not available, skipping integration test")
    except requests.RequestException:
        skip_test("External API not accessible, skipping integration test")
    
    # Proceed with integration tests
    api_client = APIClient("https://api.example.com")
    result = api_client.process_data(test_data)
    assert_true(result.success, "API call should succeed")
```

## Best Practices

### 1. Use Descriptive Test Names

```python
# Good
def test_model_inference_returns_tensor_with_correct_shape_when_given_valid_input():
    """Test model inference returns tensor with correct shape for valid input."""
    pass

# Better (more concise but still descriptive)
def test_model_inference_outputs_correct_shape():
    """Test model inference outputs correct tensor shape."""
    pass
```

### 2. Write Focused Tests

```python
# Good - Single responsibility
def test_tensor_addition_with_same_shapes():
    """Test tensor addition with same-shaped tensors."""
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6])
    expected = torch.tensor([5, 7, 9])
    
    result = tensor1 + tensor2
    assert_tensor_equal(result, expected)

def test_tensor_addition_broadcasting():
    """Test tensor addition with broadcasting."""
    tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    tensor2 = torch.tensor([1, 1, 1])  # 3
    expected = torch.tensor([[2, 3, 4], [5, 6, 7]])  # 2x3
    
    result = tensor1 + tensor2
    assert_tensor_equal(result, expected)
```

### 3. Use Meaningful Assertion Messages

```python
def test_model_prediction_accuracy():
    """Test model prediction accuracy."""
    model = create_model()
    test_data = load_test_dataset()
    
    predictions = model.predict(test_data.inputs)
    accuracy = calculate_accuracy(predictions, test_data.targets)
    
    expected_accuracy = 0.95
    assert_greater_equal(accuracy, expected_accuracy,
                        f"Model accuracy should be at least {expected_accuracy}, "
                        f"got {accuracy:.4f}. Model may need retraining.")
```

### 4. Test Edge Cases

```python
def test_edge_cases():
    """Test edge cases for tensor operations."""
    # Empty tensor
    empty_tensor = torch.empty(0)
    result = process_tensor(empty_tensor)
    assert_tensor_shape(result, (0,), "Empty tensor should be handled gracefully")
    
    # Single element tensor
    single_element = torch.tensor([42])
    result = process_tensor(single_element)
    assert_tensor_shape(result, (1,), "Single element tensor should be handled")
    
    # Very large tensor
    large_tensor = torch.randn(10000, 10000)
    result = process_tensor(large_tensor)
    assert_is_not_none(result, "Large tensor should be processed without error")
```

### 5. Test Error Conditions

```python
def test_error_conditions():
    """Test that appropriate errors are raised."""
    model = create_model()
    
    # Test with None input
    with assert_raises(ValueError):
        model.infer(None)
    
    # Test with wrong dtype
    wrong_dtype = torch.tensor([1, 2, 3], dtype=torch.int32)
    with assert_raises(TypeError):
        model.infer(wrong_dtype)
    
    # Test with wrong device
    if torch.cuda.is_available():
        cpu_tensor = torch.randn(1, 3, 224, 224)
        cuda_model = model.to("cuda")
        with assert_raises(RuntimeError):
            cuda_model(cpu_tensor)  # CPU tensor on CUDA model
```

## Common Pitfalls

### 1. Testing Implementation Instead of Behavior

```python
# Bad - Testing internal implementation details
def test_internal_variable_assignment():
    """Test that internal variable is assigned correctly."""
    model = Model()
    model._internal_var = "test"  # Testing private attribute
    assert_equal(model._internal_var, "test")

# Good - Testing observable behavior
def test_model_output_format():
    """Test that model output is in expected format."""
    model = Model()
    output = model.infer(test_input)
    assert_tensor_shape(output, expected_shape)
    assert_tensor_dtype(output, expected_dtype)
```

### 2. Flaky Tests

```python
# Bad - Test depends on external factors
def test_current_datetime_format():
    """Test current datetime format."""
    import datetime
    current_time = datetime.datetime.now()
    # This test could fail depending on when it's run
    assert_equal(current_time.hour, 10)  # Fails if run outside 10 AM

# Good - Test with controlled inputs
def test_datetime_formatting():
    """Test datetime formatting with controlled input."""
    test_datetime = datetime.datetime(2023, 1, 1, 10, 30, 45)
    formatted = format_datetime(test_datetime)
    assert_equal(formatted, "2023-01-01 10:30:45")
```

### 3. Slow Tests

```python
# Bad - Too many iterations in test
def test_model_convergence():
    """Test model convergence (too slow)."""
    model = create_model()
    # Running 10000 epochs just to test convergence
    for epoch in range(10000):
        loss = model.train_step(train_data)
    assert_less(loss, 0.1)

# Good - Test convergence logic without full training
def test_model_training_step():
    """Test single training step."""
    model = create_model()
    initial_loss = model.evaluate(test_data)
    loss_after_step = model.train_step(train_data)
    assert_less(loss_after_step, initial_loss)  # Loss should decrease
```

### 4. Insufficient Test Coverage

```python
# Bad - Only testing happy path
def test_divide_positive_numbers():
    """Test division with positive numbers."""
    result = divide(10, 2)
    assert_equal(result, 5)

# Good - Testing multiple scenarios
def test_divide_comprehensive():
    """Test division with various inputs."""
    # Normal case
    assert_equal(divide(10, 2), 5)
    
    # Negative numbers
    assert_equal(divide(-10, 2), -5)
    assert_equal(divide(10, -2), -5)
    
    # Division by zero
    with assert_raises(ZeroDivisionError):
        divide(10, 0)
    
    # Fractional results
    assert_close(divide(7, 3), 2.333333, rel_tol=1e-05)
```

---

Following these guidelines will help you write effective, maintainable, and reliable tests for the Inference-PIO project. Remember to focus on testing behavior rather than implementation, use descriptive names, and ensure your tests are independent and deterministic.
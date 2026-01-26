# Standardized Test Structure Guide

## Overview

This document describes the standardized test structure used throughout the Inference-PIO project. This structure ensures consistency, maintainability, and ease of execution across all components.

## Directory Structure

The project follows a hierarchical test structure organized by test type and component:

```
tests/
├── unit/               # Individual component tests
├── integration/        # Multi-component interaction tests
├── performance/        # Speed and resource usage tests
└── ...

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
    └── common/
        └── tests/
            ├── unit/
            ├── integration/
            └── performance/
```

## Test Categories

### Unit Tests (`unit/`)

Unit tests focus on testing individual functions, classes, or methods in isolation.

**Characteristics:**
- Test a single component or function
- Have minimal external dependencies
- Execute quickly (typically < 100ms per test)
- Are easy to debug when they fail
- Typically use mocking to isolate the component under test

**Examples:**
- `test_attention.py` - Tests attention mechanism implementations
- `test_config_loading.py` - Tests configuration loading functionality
- `test_tensor_operations.py` - Tests tensor manipulation functions

**Best Practices:**
- Test one logical concept per test function
- Use descriptive test names that explain the expected behavior
- Keep tests fast and deterministic
- Mock external dependencies when possible

### Integration Tests (`integration/`)

Integration tests verify that multiple components work together correctly.

**Characteristics:**
- Test interactions between multiple modules/components
- May involve real external dependencies
- Validate end-to-end workflows
- Ensure system components integrate properly
- Test plugin systems and cross-module functionality

**Examples:**
- `test_cross_modal_alignment_integration.py` - Tests cross-modal alignment with the full model
- `test_plugin_integration.py` - Tests plugin system integration
- `test_end_to_end.py` - Tests complete end-to-end functionality

**Best Practices:**
- Test realistic usage scenarios
- Validate data flow between components
- Test error handling across component boundaries
- Include setup and teardown phases when needed

### Performance Tests (`performance/`)

Performance tests evaluate the speed, memory usage, and efficiency of the system.

**Characteristics:**
- Measure execution time and resource usage
- Test scalability under load
- Validate optimization implementations
- Monitor performance regressions
- Benchmark different algorithms or approaches

**Examples:**
- `test_qwen3_vl_cuda_kernels.py` - Tests CUDA kernel performance
- `test_optimized_inference.py` - Tests inference performance with optimizations
- `test_memory_usage.py` - Tests memory consumption patterns

**Best Practices:**
- Establish baseline measurements
- Test with realistic data sizes
- Monitor for performance regressions
- Include statistical analysis when appropriate

## Naming Conventions

### File Names
- Use descriptive names that indicate the component being tested
- Include the test type in the name when it's not clear from the directory
- Use underscores to separate words: `test_attention_mechanism.py`

### Test Function Names
- Start with `test_` prefix
- Use descriptive names that explain what is being tested
- Include expected behavior in the name: `test_attention_returns_correct_shape`

### Class Names (for class-based tests)
- Use `Test` prefix: `TestAttentionMechanism`
- Use PascalCase: `TestClassificationModel`

## Test Organization

### Model-Specific Tests

Each model has its own test directory:

```
src/inference_pio/models/{model_name}/tests/
├── unit/
│   ├── test_attention.py
│   ├── test_config.py
│   └── test_preprocessing.py
├── integration/
│   ├── test_end_to_end.py
│   ├── test_plugin_integration.py
│   └── test_model_loading.py
└── performance/
    ├── test_inference_speed.py
    └── test_memory_usage.py
```

### Common Component Tests

Shared components have their own test directories:

```
src/inference_pio/common/tests/
├── unit/
│   ├── test_memory_manager.py
│   └── test_tensor_operations.py
├── integration/
│   ├── test_pipeline_parallel.py
│   └── test_model_surgery.py
└── performance/
    ├── test_optimization_benchmarks.py
    └── test_scaling_performance.py
```

## Test Implementation Guidelines

### Unit Tests

```python
from src.inference_pio.test_utils import assert_equal, assert_true, run_tests

def test_attention_layer_initialization():
    """Test that attention layer initializes with correct parameters."""
    layer = AttentionLayer(hidden_size=512, num_heads=8)
    
    assert_equal(layer.hidden_size, 512)
    assert_equal(layer.num_heads, 8)
    assert_true(layer.training)  # Default to training mode

if __name__ == '__main__':
    run_tests([test_attention_layer_initialization])
```

### Integration Tests

```python
from src.inference_pio.test_utils import assert_equal, assert_true, run_tests

def test_model_with_config_integration():
    """Test that model loads correctly with configuration."""
    config = ModelConfig(hidden_size=512, num_layers=6)
    model = Model(config)
    
    # Verify model properties match config
    assert_equal(model.config.hidden_size, 512)
    assert_equal(model.num_parameters(), calculate_expected_params(config))
    
    # Test forward pass
    input_tensor = torch.randn(1, 10, 512)
    output = model(input_tensor)
    assert_equal(output.shape, (1, 10, 512))

if __name__ == '__main__':
    run_tests([test_model_with_config_integration])
```

### Performance Tests

```python
import time
from src.inference_pio.test_utils import assert_less, run_tests

def test_inference_performance():
    """Test that inference completes within acceptable time."""
    model = load_test_model()
    input_data = generate_test_input(batch_size=32, seq_len=128)
    
    start_time = time.time()
    output = model(input_data)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # Assert that inference takes less than 1 second for this input size
    assert_less(execution_time, 1.0, 
              f"Inference took {execution_time}s, expected < 1.0s")

if __name__ == '__main__':
    run_tests([test_inference_performance])
```

## Running Tests

### By Category

To run tests by category:

```bash
# Run all unit tests
python -m src.inference_pio.test_discovery --category unit

# Run all integration tests  
python -m src.inference_pio.test_discovery --category integration

# Run all performance tests
python -m src.inference_pio.test_discovery --category performance
```

### By Component

To run tests for a specific component:

```bash
# Run all tests for a specific model
python -c "from src.inference_pio.test_discovery import run_model_tests; run_model_tests('qwen3_vl_2b')"

# Run plugin system tests
python -c "from src.inference_pio.test_discovery import run_plugin_system_tests; run_plugin_system_tests()"
```

### Manual Execution

Individual test files can be run directly:

```bash
python src/inference_pio/models/qwen3_vl_2b/tests/unit/test_attention.py
```

## Migration and Maintenance

### Adding New Tests

1. Identify the appropriate category (unit, integration, performance)
2. Place the test file in the corresponding directory
3. Follow the naming conventions
4. Use the appropriate assertion functions
5. Include meaningful test descriptions

### Updating Existing Tests

- Review tests when refactoring components
- Update assertions when interfaces change
- Maintain test isolation
- Ensure tests remain fast and reliable

## Quality Standards

### Test Coverage

- Aim for high unit test coverage (>80%) for critical components
- Include integration tests for key workflows
- Add performance tests for time-sensitive operations

### Test Reliability

- Tests should be deterministic
- Avoid flaky tests that sometimes pass and sometimes fail
- Minimize external dependencies
- Use appropriate mocking strategies

### Performance

- Unit tests should execute quickly
- Avoid heavy computations in unit tests
- Isolate slow operations in dedicated performance tests
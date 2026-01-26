# Standardized Test Structure

This project follows a standardized test structure to organize tests by their purpose and scope. This makes it easier to understand, maintain, and run tests efficiently.

## Directory Structure

```
tests/
├── unit/           # Tests for individual functions/components
├── integration/    # Tests for multiple components working together
└── performance/    # Tests for speed, memory usage, etc.
```

## Test Categories

### Unit Tests (`unit/`)
Unit tests focus on testing individual functions, classes, or methods in isolation. They:
- Test a single component or function
- Have minimal external dependencies
- Execute quickly
- Are easy to debug when they fail
- Typically use mocking to isolate the component under test

**Examples:**
- `test_attention.py` - Tests attention mechanism implementations
- `test_config_loading.py` - Tests configuration loading functionality

### Integration Tests (`integration/`)
Integration tests verify that multiple components work together correctly. They:
- Test interactions between multiple modules/components
- May involve real external dependencies
- Validate end-to-end workflows
- Ensure system components integrate properly
- Test plugin systems and cross-module functionality

**Examples:**
- `test_cross_modal_alignment_integration.py` - Tests cross-modal alignment with the full model
- `test_plugin_integration.py` - Tests plugin system integration
- `test_end_to_end.py` - Tests complete end-to-end functionality

### Performance Tests (`performance/`)
Performance tests evaluate the speed, memory usage, and efficiency of the system. They:
- Measure execution time and resource usage
- Test scalability under load
- Validate optimization implementations
- Monitor performance regressions
- Benchmark different algorithms or approaches

**Examples:**
- `test_qwen3_vl_cuda_kernels.py` - Tests CUDA kernel performance
- `test_optimized_inference.py` - Tests inference performance with optimizations

## Best Practices

### Writing New Tests
1. **Choose the right category**: Place new tests in the most appropriate directory based on their scope
2. **Follow naming conventions**: Use descriptive names that indicate what is being tested
3. **Keep tests focused**: Each test should validate a specific behavior or functionality
4. **Use appropriate fixtures**: Set up test data and environments consistently

### Running Tests
To run tests by category:
```bash
# Run all unit tests
pytest tests/unit/

# Run all integration tests  
# pytest tests/integration/

# Run all performance tests
pytest tests/performance/

# Run tests for a specific model
pytest src/inference_pio/models/qwen3_vl_2b/tests/
```

## Migration Notes
Existing tests have been automatically categorized based on:
- File names containing keywords like `_unit`, `_integration`, `_performance`
- Content analysis looking for integration/performance related terms
- Default assignment to `unit` for tests that don't clearly fit other categories

Some manual review may be needed to ensure tests are in the most appropriate category.
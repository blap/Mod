# Test Structure Guide

The Inference-PIO project uses a standardized testing structure to ensure reliability and maintainability across all supported models.

## Directory Organization

### Global Tests (`tests/`)
These tests cover the core framework, common utilities, and cross-cutting concerns.

- `tests/unit/`: Unit tests for common components.
- `tests/integration/`: Integration tests for the plugin system and common workflows.
- `tests/performance/`: Performance regression tests for the core system.

### Model-Specific Tests (`src/inference_pio/models/<model>/tests/`)
Each model plugin is self-contained and includes its own test suite.

- `tests/unit/`: Tests for model-specific logic (e.g., adapters, kernels).
- `tests/integration/`: Tests verifying the model integrates correctly with the plugin system.
- `tests/performance/`: Benchmarks and performance tests specific to the model (e.g., CUDA kernel speed).

## Running Tests

To run all tests:
```bash
python -m pytest
```

To run tests for a specific model:
```bash
python -m pytest src/inference_pio/models/qwen3_vl_2b/tests/
```

## Best Practices

1.  **Isolation:** Model-specific tests should not depend on other models.
2.  **Naming:** Use `test_*.py` for all test files.
3.  **Mocking:** Use mocks for heavy resources (like model weights) in unit tests.

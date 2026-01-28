# Test Structure Guide

The Inference-PIO project uses a standardized testing structure to ensure reliability and maintainability across all supported models.

## Directory Organization

### Global Tests (`tests/`)
These tests cover the core framework, common utilities, and cross-cutting concerns.

- `tests/unit/`: Unit tests for common components.
  - `tests/unit/common/`: Tests for shared utilities like configuration and optimization.
- `tests/integration/`: Integration tests for the plugin system and common workflows.
  - `tests/integration/common/`: Tests for shared integration scenarios.
- `tests/performance/`: Performance regression tests for the core system.

### Model-Specific Tests (`tests/models/`)
Model-specific tests have been moved to a centralized directory for easier management, although some plugin-specific logic may remain near the source.

- `tests/models/`: Contains tests specific to particular model implementations (e.g., `test_qwen3_vl_2b_model.py`).

## Running Tests

To run tests, use the optimized test runner script:

```bash
python scripts/run_tests.py
```

To run tests for a specific directory:

```bash
python scripts/run_tests.py -d tests/unit/common
```

To list available tests:

```bash
python scripts/run_tests.py --list
```

## Best Practices

1.  **Isolation:** Model-specific tests should not depend on other models.
2.  **Naming:** Use `test_*.py` for all test files.
3.  **Mocking:** Use mocks for heavy resources (like model weights) in unit tests.
4.  **Imports:** Always use absolute imports rooted at `src` (e.g., `from src.inference_pio.common import ...`).

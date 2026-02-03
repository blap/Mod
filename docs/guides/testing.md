# Testing Guide

This comprehensive guide covers the testing ecosystem of the Inference-PIO project, including the custom framework, execution tools, and best practices.

## 1. Overview

The project uses a custom-built testing framework (`tests.utils`) designed for minimal dependencies and fast execution. It avoids external runners like `pytest` for the core execution logic, though `pytest` plugins are supported for reporting.

### Core Philosophy
*   **Minimal Dependencies:** Rely on internal tools where possible.
*   **Fast Feedback:** Unit tests should be <100ms.
*   **Real Code:** Use real components over mocks for integration tests.

## 2. Test Execution

The primary entry point for running tests is `src/inference_pio/core/tools/scripts/testing/run_tests.py`.

### Basic Usage
```bash
# Run all tests
python src/inference_pio/core/tools/scripts/testing/run_tests.py

# Run a specific category
python src/inference_pio/core/tools/scripts/testing/run_tests.py --category unit

# Run tests in a specific directory
python src/inference_pio/core/tools/scripts/testing/run_tests.py --directory src/inference_pio/tests/unit/common

# List available tests
python src/inference_pio/core/tools/scripts/testing/run_tests.py --list
```

### Advanced Options
*   `--parallel`: Run tests concurrently (if supported).
*   `--no-cache`: Ignore cached results.
*   `--report <file>`: Generate a JSON report.

## 3. Writing Tests

Tests are simple Python functions or classes. They must be placed in `src/inference_pio/tests/` or model-specific `src/inference_pio/models/<model>/tests/` directories and start with `test_`.

### The `tests.utils` Framework
Import utilities from `src.inference_pio.tests.utils.test_utils`.

```python
from tests.utils.test_utils import assert_equal, assert_true, run_tests

def test_example():
    """Example test function."""
    result = 2 + 2
    assert_equal(result, 4, "Math should work")

if __name__ == '__main__':
    run_tests([test_example])
```

### Available Assertions

| Assertion | Description |
|-----------|-------------|
| `assert_true(cond, msg)` | Verify condition is True |
| `assert_false(cond, msg)` | Verify condition is False |
| `assert_equal(a, b, msg)` | Verify a == b |
| `assert_not_equal(a, b, msg)` | Verify a != b |
| `assert_is_none(val, msg)` | Verify val is None |
| `assert_is_not_none(val, msg)` | Verify val is not None |
| `assert_in(item, container, msg)` | Verify item in container |
| `assert_raises(exc, func, *args)` | Verify func raises exception |
| `assert_tensor_equal(t1, t2, msg)` | Verify tensor equality (for PyTorch) |
| `assert_file_exists(path, msg)` | Verify file existence |

## 4. Test Categories & Structure

Tests are organized in `src/inference_pio/tests/`:

*   **`src/inference_pio/tests/unit/`**: Isolated logic. Fast. Mock external I/O.
*   **`src/inference_pio/tests/integration/`**: Component interaction. Real I/O allowed.
*   **`src/inference_pio/tests/performance/`**: Timing and resource tracking.

### Model-Specific Tests
Model tests follow the self-contained architecture and are located in `src/inference_pio/models/<model>/tests/`.

The structure is as follows:

```
src/
└── inference_pio/
    ├── models/
    │   ├── glm_4_7_flash/
    │   │   └── tests/
    │   │       ├── unit/
    │   │       ├── integration/
    │   │       └── performance/
    │   ├── qwen3_0_6b/
    │   │   └── tests/
    │   │       ├── unit/
    │   │       ├── integration/
    │   │       └── performance/
    │   └── ...
    └── tests/
        ├── unit/                       # Shared unit tests
        ├── integration/                # Shared integration tests
        └── performance/                # Shared performance tests
```

Each model plugin is completely independent with its own tests and benchmarks.

## 5. Reporting & CI/CD

The system generates reports in JSON, XML, HTML, and Markdown.

### Generation
The `tests.utils.reporting` module handles report generation.
To generate reports during a run:

```bash
python scripts/ci_test_reporting.py --pytest-args "tests/unit" --ci-platform github-actions
```

### CI/CD Integration
*   **GitHub Actions:** Automatically summarizes test results and posts them to PRs.
*   **Performance Tracking:** Checks for regressions against historical baselines.

## 6. Examples

### Integration Test Example
```python
from tests.utils.test_utils import assert_true, run_tests
from src.inference_pio.common.config_manager import ConfigManager

def test_config_integration():
    """Verify config manager loads default values."""
    manager = ConfigManager()
    config = manager.get_config("default")
    assert_true(config.is_valid, "Config should be valid")
```

### Performance Test Example
```python
import time
from tests.utils.test_utils import assert_less, run_tests

def test_inference_speed():
    start = time.time()
    # model.infer(...)
    duration = time.time() - start
    assert_less(duration, 0.5, "Inference took too long")
```

## 7. Advanced Features

### Unified Test Discovery
The `UnifiedTestDiscovery` class automatically finds tests and benchmarks using flexible naming patterns:
*   Tests: `test_*`, `should_*`, `verify_*`.
*   Benchmarks: `benchmark_*`, `run_*`, `measure_*`.

### Test Optimization
The `optimized_test_runner.py` script (wrapped by `scripts/run_tests.py`) provides:
*   **Parallel Execution:** Uses `max_workers` to run independent tests concurrently.
*   **Result Caching:** Caches successful test results for 24 hours to speed up local development cycles. Use `--no-cache` to force re-execution.

## 8. Test Class Hierarchy

The project provides standardized base classes in `src.inference_pio.tests.base` to ensure consistency and reuse.

### 8.1 Unit Tests (`unit_test_base.py`)
- **`BaseUnitTest`**: General purpose unit tests.
- **`ModelUnitTest`**: Specialized for model plugins.
- **`PluginUnitTest`**: Specialized for plugin components.

### 8.2 Integration Tests (`integration_test_base.py`)
- **`BaseIntegrationTest`**: General integration tests.
- **`ModelIntegrationTest`**: Model integration scenarios.
- **`PipelineIntegrationTest`**: Pipeline component integration.

### 8.3 Functional Tests (`functional_test_base.py`)
- **`BaseFunctionalTest`**: End-to-end user flows.
- **`ModelFunctionalTest`**: Full model workflows.
- **`SystemFunctionalTest`**: Complete system simulation.

### 8.4 Benchmarks (`benchmark_test_base.py`)
- **`BaseBenchmarkTest`**: Performance measurement base.
- **`ModelBenchmarkTest`**: Model performance tracking.
- **`SystemBenchmarkTest`**: System-wide benchmarks.

### 8.5 Regression Tests (`regression_test_base.py`)
- **`BaseRegressionTest`**: Baseline comparison.
- **`ModelRegressionTest`**: Output consistency checks.
- **`FeatureRegressionTest`**: Feature stability checks.

### Usage Example

```python
from src.inference_pio.tests.base.unit_test_base import ModelUnitTest

class TestMyModel(ModelUnitTest):
    def get_model_plugin_class(self):
        from src.inference_pio.models.my_model.plugin import MyModelPlugin
        return MyModelPlugin

    def test_required_functionality(self):
        # Specific test implementation
        pass
```

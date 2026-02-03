# Testing Standards

**Reference Guide:** For instructions on writing and running tests, see [Testing Guide](../guides/testing.md).

## 1. Directory Structure Requirements
*   All global tests **MUST** be located in `src/inference_pio/tests/`.
*   Model-specific tests **MUST** be located in `src/inference_pio/models/<model>/tests/`.
*   Tests **MUST** be categorized into `unit/`, `integration/`, or `performance/`.

## 2. Naming Conventions
*   **Files:** Must start with `test_` (e.g., `test_attention.py`).
*   **Functions:** Must start with `test_` (e.g., `test_forward_pass`).
*   **Classes:** Must start with `Test` (e.g., `TestAttention`).

## 3. Implementation Rules
*   **Imports:** Use absolute imports starting from `src` (e.g., `from src.inference_pio import ...`).
*   **Assertions:** You **MUST** use `tests.utils.test_utils` assertions (`assert_equal`, `assert_true`) instead of Python's `assert` statement for better error reporting.
*   **Dependencies:** Unit tests **MUST NOT** rely on external services or large model weights (use mocks or small fixtures).
*   **Cleanup:** Tests creating temporary files **MUST** clean them up (use `tempfile` or `try/finally`).

## 4. Performance Tests
*   Must track metrics (time, memory) explicitly.
*   Must be deterministic where possible (seed RNGs).

## 5. Method Signatures

### Unit Tests
*   **Setup/Teardown:** `setUp(self)`, `tearDown(self)`, `setUpClass(cls)`, `tearDownClass(cls)`.
*   **Test Methods:** `test_test_name(self)`. Arguments: `self` only. Return: None.

### Benchmarks
*   **Setup:** `setUp(self)`, `setUpClass(cls)`.
*   **Benchmark Methods:** `test_benchmark_name(self)`. Arguments: `self` only. Return: Dictionary with results (if using custom runner) or None (if using assertions).
*   **Measurement Helper:** `run_benchmark(self, target_function, *args, **kwargs)`. Return: Dictionary with performance metrics.

### Utility Functions
*   **Resource Creation:** `create_test_model(model_name, config=None)`, `create_test_plugin(plugin_type, config=None)`.
*   **Validation:** `validate_model_output(output, expected_type=None)`, `validate_plugin_interface(plugin_instance)`.

### Fixtures
*   **Data:** `get_sample_text_data(size="small")`, `get_sample_tensor_data(shape=(10, 10))`.

### Custom Assertions
*   `assert_model_initialized(self, model)`
*   `assert_plugin_has_method(self, plugin, method_name)`

## 6. General Guidelines
1.  All test methods must start with `test_`.
2.  Methods should have descriptive names indicating their function.
3.  Avoid unnecessary parameters.
4.  Use type hinting for clarity.
5.  Return values should be consistent with the method's purpose.

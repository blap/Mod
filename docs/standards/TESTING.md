# Testing Standards

**Reference Guide:** For instructions on writing and running tests, see [Testing Guide](../guides/testing.md).

## 1. Directory Structure Requirements
*   All global tests **MUST** be located in `tests/`.
*   Model-specific tests **MAY** be located in `src/inference_pio/models/<model>/tests/` but moving them to `tests/models/<model>/` is preferred.
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

# Qwen3-0.6B Model Tests

This directory contains all tests specific to the Qwen3-0.6B model. All tests in this directory are independent and do not rely on other model implementations.

## Directory Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for component interactions
- `performance/` - Performance and benchmark tests
- `__main__.py` - Test discovery and execution entry point

## Running Tests

To run all tests for this model:

```bash
cd tests/models/qwen3_0_6b
python __main__.py
```

Or run specific test files directly:

```bash
python test_interface_compliance.py
```

## Independence

Each model's tests are completely independent and can be run separately without affecting other models.
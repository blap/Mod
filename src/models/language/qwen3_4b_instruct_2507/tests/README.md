# Qwen3-4B-Instruct-2507 Model Tests

This directory contains all tests specific to the Qwen3-4B-Instruct-2507 model. All tests in this directory are independent and do not rely on other model implementations.

## Directory Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for component interactions  
- `performance/` - Performance and benchmark tests
- `__main__.py` - Test discovery and execution entry point

## Running Tests

To run all tests for this model:

```bash
cd src/models/language/qwen3_4b_instruct_2507/tests
python __main__.py
```

Or run specific test files directly:

```bash
python test_interface_compliance.py
```

## Independence

Each model's tests are completely independent and can be run separately without affecting other models.
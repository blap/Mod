# Inference-PIO Tests

This directory contains tests for the Inference-PIO self-contained plugin architecture.

## Test Structure

The tests are organized to verify the functionality of each self-contained plugin:

- `basic_tests.py`: Contains basic unit tests for each model plugin
- `__main__.py`: Test runner to execute tests

## Running Tests

To run all tests:

```bash
cd src/inference_pio/tests
python -m pytest
```

Or using the test runner:

```bash
cd src/inference_pio/tests
python __main__.py --all
```

To run tests for a specific model:

```bash
python __main__.py --suite glm_4_7
python __main__.py --suite qwen3_coder_30b
python __main__.py --suite qwen3_vl_2b
python __main__.py --suite qwen3_4b_instruct_2507
```

## Test Categories

- **Unit Tests**: Verify individual plugin functionality
- **Integration Tests**: Verify plugins work with the main system
- **Performance Tests**: Measure performance characteristics
- **Compatibility Tests**: Verify compatibility with generic functions

## Adding New Tests

When adding new models to the self-contained architecture, add corresponding tests in the basic_tests.py file following the existing pattern.
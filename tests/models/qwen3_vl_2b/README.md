# Qwen3-VL-2B Model Tests

This directory contains all tests specific to the Qwen3-VL-2B model. All tests in this directory are independent and do not rely on other model implementations.

## Directory Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for component interactions  
- `performance/` - Performance and benchmark tests
- `multimodal_projector/` - Tests for multimodal projector components
- `vision_transformer/` - Tests for vision transformer components
- `visual_resource_compression/` - Tests for visual resource compression
- `__main__.py` - Test discovery and execution entry point

## Running Tests

To run all tests for this model:

```bash
cd tests/models/qwen3_vl_2b
python __main__.py
```

Or run specific test files directly:

```bash
python final_verification_image_tokenization.py
```

## Independence

Each model's tests are completely independent and can be run separately without affecting other models.
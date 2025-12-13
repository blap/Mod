# Quick Start Guide

This guide provides a quick example to get you started with the Qwen3-VL multimodal model.

## Basic Usage

### 1. Import the Model

```python
from qwen3_vl import Qwen3VLModel

# Initialize the model
model = Qwen3VLModel.from_pretrained("path/to/model")
```

### 2. Prepare Input Data

```python
# Example with text and image input
text_input = "Describe this image in detail"
image_input = "path/to/image.jpg"

# Process the inputs
inputs = model.preprocess(text_input, image_input)
```

### 3. Run Inference

```python
# Generate output
output = model.generate(inputs)

# Print the result
print(output)
```

## Using the CLI

The project also provides a command-line interface:

```bash
qwen3-vl-infer --model-path path/to/model --text "Describe this image" --image path/to/image.jpg
```

## Running Tests

To verify your installation:

```bash
# Run unit tests
python -m pytest tests/unit/ -v

# Run a specific test
python -m pytest tests/unit/test_model.py -v
```

## Next Steps

- Check the [Architecture Documentation](../architecture/README.md) for system design details
- Review the [Performance Optimizations](../performance/README.md) for efficiency improvements
- Look at the [API Reference](../qwen3_vl/README.md) for detailed function documentation
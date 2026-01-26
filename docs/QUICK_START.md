# Quick Start Guide for Inference-PIO

## Overview

This guide will help you get started with the Inference-PIO system quickly. Inference-PIO is an advanced inference system featuring a self-contained plugin architecture where each model has its own complete implementation with all necessary components in a single directory.

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (recommended for performance)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/inference-pio/inference-pio.git
cd inference-pio
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Basic Usage

### Loading and Using a Model

Here's how to load and use a model with the Inference-PIO system:

```python
from inference_pio import create_glm_4_7_flash_plugin

# Create a plugin instance
plugin = create_glm_4_7_flash_plugin()

# Initialize the plugin
plugin.initialize()

# Load the model
plugin.load_model()

# Perform inference
result = plugin.infer("What is the capital of France?")

# Print the result
print(result)

# Clean up resources
plugin.cleanup()
```

### Using Different Models

The system supports multiple models. Here are examples for each:

#### GLM-4.7-Flash (Advanced reasoning)
```python
from inference_pio import create_glm_4_7_flash_plugin

plugin = create_glm_4_7_flash_plugin()
plugin.initialize()
plugin.load_model()
result = plugin.infer("Solve this mathematical equation: 2x + 5 = 15")
plugin.cleanup()
```

#### Qwen3-Coder-30B (Code generation)
```python
from inference_pio import create_qwen3_coder_30b_plugin

plugin = create_qwen3_coder_30b_plugin()
plugin.initialize()
plugin.load_model()
result = plugin.generate_text("Write a Python function to calculate factorial")
plugin.cleanup()
```

#### Qwen3-VL-2B (Vision-Language)
```python
from inference_pio import create_qwen3_vl_2b_instruct_plugin

plugin = create_qwen3_vl_2b_instruct_plugin()
plugin.initialize()
plugin.load_model()

# For multimodal input
result = plugin.infer({
    "text": "Describe this image:",
    "image": "path/to/image.jpg"
})

plugin.cleanup()
```

#### Qwen3-4B-Instruct-2507 (Instruction following)
```python
from inference_pio import create_qwen3_4b_instruct_2507_plugin

plugin = create_qwen3_4b_instruct_2507_plugin()
plugin.initialize()
plugin.load_model()

# Chat completion
messages = [
    {"role": "user", "content": "Explain quantum computing in simple terms"}
]
result = plugin.chat_completion(messages)

plugin.cleanup()
```

## Advanced Usage

### Custom Configuration

You can customize the model behavior with configuration options:

```python
from inference_pio import create_glm_4_7_flash_plugin

plugin = create_glm_4_7_flash_plugin()
plugin.initialize(
    device="cuda:0",           # Specify device
    max_new_tokens=1024,       # Max tokens to generate
    temperature=0.7,           # Sampling temperature
    do_sample=True,            # Enable sampling
    top_p=0.9,                 # Top-p sampling
    top_k=50,                  # Top-k sampling
    enable_memory_management=True,  # Enable memory optimizations
    enable_tensor_paging=True,      # Enable tensor paging
    enable_disk_offloading=True,    # Enable disk offloading
    enable_model_surgery=True,      # Enable model surgery
    use_flash_attention_2=True,     # Enable FlashAttention 2.0
    use_sparse_attention=True,      # Enable sparse attention
    use_tensor_parallelism=False     # Disable tensor parallelism (not needed for this model)
)

plugin.load_model()
result = plugin.infer("Generate a creative story")
plugin.cleanup()
```

### Using the Plugin Manager

For managing multiple plugins:

```python
from inference_pio import get_plugin_manager

# Get the global plugin manager
pm = get_plugin_manager()

# Activate a plugin
pm.activate_plugin("glm_4_7_flash")

# Execute inference
result = pm.execute_plugin("glm_4_7_flash", "Your input text")

# List available plugins
plugins = pm.list_plugins()
print("Available plugins:", plugins)
```

## Understanding the Architecture

### Self-Contained Plugins

Each model in Inference-PIO is completely self-contained:

- **Isolated**: Each model exists in its own directory with all necessary components
- **Independent**: Models can be developed and optimized separately
- **Modular**: Easy to add new models following the same pattern

### Directory Structure

```
src/inference_pio/models/[model_name]/
├── __init__.py
├── config.py
├── model.py
├── plugin.py
├── safe_model.py
├── architecture_registration.py
├── config_integration.py
├── attention/
├── benchmarks/
├── cuda_kernels/
├── fused_layers/
├── kv_cache/
├── linear_optimizations/
├── optimizations/
├── plugin_modules/
├── prefix_caching/
├── rotary_embeddings/
├── specific_optimizations/
├── tensor_parallel/
└── tests/
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce batch size
   - Use CPU instead of GPU: `plugin.initialize(device="cpu")`
   - Enable memory optimizations

2. **Model Loading Failure**:
   - Check if model files exist at the specified path
   - Verify PyTorch and CUDA compatibility

3. **Slow Performance**:
   - Ensure CUDA is properly configured
   - Use appropriate model size for your hardware
   - Enable optimizations in configuration

### Performance Tips

- Use GPU when available for better performance
- Choose the right model size for your hardware
- Configure memory management options appropriately
- Use batch processing when possible

## Next Steps

After completing this quick start:

1. Read the [Developer Guide](DEVELOPER_GUIDE.md) for more detailed information
2. Explore the [Architecture Overview](ARCHITECTURE_OVERVIEW.md) for system design details
3. Look at the [API Reference](PLUGIN_SYSTEM_API.md) for detailed plugin system information
4. Contribute to the project by following the [Contribution Guidelines](CONTRIBUTING.md)

## Support

If you encounter issues:

1. Check the existing documentation
2. Look at the examples in the `examples/` directory
3. Open an issue on GitHub if you can't find a solution
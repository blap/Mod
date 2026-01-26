# Inference-PIO: Self-Contained Plugin Architecture

Inference-PIO is an advanced inference system featuring a self-contained plugin architecture where each model has its own complete implementation with all necessary components in a single directory. This architecture enables maximum modularity, maintainability, and optimization for each specific model while providing a unified interface.

## Documentation

For comprehensive documentation, please refer to:

- [Quick Start Guide](./docs/QUICK_START.md) - Get started with Inference-PIO quickly
- [Architecture Overview](./docs/ARCHITECTURE_OVERVIEW.md) - Understand the system architecture
- [Developer Guide](./docs/DEVELOPER_GUIDE.md) - Detailed information for developers
- [Plugin System API](./docs/PLUGIN_SYSTEM_API.md) - Documentation for the plugin system
- [Testing Framework Guide](./docs/TESTING_FRAMEWORK_GUIDE.md) - Comprehensive guide to the custom testing framework
- [Contributing Guidelines](./docs/CONTRIBUTING.md) - How to contribute to the project

## Key Features

- **Self-Contained Models**: Each model plugin contains all its specific implementations
- **Modularity**: Each model is completely isolated with its own dependencies and optimizations
- **Scalability**: Easy to add new models following the same pattern without affecting existing ones
- **Optimization**: Each model can have specific optimizations tailored to its architecture
- **Common Components**: Shared functionality available in the `common/` directory to avoid duplication
- **Unified Test Discovery**: Comprehensive system for discovering and running both tests and benchmarks
- **Flexible Naming Conventions**: Support for multiple test and benchmark naming patterns
- **Standardized Structure**: Organized test and benchmark directories for better maintainability

## Supported Models

- **GLM-4.7-Flash**: Advanced reasoning language model with 4.7B parameters, featuring MoE architecture with 64 experts and optimized for high-performance inference
- **Qwen3-Coder-30B**: Code generation and understanding model with 30B parameters
- **Qwen3-VL-2B**: Vision-language multimodal model with 2B parameters
- **Qwen3-4B-Instruct-2507**: Instruction-following language model with 4B parameters

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/inference-pio/inference-pio.git
cd inference-pio

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -r requirements.txt
pip install -e .
```

### From PyPI

```bash
pip install inference-pio
```

## Usage

### Basic Usage

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
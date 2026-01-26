# Developer Guide for Inference-PIO

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Getting Started](#getting-started)
4. [Plugin Development](#plugin-development)
5. [Model Development](#model-development)
6. [Common Components](#common-components)
7. [Testing](#testing)
8. [Performance Optimization](#performance-optimization)
9. [Design Patterns](#design-patterns)
10. [Best Practices](#best-practices)

## Introduction

This guide provides comprehensive information for developers working with the Inference-PIO system. The system follows a self-contained plugin architecture where each model has its own directory with all necessary components.

## Architecture Overview

### Self-Contained Plugin Architecture

The Inference-PIO system is built around a self-contained plugin architecture where each model implementation is completely isolated:

- Each model has its own directory with all necessary components
- Models are independent and can be developed separately
- Common functionality is shared through the `common/` module
- A unified plugin interface ensures consistency across models

### Directory Structure

```
├── benchmarks/             # Benchmark execution scripts
├── benchmark_results/      # Benchmark results and reports
├── config/                 # Configuration files
├── dev_artifacts/          # Development artifacts and temporary files
├── dev_tools/              # Development tools and utilities
├── docs/                   # Documentation files
├── examples/               # Example usage files
├── offload/                # Offload-related files
├── pipeline_checkpoints/   # Pipeline checkpoint files
├── plugin_configs/         # Plugin configuration files
├── src/                    # Source code
│   └── inference_pio/      # Main package
│       ├── __init__.py     # Package initialization and exports
│       ├── __main__.py     # Main entry point for CLI
│       ├── common/         # Common utilities
│       ├── design_patterns/ # Design pattern implementations
│       ├── models/         # Model implementations
│       │   ├── __init__.py # Models package initialization
│       │   ├── glm_4_7_flash/    # GLM-4.7-Flash model files
│       │   ├── qwen3_4b_instruct_2507/  # Qwen3-4B-Instruct-2507 model files
│       │   ├── qwen3_coder_30b/         # Qwen3-Coder-30B model files
│       │   └── qwen3_vl_2b/             # Qwen3-VL-2B model files
│       ├── plugin_system/  # Plugin system implementation
│       ├── tests/          # Core tests
│       ├── test_discovery.py # Test discovery system
│       └── test_utils.py   # Custom test utilities
├── test_shards/            # Test shards
├── tensor_swap/            # Tensor swap implementations
└── text_tensor_swap/       # Text tensor swap implementations
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (for GPU acceleration)

### Installation

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

### Basic Usage

```python
from inference_pio import (
    create_glm_4_7_flash_plugin,
    create_qwen3_coder_30b_plugin,
    create_qwen3_vl_2b_instruct_plugin,
    create_qwen3_4b_instruct_2507_plugin
)

# Create and use a plugin
plugin = create_glm_4_7_flash_plugin()
plugin.initialize()
plugin.load_model()
result = plugin.infer("Your input text")
plugin.cleanup()
```

## Plugin Development

### Plugin Interface

All plugins must implement the `ModelPluginInterface`:

```python
from inference_pio.common.base_plugin_interface import ModelPluginInterface

class MyModelPlugin(ModelPluginInterface):
    def __init__(self):
        # Initialize with metadata
        super().__init__(metadata)
        self._model = None
        self._tokenizer = None

    def initialize(self, **kwargs) -> bool:
        # Initialize the plugin with configuration
        pass

    def load_model(self, config=None) -> nn.Module:
        # Load the model with the given configuration
        pass

    def infer(self, data: Any) -> Any:
        # Perform inference on the given data
        pass

    def cleanup(self) -> bool:
        # Clean up resources used by the plugin
        pass

    def execute(self, *args, **kwargs) -> Any:
        # Execute the model with given inputs
        pass

    def get_model_info(self) -> dict:
        # Get information about the loaded model
        pass

    def update_config(self, **kwargs) -> bool:
        # Update the plugin configuration
        pass

    def supports_config(self, config) -> bool:
        # Check if this plugin supports the given configuration
        pass
```

### Creating a Factory Function

Each plugin should have a factory function:

```python
def create_my_model_plugin() -> MyModelPlugin:
    return MyModelPlugin()
```

## Model Development

### Model Directory Structure

Each model should follow this structure:

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

### Model Configuration

Create a configuration class for your model:

```python
# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class MyModelConfig:
    model_path: str = "path/to/model"
    device: str = "cuda:0"
    max_new_tokens: int = 512
    temperature: float = 0.7
    # Add model-specific configuration parameters
```

### Model Implementation

Implement your model in the `model.py` file:

```python
# model.py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Initialize model components

    def forward(self, *args, **kwargs):
        # Implement forward pass
        pass

    def get_tokenizer(self):
        # Return the tokenizer for this model
        pass
```

### Plugin Implementation

Implement the plugin interface in `plugin.py`:

```python
# plugin.py
from inference_pio.common.base_plugin_interface import ModelPluginInterface
from .model import MyModel
from .config import MyModelConfig

class MyModelPlugin(ModelPluginInterface):
    def __init__(self):
        # Initialize with metadata
        super().__init__(metadata)
        self._model = None
        self._tokenizer = None
        self._config = MyModelConfig()

    def initialize(self, **kwargs) -> bool:
        # Initialize the plugin with configuration
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        # Load the model
        self.load_model()
        return True

    def load_model(self, config=None) -> nn.Module:
        if config is not None:
            self._config = config

        self._model = MyModel(self._config)
        return self._model

    def infer(self, data: Any) -> Any:
        if self._model is None:
            self.load_model()

        # Implement inference logic
        pass

    def cleanup(self) -> bool:
        if self._model is not None:
            del self._model
            self._model = None
        return True

def create_my_model_plugin() -> MyModelPlugin:
    return MyModelPlugin()
```

## Common Components

### Base Plugin Interface

The `base_plugin_interface.py` file contains the base classes for plugins:

- `ModelPluginInterface`: Base interface for all model plugins
- `ModelPluginMetadata`: Metadata container for plugins

### Memory Management

The system provides utilities for memory management:

```python
from inference_pio.common.memory_manager import MemoryManager

# Create a memory manager
memory_manager = MemoryManager(max_memory_ratio=0.8)
```

### Model Surgery

The system provides utilities for model optimization:

```python
from inference_pio.common.model_surgery import apply_model_surgery

# Apply model surgery to optimize the model
optimized_model = apply_model_surgery(original_model, components_to_remove=['dropout'])
```

## Testing

### Test Utilities

The project uses custom test utilities in `src/inference_pio/test_utils.py`:

```python
from inference_pio.test_utils import (
    assert_true, assert_false, assert_equal, assert_is_not_none,
    assert_is_instance, assert_in, assert_greater, run_tests
)

def test_example():
    assert_true(1 + 1 == 2, "Basic math should work")
    assert_is_not_none("hello", "String should not be None")

if __name__ == '__main__':
    run_tests([test_example])
```

### Running Tests

To run all tests using the discovery system:

```python
from src.inference_pio.test_discovery import run_all_project_tests

run_all_project_tests()
```

Or run specific model tests:

```python
from src.inference_pio.test_discovery import run_model_tests

run_model_tests('glm_4_7_flash')
```

## Performance Optimization

### Attention Mechanisms

Each model can implement various attention mechanisms:

- FlashAttention 2.0
- Sparse Attention
- Sliding Window Attention
- Multi-Query/Grouped-Query Attention
- Paged Attention

### Memory Optimizations

- KV-Cache Compression
- Paged KV-Cache
- Prefix Caching
- Gradient Checkpointing
- Tensor Parallelism

### Hardware Optimizations

- CUDA Kernels
- Fused Operations
- Mixed Precision
- Tensor Cores

## Design Patterns

The system implements several design patterns:

- **Factory Pattern**: For creating plugin instances
- **Singleton Pattern**: For plugin manager
- **Strategy Pattern**: For different optimization strategies
- **Adapter Pattern**: For integrating different model architectures
- **Decorator Pattern**: For adding functionality to plugins

## Best Practices

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all classes and functions
- Keep functions focused and small (preferably under 50 lines)
- Use meaningful variable and function names
- Follow the DRY (Don't Repeat Yourself) principle
- Use constants for magic numbers and strings

### Model Development

- Each model should be self-contained in its own directory
- Follow the same directory structure across all models
- Implement the standardized plugin interface
- Include comprehensive tests
- Document model-specific parameters and features
- Implement proper error handling

### Plugin Development

- Implement all required methods from the plugin interface
- Handle different input types appropriately
- Implement proper resource cleanup
- Include comprehensive error handling
- Follow the factory pattern for plugin creation
- Use the metadata system to describe your plugin

### Testing

- Write comprehensive unit tests
- Include integration tests
- Test error conditions
- Verify resource cleanup
- Test with different input types and sizes
- Include performance benchmarks
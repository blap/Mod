# Creating a Model Plugin

## Overview

This guide outlines how to create a new model plugin for the Inference-PIO system. The project strictly adheres to a **Dependency-Free** and **Self-Contained** architecture.

**CRITICAL RULE:** Do NOT import `torch`, `numpy`, `transformers`, or `accelerate`. All tensor operations must use the custom `backend.Tensor` and related C/CUDA kernels.

## 1. Directory Structure

Create a new directory in `src/inference_pio/models/<model_name>/`. It must contain:

```
src/inference_pio/models/<model_name>/
├── __init__.py         # Exports the Model and Config classes
├── config.py           # Configuration dataclass inheriting BaseConfig
├── model.py            # Core model logic using backend.Tensor
├── plugin.py           # Adapter implementing ModelPluginInterface
├── plugin_manifest.json # Metadata for discovery
├── architecture/       # (Optional) Model-specific layers
├── tests/              # (Mandatory) Unit tests
└── README.md           # Documentation
```

## 2. Implementing the Configuration (`config.py`)

Define your model's hyperparameters using `dataclasses`. Inherit from `BaseConfig`.

```python
from dataclasses import dataclass
from ...common.config.model_config_base import BaseConfig

@dataclass
class MyModelConfig(BaseConfig):
    hidden_size: int = 4096
    num_layers: int = 32
    vocab_size: int = 32000
    model_name: str = "my_model"
```

## 3. Implementing the Model (`model.py`)

Implement your model using `src.inference_pio.core.engine.backend`.

**Key Components:**
*   `Tensor`: The main data structure.
*   `Module`: Base class for layers.
*   `Linear`, `Embedding`, `RMSNorm`: Built-in layers.
*   `scaled_dot_product_attention`: Optimized fused kernel.
*   `rope`: Rotary Positional Embedding kernel.

```python
from ...core.engine.backend import Module, Linear, Tensor, RMSNorm

class MyModel(Module):
    def __init__(self, config):
        super().__init__()
        self.embed = Linear(config.hidden_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, x: Tensor):
        # x is a backend.Tensor
        h = self.embed(x)
        return self.norm(h)
```

**Note:** Ensure you handle tensor shapes correctly. `rope` and attention kernels often expect 4D tensors `[Batch, Seq, Heads, HeadDim]`.

## 4. Implementing the Plugin Interface (`plugin.py`)

Create an adapter that connects your model to the system. Implement `ModelPluginInterface`.

```python
from ...common.interfaces.improved_base_plugin_interface import ModelPluginInterface
from .model import MyModel
from .config import MyModelConfig

class MyModelPlugin(ModelPluginInterface):
    def initialize(self, **kwargs):
        self.config = MyModelConfig(**kwargs)
        self.model = MyModel(self.config)

    def load_model(self):
        # Load weights using CustomModelLoader (safetensors)
        pass

    def infer(self, input_data):
        # Handle tokenization and generation loop here
        pass
```

## 5. Metadata (`plugin_manifest.json`)

```json
{
    "name": "my_model",
    "version": "1.0.0",
    "type": "model",
    "description": "A new dependency-free model."
}
```

## 6. Testing

Add tests in `tests/` that verify your model logic without external dependencies. Use `unittest`.

## 7. Registration

Once created, the `ModelFactory` will automatically discover your plugin if it's placed in `src/inference_pio/models/` and has a valid manifest. Alternatively, you can manually register it.

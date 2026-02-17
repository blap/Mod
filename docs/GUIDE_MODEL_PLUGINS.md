# Creating Model Plugins

Inference-PIO uses a plugin system to support different model architectures. A model plugin defines the model's structure, configuration, loading logic, and tokenization.

## Structure

A model plugin must reside in `src/inference_pio/models/<model_name>/` and contain the following files:

```
src/inference_pio/models/<model_name>/
├── __init__.py           # Exports model class, config, and plugin
├── architecture.py       # (Optional) Core PyTorch-like module definitions
├── config.py             # Configuration class inheriting from BaseConfig
├── model.py              # Main model wrapper (loading, generation)
├── plugin.py             # Plugin class inheriting from TextModelPluginInterface
└── plugin_manifest.json  # Metadata
```

## Step-by-Step Guide

### 1. `plugin_manifest.json`

Define the plugin metadata.

```json
{
    "name": "MyNewModel",
    "version": "1.0.0",
    "type": "model",
    "description": "Implementation of MyNewModel architecture",
    "entry_point": "plugin:create_my_new_model_plugin",
    "dependencies": []
}
```

### 2. `config.py`

Create a configuration class inheriting from `BaseConfig`.

```python
from ...common.config.model_config_base import BaseConfig

class MyNewModelConfig(BaseConfig):
    def __init__(self, **kwargs):
        self.hidden_size = 4096
        self.num_hidden_layers = 32
        self.num_attention_heads = 32
        self.vocab_size = 32000
        super().__init__(**kwargs)
```

### 3. `model.py` (and `architecture.py`)

Implement the model using the custom `backend.py` primitives (`Tensor`, `Linear`, `Module`). **Do not import torch.**

```python
from ...core.engine.backend import Module, Tensor, Linear, RMSNorm, scaled_dot_product_attention

class MyNewModelBlock(Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MyAttention(config)
        self.mlp = MyMLP(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(self, x):
        h = self.attn(self.norm1(x))
        x = x + h
        h = self.mlp(self.norm2(x))
        return x + h

class MyNewModel(Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for i in range(config.num_hidden_layers):
            self.layers.append(MyNewModelBlock(config))

    def forward(self, input_ids):
        # Implement embedding + layer loop
        pass
```

**Crucial:** Implement `_resolve_model_path` and `_initialize_model` to handle loading from `H:/`, local cache, or downloading.

### 4. `plugin.py`

Implement the plugin interface.

```python
from ...common.base.plugin_interface import TextModelPluginInterface
from .config import MyNewModelConfig
from .model import MyNewModel

class MyNewModelPlugin(TextModelPluginInterface):
    def initialize(self, config_path: str = None, **kwargs):
        self._config = MyNewModelConfig(**kwargs)
        self._model = MyNewModel(self._config)
        # Initialize BatchManager here

    def load_weights(self, path: str):
        # Delegate to model's loader
        pass

    def generate_text(self, prompt: str, **kwargs):
        # Tokenize, Forward, Detokenize
        pass

def create_my_new_model_plugin():
    return MyNewModelPlugin()
```

### 5. Register

The `PluginManager` will automatically discover the plugin if it's in the `src/inference_pio/models/` directory and has a valid manifest.

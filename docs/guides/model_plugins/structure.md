# Model Plugin Structure and Configuration

## Directory Structure

When creating a new model, you must follow the directory structure below:

```
src/
└── inference_pio/
    └── models/
        └── model_name/
            ├── __init__.py
            ├── config.py
            ├── model.py
            ├── plugin.py
            ├── plugin_manifest.json
            ├── architecture/
            ├── attention/
            ├── fused_layers/
            ├── kv_cache/
            ├── mlp/
            ├── rotary_embeddings/
            ├── specific_optimizations/
            ├── tests/
            └── benchmarks/
```

## Step 1: Create the Directory Structure

Create a folder for your model in `src/inference_pio/models/model_name/`.

## Step 2: Create the Plugin Manifest

Create a `plugin_manifest.json` file with your model's information:

```json
{
  "name": "ModelName",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "Description of your model",
  "plugin_type": "MODEL_COMPONENT",
  "dependencies": [
    "torch",
    "transformers"
  ],
  "compatibility": {
    "torch_version": ">=2.0.0",
    "transformers_version": ">=4.30.0",
    "python_version": ">=3.8",
    "min_memory_gb": 8.0
  },
  "model_architecture": "Architecture Type",
  "model_size": "Model Size",
  "required_memory_gb": 8.0,
  "supported_modalities": ["text"],
  "license": "MIT",
  "tags": ["description", "of", "model"],
  "model_family": "Model Family",
  "num_parameters": 0,
  "entry_point": {
    "module": ".plugin",
    "class": "ModelName_Plugin",
    "factory_function": "create_model_name_plugin"
  }
}
```

## Step 3: Create the Model Configuration

Create a `config.py` file inheriting from `BaseConfig`:

```python
from dataclasses import dataclass
from typing import Optional
from src.inference_pio.common.config.model_config_base import BaseConfig

@dataclass
class ModelNameConfig(BaseConfig):
    """
    Configuration for the ModelName model.
    """
    # Model specific parameters
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: int = 5504
    vocab_size: int = 152064
    max_position_embeddings: int = 32768

    def __post_init__(self):
        """
        Post-initialization adjustments.
        """
        super().__post_init__()
        # Add model specific adjustments here
```

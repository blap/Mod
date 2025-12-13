# Configuration Module

This module handles configuration management for the Flexible Model System.

## Components

- `Config`: Base configuration class
- `ModelConfig`: Model-specific configuration with defaults
- Configuration loading from JSON files
- Runtime configuration updates

## Usage

```python
from src.config.manager import Config, ModelConfig

# Load from dictionary
config = Config({'learning_rate': 0.001})

# Load from JSON file
model_config = ModelConfig.from_json('model_config.json')

# Access values
lr = model_config.get('learning_rate', 0.01)
```
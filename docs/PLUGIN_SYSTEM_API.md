# Plugin System and Management APIs

## Overview

The Inference-PIO plugin system provides a flexible and extensible architecture for managing different model implementations. This system allows for dynamic loading, activation, and execution of model plugins while maintaining a consistent interface.

## Plugin Manager

The `PluginManager` class is the central component for managing plugins in the system.

### Getting the Global Plugin Manager

```python
from inference_pio import get_plugin_manager

# Get the global plugin manager instance
pm = get_plugin_manager()
```

### Plugin Registration

Register a plugin with the manager:

```python
from inference_pio import register_plugin

# Register a plugin instance
success = register_plugin(plugin_instance, name="my_plugin")

# Or use the manager directly
pm = get_plugin_manager()
success = pm.register_plugin(plugin_instance, name="my_plugin")
```

### Loading Plugins

Load plugins from different sources:

```python
from inference_pio import load_plugin_from_path, load_plugins_from_directory

# Load a single plugin from a file path
success = load_plugin_from_path("/path/to/plugin.py")

# Load all plugins from a directory
count = load_plugins_from_directory("/path/to/plugins/")
```

### Activating and Deactivating Plugins

Activate a plugin before using it:

```python
from inference_pio import activate_plugin

# Activate a plugin by name
success = activate_plugin("glm_4_7_flash", device="cuda:0", max_tokens=512)

# Or use the manager directly
pm = get_plugin_manager()
success = pm.activate_plugin("glm_4_7_flash", device="cuda:0", max_tokens=512)
```

Deactivate a plugin when done:

```python
# Deactivate a plugin
success = pm.deactivate_plugin("glm_4_7_flash")
```

### Executing Plugin Functions

Execute plugin functionality:

```python
from inference_pio import execute_plugin

# Execute a plugin's main function
result = execute_plugin("glm_4_7_flash", "Your input text")

# Or use the manager directly
pm = get_plugin_manager()
result = pm.execute_plugin("glm_4_7_flash", "Your input text", max_new_tokens=100)
```

### Listing Plugins

Get information about available plugins:

```python
# List all registered plugins
all_plugins = pm.list_plugins()

# List active plugins
active_plugins = pm.list_active_plugins()

print(f"All plugins: {all_plugins}")
print(f"Active plugins: {active_plugins}")
```

## Plugin Interface

All plugins must implement the `ModelPluginInterface` or extend `TextModelPluginInterface`.

### Base Plugin Interface Methods

```python
from inference_pio.common.base_plugin_interface import ModelPluginInterface

class MyPlugin(ModelPluginInterface):
    def initialize(self, **kwargs) -> bool:
        """Initialize the plugin with configuration parameters."""
        pass

    def load_model(self, config=None) -> nn.Module:
        """Load the model with the given configuration."""
        pass

    def infer(self, data: Any) -> Any:
        """Perform inference on the given data."""
        pass

    def cleanup(self) -> bool:
        """Clean up resources used by the plugin."""
        pass
```

### Text Model Plugin Interface Methods

For text-based models, extend `TextModelPluginInterface`:

```python
from inference_pio.common.base_plugin_interface import TextModelPluginInterface

class MyTextPlugin(TextModelPluginInterface):
    def tokenize(self, text: str, **kwargs) -> Any:
        """Tokenize the given text."""
        pass

    def detokenize(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """Decode token IDs back to text."""
        pass

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        """Generate text based on the given prompt."""
        pass

    def chat_completion(self, messages: List[Dict[str, str]], max_new_tokens: int = 1024, **kwargs) -> str:
        """Perform chat completion with the model."""
        pass
```

## Plugin Creation

### Factory Functions

Each plugin should have a factory function:

```python
def create_my_plugin() -> MyPlugin:
    """Factory function to create a MyPlugin instance."""
    return MyPlugin()
```

### Plugin Metadata

Plugins use `ModelPluginMetadata` to describe themselves:

```python
from inference_pio.common.base_plugin_interface import ModelPluginMetadata, PluginType
from datetime import datetime

metadata = ModelPluginMetadata(
    name="MyModel",
    version="1.0.0",
    author="Your Name",
    description="Description of your model",
    plugin_type=PluginType.MODEL_COMPONENT,
    dependencies=["torch", "transformers"],
    compatibility={
        "torch_version": ">=2.0.0",
        "python_version": ">=3.8"
    },
    created_at=datetime.now(),
    updated_at=datetime.now(),
    model_architecture="Transformer-based model",
    model_size="7B",
    required_memory_gb=8.0,
    supported_modalities=["text"],
    license="MIT",
    tags=["language-model", "text-generation"],
    model_family="MyFamily",
    num_parameters=7000000000,
    test_coverage=0.95,
    validation_passed=True
)
```

## Advanced Plugin Features

### Memory Management

Plugins can implement memory management features:

```python
class MyPlugin(ModelPluginInterface):
    def setup_memory_management(self, **kwargs) -> bool:
        """Set up memory management including swap and paging configurations."""
        pass

    def enable_tensor_paging(self, **kwargs) -> bool:
        """Enable tensor paging for the model."""
        pass

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics for the plugin."""
        pass

    def force_memory_cleanup(self) -> bool:
        """Force cleanup of memory resources."""
        pass
```

### Model Surgery

Plugins can implement model optimization features:

```python
class MyPlugin(ModelPluginInterface):
    def setup_model_surgery(self, **kwargs) -> bool:
        """Set up model surgery system for identifying and removing non-essential components."""
        pass

    def perform_model_surgery(self, model: nn.Module = None,
                            components_to_remove: Optional[List[str]] = None,
                            preserve_components: Optional[List[str]] = None) -> nn.Module:
        """Perform model surgery by identifying and removing non-essential components."""
        pass

    def analyze_model_for_surgery(self, model: nn.Module = None) -> Dict[str, Any]:
        """Analyze a model to identify potential candidates for surgical removal."""
        pass
```

### Distributed Simulation

For distributed execution simulation:

```python
class MyTextPlugin(TextModelPluginInterface):
    def setup_distributed_simulation(self, **kwargs) -> bool:
        """Set up distributed simulation system for multi-GPU execution simulation."""
        pass

    def enable_distributed_execution(self, **kwargs) -> bool:
        """Enable distributed execution simulation."""
        pass

    def get_distributed_stats(self) -> Dict[str, Any]:
        """Get statistics about distributed execution."""
        pass
```

### Tensor Compression

For model compression:

```python
class MyPlugin(ModelPluginInterface):
    def setup_tensor_compression(self, **kwargs) -> bool:
        """Set up tensor compression system for model weights and activations."""
        pass

    def compress_model_weights(self, compression_ratio: float = 0.5, **kwargs) -> bool:
        """Compress model weights using tensor compression techniques."""
        pass

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get tensor compression statistics."""
        pass
```

## Plugin Lifecycle

The typical lifecycle of a plugin is:

1. **Creation**: Plugin instance is created via factory function
2. **Registration**: Plugin is registered with the plugin manager
3. **Initialization**: Plugin is initialized with configuration
4. **Activation**: Plugin is activated and made ready for use
5. **Execution**: Plugin performs inference or other operations
6. **Deactivation**: Plugin is deactivated when no longer needed
7. **Cleanup**: Plugin resources are cleaned up

## Error Handling

The plugin system includes comprehensive error handling:

```python
try:
    # Attempt to activate a plugin
    if not activate_plugin("my_plugin", device="cuda:0"):
        print("Failed to activate plugin")
        # Handle activation failure
except Exception as e:
    print(f"Error activating plugin: {e}")

try:
    # Execute plugin functionality
    result = execute_plugin("my_plugin", "input data")
    if result is None:
        print("Plugin execution failed")
except Exception as e:
    print(f"Error executing plugin: {e}")
```

## Performance Considerations

When working with the plugin system:

- Initialize plugins once and reuse them for multiple operations
- Activate plugins only when needed
- Clean up plugins when finished to free resources
- Monitor memory usage with the provided statistics methods
- Use appropriate configuration parameters for your hardware
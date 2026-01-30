# Contributing to Inference-PIO

We welcome contributions to the Inference-PIO project! This document outlines the guidelines for contributing to this repository.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Development Setup](#development-setup)
3. [Adding New Models](#adding-new-models)
4. [Code Style](#code-style)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)
7. [Architecture Guidelines](#architecture-guidelines)

## Project Structure

The project follows a self-contained plugin architecture with each model having its own directory containing all necessary components:

```
├── .flake8                 # Flake8 configuration
├── .gitignore              # Git ignore rules
├── .pylintrc               # Pylint configuration
├── CONTRIBUTING.md         # Contribution guidelines
├── pyproject.toml          # Project metadata and build system configuration
├── README.md               # Project overview
├── requirements.txt        # Core project dependencies
├── requirements_api.txt    # API-specific dependencies
├── requirements_benchmark.txt # Benchmark-specific dependencies
├── setup.py                # Setup script
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

Within each model directory:
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

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/inference-pio.git
   cd inference-pio
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Adding New Models

To add a new model to the system:

1. Create a new directory under `src/inference_pio/models/` with the model name
2. Implement the required components following the same structure:
   - `__init__.py` - Package initialization
   - `config.py` - Model configuration
   - `model.py` - Model implementation
   - `plugin.py` - Plugin interface implementation
   - `safe_model.py` - Safe model implementation
   - `architecture_registration.py` - Architecture registration
   - `config_integration.py` - Configuration integration
   - `attention/` - Attention mechanisms
   - `cuda_kernels/` - Hardware-specific optimizations
   - `tensor_parallel/` - Parallelism implementations
   - `kv_cache/` - KV-cache optimizations
   - `rotary_embeddings/` - Rotary embedding implementations
   - `fused_layers/` - Fused operations
   - `linear_optimizations/` - Linear layer optimizations
   - `optimizations/` - General optimizations
   - `plugin_modules/` - Plugin-specific modules
   - `prefix_caching/` - Prefix caching implementations
   - `specific_optimizations/` - Model-specific optimizations
   - `tests/` - Comprehensive test suite
   - `benchmarks/` - Performance benchmarks
3. Implement the plugin interface following the standard pattern
4. Add comprehensive tests in the `tests/` subdirectory
5. Add performance benchmarks in the `benchmarks/` subdirectory
6. Update the main `__init__.py` to expose the new plugin

### Model Plugin Interface

All models must implement the standard plugin interface:

```python
from inference_pio.common.base_plugin_interface import ModelPluginInterface

class NewModelPlugin(ModelPluginInterface):
    def __init__(self):
        # Initialize with metadata
        super().__init__(metadata)
        # Initialize model-specific attributes

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

def create_new_model_plugin() -> NewModelPlugin:
    return NewModelPlugin()
```

## Code Style

We follow PEP 8 guidelines for Python code. Additionally:

- Use type hints for all function parameters and return values
- Write docstrings for all classes and functions
- Keep functions focused and small (preferably under 50 lines)
- Use meaningful variable and function names
- Follow the DRY (Don't Repeat Yourself) principle
- Use constants for magic numbers and strings
- Follow the architecture guidelines below

## Testing

The project uses a direct testing approach without external testing frameworks like pytest or unittest. This approach provides:

- Faster execution without framework overhead
- Simpler dependency management
- More control over test execution
- Direct integration with the codebase

### Writing Tests

Tests use the custom assertion functions in `tests.utils.test_utils.py`:

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

## Submitting Changes

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes following the code style guidelines
4. Add or update tests as needed
5. Run the existing tests to ensure no regressions
6. Update documentation as needed
7. Submit a pull request with a clear description of your changes

## Architecture Guidelines

### Self-Contained Plugins

Each model plugin should be completely self-contained:

- All model-specific code should be in the model's directory
- Dependencies should be minimal and well-defined
- The plugin should work independently of other models
- Configuration should be model-specific

### Common Components

Only put code in the `common/` directory if it's truly reusable across multiple models:

- Base classes and interfaces
- General-purpose utilities
- Shared optimization techniques
- Common data structures

### Plugin Interface Compliance

All plugins must implement the required methods from the plugin interface:

- `initialize()` - Initialize the plugin with configuration
- `load_model()` - Load the model with the given configuration
- `infer()` - Perform inference on the given data
- `cleanup()` - Clean up resources used by the plugin
- `execute()` - Execute the model with given inputs
- `get_model_info()` - Get information about the loaded model
- `update_config()` - Update the plugin configuration
- `supports_config()` - Check if this plugin supports the given configuration

### Performance Considerations

When implementing new features:

- Consider memory efficiency
- Implement proper resource cleanup
- Use appropriate data types
- Follow best practices for PyTorch/TensorFlow
- Profile performance-critical code

### Documentation

All contributions should include:

- Docstrings for all public functions and classes
- Inline comments for complex logic
- Updates to relevant documentation files
- Examples of usage when appropriate
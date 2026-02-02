# System Architecture & Plugin API

## 1. Architecture Overview

Inference-PIO is designed as a modular, self-contained plugin system. Each model acts as an independent unit with its own dependencies, optimizations, and logic, tied together by a standardized interface. Every model plugin contains its own configuration, tests, and benchmarks within its directory.

### High-Level Structure
```
src/
├── common/               # Shared utilities (BasePlugin, ConfigManager, interfaces)
├── models/               # Self-contained model plugins (each with own config/tests/benchmarks)
│   ├── glm_4_7_flash/    # GLM-4.7 Flash model with all components
│   │   ├── __init__.py   # Module entry point
│   │   ├── config.py     # Model-specific config
│   │   ├── model.py      # Core model implementation
│   │   ├── plugin.py     # Plugin interface implementation
│   │   ├── plugin_manifest.json # Plugin metadata for discovery
│   │   ├── architecture/ # Architecture-specific implementations
│   │   ├── attention/    # Attention mechanisms
│   │   ├── fused_layers/ # Fused layer implementations
│   │   ├── kv_cache/     # KV cache management
│   │   ├── mlp/          # MLP implementations
│   │   ├── rotary_embeddings/ # Rotary embedding implementations
│   │   ├── specific_optimizations/ # Model-specific optimizations
│   │   ├── configs/      # Configuration files
│   │   ├── tests/        # Model-specific tests
│   │   ├── benchmarks/   # Model-specific benchmarks
│   │   └── README.md     # Model-specific documentation
│   ├── qwen3_0_6b/       # Qwen3-0.6B model with all components
│   ├── qwen3_4b_instruct_2507/ # Qwen3-4B-Instruct-2507 model with all components
│   ├── qwen3_coder_30b/  # Qwen3-Coder-30B model with all components
│   ├── qwen3_vl_2b/      # Qwen3-VL-2B model with all components
│   └── ...
├── plugins/              # Hardware and system plugins
│   ├── base/             # Base plugin interfaces
│   ├── cpu/              # CPU-specific plugins
│   ├── intel/            # Intel-specific plugins
│   └── manager.py        # Plugin manager implementation
├── inference/            # Inference engine components
├── utils/                # Utility functions
└── configs/              # Global configuration
```

## 2. Plugin System API

The `PluginManager` is the central orchestrator.

### Usage
```python
from src.plugins.manager import get_plugin_manager

pm = get_plugin_manager()

# Discover and load all plugins from models directory
pm.discover_and_load_plugins()

# Or load plugins from a specific directory
pm.load_plugins_from_directory("./src/models")

# Activate
pm.activate_plugin("qwen3_vl_2b", device="cuda:0")

# Execute
result = pm.execute_plugin("qwen3_vl_2b", "Describe this image", image="img.jpg")

# Deactivate when done
pm.deactivate_plugin("qwen3_vl_2b")
```

### Key Components
*   **Factory Pattern:** Each plugin exposes a factory function (e.g., `create_qwen3_vl_2b_plugin`).
*   **Singleton Manager:** Ensures a single point of control for resources.
*   **Automatic Discovery:** Plugins are discovered automatically via `plugin_manifest.json` files.
*   **Metadata:** Plugins self-describe via `ModelPluginMetadata` (version, memory reqs, dependencies).
*   **Self-Contained:** Each model plugin is completely independent with its own configuration, tests, and benchmarks.

## 3. The Standard Plugin Interface

All models implement `ModelPluginInterface`.

### Core Methods
| Method | Description |
|--------|-------------|
| `initialize(**kwargs)` | Setup resources. |
| `load_model(config)` | Load weights and apply optimizations. |
| `infer(data)` | Run inference. |
| `cleanup()` | Release memory and handles. |
| `supports_config(conf)`| Validate compatibility. |
| `install()` | Install dependencies and prepare for execution. |

### Text Extensions
Models extending `TextModelPluginInterface` also support:
*   `tokenize(text)`
*   `generate_text(prompt)`
*   `chat_completion(messages)`

### Advanced Features
All model plugins support advanced features for memory management and optimization:
*   `optimize_model()` - Runtime optimization using torch.compile
*   `enable_disk_offloading()` - Move model parts between RAM and disk
*   `enable_activation_offloading()` - Manage intermediate activations
*   `enable_sharding()` - Extreme sharding for large models
*   `setup_memory_management()` - Configure memory usage

## 4. Documentation Standards

All components of the system follow strict documentation standards to ensure maintainability and usability:

### Docstring Standards
All classes, methods, and functions must include Google-style docstrings as specified in [DOCSTRINGS.md](../standards/DOCSTRINGS.md). This includes:
- Brief summary of functionality
- Complete parameter documentation with type hints
- Return value documentation
- Exception documentation
- Usage examples when beneficial

### Comment Standards
Inline and block comments follow the standards outlined in [COMMENTS.md](../standards/COMMENTS.md), including:
- Explanations of complex algorithms
- TODO markers for future improvements
- Model-specific functionality notes
- Performance-related annotations
- Security-related annotations

## 5. Configuration & Security

### Dynamic Config
The system uses `ConfigManager` to adapt parameters at runtime (e.g., switching to 4-bit quantization on low VRAM).

### Security
Plugins operate under a Trust Level (Low to Maximum).
*   **Resource Limits:** CPU, RAM, Disk usage caps.
*   **Path Validation:** Strict checking of file access for `load_model`.
*   **Isolation:** Plugins run in isolated environments with limited access.
*   **Validation:** Each plugin validates its own configuration and dependencies.

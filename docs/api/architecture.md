# System Architecture & Plugin API

## 1. Architecture Overview

Inference-PIO is designed as a modular, self-contained plugin system. Each model acts as an independent unit with its own dependencies, optimizations, and logic, tied together by a standardized interface.

### High-Level Structure
```
src/
└── inference_pio/
    ├── common/             # Shared utilities (BasePlugin, ConfigManager)
    ├── models/             # Self-contained model plugins
    │   ├── glm_4_7_flash/
    │   ├── qwen3_vl_2b/
    │   └── ...
    └── plugin_system/      # Core logic (Loader, Registry)
```

## 2. Plugin System API

The `PluginManager` is the central orchestrator.

### Usage
```python
from inference_pio import get_plugin_manager

pm = get_plugin_manager()

# Load
pm.load_plugins_from_directory("./src/inference_pio/models")

# Activate
pm.activate_plugin("qwen3_vl_2b", device="cuda:0")

# Execute
result = pm.execute_plugin("qwen3_vl_2b", "Describe this image", image="img.jpg")
```

### Key Components
*   **Factory Pattern:** Each plugin exposes a factory function (e.g., `create_qwen3_vl_2b_plugin`).
*   **Singleton Manager:** Ensures a single point of control for resources.
*   **Metadata:** Plugins self-describe via `ModelPluginMetadata` (version, memory reqs).

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

### Text Extensions
Models extending `TextModelPluginInterface` also support:
*   `tokenize(text)`
*   `generate_text(prompt)`
*   `chat_completion(messages)`

## 4. Configuration & Security

### Dynamic Config
The system uses `ConfigManager` to adapt parameters at runtime (e.g., switching to 4-bit quantization on low VRAM).

### Security
Plugins operate under a Trust Level (Low to Maximum).
*   **Resource Limits:** CPU, RAM, Disk usage caps.
*   **Path Validation:** Strict checking of file access for `load_model`.

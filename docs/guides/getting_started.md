# Getting Started with Inference-PIO

## 1. Introduction
Inference-PIO is a modular inference system designed for high-performance deployment of models like Qwen3-VL and GLM-4.7. It uses a plugin-based architecture where each model is completely independent with its own configuration, tests, and benchmarks.

## 2. Installation

### Prerequisites
*   Python 3.8+
*   PyTorch 2.0+
*   CUDA-compatible GPU (Recommended)

### Steps
```bash
git clone https://github.com/inference-pio/inference-pio.git
cd inference-pio
pip install -r requirements.txt
pip install -e .
```

## 3. Basic Usage

### Loading a Model
Models are loaded via factory functions.

```python
from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin

# 1. Create Plugin
plugin = create_glm_4_7_flash_plugin()

# 2. Initialize & Load
plugin.initialize(device="cuda:0")
plugin.load_model()

# 3. Infer
result = plugin.infer("Explain quantum computing.")
print(result)

# 4. Cleanup
plugin.cleanup()
```

### Using the Plugin Manager
For managing multiple models dynamically:

```python
from src.inference_pio.plugins.manager import get_plugin_manager

pm = get_plugin_manager()
pm.activate_plugin("qwen3_vl_2b")
result = pm.execute_plugin("qwen3_vl_2b", {"text": "Describe image", "image": "img.jpg"})
```

### Using the Model Factory
Alternatively, you can use the model factory for simplified model loading:

```python
from src.inference_pio.core.model_factory import create_model

# Load any model by name
model = create_model("glm_4_7_flash")
model.initialize()
result = model.infer("Explain quantum computing.")
print(result)
```

## 4. Configuration
You can customize model behavior during initialization:

```python
plugin.initialize(
    max_new_tokens=1024,
    use_flash_attention_2=True,
    load_in_4bit=True  # For low VRAM
)
```

## 5. Adding New Models
Each model in the `src/inference_pio/models/` directory is completely self-contained with its own:
- Configuration files
- Model implementation
- Plugin interface
- Tests
- Benchmarks
- Optimization implementations

To add a new model, simply create a new directory following the standardized structure described in `docs/guides/model_plugins/structure.md`.

## 6. Troubleshooting
*   **OOM Errors:** Enable 4-bit loading, reduce batch size, or enable disk offloading.
*   **Missing Weights:** Ensure the model path in `config.py` is correct or use the automatic download features.
*   **Plugin Discovery:** Make sure your model has a proper `plugin_manifest.json` file for automatic discovery.

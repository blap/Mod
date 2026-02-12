# Developer Guide

## 1. Plugin Development

### The Interface
All plugins must implement `TextModelPluginInterface` (see `src/inference_pio/common/interfaces/improved_base_plugin_interface.py`).

### Self-Contained Architecture
Each model plugin is completely independent with its own:
- Configuration in `config.py`
- Model implementation in `model.py`
- Plugin interface in `plugin.py`
- Tests in `tests/`
- Benchmarks in `benchmarks/`

### Steps to Add a New Model
1.  **Create Directory:** `src/inference_pio/models/<new_model>/`
2.  **Implement Config:** Create `config.py` defining hyperparameters.
3.  **Implement Model:** Create `model.py` using `src.inference_pio.core.engine.backend`.
4.  **Implement Plugin:** Create `plugin.py` implementing `initialize`, `load_model`, `infer`, `tokenize`, `detokenize`, `infer_batch`.
5.  **Tests:** Add unit tests in `tests/`.

## 2. Documentation Standards

### Docstrings
All classes, methods, and functions must include Google-style docstrings.
- Brief summary of functionality
- Complete parameter documentation with type hints
- Return value documentation

### Comments
- Explain complex algorithms (especially custom C kernel usage).
- Use TODO markers for future improvements.

## 3. Best Practices

### Code Style
*   **No PyTorch/NumPy:** Use `src.inference_pio.core.engine.backend.Tensor`.
*   **Type Hints:** Mandatory.
*   **Imports:** Absolute from `src`.

### Performance Optimization
*   **Custom Backend:** Use `libtensor_ops` via `backend.Tensor` methods.
*   **Fused Kernels:** Use `swiglu`, `rope`, `scaled_dot_product_attention`, `fused_add_rms_norm` which map to optimized C/CUDA kernels.
*   **Memory:** Use `safetensors` with `mmap` for fast loading. Use `Static KV Cache` (pre-allocated) to avoid fragmentation.
*   **Batching:** Integrate with `BatchManager` for efficient request handling.

### Testing
*   Write unit tests in `src/inference_pio/models/<model_name>/tests/`.
*   Avoid large model downloads in CI tests; use mocks or dummy tensors.
*   Run `verify_textual_standardization.py` to ensure interface compliance.

## 4. Design Patterns
*   **Backend Abstraction:** `Tensor` class hides C/CUDA details.
*   **Plugin Interface:** Standardized `TextModelPluginInterface`.
*   **Factory:** `create_<model>_plugin` functions.

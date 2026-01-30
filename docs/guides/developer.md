# Developer Guide

## 1. Plugin Development

### The Interface
All plugins must implement `ModelPluginInterface` (see `src/inference_pio/common/standard_plugin_interface.py`).

### Steps to Add a New Model
1.  **Create Directory:** `src/inference_pio/models/<new_model>/`
2.  **Implement Config:** Create `config.py` inheriting `BaseModelConfig`.
3.  **Implement Plugin:** Create `plugin.py` implementing `initialize`, `load_model`, `infer`, `cleanup`.
4.  **Register:** Add factory function in `src/inference_pio/__init__.py`.

## 2. Best Practices

### Code Style
*   Use **Google-style docstrings**.
*   Type hints are **mandatory**.
*   No strict line length, but aim for readability (~100 chars).
*   Imports: Absolute from `src.inference_pio`.

### Performance Optimization
*   **FlashAttention:** Use `flash_attn` library where possible.
*   **Fused Kernels:** Implement custom CUDA kernels in `cuda_kernels/` for critical paths.
*   **Memory:** Use `HardwareAnalyzer` to check available VRAM and adapt config (e.g., enable paging) automatically.

### Testing
*   Write unit tests in `tests/`.
*   Use `tests.utils` assertions.
*   Avoid large model downloads in CI tests; use mocks or lightweight checks.

## 3. Design Patterns
*   **Factory:** Used for plugin creation.
*   **Singleton:** `PluginManager` and `HardwareAnalyzer`.
*   **Strategy:** Optimization strategies (e.g., `AttentionStrategy`).

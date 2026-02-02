# Developer Guide

## 1. Plugin Development

### The Interface
All plugins must implement `ModelPluginInterface` (see `src/common/improved_base_plugin_interface.py`).

### Self-Contained Architecture
Each model plugin is completely independent with its own:
- Configuration files in `configs/`
- Model implementation in `model.py`
- Plugin interface in `plugin.py`
- Tests in `tests/` (organized by model)
- Benchmarks in `benchmarks/` (organized by model)
- Optimization implementations in dedicated subdirectories

### Steps to Add a New Model
1.  **Create Directory:** `src/models/<new_model>/`
2.  **Implement Config:** Create `config.py` inheriting `BaseConfig`.
3.  **Implement Plugin:** Create `plugin.py` implementing `initialize`, `load_model`, `infer`, `cleanup`.
4.  **Add Manifest:** Create `plugin_manifest.json` for automatic discovery.
5.  **Organize:** Place all model-specific components within the model directory.
6.  **Document:** Ensure all code follows documentation standards (see below).

## 2. Documentation Standards

### Docstrings
All classes, methods, and functions must include Google-style docstrings as specified in [DOCSTRINGS.md](../../docs/standards/DOCSTRINGS.md). This includes:
- Brief summary of functionality
- Complete parameter documentation with type hints
- Return value documentation
- Exception documentation
- Usage examples when beneficial

### Comments
Follow comment standards as specified in [COMMENTS.md](../../docs/standards/COMMENTS.md), including:
- Explanations of complex algorithms
- TODO markers for future improvements
- Model-specific functionality notes
- Performance-related annotations
- Security-related annotations

### Self-Contained Architecture Documentation
Since each model plugin is completely independent with its own configuration, tests, and benchmarks, documentation should:
- Clearly indicate model-specific functionality
- Reference model-specific configurations and optimizations
- Include model-specific usage examples
- Document model-specific parameters and options

## 3. Best Practices

### Code Style
*   Use **Google-style docstrings** as per [DOCSTRINGS.md](../../docs/standards/DOCSTRINGS.md).
*   Follow **comment standards** as per [COMMENTS.md](../../docs/standards/COMMENTS.md).
*   Type hints are **mandatory**.
*   No strict line length, but aim for readability (~100 chars).
*   Imports: Absolute from `src`.

### Performance Optimization
*   **FlashAttention:** Use `flash_attn` library where possible.
*   **Fused Kernels:** Implement custom CUDA kernels in `cuda_kernels/` for critical paths.
*   **Memory:** Use `HardwareAnalyzer` to check available VRAM and adapt config (e.g., enable paging) automatically.

### Testing
*   Write unit tests in `tests/models/<model_name>/unit/`.
*   Write integration tests in `tests/models/<model_name>/integration/`.
*   Write performance tests in `tests/models/<model_name>/performance/`.
*   Use `tests.utils` assertions.
*   Avoid large model downloads in CI tests; use mocks or lightweight checks.

### Self-Contained Principle
*   Each model plugin must be completely independent.
*   All dependencies and configurations should be within the model's directory.
*   Tests and benchmarks should be organized by model.
*   Use the plugin manifest system for automatic discovery.

## 4. Design Patterns
*   **Factory:** Used for plugin creation.
*   **Singleton:** `PluginManager` and `HardwareAnalyzer`.
*   **Strategy:** Optimization strategies (e.g., `AttentionStrategy`).
*   **Self-Contained Architecture:** Each model plugin is independent with its own resources.

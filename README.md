# Inference-PIO

Inference-PIO is a modular, high-performance inference system built on a self-contained plugin architecture. It supports advanced models like GLM-4.7, Qwen3-VL, and Qwen3-Coder.

## 📚 Documentation

*   **[Getting Started](docs/guides/getting_started.md):** Installation, basic usage, and configuration.
*   **[Supported Models](docs/api/models.md):** List of models and their capabilities.
*   **[System Architecture](docs/api/architecture.md):** Deep dive into the plugin system and design.
*   **[Advanced Features](docs/api/advanced_features.md):** Multimodal attention, streaming, and NAS.
*   **[Benchmarking](docs/guides/benchmarking.md):** Performance measurement guide.

## 🛠 Project Structure

```
.
├── benchmarks/             # Performance scripts
├── benchmark_results/      # Output data
├── docs/                   # Documentation (Guides, API, Standards)
├── scripts/                # Utility scripts (Test runner, CI)
├── src/
│   └── inference_pio/
│       ├── common/         # Shared utilities
│       ├── models/         # Self-contained model plugins
│       └── plugin_system/  # Core infrastructure
└── tests/                  # Global tests
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python -c "from inference_pio import create_glm_4_7_flash_plugin; p=create_glm_4_7_flash_plugin(); p.initialize(); print(p.infer('Hello'))"
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for developer guidelines.

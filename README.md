# Inference-PIO

Inference-PIO is a modular, high-performance inference system built on a self-contained plugin architecture. Each model is completely independent with its own configuration, tests, and benchmarks. The system supports advanced models like GLM-4.7, Qwen3-VL, and Qwen3-Coder.

## 📚 Documentation

*   **[Getting Started](docs/guides/getting_started.md):** Installation, basic usage, and configuration.
*   **[Creating a Model Plugin](docs/guides/model_plugins/overview.md):** Guide to creating new model plugins.
*   **[Supported Models](docs/api/models.md):** List of models and their capabilities.
*   **[System Architecture](docs/api/architecture.md):** Deep dive into the plugin system and design.
*   **[Advanced Features](docs/api/advanced_features.md):** Multimodal attention, streaming, and NAS.
*   **[Benchmarking](docs/guides/benchmarking.md):** Performance measurement guide.
*   **[Coding Standards](docs/standards/CODING.md):** Code style and naming conventions.
*   **[Docstring Standards](docs/standards/DOCSTRINGS.md):** Documentation format guidelines.
*   **[Comment Standards](docs/standards/COMMENTS.md):** Inline and block comment guidelines.
*   **[Testing Standards](docs/standards/TESTING.md):** Test organization and naming conventions.
*   **[Benchmarking Standards](docs/standards/BENCHMARKS.md):** Performance measurement guidelines.

## 🛠 Project Structure

```
.
├── benchmark_results/              # General benchmark results
│   └── general/                    # Cross-model benchmark data
├── docs/                           # Documentation (Guides, API, Standards)
├── examples/                       # Example usage scripts
├── src/
│   └── inference_pio/
│       ├── benchmarks/             # General benchmarks
│       ├── common/                 # Shared utilities and interfaces
│       ├── configs/                # Global configuration
│       ├── core/                   # Core system components (tools, scripts, factory)
│       │   ├── tools/
│       │   │   ├── scripts/        # Utility scripts (testing, benchmarking)
│       │   │   └── ...
│       │   └── model_factory.py    # Model creation factory
│       ├── models/                 # Individual self-contained model plugins
│       │   ├── glm_4_7_flash/      # GLM-4.7 Flash model
│       │   ├── qwen3_0_6b/         # Qwen3-0.6B model
│       │   ├── qwen3_4b_instruct_2507/ # Qwen3-4B-Instruct-2507 model
│       │   ├── qwen3_coder_30b/    # Qwen3-Coder-30B model
│       │   └── qwen3_vl_2b/        # Qwen3-VL-2B model
│       ├── plugins/                # Plugin system infrastructure
│       │   ├── base/               # Base plugin interfaces
│       │   ├── cpu/                # CPU-specific plugins
│       │   ├── intel/              # Intel-specific plugins
│       │   └── manager.py          # Plugin manager implementation
│       ├── tests/                  # Global test structure
│       │   ├── base/               # Test base classes
│       │   ├── functional/         # Functional tests
│       │   ├── integration/        # Integration tests
│       │   ├── performance/        # Performance tests
│       │   └── unit/               # Unit tests
│       └── utils/                  # Utility functions
└── ...
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python -c "from src.inference_pio.core.model_factory import create_model; m=create_model('glm_4_7_flash'); m.initialize(); print(m.infer('Hello'))"
```

## 🧩 Plugin Discovery System

The system automatically discovers new plugins through:
1. **Directory scanning**: Looks for model directories in `src/inference_pio/models/`
2. **Manifest files**: Each model has a `plugin_manifest.json` file
3. **Auto-registration**: Plugins are automatically registered without manual imports

## 🏗️ Self-Contained Architecture

Each model plugin is completely independent with its own:
- Configuration files in `configs/`
- Model implementation in `model.py`
- Plugin interface in `plugin.py`
- Tests in `tests/` (organized by model)
- Benchmarks in `benchmarks/` (organized by model)
- Optimization implementations in dedicated subdirectories

This ensures that each model can be developed, tested, and deployed independently.

## 🧪 Testing

The project uses an organized test structure. Global tests are in `src/inference_pio/tests/`, and model-specific tests are in `src/inference_pio/models/<model_name>/tests/`.

To run all tests:
```bash
python src/inference_pio/core/tools/scripts/testing/run_tests.py
```

To run tests for a specific category (e.g., unit):
```bash
python src/inference_pio/core/tools/scripts/testing/run_tests.py --category unit
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for developer guidelines.

## 🧪 Tests with Real Functionalities

The project includes a comprehensive test suite that uses real functionalities instead of excessive simulations. These tests exercise critical system paths with real data and operations, maintaining efficiency while increasing fidelity to reality.

### Types of Real Tests

- **Functionality Tests**: Verify basic system functionality using real components
- **Integration Tests**: Test interaction between multiple system components
- **Performance Tests**: Measure real performance metrics instead of simulations
- **Functional Tests**: Verify system behavior from a user perspective
- **Regression Tests**: Ensure changes do not break existing functionalities

### Running Real Tests

To run all tests with real functionalities:

```bash
python src/inference_pio/core/tools/scripts/testing/run_tests.py
```

## 🔌 Extensible Architecture

The project implements a flexible and extensible architecture for easy inclusion of new models and test/benchmark types. Each model/plugin is completely independent with its own configuration, tests, and benchmarks.

### Adding New Models

To add a new model, create a new directory in `src/inference_pio/models/` following the standard structure. Refer to [Creating a Model Plugin](docs/guides/model_plugins/overview.md) for details.

### Adding New Test/Benchmark Types

New test and benchmark types can be added by extending the base classes in `src/inference_pio/tests/base/` or creating new scripts in `src/inference_pio/core/tools/scripts/`.

For more details on the extensible architecture, consult the documentation in `docs/guides/`.

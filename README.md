# Inference-PIO

Inference-PIO is a modular, plugin-based architecture for advanced model inference. It supports multiple models like GLM-4.7, Qwen3-4B, Qwen3-Coder, and Qwen3-VL through a standardized interface while allowing for model-specific optimizations.

## Project Structure

The project is organized to ensure strict separation between core infrastructure and model-specific plugins.

```
.
├── benchmarks_results/         # Output directory for benchmark results
├── docs/                       # Project documentation
├── scripts/                    # Utility scripts
├── src/
│   └── inference_pio/
│       ├── benchmarks/         # Global benchmark discovery and runners
│       ├── common/             # Shared utilities and interfaces (BasePlugin, etc.)
│       ├── models/             # Self-contained model plugins
│       │   ├── glm_4_7_flash/
│       │   ├── qwen3_4b_instruct_2507/
│       │   ├── qwen3_coder_30b/
│       │   └── qwen3_vl_2b/
│       └── plugin_system/      # Core plugin loading and management logic
└── tests/                      # Global integration and unit tests
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[Testing Framework](docs/TESTING_FRAMEWORK.md)**: Guide to the testing infrastructure.
- **[Benchmark Discovery](docs/BENCHMARK_DISCOVERY_IMPLEMENTATION.md)**: How the automated benchmark system works.
- **[Test Structure](docs/TEST_STRUCTURE.md)**: Explanation of the test directory organization.

## Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Tests:**
    The project uses a custom optimized test runner.
    ```bash
    python optimized_test_runner.py
    ```
    To verify configuration and list tests:
    ```bash
    python optimized_test_runner.py --list
    ```

3.  **Run Benchmarks:**
    ```bash
    python -m inference_pio.benchmarks.discovery
    ```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

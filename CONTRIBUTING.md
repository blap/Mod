# Contributing to Inference-PIO

We welcome contributions! This document is a high-level entry point. Please refer to the specific guides for detailed instructions.

## 🏗️ Project Structure

Each model/plugin in Inference-PIO follows a self-contained structure within the src directory:

```
src/inference_pio/models/{model_name}/
├── configs/          # Configuration files specific to the model
├── tests/            # Model-specific tests
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── performance/  # Performance tests
└── benchmarks/       # Model-specific benchmarks
    ├── unit/         # Unit benchmarks
    ├── integration/  # Integration benchmarks
    ├── performance/  # Performance benchmarks
    └── results/      # Benchmark results
```

This ensures that each model/plugin is completely independent with its own configurations, tests, and benchmarks.

## 📖 Developer Guides

*   **[Developer Guide](docs/guides/developer.md):** Coding standards, plugin architecture, and how to add new models.
*   **[Testing Guide](docs/guides/testing.md):** How to run tests, write new tests, and usage of the `tests.utils` framework.
*   **[Benchmarking Guide](docs/guides/benchmarking.md):** How to measure and report performance.
*   **[Coding Standards](docs/standards/CODING.md):** Code style and naming conventions.
*   **[Docstring Standards](docs/standards/DOCSTRINGS.md):** Documentation format guidelines.
*   **[Comment Standards](docs/standards/COMMENTS.md):** Inline and block comment guidelines.
*   **[Testing Standards](docs/standards/TESTING.md):** Test organization and naming conventions.
*   **[Benchmarking Standards](docs/standards/BENCHMARKS.md):** Performance measurement guidelines.

## ⚡ Workflow

1.  **Fork & Clone:** Clone the repository and set up your environment (`pip install -e ".[dev]"`).
2.  **Branch:** Create a feature branch (`feat/my-new-model`).
3.  **Implement:** Follow the [Developer Guide](docs/guides/developer.md).
4.  **Document:** Ensure all code follows the documentation standards ([DOCSTRINGS.md](docs/standards/DOCSTRINGS.md), [COMMENTS.md](docs/standards/COMMENTS.md)).
5.  **Test:** Ensure all tests pass (`python scripts/run_tests.py`).
6.  **Benchmark:** Run benchmarks for your model (`python scripts/benchmarking/validate_benchmarks.py`).
7.  **Submit:** Open a Pull Request.

## 📝 Standards Checklist

*   [ ] **Type Hints:** All new code must be typed.
*   [ ] **Docstrings:** Use Google-style docstrings as per [DOCSTRINGS.md](docs/standards/DOCSTRINGS.md).
*   [ ] **Comments:** Follow comment standards as per [COMMENTS.md](docs/standards/COMMENTS.md).
*   [ ] **Self-Contained:** Each model plugin is completely independent with its own configuration, tests, and benchmarks.
*   [ ] **Tests:** Add unit tests for new logic.
*   [ ] **Cleanliness:** Remove debug prints and unused imports.

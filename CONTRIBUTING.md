# Contributing to Inference-PIO

We welcome contributions! This document is a high-level entry point. Please refer to the specific guides for detailed instructions.

## 📖 Developer Guides

*   **[Developer Guide](docs/guides/developer.md):** Coding standards, plugin architecture, and how to add new models.
*   **[Testing Guide](docs/guides/testing.md):** How to run tests, write new tests, and usage of the `tests.utils` framework.
*   **[Benchmarking Guide](docs/guides/benchmarking.md):** How to measure and report performance.

## ⚡ Workflow

1.  **Fork & Clone:** Clone the repository and set up your environment (`pip install -e ".[dev]"`).
2.  **Branch:** Create a feature branch (`feat/my-new-model`).
3.  **Implement:** Follow the [Developer Guide](docs/guides/developer.md).
4.  **Test:** Ensure all tests pass (`python scripts/run_tests.py`).
5.  **Submit:** Open a Pull Request.

## 📝 Standards Checklist

*   [ ] **Type Hints:** All new code must be typed.
*   [ ] **Docstrings:** Use Google-style docstrings.
*   [ ] **Tests:** Add unit tests for new logic.
*   [ ] **Cleanliness:** Remove debug prints and unused imports.

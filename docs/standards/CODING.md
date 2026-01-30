# Coding Standards & Naming Conventions

This document outlines the coding standards and naming conventions for the Inference-PIO project.

## Directory Structure

The project follows a modular structure:

```
src/
└── inference_pio/
    ├── models/
    │   └── {model_name}/
    │       ├── model.py
    │       ├── config.py
    │       ├── processor.py
    │       └── attention/
    ├── common/
    │   └── hardware_analyzer.py
    └── plugin_system/
tests/
├── unit/
├── integration/
└── performance/
benchmarks/
scripts/
docs/
```

## File Naming

- **Source Files**: `snake_case.py` (e.g., `model_factory.py`, `hardware_analyzer.py`).
- **Test Files**: `test_{component}.py` (e.g., `test_hardware_analyzer.py`).
- **Benchmark Files**: `benchmark_{metric}.py` (e.g., `benchmark_throughput.py`).

## Code Style

- **Python Version**: 3.10+
- **Line Length**: 120 characters preferred.
- **Imports**: Absolute imports for `src`. Example: `from src.inference_pio.common.config_manager import ConfigManager`.
- **Type Hints**: Use type hints for function arguments and return values.
- **Docstrings**: Google style docstrings for classes and methods.

## Refactoring Guidelines

- **Real Code**: Prefer instantiating real classes over mocks when feasible.
- **No Temporary Files**: Clean up temporary files immediately after use or use `tempfile` module.
- **Modular Design**: Keep components loosely coupled.
- **Legacy Code**: Ruthlessly delete obsolete or legacy code.

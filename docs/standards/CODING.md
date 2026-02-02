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
- **Docstrings**: Google style docstrings for classes and methods as specified in DOCSTRINGS.md.
- **Comments**: Follow comment standards as specified in COMMENTS.md.

## Refactoring Guidelines

- **Real Code**: Prefer instantiating real classes over mocks when feasible.
- **No Temporary Files**: Clean up temporary files immediately after use or use `tempfile` module.
- **Modular Design**: Keep components loosely coupled.
- **Legacy Code**: Ruthlessly delete obsolete or legacy code.

## Self-Contained Architecture

Each model plugin is completely independent with its own configuration, tests, and benchmarks. This means:

- All model-specific code must reside within the model's directory
- Model plugins should not depend on other model implementations
- Configuration, tests, and benchmarks must be model-specific
- Documentation should clearly indicate which components belong to which model

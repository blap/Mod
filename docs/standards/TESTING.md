# Testing Standards & Framework

This document outlines the standardized testing framework, directory structure, and best practices for the Inference-PIO project.

## Directory Structure

Tests are centralized in the `tests/` directory at the project root.

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── models/           # Unit tests for specific models
│   ├── common/           # Unit tests for common components
│   └── utils/            # Unit tests for utility functions
├── integration/          # Integration tests for component interactions
│   ├── models/           # Integration tests for specific models
│   ├── common/           # Integration tests for common components
│   └── end_to_end/       # End-to-end integration tests
├── performance/          # Performance and benchmarking tests
    ├── models/           # Performance tests for specific models
    └── common/           # Performance tests for common components
└── utils/                # Shared test utilities (fixtures, helpers)
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Focus**: Individual functions, classes, or methods in isolation.
- **Characteristics**: Fast (<100ms), minimal dependencies, mock external interactions.
- **Goal**: Verify logic correctness.

### Integration Tests (`tests/integration/`)
- **Focus**: Interaction between multiple components.
- **Characteristics**: May involve real file I/O, model loading (preferably small/fallback models), and system flows.
- **Goal**: Validate that components work together.

### Performance Tests (`tests/performance/`)
- **Focus**: Speed, memory usage, efficiency.
- **Characteristics**: Measure execution time, resource usage, scalability.
- **Goal**: Detect regressions and benchmark optimizations.

## Naming Conventions

- **Files**: `test_{component}.py` (e.g., `test_attention.py`, `test_qwen_loader.py`).
- **Classes**: `Test{Component}` (e.g., `TestAttentionMechanism`).
- **Functions**: `test_{behavior}` (e.g., `test_returns_correct_shape`).

## Writing Tests

### Guidelines
1.  **Real Code Preference**: Avoid mocks where possible, especially for data structures and configuration. Use real `ConfigManager`, `SystemProfile`, etc.
2.  **Model Loading**: For integration tests involving models, use lightweight fallbacks or the smallest available real model to ensure tests run in reasonable time.
3.  **Imports**: Use absolute imports: `from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel`.

### Example (Unit)
```python
import unittest
from src.inference_pio.common.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    def test_load_config(self):
        manager = ConfigManager()
        # ... assertions
```

## Running Tests

The project uses a custom test runner located at `scripts/run_tests.py`.

```bash
# Run all tests
python scripts/run_tests.py

# Run specific category
python scripts/run_tests.py --category unit

# Run specific test file
python scripts/run_tests.py --file tests/unit/models/test_qwen.py
```

# Developer Guide for Inference-PIO

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Getting Started](#getting-started)
4. [Plugin Development](#plugin-development)
5. [Model Development](#model-development)
6. [Common Components](#common-components)
7. [Testing](#testing)
8. [Performance Optimization](#performance-optimization)
9. [Design Patterns](#design-patterns)
10. [Best Practices](#best-practices)

## Introduction

This guide provides comprehensive information for developers working with the Inference-PIO system. The system follows a self-contained plugin architecture where each model has its own directory with all necessary components.

## Architecture Overview

### Self-Contained Plugin Architecture

The Inference-PIO system is built around a self-contained plugin architecture where each model implementation is completely isolated:

- Each model has its own directory with all necessary components
- Models are independent and can be developed separately
- Common functionality is shared through the `common/` module
- A unified plugin interface ensures consistency across models

### Directory Structure

```
├── benchmarks/             # Benchmark execution scripts
├── benchmark_results/      # Benchmark results and reports
├── config/                 # Configuration files
├── docs/                   # Documentation files
├── plugin_configs/         # Plugin configuration files
├── scripts/                # Utility scripts
├── src/                    # Source code
│   └── inference_pio/      # Main package
│       ├── __init__.py     # Package initialization and exports
│       ├── __main__.py     # Main entry point for CLI
│       ├── common/         # Common utilities
│       ├── design_patterns/ # Design pattern implementations
│       ├── models/         # Model implementations
│       │   ├── __init__.py # Models package initialization
│       │   ├── glm_4_7_flash/    # GLM-4.7-Flash model files
│       │   ├── qwen3_4b_instruct_2507/  # Qwen3-4B-Instruct-2507 model files
│       │   ├── qwen3_coder_30b/         # Qwen3-Coder-30B model files
│       │   └── qwen3_vl_2b/             # Qwen3-VL-2B model files
│       ├── plugin_system/  # Plugin system implementation
├── tests/                  # Core tests
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/inference-pio/inference-pio.git
cd inference-pio

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.inference_pio import (
    create_glm_4_7_flash_plugin,
    create_qwen3_coder_30b_plugin,
    create_qwen3_vl_2b_instruct_plugin,
    create_qwen3_4b_instruct_2507_plugin
)

# Create and use a plugin
plugin = create_glm_4_7_flash_plugin()
plugin.initialize()
plugin.load_model()
result = plugin.infer("Your input text")
plugin.cleanup()
```

## Plugin Development

### Plugin Interface

All plugins must implement the `ModelPluginInterface`.

## Model Development

### Model Configuration

Create a configuration class for your model.

### Model Implementation

Implement your model in the `model.py` file.

### Plugin Implementation

Implement the plugin interface in `plugin.py`.

## Testing

### Test Utilities

The project uses custom test utilities in `src/inference_pio/test_utils.py`.

### Running Tests

To run all tests using the runner script:

```bash
python scripts/run_tests.py
```

To run unit tests for common components:

```bash
python scripts/run_tests.py -d tests/unit/common
```

## Performance Optimization

### Attention Mechanisms

Each model can implement various attention mechanisms:

- FlashAttention 2.0
- Sparse Attention
- Sliding Window Attention
- Multi-Query/Grouped-Query Attention
- Paged Attention

### Memory Optimizations

- KV-Cache Compression
- Paged KV-Cache
- Prefix Caching
- Gradient Checkpointing
- Tensor Parallelism

### Hardware Optimizations

- CUDA Kernels
- Fused Operations
- Mixed Precision
- Tensor Cores

## Design Patterns

The system implements several design patterns:

- **Factory Pattern**: For creating plugin instances
- **Singleton Pattern**: For plugin manager
- **Strategy Pattern**: For different optimization strategies
- **Adapter Pattern**: For integrating different model architectures
- **Decorator Pattern**: For adding functionality to plugins

## Best Practices

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all classes and functions
- Keep functions focused and small (preferably under 50 lines)
- Use meaningful variable and function names
- Follow the DRY (Don't Repeat Yourself) principle
- Use constants for magic numbers and strings

### Testing

- Write comprehensive unit tests
- Include integration tests
- Test error conditions
- Verify resource cleanup
- Test with different input types and sizes
- Include performance benchmarks

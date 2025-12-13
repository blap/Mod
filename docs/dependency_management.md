# Dependency Management for Qwen3-VL

This document explains the dependency management approach for the Qwen3-VL multimodal model optimization project.

## Overview

The project uses a consolidated dependency management approach to simplify installation and maintenance. Dependencies are organized into two main files:

1. `requirements.txt` - Contains core runtime dependencies, power management, and testing dependencies
2. `requirements-dev.txt` - Contains development dependencies (linting, formatting, type checking)

## Requirements Files

### requirements.txt

This file contains all the essential dependencies needed to run the Qwen3-VL model:

- **Core ML Dependencies**: PyTorch, Transformers, Tokenizers, etc.
- **Power Management**: psutil for system monitoring
- **Testing**: pytest and related testing tools
- **Performance Monitoring**: memory-profiler, nvidia-ml-py3
- **Utility Libraries**: numpy, pillow, pandas, etc.

### requirements-dev.txt

This file contains dependencies needed for development:

- **Code Quality**: black (formatting), flake8 (linting), mypy (type checking)
- **Pre-commit Hooks**: pre-commit for enforcing code quality standards

## Installation Methods

### Method 1: Direct pip install from requirements

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Method 2: Using setup.py extras

The setup.py file defines several optional dependency groups:

```bash
# Install with core dependencies
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Install with power management dependencies
pip install -e .[power]

# Install with testing dependencies
pip install -e .[test]

# Install with performance monitoring dependencies
pip install -e .[perf]

# Install with all optional dependencies
pip install -e .[dev,test,perf,power]
```

### Method 3: Using the setup script

```bash
# Basic installation
python scripts/setup_env.py

# Development installation
python scripts/setup_env.py --dev
```

## Version Management

All dependencies specify minimum versions to ensure compatibility while allowing for security updates. When adding new dependencies:

1. Add them to the appropriate requirements file
2. Specify a minimum version that satisfies the project's needs
3. Test compatibility with the specified version range

## Development Workflow

For development work, it's recommended to:

1. Create a virtual environment
2. Install with development dependencies: `pip install -e .[dev,test,perf]`
3. Use pre-commit hooks to maintain code quality: `pre-commit install`

## Migration Notes

This consolidated approach replaces the previous multiple requirements files:
- `requirements_power_management.txt` dependencies have been moved to `requirements.txt`
- Development dependencies remain in `requirements-dev.txt`
- The setup.py file now defines clear optional dependency groups

This consolidation simplifies dependency management while maintaining flexibility for different use cases.
# Dependency Consolidation Summary

## Overview
This document summarizes the changes made to consolidate multiple requirements files into a clean, organized dependency management system.

## Changes Made

### 1. Requirements File Consolidation
- **Removed**: `requirements_power_management.txt` (dependencies moved to consolidated file)
- **Updated**: `requirements.txt` (now includes all core dependencies: runtime, power management, and testing)
- **Updated**: `requirements-dev.txt` (contains development dependencies only)

### 2. Content of requirements.txt
The consolidated `requirements.txt` now includes:
- Core ML dependencies (torch, transformers, tokenizers, etc.)
- Power management dependencies (psutil)
- Testing dependencies (pytest and related tools)
- Performance monitoring tools (memory-profiler, nvidia-ml-py3)
- Utility libraries (numpy, pillow, pandas, etc.)

### 3. Content of requirements-dev.txt
The `requirements-dev.txt` file now contains only development dependencies:
- Code formatting: black
- Code linting: flake8
- Import sorting: isort
- Type checking: mypy
- Pre-commit hooks: pre-commit

### 4. Updated setup.py
- Added multiple optional dependency groups:
  - `dev`: Development tools
  - `torch`: PyTorch-specific dependencies
  - `power`: Power management dependencies
  - `test`: Testing dependencies
  - `perf`: Performance monitoring dependencies
- Maintained backward compatibility with requirements.txt

### 5. Enhanced setup_env.py script
- Added support for development installation with `--dev` flag
- Improved error handling and user feedback
- Maintained backward compatibility

### 6. Updated Documentation
- Enhanced README.md with comprehensive installation instructions
- Created new docs/dependency_management.md with detailed explanation
- Updated cleanup summary document

## Benefits

1. **Simplified Dependency Management**: Single consolidated requirements file for core dependencies
2. **Clear Separation**: Development dependencies separated from core dependencies
3. **Flexible Installation**: Multiple installation methods using pip extras
4. **Maintainability**: Easier to track and update dependencies in one place
5. **Documentation**: Comprehensive documentation of the new approach
6. **Backward Compatibility**: Existing installation methods still work

## Installation Options

### Method 1: Direct requirements install
```bash
pip install -r requirements.txt  # Core dependencies
pip install -r requirements-dev.txt  # Development dependencies
```

### Method 2: Using pip extras
```bash
pip install -e .  # Core only
pip install -e .[dev]  # Core + development
pip install -e .[dev,test,perf,power]  # All optional dependencies
```

### Method 3: Using setup script
```bash
python scripts/setup_env.py  # Basic installation
python scripts/setup_env.py --dev  # Development installation
```

## Migration Path
Projects previously using:
- `requirements.txt` → Continue to work as before
- `requirements-dev.txt` → Continue to work as before  
- `requirements_power_management.txt` → Dependencies now in main requirements.txt
- Direct pip install → Now supports optional extras

The consolidation maintains full backward compatibility while providing a cleaner, more maintainable dependency management system.
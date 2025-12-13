# Installation Guide

This guide explains how to install the Qwen3-VL multimodal model optimization project.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/example/qwen3-vl.git
cd qwen3-vl
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

#### Basic Installation

To install the core dependencies:

```bash
pip install -r requirements.txt
```

#### Development Installation

For development, install both core and development dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### Using pip with extras

You can also install using pip with extras from setup.py:

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

## Hardware Requirements

- NVIDIA GPU with CUDA support (for optimal performance)
- At least 16GB RAM (32GB recommended for large models)
- Sufficient disk space for model weights

## Verification

After installation, you can run the basic tests to verify everything is working:

```bash
python -m pytest tests/unit/ -v
```
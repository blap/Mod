# Contributing to Qwen3-VL

Thank you for your interest in contributing to the Qwen3-VL project! This document provides guidelines and instructions for contributing to the project. Please read it carefully before making any contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and considerate in all interactions.

## Documentation

For comprehensive documentation about the Qwen3-VL project, including user guides, developer documentation, and API references, please see our [main documentation](docs/README.md).

## Getting Started

Before contributing, please:

1. Search existing [Issues](https://github.com/example/qwen3-vl/issues) and [Pull Requests](https://github.com/example/qwen3-vl/pulls) to avoid duplication
2. Discuss significant changes with the maintainers by opening an issue first
3. Ensure your changes are well-documented and tested

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- CUDA-compatible GPU (for GPU functionality, optional for basic development)

### Installation

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/qwen3-vl.git
   cd qwen3-vl
   ```

3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install the package in development mode:
   ```bash
   # Option 1: Using the setup script
   python scripts/setup_env.py --dev
   
   # Option 2: Manual installation
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

5. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

### Using pip with extras

Alternatively, you can install using pip with extras:

```bash
# Install with development dependencies
pip install -e .[dev]

# Install with testing dependencies
pip install -e .[test]

# Install with all optional dependencies
pip install -e .[dev,test,perf,power]
```

## Project Structure

The project is organized as follows:

```
qwen3-vl/
├── configs/                    # Configuration files
│   ├── project_config/         # Build and project configuration
│   ├── model_configs/          # Model-specific configurations
│   ├── default_config.json     # Default application configuration
│   ├── model_config.json       # Model-specific configuration
│   └── training_config.json    # Training configuration
├── src/                        # Source code
│   └── qwen3_vl/               # Main package
│       ├── components/         # Core components
│       ├── cuda_kernels/       # CUDA kernels and GPU-specific code
│       ├── language/           # Language processing modules
│       ├── models/             # Model implementations
│       ├── multimodal/         # Multimodal processing logic
│       ├── utils/              # Utility functions and helpers
│       └── vision/             # Vision processing modules
├── tests/                      # Test files
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── performance/            # Performance tests
│   └── multimodal/             # Multimodal-specific tests
├── docs/                       # Documentation
├── examples/                   # Example implementations
├── benchmarks/                 # Benchmarking tools
├── dev_tools/                  # Development tools
├── scripts/                    # Utility scripts
├── .github/                    # GitHub configuration
├── requirements.txt            # Core dependencies
├── requirements-dev.txt        # Development dependencies
├── setup.py                    # Package setup
└── README.md                   # Project overview
```

## Coding Standards

### Python Code

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings for functions and classes
- Write type hints for all function parameters and return values
- Keep functions and methods focused and reasonably small
- Use descriptive variable and function names
- Follow naming conventions:
  - `snake_case` for variables and functions
  - `UPPER_SNAKE_CASE` for constants
  - `PascalCase` for classes

### Code Quality Tools

The project uses several tools to maintain code quality:

- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking
- `isort` for import sorting

### Example Function Documentation

```python
def process_multimodal_input(
    image: PIL.Image.Image,
    text: str,
    config: Optional[dict] = None
) -> dict:
    """
    Process multimodal input combining image and text.

    Args:
        image: Input image to process
        text: Text input to process
        config: Optional configuration dictionary

    Returns:
        Dictionary containing processed multimodal representation

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If processing fails
    """
    # Implementation here
```

## Testing

### Test Structure

Tests are organized by category:
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for component interactions
- `tests/performance/` - Performance regression tests
- `tests/multimodal/` - Multimodal-specific tests

### Running Tests

To run all tests:
```bash
pytest tests/
```

To run specific test categories:
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Performance tests only
pytest tests/performance/
```

To run with coverage:
```bash
pytest tests/ -v --cov=src/
```

### Test Markers

The test suite uses markers for categorization:
- `multimodal`: Multimodal processing tests
- `cpu`: CPU-specific optimization tests
- `gpu`: GPU-related tests
- `performance`: Performance benchmark tests
- `accuracy`: Accuracy validation tests

To run tests with specific markers:
```bash
pytest tests/ -m "cpu"  # Only CPU-related tests
```

### Writing Tests

When adding new functionality, please include appropriate tests:

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Place performance tests in `tests/performance/`
4. Use descriptive test names following the pattern `test_[what]_[condition]`
5. Include proper assertions and error handling
6. Use fixtures from `conftest.py` where appropriate
7. Add appropriate markers for test categorization

## Submitting Changes

### Pull Request Process

1. Create a new branch from the `main` branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards
3. Add or update tests as needed
4. Ensure all tests pass:
   ```bash
   pytest tests/
   ```
5. Update documentation as needed
6. Commit your changes with a clear, descriptive message:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```
7. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. Open a pull request from your branch to the `main` branch of the original repository

### Pull Request Guidelines

- Use a clear, descriptive title for your PR
- Provide a detailed description of the changes
- Link to any relevant issues
- Ensure your PR addresses a single issue or adds a single feature
- Include tests for new functionality
- Update documentation as needed
- Keep PRs reasonably small (ideally under 500 lines of code)
- Respond to review comments in a timely manner

### Code Review Process

- PRs require at least one approval from maintainers
- Address all review comments before merging
- Maintainers may request changes to ensure code quality
- Once approved, the PR will be merged by a maintainer

## Reporting Issues

### Before Submitting an Issue

- Search existing issues to avoid duplicates
- Ensure you're using the latest version of the code
- Verify the issue still exists in the main branch

### Creating an Issue

When reporting an issue, please provide:

1. **Clear title** - Summarize the issue in a few words
2. **Detailed description** - Explain what happened and what you expected
3. **Steps to reproduce** - Provide minimal steps to reproduce the issue
4. **Environment details**:
   - Python version
   - Operating system
   - Hardware specifications (if relevant)
   - Versions of relevant dependencies
5. **Error messages** - Include full error messages and stack traces
6. **Code snippets** - Provide minimal code examples that reproduce the issue

### Issue Templates

For bug reports, feature requests, and other types of issues, please follow the appropriate template when available.

## Changelog

For details on project releases and changes, see the [CHANGELOG.md](../CHANGELOG.md) file in the project root.

## Community

- Join our discussions on [GitHub Issues](https://github.com/example/qwen3-vl/issues)
- For questions, feel free to open an issue with the "question" label

## Questions?

If you have questions about contributing that aren't covered in this document, please open an issue with the "question" label.

Thank you for contributing to Qwen3-VL!
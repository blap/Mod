# Qwen3-VL Multimodal Model Optimization Project

This project focuses on optimizing the Qwen3-VL multimodal large language model with advanced techniques including:

- Advanced attention mechanisms (Flash Attention 2, Sparse Attention, Dynamic Sparse Attention)
- Memory optimization strategies (KV cache optimization, memory pooling, compression, swapping)
- Performance improvements through hardware-specific optimizations
- Thread safety implementations
- Flexible model system supporting multiple architectures
- Comprehensive error handling and correction systems
- Consolidated module architecture

## Project Structure

```
├── CHANGELOG.md                  # Project changelog
├── CODE_OF_CONDUCT.md            # Code of conduct
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # Project license
├── SECURITY.md                   # Security policy
├── configs/                      # Configuration files
│   ├── app/                      # Runtime application configurations
│   │   ├── default_config.json   # Default application settings
│   │   ├── model_config.json     # Model architecture parameters
│   │   ├── training_config.json  # Training parameters
│   │   ├── inference_config.json # Inference settings
│   │   └── schema.json           # JSON schema for validation
│   ├── dev/                      # Development configurations
│   │   └── .pre-commit-config.yaml # Pre-commit configuration
│   ├── env/                      # Environment configurations
│   │   └── .env.example          # Environment variable template
│   └── infra/                    # Infrastructure configurations
│       └── github-actions/       # GitHub Actions reusable configs
│           ├── action-templates/ # Template actions
│           ├── common-env.sh     # Common environment setup
│           └── matrix-strategy.json # Shared matrix definitions
├── docs/                         # Documentation (organized by category)
│   ├── api/                      # API documentation
│   ├── architecture/             # Architecture documentation
│   │   ├── attention/            # Attention mechanism documentation
│   │   ├── corrections/          # Error correction documentation
│   │   ├── models/               # Model architecture documentation
│   │   └── ...
│   ├── dev/                      # Developer documentation
│   └── user/                     # User documentation
├── src/                          # Source code
│   └── qwen3_vl/                 # Main Qwen3-VL package (monolithic structure)
│       ├── architectures/        # Transformer architecture variants
│       ├── attention/            # Attention mechanism implementations
│       ├── components/           # Core system components (renamed from components_original)
│       ├── components_original/  # Original components module (preserved for compatibility)
│       ├── config/               # Configuration management
│       ├── config_package/       # Original config module (preserved for compatibility)
│       ├── core/                 # Core model implementations
│       ├── cuda_kernels/         # CUDA kernel implementations
│       ├── hardware/             # Hardware abstraction and optimization
│       ├── hardware_optimization/ # Hardware-specific optimizations
│       ├── inference/            # Inference pipeline and utilities
│       ├── language/             # Language model components
│       ├── memory/               # Memory management components
│       ├── memory_management/    # Memory optimization systems
│       ├── model_layers/         # Model layer implementations
│       ├── models/               # Model architecture definitions
│       ├── multimodal/           # Multimodal fusion components
│       ├── optimization/         # General optimization utilities
│       ├── plugin_system/        # Plugin architecture
│       ├── training_strategies/  # Training-specific optimizations
│       ├── utils/                # Utility functions
│       └── vision/               # Vision model components
├── tests/                        # Test files
├── examples/                     # Example implementations
├── benchmarks/                   # Benchmarking tools
├── dev_tools/                    # Development tools
└── scripts/                      # Utility scripts
```

## Installation

### Basic Installation

To install the project dependencies:

```bash
pip install .
```

### Development Installation

For development, install the package with all optional dependencies:

```bash
pip install -e .[all-dev]
```

Or install with specific optional dependency groups:

```bash
# Install with development dependencies
pip install -e .[dev]

# Install with power management dependencies
pip install -e .[power]

# Install with testing dependencies
pip install -e .[test]

# Install with performance monitoring dependencies
pip install -e .[perf]

# Install with all optional dependencies
pip install -e .[all-dev]
```

## Dependency Management

This project uses `pyproject.toml` for dependency management:

- `pyproject.toml` - Defines all dependencies and optional dependency groups for flexible installation

## Code Quality and Pre-commit Hooks

This project uses pre-commit hooks to maintain code quality and consistency. Pre-commit is a framework for managing git hooks that are run before commits are made.

### Installation

To install the pre-commit hooks:

```bash
# Install the full configuration (recommended for development)
pre-commit install

# Or install the recommended configuration for faster CI/CD
pre-commit install --config .pre-commit-config-recommended.yaml
```

### Configuration Files

The project includes pre-commit configurations:

- `configs/dev/.pre-commit-config.yaml` - Full configuration with comprehensive code quality checks

For more details about the pre-commit setup, see [PRE_COMMIT_GUIDE.md](governance/PRE_COMMIT_GUIDE.md) in the governance directory.

### Running Pre-commit Hooks

To run all hooks on all files manually:

```bash
pre-commit run --all-files
```

To run with a specific configuration:

```bash
pre-commit run --all-files --config .pre-commit-config-recommended.yaml
```

## Module Organization

The project follows a monolithic package architecture where all functionality is consolidated into a single main package:

- `src/qwen3_vl/` - Main Qwen3-VL package containing all implementations with:
  - `architectures/` - Different transformer architecture variants
  - `attention/` - Attention mechanisms including Flash Attention, Block Sparse Attention, and Dynamic Sparse Attention
  - `components/` - Core system components and dependency injection (renamed from components_original)
  - `components_original/` - Original components module (preserved for backward compatibility)
  - `config/` - Configuration management and unified configuration system
  - `config_package/` - Original config module (preserved for backward compatibility)
  - `core/` - Core model implementations
  - `cuda_kernels/` - CUDA kernel implementations for GPU acceleration
  - `hardware/` - Hardware abstraction and optimization components
  - `inference/` - Inference pipeline and CLI tools
  - `language/` - Language model components
  - `memory/` - Memory management components
  - `memory_management/` - Advanced memory optimization systems
  - `models/` - Model architecture definitions with language, vision, and multimodal components
  - `multimodal/` - Multimodal fusion components
  - `optimization/` - General optimization utilities
  - `utils/` - Utility functions specific to Qwen3-VL
  - `vision/` - Vision model components

## Configuration

All configuration files are located in the `configs/` directory, organized by purpose:
- `configs/app/` - Runtime application configurations
- `configs/dev/` - Development configurations
- `configs/env/` - Environment configurations
- `configs/infra/` - Infrastructure configurations

The project uses a hierarchical configuration system that allows for default, model-specific, and environment-specific settings. The unified configuration system in `src/qwen3_vl/config/` provides hardware-aware optimization settings and runtime configuration updates.

## Changelog

See the [CHANGELOG.md](CHANGELOG.md) file for details on project releases and changes.

## Documentation

Comprehensive documentation about the Qwen3-VL project is organized by category:

- **Architecture Documentation**: Detailed information about system design, attention mechanisms, error handling, and model architecture
  - [Attention Mechanisms](docs/architecture/attention/README.md) - Information about attention consolidation and optimization
  - [Error Correction System](docs/architecture/corrections/README.md) - Details about corrections applied to fix attribute errors
  - [Model Architecture](docs/architecture/models/README.md) - Documentation on the flexible model system
  - [Complete Architecture Overview](docs/architecture/README.md) - High-level architecture overview

- **Developer Documentation**: Resources for developers contributing to or extending the project
  - [Development Guides](docs/dev/guides/README.md) - Information for setting up development environment and practices
  - [API Documentation](docs/api/README.md) - Technical reference for Qwen3-VL APIs

- **User Documentation**: Resources for end users of the Qwen3-VL model
  - [Getting Started Guide](docs/user/guides/getting_started/README.md) - Basic setup and first steps

For a complete table of contents and additional resources, see our [documentation index](docs/INDEX.md) and [full documentation summary](docs/SUMMARY.md).

## Contributing

We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md) for details on how to get started.

## Usage Examples

Check out the [examples/](examples/) directory for practical usage examples demonstrating the new consolidated module architecture:

- `examples/qwen3_vl/inference/basic_usage.py` - Basic inference example
- `examples/components/configuration/demo_unified_config_system.py` - Configuration system demonstration
- `examples/qwen3_vl/memory_management/` - Memory optimization examples
- `examples/qwen3_vl/optimization/` - Optimization strategy examples

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
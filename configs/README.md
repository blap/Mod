# Configuration Files

This directory contains all configuration files for the Qwen3-VL multimodal model optimization project, organized by category.

## Directory Structure

```
configs/
├── model_configs/           # Model-specific configurations
│   └── ...                  # Various model configuration files
├── default_config.json      # Default application configuration
├── model_config.json        # Primary model configuration
├── training_config.json     # Training configuration parameters
├── schema.json              # JSON schema for configuration validation
└── docs/
    └── CONFIGURATION.md     # Comprehensive configuration documentation
```

## Configuration Categories

### Project Configuration
- **pyproject.toml** (in root): Contains build system configuration, dependencies, and project metadata

### Model Configuration
- **default_config.json**: Default application settings and parameters
- **model_config.json**: Model-specific architecture and hyperparameters
- **training_config.json**: Training-specific parameters and settings
- **model_configs/**: Additional model-specific configuration files

### Validation & Documentation
- **schema.json**: JSON schema for validating configuration files
- **docs/CONFIGURATION.md**: Comprehensive documentation for all configuration settings

## Usage

When running the application or tests, ensure the configuration files are accessible. The application will automatically load configurations from this directory.

## Best Practices

- Keep all configuration files in this directory
- Use appropriate subdirectories for organization
- Maintain consistent naming conventions
- Document any custom configuration parameters
- Refer to docs/CONFIGURATION.md for detailed explanations of all settings
- Use schema.json to validate configuration files
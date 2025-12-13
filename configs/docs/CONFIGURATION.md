# Configuration Guide for Qwen3-VL

This document provides a comprehensive guide to all configuration files and settings used in the Qwen3-VL multimodal model project.

## Table of Contents

1. [Project Configuration](#project-configuration)
2. [Model Configuration](#model-configuration)
3. [Training Configuration](#training-configuration)
4. [Inference Configuration](#inference-configuration)
5. [Development Configuration](#development-configuration)

## Project Configuration

### pyproject.toml

The main project configuration file containing build system, dependencies, and tool configurations.

#### Project Metadata
- `name`: "qwen3-vl" - Project name
- `version`: "1.0.0" - Current project version
- `authors`: Development team information
- `description`: Brief project description
- `readme`: Path to README.md file
- `license`: MIT license
- `requires-python`: Minimum Python version (>=3.8)

#### Dependencies
- Core dependencies include PyTorch, Transformers, and other essential libraries
- Optional dependency groups for different use cases:
  - `dev`: Development tools (black, flake8, isort, mypy, pre-commit)
  - `test`: Testing frameworks (pytest and related plugins)
  - `perf`: Performance monitoring tools
  - `power`: Power management tools
  - `all-dev`: All development-related dependencies

#### Entry Points
- `qwen3-vl-infer`: Command-line interface for model inference

#### Tool Configurations
- **Black**: Code formatter with line length 88
- **Isort**: Import sorter with Black profile
- **Mypy**: Static type checker for Python 3.8
- **Pytest**: Test runner with coverage reporting

## Model Configuration

### default_config.json

Contains default model architecture parameters:

- `model_name`: "Qwen3-VL-2B-Instruct" - Name of the model
- `model_type`: "qwen3_vl" - Type identifier
- `task_type`: "multimodal" - Task category
- `hidden_size`: 2048 - Size of hidden layers
- `num_hidden_layers`: 32 - Number of hidden transformer layers
- `num_attention_heads`: 32 - Number of attention heads
- `intermediate_size`: 11008 - Size of intermediate layers
- `hidden_act`: "silu" - Activation function
- `max_position_embeddings`: 32768 - Maximum sequence length
- `vocab_size`: 152064 - Vocabulary size
- `vision_hidden_size`: 1152 - Vision encoder hidden size
- `vision_num_hidden_layers`: 24 - Vision encoder layers
- `vision_num_attention_heads`: 16 - Vision encoder attention heads
- `vision_patch_size`: 14 - Vision patch size
- `vision_image_size`: 448 - Input image size
- `num_query_tokens`: 64 - Number of query tokens
- `torch_dtype`: "float16" - Default tensor type
- `use_gradient_checkpointing`: true - Enable gradient checkpointing
- Advanced features (all false by default):
  - `use_adapters`
  - `use_sparsity`
  - `use_dynamic_sparse_attention`
  - `use_adaptive_depth`
  - `use_context_adaptive_positional_encoding`
  - `use_conditional_feature_extraction`

### model_config.json

Extends default_config.json with additional training and inference parameters:

#### Training Configuration
- `learning_rate`: 5e-5 - Initial learning rate
- `batch_size`: 16 - Training batch size
- `num_epochs`: 3 - Number of training epochs
- `warmup_steps`: 100 - Number of warmup steps
- `weight_decay`: 0.01 - Weight decay factor
- `adam_epsilon`: 1e-8 - Adam epsilon value
- `max_grad_norm`: 1.0 - Maximum gradient norm

#### Inference Configuration
- `max_new_tokens`: 512 - Maximum new tokens to generate
- `temperature`: 0.7 - Generation temperature
- `top_k`: 50 - Top-k sampling parameter
- `top_p`: 0.95 - Nucleus sampling parameter
- `repetition_penalty`: 1.2 - Repetition penalty factor

### training_config.json

Contains detailed training-specific parameters:

#### Training Parameters
- `logging_steps`: 10 - Steps between logging
- `eval_steps`: 500 - Steps between evaluation
- `save_steps`: 1000 - Steps between saving checkpoints
- `max_steps`: -1 (unlimited) - Maximum training steps
- `gradient_accumulation_steps`: 1 - Gradient accumulation steps
- `warmup_ratio`: 0.1 - Warmup ratio
- `lr_scheduler_type`: "linear" - Learning rate scheduler type
- `fp16`: true - Use FP16 precision
- `dataloader_num_workers`: 4 - Number of data loader workers
- `load_best_model_at_end`: true - Load best model at end
- `metric_for_best_model`: "loss" - Metric for best model selection
- `greater_is_better`: false - Whether higher metric values are better

#### Model Configuration in Training
- `use_adapters`: true - Enable adapter layers
- `adapter_config`: Configuration for adapter layers
  - `reduction_factor`: 16 - Adapter reduction factor
  - `non_linearity`: "relu" - Adapter activation function
  - `scaling_factor`: 1.0 - Adapter scaling factor

## Development Configuration

### .pre-commit-config.yaml

Configuration for pre-commit hooks that run automatically before each commit:

- **Black**: Code formatting
- **Isort**: Import organization
- **Flake8**: Code linting
- **Mypy**: Type checking

### .gitignore

Comprehensive ignore patterns for Python projects including:

- Python cache files and directories
- Virtual environments
- IDE-specific files
- Test cache and coverage reports
- Log files
- Environment files
- OS-specific files
- Jupyter notebook checkpoints
- Model files and datasets
- Distribution packages

### Schema Validation

The `schema.json` file provides JSON Schema definitions for validating configuration files. This ensures configuration files adhere to the expected structure and data types.

## Best Practices

1. Always validate configuration files against the schema before deployment
2. Use environment-specific overrides for sensitive configurations
3. Keep default configurations minimal and extendable
4. Document any custom configuration parameters
5. Use consistent naming conventions across all configuration files
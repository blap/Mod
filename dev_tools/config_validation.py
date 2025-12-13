"""
DEPRECATED: Configuration Validation Tools

This module has been moved to configs.utils.config_validation
Please update your imports to use the new location.

This file is kept for backward compatibility and will be removed in a future version.
"""

import warnings

# Import everything from the new location
from configs.utils.config_validation import (
    ConfigSchema,
    ConfigValidationError,
    ConfigValidator,
    ConfigManager,
    ConfigValidatorDecorator,
    global_config_validator,
    config_manager,
    validate_config_arg,
    setup_qwen3_vl_schemas,
    example_validation,
    example_config_manager
)

# Issue a deprecation warning
warnings.warn(
    "config_validation module has been moved to configs.utils.config_validation. "
    "Please update your imports to use the new location.",
    DeprecationWarning,
    stacklevel=2
)
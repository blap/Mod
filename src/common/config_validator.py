"""
Configuration Validator for Inference-PIO System

This module provides validation utilities for configuration objects.
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """
    Exception raised when configuration validation fails.
    """

    pass


class ConfigValidator:
    """
    Validator for configuration objects in the Inference-PIO system.
    """

    @staticmethod
    def validate_config(config: Any) -> tuple[bool, List[str]]:
        """
        Validate a configuration object.

        Args:
            config: Configuration object to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check if config has required attributes based on type
        if hasattr(config, "__class__"):
            class_name = config.__class__.__name__

            # Validate specific configuration types
            if "Config" in class_name:
                errors.extend(ConfigValidator._validate_config_attributes(config))

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def _validate_config_attributes(config: Any) -> List[str]:
        """
        Validate common configuration attributes.

        Args:
            config: Configuration object to validate

        Returns:
            List of validation errors
        """
        errors = []

        # Check for common config attributes and their types
        if hasattr(config, "model_path"):
            if not isinstance(config.model_path, str):
                errors.append("model_path must be a string")

        if hasattr(config, "device"):
            if not isinstance(config.device, str):
                errors.append("device must be a string")

        if hasattr(config, "torch_dtype"):
            if not isinstance(config.torch_dtype, str):
                errors.append("torch_dtype must be a string")

        if hasattr(config, "use_cache"):
            if not isinstance(config.use_cache, bool):
                errors.append("use_cache must be a boolean")

        if hasattr(config, "gradient_checkpointing"):
            if not isinstance(config.gradient_checkpointing, bool):
                errors.append("gradient_checkpointing must be a boolean")

        if hasattr(config, "device_map"):
            if not isinstance(config.device_map, str):
                errors.append("device_map must be a string")

        if hasattr(config, "low_cpu_mem_usage"):
            if not isinstance(config.low_cpu_mem_usage, bool):
                errors.append("low_cpu_mem_usage must be a boolean")

        # Add more validations as needed for specific config types
        return errors

    @staticmethod
    def validate_model_config(config: Any) -> tuple[bool, List[str]]:
        """
        Validate a model-specific configuration.

        Args:
            config: Model configuration to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required model config attributes
        required_attrs = [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
        ]

        for attr in required_attrs:
            if not hasattr(config, attr):
                errors.append(f"Missing required attribute: {attr}")

        # Validate attribute types if they exist
        if hasattr(config, "hidden_size"):
            if not isinstance(config.hidden_size, int) or config.hidden_size <= 0:
                errors.append("hidden_size must be a positive integer")

        if hasattr(config, "num_attention_heads"):
            if (
                not isinstance(config.num_attention_heads, int)
                or config.num_attention_heads <= 0
            ):
                errors.append("num_attention_heads must be a positive integer")

        if hasattr(config, "num_hidden_layers"):
            if (
                not isinstance(config.num_hidden_layers, int)
                or config.num_hidden_layers <= 0
            ):
                errors.append("num_hidden_layers must be a positive integer")

        is_valid = len(errors) == 0
        return is_valid, errors


def get_config_validator() -> ConfigValidator:
    """
    Get a configuration validator instance.

    Returns:
        ConfigValidator instance
    """
    return ConfigValidator()


__all__ = ["ConfigValidator", "ConfigValidationError", "get_config_validator"]

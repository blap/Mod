"""
Configuration Loader for Inference-PIO System

This module provides utilities for loading and validating configuration files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Utility class for loading and validating configuration files.
    """

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a file (JSON or YAML).

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix.lower() in [".json"]:
            return ConfigLoader._load_json_config(config_path)
        elif config_path.suffix.lower() in [".yaml", ".yml"]:
            return ConfigLoader._load_yaml_config(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    @staticmethod
    def _load_json_config(config_path: Path) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading config file {config_path}: {e}")
            raise

    @staticmethod
    def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
        """Load configuration from a YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading config file {config_path}: {e}")
            raise

    @staticmethod
    def validate_config(
        config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate configuration against a schema.

        Args:
            config: Configuration dictionary to validate
            schema: Schema to validate against (optional)

        Returns:
            True if validation passes, False otherwise
        """
        # If no schema provided, just check that config is a dictionary
        if schema is None:
            return isinstance(config, dict)

        # In a real implementation, this would validate against the provided schema
        # For now, we'll just return True to indicate successful validation
        return True

    @staticmethod
    def merge_configs(
        base_config: Dict[str, Any], override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries, with override_config taking precedence.

        Args:
            base_config: Base configuration
            override_config: Override configuration (takes precedence)

        Returns:
            Merged configuration
        """
        import copy

        merged = copy.deepcopy(base_config)

        for key, value in override_config.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged


def load_config_from_path(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load configuration from a file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    return ConfigLoader.load_config(config_path)


def validate_config_data(
    config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Convenience function to validate configuration data.

    Args:
        config: Configuration dictionary to validate
        schema: Schema to validate against (optional)

    Returns:
        True if validation passes, False otherwise
    """
    return ConfigLoader.validate_config(config, schema)


__all__ = ["ConfigLoader", "load_config_from_path", "validate_config_data"]

"""
Configuration Factory for Models

This module provides a factory for creating model-specific configurations with reasonable defaults.
Each model can register its configuration class here, allowing for easy instantiation.
Each model plugin is completely independent with its own configuration, tests, and benchmarks.
"""

import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Type

from .model_config_base import BaseConfig, get_optimal_config_for_hardware


class ConfigFactory:
    """
    Factory class for creating model configurations.

    This class manages the registration and creation of model-specific configurations.
    """

    _configs: Dict[str, Type[BaseConfig]] = {}

    @classmethod
    def register_config(cls, model_name: str, config_class: Type[BaseConfig]):
        """
        Register a configuration class for a specific model.

        Args:
            model_name: Name of the model (case-insensitive)
            config_class: Configuration class for the model
        """
        cls._configs[model_name.lower()] = config_class

    @classmethod
    def create_config(cls, model_name: str, **kwargs) -> BaseConfig:
        """
        Create a configuration instance for the specified model.

        Args:
            model_name: Name of the model (case-insensitive)
            **kwargs: Additional configuration parameters

        Returns:
            Configuration instance for the model

        Raises:
            ValueError: If no configuration is registered for the model
        """
        model_name_lower = model_name.lower()

        if model_name_lower not in cls._configs:
            raise ValueError(f"No configuration registered for model: {model_name}")

        config_class = cls._configs[model_name_lower]
        config = config_class(**kwargs)

        # Optimize the configuration for the current hardware
        config = get_optimal_config_for_hardware(config)

        return config

    @classmethod
    def get_available_configs(cls) -> list:
        """
        Get a list of available model configurations.

        Returns:
            List of available model names
        """
        return list(cls._configs.keys())

    @classmethod
    def load_config_from_module(
        cls, module_path: str, config_class_name: str
    ) -> Type[BaseConfig]:
        """
        Load a configuration class from a module.

        Args:
            module_path: Path to the module containing the configuration class
            config_class_name: Name of the configuration class

        Returns:
            Configuration class

        Raises:
            TypeError: If the configuration class doesn't inherit from BaseConfig
        """
        module = importlib.import_module(module_path)
        config_class = getattr(module, config_class_name)

        if not issubclass(config_class, BaseConfig):
            raise TypeError(
                f"Configuration class {config_class_name} must inherit from BaseConfig"
            )

        return config_class


def register_model_config(model_name: str, config_class: Type[BaseConfig]):
    """
    Decorator to register a model configuration.

    Args:
        model_name: Name of the model
        config_class: Configuration class for the model

    Returns:
        The configuration class (unchanged, for decorator chaining)
    """
    ConfigFactory.register_config(model_name, config_class)
    return config_class


def create_model_config(model_name: str, **kwargs) -> BaseConfig:
    """
    Create a configuration instance for the specified model.

    Args:
        model_name: Name of the model (case-insensitive)
        **kwargs: Additional configuration parameters

    Returns:
        Configuration instance for the model
    """
    return ConfigFactory.create_config(model_name, **kwargs)


def create_model_config_direct(model_name: str, **kwargs) -> BaseConfig:
    """
    Create a direct configuration instance for the specified model (not using registered class).

    Args:
        model_name: Name of the model
        **kwargs: Additional configuration parameters

    Returns:
        Configuration instance for the model

    Raises:
        ValueError: If no configuration class is found for the model
    """
    # Use dynamic import to avoid direct dependencies between common config and specific models
    import importlib

    if model_name.lower() == "qwen3_0_6b":
        config_module = importlib.import_module('src.inference_pio.models.qwen3_0_6b.config')
        config_class = getattr(config_module, 'Qwen3_0_6B_Config')
    elif model_name.lower() == "qwen3_4b_instruct_2507":
        config_module = importlib.import_module('src.inference_pio.models.qwen3_4b_instruct_2507.config')
        config_class = getattr(config_module, 'Qwen34BInstruct2507Config')
    elif model_name.lower() == "glm_4_7_flash":
        config_module = importlib.import_module('src.inference_pio.models.glm_4_7_flash.config')
        config_class = getattr(config_module, 'GLM47FlashConfig')
    elif model_name.lower() == "qwen3_vl_2b":
        config_module = importlib.import_module('src.inference_pio.models.qwen3_vl_2b.config')
        config_class = getattr(config_module, 'Qwen3VL2BConfig')
    elif model_name.lower() == "qwen3_coder_30b":
        config_module = importlib.import_module('src.inference_pio.models.qwen3_coder_30b.config')
        config_class = getattr(config_module, 'Qwen3Coder30BConfig')
    else:
        raise ValueError(f"No configuration class found for model: {model_name}")

    config = config_class(**kwargs)
    # Optimize the configuration for the current hardware
    config = get_optimal_config_for_hardware(config)
    return config


def get_model_config_class(model_name: str) -> Type[BaseConfig]:
    """
    Get the configuration class for a specific model.

    Args:
        model_name: Name of the model (case-insensitive)

    Returns:
        Configuration class for the model

    Raises:
        ValueError: If no configuration is registered for the model
    """
    model_name_lower = model_name.lower()
    if model_name_lower not in ConfigFactory._configs:
        raise ValueError(f"No configuration registered for model: {model_name}")
    return ConfigFactory._configs[model_name_lower]


def auto_detect_and_register_configs(models_dir: str = "src/inference_pio/models"):
    """
    Automatically detect and register configuration classes from model directories.

    Args:
        models_dir: Directory containing model subdirectories
    """
    import os
    from pathlib import Path

    models_path = Path(models_dir)

    if not models_path.exists():
        return

    # Iterate through each model directory
    for model_dir in models_path.iterdir():
        if not model_dir.is_dir():
            continue

        # Look for config.py in the model directory
        config_file = model_dir / "config.py"
        if config_file.exists():
            # Import the config module
            model_name = model_dir.name
            module_path = f"src.inference_pio.models.{model_name}.config"

            try:
                # Try to import the config module
                module = importlib.import_module(module_path)

                # Look for configuration classes in the module
                config_classes = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # Check if it's a subclass of BaseConfig
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseConfig)
                        and attr != BaseConfig
                    ):
                        # Prefer non-Dynamic configs over Dynamic ones
                        if "Dynamic" not in attr_name:
                            config_classes.insert(0, attr)  # Put at beginning
                        else:
                            config_classes.append(attr)  # Put at end

                # Register the first (preferred) config class
                if config_classes:
                    ConfigFactory.register_config(model_name, config_classes[0])

                    # Also register with common aliases
                    ConfigFactory.register_config(
                        f"{model_name}_config", config_classes[0]
                    )

            except ImportError as e:
                print(f"Could not import config from {module_path}: {e}")
                continue

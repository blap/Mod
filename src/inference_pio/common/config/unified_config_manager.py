"""
Unified Configuration Manager for Inference-PIO System

This module provides a consolidated configuration system that combines the functionality
from multiple configuration modules to reduce duplication and improve maintainability.
"""

import copy
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import yaml

from .model_config_base import BaseConfig, ModelConfigError
from .optimization_manager import OptimizationConfig, get_optimization_manager

# from .optimization_profiles import (
#     OptimizationProfile,
#     PerformanceProfile,
#     MemoryEfficientProfile,
#     BalancedProfile,
#     GLM47Profile,
#     Qwen34BProfile,
#     Qwen3CoderProfile,
#     Qwen3VLProfile
# )
# Optimization profiles have been temporarily removed due to restructuring
# Will be reimplemented in a future update

logger = logging.getLogger(__name__)


class UnifiedConfigManager:
    """
    Centralized manager for handling unified configurations across all models.
    Combines functionality from config_manager, config_loader, and config_validator.
    """

    def __init__(self):
        self._configs: Dict[str, BaseConfig] = {}
        self._config_history: Dict[str, List[BaseConfig]] = {}
        self._active_configs: Dict[str, str] = {}  # model_id -> config_name mapping
        self._config_templates: Dict[str, BaseConfig] = {}

        # Optimization-related managers
        self._optimization_manager = get_optimization_manager()

    def register_config_template(self, name: str, template: BaseConfig) -> bool:
        """
        Register a configuration template that can be used to create new configurations.

        Args:
            name: Name of the template
            template: Template configuration object

        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Deep copy the template to avoid reference issues
            self._config_templates[name] = copy.deepcopy(template)
            logger.info(f"Registered configuration template: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register configuration template {name}: {e}")
            return False

    def create_config_from_template(
        self,
        template_name: str,
        config_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[BaseConfig]:
        """
        Create a new configuration from a template with optional overrides.

        Args:
            template_name: Name of the template to use
            config_name: Name for the new configuration
            overrides: Optional dictionary of field overrides

        Returns:
            New configuration object or None if creation failed
        """
        if template_name not in self._config_templates:
            logger.error(f"Template '{template_name}' not found")
            return None

        try:
            # Clone the template
            new_config = copy.deepcopy(self._config_templates[template_name])

            # Apply overrides if provided
            if overrides:
                for key, value in overrides.items():
                    if hasattr(new_config, key):
                        setattr(new_config, key, value)

            # Store the configuration
            self._configs[config_name] = new_config

            # Add to history
            if config_name not in self._config_history:
                self._config_history[config_name] = []
            self._config_history[config_name].append(copy.deepcopy(new_config))

            logger.info(
                f"Created configuration '{config_name}' from template '{template_name}'"
            )
            return new_config
        except Exception as e:
            logger.error(
                f"Failed to create configuration from template {template_name}: {e}"
            )
            return None

    def register_config(self, name: str, config: BaseConfig) -> bool:
        """
        Register a configuration with the manager.

        Args:
            name: Name of the configuration
            config: Configuration object

        Returns:
            True if registration was successful, False otherwise
        """
        try:
            self._configs[name] = copy.deepcopy(config)

            # Add to history
            if name not in self._config_history:
                self._config_history[name] = []
            self._config_history[name].append(copy.deepcopy(config))

            logger.info(f"Registered configuration: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register configuration {name}: {e}")
            return False

    def get_config(self, name: str) -> Optional[BaseConfig]:
        """
        Get a configuration by name.

        Args:
            name: Name of the configuration

        Returns:
            Configuration object or None if not found
        """
        return self._configs.get(name)

    def update_config(self, name: str, updates: Dict[str, Any]) -> bool:
        """
        Update a configuration with new values.

        Args:
            name: Name of the configuration to update
            updates: Dictionary of field updates

        Returns:
            True if update was successful, False otherwise
        """
        if name not in self._configs:
            logger.error(f"Configuration '{name}' not found")
            return False

        try:
            config = self._configs[name]

            # Update configuration fields
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Field '{key}' not found in configuration '{name}'")

            # Add to history
            if name not in self._config_history:
                self._config_history[name] = []
            self._config_history[name].append(copy.deepcopy(config))

            logger.info(f"Updated configuration: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration {name}: {e}")
            return False

    def delete_config(self, name: str) -> bool:
        """
        Delete a configuration.

        Args:
            name: Name of the configuration to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        if name not in self._configs:
            logger.error(f"Configuration '{name}' not found")
            return False

        try:
            del self._configs[name]
            if name in self._config_history:
                del self._config_history[name]

            # Remove from active configs if present
            keys_to_remove = []
            for model_id, config_name in self._active_configs.items():
                if config_name == name:
                    keys_to_remove.append(model_id)
            for key in keys_to_remove:
                del self._active_configs[key]

            logger.info(f"Deleted configuration: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete configuration {name}: {e}")
            return False

    def save_config(self, name: str, filepath: str, format: str = "json") -> bool:
        """
        Save a configuration to a file.

        Args:
            name: Name of the configuration to save
            filepath: Path to save the file
            format: Format to save in ("json" or "yaml")

        Returns:
            True if save was successful, False otherwise
        """
        config = self.get_config(name)
        if not config:
            logger.error(f"Configuration '{name}' not found")
            return False

        try:
            data = config.to_dict()

            if format.lower() == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format.lower() == "yaml":
                with open(filepath, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Saved configuration '{name}' to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration {name} to {filepath}: {e}")
            return False

    def load_config_from_file(self, config_path: str, config_name: str) -> bool:
        """
        Load a configuration from a file.

        Args:
            config_path: Path to the configuration file
            config_name: Name to assign to the loaded configuration

        Returns:
            True if loading was successful, False otherwise
        """
        try:
            path = Path(config_path)
            if not path.exists():
                logger.error(f"Configuration file does not exist: {config_path}")
                return False

            # Determine file format based on extension
            if path.suffix.lower() in [".yaml", ".yml"]:
                with open(path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            else:
                logger.error(f"Unsupported file format: {path.suffix}")
                return False

            # Determine the appropriate config class based on model name in config
            model_name = config_data.get("model_name", "").lower()

            # For now, we'll create a generic config and update it
            # In practice, you'd want to use the appropriate config class
            from .model_config_base import BaseConfig

            class GenericConfig(BaseConfig):
                def __init__(self, **kwargs):
                    # Initialize with default values
                    super().__init__()
                    # Update with provided values
                    for k, v in kwargs.items():
                        if hasattr(self, k):
                            setattr(self, k, v)

                def get_model_specific_params(self) -> Dict[str, Any]:
                    return {}

            config_obj = GenericConfig(**config_data)
            return self.register_config(config_name, config_obj)
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            return False

    def list_configs(self) -> List[str]:
        """
        List all registered configurations.

        Returns:
            List of configuration names
        """
        return list(self._configs.keys())

    def get_config_history(self, name: str) -> List[BaseConfig]:
        """
        Get the history of changes for a configuration.

        Args:
            name: Name of the configuration

        Returns:
            List of previous versions of the configuration
        """
        return self._config_history.get(name, [])

    def activate_config_for_model(self, model_id: str, config_name: str) -> bool:
        """
        Activate a configuration for a specific model.

        Args:
            model_id: Identifier for the model
            config_name: Name of the configuration to activate

        Returns:
            True if activation was successful, False otherwise
        """
        if config_name not in self._configs:
            logger.error(f"Configuration '{config_name}' not found")
            return False

        self._active_configs[model_id] = config_name
        logger.info(f"Activated configuration '{config_name}' for model '{model_id}'")
        return True

    def get_active_config_for_model(self, model_id: str) -> Optional[BaseConfig]:
        """
        Get the active configuration for a specific model.

        Args:
            model_id: Identifier for the model

        Returns:
            Active configuration or None if no configuration is active
        """
        config_name = self._active_configs.get(model_id)
        if config_name:
            return self.get_config(config_name)
        return None

    def get_config_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a configuration.

        Args:
            name: Name of the configuration

        Returns:
            Metadata dictionary or None if config not found
        """
        config = self.get_config(name)
        if not config:
            return None

        return {
            "name": name,
            "type": type(config).__name__,
            "field_count": len([f for f in dir(config) if not f.startswith("_")]),
            "history_length": len(self.get_config_history(name)),
            "is_active_in_models": [
                model_id
                for model_id, config_name in self._active_configs.items()
                if config_name == name
            ],
        }

    def validate_config(self, config: BaseConfig) -> tuple[bool, List[str]]:
        """
        Validate a configuration object.

        Args:
            config: Configuration object to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Basic validation checks
        if not config.model_path:
            errors.append("Model path must be specified")

        if config.torch_dtype not in ["float16", "float32", "bfloat16"]:
            errors.append(f"Invalid torch_dtype: {config.torch_dtype}")

        # Additional validation can be added here based on specific requirements
        return len(errors) == 0, errors


# Global unified configuration manager instance
unified_config_manager = UnifiedConfigManager()


def get_unified_config_manager() -> UnifiedConfigManager:
    """
    Get the global unified configuration manager instance.

    Returns:
        UnifiedConfigManager instance
    """
    return unified_config_manager


# Helper functions for creating configurations from optimization profiles
def create_config_from_profile(
    model_type: str,
    profile_name: str,
    config_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Create a configuration from an optimization profile.

    Args:
        model_type: Type of model ('glm', 'qwen3_4b', 'qwen3_coder', 'qwen3_vl')
        profile_name: Name of the profile to use (will be created from template if it doesn't exist)
        config_name: Name for the new configuration
        overrides: Optional dictionary of field overrides

    Returns:
        True if creation was successful, False otherwise
    """
    try:
        manager = get_unified_config_manager()

        # Create a base configuration based on model type using dynamic imports to avoid cross-dependencies
        if model_type == "glm":
            # Use dynamic import to avoid direct dependency
            import importlib
            config_module = importlib.import_module('.models.glm_4_7_flash.config', package='src.inference_pio.common.config')
            config_class = getattr(config_module, 'GLM47FlashConfig')
        elif model_type == "qwen3_4b":
            # Use dynamic import to avoid direct dependency
            import importlib
            config_module = importlib.import_module('.models.qwen3_4b_instruct_2507.config', package='src.inference_pio.common.config')
            config_class = getattr(config_module, 'Qwen34BInstruct2507Config')
        elif model_type == "qwen3_coder":
            # Use dynamic import to avoid direct dependency
            import importlib
            config_module = importlib.import_module('.models.qwen3_coder_30b.config', package='src.inference_pio.common.config')
            config_class = getattr(config_module, 'Qwen3Coder30BConfig')
        elif model_type == "qwen3_vl":
            # Use dynamic import to avoid direct dependency
            import importlib
            config_module = importlib.import_module('.models.qwen3_vl_2b.config', package='src.inference_pio.common.config')
            config_class = getattr(config_module, 'Qwen3VL2BConfig')
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create an instance of the config
        config = config_class()

        # Apply profile-specific settings based on profile name
        if profile_name == "performance":
            # Performance-optimized settings
            config.use_flash_attention_2 = True
            config.use_sparse_attention = True
            config.use_paged_attention = True
            config.gradient_checkpointing = False  # Disable for better performance
            config.torch_compile_mode = "reduce-overhead"
            config.torch_compile_fullgraph = False
            config.enable_kernel_fusion = True
            config.use_tensor_parallelism = True
            config.enable_adaptive_batching = True
            config.max_batch_size = 32
            config.use_quantization = True
            config.quantization_bits = 8
        elif profile_name == "memory_efficient":
            # Memory-efficient settings
            config.use_flash_attention_2 = True
            config.use_sparse_attention = True
            config.use_paged_attention = True
            config.gradient_checkpointing = True  # Enable for memory savings
            config.torch_dtype = "float16"  # Use half precision
            config.enable_memory_management = True
            config.max_memory_ratio = 0.6  # Limit memory usage
            config.enable_disk_offloading = True
            config.enable_activation_offloading = True
            config.enable_tensor_compression = True
            config.tensor_compression_ratio = 0.5
            config.use_tensor_decomposition = True
            config.use_structured_pruning = True
            config.pruning_ratio = 0.2
            config.use_quantization = True
            config.quantization_bits = 4
        elif profile_name == "balanced":
            # Balanced settings
            config.use_flash_attention_2 = True
            config.use_sparse_attention = True
            config.use_paged_attention = True
            config.gradient_checkpointing = True
            config.torch_compile_mode = "reduce-overhead"
            config.torch_compile_fullgraph = False
            config.enable_kernel_fusion = True
            config.enable_memory_management = True
            config.max_memory_ratio = 0.8
            config.enable_disk_offloading = True
            config.enable_activation_offloading = True
            config.enable_tensor_compression = True
            config.tensor_compression_ratio = 0.4
            config.use_quantization = True
            config.quantization_bits = 8
            config.enable_adaptive_batching = True
            config.max_batch_size = 16

        # Apply overrides if provided
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Register the config
        return manager.register_config(config_name, config)
    except Exception as e:
        logger.error(
            f"Failed to create config from profile {profile_name} for model type {model_type}: {e}"
        )
        return False


__all__ = [
    "UnifiedConfigManager",
    "get_unified_config_manager",
    "unified_config_manager",
    "create_config_from_profile",
]

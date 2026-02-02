"""
Base Configuration System for Models

This module provides a standardized configuration system that can be used by all models
in the Inference-PIO system. It includes base classes and utilities for model configuration.
Each model plugin is completely independent with its own configuration, tests, and benchmarks.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class ModelConfigError(Exception):
    """Custom exception for model configuration errors."""

    pass


@dataclass
class BaseConfig(ABC):
    """
    Abstract base class for all model configurations.

    This class defines the interface that all model configurations must implement.
    It provides common parameters and methods for model configuration management.
    """

    # Model identification
    model_path: str = ""
    model_name: str = ""

    # Device settings
    device: Optional[str] = None
    device_map: str = "auto"

    # Memory optimization
    gradient_checkpointing: bool = True
    use_cache: bool = True
    low_cpu_mem_usage: bool = True
    max_memory: Optional[Dict] = None

    # Data type
    torch_dtype: str = "float16"

    # Hardware optimization
    use_tensor_parallelism: bool = False

    # Attention mechanisms
    use_flash_attention_2: bool = True
    use_sdpa: bool = True

    # KV Cache
    use_paged_kv_cache: bool = True
    paged_attention_page_size: int = 16

    # Throughput
    use_continuous_batching: bool = True

    def __post_init__(self):
        """Initialize the configuration after creation."""
        self._validate_config()

    @abstractmethod
    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return model-specific parameters."""
        pass

    def _validate_config(self):
        """
        Validate the configuration parameters.

        This method checks that all required configuration parameters are set correctly
        and raises ModelConfigError if any validation fails.
        """
        if not self.model_path:
            raise ModelConfigError("Model path must be specified")

        if self.torch_dtype not in ["float16", "float32", "bfloat16"]:
            raise ModelConfigError(f"Invalid torch_dtype: {self.torch_dtype}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        Returns:
            A dictionary representation of the configuration object.
        """
        result = {}
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if isinstance(value, BaseConfig):
                result[field_name] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                result[field_name] = [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in value
                ]
            elif isinstance(value, dict):
                result[field_name] = {
                    k: v.to_dict() if hasattr(v, "to_dict") else v
                    for k, v in value.items()
                }
            else:
                result[field_name] = value
        return result

    def save_to_file(self, filepath: Union[str, Path]):
        """
        Save the configuration to a JSON file.

        Args:
            filepath: Path to the file where the configuration should be saved.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "BaseConfig":
        """
        Load the configuration from a JSON file.

        Args:
            filepath: Path to the file to load the configuration from.

        Returns:
            A new instance of the configuration class loaded from the file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """
        Create a configuration instance from a dictionary.

        Args:
            data: Dictionary containing configuration parameters.

        Returns:
            A new instance of the configuration class created from the dictionary.
        """
        # This is a simplified implementation - in practice, you'd need to handle
        # nested objects and type conversion properly
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def update(self, **kwargs):
        """
        Update configuration parameters.

        Args:
            **kwargs: Key-value pairs of configuration parameters to update.

        Raises:
            ModelConfigError: If an unknown configuration parameter is provided.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ModelConfigError(f"Unknown configuration parameter: {key}")


class ConfigurableModelMixin:
    """
    A mixin class that adds configuration capabilities to model classes.
    """

    def __init__(self, config: BaseConfig):
        self.config = config

    def get_config(self) -> BaseConfig:
        """Get the current configuration."""
        return self.config

    def update_config(self, **kwargs):
        """Update the configuration."""
        self.config.update(**kwargs)

    def save_config(self, filepath: Union[str, Path]):
        """Save the current configuration to a file."""
        self.config.save_to_file(filepath)


def get_default_model_path(model_name: str) -> str:
    """
    Get a default model path based on the model name.

    Args:
        model_name: Name of the model

    Returns:
        Default path for the model
    """
    # Try to find the model in common locations
    possible_paths = [
        f"./models/{model_name}",
        f"../models/{model_name}",
        f"H:/{model_name}",  # Common Windows location
        f"C:/{model_name}",
        f"~/.cache/huggingface/transformers/{model_name}",
    ]

    for path in possible_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return expanded_path

    # If no path exists, return a default path
    return f"./models/{model_name}"


def detect_hardware_capabilities() -> Dict[str, Any]:
    """
    Detect hardware capabilities and return appropriate configuration settings.

    Returns:
        Dictionary with hardware-specific configuration recommendations
    """
    try:
        import torch

        capabilities = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": 0,
            "gpu_memory": [],
            "mps_available": hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available(),
        }

        if torch.cuda.is_available():
            capabilities["cuda_devices"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                capabilities["gpu_memory"].append(
                    {
                        "device": i,
                        "name": gpu_props.name,
                        "total_memory": gpu_props.total_memory,
                        "major": gpu_props.major,
                        "minor": gpu_props.minor,
                    }
                )

        return capabilities
    except ImportError:
        # PyTorch not available, return basic capabilities
        return {
            "cuda_available": False,
            "cuda_devices": 0,
            "gpu_memory": [],
            "mps_available": False,
        }


def get_optimal_config_for_hardware(base_config: BaseConfig) -> BaseConfig:
    """
    Adjust the configuration based on detected hardware capabilities.

    Args:
        base_config: Base configuration to adjust

    Returns:
        Adjusted configuration optimized for hardware
    """
    hardware_caps = detect_hardware_capabilities()

    # Adjust memory settings based on available GPU memory
    if hardware_caps["cuda_available"] and hardware_caps["gpu_memory"]:
        # Use the first GPU's memory as reference
        gpu_memory_bytes = hardware_caps["gpu_memory"][0]["total_memory"]
        gpu_memory_gb = gpu_memory_bytes / (1024**3)

        # Set max memory to 80% of available GPU memory
        base_config.max_memory = {0: f"{int(gpu_memory_gb * 0.8)}GB", "cpu": "20GB"}

        # Adjust dtype based on GPU capability
        if gpu_memory_gb < 8:
            # For lower memory GPUs, use float16
            base_config.torch_dtype = "float16"
        elif gpu_memory_gb >= 24:
            # For high memory GPUs, use bfloat16 if supported
            base_config.torch_dtype = "bfloat16"

    # Adjust tensor parallelism based on number of GPUs
    if hardware_caps["cuda_devices"] > 1:
        base_config.use_tensor_parallelism = True

    return base_config

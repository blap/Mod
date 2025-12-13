"""
Configuration factory for Qwen3-VL model with dependency injection support.

This factory provides methods to load, save, and validate configurations
for the Qwen3-VL model with proper support for modular configuration components.
"""
import json
import os
from typing import Dict, Any, Optional
from dataclasses import asdict, fields
from .config import Qwen3VLConfig
from .memory_config import MemoryConfig
from .attention_config import AttentionConfig
from .routing_config import RoutingConfig
from .hardware_config import HardwareConfig


class ConfigFactory:
    """
    Factory class for creating and managing configurations with dependency injection support.
    Provides methods to load, save, and validate configurations for the Qwen3-VL model.
    """

    @staticmethod
    def from_json_file(file_path: str) -> Qwen3VLConfig:
        """
        Load configuration from a JSON file.

        Args:
            file_path: Path to the JSON configuration file

        Returns:
            Qwen3VLConfig instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        return ConfigFactory.from_dict(config_dict)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> Qwen3VLConfig:
        """
        Create configuration from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            Qwen3VLConfig instance
        """
        # Extract modular configuration components from the dict
        memory_config_dict = config_dict.pop('memory_config', {})
        attention_config_dict = config_dict.pop('attention_config', {})
        routing_config_dict = config_dict.pop('routing_config', {})
        hardware_config_dict = config_dict.pop('hardware_config', {})

        # Filter out keys that are not in the main dataclass
        valid_keys = {field.name for field in fields(Qwen3VLConfig) if field.name not in 
                      ['memory_config', 'attention_config', 'routing_config', 'hardware_config']}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        # Create modular configuration instances
        memory_config = MemoryConfig(**memory_config_dict) if memory_config_dict else None
        attention_config = AttentionConfig(**attention_config_dict) if attention_config_dict else None
        routing_config = RoutingConfig(**routing_config_dict) if routing_config_dict else None
        hardware_config = HardwareConfig(**hardware_config_dict) if hardware_config_dict else None

        # Create main config with modular components
        config = Qwen3VLConfig(**filtered_dict)
        
        # Assign modular configurations if they were provided
        if memory_config:
            config.memory_config = memory_config
        if attention_config:
            config.attention_config = attention_config
        if routing_config:
            config.routing_config = routing_config
        if hardware_config:
            config.hardware_config = hardware_config

        return config

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str, **kwargs) -> Qwen3VLConfig:
        """
        Load configuration from a pretrained model path or hub.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or model name
            **kwargs: Additional arguments to override configuration

        Returns:
            Qwen3VLConfig instance
        """
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")

        if not os.path.exists(config_path):
            # If config.json doesn't exist, try to find it in common locations
            possible_paths = [
                os.path.join(pretrained_model_name_or_path, "config.json"),
                os.path.join(pretrained_model_name_or_path, "model_config.json"),
                os.path.join(pretrained_model_name_or_path, "default_config.json"),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            else:
                # If no config file exists, create with default values
                config = Qwen3VLConfig()
                # Override with any provided kwargs
                for key, value in kwargs.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                return config

        # Load from file
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # Override with any provided kwargs
        for key, value in kwargs.items():
            if key in config_dict or hasattr(Qwen3VLConfig, key):
                config_dict[key] = value

        return ConfigFactory.from_dict(config_dict)

    @staticmethod
    def save_config(config: Qwen3VLConfig, save_path: str) -> None:
        """
        Save configuration to a JSON file.

        Args:
            config: Qwen3VLConfig instance to save
            save_path: Path where to save the configuration
        """
        # Convert the main config to dict
        config_dict = asdict(config)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @staticmethod
    def create_default_config() -> Qwen3VLConfig:
        """
        Create a default configuration instance.

        Returns:
            Qwen3VLConfig instance with default values
        """
        return Qwen3VLConfig()

    @staticmethod
    def create_optimized_config_for_hardware(hardware_type: str = "intel_i5_10210u") -> Qwen3VLConfig:
        """
        Create a configuration optimized for specific hardware.

        Args:
            hardware_type: Type of hardware to optimize for

        Returns:
            Qwen3VLConfig instance with hardware-optimized settings
        """
        config = Qwen3VLConfig()

        if hardware_type.lower() == "intel_i5_10210u":
            # Apply optimizations specific to Intel i5-10210U
            # These are based on the advanced optimizations implemented in the project
            if config.memory_config:
                config.memory_config.use_gradient_checkpointing = True  # Memory efficiency
                config.memory_config.use_activation_sparsity = True  # Activation sparsity for efficiency
                config.memory_config.sparsity_ratio = 0.5  # 50% sparsity
                config.memory_config.kv_cache_strategy = "hybrid"  # Hybrid KV cache optimization
            config.torch_dtype = "float16"  # Better performance on this hardware
            if config.routing_config:
                config.routing_config.use_moe = True  # Mixture of Experts for parameter efficiency
                config.routing_config.moe_num_experts = 4  # 4 experts as specified in architecture
                config.routing_config.moe_top_k = 2  # Top-2 routing as specified
            if config.attention_config:
                config.attention_config.use_flash_attention_2 = True  # Efficient attention
                config.attention_config.attention_implementation = "flash_attention_2"  # Set implementation
                config.attention_config.use_dynamic_sparse_attention = True  # Dynamic sparse attention
            config.use_adaptive_depth = True  # Adaptive depth for efficiency
            config.use_context_adaptive_positional_encoding = True  # Learned positional encodings
            config.use_conditional_feature_extraction = True  # Conditional features
        elif hardware_type.lower() == "nvidia_sm61":
            # Apply optimizations specific to NVIDIA SM61
            if config.memory_config:
                config.memory_config.use_gradient_checkpointing = True
            config.torch_dtype = "float16"
            if config.attention_config:
                config.attention_config.use_flash_attention_2 = True
                config.attention_config.attention_implementation = "flash_attention_2"  # Set implementation
            if config.memory_config:
                config.memory_config.kv_cache_strategy = "low_rank"
            if config.routing_config:
                config.routing_config.use_moe = True
                config.routing_config.moe_num_experts = 4
                config.routing_config.moe_top_k = 2
            if config.hardware_config:
                config.hardware_config.enable_tensor_cores = True
        elif hardware_type.lower() == "generic":
            # Generic optimization
            if config.memory_config:
                config.memory_config.use_gradient_checkpointing = True
            config.torch_dtype = "float16"
        else:
            # Default config for unknown hardware
            pass

        return config
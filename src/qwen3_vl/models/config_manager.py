"""
Model configuration manager for handling different model architectures and requirements.

This module provides model-specific configuration loading that adapts to different architectures.
"""

import json
import yaml
from typing import Dict, Any, Optional, Union, Type
from pathlib import Path
import logging
from dataclasses import dataclass, fields
import torch


@dataclass
class ModelConfig:
    """Base model configuration class."""
    model_name: str = ""
    model_type: str = "language"  # "language", "vision", "multimodal", "other"
    torch_dtype: Optional[str] = None
    device_map: Optional[Dict[str, Any]] = None
    memory_requirements: Dict[str, float] = None  # Memory requirements in GB
    performance_profile: str = "balanced"  # "balanced", "memory_efficient", "performance"
    
    def __post_init__(self):
        if self.memory_requirements is None:
            self.memory_requirements = {
                "min_memory_gb": 4.0,
                "recommended_memory_gb": 8.0,
                "max_memory_gb": 16.0
            }


class ConfigManager:
    """
    Manager for handling different model configurations and adapting to different architectures.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._config_templates: Dict[str, Dict[str, Any]] = {}
        self._config_validators: Dict[str, callable] = {}
    
    def register_config_template(self, model_name: str, template: Dict[str, Any]) -> bool:
        """
        Register a configuration template for a specific model.
        
        Args:
            model_name: Name of the model
            template: Configuration template dictionary
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            self._config_templates[model_name] = template
            self._logger.info(f"Configuration template registered for {model_name}")
            return True
        except Exception as e:
            self._logger.error(f"Error registering config template for {model_name}: {e}")
            return False
    
    def register_config_validator(self, model_name: str, validator: callable) -> bool:
        """
        Register a configuration validator for a specific model.
        
        Args:
            model_name: Name of the model
            validator: Validation function
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            self._config_validators[model_name] = validator
            self._logger.info(f"Configuration validator registered for {model_name}")
            return True
        except Exception as e:
            self._logger.error(f"Error registering config validator for {model_name}: {e}")
            return False
    
    def load_config(
        self,
        model_name: str,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load configuration for a specific model.
        
        Args:
            model_name: Name of the model
            config_path: Path to configuration file (optional)
            config_dict: Configuration dictionary (optional)
            
        Returns:
            Configuration dictionary
        """
        # Start with default config
        config = self._get_default_config(model_name)
        
        # Load from file if provided
        if config_path:
            config_path = Path(config_path)
            if config_path.suffix.lower() in ['.json', '.yaml', '.yml']:
                loaded_config = self._load_config_from_file(config_path)
                config.update(loaded_config)
        
        # Update with provided config dict
        if config_dict:
            config.update(config_dict)
        
        # Validate config if validator exists
        if model_name in self._config_validators:
            try:
                self._config_validators[model_name](config)
            except Exception as e:
                self._logger.warning(f"Configuration validation failed for {model_name}: {e}")
        
        return config
    
    def _get_default_config(self, model_name: str) -> Dict[str, Any]:
        """Get default configuration for a model."""
        if model_name in self._config_templates:
            return self._config_templates[model_name].copy()
        else:
            # Return a generic default config
            return {
                "model_name": model_name,
                "model_type": "language",
                "torch_dtype": "float16",
                "device_map": None,
                "memory_requirements": {
                    "min_memory_gb": 4.0,
                    "recommended_memory_gb": 8.0,
                    "max_memory_gb": 16.0
                },
                "performance_profile": "balanced"
            }
    
    def _load_config_from_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from a file."""
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def adapt_config_for_hardware(
        self,
        config: Dict[str, Any],
        available_memory_gb: float,
        compute_capability: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        Adapt configuration based on available hardware resources.
        
        Args:
            config: Original configuration
            available_memory_gb: Available memory in GB
            compute_capability: Hardware compute capability (optional)
            
        Returns:
            Adapted configuration
        """
        adapted_config = config.copy()
        
        # Adjust memory-related settings based on available memory
        memory_requirements = config.get('memory_requirements', {})
        min_memory = memory_requirements.get('min_memory_gb', 4.0)
        
        if available_memory_gb < min_memory:
            self._logger.warning(
                f"Available memory ({available_memory_gb}GB) is less than minimum required "
                f"({min_memory}GB). This may cause performance issues or failures."
            )
        
        # Adjust dtype based on memory if needed
        if available_memory_gb < 6.0 and config.get('torch_dtype') == 'float32':
            self._logger.info("Reducing precision from float32 to float16 due to memory constraints")
            adapted_config['torch_dtype'] = 'float16'
        
        # Adjust performance profile based on memory
        if available_memory_gb < 6.0:
            adapted_config['performance_profile'] = 'memory_efficient'
        elif available_memory_gb >= 12.0:
            adapted_config['performance_profile'] = 'performance'
        else:
            adapted_config['performance_profile'] = 'balanced'
        
        # Add hardware-specific optimizations if compute capability is provided
        if compute_capability:
            major, minor = compute_capability
            if major >= 8:  # Modern GPUs with better tensor cores
                adapted_config.setdefault('use_flash_attention', True)
                adapted_config.setdefault('use_tensor_cores', True)
            else:
                adapted_config.setdefault('use_flash_attention', False)
                adapted_config.setdefault('use_tensor_cores', False)
        
        return adapted_config
    
    def validate_config(self, model_name: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a specific model.
        
        Args:
            model_name: Name of the model
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        if model_name in self._config_validators:
            try:
                self._config_validators[model_name](config)
                return True
            except Exception as e:
                self._logger.error(f"Configuration validation failed for {model_name}: {e}")
                return False
        else:
            # If no specific validator, just check for basic required fields
            required_fields = ['model_name', 'model_type']
            return all(field in config for field in required_fields)


# Global config manager instance
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """
    Get the global config manager instance.
    
    Returns:
        ConfigManager instance
    """
    return config_manager


# Register default configuration templates for known models
def _register_default_configs():
    """Register default configuration templates for known models."""
    
    # Qwen3-VL configuration template
    qwen3_vl_config = {
        "model_name": "Qwen3-VL",
        "model_type": "multimodal",
        "torch_dtype": "float16",
        "device_map": None,
        "memory_requirements": {
            "min_memory_gb": 8.0,
            "recommended_memory_gb": 16.0,
            "max_memory_gb": 24.0
        },
        "performance_profile": "balanced",
        # Qwen3-VL specific parameters
        "vocab_size": 152064,
        "hidden_size": 2048,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "max_position_embeddings": 32768,
        "use_flash_attention_2": True,
        "use_gradient_checkpointing": True,
        "use_mixed_precision": True
    }
    
    config_manager.register_config_template("Qwen3-VL", qwen3_vl_config)
    
    # Qwen3-4B configuration template
    qwen3_4b_config = {
        "model_name": "Qwen3-4B-Instruct-2507",
        "model_type": "language",
        "torch_dtype": "float16",
        "device_map": None,
        "memory_requirements": {
            "min_memory_gb": 6.0,
            "recommended_memory_gb": 8.0,
            "max_memory_gb": 12.0
        },
        "performance_profile": "balanced",
        # Qwen3-4B specific parameters
        "vocab_size": 151936,
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "num_key_value_heads": 20,
        "intermediate_size": 6912,
        "max_position_embeddings": 32768,
        "use_flash_attention_2": True,
        "use_gradient_checkpointing": True,
        "use_mixed_precision": True,
        "rope_theta": 1000000.0
    }
    
    config_manager.register_config_template("Qwen3-4B-Instruct-2507", qwen3_4b_config)


_register_default_configs()
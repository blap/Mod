"""
Configuration Loader for Inference-PIO System

This module provides utilities for loading, saving, and managing configurations
for different models in the Inference-PIO system.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .config_manager import (
    get_config_manager,
    GLM47DynamicConfig,
    Qwen34BDynamicConfig,
    Qwen3CoderDynamicConfig,
    Qwen3VLDynamicConfig
)
from .optimization_profiles import (
    get_profile_manager,
    PerformanceProfile,
    MemoryEfficientProfile,
    BalancedProfile,
    GLM47Profile,
    Qwen34BProfile,
    Qwen3CoderProfile,
    Qwen3VLProfile
)
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Utility class for loading and saving configurations for different models.
    """

    def __init__(self):
        self.config_manager = get_config_manager()
        self.profile_manager = get_profile_manager()
    
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
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                logger.error(f"Unsupported file format: {path.suffix}")
                return False
            
            # Determine the appropriate config class based on model name in config
            model_name = config_data.get('model_name', '').lower()
            
            if 'glm' in model_name:
                config_obj = GLM47DynamicConfig(**config_data)
            elif 'qwen3' in model_name and 'coder' in model_name:
                config_obj = Qwen3CoderDynamicConfig(**config_data)
            elif 'qwen3' in model_name and ('vl' in model_name or 'vision' in model_name):
                config_obj = Qwen3VLDynamicConfig(**config_data)
            elif 'qwen3' in model_name:
                config_obj = Qwen34BDynamicConfig(**config_data)
            else:
                # Default to a generic config if model type is not recognized
                from .config_manager import DynamicConfig
                
                class GenericConfig(DynamicConfig):
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                
                config_obj = GenericConfig(**config_data)
            
            return self.config_manager.register_config(config_name, config_obj)
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            return False
    
    def save_config_to_file(self, config_name: str, config_path: str, 
                          format: str = "json") -> bool:
        """
        Save a configuration to a file.
        
        Args:
            config_name: Name of the configuration to save
            config_path: Path to save the configuration file
            format: Format to save in ("json" or "yaml")
            
        Returns:
            True if saving was successful, False otherwise
        """
        return self.config_manager.save_config(config_name, config_path, format)
    
    def create_config_from_template(self, template_name: str, config_name: str, 
                                  overrides: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a configuration from a template with optional overrides.
        
        Args:
            template_name: Name of the template to use
            config_name: Name for the new configuration
            overrides: Optional dictionary of field overrides
            
        Returns:
            True if creation was successful, False otherwise
        """
        result = self.config_manager.create_config_from_template(
            template_name, config_name, overrides
        )
        return result is not None
    
    def get_available_templates(self) -> list:
        """
        Get list of available configuration templates.
        
        Returns:
            List of template names
        """
        # Access the internal templates dictionary to get available templates
        return list(self.config_manager._config_templates.keys())
    
    def list_saved_configs(self, config_dir: str) -> list:
        """
        List saved configuration files in a directory.
        
        Args:
            config_dir: Directory to search for configuration files
            
        Returns:
            List of configuration file paths
        """
        config_dir_path = Path(config_dir)
        if not config_dir_path.exists():
            logger.warning(f"Configuration directory does not exist: {config_dir}")
            return []
        
        config_files = []
        for ext in ['.json', '.yaml', '.yml']:
            config_files.extend(config_dir_path.glob(f"*{ext}"))
        
        return [str(f) for f in config_files]

    def apply_profile_to_config(self, config_name: str, profile_name: str) -> bool:
        """
        Apply an optimization profile to a configuration.

        Args:
            config_name: Name of the configuration to modify
            profile_name: Name of the optimization profile to apply

        Returns:
            True if application was successful, False otherwise
        """
        return self.profile_manager.apply_profile_to_config(profile_name, self.config_manager.get_config(config_name))

    def create_config_from_profile(self, model_type: str, profile_name: str, config_name: str,
                                 overrides: Optional[Dict[str, Any]] = None) -> bool:
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
            # Create a base configuration based on model type
            if model_type == 'glm':
                config_class = GLM47DynamicConfig
            elif model_type == 'qwen3_4b':
                config_class = Qwen34BDynamicConfig
            elif model_type == 'qwen3_coder':
                config_class = Qwen3CoderDynamicConfig
            elif model_type == 'qwen3_vl':
                config_class = Qwen3VLDynamicConfig
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Create an instance of the config
            config = config_class()

            # Check if profile exists, if not try to create it from template
            profile_exists = self.profile_manager.get_profile(profile_name) is not None
            if not profile_exists:
                # Try to create profile from template
                success = self.profile_manager.create_profile_from_template(profile_name, profile_name, {})
                if not success:
                    logger.error(f"Profile '{profile_name}' not found and could not be created from template")
                    return False

            # Apply the profile to the config
            success = self.profile_manager.apply_profile_to_config(profile_name, config)
            if not success:
                return False

            # Apply overrides if provided
            if overrides:
                for key, value in overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            # Register the config
            return self.config_manager.register_config(config_name, config)
        except Exception as e:
            logger.error(f"Failed to create config from profile {profile_name} for model type {model_type}: {e}")
            return False

    def list_available_profiles(self) -> list:
        """
        Get list of available optimization profiles.

        Returns:
            List of profile names
        """
        return self.profile_manager.list_profiles()

    def get_profile_metadata(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about an optimization profile.

        Args:
            profile_name: Name of the profile

        Returns:
            Metadata dictionary or None if profile not found
        """
        return self.profile_manager.get_profile_metadata(profile_name)


# Global configuration loader instance
config_loader = ConfigLoader()


def get_config_loader() -> ConfigLoader:
    """
    Get the global configuration loader instance.
    
    Returns:
        ConfigLoader instance
    """
    return config_loader


# Predefined configuration profiles for different use cases
def create_performance_optimized_config(model_type: str, **overrides) -> Dict[str, Any]:
    """
    Create a performance-optimized configuration for a specific model type.
    
    Args:
        model_type: Type of model ('glm', 'qwen3_4b', 'qwen3_coder', 'qwen3_vl')
        **overrides: Additional overrides to apply
        
    Returns:
        Dictionary of configuration parameters
    """
    base_config = {
        'use_flash_attention_2': True,
        'use_sparse_attention': True,
        'use_paged_attention': True,
        'gradient_checkpointing': False,  # Disable for better performance
        'torch_compile_mode': 'reduce-overhead',
        'torch_compile_fullgraph': False,
        'enable_kernel_fusion': True,
        'use_tensor_parallelism': True,
        'enable_adaptive_batching': True,
        'max_batch_size': 32,
        'use_quantization': True,
        'quantization_bits': 8,
    }
    
    # Model-specific optimizations
    if model_type == 'glm':
        model_specific = {
            'use_glm_attention_patterns': True,
            'use_glm_ffn_optimization': True,
            'use_glm_memory_efficient_kv': True,
        }
    elif model_type == 'qwen3_4b':
        model_specific = {
            'use_qwen3_attention_optimizations': True,
            'use_qwen3_kv_cache_optimizations': True,
            'use_qwen3_instruction_optimizations': True,
        }
    elif model_type == 'qwen3_coder':
        model_specific = {
            'use_qwen3_coder_attention_optimizations': True,
            'use_qwen3_coder_kv_cache_optimizations': True,
            'use_qwen3_coder_code_optimizations': True,
        }
    elif model_type == 'qwen3_vl':
        model_specific = {
            'use_qwen3_vl_attention_optimizations': True,
            'use_qwen3_vl_kv_cache_optimizations': True,
            'use_qwen3_vl_vision_optimizations': True,
        }
    else:
        model_specific = {}
    
    # Combine base and model-specific configs
    perf_config = {**base_config, **model_specific, **overrides}
    return perf_config


def create_memory_efficient_config(model_type: str, **overrides) -> Dict[str, Any]:
    """
    Create a memory-efficient configuration for a specific model type.
    
    Args:
        model_type: Type of model ('glm', 'qwen3_4b', 'qwen3_coder', 'qwen3_vl')
        **overrides: Additional overrides to apply
        
    Returns:
        Dictionary of configuration parameters
    """
    base_config = {
        'use_flash_attention_2': True,
        'use_sparse_attention': True,
        'use_paged_attention': True,
        'gradient_checkpointing': True,  # Enable for memory savings
        'torch_dtype': 'float16',  # Use half precision
        'enable_memory_management': True,
        'max_memory_ratio': 0.6,  # Limit memory usage
        'enable_disk_offloading': True,
        'enable_activation_offloading': True,
        'enable_tensor_compression': True,
        'tensor_compression_ratio': 0.5,
        'use_tensor_decomposition': True,
        'use_structured_pruning': True,
        'pruning_ratio': 0.2,
        'use_quantization': True,
        'quantization_bits': 4,
    }
    
    # Model-specific optimizations
    if model_type == 'glm':
        model_specific = {
            'use_glm_attention_patterns': True,
            'use_glm_memory_efficient_kv': True,
            'glm_kv_cache_compression_ratio': 0.6,
        }
    elif model_type == 'qwen3_4b':
        model_specific = {
            'use_qwen3_attention_optimizations': True,
            'use_qwen3_kv_cache_optimizations': True,
            'qwen3_kv_cache_compression_ratio': 0.6,
        }
    elif model_type == 'qwen3_coder':
        model_specific = {
            'use_qwen3_coder_attention_optimizations': True,
            'use_qwen3_coder_kv_cache_optimizations': True,
            'qwen3_coder_kv_cache_compression_ratio': 0.6,
        }
    elif model_type == 'qwen3_vl':
        model_specific = {
            'use_qwen3_vl_attention_optimizations': True,
            'use_qwen3_vl_kv_cache_optimizations': True,
            'qwen3_vl_kv_cache_compression_ratio': 0.6,
        }
    else:
        model_specific = {}
    
    # Combine base and model-specific configs
    mem_config = {**base_config, **model_specific, **overrides}
    return mem_config


def create_balanced_config(model_type: str, **overrides) -> Dict[str, Any]:
    """
    Create a balanced configuration for a specific model type.
    
    Args:
        model_type: Type of model ('glm', 'qwen3_4b', 'qwen3_coder', 'qwen3_vl')
        **overrides: Additional overrides to apply
        
    Returns:
        Dictionary of configuration parameters
    """
    base_config = {
        'use_flash_attention_2': True,
        'use_sparse_attention': True,
        'use_paged_attention': True,
        'gradient_checkpointing': True,
        'torch_compile_mode': 'reduce-overhead',
        'torch_compile_fullgraph': False,
        'enable_kernel_fusion': True,
        'enable_memory_management': True,
        'max_memory_ratio': 0.8,
        'enable_disk_offloading': True,
        'enable_activation_offloading': True,
        'enable_tensor_compression': True,
        'tensor_compression_ratio': 0.4,
        'use_quantization': True,
        'quantization_bits': 8,
        'enable_adaptive_batching': True,
        'max_batch_size': 16,
    }
    
    # Model-specific optimizations
    if model_type == 'glm':
        model_specific = {
            'use_glm_attention_patterns': True,
            'use_glm_memory_efficient_kv': True,
            'glm_kv_cache_compression_ratio': 0.5,
        }
    elif model_type == 'qwen3_4b':
        model_specific = {
            'use_qwen3_attention_optimizations': True,
            'use_qwen3_kv_cache_optimizations': True,
            'qwen3_kv_cache_compression_ratio': 0.5,
        }
    elif model_type == 'qwen3_coder':
        model_specific = {
            'use_qwen3_coder_attention_optimizations': True,
            'use_qwen3_coder_kv_cache_optimizations': True,
            'qwen3_coder_kv_cache_compression_ratio': 0.5,
        }
    elif model_type == 'qwen3_vl':
        model_specific = {
            'use_qwen3_vl_attention_optimizations': True,
            'use_qwen3_vl_kv_cache_optimizations': True,
            'qwen3_vl_kv_cache_compression_ratio': 0.5,
        }
    else:
        model_specific = {}
    
    # Combine base and model-specific configs
    balanced_config = {**base_config, **model_specific, **overrides}
    return balanced_config


# Configuration profile registry
CONFIG_PROFILES = {
    'performance': create_performance_optimized_config,
    'memory_efficient': create_memory_efficient_config,
    'balanced': create_balanced_config,
}


def create_config_from_profile(model_type: str, profile_name: str, **overrides) -> Dict[str, Any]:
    """
    Create a configuration from a predefined profile.
    
    Args:
        model_type: Type of model ('glm', 'qwen3_4b', 'qwen3_coder', 'qwen3_vl')
        profile_name: Name of the profile ('performance', 'memory_efficient', 'balanced')
        **overrides: Additional overrides to apply
        
    Returns:
        Dictionary of configuration parameters
    """
    if profile_name not in CONFIG_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(CONFIG_PROFILES.keys())}")
    
    profile_func = CONFIG_PROFILES[profile_name]
    return profile_func(model_type, **overrides)


__all__ = [
    "ConfigLoader",
    "get_config_loader",
    "config_loader",
    "create_performance_optimized_config",
    "create_memory_efficient_config",
    "create_balanced_config",
    "create_config_from_profile",
    "CONFIG_PROFILES",
]
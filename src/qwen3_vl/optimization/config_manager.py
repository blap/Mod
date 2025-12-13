"""
Configuration Management System for Qwen3-VL Optimization Techniques
Provides centralized management of optimization configurations with validation and validation.
"""
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
import copy
from enum import Enum


class OptimizationLevel(Enum):
    """Enumeration for optimization levels"""
    MINIMAL = "minimal"
    MODERATE = "moderate" 
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class OptimizationConfig:
    """Configuration for all 12 optimization techniques"""
    # Block sparse attention
    block_sparse_attention: bool = True
    block_size: int = 64
    
    # Cross-modal token merging
    cross_modal_token_merging: bool = True
    token_merging_threshold: float = 0.1
    
    # Hierarchical memory compression
    hierarchical_memory_compression: bool = True
    compression_level: str = "medium"
    
    # Learned activation routing
    learned_activation_routing: bool = True
    num_activation_functions: int = 4
    
    # Adaptive batch processing
    adaptive_batch_processing: bool = True
    max_batch_size: int = 32
    
    # Cross-layer parameter recycling
    cross_layer_parameter_recycling: bool = True
    recycling_frequency: int = 4
    
    # Adaptive sequence packing
    adaptive_sequence_packing: bool = True
    max_packed_length: int = 1024
    
    # Memory-efficient gradient accumulation
    memory_efficient_grad_accumulation: bool = True
    grad_accumulation_steps: int = 1
    
    # KV cache multiple strategies
    kv_cache_multiple_strategies: bool = True
    kv_cache_strategy: str = "hybrid"
    
    # Faster rotary embeddings
    faster_rotary_embeddings: bool = True
    use_approximated_rotary: bool = True
    
    # Distributed pipeline parallelism
    distributed_pipeline_parallelism: bool = True
    pipeline_stages: int = 4
    
    # Hardware-specific kernels
    hardware_specific_kernels: bool = True
    use_tensor_cores: bool = True
    
    # Model capacity preservation settings
    preserve_32_layers: bool = True
    preserve_32_attention_heads: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary"""
        # Create a new instance and set attributes from the dictionary
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    @classmethod
    def from_json(cls, json_str: str):
        """Create configuration from JSON string"""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]):
        """Load configuration from JSON or YAML file"""
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def save_to_file(self, file_path: Union[str, Path], format: str = 'json'):
        """Save configuration to file"""
        file_path = Path(file_path)
        config_dict = self.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'yaml' or file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_dict, f, indent=2)


class ConfigValidator:
    """Validates optimization configurations"""
    
    @staticmethod
    def validate_config(config: OptimizationConfig) -> List[str]:
        """Validate the configuration and return list of validation errors"""
        errors = []
        
        # Validate block size is positive and reasonable
        if config.block_size <= 0 or config.block_size > 1024:
            errors.append(f"block_size must be between 1 and 1024, got {config.block_size}")
        
        # Validate token merging threshold is between 0 and 1
        if config.token_merging_threshold < 0 or config.token_merging_threshold > 1:
            errors.append(f"token_merging_threshold must be between 0 and 1, got {config.token_merging_threshold}")
        
        # Validate compression level
        valid_compression_levels = ["low", "medium", "high"]
        if config.compression_level not in valid_compression_levels:
            errors.append(f"compression_level must be one of {valid_compression_levels}, got {config.compression_level}")
        
        # Validate number of activation functions
        if config.num_activation_functions <= 0 or config.num_activation_functions > 10:
            errors.append(f"num_activation_functions must be between 1 and 10, got {config.num_activation_functions}")
        
        # Validate max batch size
        if config.max_batch_size <= 0:
            errors.append(f"max_batch_size must be positive, got {config.max_batch_size}")
        
        # Validate recycling frequency
        if config.recycling_frequency <= 0:
            errors.append(f"recycling_frequency must be positive, got {config.recycling_frequency}")
        
        # Validate max packed length
        if config.max_packed_length <= 0:
            errors.append(f"max_packed_length must be positive, got {config.max_packed_length}")
        
        # Validate grad accumulation steps
        if config.grad_accumulation_steps <= 0:
            errors.append(f"grad_accumulation_steps must be positive, got {config.grad_accumulation_steps}")
        
        # Validate KV cache strategy
        valid_kv_strategies = ["standard", "low_rank", "sliding_window", "hybrid"]
        if config.kv_cache_strategy not in valid_kv_strategies:
            errors.append(f"kv_cache_strategy must be one of {valid_kv_strategies}, got {config.kv_cache_strategy}")
        
        # Validate pipeline stages
        if config.pipeline_stages <= 0 or config.pipeline_stages > 8:
            errors.append(f"pipeline_stages must be between 1 and 8, got {config.pipeline_stages}")
        
        # Validate model capacity preservation
        if not config.preserve_32_layers:
            errors.append("preserve_32_layers should be True to maintain model capacity")
        if not config.preserve_32_attention_heads:
            errors.append("preserve_32_attention_heads should be True to maintain model capacity")
        
        return errors
    
    @staticmethod
    def validate_compatibility(config: OptimizationConfig) -> List[str]:
        """Validate compatibility between optimizations and return warnings"""
        warnings = []
        
        # Check if hardware-specific kernels are enabled but no compatible hardware is detected
        if config.hardware_specific_kernels and not ConfigValidator._has_tensor_cores():
            warnings.append("hardware_specific_kernels enabled but tensor cores not detected")
        
        # Check if distributed pipeline parallelism is enabled but only one stage is specified
        if config.distributed_pipeline_parallelism and config.pipeline_stages <= 1:
            warnings.append("distributed_pipeline_parallelism enabled but pipeline_stages <= 1")
        
        return warnings
    
    @staticmethod
    def _has_tensor_cores() -> bool:
        """Check if tensor cores are available"""
        try:
            import torch
            if torch.cuda.is_available():
                # Check if GPU has compute capability >= 7.0 (for tensor cores)
                device = torch.cuda.current_device()
                capability = torch.cuda.get_device_capability(device)
                return capability[0] >= 7
            return False
        except:
            return False


class ConfigManager:
    """Centralized configuration manager for all optimization techniques"""
    
    def __init__(self):
        self.configs: Dict[str, OptimizationConfig] = {}
        self.validator = ConfigValidator()
        self.logger = logging.getLogger(__name__)
        
        # Register default configurations
        self._register_default_configs()
    
    def _register_default_configs(self):
        """Register default configurations for different optimization levels"""
        # Minimal optimization config
        minimal_config = OptimizationConfig(
            block_sparse_attention=False,
            cross_modal_token_merging=False,
            hierarchical_memory_compression=False,
            learned_activation_routing=False,
            adaptive_batch_processing=False,
            cross_layer_parameter_recycling=False,
            adaptive_sequence_packing=False,
            memory_efficient_grad_accumulation=False,
            kv_cache_multiple_strategies=False,
            faster_rotary_embeddings=False,
            distributed_pipeline_parallelism=False,
            hardware_specific_kernels=False
        )
        self.register_config("minimal", minimal_config)
        
        # Moderate optimization config
        moderate_config = OptimizationConfig(
            block_sparse_attention=True,
            cross_modal_token_merging=True,
            hierarchical_memory_compression=True,
            learned_activation_routing=True,
            adaptive_batch_processing=True,
            cross_layer_parameter_recycling=False,
            adaptive_sequence_packing=True,
            memory_efficient_grad_accumulation=True,
            kv_cache_multiple_strategies=True,
            faster_rotary_embeddings=True,
            distributed_pipeline_parallelism=False,
            hardware_specific_kernels=True
        )
        self.register_config("moderate", moderate_config)
        
        # Aggressive optimization config
        aggressive_config = OptimizationConfig(
            block_sparse_attention=True,
            cross_modal_token_merging=True,
            hierarchical_memory_compression=True,
            learned_activation_routing=True,
            adaptive_batch_processing=True,
            cross_layer_parameter_recycling=True,
            adaptive_sequence_packing=True,
            memory_efficient_grad_accumulation=True,
            kv_cache_multiple_strategies=True,
            faster_rotary_embeddings=True,
            distributed_pipeline_parallelism=True,
            hardware_specific_kernels=True
        )
        self.register_config("aggressive", aggressive_config)
    
    def register_config(self, name: str, config: OptimizationConfig) -> bool:
        """Register a named configuration"""
        try:
            # Validate the configuration before registering
            errors = self.validator.validate_config(config)
            if errors:
                self.logger.error(f"Configuration '{name}' validation failed: {errors}")
                return False
            
            self.configs[name] = copy.deepcopy(config)
            self.logger.info(f"Registered configuration '{name}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register configuration '{name}': {e}")
            return False
    
    def get_config(self, name: str) -> Optional[OptimizationConfig]:
        """Get a named configuration"""
        if name not in self.configs:
            self.logger.warning(f"Configuration '{name}' not found")
            return None
        return copy.deepcopy(self.configs[name])
    
    def get_config_names(self) -> List[str]:
        """Get list of all registered configuration names"""
        return list(self.configs.keys())
    
    def update_config(self, name: str, new_config: OptimizationConfig) -> bool:
        """Update an existing configuration"""
        if name not in self.configs:
            self.logger.warning(f"Configuration '{name}' not found for update")
            return False
        
        try:
            # Validate the new configuration
            errors = self.validator.validate_config(new_config)
            if errors:
                self.logger.error(f"New configuration for '{name}' validation failed: {errors}")
                return False
            
            self.configs[name] = copy.deepcopy(new_config)
            self.logger.info(f"Updated configuration '{name}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update configuration '{name}': {e}")
            return False
    
    def delete_config(self, name: str) -> bool:
        """Delete a configuration"""
        if name not in self.configs:
            self.logger.warning(f"Configuration '{name}' not found for deletion")
            return False
        
        del self.configs[name]
        self.logger.info(f"Deleted configuration '{name}'")
        return True
    
    def validate_config_by_name(self, name: str) -> List[str]:
        """Validate a named configuration"""
        config = self.get_config(name)
        if config is None:
            return [f"Configuration '{name}' not found"]
        
        return self.validator.validate_config(config)
    
    def create_config_from_level(self, level: OptimizationLevel) -> OptimizationConfig:
        """Create a configuration based on optimization level"""
        if level == OptimizationLevel.MINIMAL:
            return self.get_config("minimal")
        elif level == OptimizationLevel.MODERATE:
            return self.get_config("moderate")
        elif level == OptimizationLevel.AGGRESSIVE:
            return self.get_config("aggressive")
        elif level == OptimizationLevel.MAXIMUM:
            # Maximum optimization config - all optimizations enabled
            max_config = OptimizationConfig()
            return max_config
        else:
            raise ValueError(f"Unknown optimization level: {level}")
    
    def merge_configs(self, base_config: OptimizationConfig, override_config: OptimizationConfig) -> OptimizationConfig:
        """Merge two configurations, with override_config taking precedence"""
        merged_config = copy.deepcopy(base_config)
        
        # Override values from the second config
        for field_name, field_value in asdict(override_config).items():
            if hasattr(merged_config, field_name):
                setattr(merged_config, field_name, field_value)
        
        return merged_config
    
    def save_all_configs(self, directory: Union[str, Path]):
        """Save all configurations to a directory"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for name, config in self.configs.items():
            file_path = directory / f"{name}_config.json"
            config.save_to_file(file_path)
            self.logger.info(f"Saved configuration '{name}' to {file_path}")
    
    def load_configs_from_directory(self, directory: Union[str, Path]) -> bool:
        """Load all configurations from a directory"""
        directory = Path(directory)
        if not directory.exists():
            self.logger.error(f"Directory {directory} does not exist")
            return False
        
        loaded_count = 0
        for file_path in directory.glob("*.json"):
            try:
                # Extract config name from filename
                config_name = file_path.stem.replace("_config", "")
                config = OptimizationConfig.from_file(file_path)
                
                # Register the loaded config
                if self.register_config(config_name, config):
                    loaded_count += 1
                    self.logger.info(f"Loaded configuration '{config_name}' from {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to load configuration from {file_path}: {e}")
        
        self.logger.info(f"Loaded {loaded_count} configurations from {directory}")
        return loaded_count > 0
    
    def get_config_diff(self, config1_name: str, config2_name: str) -> Dict[str, Any]:
        """Get differences between two configurations"""
        config1 = self.get_config(config1_name)
        config2 = self.get_config(config2_name)
        
        if not config1 or not config2:
            return {}
        
        diff = {}
        config1_dict = asdict(config1)
        config2_dict = asdict(config2)
        
        for key in config1_dict.keys():
            if config1_dict[key] != config2_dict[key]:
                diff[key] = {
                    'config1': config1_dict[key],
                    'config2': config2_dict[key]
                }
        
        return diff


# Global configuration manager instance
config_manager = ConfigManager()


def get_global_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    return config_manager


def create_default_config(level: OptimizationLevel = OptimizationLevel.MODERATE) -> OptimizationConfig:
    """Create a default configuration with the specified optimization level"""
    return config_manager.create_config_from_level(level)


def validate_config(config: OptimizationConfig) -> bool:
    """Validate a configuration and return True if valid"""
    validator = ConfigValidator()
    errors = validator.validate_config(config)
    return len(errors) == 0


def get_default_config() -> OptimizationConfig:
    """Get the default configuration (moderate level)"""
    return config_manager.get_config("moderate") or create_default_config()
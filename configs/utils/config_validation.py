"""
Configuration Validation Tools for Qwen3-VL Model

This module provides tools to validate configurations and catch misconfigurations early
in the Qwen3-VL model development and deployment process.
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, fields
from pathlib import Path
import torch
import numpy as np
from functools import wraps
from contextlib import contextmanager


@dataclass
class ConfigSchema:
    """Schema definition for configuration validation"""
    name: str
    type: type
    required: bool = True
    default: Any = None
    description: str = ""
    validator: Optional[callable] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    nested_schema: Optional['ConfigSchema'] = None


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors"""
    pass


class ConfigValidator:
    """Main configuration validator class"""

    def __init__(self):
        self.schemas = {}
        self.validation_results = []

    def register_schema(self, name: str, schema: ConfigSchema):
        """Register a configuration schema"""
        self.schemas[name] = schema

    def validate_config(self, config: Dict[str, Any], schema_name: str = None, schema: ConfigSchema = None) -> Dict[str, Any]:
        """Validate a configuration against a schema"""
        if schema is None:
            if schema_name not in self.schemas:
                raise ConfigValidationError(f"Schema '{schema_name}' not registered")
            schema = self.schemas[schema_name]

        errors = []
        warnings = []

        # If schema is a nested structure, validate recursively
        if hasattr(schema, 'nested_schema') and schema.nested_schema:
            for key, value in config.items():
                if isinstance(value, dict):
                    nested_result = self.validate_config(value, schema=schema.nested_schema)
                    if nested_result['errors']:
                        errors.extend([f"{key}.{err}" for err in nested_result['errors']])
                    if nested_result['warnings']:
                        warnings.extend([f"{key}.{warn}" for warn in nested_result['warnings']])

        # Validate each field according to schema
        for field_name, field_value in config.items():
            field_schema = self._get_field_schema(schema, field_name)
            if field_schema:
                field_errors, field_warnings = self._validate_field(field_name, field_value, field_schema)
                errors.extend(field_errors)
                warnings.extend(field_warnings)
            else:
                # Field not in schema - warn about it
                warnings.append(f"Unknown field '{field_name}'")

        # Check for required fields
        if hasattr(schema, 'fields'):
            for field in fields(schema):
                if field.default is None and field.name not in config:
                    errors.append(f"Required field '{field.name}' is missing")

        result = {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config': config
        }

        self.validation_results.append(result)

        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")

        return result

    def _get_field_schema(self, schema: ConfigSchema, field_name: str) -> Optional[ConfigSchema]:
        """Get schema for a specific field"""
        # This is a simplified implementation - in a real system, you'd have a more complex schema structure
        if hasattr(schema, 'fields'):
            for field in fields(schema):
                if field.name == field_name:
                    return field
        return None

    def _validate_field(self, field_name: str, value: Any, schema: ConfigSchema) -> tuple:
        """Validate a single field according to its schema"""
        errors = []
        warnings = []

        # Type validation
        if schema.type and not isinstance(value, schema.type):
            errors.append(f"Field '{field_name}' has type {type(value).__name__}, expected {schema.type.__name__}")
            return errors, warnings

        # Required field validation
        if schema.required and value is None:
            errors.append(f"Required field '{field_name}' cannot be None")

        # Value range validation
        if schema.min_value is not None and isinstance(value, (int, float)):
            if value < schema.min_value:
                errors.append(f"Field '{field_name}' value {value} is less than minimum {schema.min_value}")

        if schema.max_value is not None and isinstance(value, (int, float)):
            if value > schema.max_value:
                errors.append(f"Field '{field_name}' value {value} is greater than maximum {schema.max_value}")

        # Allowed values validation
        if schema.allowed_values is not None and value not in schema.allowed_values:
            errors.append(f"Field '{field_name}' value {value} not in allowed values: {schema.allowed_values}")

        # Custom validator
        if schema.validator and callable(schema.validator):
            try:
                if not schema.validator(value):
                    errors.append(f"Field '{field_name}' failed custom validation")
            except Exception as e:
                errors.append(f"Field '{field_name}' custom validator raised exception: {str(e)}")

        return errors, warnings

    def validate_file(self, file_path: str, schema_name: str = None) -> Dict[str, Any]:
        """Validate a configuration file"""
        if not os.path.exists(file_path):
            raise ConfigValidationError(f"Configuration file '{file_path}' does not exist")

        # Determine file format and load
        ext = Path(file_path).suffix.lower()
        if ext in ['.json']:
            with open(file_path, 'r') as f:
                config = json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ConfigValidationError(f"Unsupported configuration file format: {ext}")

        return self.validate_config(config, schema_name)

    def validate_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Qwen3-VL specific model configuration"""
        # Define Qwen3-VL specific schema
        qwen3_vl_schema = {
            'model': {
                'type': str,
                'required': True,
                'allowed_values': ['qwen3_vl', 'qwen3_vl_2b', 'qwen3_vl_instruct']
            },
            'transformer_layers': {
                'type': int,
                'required': True,
                'min_value': 1,
                'max_value': 100
            },
            'attention_heads': {
                'type': int,
                'required': True,
                'min_value': 1,
                'max_value': 128
            },
            'hidden_size': {
                'type': int,
                'required': True,
                'min_value': 64,
                'max_value': 8192
            },
            'sequence_length': {
                'type': int,
                'required': True,
                'min_value': 1,
                'max_value': 32768
            },
            'batch_size': {
                'type': int,
                'required': True,
                'min_value': 1,
                'max_value': 512
            },
            'learning_rate': {
                'type': float,
                'required': False,
                'min_value': 1e-8,
                'max_value': 1.0
            },
            'optimizer': {
                'type': str,
                'required': False,
                'allowed_values': ['adam', 'adamw', 'sgd', 'rmsprop']
            },
            'device': {
                'type': str,
                'required': False,
                'allowed_values': ['cpu', 'cuda', 'auto']
            },
            'precision': {
                'type': str,
                'required': False,
                'allowed_values': ['fp32', 'fp16', 'bf16']
            }
        }

        errors = []
        warnings = []

        # Validate each field according to the schema
        for field_name, field_schema in qwen3_vl_schema.items():
            if field_name in config:
                value = config[field_name]

                # Type validation
                expected_type = field_schema.get('type')
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Field '{field_name}' has type {type(value).__name__}, expected {expected_type.__name__}")
                    continue

                # Value range validation
                min_val = field_schema.get('min_value')
                max_val = field_schema.get('max_value')
                if isinstance(value, (int, float)):
                    if min_val is not None and value < min_val:
                        errors.append(f"Field '{field_name}' value {value} is less than minimum {min_val}")
                    if max_val is not None and value > max_val:
                        errors.append(f"Field '{field_name}' value {value} is greater than maximum {max_val}")

                # Allowed values validation
                allowed_vals = field_schema.get('allowed_values')
                if allowed_vals is not None and value not in allowed_vals:
                    errors.append(f"Field '{field_name}' value {value} not in allowed values: {allowed_vals}")
            elif field_schema.get('required', False):
                errors.append(f"Required field '{field_name}' is missing")

        # Additional semantic validations
        if 'transformer_layers' in config and 'attention_heads' in config:
            # Example: Check if the combination makes sense
            layers = config['transformer_layers']
            heads = config['attention_heads']
            if layers * heads > 1024:  # Arbitrary check
                warnings.append(f"High layer*head count ({layers} * {heads} = {layers*heads}) may cause memory issues")

        if 'device' in config and config['device'] == 'cuda' and not torch.cuda.is_available():
            errors.append("CUDA device specified but CUDA is not available")

        if 'precision' in config and config['precision'] in ['fp16', 'bf16'] and config.get('device') == 'cpu':
            warnings.append(f"{config['precision']} precision may not be supported on CPU")

        result = {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config': config
        }

        self.validation_results.append(result)

        if errors:
            raise ConfigValidationError(f"Qwen3-VL configuration validation failed: {'; '.join(errors)}")

        return result

    def validate_memory_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate memory-related configurations"""
        errors = []
        warnings = []

        # Memory-related fields to validate
        memory_fields = {
            'max_memory_mb': {'type': int, 'min': 100, 'max': 1000000},
            'memory_fraction': {'type': float, 'min': 0.01, 'max': 1.0},
            'kv_cache_size': {'type': int, 'min': 1, 'max': 1000000000},
            'batch_size': {'type': int, 'min': 1, 'max': 1024},
            'sequence_length': {'type': int, 'min': 1, 'max': 32768}
        }

        for field_name, constraints in memory_fields.items():
            if field_name in config:
                value = config[field_name]

                # Type validation
                expected_type = constraints['type']
                if not isinstance(value, expected_type):
                    errors.append(f"Field '{field_name}' has type {type(value).__name__}, expected {expected_type.__name__}")
                    continue

                # Range validation
                min_val = constraints.get('min')
                max_val = constraints.get('max')
                if min_val is not None and value < min_val:
                    errors.append(f"Field '{field_name}' value {value} is less than minimum {min_val}")
                if max_val is not None and value > max_val:
                    errors.append(f"Field '{field_name}' value {value} is greater than maximum {max_val}")

        # Additional memory-specific validations
        if 'batch_size' in config and 'sequence_length' in config:
            batch_size = config['batch_size']
            seq_len = config['sequence_length']
            estimated_memory = batch_size * seq_len * 4 * 2  # Rough estimate: 4 bytes per element, 2 for key+value
            if estimated_memory > 1e9:  # More than 1GB
                warnings.append(f"Estimated memory usage ({estimated_memory/1e9:.2f}GB) may be high for batch_size={batch_size}, seq_len={seq_len}")

        result = {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config': config
        }

        self.validation_results.append(result)

        if errors:
            raise ConfigValidationError(f"Memory configuration validation failed: {'; '.join(errors)}")

        return result

    def validate_hardware_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hardware-specific configurations"""
        errors = []
        warnings = []

        # Hardware-specific validations
        if 'cpu_threads' in config:
            cpu_threads = config['cpu_threads']
            if not isinstance(cpu_threads, int) or cpu_threads <= 0:
                errors.append(f"cpu_threads must be a positive integer, got {cpu_threads}")
            elif cpu_threads > os.cpu_count():
                warnings.append(f"cpu_threads ({cpu_threads}) exceeds system CPU count ({os.cpu_count()})")

        if 'gpu_device_id' in config:
            gpu_id = config['gpu_device_id']
            if not isinstance(gpu_id, int) or gpu_id < 0:
                errors.append(f"gpu_device_id must be a non-negative integer, got {gpu_id}")
            elif torch.cuda.is_available() and gpu_id >= torch.cuda.device_count():
                errors.append(f"gpu_device_id ({gpu_id}) exceeds available GPU count ({torch.cuda.device_count()})")

        if 'use_tensor_cores' in config and config['use_tensor_cores']:
            if not torch.cuda.is_available():
                errors.append("use_tensor_cores enabled but CUDA is not available")
            elif torch.cuda.get_device_capability(0)[0] < 7:
                warnings.append("Tensor cores may not be available on this GPU (compute capability < 7.0)")

        result = {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config': config
        }

        self.validation_results.append(result)

        if errors:
            raise ConfigValidationError(f"Hardware configuration validation failed: {'; '.join(errors)}")

        return result

    def print_validation_summary(self):
        """Print a summary of validation results"""
        print("=== Configuration Validation Summary ===")
        print(f"Total validations performed: {len(self.validation_results)}")

        valid_count = sum(1 for result in self.validation_results if result['valid'])
        print(f"Valid configurations: {valid_count}/{len(self.validation_results)}")

        for i, result in enumerate(self.validation_results, 1):
            status = "OK" if result['valid'] else "FAIL"
            print(f"\n{i}. {status} Valid: {result['valid']}")
            if result['errors']:
                print(f"   Errors: {len(result['errors'])}")
                for error in result['errors']:
                    print(f"     - {error}")
            if result['warnings']:
                print(f"   Warnings: {len(result['warnings'])}")
                for warning in result['warnings']:
                    print(f"     - {warning}")


class ConfigManager:
    """Configuration management with validation"""

    def __init__(self):
        self.validator = ConfigValidator()
        self.configs = {}

    def load_config(self, config_path: str, validate: bool = True, schema_name: str = None):
        """Load and optionally validate a configuration file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load configuration based on file extension
        ext = Path(config_path).suffix.lower()
        if ext in ['.json']:
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")

        # Store config with path as key
        config_name = Path(config_path).stem
        self.configs[config_name] = config

        # Validate if requested
        if validate:
            if schema_name:
                return self.validator.validate_config(config, schema_name)
            else:
                # Try to determine validation type based on content
                if 'model' in config and 'transformer_layers' in config:
                    return self.validator.validate_model_config(config)
                elif 'max_memory_mb' in config or 'memory_fraction' in config:
                    return self.validator.validate_memory_config(config)
                elif 'cpu_threads' in config or 'gpu_device_id' in config:
                    return self.validator.validate_hardware_config(config)
                else:
                    # Use general validation
                    return self.validator.validate_config(config)

    def save_config(self, config_name: str, config_path: str):
        """Save a configuration to file"""
        if config_name not in self.configs:
            raise ValueError(f"Configuration '{config_name}' not found in manager")

        config = self.configs[config_name]

        # Determine file format and save
        ext = Path(config_path).suffix.lower()
        if ext in ['.json']:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif ext in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")

    def get_config(self, name: str) -> Dict[str, Any]:
        """Get a loaded configuration by name"""
        if name not in self.configs:
            raise ValueError(f"Configuration '{name}' not found")
        return self.configs[name]

    def set_config(self, name: str, config: Dict[str, Any]):
        """Set a configuration by name"""
        self.configs[name] = config

    def validate_config(self, config_name: str, schema_name: str = None):
        """Validate a stored configuration"""
        if config_name not in self.configs:
            raise ValueError(f"Configuration '{config_name}' not found")

        config = self.configs[config_name]
        return self.validator.validate_config(config, schema_name)

    def create_default_config(self, config_type: str = "model") -> Dict[str, Any]:
        """Create a default configuration based on type"""
        if config_type == "model":
            return {
                "model": "qwen3_vl_2b",
                "transformer_layers": 32,
                "attention_heads": 32,
                "hidden_size": 2560,
                "sequence_length": 32768,
                "batch_size": 1,
                "learning_rate": 5e-5,
                "optimizer": "adamw",
                "device": "auto",
                "precision": "fp16"
            }
        elif config_type == "memory":
            return {
                "max_memory_mb": 6144,
                "memory_fraction": 0.8,
                "kv_cache_size": 500000000,
                "batch_size": 1,
                "sequence_length": 2048
            }
        elif config_type == "hardware":
            return {
                "cpu_threads": os.cpu_count() or 4,
                "gpu_device_id": 0,
                "use_tensor_cores": True,
                "enable_mixed_precision": True
            }
        else:
            return {}


class ConfigValidatorDecorator:
    """Decorator for validating function arguments that contain configurations"""

    def __init__(self, schema_name: str = None, validator: ConfigValidator = None):
        self.schema_name = schema_name
        self.validator = validator or global_config_validator

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to find configuration in arguments
            config = None
            config_arg_idx = None

            # Check if first argument is a config dict
            if args and isinstance(args[0], dict):
                config = args[0]
                config_arg_idx = 0
            # Check keyword arguments for config
            elif 'config' in kwargs:
                config = kwargs['config']
            elif 'model_config' in kwargs:
                config = kwargs['model_config']
            elif 'settings' in kwargs:
                config = kwargs['settings']

            # Validate configuration if found
            if config and isinstance(config, dict):
                try:
                    self.validator.validate_config(config, self.schema_name)
                except ConfigValidationError as e:
                    raise ConfigValidationError(f"Configuration validation failed for function '{func.__name__}': {str(e)}")

            return func(*args, **kwargs)
        return wrapper


# Global validator instance
global_config_validator = ConfigValidator()
config_manager = ConfigManager()


def validate_config_arg(schema_name: str = None):
    """Decorator to validate configuration arguments to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find config in args or kwargs
            config = None
            if args and isinstance(args[0], dict):
                config = args[0]
            elif 'config' in kwargs:
                config = kwargs['config']
            elif 'model_config' in kwargs:
                config = kwargs['model_config']

            if config:
                try:
                    global_config_validator.validate_config(config, schema_name)
                except ConfigValidationError as e:
                    raise ConfigValidationError(f"Config validation failed in {func.__name__}: {str(e)}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


# Example configuration schemas for Qwen3-VL
def setup_qwen3_vl_schemas():
    """Setup standard schemas for Qwen3-VL configurations"""

    # Model configuration schema
    model_schema = ConfigSchema(
        name="model_config",
        type=Dict,
        required=True,
        description="Qwen3-VL model configuration"
    )

    # Memory configuration schema
    memory_schema = ConfigSchema(
        name="memory_config",
        type=Dict,
        required=True,
        description="Memory management configuration"
    )

    # Hardware configuration schema
    hardware_schema = ConfigSchema(
        name="hardware_config",
        type=Dict,
        required=True,
        description="Hardware-specific configuration"
    )

    # Register schemas
    global_config_validator.register_schema("model_config", model_schema)
    global_config_validator.register_schema("memory_config", memory_schema)
    global_config_validator.register_schema("hardware_config", hardware_schema)


def example_validation():
    """Example of configuration validation usage"""
    print("=== Configuration Validation Example ===")

    # Setup schemas
    setup_qwen3_vl_schemas()

    # Example model configuration
    model_config = {
        "model": "qwen3_vl_2b",
        "transformer_layers": 32,
        "attention_heads": 32,
        "hidden_size": 2560,
        "sequence_length": 32768,
        "batch_size": 1,
        "learning_rate": 5e-5,
        "optimizer": "adamw",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "precision": "fp16"
    }

    # Validate model configuration
    try:
        result = global_config_validator.validate_model_config(model_config)
        print(f"Model config validation: {'PASSED' if result['valid'] else 'FAILED'}")
        if result['warnings']:
            print(f"Warnings: {result['warnings']}")
    except ConfigValidationError as e:
        print(f"Model config validation failed: {e}")

    # Example memory configuration
    memory_config = {
        "max_memory_mb": 6144,
        "memory_fraction": 0.8,
        "kv_cache_size": 500000000,
        "batch_size": 1,
        "sequence_length": 2048
    }

    # Validate memory configuration
    try:
        result = global_config_validator.validate_memory_config(memory_config)
        print(f"Memory config validation: {'PASSED' if result['valid'] else 'FAILED'}")
        if result['warnings']:
            print(f"Warnings: {result['warnings']}")
    except ConfigValidationError as e:
        print(f"Memory config validation failed: {e}")

    # Example hardware configuration
    hardware_config = {
        "cpu_threads": os.cpu_count(),
        "gpu_device_id": 0 if torch.cuda.is_available() else None,
        "use_tensor_cores": torch.cuda.is_available(),
        "enable_mixed_precision": True
    }

    # Validate hardware configuration (only if CUDA available)
    if torch.cuda.is_available():
        try:
            result = global_config_validator.validate_hardware_config(hardware_config)
            print(f"Hardware config validation: {'PASSED' if result['valid'] else 'FAILED'}")
            if result['warnings']:
                print(f"Warnings: {result['warnings']}")
        except ConfigValidationError as e:
            print(f"Hardware config validation failed: {e}")

    # Print validation summary
    global_config_validator.print_validation_summary()


def example_config_manager():
    """Example of using the configuration manager"""
    print("\n=== Configuration Manager Example ===")

    # Create a default model config
    default_config = config_manager.create_default_config("model")
    config_manager.set_config("default_model", default_config)

    # Save to file
    config_manager.save_config("default_model", "temp_default_config.json")

    # Load from file
    loaded_result = config_manager.load_config("temp_default_config.json", validate=True)
    print(f"Loaded and validated config: {loaded_result['valid']}")

    # Get the config
    retrieved_config = config_manager.get_config("default_model")
    print(f"Retrieved config has {len(retrieved_config)} keys")

    # Clean up temp file
    if os.path.exists("temp_default_config.json"):
        os.remove("temp_default_config.json")


if __name__ == "__main__":
    example_validation()
    example_config_manager()
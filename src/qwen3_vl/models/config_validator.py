"""
Configuration validation for different model types.

This module provides configuration validation for different model types.
"""

from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Represents a configuration validation error."""
    field: str
    message: str
    severity: str  # "error", "warning", "info"


class ConfigValidator:
    """
    Validator for model configurations that ensures they are appropriate for each model type.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._validators: Dict[str, callable] = {}
    
    def register_validator(self, model_type: str, validator: callable) -> bool:
        """
        Register a validator for a specific model type.
        
        Args:
            model_type: Type of model ("language", "vision", "multimodal", etc.)
            validator: Validation function that takes config dict and returns list of ValidationError
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            self._validators[model_type] = validator
            self._logger.info(f"Validator registered for model type: {model_type}")
            return True
        except Exception as e:
            self._logger.error(f"Error registering validator for {model_type}: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any], model_type: str = "language") -> List[ValidationError]:
        """
        Validate a configuration for a specific model type.
        
        Args:
            config: Configuration dictionary to validate
            model_type: Type of model to validate for
            
        Returns:
            List of ValidationError instances
        """
        errors = []
        
        # Run the type-specific validator if it exists
        if model_type in self._validators:
            try:
                type_errors = self._validators[model_type](config)
                errors.extend(type_errors)
            except Exception as e:
                self._logger.error(f"Error running validator for {model_type}: {e}")
                errors.append(ValidationError(
                    field="general",
                    message=f"Validator error for {model_type}: {str(e)}",
                    severity="error"
                ))
        
        # Run general validations
        general_errors = self._run_general_validations(config)
        errors.extend(general_errors)
        
        return errors
    
    def _run_general_validations(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Run general validations that apply to all model types."""
        errors = []
        
        # Check for required fields
        required_fields = ["model_name", "model_type", "torch_dtype"]
        for field in required_fields:
            if field not in config:
                errors.append(ValidationError(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    severity="error"
                ))
        
        # Validate torch_dtype
        if "torch_dtype" in config:
            valid_dtypes = ["float16", "float32", "bfloat16", "float64", "int8", "int16", "int32", "int64"]
            dtype = config["torch_dtype"]
            if dtype not in valid_dtypes:
                errors.append(ValidationError(
                    field="torch_dtype",
                    message=f"Invalid torch_dtype: {dtype}. Valid options: {valid_dtypes}",
                    severity="error"
                ))
        
        # Validate memory requirements
        if "memory_requirements" in config:
            mem_reqs = config["memory_requirements"]
            if not isinstance(mem_reqs, dict):
                errors.append(ValidationError(
                    field="memory_requirements",
                    message="memory_requirements must be a dictionary",
                    severity="error"
                ))
            else:
                required_mem_fields = ["min_memory_gb", "recommended_memory_gb", "max_memory_gb"]
                for mem_field in required_mem_fields:
                    if mem_field not in mem_reqs:
                        errors.append(ValidationError(
                            field=f"memory_requirements.{mem_field}",
                            message=f"Required memory field '{mem_field}' is missing",
                            severity="error"
                        ))
                    elif not isinstance(mem_reqs[mem_field], (int, float)) or mem_reqs[mem_field] <= 0:
                        errors.append(ValidationError(
                            field=f"memory_requirements.{mem_field}",
                            message=f"Memory field '{mem_field}' must be a positive number",
                            severity="error"
                        ))
                
                # Check consistency of memory requirements
                if (isinstance(mem_reqs.get("min_memory_gb"), (int, float)) and
                    isinstance(mem_reqs.get("recommended_memory_gb"), (int, float)) and
                    mem_reqs["min_memory_gb"] > mem_reqs["recommended_memory_gb"]):
                    errors.append(ValidationError(
                        field="memory_requirements",
                        message="min_memory_gb should not exceed recommended_memory_gb",
                        severity="warning"
                    ))
        
        # Validate performance profile
        if "performance_profile" in config:
            valid_profiles = ["balanced", "memory_efficient", "performance"]
            profile = config["performance_profile"]
            if profile not in valid_profiles:
                errors.append(ValidationError(
                    field="performance_profile",
                    message=f"Invalid performance_profile: {profile}. Valid options: {valid_profiles}",
                    severity="error"
                ))
        
        return errors
    
    def validate_qwen_config(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate configuration for Qwen models."""
        errors = []
        
        # Qwen-specific validations
        qwen_required_fields = [
            "vocab_size", "hidden_size", "num_hidden_layers", 
            "num_attention_heads", "max_position_embeddings"
        ]
        
        for field in qwen_required_fields:
            if field not in config:
                errors.append(ValidationError(
                    field=field,
                    message=f"Required Qwen field '{field}' is missing",
                    severity="error"
                ))
        
        # Validate specific Qwen parameters
        if "vocab_size" in config:
            vocab_size = config["vocab_size"]
            if not isinstance(vocab_size, int) or vocab_size <= 0:
                errors.append(ValidationError(
                    field="vocab_size",
                    message="vocab_size must be a positive integer",
                    severity="error"
                ))
        
        if "hidden_size" in config:
            hidden_size = config["hidden_size"]
            if not isinstance(hidden_size, int) or hidden_size <= 0:
                errors.append(ValidationError(
                    field="hidden_size",
                    message="hidden_size must be a positive integer",
                    severity="error"
                ))
            # Hidden size should typically be divisible by attention heads
            if ("num_attention_heads" in config and 
                isinstance(config["num_attention_heads"], int) and
                config["num_attention_heads"] > 0):
                if hidden_size % config["num_attention_heads"] != 0:
                    errors.append(ValidationError(
                        field="hidden_size",
                        message="hidden_size should be divisible by num_attention_heads",
                        severity="warning"
                    ))
        
        if "num_hidden_layers" in config:
            layers = config["num_hidden_layers"]
            if not isinstance(layers, int) or layers <= 0:
                errors.append(ValidationError(
                    field="num_hidden_layers",
                    message="num_hidden_layers must be a positive integer",
                    severity="error"
                ))
        
        if "max_position_embeddings" in config:
            max_pos = config["max_position_embeddings"]
            if not isinstance(max_pos, int) or max_pos <= 0:
                errors.append(ValidationError(
                    field="max_position_embeddings",
                    message="max_position_embeddings must be a positive integer",
                    severity="error"
                ))
        
        # Validate attention-related parameters
        if "use_flash_attention_2" in config:
            if not isinstance(config["use_flash_attention_2"], bool):
                errors.append(ValidationError(
                    field="use_flash_attention_2",
                    message="use_flash_attention_2 must be a boolean",
                    severity="error"
                ))
        
        if "use_gradient_checkpointing" in config:
            if not isinstance(config["use_gradient_checkpointing"], bool):
                errors.append(ValidationError(
                    field="use_gradient_checkpointing",
                    message="use_gradient_checkpointing must be a boolean",
                    severity="error"
                ))
        
        if "rope_theta" in config:
            rope_theta = config["rope_theta"]
            if not isinstance(rope_theta, (int, float)) or rope_theta <= 0:
                errors.append(ValidationError(
                    field="rope_theta",
                    message="rope_theta must be a positive number",
                    severity="error"
                ))
        
        return errors
    
    def validate_multimodal_config(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate configuration for multimodal models."""
        errors = []
        
        # Multimodal-specific validations
        if "vision_hidden_size" not in config:
            errors.append(ValidationError(
                field="vision_hidden_size",
                message="vision_hidden_size is required for multimodal models",
                severity="error"
            ))
        
        if "num_query_tokens" not in config:
            errors.append(ValidationError(
                field="num_query_tokens",
                message="num_query_tokens is required for multimodal models",
                severity="error"
            ))
        
        if "vision_projection_dim" not in config:
            errors.append(ValidationError(
                field="vision_projection_dim",
                message="vision_projection_dim is required for multimodal models",
                severity="error"
            ))
        
        # Validate vision parameters
        vision_fields = ["vision_hidden_size", "vision_num_hidden_layers", "vision_num_attention_heads"]
        for field in vision_fields:
            if field in config:
                value = config[field]
                if not isinstance(value, int) or value <= 0:
                    errors.append(ValidationError(
                        field=field,
                        message=f"{field} must be a positive integer",
                        severity="error"
                    ))
        
        return errors
    
    def validate_language_config(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate configuration for language models."""
        errors = []
        
        # Language-specific validations
        if "pad_token_id" not in config:
            errors.append(ValidationError(
                field="pad_token_id",
                message="pad_token_id is required for language models",
                severity="warning"  # Warning because it might have a default
            ))
        
        if "eos_token_id" not in config:
            errors.append(ValidationError(
                field="eos_token_id",
                message="eos_token_id is required for language models",
                severity="warning"  # Warning because it might have a default
            ))
        
        return errors


# Global config validator instance
config_validator = ConfigValidator()


def get_config_validator() -> ConfigValidator:
    """
    Get the global config validator instance.
    
    Returns:
        ConfigValidator instance
    """
    return config_validator


# Register default validators
def _register_default_validators():
    """Register default validators for known model types."""
    
    # Register validator for Qwen models
    config_validator.register_validator("language", config_validator.validate_qwen_config)
    config_validator.register_validator("multimodal", config_validator.validate_multimodal_config)
    config_validator.register_validator("vision", config_validator.validate_qwen_config)  # Use same base validation
    config_validator.register_validator("other", config_validator.validate_qwen_config)   # Use same base validation


_register_default_validators()
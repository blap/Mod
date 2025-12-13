"""
Comprehensive Configuration Validation System for Qwen3-VL Model

This module provides a comprehensive validation system for configuration parameters throughout the Qwen3-VL model.
It includes validation for proper typing, acceptable ranges, and consistency across the system.
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, fields, asdict
from pathlib import Path
import logging
import torch
import math
import warnings
import os
import sys
from enum import Enum


logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Enum for validation strictness levels"""
    BASIC = "basic"
    MODERATE = "moderate"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors"""
    pass


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    config: Any  # The validated configuration object


class ConfigValidator:
    """Main configuration validator class for Qwen3-VL model configurations"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.MODERATE):
        self.level = level
        self.validation_rules = self._define_validation_rules(level)
    
    def _define_validation_rules(self, level: ValidationLevel) -> List[callable]:
        """Define validation rules based on validation level"""
        basic_rules = [
            self._validate_required_attributes,
            self._validate_positive_integer_values,
            self._validate_probability_ranges,
            self._validate_dimension_divisibility,
            self._validate_capacity_preservation
        ]
        
        if level in [ValidationLevel.MODERATE, ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            basic_rules.extend([
                self._validate_memory_config_consistency,
                self._validate_attention_config_consistency,
                self._validate_optimization_config_consistency
            ])
        
        if level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            basic_rules.extend([
                self._validate_dtype_consistency,
                self._validate_device_compatibility
            ])
        
        if level == ValidationLevel.COMPREHENSIVE:
            basic_rules.extend([
                self._validate_performance_impact,
                self._validate_accuracy_preservation
            ])
        
        return basic_rules
    
    def _validate_required_attributes(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate that required configuration attributes exist"""
        errors = []
        warnings = []
        
        required_attrs = [
            'num_hidden_layers', 'num_attention_heads', 'hidden_size', 
            'intermediate_size', 'vocab_size', 'max_position_embeddings',
            'vision_num_hidden_layers', 'vision_num_attention_heads', 'vision_hidden_size'
        ]
        
        for attr in required_attrs:
            if not hasattr(config, attr):
                errors.append(f"Missing required configuration attribute: {attr}")
        
        return errors, warnings
    
    def _validate_positive_integer_values(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate that positive integer values are properly set"""
        errors = []
        warnings = []
        
        positive_int_attrs = [
            ('num_hidden_layers', 1, 1000),
            ('num_attention_heads', 1, 128),
            ('hidden_size', 1, 8192),
            ('intermediate_size', 1, 32768),
            ('vocab_size', 1, 500000),
            ('max_position_embeddings', 1, 1000000),
            ('vision_num_hidden_layers', 1, 100),
            ('vision_num_attention_heads', 1, 128),
            ('vision_hidden_size', 1, 8192)
        ]
        
        for attr, min_val, max_val in positive_int_attrs:
            if hasattr(config, attr):
                val = getattr(config, attr)
                if not isinstance(val, int) or val <= 0:
                    errors.append(f"{attr} must be a positive integer, got {val} ({type(val).__name__})")
                elif val < min_val or val > max_val:
                    warnings.append(f"{attr} value {val} is outside recommended range [{min_val}, {max_val}]")
        
        return errors, warnings
    
    def _validate_probability_ranges(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate that probability values are within [0, 1] range"""
        errors = []
        warnings = []
        
        prob_attrs = [
            'attention_dropout_prob', 'hidden_dropout_prob', 'sparsity_ratio', 
            'compression_ratio', 'exit_threshold', 'gpu_memory_fraction'
        ]
        
        for attr in prob_attrs:
            if hasattr(config, attr):
                val = getattr(config, attr)
                if not 0.0 <= val <= 1.0:
                    errors.append(f"{attr} must be between 0.0 and 1.0, got {val}")
        
        return errors, warnings
    
    def _validate_dimension_divisibility(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate that dimensions are properly divisible"""
        errors = []
        warnings = []
        
        # Check hidden_size is divisible by num_attention_heads
        if hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
            if config.num_attention_heads > 0 and config.hidden_size % config.num_attention_heads != 0:
                errors.append(f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads ({config.num_attention_heads})")
        
        # Check vision_hidden_size is divisible by vision_num_attention_heads
        if hasattr(config, 'vision_hidden_size') and hasattr(config, 'vision_num_attention_heads'):
            if config.vision_num_attention_heads > 0 and config.vision_hidden_size % config.vision_num_attention_heads != 0:
                errors.append(f"vision_hidden_size ({config.vision_hidden_size}) must be divisible by vision_num_attention_heads ({config.vision_num_attention_heads})")
        
        return errors, warnings
    
    def _validate_capacity_preservation(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate that the model preserves full capacity"""
        errors = []
        warnings = []
        
        # For Qwen3-VL, we need to preserve 32 layers and 32 attention heads
        if hasattr(config, 'num_hidden_layers') and config.num_hidden_layers != 32:
            errors.append(f"num_hidden_layers should be 32 to preserve full capacity, got {config.num_hidden_layers}")
        
        if hasattr(config, 'num_attention_heads') and config.num_attention_heads != 32:
            errors.append(f"num_attention_heads should be 32 to preserve full capacity, got {config.num_attention_heads}")
        
        if hasattr(config, 'vision_num_hidden_layers') and config.vision_num_hidden_layers != 24:
            warnings.append(f"vision_num_hidden_layers is {config.vision_num_hidden_layers}, expected 24 for full capacity")
        
        if hasattr(config, 'vision_num_attention_heads') and config.vision_num_attention_heads != 16:
            warnings.append(f"vision_num_attention_heads is {config.vision_num_attention_heads}, expected 16 for full capacity")
        
        return errors, warnings
    
    def _validate_memory_config_consistency(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate memory configuration consistency"""
        errors = []
        warnings = []
        
        # Check memory-related attributes
        if hasattr(config, 'memory_config') and config.memory_config:
            memory_config = config.memory_config
            if hasattr(memory_config, 'memory_pool_size') and memory_config.memory_pool_size <= 0:
                errors.append(f"memory_pool_size must be positive, got {memory_config.memory_pool_size}")
        
        # Check memory pool growth factor
        if hasattr(config, 'memory_pool_growth_factor') and config.memory_pool_growth_factor <= 1.0:
            errors.append(f"memory_pool_growth_factor must be > 1.0, got {config.memory_pool_growth_factor}")
        
        return errors, warnings
    
    def _validate_attention_config_consistency(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate attention configuration consistency"""
        errors = []
        warnings = []
        
        # Check attention implementation
        if hasattr(config, 'attention_implementation'):
            valid_implementations = ["eager", "flash_attention_2", "sdpa", "kv_cache_optimized", "sparse_attention"]
            if config.attention_implementation not in valid_implementations:
                errors.append(f"attention_implementation must be one of {valid_implementations}, got {config.attention_implementation}")
        
        # Check KV cache strategy
        if hasattr(config, 'kv_cache_strategy'):
            valid_strategies = ["standard", "low_rank", "sliding_window", "hybrid"]
            if config.kv_cache_strategy not in valid_strategies:
                errors.append(f"kv_cache_strategy must be one of {valid_strategies}, got {config.kv_cache_strategy}")
        
        return errors, warnings
    
    def _validate_optimization_config_consistency(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate optimization configuration consistency"""
        errors = []
        warnings = []
        
        # Validate MoE configuration
        if hasattr(config, 'use_moe') and config.use_moe:
            if hasattr(config, 'moe_num_experts') and config.moe_num_experts < 2:
                errors.append(f"moe_num_experts must be at least 2 when MoE is enabled, got {config.moe_num_experts}")
            
            if hasattr(config, 'moe_top_k') and (config.moe_top_k < 1 or config.moe_top_k > config.moe_num_experts):
                errors.append(f"moe_top_k must be between 1 and moe_num_experts ({config.moe_num_experts}), got {config.moe_top_k}")
        
        return errors, warnings
    
    def _validate_dtype_consistency(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate data type consistency"""
        errors = []
        warnings = []
        
        if hasattr(config, 'torch_dtype'):
            valid_dtypes = ["float32", "float16", "bfloat16", "int8"]
            if config.torch_dtype not in valid_dtypes:
                errors.append(f"torch_dtype must be one of {valid_dtypes}, got {config.torch_dtype}")
        
        return errors, warnings
    
    def _validate_device_compatibility(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate configuration compatibility with device"""
        errors = []
        warnings = []
        
        # Check if configuration is compatible with available hardware
        if hasattr(config, 'use_flash_attention_2') and config.use_flash_attention_2:
            if not torch.cuda.is_available():
                warnings.append("flash_attention_2 enabled but CUDA is not available")
            elif torch.cuda.get_device_capability(0)[0] < 8:
                warnings.append("flash_attention_2 may not be fully supported on this GPU (compute capability < 8.0)")
        
        return errors, warnings
    
    def _validate_performance_impact(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate configuration for performance impact"""
        errors = []
        warnings = []
        
        # Check if optimization settings make sense together
        if (hasattr(config, 'use_sparsity') and config.use_sparsity and 
            hasattr(config, 'use_gradient_checkpointing') and config.use_gradient_checkpointing):
            # This combination is generally acceptable
            pass
        
        # Check for potentially conflicting optimizations
        if (hasattr(config, 'use_moe') and config.use_moe and
            hasattr(config, 'sparsity_ratio') and config.sparsity_ratio > 0.8):
            warnings.append("Very high sparsity combined with MoE may not provide optimal performance")
        
        return errors, warnings
    
    def _validate_accuracy_preservation(self, config: Any) -> Tuple[List[str], List[str]]:
        """Validate configuration for accuracy preservation"""
        errors = []
        warnings = []
        
        # Check if aggressive optimizations might impact accuracy
        if (hasattr(config, 'sparsity_ratio') and config.sparsity_ratio > 0.7):
            warnings.append(f"High sparsity ratio ({config.sparsity_ratio}) may impact model accuracy")
        
        if (hasattr(config, 'use_moe') and config.use_moe and
            hasattr(config, 'moe_top_k') and config.moe_top_k == 1 and
            hasattr(config, 'moe_num_experts') and config.moe_num_experts > 4):
            warnings.append("Using only 1 expert out of many in MoE may limit model capacity")
        
        return errors, warnings
    
    def validate_config(self, config: Any, strict: bool = True) -> ValidationResult:
        """
        Validate a configuration object against all rules.
        
        Args:
            config: Configuration object to validate
            strict: If True, raises ConfigValidationError on any error; if False, returns errors in result
            
        Returns:
            ValidationResult with validation results
        """
        all_errors = []
        all_warnings = []
        
        # Run all validation rules
        for rule in self.validation_rules:
            try:
                errors, warnings = rule(config)
                all_errors.extend(errors)
                all_warnings.extend(warnings)
            except Exception as e:
                all_errors.append(f"Error running validation rule: {str(e)}")
        
        # Check for deprecated parameters
        deprecated_warnings = self._check_deprecated_params(config)
        all_warnings.extend(deprecated_warnings)
        
        # Check for unused parameters
        unused_warnings = self._check_unused_params(config)
        all_warnings.extend(unused_warnings)
        
        result = ValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            config=config
        )
        
        if strict and not result.valid:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(all_errors)}")
        
        return result
    
    def _check_deprecated_params(self, config: Any) -> List[str]:
        """Check for deprecated configuration parameters"""
        warnings = []
        
        deprecated_params = [
            'deprecated_param',
            'old_feature_flag',
            'legacy_setting'
        ]
        
        for param in deprecated_params:
            if hasattr(config, param):
                warnings.append(f"Deprecated parameter '{param}' found in configuration")
        
        return warnings
    
    def _check_unused_params(self, config: Any) -> List[str]:
        """Check for unused configuration parameters"""
        warnings = []
        
        # This is a simplified check - in reality, you'd need to know which params are actually used
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            return warnings  # Can't check if not a dataclass or dict-compatible
        
        known_params = {
            'num_hidden_layers', 'num_attention_heads', 'hidden_size', 'intermediate_size',
            'vocab_size', 'max_position_embeddings', 'hidden_act', 'hidden_dropout_prob',
            'attention_dropout_prob', 'initializer_range', 'layer_norm_eps', 'pad_token_id',
            'tie_word_embeddings', 'torch_dtype', 'use_cache', 'rope_theta',
            'num_key_value_heads', 'vision_num_hidden_layers', 'vision_num_attention_heads',
            'vision_hidden_size', 'vision_intermediate_size', 'vision_patch_size',
            'vision_image_size', 'vision_window_size', 'vision_num_channels', 'vision_qkv_bias',
            'num_query_tokens', 'vision_projection_dim', 'language_projection_dim',
            'use_adapters', 'use_early_exit', 'exit_threshold', 'use_adaptive_depth',
            'use_vision_adaptive_depth', 'use_multimodal_adaptive_depth', 'min_depth_ratio',
            'max_depth_ratio', 'vision_min_depth_ratio', 'vision_max_depth_ratio',
            'depth_temperature', 'use_context_adaptive_positional_encoding',
            'use_cross_modal_positional_encoding', 'use_conditional_feature_extraction',
            'use_moe', 'moe_num_experts', 'moe_top_k', 'use_sparsity', 'sparsity_ratio',
            'use_adaptive_precision', 'use_cross_modal_compression', 'compression_ratio',
            'use_cross_layer_memory_sharing', 'use_mixed_precision', 'gpu_memory_size',
            'use_inference_memory_efficient', 'attention_implementation',
            'use_flash_attention_2', 'flash_attention_causal', 'use_memory_efficient_attention',
            'use_dynamic_sparse_attention', 'sparse_attention_sparsity_ratio',
            'vision_sparse_attention_sparsity_ratio', 'sparse_attention_pattern',
            'sparse_attention_num_blocks', 'use_rotary_embedding',
            'use_approximated_rotary_embeddings', 'rotary_embedding_scaling_factor', 'head_dim',
            'use_block_sparse_attention', 'block_sparse_block_size',
            'use_learned_attention_routing', 'learned_routing_temperature',
            'use_memory_pooling', 'memory_pool_initial_size', 'memory_pool_max_size',
            'memory_pool_growth_factor', 'use_buddy_allocation', 'memory_defragmentation_enabled',
            'memory_defragmentation_threshold', 'kv_cache_strategy', 'use_low_rank_kv_cache',
            'kv_cache_window_size', 'kv_low_rank_dimension', 'kv_cache_max_length',
            'use_gradient_checkpointing', 'use_activation_sparsity', 'memory_efficient_attention',
            'use_vision_memory_optimization', 'vision_memory_chunk_size', 'use_tensor_fusion',
            'use_pre_allocated_tensors', 'pre_allocated_cache_size', 'hardware_compute_capability',
            'memory_size_gb', 'use_nvme_cache', 'routing_method', 'routing_top_k',
            'routing_temperature', 'common_compressed_dim', 'enable_cross_modal_compression',
            'enable_cross_layer_memory_sharing', 'use_context_adaptive_positional_encoding',
            'use_conditional_feature_extraction', 'memory_pool_size', 'memory_pool_growth_factor',
            'memory_pool_dtype', 'enable_memory_tiering', 'gpu_memory_fraction',
            'compression_level', 'enable_memory_swapping', 'swap_threshold', 'swap_algorithm',
            'enable_memory_defragmentation', 'defragmentation_interval', 'defragmentation_threshold',
            'num_threads', 'num_workers', 'max_concurrent_threads', 'l1_cache_size',
            'l2_cache_size', 'l3_cache_size', 'cache_line_size', 'enable_cpu_optimizations',
            'use_hyperthreading', 'enable_simd_optimizations', 'simd_instruction_set',
            'num_preprocess_workers', 'preprocess_batch_size', 'memory_threshold',
            'transfer_async', 'gpu_compute_capability', 'max_threads_per_block',
            'shared_memory_per_block', 'memory_bandwidth_gbps', 'enable_gpu_optimizations',
            'use_tensor_cores', 'use_mixed_precision', 'enable_cuda_graphs',
            'enable_power_optimization', 'power_constraint', 'thermal_constraint',
            'performance_target', 'adaptation_frequency', 'enable_dynamic_power_scaling',
            'use_hierarchical_memory_compression', 'use_kv_cache_optimization',
            'use_cross_layer_parameter_sharing', 'use_dynamic_sparse_attention',
            'use_adaptive_precision', 'use_moe', 'use_flash_attention_2',
            'use_adaptive_depth', 'use_gradient_checkpointing',
            'use_context_adaptive_positional_encoding', 'use_conditional_feature_extraction',
            'use_cross_modal_compression', 'use_cross_layer_memory_sharing',
            'use_hierarchical_vision', 'use_learned_activation_routing',
            'use_adaptive_batch_processing', 'use_adaptive_sequence_packing',
            'use_memory_efficient_grad_accumulation', 'use_faster_rotary_embeddings',
            'use_distributed_pipeline_parallelism', 'use_hardware_specific_kernels',
            'performance_improvement_threshold', 'accuracy_preservation_threshold',
            'hardware_target', 'target_hardware', 'compute_units', 'memory_gb',
            'optimization_level', 'sparsity_pattern', 'sparsity_num_blocks',
            'memory_config', 'cpu_config', 'gpu_config', 'power_config',
            'optimization_config', 'exit_threshold', 'sparsity_temperature',
            'sparsity_top_k', 'low_rank_dimension', 'window_size',
            'use_tensor_cores', 'tensor_cores_enabled', 'enable_tensor_cores',
            'enable_memory_pooling', 'enable_cache_optimization', 'enable_tensor_fusion',
            'enable_memory_efficient_attention', 'enable_sparsity', 'enable_moe',
            'enable_flash_attention', 'enable_gradient_checkpointing', 'enable_adaptive_depth',
            'enable_context_adaptive_positional_encoding', 'enable_conditional_feature_extraction',
            'enable_cross_modal_compression', 'enable_cross_layer_memory_sharing',
            'enable_hierarchical_vision', 'enable_learned_activation_routing',
            'enable_adaptive_batch_processing', 'enable_adaptive_sequence_packing',
            'enable_memory_efficient_grad_accumulation', 'enable_faster_rotary_embeddings',
            'enable_distributed_pipeline_parallelism', 'enable_hardware_specific_kernels',
            'enable_hierarchical_memory_compression', 'enable_kv_cache_optimization',
            'enable_cross_layer_parameter_sharing', 'enable_dynamic_sparse_attention',
            'enable_adaptive_precision', 'use_mixed_precision_training',
            'enable_tensor_parallelism', 'enable_pipeline_parallelism',
            'enable_sequence_parallelism', 'enable_data_parallelism',
            'enable_parameter_efficient_tuning', 'enable_quantization',
            'enable_pruning', 'enable_knowledge_distillation',
            'enable_memory_efficient_inference', 'enable_low_precision_inference',
            'enable_sparse_inference', 'enable_moe_inference',
            'enable_flash_inference', 'enable_optimized_attention',
            'enable_optimized_mlp', 'enable_optimized_normalization',
            'enable_optimized_embeddings', 'enable_optimized_projections',
            # Add the missing parameters that were generating warnings
            'vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor'
        }
        
        for param in config_dict:
            if param not in known_params:
                warnings.append(f"Unknown parameter '{param}' found in configuration")
        
        return warnings
    
    def validate_config_from_file(self, file_path: Union[str, Path], strict: bool = True) -> ValidationResult:
        """
        Validate a configuration file.
        
        Args:
            file_path: Path to the configuration file
            strict: If True, raises ConfigValidationError on any error; if False, returns errors in result
            
        Returns:
            ValidationResult with validation results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {file_path}")
        
        # Load configuration based on file extension
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # Create a config object from the dictionary
        from .base_config import Qwen3VLConfig
        config = Qwen3VLConfig.from_dict(config_dict)
        
        return self.validate_config(config, strict)
    
    def validate_multiple_configs(self, configs: List[Any], strict: bool = True) -> List[ValidationResult]:
        """
        Validate multiple configurations.
        
        Args:
            configs: List of configuration objects to validate
            strict: If True, raises ConfigValidationError on any error; if False, returns errors in result
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        for config in configs:
            result = self.validate_config(config, strict=False)  # Always use non-strict for individual validation
            results.append(result)
            
            # If strict mode and any config fails, raise an error
            if strict and not result.valid:
                raise ConfigValidationError(f"Configuration {config} validation failed: {'; '.join(result.errors)}")
        
        return results


class ConfigValidationManager:
    """Manager class for handling configuration validation across the system"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.MODERATE):
        self.level = level
        self.validator = ConfigValidator(level)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_config(self, config: Any, strict: bool = True) -> ValidationResult:
        """Validate a single configuration"""
        return self.validator.validate_config(config, strict)
    
    def validate_config_with_context(self, config: Any, context: str = "model") -> ValidationResult:
        """
        Validate configuration with specific context (model, memory, hardware, etc.)
        
        Args:
            config: Configuration object to validate
            context: Context for validation (model, memory, hardware, optimization)
            
        Returns:
            ValidationResult with validation results
        """
        # First run general validation
        general_result = self.validator.validate_config(config, strict=False)
        
        # Then run context-specific validation
        context_errors, context_warnings = self._validate_context_specific(config, context)
        
        general_result.errors.extend(context_errors)
        general_result.warnings.extend(context_warnings)
        general_result.valid = len(general_result.errors) == 0
        
        if strict and not general_result.valid:
            raise ConfigValidationError(f"Context-specific validation failed: {'; '.join(context_errors)}")
        
        return general_result
    
    def _validate_context_specific(self, config: Any, context: str) -> Tuple[List[str], List[str]]:
        """Run context-specific validations"""
        errors = []
        warnings = []
        
        if context == "model":
            # Model-specific validations
            if hasattr(config, 'num_hidden_layers') and config.num_hidden_layers <= 0:
                errors.append(f"num_hidden_layers must be positive in model context, got {config.num_hidden_layers}")
            if hasattr(config, 'num_attention_heads') and config.num_attention_heads <= 0:
                errors.append(f"num_attention_heads must be positive in model context, got {config.num_attention_heads}")
        elif context == "memory":
            # Memory-specific validations
            if hasattr(config, 'memory_config') and config.memory_config:
                if config.memory_config.memory_pool_size <= 0:
                    errors.append(f"memory_pool_size must be positive in memory context, got {config.memory_config.memory_pool_size}")
                if config.memory_config.memory_pool_growth_factor <= 1.0:
                    errors.append(f"memory_pool_growth_factor must be > 1.0 in memory context, got {config.memory_config.memory_pool_growth_factor}")
        elif context == "hardware":
            # Hardware-specific validations
            if hasattr(config, 'hardware_compute_capability'):
                if not isinstance(config.hardware_compute_capability, tuple) or len(config.hardware_compute_capability) != 2:
                    errors.append(f"hardware_compute_capability must be a tuple of length 2, got {type(config.hardware_compute_capability)}")
                elif not all(isinstance(x, int) and x >= 0 for x in config.hardware_compute_capability):
                    errors.append(f"hardware_compute_capability must contain non-negative integers, got {config.hardware_compute_capability}")
        elif context == "optimization":
            # Optimization-specific validations
            if hasattr(config, 'optimization_config') and config.optimization_config:
                opt_config = config.optimization_config
                if hasattr(opt_config, 'sparsity_ratio') and not (0.0 <= opt_config.sparsity_ratio <= 1.0):
                    errors.append(f"sparsity_ratio must be between 0.0 and 1.0 in optimization context, got {opt_config.sparsity_ratio}")
        
        return errors, warnings
    
    def validate_config_compatibility(self, config1: Any, config2: Any) -> ValidationResult:
        """
        Validate compatibility between two configurations.
        
        Args:
            config1: First configuration object
            config2: Second configuration object
            
        Returns:
            ValidationResult with compatibility validation results
        """
        result1 = self.validator.validate_config(config1, strict=False)
        result2 = self.validator.validate_config(config2, strict=False)
        
        compatibility_errors = []
        
        # Check if both configurations are valid individually
        if not result1.valid:
            compatibility_errors.extend([f"Config 1 error: {err}" for err in result1.errors])
        if not result2.valid:
            compatibility_errors.extend([f"Config 2 error: {err}" for err in result2.errors])
        
        # Check for compatibility between specific parameters
        if (hasattr(config1, 'hidden_size') and hasattr(config2, 'hidden_size') and
            config1.hidden_size != config2.hidden_size):
            compatibility_errors.append(f"hidden_size mismatch: {config1.hidden_size} vs {config2.hidden_size}")
        
        if (hasattr(config1, 'num_attention_heads') and hasattr(config2, 'num_attention_heads') and
            config1.num_attention_heads != config2.num_attention_heads):
            compatibility_errors.append(f"num_attention_heads mismatch: {config1.num_attention_heads} vs {config2.num_attention_heads}")
        
        if (hasattr(config1, 'num_hidden_layers') and hasattr(config2, 'num_hidden_layers') and
            config1.num_hidden_layers != config2.num_hidden_layers):
            compatibility_errors.append(f"num_hidden_layers mismatch: {config1.num_hidden_layers} vs {config2.num_hidden_layers}")
        
        return ValidationResult(
            valid=len(compatibility_errors) == 0,
            errors=compatibility_errors,
            warnings=result1.warnings + result2.warnings,
            config=(config1, config2)
        )
    
    def validate_config_for_hardware(self, config: Any, hardware_specs: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration for specific hardware specifications.
        
        Args:
            config: Configuration object to validate
            hardware_specs: Dictionary with hardware specifications
            
        Returns:
            ValidationResult with hardware-specific validation results
        """
        result = self.validator.validate_config(config, strict=False)
        
        hardware_errors = []
        
        # Validate based on hardware memory
        if 'memory_gb' in hardware_specs:
            memory_gb = hardware_specs['memory_gb']
            
            # Estimate memory requirements based on config
            estimated_memory_gb = (
                config.num_hidden_layers * 
                (config.hidden_size * config.hidden_size * 4 * 2) / (1024**3)  # Approximate for attention weights
            )
            
            if estimated_memory_gb > memory_gb * 0.8:  # Using 80% of available memory
                hardware_errors.append(f"Estimated memory usage ({estimated_memory_gb:.2f}GB) exceeds available memory ({memory_gb}GB)")
        
        # Validate based on compute capability
        if 'compute_capability' in hardware_specs:
            compute_cap = hardware_specs['compute_capability']
            
            if (hasattr(config, 'use_flash_attention_2') and config.use_flash_attention_2 and
                compute_cap[0] < 8):
                result.warnings.append(f"Flash attention 2 recommended for compute capability >= 8.0, got {compute_cap}")
        
        # Add hardware-specific errors to result
        result.errors.extend(hardware_errors)
        result.valid = len(result.errors) == 0
        
        return result
    
    def get_validation_report(self, config: Any) -> Dict[str, Any]:
        """
        Get a detailed validation report for a configuration.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            Dictionary with detailed validation report
        """
        validation_result = self.validator.validate_config(config, strict=False)
        
        report = {
            'valid': validation_result.valid,
            'errors': validation_result.errors,
            'warnings': validation_result.warnings,
            'config_summary': {
                'num_hidden_layers': getattr(config, 'num_hidden_layers', 'N/A'),
                'num_attention_heads': getattr(config, 'num_attention_heads', 'N/A'),
                'hidden_size': getattr(config, 'hidden_size', 'N/A'),
                'intermediate_size': getattr(config, 'intermediate_size', 'N/A'),
                'vocab_size': getattr(config, 'vocab_size', 'N/A'),
                'max_position_embeddings': getattr(config, 'max_position_embeddings', 'N/A'),
                'vision_num_hidden_layers': getattr(config, 'vision_num_hidden_layers', 'N/A'),
                'vision_num_attention_heads': getattr(config, 'vision_num_attention_heads', 'N/A'),
                'vision_hidden_size': getattr(config, 'vision_hidden_size', 'N/A'),
                'use_sparsity': getattr(config, 'use_sparsity', 'N/A'),
                'use_moe': getattr(config, 'use_moe', 'N/A'),
                'use_flash_attention_2': getattr(config, 'use_flash_attention_2', 'N/A'),
                'use_gradient_checkpointing': getattr(config, 'use_gradient_checkpointing', 'N/A'),
                'use_adaptive_depth': getattr(config, 'use_adaptive_depth', 'N/A'),
                'use_context_adaptive_positional_encoding': getattr(config, 'use_context_adaptive_positional_encoding', 'N/A'),
                'use_conditional_feature_extraction': getattr(config, 'use_conditional_feature_extraction', 'N/A'),
                'use_cross_modal_compression': getattr(config, 'use_cross_modal_compression', 'N/A'),
                'use_cross_layer_memory_sharing': getattr(config, 'use_cross_layer_memory_sharing', 'N/A'),
                'use_hierarchical_vision': getattr(config, 'use_hierarchical_vision', 'N/A'),
                'use_learned_activation_routing': getattr(config, 'use_learned_activation_routing', 'N/A'),
                'use_adaptive_batch_processing': getattr(config, 'use_adaptive_batch_processing', 'N/A'),
                'use_adaptive_sequence_packing': getattr(config, 'use_adaptive_sequence_packing', 'N/A'),
                'use_memory_efficient_grad_accumulation': getattr(config, 'use_memory_efficient_grad_accumulation', 'N/A'),
                'use_faster_rotary_embeddings': getattr(config, 'use_faster_rotary_embeddings', 'N/A'),
                'use_distributed_pipeline_parallelism': getattr(config, 'use_distributed_pipeline_parallelism', 'N/A'),
                'use_hardware_specific_kernels': getattr(config, 'use_hardware_specific_kernels', 'N/A'),
            },
            'validation_level': self.level.value,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        return report


# Global validator instance
config_validator = ConfigValidator(ValidationLevel.MODERATE)
validation_manager = ConfigValidationManager(ValidationLevel.MODERATE)


def validate_config_object(config: Any, strict: bool = True, level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResult:
    """
    Validate a configuration object using the validation system.
    
    Args:
        config: Configuration object to validate
        strict: If True, raises ConfigValidationError on any error; if False, returns errors in result
        level: Validation level to use
        
    Returns:
        ValidationResult with validation results
    """
    validator = ConfigValidator(level)
    return validator.validate_config(config, strict)


def validate_config_file(file_path: Union[str, Path], strict: bool = True, level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResult:
    """
    Validate a configuration file using the validation system.
    
    Args:
        file_path: Path to the configuration file
        strict: If True, raises ConfigValidationError on any error; if False, returns errors in result
        level: Validation level to use
        
    Returns:
        ValidationResult with validation results
    """
    validator = ConfigValidator(level)
    return validator.validate_config_from_file(file_path, strict)


def get_config_validation_report(config: Any) -> Dict[str, Any]:
    """
    Get a detailed validation report for a configuration.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        Dictionary with detailed validation report
    """
    return validation_manager.get_validation_report(config)


def validate_config_pair_compatibility(config1: Any, config2: Any) -> ValidationResult:
    """
    Validate compatibility between two configurations.
    
    Args:
        config1: First configuration object
        config2: Second configuration object
        
    Returns:
        ValidationResult with compatibility validation results
    """
    return validation_manager.validate_config_compatibility(config1, config2)


def validate_config_for_hardware_target(config: Any, hardware_specs: Dict[str, Any]) -> ValidationResult:
    """
    Validate configuration for specific hardware target.
    
    Args:
        config: Configuration object to validate
        hardware_specs: Hardware specifications dictionary
        
    Returns:
        ValidationResult with hardware-specific validation results
    """
    return validation_manager.validate_config_for_hardware(config, hardware_specs)


# Export classes and functions
__all__ = [
    "ConfigValidator", 
    "ConfigValidationManager",
    "ValidationResult",
    "ValidationLevel",
    "ConfigValidationError", 
    "validate_config_object",
    "validate_config_file", 
    "get_config_validation_report",
    "validate_config_pair_compatibility",
    "validate_config_for_hardware_target",
    "config_validator",
    "validation_manager"
]
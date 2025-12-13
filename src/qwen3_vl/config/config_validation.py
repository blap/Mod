"""
Configuration Validation Module for Qwen3-VL Model

This module provides comprehensive validation for configuration parameters throughout the Qwen3-VL system.
It includes validation for proper typing, acceptable ranges, and consistency across the system.
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, fields, asdict
from pathlib import Path
import logging
import torch
import math
import warnings
import os
import sys


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors"""
    pass


@dataclass
class ValidationRule:
    """Definition of a validation rule"""
    name: str
    description: str
    check_function: callable
    error_message: str
    warning_message: Optional[str] = None


class ConfigValidator:
    """Main configuration validator class for Qwen3-VL model configurations"""
    
    def __init__(self):
        self.validation_rules = self._define_validation_rules()
    
    def _define_validation_rules(self) -> List[ValidationRule]:
        """Define standard validation rules for Qwen3-VL configurations"""
        return [
            # Core model configuration rules
            ValidationRule(
                name="hidden_size_positive",
                description="hidden_size must be positive",
                check_function=lambda config: getattr(config, 'hidden_size', 0) > 0,
                error_message="hidden_size must be positive"
            ),
            ValidationRule(
                name="num_attention_heads_positive",
                description="num_attention_heads must be positive",
                check_function=lambda config: getattr(config, 'num_attention_heads', 0) > 0,
                error_message="num_attention_heads must be positive"
            ),
            ValidationRule(
                name="num_hidden_layers_positive",
                description="num_hidden_layers must be positive",
                check_function=lambda config: getattr(config, 'num_hidden_layers', 0) > 0,
                error_message="num_hidden_layers must be positive"
            ),
            ValidationRule(
                name="hidden_size_divisible_by_heads",
                description="hidden_size must be divisible by num_attention_heads",
                check_function=lambda config: (
                    getattr(config, 'hidden_size', 0) % getattr(config, 'num_attention_heads', 1) == 0
                    if getattr(config, 'num_attention_heads', 1) > 0
                    else True
                ),
                error_message="hidden_size must be divisible by num_attention_heads"
            ),
            ValidationRule(
                name="vocab_size_positive",
                description="vocab_size must be positive",
                check_function=lambda config: getattr(config, 'vocab_size', 0) > 0,
                error_message="vocab_size must be positive"
            ),
            ValidationRule(
                name="max_position_embeddings_positive",
                description="max_position_embeddings must be positive",
                check_function=lambda config: getattr(config, 'max_position_embeddings', 0) > 0,
                error_message="max_position_embeddings must be positive"
            ),
            
            # Vision model configuration rules
            ValidationRule(
                name="vision_hidden_size_positive",
                description="vision_hidden_size must be positive",
                check_function=lambda config: getattr(config, 'vision_hidden_size', 0) > 0,
                error_message="vision_hidden_size must be positive"
            ),
            ValidationRule(
                name="vision_num_attention_heads_positive",
                description="vision_num_attention_heads must be positive",
                check_function=lambda config: getattr(config, 'vision_num_attention_heads', 0) > 0,
                error_message="vision_num_attention_heads must be positive"
            ),
            ValidationRule(
                name="vision_hidden_size_divisible_by_heads",
                description="vision_hidden_size must be divisible by vision_num_attention_heads",
                check_function=lambda config: (
                    getattr(config, 'vision_hidden_size', 0) % getattr(config, 'vision_num_attention_heads', 1) == 0
                    if getattr(config, 'vision_num_attention_heads', 1) > 0
                    else True
                ),
                error_message="vision_hidden_size must be divisible by vision_num_attention_heads"
            ),
            
            # Optimization configuration rules
            ValidationRule(
                name="sparsity_ratio_range",
                description="sparsity_ratio must be between 0.0 and 1.0",
                check_function=lambda config: 0.0 <= getattr(config, 'sparsity_ratio', 0.5) <= 1.0,
                error_message="sparsity_ratio must be between 0.0 and 1.0"
            ),
            ValidationRule(
                name="compression_ratio_range",
                description="compression_ratio must be between 0.0 and 1.0",
                check_function=lambda config: 0.0 <= getattr(config, 'compression_ratio', 0.5) <= 1.0,
                error_message="compression_ratio must be between 0.0 and 1.0"
            ),
            ValidationRule(
                name="exit_threshold_range",
                description="exit_threshold must be between 0.0 and 1.0",
                check_function=lambda config: 0.0 <= getattr(config, 'exit_threshold', 0.8) <= 1.0,
                error_message="exit_threshold must be between 0.0 and 1.0"
            ),
            ValidationRule(
                name="moe_num_experts_range",
                description="moe_num_experts must be at least 2 when MoE is enabled",
                check_function=lambda config: (
                    getattr(config, 'moe_num_experts', 1) >= 2
                    if getattr(config, 'use_moe', False)
                    else True
                ),
                error_message="moe_num_experts must be at least 2 when MoE is enabled"
            ),
            ValidationRule(
                name="moe_top_k_range",
                description="moe_top_k must be between 1 and moe_num_experts",
                check_function=lambda config: (
                    1 <= getattr(config, 'moe_top_k', 1) <= getattr(config, 'moe_num_experts', 4)
                    if getattr(config, 'use_moe', False)
                    else True
                ),
                error_message="moe_top_k must be between 1 and moe_num_experts"
            ),
            
            # Dropout probability rules
            ValidationRule(
                name="attention_dropout_range",
                description="attention_dropout_prob must be between 0.0 and 1.0",
                check_function=lambda config: 0.0 <= getattr(config, 'attention_dropout_prob', 0.0) <= 1.0,
                error_message="attention_dropout_prob must be between 0.0 and 1.0"
            ),
            ValidationRule(
                name="hidden_dropout_range",
                description="hidden_dropout_prob must be between 0.0 and 1.0",
                check_function=lambda config: 0.0 <= getattr(config, 'hidden_dropout_prob', 0.0) <= 1.0,
                error_message="hidden_dropout_prob must be between 0.0 and 1.0"
            ),
            
            # Layer normalization rules
            ValidationRule(
                name="layer_norm_eps_range",
                description="layer_norm_eps must be between 1e-12 and 1e-3",
                check_function=lambda config: 1e-12 <= getattr(config, 'layer_norm_eps', 1e-5) <= 1e-3,
                error_message="layer_norm_eps must be between 1e-12 and 1e-3"
            ),
            
            # Rotary embedding rules
            ValidationRule(
                name="rope_theta_range",
                description="rope_theta must be between 1000.0 and 10000000.0",
                check_function=lambda config: 1000.0 <= getattr(config, 'rope_theta', 10000.0) <= 10000000.0,
                error_message="rope_theta must be between 1000.0 and 10000000.0"
            ),
            
            # Memory configuration rules
            ValidationRule(
                name="memory_pool_size_positive",
                description="memory_pool_size must be positive",
                check_function=lambda config: getattr(config.memory_config, 'memory_pool_size', 0) > 0 if hasattr(config, 'memory_config') and config.memory_config else True,
                error_message="memory_pool_size must be positive"
            ),
            ValidationRule(
                name="memory_pool_growth_factor",
                description="memory_pool_growth_factor must be > 1.0",
                check_function=lambda config: getattr(config.memory_config, 'memory_pool_growth_factor', 1.0) > 1.0 if hasattr(config, 'memory_config') and config.memory_config else True,
                error_message="memory_pool_growth_factor must be > 1.0"
            ),
            
            # Hardware configuration rules
            ValidationRule(
                name="gpu_memory_size_positive",
                description="gpu_memory_size must be positive",
                check_function=lambda config: getattr(config.gpu_config, 'gpu_memory_size', 0) > 0 if hasattr(config, 'gpu_config') and config.gpu_config else True,
                error_message="gpu_memory_size must be positive"
            ),
            ValidationRule(
                name="compute_units_positive",
                description="compute_units must be positive",
                check_function=lambda config: getattr(config, 'compute_units', 0) > 0,
                error_message="compute_units must be positive"
            ),
        ]
    
    def validate_config(self, config: Any, strict: bool = True) -> Dict[str, Any]:
        """
        Validate a configuration object against all rules.
        
        Args:
            config: Configuration object to validate
            strict: If True, raises ConfigValidationError on any error; if False, returns errors in result
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Check if config is a dataclass or has a to_dict method
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            errors.append("Configuration object must be a dataclass or have a to_dict method")
            return {
                'valid': False,
                'errors': errors,
                'warnings': warnings,
                'config': config
            }
        
        # Run all validation rules
        for rule in self.validation_rules:
            try:
                if not rule.check_function(config):
                    errors.append(rule.error_message)
                elif hasattr(config, f'use_{rule.name.split("_")[0]}') and not rule.check_function(config):
                    # For optimization-specific rules
                    if getattr(config, f'use_{rule.name.split("_")[0]}', False):
                        errors.append(rule.error_message)
            except Exception as e:
                errors.append(f"Error validating {rule.name}: {str(e)}")
        
        # Additional custom validations
        errors.extend(self._custom_validations(config))
        
        # Check for deprecated parameters
        deprecated_warnings = self._check_deprecated_params(config_dict)
        warnings.extend(deprecated_warnings)
        
        # Check for unused parameters
        unused_warnings = self._check_unused_params(config_dict)
        warnings.extend(unused_warnings)
        
        result = {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config': config
        }
        
        if strict and not result['valid']:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return result
    
    def _custom_validations(self, config: Any) -> List[str]:
        """Perform custom validations that are not covered by standard rules"""
        errors = []
        
        # Check for capacity preservation (should be 32 layers and 32 heads)
        if hasattr(config, 'num_hidden_layers') and config.num_hidden_layers != 32:
            errors.append(f"num_hidden_layers should be 32 to preserve full capacity, got {config.num_hidden_layers}")
        
        if hasattr(config, 'num_attention_heads') and config.num_attention_heads != 32:
            errors.append(f"num_attention_heads should be 32 to preserve full capacity, got {config.num_attention_heads}")
        
        # Check for consistency between intermediate_size and hidden_size
        if hasattr(config, 'hidden_size') and hasattr(config, 'intermediate_size'):
            if config.intermediate_size < config.hidden_size:
                warnings.warn("intermediate_size is smaller than hidden_size, which may impact model capacity")
        
        # Check for consistency between vision parameters
        if hasattr(config, 'vision_num_hidden_layers') and config.vision_num_hidden_layers != 24:
            warnings.warn(f"vision_num_hidden_layers is {config.vision_num_hidden_layers}, expected 24 for full capacity")
        
        if hasattr(config, 'vision_num_attention_heads') and config.vision_num_attention_heads != 16:
            warnings.warn(f"vision_num_attention_heads is {config.vision_num_attention_heads}, expected 16 for full capacity")
        
        return errors
    
    def _check_deprecated_params(self, config_dict: Dict[str, Any]) -> List[str]:
        """Check for deprecated configuration parameters"""
        deprecated_params = [
            'deprecated_param',
            'old_feature_flag',
            'legacy_setting'
        ]
        
        warnings = []
        for param in deprecated_params:
            if param in config_dict:
                warnings.append(f"Deprecated parameter '{param}' found in configuration")
        
        return warnings
    
    def _check_unused_params(self, config_dict: Dict[str, Any]) -> List[str]:
        """Check for unused configuration parameters"""
        # This is a simplified check - in reality, you'd need to know which params are actually used
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
            'use_approximated_rotary_embeddings', 'rope_scaling_factor', 'head_dim',
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
            'optimization_level',
            # Add the missing parameters that were generating warnings
            'vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor'
        }
        
        warnings = []
        for param in config_dict:
            if param not in known_params:
                warnings.append(f"Unknown parameter '{param}' found in configuration")
        
        return warnings
    
    def validate_file(self, file_path: Union[str, Path], strict: bool = True) -> Dict[str, Any]:
        """
        Validate a configuration file.
        
        Args:
            file_path: Path to the configuration file
            strict: If True, raises ConfigValidationError on any error; if False, returns errors in result
            
        Returns:
            Dictionary with validation results
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
        from .config import Qwen3VLConfig
        config = Qwen3VLConfig.from_dict(config_dict)
        
        return self.validate_config(config, strict)
    
    def validate_config_with_context(self, config: Any, context: str = "model") -> Dict[str, Any]:
        """
        Validate configuration with specific context (model, memory, hardware, etc.)
        
        Args:
            config: Configuration object to validate
            context: Context for validation (model, memory, hardware, optimization)
            
        Returns:
            Dictionary with validation results
        """
        # First run general validation
        general_result = self.validate_config(config, strict=False)
        
        # Then run context-specific validation
        context_errors = self._validate_context_specific(config, context)
        
        general_result['errors'].extend(context_errors)
        general_result['valid'] = len(general_result['errors']) == 0
        
        return general_result
    
    def _validate_context_specific(self, config: Any, context: str) -> List[str]:
        """Run context-specific validations"""
        errors = []
        
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
        
        return errors


class ConfigValidatorService:
    """Service class for configuration validation with additional utilities"""
    
    def __init__(self):
        self.validator = ConfigValidator()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_and_report(self, config: Any) -> Dict[str, Any]:
        """Validate configuration and return detailed report"""
        validation_result = self.validator.validate_config(config, strict=False)
        
        report = {
            'valid': validation_result['valid'],
            'errors': validation_result['errors'],
            'warnings': validation_result['warnings'],
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
            }
        }
        
        return report
    
    def validate_multiple_configs(self, configs: List[Any], context: str = "model") -> Dict[str, Any]:
        """Validate multiple configurations"""
        results = []
        all_valid = True
        
        for i, config in enumerate(configs):
            try:
                result = self.validator.validate_config_with_context(config, context)
                results.append({
                    'index': i,
                    'config_name': getattr(config, 'name', f'config_{i}'),
                    'valid': result['valid'],
                    'errors': result['errors'],
                    'warnings': result['warnings']
                })
                
                if not result['valid']:
                    all_valid = False
            except Exception as e:
                results.append({
                    'index': i,
                    'config_name': getattr(config, 'name', f'config_{i}'),
                    'valid': False,
                    'errors': [f"Unexpected error validating config {i}: {str(e)}"],
                    'warnings': []
                })
                all_valid = False
        
        return {
            'all_valid': all_valid,
            'individual_results': results,
            'summary': {
                'total_configs': len(configs),
                'valid_configs': sum(1 for r in results if r['valid']),
                'invalid_configs': sum(1 for r in results if not r['valid'])
            }
        }
    
    def validate_config_compatibility(self, config1: Any, config2: Any) -> Dict[str, Any]:
        """Validate compatibility between two configurations"""
        result1 = self.validator.validate_config(config1, strict=False)
        result2 = self.validator.validate_config(config2, strict=False)
        
        compatibility_errors = []
        
        # Check if both configurations are valid individually
        if not result1['valid']:
            compatibility_errors.extend([f"Config 1 error: {err}" for err in result1['errors']])
        if not result2['valid']:
            compatibility_errors.extend([f"Config 2 error: {err}" for err in result2['errors']])
        
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
        
        return {
            'compatible': len(compatibility_errors) == 0,
            'errors': compatibility_errors,
            'config1_valid': result1['valid'],
            'config2_valid': result2['valid']
        }


# Global validator instance
config_validator = ConfigValidator()
validator_service = ConfigValidatorService()


def validate_config_object(config: Any, strict: bool = True) -> Dict[str, Any]:
    """
    Validate a configuration object using the global validator.
    
    Args:
        config: Configuration object to validate
        strict: If True, raises ConfigValidationError on any error; if False, returns errors in result
        
    Returns:
        Dictionary with validation results
    """
    return config_validator.validate_config(config, strict)


def validate_config_file(file_path: Union[str, Path], strict: bool = True) -> Dict[str, Any]:
    """
    Validate a configuration file using the global validator.
    
    Args:
        file_path: Path to the configuration file
        strict: If True, raises ConfigValidationError on any error; if False, returns errors in result
        
    Returns:
        Dictionary with validation results
    """
    return config_validator.validate_file(file_path, strict)


def get_config_validation_report(config: Any) -> Dict[str, Any]:
    """
    Get a detailed validation report for a configuration.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        Dictionary with detailed validation report
    """
    return validator_service.validate_and_report(config)


def validate_config_pair_compatibility(config1: Any, config2: Any) -> Dict[str, Any]:
    """
    Validate compatibility between two configurations.
    
    Args:
        config1: First configuration object
        config2: Second configuration object
        
    Returns:
        Dictionary with compatibility validation results
    """
    return validator_service.validate_config_compatibility(config1, config2)


# Export classes and functions
__all__ = [
    "ConfigValidator", 
    "ConfigValidationError", 
    "ValidationRule",
    "ConfigValidatorService",
    "validate_config_object",
    "validate_config_file", 
    "get_config_validation_report",
    "validate_config_pair_compatibility",
    "config_validator",
    "validator_service"
]
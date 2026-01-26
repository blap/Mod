"""
Configuration Validator for Inference-PIO System

This module provides utilities for validating configurations before they are applied
to models in the Inference-PIO system.
"""

import re
from typing import Any, Dict, List, Optional, Union
from .config_manager import DynamicConfig
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Utility class for validating configurations for different models.
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize validation rules for different configuration parameters.
        
        Returns:
            Dictionary of validation rules
        """
        return {
            # Model identification
            'model_path': {
                'type': str,
                'required': True,
                'validator': lambda x: len(x.strip()) > 0
            },
            'model_name': {
                'type': str,
                'required': True,
                'validator': lambda x: len(x.strip()) > 0
            },
            
            # Device settings
            'device': {
                'type': str,
                'required': False,
                'validator': lambda x: x is None or x in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps']
            },
            
            # Model architecture parameters
            'hidden_size': {
                'type': int,
                'required': True,
                'validator': lambda x: x > 0 and x % 8 == 0  # Hidden size should be divisible by 8
            },
            'num_attention_heads': {
                'type': int,
                'required': True,
                'validator': lambda x: x > 0
            },
            'num_hidden_layers': {
                'type': int,
                'required': True,
                'validator': lambda x: x > 0
            },
            'max_position_embeddings': {
                'type': int,
                'required': True,
                'validator': lambda x: x > 0
            },
            'rope_theta': {
                'type': (int, float),
                'required': True,
                'validator': lambda x: x > 0
            },
            'intermediate_size': {
                'type': int,
                'required': True,
                'validator': lambda x: x > 0
            },
            'vocab_size': {
                'type': int,
                'required': True,
                'validator': lambda x: x > 0
            },
            'layer_norm_eps': {
                'type': (int, float),
                'required': True,
                'validator': lambda x: 0 < x < 1
            },
            'attention_dropout_prob': {
                'type': (int, float),
                'required': True,
                'validator': lambda x: 0 <= x <= 1
            },
            'hidden_dropout_prob': {
                'type': (int, float),
                'required': True,
                'validator': lambda x: 0 <= x <= 1
            },
            'num_key_value_heads': {
                'type': int,
                'required': True,
                'validator': lambda x: x > 0
            },
            'initializer_range': {
                'type': (int, float),
                'required': True,
                'validator': lambda x: x > 0
            },
            
            # Memory optimization settings
            'gradient_checkpointing': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'use_cache': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'torch_dtype': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['float16', 'float32', 'bfloat16', 'int8', 'int4']
            },
            'device_map': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['auto', 'balanced', 'balanced_low_0', 'sequential']
            },
            'low_cpu_mem_usage': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            
            # Generation parameters
            'temperature': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 2.0
            },
            'top_p': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            'top_k': {
                'type': int,
                'required': False,
                'validator': lambda x: x >= 0
            },
            'repetition_penalty': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: x > 0
            },
            'max_new_tokens': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'do_sample': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            
            # Optimization flags
            'use_flash_attention_2': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'use_sparse_attention': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'sparse_attention_pattern': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['longformer', 'bigbird', 'block_sparse', 'local', 'random', 'strided']
            },
            'sparse_attention_sparsity_ratio': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            'sparse_attention_block_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'sparse_attention_local_window_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'use_global_attention': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'global_attention_indices': {
                'type': list,
                'required': False,
                'validator': lambda x: all(isinstance(i, int) and i >= 0 for i in x) if x else True
            },
            'use_multi_pattern_attention': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'use_sparse_attention_with_fallback': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'use_multi_query_attention': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'use_grouped_query_attention': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'use_paged_attention': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'paged_attention_page_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'use_sliding_window_attention': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'sliding_window_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'attention_type': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['mha', 'gqa', 'mqa']
            },
            'num_key_value_groups': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'use_fused_layer_norm': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'use_bias_removal_optimization': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'use_tensor_parallelism': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'tensor_parallel_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            
            # KV-cache compression settings
            'use_kv_cache_compression': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'kv_cache_compression_method': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['quantization', 'low_rank', 'adaptive_precision', 'sparse', 'combined']
            },
            'kv_cache_quantization_bits': {
                'type': int,
                'required': False,
                'validator': lambda x: x in [2, 4, 8]
            },
            'kv_cache_low_rank_dimension': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'kv_cache_adaptive_precision_threshold': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x < 1
            },
            'kv_cache_sparse_compression_ratio': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            'kv_cache_enable_dynamic_compression': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            
            # Memory management settings
            'enable_memory_management': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'max_memory_ratio': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            'page_size_mb': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'eviction_policy': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['lru', 'fifo', 'priority', 'predictive', 'intelligent']
            },
            'enable_tensor_paging': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'enable_smart_swap': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'tensor_paging_priority': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['low', 'medium', 'high', 'critical']
            },
            'memory_cleanup_interval': {
                'type': int,
                'required': False,
                'validator': lambda x: x >= 0
            },
            
            # Predictive management settings
            'enable_predictive_management': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'prediction_horizon_seconds': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'proactive_management_interval': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: x > 0
            },
            'memory_prediction_threshold': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            
            # Disk offloading settings
            'enable_disk_offloading': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'offloading_priority': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['low', 'medium', 'high', 'critical']
            },
            'enable_predictive_offloading': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'proactive_offloading_interval': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: x > 0
            },
            
            # Kernel fusion settings
            'enable_kernel_fusion': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'kernel_fusion_patterns': {
                'type': list,
                'required': False,
                'validator': lambda x: all(isinstance(p, str) for p in x) if x else True
            },
            'use_custom_cuda_kernels': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'custom_kernel_fallback_enabled': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'kernel_fusion_verbose': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            
            # Activation offloading settings
            'enable_activation_offloading': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'activation_max_memory_ratio': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            'activation_page_size_mb': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'activation_eviction_policy': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['lru', 'fifo', 'priority', 'predictive', 'intelligent']
            },
            'activation_offloading_priority': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['low', 'medium', 'high', 'critical']
            },
            'enable_predictive_activation_offloading': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'proactive_activation_offloading_interval': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: x > 0
            },
            
            # Adaptive batching settings
            'enable_adaptive_batching': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'initial_batch_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'min_batch_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'max_batch_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'memory_threshold_ratio': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            'performance_window_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'batch_adjustment_factor': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            'batch_cooldown_period': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: x > 0
            },
            'performance_target': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 <= x <= 1
            },
            
            # Tensor compression settings
            'enable_tensor_compression': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'tensor_compression_method': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['incremental_pca', 'svd', 'auto', 'cp_decomposition', 'tucker_decomposition', 'tensor_train']
            },
            'tensor_compression_ratio': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            'tensor_compression_max_components': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'compression_memory_threshold_high': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            'compression_memory_threshold_critical': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            'enable_adaptive_compression': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'enable_activation_compression': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'compression_update_frequency': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            
            # Tensor decomposition settings
            'use_tensor_decomposition': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'tensor_decomposition_method': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['cp_decomposition', 'tucker_decomposition', 'tensor_train', 'matrix_svd']
            },
            'tensor_decomposition_rank_ratio': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 < x <= 1
            },
            
            # Structured pruning settings
            'use_structured_pruning': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'pruning_ratio': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: 0 <= x < 1
            },
            'pruning_method': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['layer_removal', 'block_removal', 'head_removal', 'mlp_removal', 'adaptive_pruning']
            },
            'pruning_block_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            
            # Quantization settings
            'use_quantization': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'quantization_scheme': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['int8', 'int4', 'fp8', 'nf4', 'fp4']
            },
            'quantization_bits': {
                'type': int,
                'required': False,
                'validator': lambda x: x in [2, 4, 8]
            },
            'quantization_symmetric': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'quantization_per_channel': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            
            # Sequence parallelism settings
            'enable_sequence_parallelism': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'sequence_parallel_num_segments': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'sequence_parallel_split_method': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['chunk', 'stride', 'block']
            },
            'sequence_parallel_enable_overlap': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'sequence_parallel_overlap_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'sequence_parallel_algorithm': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['1d', '2d', 'ring']
            },
            
            # Async processing settings
            'enable_async_unimodal_processing': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'async_max_concurrent_requests': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'async_buffer_size': {
                'type': int,
                'required': False,
                'validator': lambda x: x > 0
            },
            'async_batch_timeout': {
                'type': (int, float),
                'required': False,
                'validator': lambda x: x > 0
            },
            'enable_async_batching': {
                'type': bool,
                'required': False,
                'validator': lambda x: isinstance(x, bool)
            },
            'async_processing_device': {
                'type': str,
                'required': False,
                'validator': lambda x: x in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps']
            },
        }
    
    def validate_config(self, config: DynamicConfig) -> tuple[bool, List[str]]:
        """
        Validate a configuration object.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Get all fields in the config
        fields = config.get_field_names()
        
        # Validate each field
        for field_name in fields:
            field_value = getattr(config, field_name, None)
            
            # Skip validation if field doesn't have rules
            if field_name not in self.validation_rules:
                continue
            
            rule = self.validation_rules[field_name]
            
            # Check if required field is missing
            if rule['required'] and field_value is None:
                errors.append(f"Required field '{field_name}' is missing")
                continue
            
            # Skip validation if field is None and not required
            if field_value is None:
                continue
            
            # Check type
            if not isinstance(field_value, rule['type']):
                errors.append(f"Field '{field_name}' has invalid type. Expected {rule['type']}, got {type(field_value)}")
                continue
            
            # Run custom validator
            try:
                if not rule['validator'](field_value):
                    errors.append(f"Field '{field_name}' failed validation: {field_value}")
            except Exception as e:
                errors.append(f"Field '{field_name}' validation raised exception: {e}")
        
        return len(errors) == 0, errors
    
    def validate_config_dict(self, config_dict: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for field_name, field_value in config_dict.items():
            # Skip validation if field doesn't have rules
            if field_name not in self.validation_rules:
                continue
            
            rule = self.validation_rules[field_name]
            
            # Check if required field is missing
            if rule['required'] and field_value is None:
                errors.append(f"Required field '{field_name}' is missing")
                continue
            
            # Skip validation if field is None and not required
            if field_value is None:
                continue
            
            # Check type
            if not isinstance(field_value, rule['type']):
                errors.append(f"Field '{field_name}' has invalid type. Expected {rule['type']}, got {type(field_value)}")
                continue
            
            # Run custom validator
            try:
                if not rule['validator'](field_value):
                    errors.append(f"Field '{field_name}' failed validation: {field_value}")
            except Exception as e:
                errors.append(f"Field '{field_name}' validation raised exception: {e}")
        
        return len(errors) == 0, errors
    
    def validate_model_config(self, config: DynamicConfig, model_type: str) -> tuple[bool, List[str]]:
        """
        Validate a configuration specifically for a model type.
        
        Args:
            config: Configuration object to validate
            model_type: Type of model ('glm', 'qwen3_4b', 'qwen3_coder', 'qwen3_vl')
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        is_valid, errors = self.validate_config(config)
        
        # Additional model-specific validations
        if model_type == 'glm':
            # GLM-specific validations
            if hasattr(config, 'model_name') and 'glm' not in config.model_name.lower():
                errors.append("Model name should contain 'glm' for GLM models")
        elif model_type == 'qwen3_4b':
            # Qwen3-4B-specific validations
            if hasattr(config, 'model_name') and 'qwen3' not in config.model_name.lower():
                errors.append("Model name should contain 'qwen3' for Qwen3 models")
        elif model_type == 'qwen3_coder':
            # Qwen3-Coder-specific validations
            if hasattr(config, 'model_name') and 'qwen3' not in config.model_name.lower():
                errors.append("Model name should contain 'qwen3' for Qwen3 models")
            if hasattr(config, 'model_name') and 'coder' not in config.model_name.lower():
                errors.append("Model name should contain 'coder' for Qwen3-Coder models")
        elif model_type == 'qwen3_vl':
            # Qwen3-VL-specific validations
            if hasattr(config, 'model_name') and 'qwen3' not in config.model_name.lower():
                errors.append("Model name should contain 'qwen3' for Qwen3 models")
            if hasattr(config, 'model_name') and ('vl' not in config.model_name.lower() and 'vision' not in config.model_name.lower()):
                errors.append("Model name should contain 'vl' or 'vision' for Qwen3-VL models")
        
        return len(errors) == 0, errors


# Global configuration validator instance
config_validator = ConfigValidator()


def get_config_validator() -> ConfigValidator:
    """
    Get the global configuration validator instance.
    
    Returns:
        ConfigValidator instance
    """
    return config_validator


__all__ = [
    "ConfigValidator",
    "get_config_validator",
    "config_validator",
]
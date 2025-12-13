"""
Configuration Parameter Validation Test Suite
This module implements comprehensive tests for configuration validation throughout the Qwen3-VL codebase.
Tests ensure that configuration values are within acceptable ranges, properly typed,
and consistent across the system.
"""

import unittest
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import math
from dataclasses import asdict, fields
import tempfile
import os
from pathlib import Path
import json
import yaml

from src.qwen3_vl.core.config import Qwen3VLConfig, BaseConfig, MemoryConfig, CPUConfig, GPUConfig, PowerManagementConfig, OptimizationConfig


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors"""
    pass


class ConfigValidator:
    """Validates configuration parameters and their consistency"""
    
    @staticmethod
    def validate_config_types(config: Any) -> List[str]:
        """Validate that configuration parameters have proper types"""
        errors = []
        
        # Check if config is a dataclass or has a to_dict method
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            errors.append("Configuration object must be a dataclass or have a to_dict method")
            return errors
        
        # Define expected types for key configuration parameters
        type_mapping = {
            'num_hidden_layers': int,
            'num_attention_heads': int,
            'hidden_size': int,
            'intermediate_size': int,
            'vocab_size': int,
            'max_position_embeddings': int,
            'vision_num_hidden_layers': int,
            'vision_num_attention_heads': int,
            'vision_hidden_size': int,
            'attention_dropout_prob': float,
            'hidden_dropout_prob': float,
            'layer_norm_eps': float,
            'rope_theta': float,
            'sparsity_ratio': float,
            'compression_ratio': float,
            'exit_threshold': float,
            'memory_size_gb': float,
            'use_sparsity': bool,
            'use_moe': bool,
            'use_flash_attention_2': bool,
            'use_gradient_checkpointing': bool,
            'use_adaptive_depth': bool,
            'use_context_adaptive_positional_encoding': bool,
            'use_conditional_feature_extraction': bool,
            'use_memory_pooling': bool,
            'use_hierarchical_memory_compression': bool,
            'use_memory_efficient_attention': bool,
            'use_cross_modal_compression': bool,
            'use_cross_layer_memory_sharing': bool,
            'use_hierarchical_vision': bool,
            'use_learned_activation_routing': bool,
            'use_adaptive_batch_processing': bool,
            'use_adaptive_sequence_packing': bool,
            'use_memory_efficient_grad_accumulation': bool,
            'use_faster_rotary_embeddings': bool,
            'use_distributed_pipeline_parallelism': bool,
            'use_hardware_specific_kernels': bool,
            'torch_dtype': str,
            'attention_implementation': str,
            'kv_cache_strategy': str,
            'hardware_target': str,
            'target_hardware': str,
        }
        
        for param_name, expected_type in type_mapping.items():
            if param_name in config_dict:
                value = config_dict[param_name]
                if not isinstance(value, expected_type):
                    errors.append(f"Parameter '{param_name}' has type {type(value).__name__}, expected {expected_type.__name__}")
        
        return errors
    
    @staticmethod
    def validate_config_ranges(config: Any) -> List[str]:
        """Validate that configuration parameters are within acceptable ranges"""
        errors = []
        
        # Check if config is a dataclass or has a to_dict method
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            errors.append("Configuration object must be a dataclass or have a to_dict method")
            return errors
        
        # Validate numerical ranges
        if 'num_hidden_layers' in config_dict:
            num_layers = config_dict['num_hidden_layers']
            if num_layers <= 0 or num_layers > 1000:
                errors.append(f"num_hidden_layers must be between 1 and 1000, got {num_layers}")
        
        if 'num_attention_heads' in config_dict:
            num_heads = config_dict['num_attention_heads']
            if num_heads <= 0 or num_heads > 128:
                errors.append(f"num_attention_heads must be between 1 and 128, got {num_heads}")
        
        if 'hidden_size' in config_dict:
            hidden_size = config_dict['hidden_size']
            if hidden_size <= 0 or hidden_size > 8192:
                errors.append(f"hidden_size must be between 1 and 8192, got {hidden_size}")
            
            # Check if hidden_size is divisible by num_attention_heads
            if 'num_attention_heads' in config_dict:
                num_heads = config_dict['num_attention_heads']
                if hidden_size % num_heads != 0:
                    errors.append(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_heads})")
        
        if 'intermediate_size' in config_dict:
            intermediate_size = config_dict['intermediate_size']
            if intermediate_size <= 0 or intermediate_size > 32768:
                errors.append(f"intermediate_size must be between 1 and 32768, got {intermediate_size}")
        
        if 'vocab_size' in config_dict:
            vocab_size = config_dict['vocab_size']
            if vocab_size <= 0 or vocab_size > 500000:
                errors.append(f"vocab_size must be between 1 and 500000, got {vocab_size}")
        
        if 'max_position_embeddings' in config_dict:
            max_pos = config_dict['max_position_embeddings']
            if max_pos <= 0 or max_pos > 1000000:
                errors.append(f"max_position_embeddings must be between 1 and 1000000, got {max_pos}")
        
        # Validate vision parameters
        if 'vision_num_hidden_layers' in config_dict:
            vision_num_layers = config_dict['vision_num_hidden_layers']
            if vision_num_layers <= 0 or vision_num_layers > 100:
                errors.append(f"vision_num_hidden_layers must be between 1 and 100, got {vision_num_layers}")
        
        if 'vision_num_attention_heads' in config_dict:
            vision_num_heads = config_dict['vision_num_attention_heads']
            if vision_num_heads <= 0 or vision_num_heads > 128:
                errors.append(f"vision_num_attention_heads must be between 1 and 128, got {vision_num_heads}")
        
        if 'vision_hidden_size' in config_dict:
            vision_hidden_size = config_dict['vision_hidden_size']
            if vision_hidden_size <= 0 or vision_hidden_size > 8192:
                errors.append(f"vision_hidden_size must be between 1 and 8192, got {vision_hidden_size}")
            
            # Check if vision_hidden_size is divisible by vision_num_attention_heads
            if 'vision_num_attention_heads' in config_dict:
                vision_num_heads = config_dict['vision_num_attention_heads']
                if vision_hidden_size % vision_num_heads != 0:
                    errors.append(f"vision_hidden_size ({vision_hidden_size}) must be divisible by vision_num_attention_heads ({vision_num_heads})")
        
        # Validate floating-point ranges
        if 'attention_dropout_prob' in config_dict:
            dropout_prob = config_dict['attention_dropout_prob']
            if not 0.0 <= dropout_prob <= 1.0:
                errors.append(f"attention_dropout_prob must be between 0.0 and 1.0, got {dropout_prob}")
        
        if 'hidden_dropout_prob' in config_dict:
            hidden_dropout_prob = config_dict['hidden_dropout_prob']
            if not 0.0 <= hidden_dropout_prob <= 1.0:
                errors.append(f"hidden_dropout_prob must be between 0.0 and 1.0, got {hidden_dropout_prob}")
        
        if 'layer_norm_eps' in config_dict:
            layer_norm_eps = config_dict['layer_norm_eps']
            if not 1e-12 <= layer_norm_eps <= 1e-3:
                errors.append(f"layer_norm_eps must be between 1e-12 and 1e-3, got {layer_norm_eps}")
        
        if 'rope_theta' in config_dict:
            rope_theta = config_dict['rope_theta']
            if not 1000.0 <= rope_theta <= 10000000.0:
                errors.append(f"rope_theta must be between 1000.0 and 10000000.0, got {rope_theta}")
        
        if 'sparsity_ratio' in config_dict:
            sparsity_ratio = config_dict['sparsity_ratio']
            if not 0.0 <= sparsity_ratio <= 1.0:
                errors.append(f"sparsity_ratio must be between 0.0 and 1.0, got {sparsity_ratio}")
        
        if 'compression_ratio' in config_dict:
            compression_ratio = config_dict['compression_ratio']
            if not 0.0 <= compression_ratio <= 1.0:
                errors.append(f"compression_ratio must be between 0.0 and 1.0, got {compression_ratio}")
        
        if 'exit_threshold' in config_dict:
            exit_threshold = config_dict['exit_threshold']
            if not 0.0 <= exit_threshold <= 1.0:
                errors.append(f"exit_threshold must be between 0.0 and 1.0, got {exit_threshold}")
        
        if 'memory_size_gb' in config_dict:
            memory_size_gb = config_dict['memory_size_gb']
            if memory_size_gb <= 0 or memory_size_gb > 1000:
                errors.append(f"memory_size_gb must be between 1 and 1000, got {memory_size_gb}")
        
        # Validate optimization parameters
        if 'moe_num_experts' in config_dict:
            moe_num_experts = config_dict['moe_num_experts']
            if moe_num_experts < 2 or moe_num_experts > 100:
                errors.append(f"moe_num_experts must be between 2 and 100, got {moe_num_experts}")
        
        if 'moe_top_k' in config_dict:
            moe_top_k = config_dict['moe_top_k']
            moe_num_experts = config_dict.get('moe_num_experts', 4)
            if moe_top_k < 1 or moe_top_k > moe_num_experts:
                errors.append(f"moe_top_k must be between 1 and moe_num_experts ({moe_num_experts}), got {moe_top_k}")
        
        return errors
    
    @staticmethod
    def validate_config_consistency(config: Any) -> List[str]:
        """Validate consistency between related configuration parameters"""
        errors = []
        
        # Check if config is a dataclass or has a to_dict method
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            errors.append("Configuration object must be a dataclass or have a to_dict method")
            return errors
        
        # Validate that the model preserves full capacity (32 layers, 32 heads)
        if hasattr(config, 'num_hidden_layers') and config.num_hidden_layers != 32:
            errors.append(f"num_hidden_layers should be 32 to preserve full capacity, got {config.num_hidden_layers}")
        
        if hasattr(config, 'num_attention_heads') and config.num_attention_heads != 32:
            errors.append(f"num_attention_heads should be 32 to preserve full capacity, got {config.num_attention_heads}")
        
        # Validate attention head dimensions consistency
        if 'hidden_size' in config_dict and 'num_attention_heads' in config_dict:
            hidden_size = config_dict['hidden_size']
            num_heads = config_dict['num_attention_heads']
            if num_heads > 0 and hidden_size % num_heads != 0:
                errors.append(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_heads})")
        
        # Validate vision attention head dimensions consistency
        if 'vision_hidden_size' in config_dict and 'vision_num_attention_heads' in config_dict:
            vision_hidden_size = config_dict['vision_hidden_size']
            vision_num_heads = config_dict['vision_num_attention_heads']
            if vision_num_heads > 0 and vision_hidden_size % vision_num_heads != 0:
                errors.append(f"vision_hidden_size ({vision_hidden_size}) must be divisible by vision_num_attention_heads ({vision_num_heads})")
        
        # Validate that sparsity ratio is consistent with use_sparsity flag
        if 'use_sparsity' in config_dict and 'sparsity_ratio' in config_dict:
            if config_dict['use_sparsity'] and not (0.0 < config_dict['sparsity_ratio'] < 1.0):
                errors.append(f"sparsity_ratio must be between 0 and 1 when use_sparsity is True, got {config_dict['sparsity_ratio']}")
        
        # Validate that MoE parameters are consistent with use_moe flag
        if 'use_moe' in config_dict and config_dict['use_moe']:
            if 'moe_num_experts' not in config_dict or config_dict['moe_num_experts'] < 2:
                errors.append("moe_num_experts must be at least 2 when use_moe is True")
            if 'moe_top_k' not in config_dict or config_dict['moe_top_k'] < 1:
                errors.append("moe_top_k must be at least 1 when use_moe is True")
        
        # Validate KV cache strategy consistency
        if 'kv_cache_strategy' in config_dict:
            valid_strategies = ["standard", "low_rank", "sliding_window", "hybrid"]
            if config_dict['kv_cache_strategy'] not in valid_strategies:
                errors.append(f"kv_cache_strategy must be one of {valid_strategies}, got {config_dict['kv_cache_strategy']}")
        
        # Validate attention implementation consistency
        if 'attention_implementation' in config_dict:
            valid_implementations = ["eager", "flash_attention_2", "sdpa", "kv_cache_optimized", "sparse_attention"]
            if config_dict['attention_implementation'] not in valid_implementations:
                errors.append(f"attention_implementation must be one of {valid_implementations}, got {config_dict['attention_implementation']}")
        
        return errors


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation throughout the codebase"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = Qwen3VLConfig()
        self.validator = ConfigValidator()
    
    def test_config_types_validation(self):
        """Test that configuration parameters have proper types"""
        errors = self.validator.validate_config_types(self.config)
        self.assertEqual(len(errors), 0, f"Type validation failed with errors: {errors}")
        
        # Verify a few key types
        self.assertIsInstance(self.config.num_hidden_layers, int)
        self.assertIsInstance(self.config.num_attention_heads, int)
        self.assertIsInstance(self.config.hidden_size, int)
        self.assertIsInstance(self.config.attention_dropout_prob, float)
        self.assertIsInstance(self.config.use_sparsity, bool)
        self.assertIsInstance(self.config.torch_dtype, str)
    
    def test_config_ranges_validation(self):
        """Test that configuration parameters are within acceptable ranges"""
        errors = self.validator.validate_config_ranges(self.config)
        self.assertEqual(len(errors), 0, f"Range validation failed with errors: {errors}")
        
        # Test with invalid values
        invalid_config = Qwen3VLConfig(num_hidden_layers=-1)
        errors = self.validator.validate_config_ranges(invalid_config)
        self.assertGreater(len(errors), 0)
        
        invalid_config = Qwen3VLConfig(hidden_size=100, num_attention_heads=7)  # Not divisible
        errors = self.validator.validate_config_ranges(invalid_config)
        self.assertGreater(len(errors), 0)
    
    def test_config_consistency_validation(self):
        """Test consistency between related configuration parameters"""
        errors = self.validator.validate_config_consistency(self.config)
        self.assertEqual(len(errors), 0, f"Consistency validation failed with errors: {errors}")
    
    def test_memory_config_validation(self):
        """Test memory configuration validation"""
        memory_config = MemoryConfig()
        memory_errors = []
        
        # Validate types
        memory_errors.extend(self.validator.validate_config_types(memory_config))
        
        # Validate ranges
        memory_errors.extend(self.validator.validate_config_ranges(memory_config))
        
        # Validate consistency
        memory_errors.extend(self.validator.validate_config_consistency(memory_config))
        
        self.assertEqual(len(memory_errors), 0, f"Memory config validation failed with errors: {memory_errors}")
    
    def test_cpu_config_validation(self):
        """Test CPU configuration validation"""
        cpu_config = CPUConfig()
        cpu_errors = []
        
        # Validate types
        cpu_errors.extend(self.validator.validate_config_types(cpu_config))
        
        # Validate ranges
        cpu_errors.extend(self.validator.validate_config_ranges(cpu_config))
        
        # Validate consistency
        cpu_errors.extend(self.validator.validate_config_consistency(cpu_config))
        
        self.assertEqual(len(cpu_errors), 0, f"CPU config validation failed with errors: {cpu_errors}")
    
    def test_gpu_config_validation(self):
        """Test GPU configuration validation"""
        gpu_config = GPUConfig()
        gpu_errors = []
        
        # Validate types
        gpu_errors.extend(self.validator.validate_config_types(gpu_config))
        
        # Validate ranges
        gpu_errors.extend(self.validator.validate_config_ranges(gpu_config))
        
        # Validate consistency
        gpu_errors.extend(self.validator.validate_config_consistency(gpu_config))
        
        self.assertEqual(len(gpu_errors), 0, f"GPU config validation failed with errors: {gpu_errors}")
    
    def test_power_config_validation(self):
        """Test power management configuration validation"""
        power_config = PowerManagementConfig()
        power_errors = []
        
        # Validate types
        power_errors.extend(self.validator.validate_config_types(power_config))
        
        # Validate ranges
        power_errors.extend(self.validator.validate_config_ranges(power_config))
        
        # Validate consistency
        power_errors.extend(self.validator.validate_config_consistency(power_config))
        
        self.assertEqual(len(power_errors), 0, f"Power config validation failed with errors: {power_errors}")
    
    def test_optimization_config_validation(self):
        """Test optimization configuration validation"""
        opt_config = OptimizationConfig()
        opt_errors = []
        
        # Validate types
        opt_errors.extend(self.validator.validate_config_types(opt_config))
        
        # Validate ranges
        opt_errors.extend(self.validator.validate_config_ranges(opt_config))
        
        # Validate consistency
        opt_errors.extend(self.validator.validate_config_consistency(opt_config))
        
        self.assertEqual(len(opt_errors), 0, f"Optimization config validation failed with errors: {opt_errors}")
    
    def test_config_file_validation(self):
        """Test configuration validation from file"""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(asdict(self.config), f)
            temp_file = f.name
        
        try:
            # Load config from file
            loaded_config = Qwen3VLConfig.from_file(temp_file)
            
            # Validate loaded config
            errors = self.validator.validate_config_types(loaded_config)
            self.assertEqual(len(errors), 0, f"Loaded config type validation failed: {errors}")
            
            errors = self.validator.validate_config_ranges(loaded_config)
            self.assertEqual(len(errors), 0, f"Loaded config range validation failed: {errors}")
            
            errors = self.validator.validate_config_consistency(loaded_config)
            self.assertEqual(len(errors), 0, f"Loaded config consistency validation failed: {errors}")
        finally:
            os.unlink(temp_file)
    
    def test_config_serialization_validation(self):
        """Test configuration validation after serialization/deserialization"""
        # Serialize config
        config_dict = self.config.to_dict()
        
        # Create new config from dict
        new_config = Qwen3VLConfig.from_dict(config_dict)
        
        # Validate new config
        errors = self.validator.validate_config_types(new_config)
        self.assertEqual(len(errors), 0, f"Serialized config type validation failed: {errors}")
        
        errors = self.validator.validate_config_ranges(new_config)
        self.assertEqual(len(errors), 0, f"Serialized config range validation failed: {errors}")
        
        errors = self.validator.validate_config_consistency(new_config)
        self.assertEqual(len(errors), 0, f"Serialized config consistency validation failed: {errors}")
    
    def test_model_config_preserves_capacity(self):
        """Test that model configuration preserves full capacity"""
        # Ensure the config has the required capacity
        self.assertEqual(self.config.num_hidden_layers, 32, 
                         f"num_hidden_layers should be 32 to preserve full capacity, got {self.config.num_hidden_layers}")
        self.assertEqual(self.config.num_attention_heads, 32, 
                         f"num_attention_heads should be 32 to preserve full capacity, got {self.config.num_attention_heads}")
        
        # Validate with custom config that also preserves capacity
        custom_config = Qwen3VLConfig(
            hidden_size=2048,
            intermediate_size=5504,
            vocab_size=152064,
            max_position_embeddings=32768
        )
        errors = self.validator.validate_config_consistency(custom_config)
        self.assertEqual(len(errors), 0, f"Custom config consistency validation failed: {errors}")
    
    def test_config_with_optimizations_validation(self):
        """Test configuration with various optimizations enabled"""
        # Create config with different optimizations enabled
        config_with_optimizations = Qwen3VLConfig(
            use_sparsity=True,
            sparsity_ratio=0.5,
            use_moe=True,
            moe_num_experts=4,
            moe_top_k=2,
            use_flash_attention_2=True,
            use_gradient_checkpointing=True,
            use_adaptive_depth=True,
            exit_threshold=0.8
        )
        
        # Validate the config
        errors = self.validator.validate_config_types(config_with_optimizations)
        self.assertEqual(len(errors), 0, f"Config with optimizations type validation failed: {errors}")
        
        errors = self.validator.validate_config_ranges(config_with_optimizations)
        self.assertEqual(len(errors), 0, f"Config with optimizations range validation failed: {errors}")
        
        errors = self.validator.validate_config_consistency(config_with_optimizations)
        self.assertEqual(len(errors), 0, f"Config with optimizations consistency validation failed: {errors}")
    
    def test_config_with_invalid_values(self):
        """Test that validation catches invalid configuration values"""
        # Create config with invalid values
        invalid_configs = [
            # Invalid hidden size not divisible by num attention heads
            Qwen3VLConfig(hidden_size=100, num_attention_heads=7),
            
            # Invalid dropout probabilities
            Qwen3VLConfig(attention_dropout_prob=-0.1),
            Qwen3VLConfig(attention_dropout_prob=1.5),
            
            # Invalid layer norm epsilon
            Qwen3VLConfig(layer_norm_eps=0.0),
            Qwen3VLConfig(layer_norm_eps=1.0),
            
            # Invalid sparsity ratio
            Qwen3VLConfig(use_sparsity=True, sparsity_ratio=1.5),
            Qwen3VLConfig(use_sparsity=True, sparsity_ratio=-0.1),
            
            # Invalid MoE configuration
            Qwen3VLConfig(use_moe=True, moe_num_experts=1),  # Too few experts
            Qwen3VLConfig(use_moe=True, moe_top_k=0),  # Invalid top_k
        ]
        
        for i, invalid_config in enumerate(invalid_configs):
            with self.subTest(config_index=i):
                try:
                    # This should raise an error during __post_init__
                    invalid_config.__post_init__()
                    self.fail(f"Config with invalid values should have failed validation: {invalid_config}")
                except (ValueError, AttributeError):
                    # Expected to fail
                    pass
    
    def test_nested_config_validation(self):
        """Test validation of nested configuration objects"""
        # Create config with nested objects
        nested_config = Qwen3VLConfig()
        
        # Validate the main config and nested configs
        main_errors = self.validator.validate_config_consistency(nested_config)
        self.assertEqual(len(main_errors), 0, f"Main config consistency validation failed: {main_errors}")
        
        # Validate nested memory config
        memory_errors = self.validator.validate_config_consistency(nested_config.memory_config)
        self.assertEqual(len(memory_errors), 0, f"Memory config consistency validation failed: {memory_errors}")
        
        # Validate nested CPU config
        cpu_errors = self.validator.validate_config_consistency(nested_config.cpu_config)
        self.assertEqual(len(cpu_errors), 0, f"CPU config consistency validation failed: {cpu_errors}")
        
        # Validate nested GPU config
        gpu_errors = self.validator.validate_config_consistency(nested_config.gpu_config)
        self.assertEqual(len(gpu_errors), 0, f"GPU config consistency validation failed: {gpu_errors}")
        
        # Validate nested power config
        power_errors = self.validator.validate_config_consistency(nested_config.power_config)
        self.assertEqual(len(power_errors), 0, f"Power config consistency validation failed: {power_errors}")
        
        # Validate nested optimization config
        opt_errors = self.validator.validate_config_consistency(nested_config.optimization_config)
        self.assertEqual(len(opt_errors), 0, f"Optimization config consistency validation failed: {opt_errors}")


class ConfigValidationIntegrationTest(unittest.TestCase):
    """Integration tests for configuration validation"""
    
    def test_config_validation_in_model_creation(self):
        """Test that configuration validation is applied during model creation"""
        # Create a valid config
        config = Qwen3VLConfig()
        
        # Verify that the config passes validation
        errors = ConfigValidator.validate_config_consistency(config)
        self.assertEqual(len(errors), 0, f"Default config validation failed: {errors}")
        
        # Try creating a model with the config (this should work)
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration(config)
        
        # Verify the model has the correct configuration
        self.assertEqual(model.config.num_hidden_layers, 32)
        self.assertEqual(model.config.num_attention_heads, 32)
    
    def test_config_validation_with_different_optimization_levels(self):
        """Test configuration validation across different optimization levels"""
        # Test different optimization level configs
        configs_to_test = [
            # Balanced config
            Qwen3VLConfig(
                use_memory_pooling=True,
                use_hierarchical_memory_compression=True,
                use_memory_efficient_attention=True,
                use_sparsity=True,
                sparsity_ratio=0.3,
                use_moe=False,
                use_flash_attention_2=True,
                use_adaptive_depth=True,
                use_gradient_checkpointing=True
            ),
            
            # Memory efficient config
            Qwen3VLConfig(
                use_memory_pooling=True,
                use_hierarchical_memory_compression=True,
                use_memory_efficient_attention=True,
                use_sparsity=True,
                sparsity_ratio=0.5,
                use_moe=True,
                moe_num_experts=2,
                moe_top_k=1,
                use_flash_attention_2=True,
                use_adaptive_depth=True,
                use_gradient_checkpointing=True
            ),
            
            # Performance optimized config
            Qwen3VLConfig(
                use_memory_pooling=False,
                use_hierarchical_memory_compression=False,
                use_memory_efficient_attention=False,
                use_sparsity=False,
                use_moe=False,
                use_flash_attention_2=True,
                use_adaptive_depth=False,
                use_gradient_checkpointing=False
            )
        ]
        
        for i, test_config in enumerate(configs_to_test):
            with self.subTest(config_index=i):
                # Validate config consistency
                errors = ConfigValidator.validate_config_consistency(test_config)
                self.assertEqual(len(errors), 0, f"Config {i} validation failed: {errors}")
                
                # Validate config ranges
                errors = ConfigValidator.validate_config_ranges(test_config)
                self.assertEqual(len(errors), 0, f"Config {i} range validation failed: {errors}")
                
                # Validate config types
                errors = ConfigValidator.validate_config_types(test_config)
                self.assertEqual(len(errors), 0, f"Config {i} type validation failed: {errors}")


def run_config_validation_tests():
    """Run the configuration validation test suite"""
    print("Running Configuration Validation Test Suite...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestConfigValidation))
    suite.addTest(unittest.makeSuite(ConfigValidationIntegrationTest))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nConfiguration Validation Test Results:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        print("  Failures:")
        for test, traceback in result.failures:
            print(f"    - {test}: {traceback}")
    
    if result.errors:
        print("  Errors:")
        for test, traceback in result.errors:
            print(f"    - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_config_validation_tests()
    if success:
        print("\n✅ All configuration validation tests passed!")
    else:
        print("\n❌ Some configuration validation tests failed!")
        exit(1)
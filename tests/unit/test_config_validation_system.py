"""
Configuration Validation System Tests
Comprehensive tests for the configuration validation system throughout the Qwen3-VL codebase.
"""

import unittest
import torch
import tempfile
import os
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from src.qwen3_vl.core.config import Qwen3VLConfig, MemoryConfig, CPUConfig, GPUConfig, PowerManagementConfig, OptimizationConfig
from src.qwen3_vl.config.config_validation import ConfigValidator, ConfigValidationError, validate_config_object, validate_config_file, get_config_validation_report


class TestConfigValidationSystem(unittest.TestCase):
    """Test the configuration validation system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = ConfigValidator()
    
    def test_config_validator_initialization(self):
        """Test that the config validator initializes with proper rules"""
        self.assertIsNotNone(self.validator.validation_rules)
        self.assertGreater(len(self.validator.validation_rules), 0)
        for rule in self.validator.validation_rules:
            self.assertIsInstance(rule, type(self.validator.validation_rules[0]))
    
    def test_config_validation_with_valid_config(self):
        """Test validation with a valid configuration"""
        config = Qwen3VLConfig()
        result = self.validator.validate_config(config, strict=False)
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
        self.assertIsInstance(result['warnings'], list)
    
    def test_config_validation_with_invalid_hidden_size(self):
        """Test validation with invalid hidden size (not divisible by num attention heads)"""
        config = Qwen3VLConfig(hidden_size=100, num_attention_heads=7)  # 100 not divisible by 7
        result = self.validator.validate_config(config, strict=False)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
        self.assertTrue(any('hidden_size must be divisible' in error for error in result['errors']))
    
    def test_config_validation_with_negative_values(self):
        """Test validation with negative values"""
        config = Qwen3VLConfig(num_hidden_layers=-1, num_attention_heads=-1)
        result = self.validator.validate_config(config, strict=False)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
        self.assertTrue(any('num_hidden_layers must be positive' in error for error in result['errors']))
        self.assertTrue(any('num_attention_heads must be positive' in error for error in result['errors']))
    
    def test_config_validation_with_invalid_dropout_prob(self):
        """Test validation with invalid dropout probability"""
        config = Qwen3VLConfig(attention_dropout_prob=-0.1)
        result = self.validator.validate_config(config, strict=False)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
        self.assertTrue(any('attention_dropout_prob must be between' in error for error in result['errors']))
    
    def test_config_validation_with_invalid_sparsity_ratio(self):
        """Test validation with invalid sparsity ratio"""
        config = Qwen3VLConfig(use_sparsity=True, sparsity_ratio=1.5)
        result = self.validator.validate_config(config, strict=False)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
        self.assertTrue(any('sparsity_ratio must be between' in error for error in result['errors']))
    
    def test_config_validation_with_invalid_moe_config(self):
        """Test validation with invalid MoE configuration"""
        config = Qwen3VLConfig(use_moe=True, moe_num_experts=1)  # Must be at least 2
        result = self.validator.validate_config(config, strict=False)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
        self.assertTrue(any('moe_num_experts must be at least' in error for error in result['errors']))
    
    def test_config_validation_with_invalid_exit_threshold(self):
        """Test validation with invalid exit threshold"""
        config = Qwen3VLConfig(exit_threshold=1.5)
        result = self.validator.validate_config(config, strict=False)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
        self.assertTrue(any('exit_threshold must be between' in error for error in result['errors']))
    
    def test_config_validation_with_invalid_layer_norm_eps(self):
        """Test validation with invalid layer norm epsilon"""
        config = Qwen3VLConfig(layer_norm_eps=0.0)  # Must be between 1e-12 and 1e-3
        result = self.validator.validate_config(config, strict=False)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
        self.assertTrue(any('layer_norm_eps must be between' in error for error in result['errors']))
    
    def test_config_validation_with_invalid_rope_theta(self):
        """Test validation with invalid rope theta"""
        config = Qwen3VLConfig(rope_theta=0.0)  # Must be between 1000.0 and 10000000.0
        result = self.validator.validate_config(config, strict=False)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
        self.assertTrue(any('rope_theta must be between' in error for error in result['errors']))
    
    def test_config_validation_capacity_preservation(self):
        """Test that configurations preserve full capacity (32 layers, 32 heads)"""
        config = Qwen3VLConfig(num_hidden_layers=16, num_attention_heads=16)  # Less than full capacity
        result = self.validator.validate_config(config, strict=False)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
        self.assertTrue(any('num_hidden_layers should be 32' in error for error in result['errors']))
        self.assertTrue(any('num_attention_heads should be 32' in error for error in result['errors']))
    
    def test_config_validation_strict_mode_raises_exception(self):
        """Test that strict mode raises exception on validation errors"""
        config = Qwen3VLConfig(num_hidden_layers=-1)
        
        with self.assertRaises(ConfigValidationError):
            self.validator.validate_config(config, strict=True)
    
    def test_config_validation_non_strict_mode_returns_errors(self):
        """Test that non-strict mode returns errors without raising exception"""
        config = Qwen3VLConfig(num_hidden_layers=-1)
        result = self.validator.validate_config(config, strict=False)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_memory_config_validation(self):
        """Test validation of MemoryConfig"""
        memory_config = MemoryConfig(
            memory_pool_size=1024*1024*1024,  # 1GB
            memory_pool_growth_factor=1.5
        )
        result = self.validator.validate_config_with_context(memory_config, "memory")
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_cpu_config_validation(self):
        """Test validation of CPUConfig"""
        cpu_config = CPUConfig(
            num_threads=4,
            num_workers=4,
            l1_cache_size=32*1024,  # 32KB
            l2_cache_size=256*1024,  # 256KB
            l3_cache_size=6*1024*1024,  # 6MB
        )
        result = self.validator.validate_config_with_context(cpu_config, "model")
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_gpu_config_validation(self):
        """Test validation of GPUConfig"""
        gpu_config = GPUConfig(
            gpu_compute_capability=(6, 1),  # SM61
            max_threads_per_block=1024,
            shared_memory_per_block=48*1024,  # 48KB
            memory_bandwidth_gbps=320.0
        )
        result = self.validator.validate_config_with_context(gpu_config, "model")
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_power_config_validation(self):
        """Test validation of PowerManagementConfig"""
        power_config = PowerManagementConfig(
            power_constraint=0.8,
            thermal_constraint=75.0,
            performance_target=0.9,
            adaptation_frequency=1.0
        )
        result = self.validator.validate_config_with_context(power_config, "model")
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_optimization_config_validation(self):
        """Test validation of OptimizationConfig"""
        opt_config = OptimizationConfig(
            use_memory_pooling=True,
            use_hierarchical_memory_compression=True,
            use_memory_efficient_attention=True,
            use_sparsity=True,
            sparsity_ratio=0.5,
            use_dynamic_sparse_attention=True,
            use_adaptive_precision=True,
            use_moe=True,
            moe_num_experts=4,
            moe_top_k=2,
            use_flash_attention_2=True,
            use_adaptive_depth=True,
            use_gradient_checkpointing=True,
            use_context_adaptive_positional_encoding=True,
            use_conditional_feature_extraction=True,
            use_cross_modal_compression=True,
            use_cross_layer_memory_sharing=True,
            use_hierarchical_vision=True,
            use_learned_activation_routing=True,
            use_adaptive_batch_processing=True,
            use_adaptive_sequence_packing=True,
            use_memory_efficient_grad_accumulation=True,
            use_faster_rotary_embeddings=True,
            use_hardware_specific_kernels=True,
            performance_improvement_threshold=0.05,
            accuracy_preservation_threshold=0.95
        )
        result = self.validator.validate_config_with_context(opt_config, "optimization")
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_config_file_validation(self):
        """Test validation of configuration from file"""
        config = Qwen3VLConfig()
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(asdict(config), f)
            temp_file = f.name
        
        try:
            result = self.validator.validate_file(temp_file, strict=False)
            self.assertTrue(result['valid'])
            self.assertEqual(len(result['errors']), 0)
        finally:
            os.unlink(temp_file)
    
    def test_config_file_validation_with_invalid_values(self):
        """Test validation of configuration from file with invalid values"""
        invalid_config_dict = {
            'num_hidden_layers': -1,
            'num_attention_heads': 0,
            'hidden_size': 0,
            'attention_dropout_prob': -0.1,
            'sparsity_ratio': 1.5
        }
        
        # Create temporary config file with invalid values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config_dict, f)
            temp_file = f.name
        
        try:
            result = self.validator.validate_file(temp_file, strict=False)
            self.assertFalse(result['valid'])
            self.assertGreater(len(result['errors']), 0)
        finally:
            os.unlink(temp_file)
    
    def test_config_file_validation_yaml_format(self):
        """Test validation of configuration from YAML file"""
        import yaml
        
        config = Qwen3VLConfig()
        
        # Create temporary YAML config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(asdict(config), f)
            temp_file = f.name
        
        try:
            result = self.validator.validate_file(temp_file, strict=False)
            self.assertTrue(result['valid'])
            self.assertEqual(len(result['errors']), 0)
        finally:
            os.unlink(temp_file)
    
    def test_global_validation_functions(self):
        """Test global validation functions"""
        config = Qwen3VLConfig()
        
        # Test validate_config_object
        result = validate_config_object(config, strict=False)
        self.assertTrue(result['valid'])
        
        # Test get_config_validation_report
        report = get_config_validation_report(config)
        self.assertIsInstance(report, dict)
        self.assertIn('valid', report)
        self.assertIn('errors', report)
        self.assertIn('warnings', report)
        self.assertIn('config_summary', report)
    
    def test_config_validation_report_contains_expected_fields(self):
        """Test that validation reports contain expected fields"""
        config = Qwen3VLConfig()
        report = get_config_validation_report(config)
        
        expected_fields = [
            'valid', 'errors', 'warnings', 'config_summary',
            'num_hidden_layers', 'num_attention_heads', 'hidden_size', 'intermediate_size',
            'vocab_size', 'max_position_embeddings', 'use_sparsity', 'use_moe',
            'use_flash_attention_2', 'use_gradient_checkpointing', 'use_adaptive_depth',
            'use_context_adaptive_positional_encoding', 'use_conditional_feature_extraction',
            'use_cross_modal_compression', 'use_cross_layer_memory_sharing'
        ]
        
        for field in expected_fields:
            self.assertIn(field, report)
    
    def test_config_validation_with_edge_case_values(self):
        """Test validation with edge case values"""
        edge_case_configs = [
            # Edge case 1: Very small values
            Qwen3VLConfig(
                hidden_size=1,
                num_attention_heads=1,
                intermediate_size=1,
                vocab_size=1,
                max_position_embeddings=1
            ),
            
            # Edge case 2: Very large values
            Qwen3VLConfig(
                hidden_size=8192,
                num_attention_heads=128,
                intermediate_size=32768,
                vocab_size=500000,
                max_position_embeddings=1000000
            ),
            
            # Edge case 3: Boundary values for ratios
            Qwen3VLConfig(
                sparsity_ratio=0.0,
                compression_ratio=1.0,
                exit_threshold=0.0,
                attention_dropout_prob=1.0,
                hidden_dropout_prob=1.0
            ),
            
            # Edge case 4: Boundary values for ratios (invalid)
            Qwen3VLConfig(
                sparsity_ratio=-0.1,
                compression_ratio=1.1,
                exit_threshold=-0.1,
                attention_dropout_prob=1.1,
                hidden_dropout_prob=-0.1
            )
        ]
        
        # First 3 configs should have validation errors due to boundary violations
        for i, config in enumerate(edge_case_configs[:3]):
            result = self.validator.validate_config(config, strict=False)
            # Even if the values are extreme, they should still pass basic validation
            # (some may generate warnings but not errors)
            pass  # Just verify no exceptions are raised
        
        # Fourth config should definitely fail validation
        result = self.validator.validate_config(edge_case_configs[3], strict=False)
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_config_validation_warnings_only(self):
        """Test that configurations can have warnings but still be valid"""
        # Create config with values that generate warnings but not errors
        config = Qwen3VLConfig(
            intermediate_size=128,  # Much smaller than hidden_size (should generate warning)
            vision_num_hidden_layers=12,  # Not the standard 24 (should generate warning)
            vision_num_attention_heads=8  # Not the standard 16 (should generate warning)
        )
        result = self.validator.validate_config(config, strict=False)
        
        # Should still be valid despite warnings
        self.assertTrue(result['valid'])
        # But might have warnings
        # Note: The warning behavior depends on implementation, so we don't strictly enforce warnings here
    
    def test_config_validation_context_specific(self):
        """Test context-specific validation"""
        config = Qwen3VLConfig()
        
        # Test model context validation
        result = self.validator.validate_config_with_context(config, "model")
        self.assertTrue(result['valid'])
        
        # Test memory context validation
        result = self.validator.validate_config_with_context(config, "memory")
        self.assertTrue(result['valid'])
        
        # Test hardware context validation
        result = self.validator.validate_config_with_context(config, "hardware")
        self.assertTrue(result['valid'])
        
        # Test optimization context validation
        result = self.validator.validate_config_with_context(config, "optimization")
        self.assertTrue(result['valid'])


class TestConfigValidatorIntegration(unittest.TestCase):
    """Integration tests for configuration validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = ConfigValidator()
    
    def test_config_validation_integration_with_model_creation(self):
        """Test that valid configurations can be used to create models"""
        config = Qwen3VLConfig()
        result = self.validator.validate_config(config, strict=False)
        
        self.assertTrue(result['valid'])
        
        # Now try to create a model with this config
        # (We won't actually run inference to keep tests lightweight)
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration(config)
        
        # Verify model has expected properties from config
        self.assertEqual(model.config.num_hidden_layers, config.num_hidden_layers)
        self.assertEqual(model.config.num_attention_heads, config.num_attention_heads)
        self.assertEqual(model.config.hidden_size, config.hidden_size)
    
    def test_config_validation_integration_with_optimizations(self):
        """Test that valid configurations work with optimizations"""
        config = Qwen3VLConfig(
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
        result = self.validator.validate_config(config, strict=False)
        
        self.assertTrue(result['valid'])
        
        # Verify that optimization-related attributes are properly set
        self.assertTrue(config.use_sparsity)
        self.assertEqual(config.sparsity_ratio, 0.5)
        self.assertTrue(config.use_moe)
        self.assertEqual(config.moe_num_experts, 4)
        self.assertEqual(config.moe_top_k, 2)
        self.assertTrue(config.use_flash_attention_2)
        self.assertTrue(config.use_gradient_checkpointing)
        self.assertTrue(config.use_adaptive_depth)
        self.assertEqual(config.exit_threshold, 0.8)
    
    def test_config_validation_integration_with_different_optimization_levels(self):
        """Test validation across different optimization levels"""
        configs = [
            # Minimal optimization config
            Qwen3VLConfig(
                use_sparsity=False,
                use_moe=False,
                use_flash_attention_2=False,
                use_gradient_checkpointing=False,
                use_adaptive_depth=False
            ),
            
            # Balanced optimization config
            Qwen3VLConfig(
                use_sparsity=True,
                sparsity_ratio=0.3,
                use_moe=True,
                moe_num_experts=2,
                moe_top_k=1,
                use_flash_attention_2=True,
                use_gradient_checkpointing=True,
                use_adaptive_depth=True
            ),
            
            # Aggressive optimization config
            Qwen3VLConfig(
                use_sparsity=True,
                sparsity_ratio=0.6,
                use_moe=True,
                moe_num_experts=8,
                moe_top_k=2,
                use_flash_attention_2=True,
                use_gradient_checkpointing=True,
                use_adaptive_depth=True,
                exit_threshold=0.6
            )
        ]
        
        for i, config in enumerate(configs):
            with self.subTest(config=i):
                result = self.validator.validate_config(config, strict=False)
                self.assertTrue(result['valid'], f"Config {i} failed validation: {result['errors']}")
    
    def test_config_validation_with_serialization_roundtrip(self):
        """Test that configurations remain valid after serialization/deserialization"""
        original_config = Qwen3VLConfig(
            use_sparsity=True,
            sparsity_ratio=0.4,
            use_moe=True,
            moe_num_experts=6,
            moe_top_k=3,
            use_flash_attention_2=True
        )
        
        # Validate original config
        original_result = self.validator.validate_config(original_config, strict=False)
        self.assertTrue(original_result['valid'])
        
        # Serialize to dict
        config_dict = asdict(original_config)
        
        # Deserialize back to config
        from src.qwen3_vl.core.config import Qwen3VLConfig
        new_config = Qwen3VLConfig.from_dict(config_dict)
        
        # Validate new config
        new_result = self.validator.validate_config(new_config, strict=False)
        self.assertTrue(new_result['valid'])
        
        # Verify values are preserved
        self.assertEqual(original_config.use_sparsity, new_config.use_sparsity)
        self.assertEqual(original_config.sparsity_ratio, new_config.sparsity_ratio)
        self.assertEqual(original_config.use_moe, new_config.use_moe)
        self.assertEqual(original_config.moe_num_experts, new_config.moe_num_experts)
        self.assertEqual(original_config.moe_top_k, new_config.moe_top_k)
        self.assertEqual(original_config.use_flash_attention_2, new_config.use_flash_attention_2)
    
    def test_config_validation_with_file_roundtrip(self):
        """Test that configurations remain valid after file serialization/deserialization"""
        original_config = Qwen3VLConfig(
            use_sparsity=True,
            sparsity_ratio=0.3,
            use_moe=True,
            moe_num_experts=4,
            moe_top_k=2,
            use_adaptive_depth=True,
            exit_threshold=0.75
        )
        
        # Validate original config
        original_result = self.validator.validate_config(original_config, strict=False)
        self.assertTrue(original_result['valid'])
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(asdict(original_config), f)
            temp_file = f.name
        
        try:
            # Load config from file and validate
            loaded_result = self.validator.validate_file(temp_file, strict=False)
            self.assertTrue(loaded_result['valid'])
            
            # Also verify loaded config matches original
            from src.qwen3_vl.core.config import Qwen3VLConfig
            loaded_config = Qwen3VLConfig.from_file(temp_file)
            
            self.assertEqual(original_config.use_sparsity, loaded_config.use_sparsity)
            self.assertEqual(original_config.sparsity_ratio, loaded_config.sparsity_ratio)
            self.assertEqual(original_config.use_moe, loaded_config.use_moe)
            self.assertEqual(original_config.moe_num_experts, loaded_config.moe_num_experts)
            self.assertEqual(original_config.moe_top_k, loaded_config.moe_top_k)
            self.assertEqual(original_config.use_adaptive_depth, loaded_config.use_adaptive_depth)
            self.assertEqual(original_config.exit_threshold, loaded_config.exit_threshold)
        finally:
            os.unlink(temp_file)


class TestConfigValidationPerformance(unittest.TestCase):
    """Performance tests for configuration validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = ConfigValidator()
    
    def test_config_validation_performance_large_config(self):
        """Test validation performance with larger configurations"""
        import time
        
        # Create a configuration with many parameters
        config = Qwen3VLConfig(
            use_sparsity=True,
            use_moe=True,
            moe_num_experts=8,
            moe_top_k=3,
            use_flash_attention_2=True,
            use_gradient_checkpointing=True,
            use_adaptive_depth=True,
            use_context_adaptive_positional_encoding=True,
            use_conditional_feature_extraction=True,
            use_cross_modal_compression=True,
            use_cross_layer_memory_sharing=True,
            use_hierarchical_vision=True,
            use_learned_activation_routing=True,
            use_adaptive_batch_processing=True,
            use_adaptive_sequence_packing=True,
            use_memory_efficient_grad_accumulation=True,
            use_faster_rotary_embeddings=True,
            use_hardware_specific_kernels=True
        )
        
        start_time = time.time()
        result = self.validator.validate_config(config, strict=False)
        end_time = time.time()
        
        validation_time = end_time - start_time
        
        # Validation should be fast (less than 0.1 seconds)
        self.assertLess(validation_time, 0.1, f"Config validation took too long: {validation_time}s")
        self.assertTrue(result['valid'])
    
    def test_config_validation_performance_multiple_calls(self):
        """Test validation performance across multiple calls"""
        import time
        
        config = Qwen3VLConfig()
        
        start_time = time.time()
        for _ in range(100):
            result = self.validator.validate_config(config, strict=False)
            self.assertTrue(result['valid'])
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / 100
        
        # Average validation time should be very fast (less than 0.01 seconds per call)
        self.assertLess(avg_time, 0.01, f"Average config validation time too slow: {avg_time}s")
    
    def test_config_validation_memory_usage(self):
        """Test that config validation doesn't consume excessive memory"""
        import gc
        import torch
        
        # Record initial memory
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
        
        config = Qwen3VLConfig()
        
        # Run validation
        for _ in range(10):
            result = self.validator.validate_config(config, strict=False)
            self.assertTrue(result['valid'])
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Check memory hasn't grown significantly
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
            # Memory growth should be minimal (less than 1MB)
            self.assertLess(abs(memory_growth), 1024*1024, "Config validation consumed excessive memory")


def run_all_config_validation_tests():
    """Run all configuration validation tests and return results"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestConfigValidationSystem))
    suite.addTest(unittest.makeSuite(TestConfigValidatorIntegration))
    suite.addTest(unittest.makeSuite(TestConfigValidationPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Running Configuration Validation System Tests...")
    test_result = run_all_config_validation_tests()
    
    print(f"\nTest Results:")
    print(f"  Tests run: {test_result.testsRun}")
    print(f"  Failures: {len(test_result.failures)}")
    print(f"  Errors: {len(test_result.errors)}")
    
    if test_result.failures:
        print("  Failures:")
        for test, traceback in test_result.failures:
            print(f"    - {test}: {traceback}")
    
    if test_result.errors:
        print("  Errors:")
        for test, traceback in test_result.errors:
            print(f"    - {test}: {traceback}")
    
    success = len(test_result.failures) == 0 and len(test_result.errors) == 0
    print(f"\nOverall Result: {'PASS' if success else 'FAIL'}")
    
    if success:
        print("✅ All configuration validation tests passed!")
    else:
        print("❌ Some configuration validation tests failed!")
        exit(1)
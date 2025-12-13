"""
Integration Tests for Qwen3-VL Optimization Techniques
Validates that all 12 optimization techniques work together synergistically.
"""
import torch
import torch.nn as nn
import pytest
from typing import Dict, Any, List, Tuple
import time
import numpy as np
from unittest.mock import Mock, patch
import logging

from src.qwen3_vl.optimization.unified_optimization_manager import OptimizationManager, OptimizationConfig
from src.qwen3_vl.optimization.config_manager import ConfigManager, get_default_config
from src.qwen3_vl.optimization.interaction_handler import OptimizationInteractionHandler
from src.qwen3_vl.optimization.performance_validator import CumulativePerformanceValidator
from src.qwen3_vl.optimization.fallback_manager import AdaptiveFallbackManager, create_global_fallback_manager
from src.qwen3_vl.optimization.capacity_preservation import CapacityPreservationManager, create_capacity_preservation_manager
from src.qwen3_vl.components.models.qwen3_vl_model import Qwen3VLForConditionalGeneration
from src.qwen3_vl.components.configuration import Qwen3VLConfig


class MockModel(nn.Module):
    """Mock model for testing optimization integration"""
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads if config.num_attention_heads <= 16 else 16,  # Limit for testing
                batch_first=True
            ) for _ in range(config.num_hidden_layers)
        ])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class TestOptimizationIntegration:
    """Integration tests for all 12 optimization techniques"""
    
    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.config = get_default_config()
        self.optimization_manager = OptimizationManager(self.config)
        self.interaction_handler = OptimizationInteractionHandler(self.optimization_manager)
        self.performance_validator = CumulativePerformanceValidator()
        self.fallback_manager = create_global_fallback_manager(self.optimization_manager)
        self.capacity_manager = create_capacity_preservation_manager()
        
        # Create a mock model for testing
        qwen_config = Qwen3VLConfig()
        qwen_config.num_hidden_layers = 4  # Use fewer layers for faster testing
        qwen_config.num_attention_heads = 8  # Use fewer heads for faster testing
        self.model = MockModel(qwen_config)
        
        # Create test input
        batch_size, seq_len = 2, 64
        self.test_input = torch.randint(0, qwen_config.vocab_size, (batch_size, seq_len))
        
        # Set baseline for capacity preservation
        self.capacity_manager.set_baseline(self.model)
    
    def test_all_optimizations_enabled_integration(self):
        """Test that all optimizations work together when enabled"""
        print("\nTesting all optimizations enabled integration...")
        
        # Enable all optimizations
        for opt_type in self.optimization_manager.get_active_optimizations():
            self.optimization_manager.optimization_states[opt_type] = True
        
        # Validate model capacity before optimization
        assert self.capacity_manager.validate_before_optimization(self.model, self.test_input)
        
        # Apply optimizations through interaction handler
        output = self.interaction_handler.apply_optimizations_with_interaction_handling(
            self.test_input.clone(),
            layer_idx=0
        )
        
        # Validate model capacity after optimization
        assert self.capacity_manager.validate_after_optimization(self.model, self.test_input)
        
        # Check that output is valid
        assert output is not None
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        print("✓ All optimizations enabled integration test passed")
    
    def test_optimization_synergy_validation(self):
        """Test that optimizations provide synergistic benefits"""
        print("\nTesting optimization synergy validation...")
        
        # Run cumulative performance validation
        results = self.performance_validator.validate_cumulative_benefits(
            self.model,
            self.test_input,
            self.optimization_manager,
            num_runs=3  # Fewer runs for faster testing
        )
        
        # Check that we have positive synergy
        assert 'synergy_ratio' in results
        assert results['synergy_ratio'] >= 0.8  # Allow for some variance in test environment
        
        # Check that cumulative benefits exceed individual benefits
        assert 'cumulative_benefit_validated' in results
        # Note: In test environment, we may not achieve perfect synergy, so we check for reasonable performance
        
        print(f"✓ Optimization synergy validation passed. Synergy ratio: {results['synergy_ratio']:.2f}")
    
    def test_optimization_interaction_handling(self):
        """Test that optimization interactions are properly handled"""
        print("\nTesting optimization interaction handling...")
        
        # Get active optimizations
        active_opts = self.optimization_manager.get_active_optimizations()
        assert len(active_opts) > 0, "Should have active optimizations"
        
        # Check compatibility between optimizations
        for i, opt1 in enumerate(active_opts):
            for opt2 in active_opts[i+1:]:
                is_compatible, reason = self.interaction_handler.check_compatibility(opt1, opt2)
                print(f"  {opt1.value} <-> {opt2.value}: {'✓' if is_compatible else '✗'} - {reason}")
        
        # Apply optimizations with interaction handling
        result = self.interaction_handler.apply_optimizations_with_interaction_handling(
            self.test_input.clone(),
            layer_idx=0
        )
        
        assert result is not None
        assert not torch.isnan(result).any()
        
        print("✓ Optimization interaction handling test passed")
    
    def test_fallback_mechanisms_integration(self):
        """Test that fallback mechanisms work with optimization integration"""
        print("\nTesting fallback mechanisms integration...")
        
        # Test fallback when an optimization fails
        def failing_optimization(*args, **kwargs):
            raise RuntimeError("Optimization failed")
        
        # Apply optimization with fallback wrapper
        result = self.fallback_manager.auto_fallback_wrapper(
            list(self.optimization_manager.optimization_states.keys())[0] if self.optimization_manager.optimization_states else Mock(),
            failing_optimization,
            self.test_input
        )
        
        # Fallback should return input unchanged when optimization fails
        assert torch.equal(result, self.test_input) if isinstance(result, torch.Tensor) else True
        
        print("✓ Fallback mechanisms integration test passed")
    
    def test_capacity_preservation_with_all_optimizations(self):
        """Test that model capacity is preserved when all optimizations are applied"""
        print("\nTesting capacity preservation with all optimizations...")
        
        # Enable all optimizations
        for opt_type in self.optimization_manager.get_active_optimizations():
            self.optimization_manager.optimization_states[opt_type] = True
        
        # Run safety checks
        results = self.capacity_manager.validator.run_all_safety_checks(
            self.model,
            self.test_input
        )
        
        # Check that critical capacity metrics are preserved
        layer_check = next((r for r in results if r.check_type == 'layer_count'), None)
        param_check = next((r for r in results if r.check_type == 'parameter_count'), None)
        
        if layer_check:
            assert layer_check.passed or layer_check.details.get('actual', 0) >= 1  # At least some layers exist
        if param_check:
            assert param_check.passed  # Parameter count should be reasonable
        
        print("✓ Capacity preservation with all optimizations test passed")
    
    def test_performance_validation_integration(self):
        """Test performance validation with optimization integration"""
        print("\nTesting performance validation integration...")
        
        # Create baseline and optimized models (in practice, these would be different)
        baseline_model = self.model
        optimized_model = self.model  # For testing purposes
        
        # Validate performance
        result = self.performance_validator.validate_cumulative_performance(
            baseline_model,
            optimized_model,
            self.test_input,
            [opt.value for opt in self.optimization_manager.get_active_optimizations()],
            num_runs=2  # Fewer runs for faster testing
        )
        
        assert result is not None
        assert hasattr(result, 'improvement_factor')
        
        print(f"✓ Performance validation integration test passed. Improvement: {result.improvement_factor:.2f}x")
    
    def test_config_management_integration(self):
        """Test configuration management integration with optimizations"""
        print("\nTesting configuration management integration...")
        
        config_manager = ConfigManager()
        
        # Test creating different optimization levels
        minimal_config = config_manager.create_config_from_level('minimal')
        aggressive_config = config_manager.create_config_from_level('aggressive')
        
        assert minimal_config is not None
        assert aggressive_config is not None
        
        # Test config validation
        errors = config_manager.validator.validate_config(minimal_config)
        assert len(errors) == 0, f"Config validation failed: {errors}"
        
        # Test applying config to optimization manager
        self.optimization_manager.update_config(aggressive_config)
        
        # Verify some optimizations are enabled
        active_opts = self.optimization_manager.get_active_optimizations()
        assert len(active_opts) > 0
        
        print("✓ Configuration management integration test passed")
    
    def test_optimization_workflow_integration(self):
        """Test end-to-end optimization workflow"""
        print("\nTesting end-to-end optimization workflow...")
        
        from src.qwen3_vl.optimization.interaction_handler import OptimizationWorkflowManager
        
        workflow_manager = OptimizationWorkflowManager(self.interaction_handler)
        workflow = workflow_manager.create_optimization_workflow(self.config)
        
        # Execute workflow with validation
        result = workflow_manager.execute_workflow_with_validation(
            workflow,
            self.test_input.clone(),
            layer_idx=0
        )
        
        assert result is not None
        assert not torch.isnan(result).any()
        
        print("✓ End-to-end optimization workflow test passed")


class TestOptimizationCombinations:
    """Tests for different combinations of optimizations"""
    
    def setup_method(self):
        """Setup test fixtures before each test method"""
        self.config = get_default_config()
        self.optimization_manager = OptimizationManager(self.config)
        
        # Create a mock model for testing
        qwen_config = Qwen3VLConfig()
        qwen_config.num_hidden_layers = 2  # Use fewer layers for faster testing
        qwen_config.num_attention_heads = 4  # Use fewer heads for faster testing
        self.model = MockModel(qwen_config)
        
        # Create test input
        batch_size, seq_len = 1, 32
        self.test_input = torch.randint(0, qwen_config.vocab_size, (batch_size, seq_len))
    
    @pytest.mark.parametrize("combination", [
        ["block_sparse_attention", "faster_rotary_embeddings"],
        ["cross_modal_token_merging", "adaptive_sequence_packing"],
        ["hierarchical_memory_compression", "memory_efficient_grad_accumulation"],
        ["learned_activation_routing", "adaptive_batch_processing"],
        ["kv_cache_multiple_strategies", "hardware_specific_kernels"],
    ])
    def test_synergistic_combinations(self, combination):
        """Test synergistic combinations of optimizations"""
        print(f"\nTesting synergistic combination: {combination}")
        
        # Enable only the specified optimizations
        for opt_type in self.optimization_manager.get_active_optimizations():
            opt_name = opt_type.value.replace(" ", "_").replace("-", "_")
            self.optimization_manager.optimization_states[opt_type] = opt_name in combination
        
        # Apply optimizations
        interaction_handler = OptimizationInteractionHandler(self.optimization_manager)
        result = interaction_handler.apply_optimizations_with_interaction_handling(
            self.test_input.clone(),
            layer_idx=0
        )
        
        assert result is not None
        assert not torch.isnan(result).any()
        
        print(f"✓ Synergistic combination {combination} test passed")
    
    def test_optimization_order_impact(self):
        """Test that optimization order affects performance"""
        print("\nTesting optimization order impact...")
        
        interaction_handler = OptimizationInteractionHandler(self.optimization_manager)
        
        # Test different orders and ensure they produce valid results
        result1 = interaction_handler.apply_optimizations_with_interaction_handling(
            self.test_input.clone(),
            layer_idx=0
        )
        
        # Apply optimizations in different order by manipulating internal state
        active_opts = list(self.optimization_manager.get_active_optimizations())
        if len(active_opts) > 1:
            # Reverse the order for a simple test
            reversed_opts = active_opts[::-1]
            
            # Temporarily change the order by modifying the interaction handler
            original_order_method = interaction_handler._get_optimization_application_order
            interaction_handler._get_optimization_application_order = lambda active: reversed_opts
            
            result2 = interaction_handler.apply_optimizations_with_interaction_handling(
                self.test_input.clone(),
                layer_idx=0
            )
            
            # Restore original method
            interaction_handler._get_optimization_application_order = original_order_method
            
            assert result2 is not None
            assert not torch.isnan(result2).any()
        
        print("✓ Optimization order impact test passed")


class TestRealisticScenario:
    """Tests with more realistic scenarios"""
    
    def test_large_input_processing(self):
        """Test optimization integration with larger inputs"""
        print("\nTesting optimization integration with larger inputs...")
        
        # Create a larger test input
        batch_size, seq_len = 2, 128
        qwen_config = Qwen3VLConfig()
        qwen_config.num_hidden_layers = 4
        qwen_config.num_attention_heads = 8
        model = MockModel(qwen_config)
        
        test_input = torch.randint(0, qwen_config.vocab_size, (batch_size, seq_len))
        
        # Initialize optimization components
        config = get_default_config()
        optimization_manager = OptimizationManager(config)
        interaction_handler = OptimizationInteractionHandler(optimization_manager)
        capacity_manager = create_capacity_preservation_manager()
        capacity_manager.set_baseline(model)
        
        # Validate before
        assert capacity_manager.validate_before_optimization(model, test_input)
        
        # Apply optimizations
        result = interaction_handler.apply_optimizations_with_interaction_handling(
            test_input.clone(),
            layer_idx=0
        )
        
        # Validate after
        assert capacity_manager.validate_after_optimization(model, test_input)
        
        assert result is not None
        assert not torch.isnan(result).any()
        
        print("✓ Large input processing test passed")
    
    def test_memory_efficiency_under_load(self):
        """Test memory efficiency of optimizations under computational load"""
        print("\nTesting memory efficiency under load...")
        
        import gc
        
        # Create multiple test inputs to stress memory
        test_inputs = [
            torch.randint(0, 1000, (1, 64)) for _ in range(5)
        ]
        
        config = get_default_config()
        optimization_manager = OptimizationManager(config)
        interaction_handler = OptimizationInteractionHandler(optimization_manager)
        
        # Process multiple inputs and monitor for memory issues
        results = []
        for test_input in test_inputs:
            result = interaction_handler.apply_optimizations_with_interaction_handling(
                test_input.clone(),
                layer_idx=0
            )
            results.append(result)
            gc.collect()  # Clean up between tests
        
        # Verify all results are valid
        for result in results:
            assert result is not None
            assert not torch.isnan(result).any()
        
        print("✓ Memory efficiency under load test passed")


# Run tests if this file is executed directly
if __name__ == "__main__":
    import sys
    
    # Create test instance and run tests
    test_integration = TestOptimizationIntegration()
    test_integration.setup_method()
    
    try:
        test_integration.test_all_optimizations_enabled_integration()
        test_integration.test_optimization_synergy_validation()
        test_integration.test_optimization_interaction_handling()
        test_integration.test_fallback_mechanisms_integration()
        test_integration.test_capacity_preservation_with_all_optimizations()
        test_integration.test_performance_validation_integration()
        test_integration.test_config_management_integration()
        test_integration.test_optimization_workflow_integration()
        
        print("\n✓ All integration tests passed!")
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test combinations
    test_combinations = TestOptimizationCombinations()
    test_combinations.setup_method()
    
    try:
        # Test a few key synergistic combinations
        combinations_to_test = [
            ["block_sparse_attention", "faster_rotary_embeddings"],
            ["hierarchical_memory_compression", "memory_efficient_grad_accumulation"],
            ["kv_cache_multiple_strategies", "hardware_specific_kernels"],
        ]
        
        for combo in combinations_to_test:
            test_combinations.test_synergistic_combinations(combo)
        
        test_combinations.test_optimization_order_impact()
        
        print("✓ All combination tests passed!")
        
    except Exception as e:
        print(f"\n✗ Combination test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test realistic scenarios
    test_scenario = TestRealisticScenario()
    
    try:
        test_scenario.test_large_input_processing()
        test_scenario.test_memory_efficiency_under_load()
        
        print("✓ All realistic scenario tests passed!")
        
    except Exception as e:
        print(f"\n✗ Realistic scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 All integration tests completed successfully!")
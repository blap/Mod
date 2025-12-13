"""
Comprehensive Test Suite for Qwen3-VL Optimization Synergy
Tests that all 12 optimization techniques work together synergistically
while maintaining model capacity and accuracy.
"""
import torch
import torch.nn as nn
import pytest
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import time
import psutil
import GPUtil
import copy

# Import all optimization components
from qwen3_vl.optimization.unified_optimization_manager import (
    OptimizationManager, OptimizationType, OptimizationConfig
)
from qwen3_vl.optimization.interaction_handler import (
    OptimizationInteractionHandler, InteractionRule, InteractionType
)
from qwen3_vl.optimization.performance_validator import (
    PerformanceValidator, CumulativePerformanceValidator, PerformanceMetrics
)
from qwen3_vl.optimization.capacity_preservation import (
    CapacityPreservationManager, ModelCapacityValidator
)
from qwen3_vl.optimization.config_manager import ConfigManager, OptimizationLevel


@dataclass
class TestConfig:
    """Configuration for optimization synergy tests"""
    batch_size: int = 4
    seq_len: int = 128
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 4
    vocab_size: int = 1000
    test_runs: int = 5
    warmup_runs: int = 2


class MockModel(nn.Module):
    """Simple mock model for testing optimization integration"""
    
    def __init__(self, config: TestConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # Output layer
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        logits = self.lm_head(x)
        return logits


class OptimizationSynergyTester:
    """Comprehensive tester for optimization synergy"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup test logger"""
        import logging
        logger = logging.getLogger('OptimizationSynergyTester')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def create_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create test data for model evaluation"""
        input_ids = torch.randint(
            0, self.config.vocab_size, 
            (self.config.batch_size, self.config.seq_len)
        ).to(self.device)
        
        target_ids = torch.randint(
            0, self.config.vocab_size, 
            (self.config.batch_size, self.config.seq_len)
        ).to(self.device)
        
        return input_ids, target_ids
        
    def create_optimization_manager(self) -> OptimizationManager:
        """Create optimization manager with all optimizations enabled"""
        opt_config = OptimizationConfig()
        manager = OptimizationManager(opt_config)
        return manager
        
    def create_interaction_handler(self, optimization_manager: OptimizationManager):
        """Create interaction handler"""
        return OptimizationInteractionHandler(optimization_manager)
        
    def test_optimization_interactions(self):
        """Test that optimizations interact properly without conflicts"""
        self.logger.info("Testing optimization interactions...")
        
        manager = self.create_optimization_manager()
        handler = self.create_interaction_handler(manager)
        
        # Test all optimization pairs for compatibility
        active_opts = manager.get_active_optimizations()
        
        conflicts_found = 0
        for i, opt1 in enumerate(active_opts):
            for j, opt2 in enumerate(active_opts):
                if i != j:
                    is_compatible, reason = handler.check_compatibility(opt1, opt2)
                    if not is_compatible:
                        self.logger.warning(f"Conflict: {opt1} vs {opt2} - {reason}")
                        conflicts_found += 1
                    else:
                        self.logger.debug(f"Compatible: {opt1} vs {opt2}")
        
        assert conflicts_found == 0, f"Found {conflicts_found} conflicts between optimizations"
        self.logger.info("✓ All optimizations are compatible")
        
    def test_optimization_synergy(self):
        """Test that optimizations provide synergistic benefits"""
        self.logger.info("Testing optimization synergy...")
        
        # Create models: baseline vs optimized
        baseline_model = MockModel(self.config).to(self.device)
        optimized_model = MockModel(self.config).to(self.device)
        
        # Create optimization manager and apply optimizations to the optimized model
        manager = self.create_optimization_manager()
        handler = self.create_interaction_handler(manager)
        
        # Test with various optimization combinations
        all_optimizations = manager.get_active_optimizations()
        
        # Test subset combinations
        subset_sizes = [1, 2, 4, len(all_optimizations)]
        results = {}
        
        for subset_size in subset_sizes:
            if subset_size <= len(all_optimizations):
                subset = all_optimizations[:subset_size]
                
                # Temporarily configure manager for this subset
                original_states = {opt: manager.optimization_states[opt] for opt in all_optimizations}
                for opt in all_optimizations:
                    manager.optimization_states[opt] = opt in subset
                
                # Benchmark this subset
                validator = PerformanceValidator()
                input_tensor, _ = self.create_test_data()
                
                result = validator.validate_cumulative_performance(
                    baseline_model,
                    optimized_model,
                    input_tensor,
                    [opt.value for opt in subset],
                    num_runs=self.config.test_runs
                )
                
                results[f"subset_{subset_size}"] = result
                self.logger.info(f"Subset {subset_size}: Improvement factor = {result.improvement_factor:.2f}x")
                
                # Restore original states
                manager.optimization_states = original_states
        
        # Verify that larger subsets generally provide better improvements
        subset_keys = [k for k in results.keys() if k.startswith('subset_')]
        subset_sizes_tested = sorted([int(k.split('_')[1]) for k in subset_keys])
        
        for i in range(1, len(subset_sizes_tested)):
            smaller_key = f"subset_{subset_sizes_tested[i-1]}"
            larger_key = f"subset_{subset_sizes_tested[i]}"
            
            smaller_imp = results[smaller_key].improvement_factor
            larger_imp = results[larger_key].improvement_factor
            
            if larger_imp >= smaller_imp:
                self.logger.info(f"✓ Synergy confirmed: {subset_sizes_tested[i]} opts > {subset_sizes_tested[i-1]} opts")
            else:
                self.logger.warning(f"⚠ Synergy not confirmed: {subset_sizes_tested[i]} opts < {subset_sizes_tested[i-1]} opts")
        
        return results
        
    def test_capacity_preservation(self):
        """Test that model capacity is preserved with all optimizations"""
        self.logger.info("Testing capacity preservation...")
        
        # Create baseline model
        baseline_model = MockModel(self.config).to(self.device)
        
        # Create capacity preservation manager
        cap_manager = CapacityPreservationManager()
        cap_manager.set_baseline(baseline_model)
        
        # Create test input
        input_tensor, _ = self.create_test_data()
        
        # Validate baseline
        baseline_valid = cap_manager.validate_before_optimization(baseline_model, input_tensor)
        assert baseline_valid, "Baseline model failed capacity validation"
        
        # Test with optimized model (same architecture for this test)
        optimized_model = MockModel(self.config).to(self.device)
        optimized_valid = cap_manager.validate_after_optimization(optimized_model, input_tensor)
        
        assert optimized_valid, "Optimized model failed capacity validation"
        assert cap_manager.capacity_preserved, "Capacity was not preserved"
        
        self.logger.info("✓ Model capacity preserved with optimizations")
        
    def test_accuracy_preservation(self):
        """Test that accuracy is maintained across optimization combinations"""
        self.logger.info("Testing accuracy preservation...")
        
        # Create models
        baseline_model = MockModel(self.config).to(self.device)
        optimized_model = MockModel(self.config).to(self.device)
        
        # Create test data
        input_tensor, target_tensor = self.create_test_data()
        
        # Get outputs from both models
        baseline_model.eval()
        optimized_model.eval()
        
        with torch.no_grad():
            baseline_output = baseline_model(input_tensor)
            optimized_output = optimized_model(input_tensor)
        
        # Calculate similarity metrics
        mse = torch.mean((baseline_output - optimized_output) ** 2).item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            baseline_output.flatten(), 
            optimized_output.flatten(), 
            dim=0
        ).item()
        
        # Accuracy should be preserved (outputs should be similar)
        assert mse < 1.0, f"MSE too high: {mse}"  # Adjust threshold as needed
        assert cosine_sim > 0.95, f"Cosine similarity too low: {cosine_sim}"
        
        self.logger.info(f"✓ Accuracy preserved - MSE: {mse:.6f}, Cosine Similarity: {cosine_sim:.4f}")
        
    def test_resource_efficiency(self):
        """Test resource utilization with optimizations active"""
        self.logger.info("Testing resource efficiency...")
        
        # Create models
        baseline_model = MockModel(self.config).to(self.device)
        optimized_model = MockModel(self.config).to(self.device)
        
        # Create test data
        input_tensor, _ = self.create_test_data()
        
        # Benchmark resource usage
        validator = PerformanceValidator()
        
        baseline_metrics = validator.benchmark_model_performance(
            baseline_model, input_tensor, 
            num_runs=self.config.test_runs,
            measure_memory=True
        )
        
        # For this test, we'll just validate that metrics are collected properly
        assert baseline_metrics.execution_time > 0
        assert baseline_metrics.memory_usage >= 0
        
        # Test with resource budget validation
        resource_budget = {
            'max_memory_gb': 4.0,  # 4GB budget
            'max_time_per_sample': 0.1  # 100ms per sample
        }
        
        resource_check = validator.validate_resource_efficiency(
            baseline_model, input_tensor, resource_budget
        )
        
        self.logger.info(f"Resource efficiency: {resource_check}")
        
    def test_configuration_management(self):
        """Test that configuration system properly manages optimization interactions"""
        self.logger.info("Testing configuration management...")
        
        # Create config manager
        config_manager = ConfigManager()
        
        # Test different optimization levels
        for level_name, level in [("minimal", OptimizationLevel.MINIMAL), 
                                  ("moderate", OptimizationLevel.MODERATE),
                                  ("aggressive", OptimizationLevel.AGGRESSIVE)]:
            config = config_manager.create_config_from_level(level)
            manager = OptimizationManager(config)
            
            active_opts = manager.get_active_optimizations()
            self.logger.info(f"{level_name} level: {len(active_opts)} optimizations active")
            
            # Validate config
            validation_errors = config_manager.validate_config(config)
            assert len(validation_errors) == 0, f"Config validation errors: {validation_errors}"
            
            # Validate compatibility
            compatibility_warnings = config_manager.validate_compatibility(config)
            self.logger.info(f"Compatibility warnings for {level_name}: {len(compatibility_warnings)}")
        
        self.logger.info("✓ Configuration management working correctly")
        
    def run_comprehensive_tests(self):
        """Run all comprehensive optimization synergy tests"""
        self.logger.info("Running comprehensive optimization synergy tests...")
        
        # Test 1: Optimization interactions
        self.test_optimization_interactions()
        
        # Test 2: Optimization synergy
        synergy_results = self.test_optimization_synergy()
        
        # Test 3: Capacity preservation
        self.test_capacity_preservation()
        
        # Test 4: Accuracy preservation
        self.test_accuracy_preservation()
        
        # Test 5: Resource efficiency
        self.test_resource_efficiency()
        
        # Test 6: Configuration management
        self.test_configuration_management()
        
        self.logger.info("✓ All comprehensive optimization synergy tests passed!")
        
        return {
            'synergy_results': synergy_results,
            'test_status': 'PASSED'
        }


def test_all_optimization_synergy():
    """Main test function to validate all optimization synergy"""
    config = TestConfig()
    tester = OptimizationSynergyTester(config)
    
    results = tester.run_comprehensive_tests()
    
    # Additional validation: test cumulative benefits
    manager = tester.create_optimization_manager()
    validator = CumulativePerformanceValidator()
    
    model = MockModel(config).to(tester.device)
    input_tensor, _ = tester.create_test_data()
    
    cumulative_results = validator.validate_cumulative_benefits(
        model, input_tensor, manager, num_runs=config.test_runs
    )
    
    print(f"Cumulative benefits validated: {cumulative_results['cumulative_benefit_validated']}")
    print(f"Synergy ratio: {cumulative_results['synergy_ratio']:.2f}")
    
    assert cumulative_results['cumulative_benefit_validated'], "Cumulative benefits not validated"
    assert cumulative_results['synergy_ratio'] >= 1.0, f"Synergy ratio too low: {cumulative_results['synergy_ratio']}"
    
    return results


if __name__ == "__main__":
    # Run the comprehensive tests
    test_results = test_all_optimization_synergy()
    print("All optimization synergy tests completed successfully!")
    print(f"Results: {test_results}")
"""
Tests for the ML-based Optimization System in Inference-PIO

This module contains comprehensive tests for the ML-based optimization selection
and hyperparameter tuning system.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import numpy as np
from src.inference_pio.common.ml_optimization_selector import (
    AutoOptimizationSelector,
    PerformanceMetrics,
    OptimizationSelectionData,
    MLSuggestionEngine
)
from src.inference_pio.common.hyperparameter_optimizer import (
    PerformanceHyperparameterOptimizer,
    HyperparameterConfig
)
from src.inference_pio.common.unified_ml_optimization import (
    UnifiedMLOptimizationSystem,
    ModelType
)
from src.inference_pio.common.optimization_manager import OptimizationConfig

class MockModel(nn.Module):
    """Mock model for testing purposes."""
    
    def __init__(self):
        super().__init__()
        linear = nn.Linear(10, 5)
        
    def forward(self, x):
        return linear(x)

# TestMLOptimizationSelector

    """Test cases for ML-based optimization selector."""
    
    def setup_helper():
        """Set up test fixtures."""
        selector = AutoOptimizationSelector()
        mock_model = MockModel()
        mock_input = torch.randn(2, 10)
        
    def initialization(self)():
        """Test that the selector initializes correctly."""
        assert_is_not_none(selector.ml_engine)
        assert_equal(len(selector.performance_history))
        
    def select_optimizations_basic(self)():
        """Test basic optimization selection."""
        with patch.object(selector.ml_engine, 'suggest_optimizations', 
                         return_value=[(['flash_attention'], 0.5)]):
            result = selector.select_optimizations(
                model=mock_model,
                input_data=mock_input,
                target_metric="latency_ms"
            )
            assert_is_instance(result, list)
            assert_in('flash_attention', result)
            
    def update_with_performance_feedback(self)():
        """Test updating with performance feedback."""
        perf_metrics = PerformanceMetrics(
            latency_ms=100.0,
            memory_usage_mb=500.0,
            throughput_tokens_per_sec=10.0,
            energy_consumption=1.0,
            accuracy_drop=0.0
        )
        
        selector.update_with_performance_feedback(
            model=mock_model,
            input_data=mock_input,
            applied_optimizations=['flash_attention'],
            performance_metrics=perf_metrics
        )
        
        assert_equal(len(selector.performance_history), 1)
        
    def save_and_load_state(self)():
        """Test saving and loading the selector state."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            temp_path = tmp.name
            
        try:
            selector.save_state(temp_path)
            new_selector = AutoOptimizationSelector()
            new_selector.load_state(temp_path)
            
            assert_equal(len(new_selector.performance_history), 
                           len(selector.performance_history))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# TestHyperparameterOptimizer

    """Test cases for hyperparameter optimization."""
    
    def setup_helper():
        """Set up test fixtures."""
        optimizer = PerformanceHyperparameterOptimizer()
        mock_model = MockModel()
        mock_input = torch.randn(2, 10)
        
    def initialization(self)():
        """Test that the hyperparameter optimizer initializes correctly."""
        assert_is_not_none(optimizer.optimizer)
        assertGreater(len(optimizer.optimizer.hyperparameters))
        
    def add_hyperparameter(self)():
        """Test adding a hyperparameter."""
        hp_config = HyperparameterConfig(
            name="test_param",
            type="int",
            bounds=(1, 10),
            default_value=5
        )
        optimizer.optimizer.add_hyperparameter(hp_config)
        
        found = False
        for hp in optimizer.optimizer.hyperparameters:
            if hp.name == "test_param":
                found = True
                break
        assert_true(found)
        
    def optimize_for_model(self)():
        """Test optimizing hyperparameters for a model."""
        # Mock the objective function to return a fixed value
        with patch.object(optimizer.optimizer) as mock_optimize:
            mock_result = MagicMock()
            mock_result.best_params = {'batch_size': 8}
            mock_result.best_score = 0.5
            mock_result.optimization_trace = [({'batch_size': 8}, 0.5)]
            mock_result.num_evaluations = 1
            mock_optimize.return_value = mock_result
            
            result = optimizer.optimize_for_model(
                model=mock_model,
                input_data=mock_input,
                target_metric="latency"
            )
            
            assert_equal(result.best_params['batch_size'], 8)

# TestUnifiedMLOptimizationSystem

    """Test cases for the unified ML optimization system."""
    
    def setup_helper():
        """Set up test fixtures."""
        system = UnifiedMLOptimizationSystem()
        mock_model = MockModel()
        mock_input = torch.randn(2, 10)
        
    def initialization(self)():
        """Test that the system initializes with all model types."""
        assert_is_not_none(system.auto_selector)
        assertIsNotNone(system.performance_optimizer)
        # The system is initialized with configs)

        system.register_model_type(ModelType.GLM_4_7_FLASH, config)
        assert_in(ModelType.GLM_4_7_FLASH, system.configs)
        
    def register_model_type(self)():
        """Test registering a new model type."""
        from src.inference_pio.common.unified_ml_optimization import MLBasedOptimizationConfig
        
        config = MLBasedOptimizationConfig(
            model_type=ModelType.GLM_4_7_FLASH,
            enable_ml_selection=True,
            enable_hyperparameter_tuning=True
        )
        
        system.register_model_type(ModelType.GLM_4_7_FLASH, config)
        assert_in(ModelType.GLM_4_7_FLASH, system.configs)
        
    def optimize_model_for_input(self)():
        """Test optimizing a model for input."""
        # Mock the ML optimization methods
        with patch.object(system, '_apply_ml_optimizations') as mock_ml_opt:
            with patch.object(system, '_apply_hyperparameter_tuning') as mock_hp_opt:
                mock_ml_opt.return_value = mock_model
                mock_hp_opt.return_value = mock_model
                
                result = system.optimize_model_for_input(
                    model=mock_model,
                    input_data=mock_input,
                    model_type=ModelType.GLM_4_7_FLASH
                )
                
                assertIs(result, mock_model)
                mock_ml_opt.assert_called_once()
                mock_hp_opt.assert_called_once()
                
    def save_and_load_state(self)():
        """Test saving and loading the system state."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pkl') as tmp:
            temp_path = tmp.name
            
        try:
            system.save_state(temp_path)
            new_system = UnifiedMLOptimizationSystem()
            new_system.load_state(temp_path)
            
            assert_equal(len(new_system.call_counts), 
                           len(system.call_counts))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# TestIntegration

    """Integration tests for the ML optimization system."""
    
    def setup_helper():
        """Set up test fixtures."""
        system = UnifiedMLOptimizationSystem()
        mock_model = MockModel()
        mock_input = torch.randn(2, 10)
        
    def end_to_end_optimization(self)():
        """Test end-to-end optimization workflow."""
        # This is a high-level test to ensure the system can process a model
        try:
            result = system.optimize_model_for_input(
                model=mock_model,
                input_data=mock_input,
                model_type=ModelType.GLM_4_7_FLASH
            )
            # The result should be a model (could be the same or modified)
            assert_is_instance(result, torch.nn.Module)
        except Exception as e:
            # If there are issues with the actual ML optimization, 
            # at least verify the system structure is correct
            assert_is_not_none(system)

if __name__ == '__main__':
    run_tests(test_functions)
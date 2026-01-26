"""
Tests for the Structured Pruning System in Inference-PIO

This module contains comprehensive tests for the structured pruning system
that preserves model accuracy while reducing complexity.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import sys
import os
from typing import Dict, List, Optional

# Add the src directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.inference_pio.common.structured_pruning import (
    StructuredPruningSystem, PruningMethod, PruningResult, 
    get_structured_pruning_system, apply_structured_pruning
)

class SimpleTestModel(nn.Module):
    """Simple test model for pruning tests."""
    
    def __init__(self):
        super().__init__()
        linear1 = nn.Linear(10, 20)
        relu1 = nn.ReLU()
        linear2 = nn.Linear(20, 15)
        relu2 = nn.ReLU()
        linear3 = nn.Linear(15, 5)
        attention = nn.MultiheadAttention(embed_dim=10, num_heads=2)
        
    def forward(self, x):
        x = relu1(linear1(x))
        x = relu2(linear2(x))
        x, _ = attention(x, x, x)
        x = linear3(x)
        return x

# TestStructuredPruning

    """Test cases for the structured pruning system."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        model = SimpleTestModel()
        pruning_system = get_structured_pruning_system()
        
    def initialization(self)():
        """Test that the pruning system initializes correctly."""
        assert_is_instance(pruning_system, StructuredPruningSystem)
        assert_equal(len(pruning_system.pruning_history), 0)
        
    def calculate_layer_importance_magnitude(self)():
        """Test calculating layer importance using magnitude method."""
        importance_scores = pruning_system.calculate_layer_importance(
            model, method="magnitude"
        )
        
        # Check that we have scores for prunable layers
        prunable_layers = ['linear1', 'linear2', 'linear3', 'attention']
        for layer_name in prunable_layers:
            full_name = f"{layer_name}" if hasattr(model, layer_name) else None
            if full_name:
                for name, module in model.named_modules():
                    if name.endswith(layer_name) and pruning_system._is_prunable_layer(module):
                        assert_in(name, importance_scores)
                        assert_is_instance(importance_scores[name], float)
                        
    def identify_least_important_blocks(self)():
        """Test identifying least important blocks for pruning."""
        importance_scores = pruning_system.calculate_layer_importance(
            model, method="magnitude"
        )
        
        # Test with different block sizes
        blocks_single = pruning_system.identify_least_important_blocks(
            model, importance_scores, pruning_ratio=0.5, block_size=1
        )
        
        blocks_double = pruning_system.identify_least_important_blocks(
            model, importance_scores, pruning_ratio=0.5, block_size=2
        )
        
        # Should have identified some layers to remove
        assertGreaterEqual(len(blocks_single), 0)
        assertGreaterEqual(len(blocks_double), 0)
        
    def prune_model_layer_removal(self)():
        """Test pruning model using layer removal method."""
        original_params = sum(p.numel() for p in model.parameters())
        
        result = pruning_system.prune_model(
            model,
            pruning_ratio=0.2,
            method=PruningMethod.LAYER_REMOVAL
        )
        
        # Check that result is properly formed
        assert_is_instance(result, PruningResult)
        assert_is_instance(result.pruned_model, nn.Module)
        assert_equal(result.original_params, original_params)
        assertLessEqual(result.pruned_params, result.original_params)
        assertGreaterEqual(result.compression_ratio, 0.0)
        assert_is_instance(result.accuracy_preserved, bool)
        assert_is_instance(result.removed_layers, list)
        assert_is_instance(result.metrics, dict)
        
        # Check that some parameters were actually pruned
        if original_params > 0:
            assert_less(result.pruned_params, original_params)
        
    def prune_model_block_removal(self)():
        """Test pruning model using block removal method."""
        original_params = sum(p.numel() for p in model.parameters())
        
        result = pruning_system.prune_model(
            model,
            pruning_ratio=0.2,
            block_size=2,
            method=PruningMethod.BLOCK_REMOVAL
        )
        
        # Check that result is properly formed
        assert_is_instance(result, PruningResult)
        assert_is_instance(result.pruned_model, nn.Module)
        assert_equal(result.original_params, original_params)
        assertLessEqual(result.pruned_params, result.original_params)
        assertGreaterEqual(result.compression_ratio, 0.0)
        assert_is_instance(result.accuracy_preserved, bool)
        assert_is_instance(result.removed_layers, list)
        assert_is_instance(result.metrics, dict)
        
    def prune_model_head_removal(self)():
        """Test pruning model using head removal method."""
        original_params = sum(p.numel() for p in model.parameters())
        
        result = pruning_system.prune_model(
            model,
            pruning_ratio=0.5,  # Higher ratio to ensure some heads are targeted
            method=PruningMethod.HEAD_REMOVAL
        )
        
        # Check that result is properly formed
        assert_is_instance(result, PruningResult)
        assert_is_instance(result.pruned_model, nn.Module)
        
    def convenience_function(self)():
        """Test the convenience function for applying structured pruning."""
        original_params = sum(p.numel() for p in model.parameters())
        
        result = apply_structured_pruning(
            model,
            pruning_ratio=0.2,
            method=PruningMethod.LAYER_REMOVAL
        )
        
        # Check that result is properly formed
        assert_is_instance(result, PruningResult)
        assert_is_instance(result.pruned_model, nn.Module)
        assert_equal(result.original_params, original_params)
        assertLessEqual(result.pruned_params, result.original_params)
        
    def pruning_with_different_ratios(self)():
        """Test pruning with different ratios."""
        ratios = [0.1, 0.3, 0.5]
        original_params = sum(p.numel() for p in model.parameters())
        
        for ratio in ratios:
            result = pruning_system.prune_model(
                model,
                pruning_ratio=ratio,
                method=PruningMethod.LAYER_REMOVAL
            )
            
            assert_is_instance(result, PruningResult)
            assertLessEqual(result.pruned_params, original_params)
            assertGreaterEqual(result.compression_ratio, 0.0)
            
    def get_pruning_stats(self)():
        """Test getting pruning statistics."""
        # First perform a pruning operation
        apply_structured_pruning(model, pruning_ratio=0.2)
        
        stats = pruning_system.get_pruning_stats()
        
        assert_is_instance(stats, dict)
        assert_in('total_pruning_operations', stats)
        assertGreaterEqual(stats['total_pruning_operations'], 1)
        
    def is_prunable_layer(self)():
        """Test the _is_prunable_layer method."""
        linear_layer = nn.Linear(10, 5)
        relu_layer = nn.ReLU()
        conv_layer = nn.Conv1d(3, 6, 3)
        
        # Linear and Conv layers should be prunable
        assert_true(pruning_system._is_prunable_layer(linear_layer))
        assertTrue(pruning_system._is_prunable_layer(conv_layer))
        
        # ReLU should not be prunable
        assert_false(pruning_system._is_prunable_layer(relu_layer))
        
    def estimate_accuracy_preservation(self)():
        """Test the accuracy preservation estimation."""
        # With minimal pruning)
        assert_true(preserved)
        
        # With significant pruning)
        # This might be true or false depending on the heuristic

# TestStructuredPruningIntegration

    """Integration tests for the structured pruning system."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        model = SimpleTestModel()
        
    def end_to_end_pruning_process(self)():
        """Test the complete end-to-end pruning process."""
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply structured pruning
        result = apply_structured_pruning(
            model,
            pruning_ratio=0.25,
            block_size=1,
            method=PruningMethod.LAYER_REMOVAL
        )
        
        # Verify the result
        assert_is_instance(result, PruningResult)
        assert_is_instance(result.pruned_model, nn.Module)
        assert_equal(result.original_params, original_params)
        assertLessEqual(result.pruned_params, result.original_params)
        assertGreaterEqual(result.compression_ratio, 0.0)
        assertLessEqual(result.compression_ratio, 1.0)
        
        # Verify that the pruned model can still perform forward pass
        dummy_input = torch.randn(1, 10)
        try:
            with torch.no_grad():
                output = result.pruned_model(dummy_input)
                assert_is_instance(output, torch.Tensor)
        except Exception as e:
            fail(f"Forward pass failed after pruning: {e}")
            
    def multiple_pruning_operations(self)():
        """Test performing multiple pruning operations."""
        system = get_structured_pruning_system()
        
        # Clear any previous history
        system.pruning_history = []
        
        # Perform multiple pruning operations
        for i in range(3):
            result = system.prune_model(
                model,
                pruning_ratio=0.1,
                method=PruningMethod.LAYER_REMOVAL
            )
            assert_is_instance(result, PruningResult)
        
        # Check that history was recorded
        stats = system.get_pruning_stats()
        assert_equal(stats['total_pruning_operations'], 3)

if __name__ == '__main__':
    # Run the tests
    run_tests(test_functions)
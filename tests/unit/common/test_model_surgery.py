"""
Test suite for the Model Surgery System in Inference-PIO.

This module contains comprehensive tests for the Model Surgery functionality
that identifies and temporarily removes non-essential components during inference.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import sys
import os

# Add the src directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ..model_surgery import ModelSurgerySystem, ComponentType, SurgicalComponent, apply_model_surgery, restore_model_from_surgery

class TestModelWithRemovableComponents(nn.Module):
    """A test model with various components that can be surgically removed."""
    
    def __init__(self):
        super().__init__()
        linear1 = nn.Linear(10, 20)
        dropout = nn.Dropout(0.5)
        norm1 = nn.LayerNorm(20)
        relu = nn.ReLU()
        linear2 = nn.Linear(20, 10)
        batch_norm = nn.BatchNorm1d(10)
        identity = nn.Identity()
        
    def forward(self, x):
        x = linear1(x)
        x = dropout(x)
        x = norm1(x)
        x = relu(x)
        x = linear2(x)
        x = batch_norm(x)
        return x

# TestModelSurgery

    """Test cases for the Model Surgery System."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        surgery_system = ModelSurgerySystem()
        test_model = TestModelWithRemovableComponents()
        
    def identify_removable_components(self)():
        """Test identification of removable components."""
        removable = surgery_system.identify_removable_components(test_model)
        
        # Check that we found some removable components
        assert_greater(len(removable), 0)
        
        # Check that specific types are identified
        component_types = [comp.type for comp in removable]
        assert_in(ComponentType.DROPOUT_LAYER, component_types)
        assert_in(ComponentType.LAYER_NORM, component_types)
        assert_in(ComponentType.ACTIVATION_LAYER, component_types)
        
    def can_safely_remove(self)():
        """Test the _can_safely_remove method."""
        # Test dropout
        can_remove, reason, priority = surgery_system._can_safely_remove(
            nn.Dropout(0.5), ComponentType.DROPOUT_LAYER, "test_dropout"
        )
        assert_true(can_remove)
        assert_equal(priority)  # Highest priority
        
        # Test layer norm
        can_remove, reason, priority = surgery_system._can_safely_remove(
            nn.LayerNorm(10), ComponentType.LAYER_NORM, "test_norm"
        )
        assert_true(can_remove)
        assert_equal(priority)  # Second highest priority
        
    def perform_surgery(self)():
        """Test performing model surgery."""
        # Get original parameter count
        original_params = sum(p.numel() for p in test_model.parameters())
        
        # Perform surgery
        modified_model = surgery_system.perform_surgery(test_model)
        
        # Check that the model still works
        test_input = torch.randn(5, 10)
        original_output = test_model(test_input)
        modified_output = modified_model(test_input)
        
        # The outputs should be different due to removed components like dropout
        # but the model should still run
        assert_equal(original_output.shape, modified_output.shape)
        
        # Check that the model still has parameters (some may have been reduced due to normalization layers)
        modified_params = sum(p.numel() for p in modified_model.parameters())
        # The modified model may have fewer parameters if normalization layers were replaced
        # with identity layers (which don't have parameters)
        assertGreaterEqual(modified_params, 0)
        
    def perform_surgery_with_specific_removals(self)():
        """Test performing surgery with specific components to remove."""
        # Perform surgery on specific components
        modified_model = surgery_system.perform_surgery(
            test_model,
            components_to_remove=['dropout', 'batch_norm']
        )
        
        # Check that specific components were replaced
        assert_is_instance(modified_model.dropout, nn.Identity)
        assert_is_instance(modified_model.batch_norm, nn.Identity)
        
        # Other components should remain unchanged
        assertNotIsInstance(modified_model.norm1, nn.Identity)
        
    def restore_model(self)():
        """Test restoring a model from surgery."""
        # Perform surgery
        modified_model = surgery_system.perform_surgery(test_model)
        
        # Check that components were replaced
        assert_is_instance(modified_model.dropout, nn.Identity)
        
        # Restore the model
        restored_model = surgery_system.restore_model(modified_model)
        
        # Check that the original components are back
        assert_is_instance(restored_model.dropout, nn.Dropout)
        assert_is_instance(restored_model.batch_norm, nn.BatchNorm1d)
        
    def analyze_model_for_surgery(self)():
        """Test model analysis for surgery."""
        analysis = surgery_system.analyze_model_for_surgery(test_model)
        
        # Check that analysis contains expected keys
        assert_in('total_parameters', analysis)
        assert_in('total_modules', analysis)
        assert_in('removable_components', analysis)
        assert_in('recommendations', analysis)
        
        # Check that we have some removable components identified
        assert_greater(len(analysis['removable_components']), 0)
        
    def convenience_functions(self)():
        """Test the convenience functions."""
        # Test apply_model_surgery
        modified_model = apply_model_surgery(test_model)
        
        # Check that some components were replaced
        has_identity = any(isinstance(module, nn.Identity) for module in modified_model.modules())
        assert_true(has_identity)
        
        # Test restore_model_from_surgery
        restored_model = restore_model_from_surgery(modified_model)
        
        # Check that original components are back
        has_dropout = any(isinstance(module) for module in restored_model.modules())
        assert_true(has_dropout)
        
    def preserve_components(self)():
        """Test preserving specific components during surgery."""
        # Perform surgery while preserving specific components
        modified_model = surgery_system.perform_surgery(
            test_model,
            preserve_components=['norm1']
        )
        
        # Check that preserved component remains unchanged
        assertNotIsInstance(modified_model.norm1, nn.Identity)
        
        # Check that other removable components were still removed
        assert_is_instance(modified_model.dropout, nn.Identity)
        
    def surgery_stats(self)():
        """Test getting surgery statistics."""
        stats = surgery_system.get_surgery_stats()
        
        # Initially should have no surgeries
        assert_equal(stats['total_surgeries_performed'], 0)
        
        # Perform a surgery
        surgery_system.perform_surgery(test_model)
        
        # Check stats after surgery
        stats = surgery_system.get_surgery_stats()
        assert_greater(stats['total_surgeries_performed'], 0)

# TestAdvancedModelSurgery

    """Advanced test cases for complex model architectures."""
    
    def deep_model_surgery(self)():
        """Test surgery on a deeper model with nested modules."""
        class DeepTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                layer1 = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.Dropout(0.3),
                    nn.LayerNorm(20),
                    nn.ReLU()
                )
                layer2 = nn.Sequential(
                    nn.Linear(20, 10),
                    nn.BatchNorm1d(10),
                    nn.Dropout(0.2)
                )
                
            def forward(self, x):
                x = layer1(x)
                x = layer2(x)
                return x
        
        model = DeepTestModel()
        surgery_system = ModelSurgerySystem()
        
        # Perform surgery
        modified_model = surgery_system.perform_surgery(model)
        
        # Check that nested components were handled
        assert_is_instance(modified_model.layer1[1], nn.Identity)  # dropout
        assert_is_instance(modified_model.layer1[2], nn.Identity)  # layer norm
        assert_is_instance(modified_model.layer2[2], nn.Identity)  # dropout
        
        # But linear layers should remain
        assert_is_instance(modified_model.layer1[0], nn.Linear)
        assert_is_instance(modified_model.layer2[0], nn.Linear)
        
    def error_handling(self)():
        """Test error handling in surgery operations."""
        surgery_system = ModelSurgerySystem()
        
        # Test with invalid model
        with assert_raises(Exception):
            surgery_system.perform_surgery(None)
        
        # Test with empty model
        empty_model = nn.Module()
        modified_model = surgery_system.perform_surgery(empty_model)
        assert_is_not_none(modified_model)

if __name__ == '__main__':
    run_tests(test_functions)
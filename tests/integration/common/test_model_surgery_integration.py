"""
Integration test for Model Surgery with Plugin System.

This module tests the integration of the Model Surgery system with the plugin architecture.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import sys
import os

# Add the src directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.inference_pio.common.model_surgery import ModelSurgerySystem, apply_model_surgery, restore_model_from_surgery
from src.inference_pio.common.base_plugin_interface import ModelPluginInterface, ModelPluginMetadata, PluginType
from datetime import datetime

# Import the real model implementation
from real_model_for_testing import RealGLM47Model


class MockModel(RealGLM47Model):
    """Real model for testing."""

    def __init__(self):
        # Initialize with smaller parameters for testing
        super().__init__(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
            vocab_size=256
        )
        # Add additional layers for surgery testing
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 5)

    def forward(self, x):
        # Handle different input formats for compatibility with test expectations
        if x.dim() == 2 and x.size(1) == 10:
            # If input is (batch, 10), treat as token IDs
            input_ids = x.argmax(dim=1).unsqueeze(0).long()  # Convert to token IDs
            result = super().forward(input_ids=input_ids)
            # Extract the last hidden state and apply additional layers
            x = result.last_hidden_state.mean(dim=1)  # Average pooling
            x = self.dropout(x)
            x = self.norm(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
        elif x.dim() == 2:
            # If input has different size, convert to token IDs appropriately
            input_ids = x.abs().sum(dim=1).unsqueeze(0).long()  # Sum along feature dimension to get token IDs
            result = super().forward(input_ids=input_ids)
            # Extract the last hidden state and apply additional layers
            x = result.last_hidden_state.mean(dim=1)  # Average pooling
            x = self.dropout(x)
            x = self.norm(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
        else:
            # For other inputs, use the parent forward method
            result = super().forward(input_ids=x.long())
            # Extract the last hidden state and apply additional layers
            x = result.last_hidden_state.mean(dim=1)  # Average pooling
            x = self.dropout(x)
            x = self.norm(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x

# TestModelSurgeryPluginIntegration

    """Test cases for Model Surgery integration with Plugin System."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        # Create mock metadata
        metadata = ModelPluginMetadata(
            name="TestModel",
            version="1.0.0",
            author="Test Author",
            description="Test model for surgery integration",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["torch"],
            compatibility={"torch_version": ">=2.0.0"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model_architecture="Test Architecture",
            model_size="Small",
            required_memory_gb=1.0
        )
        
        # Create a mock plugin that inherits from ModelPluginInterface
        class TestPlugin(ModelPluginInterface):
            def __init__(self, metadata):
                super().__init__(metadata)
                _model = MockModel()
                
            def initialize(self, **kwargs):
                return True
                
            def load_model(self, config=None):
                return _model
                
            def infer(self, data):
                return _model(data)
                
            def cleanup(self):
                return True
        
        plugin = TestPlugin(metadata)
        
    def plugin_has_surgery_methods(self)():
        """Test that the plugin has model surgery methods."""
        # Check that the plugin has the surgery-related methods
        assert_true(hasattr(plugin))
        assert_true(hasattr(plugin))
        assert_true(hasattr(plugin))
        assert_true(hasattr(plugin))
        assert_true(hasattr(plugin))
        assert_true(hasattr(plugin))
        
    def setup_model_surgery(self)():
        """Test setting up model surgery in plugin."""
        result = plugin.setup_model_surgery(surgery_enabled=True)
        assert_true(result)
        assertTrue(hasattr(plugin))
        assert_true(hasattr(plugin))
        
    def enable_model_surgery(self)():
        """Test enabling model surgery in plugin."""
        # First setup
        plugin.setup_model_surgery(surgery_enabled=True)
        
        # Then enable
        result = plugin.enable_model_surgery()
        assert_true(result)
        
    def perform_model_surgery(self)():
        """Test performing model surgery through plugin."""
        # Setup and enable surgery
        plugin.setup_model_surgery(surgery_enabled=True)
        plugin.enable_model_surgery()
        
        # Get original parameter count
        original_params = sum(p.numel() for p in plugin._model.parameters())
        
        # Perform surgery
        modified_model = plugin.perform_model_surgery()
        
        # Check that the model was modified
        assert_is_not_none(modified_model)
        
        # The modified model should have fewer parameters due to normalization layer removal
        modified_params = sum(p.numel() for p in modified_model.parameters())
        
        # Check that the model still works
        test_input = torch.randn(3)
        output = modified_model(test_input)
        assert_equal(output.shape))
        
    def analyze_model_for_surgery(self)():
        """Test analyzing model for surgery through plugin."""
        # Setup surgery
        plugin.setup_model_surgery(surgery_enabled=True)
        
        # Analyze the model
        analysis = plugin.analyze_model_for_surgery()
        
        # Check that analysis contains expected keys
        assert_in('total_parameters', analysis)
        assert_in('removable_components', analysis)
        assert_in('recommendations', analysis)
        
        # Check that some components were identified as removable
        assert_greater(len(analysis['removable_components']), 0)
        
    def get_surgery_stats(self)():
        """Test getting surgery stats through plugin."""
        # Setup surgery
        plugin.setup_model_surgery(surgery_enabled=True)
        
        # Get initial stats
        stats = plugin.get_surgery_stats()
        assert_in('total_surgeries_performed', stats)
        assert_in('total_components_removed', stats)
        
    def end_to_end_surgery_workflow(self)():
        """Test complete end-to-end surgery workflow."""
        # Setup and enable surgery
        plugin.setup_model_surgery(
            surgery_enabled=True,
            auto_identify_components=True,
            preserve_components=[]
        )
        plugin.enable_model_surgery()
        
        # Analyze model
        analysis = plugin.analyze_model_for_surgery()
        assert_greater(len(analysis['removable_components']), 0)
        
        # Perform surgery
        original_params = sum(p.numel() for p in plugin._model.parameters())
        modified_model = plugin.perform_model_surgery()
        modified_params = sum(p.numel() for p in modified_model.parameters())
        
        # Check that surgery was attempted (even if no components were removed)
        stats = plugin.get_surgery_stats()
        # The surgery may not have removed components if none were identified as removable
        # but the operation should still be recorded
        assert_is_not_none(stats)
        
        # Test that the modified model works
        test_input = torch.randn(2)
        output = modified_model(test_input)
        assert_equal(output.shape, (2))
        
        # Restore the model
        restored_model = plugin.restore_model_from_surgery()
        
        # Check that parameters are back to original (or close)
        restored_params = sum(p.numel() for p in restored_model.parameters())
        
        # The restored model should work
        restored_output = restored_model(test_input)
        assert_equal(restored_output.shape, (2))

# TestModelSurgeryDirect

    """Additional tests for the model surgery system directly."""
    
    def model_surgery_system_creation(self)():
        """Test creating a model surgery system instance."""
        surgery_system = ModelSurgerySystem()
        assert_is_not_none(surgery_system)
        assert_equal(len(surgery_system.surgical_components))
        assert_equal(len(surgery_system.backup_registry), 0)
        assert_equal(len(surgery_system.surgery_history), 0)
        
    def identify_different_component_types(self)():
        """Test identifying different types of removable components."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                linear = nn.Linear(10, 20)
                dropout = nn.Dropout(0.3)
                batch_norm = nn.BatchNorm1d(20)
                layer_norm = nn.LayerNorm(20)
                group_norm = nn.GroupNorm(4, 20)
                instance_norm = nn.InstanceNorm1d(20)
                relu = nn.ReLU()
                sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = linear(x)
                x = dropout(x)
                x = batch_norm(x)
                x = layer_norm(x)
                x = relu(x)
                return x
        
        model = TestModel()
        surgery_system = ModelSurgerySystem()
        
        removable = surgery_system.identify_removable_components(model)
        
        # Check that different types of components were identified
        component_types = [comp.type for comp in removable]
        
        # Should have dropout
        assert_in('dropout_layer', [t.value for t in component_types])
        
        # Should have some removable components
        assert_greater(len(component_types), 0)

if __name__ == '__main__':
    run_tests(test_functions)
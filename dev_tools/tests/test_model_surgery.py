"""
Test suite for Model Surgery functionality in model plugins.

This test verifies that the model surgery system works correctly across all model plugins.
"""
import torch
import torch.nn as nn

from src.inference_pio.common.model_surgery import (
    ModelSurgerySystem,
    apply_model_surgery,
    restore_model_from_surgery,
)
from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import (
    Qwen3_4B_Instruct_2507_Plugin,
)
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin
from tests.utils.test_utils import (
    assert_equal,
    assert_false,
    assert_greater,
    assert_in,
    assert_is_instance,
    assert_is_none,
    assert_is_not_none,
    assert_less,
    assert_not_equal,
    assert_not_in,
    assert_raises,
    assert_true,
    run_tests,
)

# TestModelSurgery

    """Test cases for model surgery functionality."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        plugins = [
            GLM_4_7_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin(),
            Qwen3_VL_2B_Plugin()
        ]
        
        # Create a simple test model for surgery tests
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                dropout = nn.Dropout(0.1)
                norm = nn.LayerNorm(10)
                linear1 = nn.Linear(10, 5)
                relu = nn.ReLU()
                linear2 = nn.Linear(5, 1)
                
            def forward(self, x):
                x = dropout(x)
                x = norm(x)
                x = linear1(x)
                x = relu(x)
                x = linear2(x)
                return x
        
        simple_model = SimpleModel()

    def model_surgery_system_creation(self)():
        """Test that the model surgery system can be created and accessed."""
        surgery_system = ModelSurgerySystem()
        assert_is_instance(surgery_system, ModelSurgerySystem)
        
        # Test global instance
        from src.inference_pio.common.model_surgery import get_model_surgery_system
        global_system = get_model_surgery_system()
        assert_is_instance(global_system, ModelSurgerySystem)

    def identify_removable_components(self)():
        """Test that the system can identify removable components."""
        surgery_system = ModelSurgerySystem()
        
        # Test with our simple model
        removable = surgery_system.identify_removable_components(simple_model)
        
        # Should find dropout and possibly normalization layers
        assertGreaterEqual(len(removable), 1, 
                               "Should find at least one removable component (dropout)")
        
        # Check that identified components have required attributes
        for comp in removable:
            assert_is_not_none(comp.name)
            assertIsNotNone(comp.module)
            assertIsNotNone(comp.type)
            assert_is_instance(comp.priority)

    def perform_surgery(self)():
        """Test performing model surgery."""
        surgery_system = ModelSurgerySystem()
        
        # Get original parameter count
        original_params = sum(p.numel() for p in simple_model.parameters())
        
        # Perform surgery
        modified_model = surgery_system.perform_surgery(simple_model)
        
        # Parameter count should remain the same (we replace with identity, not remove)
        modified_params = sum(p.numel() for p in modified_model.parameters())
        
        # The model should still work
        test_input = torch.randn(2, 10)
        original_output = simple_model(test_input)
        modified_output = modified_model(test_input)
        
        # Outputs should be similar (though not identical due to dropout removal)
        assert_equal(original_output.shape, modified_output.shape)

    def apply_model_surgery_function(self)():
        """Test the convenience function for applying model surgery."""
        # Test with our simple model
        modified_model = apply_model_surgery(simple_model)
        
        # Should return a model
        assert_is_instance(modified_model, nn.Module)
        
        # Should still be functional
        test_input = torch.randn(2, 10)
        output = modified_model(test_input)
        assert_equal(output.shape, (2))

    def restore_model_from_surgery(self)():
        """Test restoring a model from surgery."""
        # First apply surgery
        modified_model = apply_model_surgery(simple_model)
        
        # Then restore
        restored_model = restore_model_from_surgery(modified_model)
        
        # Should return a model
        assert_is_instance(restored_model, nn.Module)
        
        # Should still be functional
        test_input = torch.randn(2, 10)
        output = restored_model(test_input)
        assert_equal(output.shape, (2))

    def plugin_model_surgery_setup(self)():
        """Test that all plugins can set up model surgery."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize(enable_model_surgery=True)
            assert_true(success)
            
            # Check that model surgery methods are available
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))

    def plugin_perform_model_surgery(self)():
        """Test that plugins can perform model surgery."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with model surgery enabled
            success = plugin.initialize(enable_model_surgery=True)
            assert_true(success)
            
            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()
            
            # Perform model surgery
            original_model = plugin._model
            modified_model = plugin.perform_model_surgery()
            
            # Should return a model
            assert_is_instance(modified_model, type(original_model))
            
            # Should have performed some modifications
            surgery_stats = plugin.get_surgery_stats()
            assert_is_instance(surgery_stats, dict)

    def plugin_analyze_model_for_surgery(self)():
        """Test that plugins can analyze models for surgery."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin
            success = plugin.initialize()
            assert_true(success)
            
            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()
            
            # Analyze the model for surgery
            analysis = plugin.analyze_model_for_surgery()
            
            # Should return a dictionary with analysis results
            assert_is_instance(analysis, dict)
            assert_in('total_parameters', analysis)
            assert_in('removable_components', analysis)
            assert_in('recommendations', analysis)

    def plugin_get_surgery_stats(self)():
        """Test that plugins can report surgery statistics."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize()
            assert_true(success)
            
            # Get surgery stats (should work even without surgery performed)
            stats = plugin.get_surgery_stats()
            
            # Should return a dictionary with stats
            assert_is_instance(stats, dict)
            assert_in('total_surgeries_performed', stats)
            assert_in('total_components_removed', stats)

    def model_surgery_with_preserve_components(self)():
        """Test model surgery with preserved components."""
        surgery_system = ModelSurgerySystem()
        
        # Identify components to remove
        removable = surgery_system.identify_removable_components(simple_model)
        components_to_remove = [comp.name for comp in removable]
        
        # Preserve the linear layers
        preserve_components = ['linear1', 'linear2']
        
        # Perform surgery while preserving certain components
        modified_model = surgery_system.perform_surgery(
            simple_model,
            components_to_remove=components_to_remove,
            preserve_components=preserve_components
        )
        
        # Should still work
        test_input = torch.randn(2, 10)
        output = modified_model(test_input)
        assert_equal(output.shape, (2))

    def cleanup_helper():
        """Clean up after each test method."""
        # Clean up any resources used by the plugins
        for plugin in plugins:
            if hasattr(plugin, 'cleanup'):
                plugin.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)
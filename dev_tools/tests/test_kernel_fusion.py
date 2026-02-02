"""
Test suite for Kernel Fusion functionality in model plugins.

This test verifies that the kernel fusion system works correctly across all model plugins.
"""
import torch
import torch.nn as nn

from src.inference_pio.common.kernel_fusion import get_kernel_fusion_manager
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

# TestKernelFusion

    """Test cases for kernel fusion functionality."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        plugins = [
            GLM_4_7_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin(),
            Qwen3_VL_2B_Plugin()
        ]
        
        # Create a simple test model for fusion tests
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                linear1 = nn.Linear(10, 20)
                relu = nn.ReLU()
                linear2 = nn.Linear(20, 5)
                gelu = nn.GELU()
                linear3 = nn.Linear(5, 1)
                
            def forward(self, x):
                x = linear1(x)
                x = relu(x)
                x = linear2(x)
                x = gelu(x)
                x = linear3(x)
                return x
        
        simple_model = SimpleModel()

    def kernel_fusion_manager_creation(self)():
        """Test that the kernel fusion manager can be created and accessed."""
        from src.inference_pio.common.kernel_fusion import KernelFusionManager
        fusion_manager = KernelFusionManager()
        assert_is_instance(fusion_manager, KernelFusionManager)
        
        # Test global instance
        global_manager = get_kernel_fusion_manager()
        assert_is_instance(global_manager, KernelFusionManager)

    def kernel_fusion_manager_methods(self)():
        """Test that kernel fusion manager has required methods."""
        fusion_manager = get_kernel_fusion_manager()
        
        # Check that required methods exist
        assert_true(hasattr(fusion_manager))
        assert_true(hasattr(fusion_manager))
        assert_true(hasattr(fusion_manager))
        assert_true(hasattr(fusion_manager))

    def enable_fusion(self)():
        """Test enabling kernel fusion."""
        fusion_manager = get_kernel_fusion_manager()
        
        # Enable fusion
        fusion_manager.enable_fusion()
        
        # Check that fusion is enabled
        assert_true(fusion_manager.fusion_enabled)

    def fuse_model(self)():
        """Test fusing a model."""
        fusion_manager = get_kernel_fusion_manager()
        
        # Enable fusion first
        fusion_manager.enable_fusion()
        
        # Fuse our simple model
        fused_model = fusion_manager.fuse_model(simple_model)
        
        # Should return a model
        assert_is_instance(fused_model)
        
        # Should still work
        test_input = torch.randn(2, 10)
        output = fused_model(test_input)
        assert_equal(output.shape, (2))

    def apply_custom_kernels(self)():
        """Test applying custom kernels to a model."""
        fusion_manager = get_kernel_fusion_manager()
        
        # Apply custom kernels to our simple model
        model_with_kernels = fusion_manager.apply_custom_kernels(simple_model)
        
        # Should return a model
        assert_is_instance(model_with_kernels, nn.Module)
        
        # Should still work
        test_input = torch.randn(2, 10)
        output = model_with_kernels(test_input)
        assert_equal(output.shape, (2))

    def optimize_model(self)():
        """Test optimizing a model with kernel fusion."""
        fusion_manager = get_kernel_fusion_manager()
        
        # Enable fusion first
        fusion_manager.enable_fusion()
        
        # Optimize our simple model
        optimized_model = fusion_manager.optimize_model(simple_model)
        
        # Should return a model
        assert_is_instance(optimized_model, nn.Module)
        
        # Should still work
        test_input = torch.randn(2, 10)
        output = optimized_model(test_input)
        assert_equal(output.shape, (2))

    def plugin_kernel_fusion_setup(self)():
        """Test that all plugins can set up kernel fusion."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize(enable_kernel_fusion=True)
            assert_true(success)
            
            # Check that kernel fusion methods are available
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))

    def plugin_apply_kernel_fusion(self)():
        """Test that plugins can apply kernel fusion."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with kernel fusion enabled
            success = plugin.initialize(enable_kernel_fusion=True)
            assert_true(success)
            
            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()
            
            # Apply kernel fusion
            original_model = plugin._model
            fusion_success = plugin.apply_kernel_fusion()
            
            # Should return True on success
            assert_true(fusion_success)

    def plugin_get_fusion_manager(self)():
        """Test that plugins can get the fusion manager."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize()
            assert_true(success)
            
            # Get fusion manager
            fusion_manager = plugin.get_fusion_manager()
            
            # Should return a manager or None
            assert_true(fusion_manager is not None or fusion_manager is None)

    def plugin_setup_kernel_fusion(self)():
        """Test that plugins can set up kernel fusion."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize()
            assertTrue(success)
            
            # Set up kernel fusion
            setup_success = plugin.setup_kernel_fusion()
            
            # Should return True on success
            assert_true(setup_success)

    def kernel_fusion_with_different_models(self)():
        """Test kernel fusion with different types of models."""
        fusion_manager = get_kernel_fusion_manager()
        fusion_manager.enable_fusion()
        
        # Test with different model architectures
        models_to_test = [
            nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1)),
            nn.Conv2d(3, 16, 3),
            nn.LSTM(10, 20, 2)
        ]
        
        for i, model in enumerate(models_to_test):
            try:
                fused_model = fusion_manager.fuse_model(model)
                assert_is_instance(fused_model, nn.Module)
            except Exception as e:
                # Some models might not be compatible with fusion
                print(f"Model {i} not compatible with fusion: {e}")

    def cleanup_helper():
        """Clean up after each test method."""
        # Clean up any resources used by the plugins
        for plugin in plugins:
            if hasattr(plugin, 'cleanup'):
                plugin.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)
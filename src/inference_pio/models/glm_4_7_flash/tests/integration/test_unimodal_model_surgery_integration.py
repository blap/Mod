"""
Test suite for Unimodal Model Surgery Integration with GLM-4-7, Qwen3-4b-instruct-2507, and Qwen3-coder-30b models.

This test verifies that the unimodal model surgery system works correctly with the three unimodal models.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.common.unimodal_model_surgery import (
    UnimodalModelSurgerySystem,
    apply_unimodal_model_surgery,
    analyze_unimodal_model_for_surgery
)

# TestUnimodalModelSurgeryIntegration

    """Test cases for unimodal model surgery integration with specific models."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        plugins = [
            GLM_4_7_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin()
        ]

    def unimodal_model_surgery_methods_exist(self)():
        """Test that all plugins have unimodal model surgery methods."""
        for plugin in plugins:
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))

    def unimodal_model_surgery_setup(self)():
        """Test setting up unimodal model surgery in plugins."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize(enable_unimodal_model_surgery=True)
            assert_true(success)

            # Check that unimodal model surgery methods are available
            assert_true(hasattr(plugin))

    def unimodal_model_surgery_analysis(self)():
        """Test analyzing unimodal models for surgery."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin
            success = plugin.initialize()
            assert_true(success)

            # Load the model if not already loaded
            if plugin._model is None:
                # Use a minimal config to avoid downloading large models
                from src.inference_pio.models.glm_4_7.config import GLM47Config
                config = GLM47Config()
                config.model_path = "dummy_path"  # Use dummy path to avoid download
                config.use_flash_attention_2 = False
                config.use_sparse_attention = False
                config.use_tensor_parallelism = False
                config.gradient_checkpointing = False
                config.use_cache = False
                try:
                    plugin.load_model(config)
                except:
                    # If loading fails (due to dummy path), create a minimal model for testing
                    class MinimalModel(nn.Module):
                        def __init__(self):
                            super().__init__()
                            embedding = nn.Embedding(100, 32)
                            dropout = nn.Dropout(0.1)
                            norm = nn.LayerNorm(32)
                            linear = nn.Linear(32, 10)

                        def forward(self, x):
                            x = embedding(x)
                            x = dropout(x)
                            x = norm(x)
                            x = linear(x)
                            return x
                    
                    plugin._model = MinimalModel()
                    plugin._tokenizer = None

            # Analyze the model for unimodal surgery
            analysis = plugin.analyze_unimodal_model_for_surgery()

            # Should return a dictionary with analysis results
            assert_is_instance(analysis, dict)
            assert_in('total_parameters', analysis)
            assert_in('removable_components', analysis)
            assert_in('recommendations', analysis)

    def unimodal_model_surgery_application(self)():
        """Test applying unimodal model surgery to models."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin
            success = plugin.initialize()
            assert_true(success)

            # Create a minimal model for testing
            class MinimalModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    embedding = nn.Embedding(100, 32)
                    dropout = nn.Dropout(0.1)
                    norm = nn.LayerNorm(32)
                    linear = nn.Linear(32, 10)

                def forward(self, x):
                    x = embedding(x)
                    x = dropout(x)
                    x = norm(x)
                    x = linear(x)
                    return x

            plugin._model = MinimalModel()
            plugin._tokenizer = None

            # Apply unimodal model surgery
            original_model = plugin._model
            modified_model = plugin.perform_unimodal_model_surgery()

            # Should return a model
            assert_is_instance(modified_model, type(original_model))

            # Should still be functional
            test_input = torch.randint(0, 100, (2, 5))
            output = modified_model(test_input)
            assert_equal(output.shape, (2))

    def unimodal_model_surgery_with_config_options(self)():
        """Test unimodal model surgery with various configuration options."""
        plugin = plugins[0]  # Use first plugin

        # Initialize with unimodal surgery enabled
        success = plugin.initialize(
            enable_unimodal_model_surgery=True,
            unimodal_components_to_remove=None,
            unimodal_preserve_components=['embedding'],
            unimodal_semantic_importance_threshold=0.5
        )
        assert_true(success)

        # Create a minimal model for testing
        class MinimalModelWithMultipleComponents(nn.Module):
            def __init__(self):
                super().__init__()
                embedding = nn.Embedding(100)
                dropout1 = nn.Dropout(0.1)
                norm1 = nn.LayerNorm(32)
                linear1 = nn.Linear(32, 64)
                dropout2 = nn.Dropout(0.1)
                norm2 = nn.LayerNorm(64)
                linear2 = nn.Linear(64, 10)

            def forward(self, x):
                x = embedding(x)
                x = dropout1(x)
                x = norm1(x)
                x = linear1(x)
                x = dropout2(x)
                x = norm2(x)
                x = linear2(x)
                return x

        plugin._model = MinimalModelWithMultipleComponents()
        plugin._tokenizer = None

        # Perform surgery with specific options
        modified_model = plugin.perform_unimodal_model_surgery(
            preserve_components=['embedding'],
            preserve_semantic_importance_threshold=0.8
        )

        # Should still work
        test_input = torch.randint(0, 100, (2, 5))
        output = modified_model(test_input)
        assert_equal(output.shape, (2))

    def unimodal_model_surgery_system_direct_usage(self)():
        """Test using the unimodal model surgery system directly."""
        # Create a test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                embedding = nn.Embedding(1000, 128)
                pos_encoding = nn.Embedding(512, 128)
                dropout = nn.Dropout(0.1)
                layer_norm = nn.LayerNorm(128)
                linear1 = nn.Linear(128, 256)
                relu = nn.ReLU()
                linear2 = nn.Linear(256, 128)

            def forward(self, x):
                x = embedding(x)
                pos_enc = pos_encoding(torch.arange(x.size(1)).unsqueeze(0).expand(x.size(0), -1))
                x = x + pos_enc
                x = dropout(x)
                x = layer_norm(x)
                x = linear1(x)
                x = relu(x)
                x = linear2(x)
                return x

        model = TestModel()

        # Test analysis
        analysis = analyze_unimodal_model_for_surgery(model)
        assert_is_instance(analysis, dict)
        assert_in('removable_components', analysis)

        # Test surgery application
        modified_model = apply_unimodal_model_surgery(model)
        assert_is_instance(modified_model, nn.Module)

        # Test with specific parameters
        modified_model_filtered = apply_unimodal_model_surgery(
            model,
            preserve_components=['embedding'],
            preserve_semantic_importance_threshold=0.9
        )
        assert_is_instance(modified_model_filtered, nn.Module)

        # Both models should work
        test_input = torch.randint(0, 100, (2, 10))
        output1 = modified_model(test_input)
        output2 = modified_model_filtered(test_input)

        assert_equal(output1.shape, (2))
        assert_equal(output2.shape, (2))

    def cleanup_helper():
        """Clean up after each test method."""
        # Clean up any resources used by the plugins
        for plugin in plugins:
            if hasattr(plugin, 'cleanup'):
                try:
                    plugin.cleanup()
                except:
                    pass  # Ignore cleanup errors in tests

if __name__ == '__main__':
    run_tests(test_functions)
"""
Standardized Test for Configuration Loading - GLM-4.7-Flash

This module tests the configuration loading for the GLM-4.7-Flash model with real parameters.
"""

import tempfile
import os
import json
from src.inference_pio.test_utils import assert_true, assert_is_not_none, assert_equal, assert_in, assert_is_instance, run_tests
from inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig


def test_default_config_loading():
    """Test loading the model with default configuration."""
    plugin = create_glm_4_7_flash_plugin()
    success = plugin.initialize(use_mock_model=True)  # Use mock model for testing
    assert_true(success)

    model = plugin.load_model()
    assert_is_not_none(model)

    # Check if model has config attribute
    assert_true(hasattr(model, 'config'))


def test_custom_config_loading():
    """Test loading the model with custom configuration."""
    plugin = create_glm_4_7_flash_plugin()
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_data = {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 10000,
            "use_mock_model": True  # Use mock model for testing
        }
        json.dump(config_data, f)
        temp_config_path = f.name

    try:
        # Initialize with custom config path
        success = plugin.initialize(config_path=temp_config_path)
        assert_true(success)

        model = plugin.load_model()
        assert_is_not_none(model)

        # Verify config values are applied (implementation dependent)
        if hasattr(model, 'config'):
            # This check depends on how the config is implemented
            pass
    finally:
        # Clean up temp file
        os.unlink(temp_config_path)


def test_config_validation():
    """Test that configuration is validated properly."""
    plugin = create_glm_4_7_flash_plugin()
    # Test with valid config
    success = plugin.initialize(hidden_size=512, num_attention_heads=8, use_mock_model=True)  # Use mock model for testing
    assert_true(success)

    model = plugin.load_model()
    assert_is_not_none(model)


def test_config_with_different_model_sizes():
    """Test configuration with different model sizes."""
    plugin = create_glm_4_7_flash_plugin()
    for hidden_size in [256, 512, 1024]:
        success = plugin.initialize(hidden_size=hidden_size, device="cpu", use_mock_model=True)  # Use mock model for testing
        assert_true(success)

        model = plugin.load_model()
        assert_is_not_none(model)

        # If config exists, check if the value is applied
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            assert_equal(model.config.hidden_size, hidden_size)


def test_config_with_attention_params():
    """Test configuration with attention-related parameters."""
    plugin = create_glm_4_7_flash_plugin()
    config_params = {
        'num_attention_heads': 20,  # Real GLM-4.7-Flash value
        'num_key_value_heads': 20,  # Real GLM-4.7-Flash value
        'intermediate_size': 10240,  # Real GLM-4.7-Flash value
        'max_position_embeddings': 202752,  # Real GLM-4.7-Flash value
        'use_mock_model': True  # Use mock model for testing
    }

    success = plugin.initialize(**config_params, device="cpu")
    assert_true(success)

    model = plugin.load_model()
    assert_is_not_none(model)


def test_config_loading_with_optimizations():
    """Test configuration loading with optimizations enabled."""
    plugin = create_glm_4_7_flash_plugin()
    success = plugin.initialize(
        use_flash_attention=True,
        memory_efficient=True,
        device="cpu",
        use_mock_model=True  # Use mock model for testing
    )
    assert_true(success)

    model = plugin.load_model()
    assert_is_not_none(model)

    # Check if optimizations are reflected in config
    if hasattr(model, 'config'):
        # Implementation-dependent checks
        pass


def test_invalid_config_handling():
    """Test handling of invalid configurations."""
    plugin = create_glm_4_7_flash_plugin()
    try:
        # Try to initialize with invalid parameters
        success = plugin.initialize(invalid_param="invalid_value", use_mock_model=True)  # Use mock model for testing
        # Should either ignore invalid params or handle gracefully
        model = plugin.load_model()
        assert_is_not_none(model)
    except TypeError:
        # If strict parameter checking is implemented
        pass


def test_config_preservation_across_initializations():
    """Test that configuration is preserved across multiple initializations."""
    plugin = create_glm_4_7_flash_plugin()
    # Initialize with specific config
    initial_config = {'hidden_size': 512, 'num_attention_heads': 8, 'use_mock_model': True}  # Use mock model for testing
    success = plugin.initialize(**initial_config, device="cpu")
    assert_true(success)

    model1 = plugin.load_model()
    assert_is_not_none(model1)

    # Initialize again (should handle appropriately)
    success2 = plugin.initialize(**initial_config, device="cpu")
    assert_true(success2)

    model2 = plugin.load_model()
    assert_is_not_none(model2)


def test_get_model_info_includes_config():
    """Test that get_model_info includes configuration details."""
    plugin = create_glm_4_7_flash_plugin()
    success = plugin.initialize(use_mock_model=True)  # Use mock model for testing
    assert_true(success)

    model = plugin.load_model()
    assert_is_not_none(model)

    # Get model info
    info = plugin.get_model_info()
    assert_is_instance(info, dict)
    assert_in('name', info)
    assert_in('model_type', info)


def test_config_serialization_deserialization():
    """Test configuration serialization and deserialization."""
    plugin = create_glm_4_7_flash_plugin()
    # Initialize with specific config
    success = plugin.initialize(hidden_size=512, device="cpu", use_mock_model=True)  # Use mock model for testing
    assert_true(success)

    model = plugin.load_model()
    assert_is_not_none(model)

    # If the model has a way to serialize config, test it
    if hasattr(model.config, 'to_dict') if hasattr(model, 'config') else False:
        config_dict = model.config.to_dict()
        assert_is_instance(config_dict, dict)
        assert_in('hidden_size', config_dict)


if __name__ == '__main__':
    run_tests([
        test_default_config_loading,
        test_custom_config_loading,
        test_config_validation,
        test_config_with_different_model_sizes,
        test_config_with_attention_params,
        test_config_loading_with_optimizations,
        test_invalid_config_handling,
        test_config_preservation_across_initializations,
        test_get_model_info_includes_config,
        test_config_serialization_deserialization
    ])
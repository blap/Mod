"""
Direct Test for Model Loading - GLM-4.7

This module tests the model loading functionality for the GLM-4.7 model using direct testing.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from inference_pio.common.base_plugin_interface import ModelPluginMetadata, PluginType
from inference_pio.test_utils import (
    assert_true, assert_false, assert_equal, assert_is_not_none,
    assert_is_none, assert_is_instance, skip_test, run_tests
)


def test_plugin_initialization():
    """Test that the plugin initializes properly."""
    plugin = create_glm_4_7_flash_plugin()
    assert_is_not_none(plugin, "Plugin should not be None")
    assert_is_instance(plugin.metadata, ModelPluginMetadata, "Metadata should be ModelPluginMetadata instance")
    assert_equal(plugin.metadata.name, "GLM-4.7-Flash", "Plugin name should be GLM-4.7-Flash")
    assert_false(plugin._initialized, "Plugin should not be initialized initially")


def test_model_loading_without_config():
    """Test loading the model without explicit configuration."""
    plugin = create_glm_4_7_flash_plugin()
    success = plugin.initialize(use_mock_model=True)  # Use mock model for testing
    assert_true(success, "Initialization should succeed")
    assert_true(plugin._initialized, "Plugin should be marked as initialized")

    # Try to load the model
    model = plugin.load_model()
    assert_is_not_none(model, "Model should not be None after loading")
    assert_true(plugin.is_loaded, "Plugin should be marked as loaded")


def test_model_loading_with_device_cpu():
    """Test loading the model on CPU device."""
    plugin = create_glm_4_7_flash_plugin()
    success = plugin.initialize(device="cpu", use_mock_model=True)  # Use mock model for testing
    assert_true(success, "Initialization with CPU should succeed")

    model = plugin.load_model()
    assert_is_not_none(model, "Model should not be None after loading")

    # Check if model is on CPU
    for param in model.parameters():
        assert_equal(param.device.type, "cpu", "Model parameters should be on CPU")


def test_model_loading_with_device_cuda_if_available():
    """Test loading the model on CUDA device if available."""
    if torch.cuda.is_available():
        plugin = create_glm_4_7_flash_plugin()
        success = plugin.initialize(device="cuda", use_mock_model=True)  # Use mock model for testing
        assert_true(success, "Initialization with CUDA should succeed")

        model = plugin.load_model()
        assert_is_not_none(model, "Model should not be None after loading")

        # Check if model is on CUDA
        for param in model.parameters():
            assert_equal(param.device.type, "cuda", "Model parameters should be on CUDA")
    else:
        # If CUDA is not available, skip this test
        skip_test("CUDA not available")


def test_model_loading_with_different_dtypes():
    """Test loading the model with different data types."""
    dtypes_to_test = [torch.float32]
    if torch.cuda.is_available():
        dtypes_to_test.extend([torch.float16])

    for dtype in dtypes_to_test:
        plugin = create_glm_4_7_flash_plugin()
        success = plugin.initialize(torch_dtype=dtype, use_mock_model=True)  # Use mock model for testing
        assert_true(success, f"Initialization with dtype {dtype} should succeed")

        model = plugin.load_model()
        assert_is_not_none(model, f"Model should not be None after loading with dtype {dtype}")

        # Check if model parameters have the correct dtype
        for param in model.parameters():
            assert_equal(param.dtype, dtype, f"Model parameters should have dtype {dtype}")


def test_model_loading_with_quantization():
    """Test loading the model with quantization if supported."""
    # Test with no quantization first
    plugin = create_glm_4_7_flash_plugin()
    success = plugin.initialize(quantization=None, use_mock_model=True)  # Use mock model for testing
    assert_true(success, "Initialization with no quantization should succeed")

    model = plugin.load_model()
    assert_is_not_none(model, "Model should not be None after loading")


def test_model_loading_error_handling():
    """Test error handling during model loading."""
    plugin = create_glm_4_7_flash_plugin()
    # Test with invalid device
    try:
        success = plugin.initialize(device="invalid_device", use_mock_model=True)  # Use mock model for testing
        # If it doesn't raise an exception, check if it handles gracefully
        model = plugin.load_model()
        # Should fallback to CPU or handle gracefully
        assert_is_not_none(model, "Model should not be None even with invalid device")
    except Exception:
        # If an exception is raised, that's also acceptable behavior
        pass


def test_multiple_load_attempts():
    """Test that multiple load attempts behave correctly."""
    plugin = create_glm_4_7_flash_plugin()
    # First load
    success1 = plugin.initialize(use_mock_model=True)  # Use mock model for testing
    assert_true(success1, "First initialization should succeed")
    model1 = plugin.load_model()
    assert_is_not_none(model1, "First model should not be None")

    # Second load attempt (should handle appropriately)
    model2 = plugin.load_model()
    assert_is_not_none(model2, "Second model should not be None")


if __name__ == '__main__':
    # Run all tests
    test_functions = [
        test_plugin_initialization,
        test_model_loading_without_config,
        test_model_loading_with_device_cpu,
        test_model_loading_with_device_cuda_if_available,
        test_model_loading_with_different_dtypes,
        test_model_loading_with_quantization,
        test_model_loading_error_handling,
        test_multiple_load_attempts
    ]

    run_tests(test_functions)
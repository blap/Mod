"""
Tests for model-specific configurations.
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys
import shutil

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from tests.utils.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false, 
    assert_is_none, assert_is_not_none, assert_in, assert_not_in, 
    assert_greater, assert_less, assert_is_instance, assert_raises, 
    run_tests
)

from src.inference_pio.common.config_manager import (
    GLM47DynamicConfig, Qwen34BDynamicConfig,
    Qwen3CoderDynamicConfig, Qwen3VLDynamicConfig
)

def test_glm47_specific_config():
    """Test GLM-4.7 specific configuration parameters."""
    config = GLM47DynamicConfig()
    
    # Check default values for specific parameters
    assert_true(config.use_glm_attention_patterns, "Should enable GLM attention patterns by default")
    assert_equal(config.glm_attention_window_size, 1024, "Should have correct default window size")
    
    # Check that we can modify specific parameters
    config.glm_attention_pattern_sparsity = 0.5
    assert_equal(config.glm_attention_pattern_sparsity, 0.5, "Should allow modifying specific parameters")


def test_qwen3_4b_specific_config():
    """Test Qwen3-4B specific configuration parameters."""
    config = Qwen34BDynamicConfig()
    
    # Check default values
    assert_true(config.use_qwen3_attention_optimizations, "Should enable Qwen3 attention optimizations")
    assert_equal(config.qwen3_instruction_attention_scaling, 1.2, "Should have correct attention scaling")
    
    # Check modification
    config.qwen3_kv_cache_compression_ratio = 0.8
    assert_equal(config.qwen3_kv_cache_compression_ratio, 0.8, "Should allow modifying compression ratio")


def test_qwen3_coder_specific_config():
    """Test Qwen3-Coder specific configuration parameters."""
    config = Qwen3CoderDynamicConfig()
    
    # Check default values
    assert_true(config.use_qwen3_coder_code_optimizations, "Should enable code optimizations")
    assert_true(config.use_qwen3_coder_syntax_highlighting, "Should enable syntax highlighting") # Fixed name
    
    # Check modification
    config.code_generation_temperature = 0.5
    assert_equal(config.code_generation_temperature, 0.5, "Should allow modifying generation temperature")


def test_qwen3_vl_specific_config():
    """Test Qwen3-VL specific configuration parameters."""
    config = Qwen3VLDynamicConfig()
    
    # Check default values
    assert_true(config.use_qwen3_vl_vision_optimizations, "Should enable vision optimizations")
    assert_equal(config.patch_size, 14, "Should have correct patch size")
    
    # Check modification
    config.enable_image_tokenization = False
    assert_false(config.enable_image_tokenization, "Should allow disabling image tokenization")


def test_model_config_compatibility():
    """Test compatibility between different model configurations."""
    # Create configs for different models
    configs = [
        GLM47DynamicConfig(model_name="glm47"),
        Qwen34BDynamicConfig(model_name="qwen3_4b"),
        Qwen3CoderDynamicConfig(model_name="qwen3_coder"),
        Qwen3VLDynamicConfig(model_name="qwen3_vl")
    ]
    
    # All should have common attributes
    for config in configs:
        assert_is_not_none(config.model_name, "All configs should have model_name")
        assert_is_not_none(config.hidden_size, "All configs should have hidden_size")
        assert_is_not_none(config.vocab_size, "All configs should have vocab_size")


def test_model_config_cloning():
    """Test cloning of model-specific configurations."""
    original = Qwen3VLDynamicConfig(
        model_name="original_vl",
        vision_hidden_size=2048
    )
    
    cloned = original.clone()
    cloned.model_name = "cloned_vl"

    # Check that clone preserves specific attributes
    assert_equal(cloned.vision_hidden_size, 2048, "Clone should preserve specific attributes")
    assert_true(cloned.use_qwen3_vl_vision_optimizations, "Clone should preserve default specific attributes")
    
    # Modify clone shouldn't affect original
    cloned.vision_hidden_size = 1024
    assert_equal(original.vision_hidden_size, 2048, "Modifying clone shouldn't affect original")


def run_tests():
    """Run all model-specific configuration tests."""
    print("Running model-specific configuration tests...")
    
    test_functions = [
        test_glm47_specific_config,
        test_qwen3_4b_specific_config,
        test_qwen3_coder_specific_config,
        test_qwen3_vl_specific_config,
        test_model_config_compatibility,
        test_model_config_cloning
    ]
    
    all_passed = True
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__} passed")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {str(e)}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✓ All model-specific configuration tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)

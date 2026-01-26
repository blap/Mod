"""
Tests for model-specific dynamic configurations.
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys
import shutil

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from src.inference_pio.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false, 
    assert_is_none, assert_is_not_none, assert_in, assert_not_in, 
    assert_greater, assert_less, assert_is_instance, assert_raises, 
    run_tests
)


def test_glm47_specific_config():
    """Test GLM-4.7 specific configuration."""
    from src.inference_pio.config.dynamic_config import GLM47DynamicConfig
    
    config = GLM47DynamicConfig(
        model_name="glm47_test",
        temperature=0.7,
        glm47_specific_param="special_value"
    )
    
    assert_equal(config.model_name, "glm47_test", "Config should have correct model name")
    assert_equal(config.glm47_specific_param, "special_value", "Config should have GLM-4.7 specific param")
    assert_equal(config.temperature, 0.7, "Config should have correct temperature")


def test_qwen3_4b_specific_config():
    """Test Qwen3-4b specific configuration."""
    from src.inference_pio.config.dynamic_config import Qwen34BDynamicConfig
    
    config = Qwen34BDynamicConfig(
        model_name="qwen3_4b_test",
        temperature=0.8,
        qwen3_4b_specific_param="special_value"
    )
    
    assert_equal(config.model_name, "qwen3_4b_test", "Config should have correct model name")
    assert_equal(config.qwen3_4b_specific_param, "special_value", "Config should have Qwen3-4b specific param")
    assert_equal(config.temperature, 0.8, "Config should have correct temperature")


def test_qwen3_coder_specific_config():
    """Test Qwen3-Coder specific configuration."""
    from src.inference_pio.config.dynamic_config import Qwen3CoderDynamicConfig
    
    config = Qwen3CoderDynamicConfig(
        model_name="qwen3_coder_test",
        temperature=0.6,
        qwen3_coder_specific_param="special_value"
    )
    
    assert_equal(config.model_name, "qwen3_coder_test", "Config should have correct model name")
    assert_equal(config.qwen3_coder_specific_param, "special_value", "Config should have Qwen3-Coder specific param")
    assert_equal(config.temperature, 0.6, "Config should have correct temperature")


def test_qwen3_vl_specific_config():
    """Test Qwen3-VL specific configuration."""
    from src.inference_pio.config.dynamic_config import Qwen3VL2BDynamicConfig
    
    config = Qwen3VL2BDynamicConfig(
        model_name="qwen3_vl_test",
        temperature=0.5,
        qwen3_vl_specific_param="special_value"
    )
    
    assert_equal(config.model_name, "qwen3_vl_test", "Config should have correct model name")
    assert_equal(config.qwen3_vl_specific_param, "special_value", "Config should have Qwen3-VL specific param")
    assert_equal(config.temperature, 0.5, "Config should have correct temperature")


def test_model_config_compatibility():
    """Test compatibility between different model configurations."""
    from src.inference_pio.config.dynamic_config import (
        GLM47DynamicConfig, Qwen34BDynamicConfig, 
        Qwen3CoderDynamicConfig, Qwen3VL2BDynamicConfig
    )
    
    # Create configs for different models
    configs = [
        GLM47DynamicConfig(model_name="glm47_compat", temperature=0.7),
        Qwen34BDynamicConfig(model_name="qwen3_4b_compat", temperature=0.8),
        Qwen3CoderDynamicConfig(model_name="qwen3_coder_compat", temperature=0.6),
        Qwen3VL2BDynamicConfig(model_name="qwen3_vl_compat", temperature=0.5)
    ]
    
    # All should have common attributes
    for config in configs:
        assert_is_not_none(config.model_name, "All configs should have model_name")
        assert_is_not_none(config.temperature, "All configs should have temperature")
        assert_true(config.validate(), "All configs should validate")


def test_model_config_cloning():
    """Test cloning of model-specific configurations."""
    from src.inference_pio.config.dynamic_config import (
        GLM47DynamicConfig, Qwen34BDynamicConfig, 
        Qwen3CoderDynamicConfig, Qwen3VL2BDynamicConfig
    )
    
    # Test cloning for each model-specific config
    original_configs = [
        GLM47DynamicConfig(model_name="clone_glm47", temperature=0.7, glm47_specific_param="value1"),
        Qwen34BDynamicConfig(model_name="clone_qwen3_4b", temperature=0.8, qwen3_4b_specific_param="value2"),
        Qwen3CoderDynamicConfig(model_name="clone_qwen3_coder", temperature=0.6, qwen3_coder_specific_param="value3"),
        Qwen3VL2BDynamicConfig(model_name="clone_qwen3_vl", temperature=0.5, qwen3_vl_specific_param="value4")
    ]
    
    for original in original_configs:
        cloned = original.clone(new_model_name=f"cloned_{original.model_name}")
        
        # Check that the clone has the new name
        assert_equal(cloned.model_name, f"cloned_{original.model_name}", "Cloned config should have new name")
        
        # Check that other attributes are preserved
        assert_equal(cloned.temperature, original.temperature, "Temperature should be preserved in clone")
        
        # Check that model-specific attributes are preserved
        if hasattr(original, 'glm47_specific_param'):
            assert_equal(cloned.glm47_specific_param, original.glm47_specific_param, "GLM47 specific param should be preserved")
        elif hasattr(original, 'qwen3_4b_specific_param'):
            assert_equal(cloned.qwen3_4b_specific_param, original.qwen3_4b_specific_param, "Qwen3-4b specific param should be preserved")
        elif hasattr(original, 'qwen3_coder_specific_param'):
            assert_equal(cloned.qwen3_coder_specific_param, original.qwen3_coder_specific_param, "Qwen3-Coder specific param should be preserved")
        elif hasattr(original, 'qwen3_vl_specific_param'):
            assert_equal(cloned.qwen3_vl_specific_param, original.qwen3_vl_specific_param, "Qwen3-VL specific param should be preserved")


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
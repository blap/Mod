"""
Tests for the dynamic configuration system components.
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


def test_dynamic_config_creation():
    """Test creation of dynamic configurations."""
    from src.inference_pio.config.dynamic_config import GLM47DynamicConfig
    
    # Create a config with default values
    config = GLM47DynamicConfig(model_name="test_config")
    assert_equal(config.model_name, "test_config", "Config should have correct model name")
    assert_is_not_none(config.temperature, "Config should have default temperature")
    assert_is_not_none(config.max_new_tokens, "Config should have default max_new_tokens")


def test_dynamic_config_attribute_modification():
    """Test modification of dynamic configuration attributes."""
    from src.inference_pio.config.dynamic_config import GLM47DynamicConfig
    
    config = GLM47DynamicConfig(model_name="attr_test", temperature=0.5)
    assert_equal(config.temperature, 0.5, "Initial temperature should be 0.5")
    
    # Modify the attribute
    config.temperature = 0.8
    assert_equal(config.temperature, 0.8, "Modified temperature should be 0.8")


def test_dynamic_config_clone():
    """Test cloning of dynamic configurations."""
    from src.inference_pio.config.dynamic_config import GLM47DynamicConfig
    
    original_config = GLM47DynamicConfig(
        model_name="clone_test_1",
        temperature=0.7,
        max_new_tokens=100
    )
    
    cloned_config = original_config.clone(new_model_name="clone_test_2")
    
    # Check that the clone has the new name
    assert_equal(cloned_config.model_name, "clone_test_2", "Cloned config should have new name")
    
    # Check that other attributes are preserved
    assert_equal(cloned_config.temperature, original_config.temperature, "Temperature should be preserved in clone")
    assert_equal(cloned_config.max_new_tokens, original_config.max_new_tokens, "Max new tokens should be preserved in clone")
    
    # Check that they are different objects
    assert_not_equal(id(original_config), id(cloned_config), "Original and clone should be different objects")


def test_dynamic_config_merge():
    """Test merging of dynamic configurations."""
    from src.inference_pio.config.dynamic_config import GLM47DynamicConfig
    
    base_config = GLM47DynamicConfig(
        model_name="merge_base",
        temperature=0.5,
        max_new_tokens=50
    )
    
    override_config = GLM47DynamicConfig(
        model_name="merge_override",
        temperature=0.9,  # This should override base
        top_p=0.9  # This should be added
    )
    
    merged_config = base_config.merge_with(override_config)
    
    # Temperature should come from override
    assert_equal(merged_config.temperature, 0.9, "Merged config should have overridden temperature")
    
    # Max new tokens should come from base
    assert_equal(merged_config.max_new_tokens, 50, "Merged config should have base max_new_tokens")
    
    # Top_p should come from override
    assert_equal(merged_config.top_p, 0.9, "Merged config should have added top_p")


def test_dynamic_config_validation():
    """Test validation of dynamic configurations."""
    from src.inference_pio.config.dynamic_config import GLM47DynamicConfig
    
    # Valid config
    valid_config = GLM47DynamicConfig(model_name="valid_test", temperature=0.7)
    assert_true(valid_config.validate(), "Valid config should pass validation")
    
    # Invalid temperature
    invalid_config = GLM47DynamicConfig(model_name="invalid_test", temperature=-1.0)
    assert_false(invalid_config.validate(), "Invalid temperature should fail validation")


def test_dynamic_config_serialization():
    """Test serialization and deserialization of dynamic configurations."""
    from src.inference_pio.config.dynamic_config import GLM47DynamicConfig
    import json
    
    original_config = GLM47DynamicConfig(
        model_name="serialize_test",
        temperature=0.6,
        max_new_tokens=128
    )
    
    # Serialize to dict
    serialized = original_config.to_dict()
    assert_is_instance(serialized, dict, "Serialized config should be a dict")
    assert_in('model_name', serialized, "Serialized config should contain model_name")
    assert_in('temperature', serialized, "Serialized config should contain temperature")
    
    # Deserialize from dict
    deserialized_config = GLM47DynamicConfig.from_dict(serialized)
    assert_equal(deserialized_config.model_name, original_config.model_name, "Deserialized config should have same name")
    assert_equal(deserialized_config.temperature, original_config.temperature, "Deserialized config should have same temperature")
    assert_equal(deserialized_config.max_new_tokens, original_config.max_new_tokens, "Deserialized config should have same max_new_tokens")


def run_tests():
    """Run all dynamic configuration tests."""
    print("Running dynamic configuration tests...")
    
    test_functions = [
        test_dynamic_config_creation,
        test_dynamic_config_attribute_modification,
        test_dynamic_config_clone,
        test_dynamic_config_merge,
        test_dynamic_config_validation,
        test_dynamic_config_serialization
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
        print("\n✓ All dynamic configuration tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
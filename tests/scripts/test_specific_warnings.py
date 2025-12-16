"""
Test script to specifically check if the three parameters are still generating warnings.
This will help us confirm the fix is working properly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.config.validation_system import ConfigValidator as ValidationSystemValidator
from src.qwen3_vl.config.config_validation import ConfigValidator as ConfigValidationValidator
from dataclasses import asdict


def test_specific_parameter_warnings():
    """Test that the three specific parameters don't generate warnings."""
    print("Testing specific parameter warnings...")

    # Create a config with the three parameters that were problematic
    config = Qwen3VLConfig()
    
    # Print the actual values to confirm they exist
    print(f"config.vision_model_type: {config.vision_model_type}")
    print(f"config.pretraining_tp: {config.pretraining_tp}")
    print(f"config.rotary_embedding_scaling_factor: {config.rotary_embedding_scaling_factor}")

    # Test with validation system
    print("\n1. Testing with validation_system.py:")
    validation_system_validator = ValidationSystemValidator()
    result1 = validation_system_validator.validate_config(config, strict=False)

    # Check for warnings about the three parameters specifically
    target_warnings1 = []
    for w in result1.warnings:
        if any(param in w for param in ['vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor']):
            target_warnings1.append(w)
    
    print(f"   Warnings about target parameters: {target_warnings1}")
    print(f"   Total warnings: {len(result1.warnings)}")

    # Test with config_validation system
    print("\n2. Testing with config_validation.py:")
    config_validation_validator = ConfigValidationValidator()
    result2 = config_validation_validator.validate_config(config, strict=False)

    # Check for warnings about the three parameters specifically
    target_warnings2 = []
    for w in result2['warnings']:
        if any(param in w for param in ['vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor']):
            target_warnings2.append(w)
    
    print(f"   Warnings about target parameters: {target_warnings2}")
    print(f"   Total warnings: {len(result2['warnings'])}")

    # Now test with a config dict that has unknown parameters to make sure the validation still works
    print("\n3. Testing that validation still catches unknown parameters...")
    config_dict = asdict(config)
    # Add an unknown parameter
    config_dict['unknown_parameter'] = 'some_value'
    
    # Test with validation system
    from src.qwen3_vl.config.factory import ConfigFactory
    test_config = ConfigFactory.from_dict(config_dict)
    result3 = validation_system_validator.validate_config(test_config, strict=False)
    
    unknown_warnings = [w for w in result3.warnings if 'unknown_parameter' in w]
    print(f"   Warnings about unknown_parameter: {len(unknown_warnings) > 0}")

    print("\nTest completed.")


if __name__ == "__main__":
    test_specific_parameter_warnings()
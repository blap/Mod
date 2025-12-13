"""
Test script to verify the configuration validation warnings for missing parameters.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.config.validation_system import ConfigValidator as ValidationSystemValidator
from src.qwen3_vl.config.config_validation import ConfigValidator as ConfigValidationValidator


def test_validation_warnings():
    """Test that the three parameters don't generate warnings after fix."""
    print("Testing configuration validation for missing parameters...")
    
    # Create a config with the three problematic parameters
    config = Qwen3VLConfig()
    
    # Test with validation system
    print("\n1. Testing with validation_system.py:")
    validation_system_validator = ValidationSystemValidator()
    result1 = validation_system_validator.validate_config(config, strict=False)
    
    # Check for warnings about the three parameters
    warnings1 = [w for w in result1.warnings if any(param in w for param in 
                                                    ['vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor'])]
    print(f"   Warnings about target parameters: {warnings1}")
    print(f"   Total warnings: {len(result1.warnings)}")
    
    # Test with config_validation system
    print("\n2. Testing with config_validation.py:")
    config_validation_validator = ConfigValidationValidator()
    result2 = config_validation_validator.validate_config(config, strict=False)
    
    # Check for warnings about the three parameters
    warnings2 = [w for w in result2['warnings'] if any(param in w for param in 
                                                       ['vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor'])]
    print(f"   Warnings about target parameters: {warnings2}")
    print(f"   Total warnings: {len(result2['warnings'])}")
    
    print("\nTest completed.")


if __name__ == "__main__":
    test_validation_warnings()
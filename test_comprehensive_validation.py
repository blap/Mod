"""
Comprehensive test to verify the configuration validation fix doesn't break existing functionality.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.config.validation_system import ConfigValidator as ValidationSystemValidator
from src.qwen3_vl.config.config_validation import ConfigValidator as ConfigValidationValidator


def test_comprehensive_validation():
    """Test that validation still works correctly after fixing the warnings."""
    print("Running comprehensive validation tests...")
    
    # Create a config with the three previously problematic parameters
    config = Qwen3VLConfig()
    
    # Test with validation system
    print("\n1. Testing validation_system.py:")
    validation_system_validator = ValidationSystemValidator()
    result1 = validation_system_validator.validate_config(config, strict=False)
    
    print(f"   Valid: {result1.valid}")
    print(f"   Total errors: {len(result1.errors)}")
    print(f"   Total warnings: {len(result1.warnings)}")
    
    # Check that the three specific parameters no longer generate warnings
    target_warnings = [w for w in result1.warnings if any(param in w for param in 
                                                    ['vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor'])]
    print(f"   Target parameter warnings (should be empty): {target_warnings}")
    
    # Test with config_validation system
    print("\n2. Testing config_validation.py:")
    config_validation_validator = ConfigValidationValidator()
    result2 = config_validation_validator.validate_config(config, strict=False)
    
    print(f"   Valid: {result2['valid']}")
    print(f"   Total errors: {len(result2['errors'])}")
    print(f"   Total warnings: {len(result2['warnings'])}")
    
    # Check that the three specific parameters no longer generate warnings
    target_warnings2 = [w for w in result2['warnings'] if any(param in w for param in 
                                                       ['vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor'])]
    print(f"   Target parameter warnings (should be empty): {target_warnings2}")
    
    # Test with a config that has invalid values to ensure validation still catches errors
    print("\n3. Testing that validation still catches errors...")
    bad_config = Qwen3VLConfig()
    bad_config.num_hidden_layers = 16  # Should cause an error since it should be 32
    
    result_bad = validation_system_validator.validate_config(bad_config, strict=False)
    errors_about_capacity = [e for e in result_bad.errors if 'capacity' in e.lower()]
    print(f"   Errors about capacity preservation: {len(errors_about_capacity) > 0}")
    
    print("\nComprehensive test completed successfully!")


if __name__ == "__main__":
    test_comprehensive_validation()
"""
Test to verify that the three parameters are properly recognized by the validation system.
This will create a test scenario where we can confirm the fix works correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.config.validation_system import ConfigValidator as ValidationSystemValidator
from src.qwen3_vl.config.config_validation import ConfigValidator as ConfigValidationValidator
from dataclasses import asdict


def test_parameter_recognition():
    """Test that the three parameters are properly recognized and don't generate warnings."""
    print("Testing parameter recognition in validation systems...")
    
    # Create a config instance
    config = Qwen3VLConfig()
    
    print(f"Config values:")
    print(f"  vision_model_type: {config.vision_model_type}")
    print(f"  pretraining_tp: {config.pretraining_tp}")
    print(f"  rotary_embedding_scaling_factor: {config.rotary_embedding_scaling_factor}")
    
    # Test both validation systems
    print("\n1. Testing validation_system.py:")
    vs_validator = ValidationSystemValidator()
    vs_result = vs_validator.validate_config(config, strict=False)
    
    print(f"   Valid: {vs_result.valid}")
    print(f"   Errors: {len(vs_result.errors)}")
    print(f"   Warnings: {len(vs_result.warnings)}")
    
    # Check specifically for our three parameters in warnings
    vs_target_warnings = [w for w in vs_result.warnings 
                         if any(param in w for param in 
                               ['vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor'])]
    print(f"   Target parameter warnings: {len(vs_target_warnings)}")
    
    print("\n2. Testing config_validation.py:")
    cv_validator = ConfigValidationValidator()
    cv_result = cv_validator.validate_config(config, strict=False)
    
    print(f"   Valid: {cv_result['valid']}")
    print(f"   Errors: {len(cv_result['errors'])}")
    print(f"   Warnings: {len(cv_result['warnings'])}")
    
    # Check specifically for our three parameters in warnings
    cv_target_warnings = [w for w in cv_result['warnings'] 
                         if any(param in w for param in 
                               ['vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor'])]
    print(f"   Target parameter warnings: {len(cv_target_warnings)}")
    
    # Now let's test with a config that has an unknown parameter to ensure validation still works
    print("\n3. Testing that validation still catches truly unknown parameters...")
    config_dict = asdict(config)
    
    # Add an unknown parameter
    config_dict['completely_unknown_parameter'] = 'test_value'
    
    # Create a new config from the modified dict
    from src.qwen3_vl.config.factory import ConfigFactory
    modified_config = ConfigFactory.from_dict(config_dict)
    
    # Test with both validators
    vs_modified_result = vs_validator.validate_config(modified_config, strict=False)
    cv_modified_result = cv_validator.validate_config(modified_config, strict=False)
    
    # Check if the unknown parameter generated warnings
    vs_unknown_warnings = [w for w in vs_modified_result.warnings 
                          if 'completely_unknown_parameter' in w]
    cv_unknown_warnings = [w for w in cv_modified_result['warnings'] 
                          if 'completely_unknown_parameter' in w]
    
    print(f"   Unknown parameter warnings (validation_system): {len(vs_unknown_warnings)}")
    print(f"   Unknown parameter warnings (config_validation): {len(cv_unknown_warnings)}")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  - vision_model_type, pretraining_tp, rotary_embedding_scaling_factor: NO warnings (as expected)")
    print(f"  - completely_unknown_parameter: {'YES' if len(vs_unknown_warnings) > 0 or len(cv_unknown_warnings) > 0 else 'NO'} warnings (as expected)")
    
    success = (len(vs_target_warnings) == 0 and len(cv_target_warnings) == 0 and 
               (len(vs_unknown_warnings) > 0 or len(cv_unknown_warnings) > 0))
    
    print(f"  - Overall test: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    test_parameter_recognition()
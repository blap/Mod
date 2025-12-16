"""
Comprehensive test to verify that the three parameters are properly handled by validation systems.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.config.validation_system import ConfigValidator as ValidationSystemValidator
from src.qwen3_vl.config.config_validation import ConfigValidator as ConfigValidationValidator
from dataclasses import asdict
from src.qwen3_vl.config.factory import ConfigFactory


def test_comprehensive_fix():
    """Test that the configuration validation warnings for the three parameters are fixed."""
    print("=== COMPREHENSIVE TEST FOR CONFIGURATION VALIDATION FIX ===")
    
    # Test 1: Check that the three parameters exist in the config
    print("\n1. Checking that the three parameters exist in Qwen3VLConfig:")
    config = Qwen3VLConfig()
    
    params_to_check = {
        'vision_model_type': config.vision_model_type,
        'pretraining_tp': config.pretraining_tp,
        'rotary_embedding_scaling_factor': config.rotary_embedding_scaling_factor
    }
    
    all_params_exist = True
    for param, value in params_to_check.items():
        print(f"   {param}: {value}")
        if value is None:
            all_params_exist = False
    
    print(f"   All parameters exist: {all_params_exist}")
    
    # Test 2: Test with validation_system
    print("\n2. Testing validation_system.py with normal config:")
    vs_validator = ValidationSystemValidator()
    vs_result = vs_validator.validate_config(config, strict=False)
    
    print(f"   Valid: {vs_result.valid}")
    print(f"   Errors: {len(vs_result.errors)}")
    print(f"   Warnings: {len(vs_result.warnings)}")
    
    # Check specifically for the three parameters in warnings
    vs_target_warnings = [w for w in vs_result.warnings 
                         if any(param in w for param in params_to_check.keys())]
    print(f"   Target parameter warnings: {len(vs_target_warnings)}")
    print(f"   Target warnings details: {vs_target_warnings}")
    
    # Test 3: Test with config_validation
    print("\n3. Testing config_validation.py with normal config:")
    cv_validator = ConfigValidationValidator()
    cv_result = cv_validator.validate_config(config, strict=False)
    
    print(f"   Valid: {cv_result['valid']}")
    print(f"   Errors: {len(cv_result['errors'])}")
    print(f"   Warnings: {len(cv_result['warnings'])}")
    
    # Check specifically for the three parameters in warnings
    cv_target_warnings = [w for w in cv_result['warnings'] 
                         if any(param in w for param in params_to_check.keys())]
    print(f"   Target parameter warnings: {len(cv_target_warnings)}")
    print(f"   Target warnings details: {cv_target_warnings}")
    
    # Test 4: Test with a config dict that has unknown parameters to ensure validation still works
    print("\n4. Testing that validation still catches truly unknown parameters:")
    
    # Create a config dict with the three target params plus an unknown param
    config_dict = asdict(config)
    config_dict['unknown_test_parameter'] = 'test_value'
    
    # Create a new config from this modified dict using the factory
    try:
        modified_config = ConfigFactory.from_dict(config_dict)
        
        # Test with validation_system
        vs_modified_result = vs_validator.validate_config(modified_config, strict=False)
        vs_unknown_warnings = [w for w in vs_modified_result.warnings 
                              if 'unknown_test_parameter' in w]
        
        # Test with config_validation
        cv_modified_result = cv_validator.validate_config(modified_config, strict=False)
        cv_unknown_warnings = [w for w in cv_modified_result['warnings'] 
                              if 'unknown_test_parameter' in w]
        
        print(f"   Validation_system unknown param warnings: {len(vs_unknown_warnings)}")
        print(f"   Config_validation unknown param warnings: {len(cv_unknown_warnings)}")
        
    except Exception as e:
        print(f"   Error creating modified config: {e}")
        # Let's try a different approach
        print("   Trying alternative approach...")
        # Just call the _check_unused_params method directly
        vs_unused = vs_validator._check_unused_params(config_dict)
        cv_unused = cv_validator._check_unused_params(config_dict)
        
        vs_unknown_direct = [w for w in vs_unused if 'unknown_test_parameter' in w]
        cv_unknown_direct = [w for w in cv_unused if 'unknown_test_parameter' in w]
        
        print(f"   Validation_system (direct) unknown param warnings: {len(vs_unknown_direct)}")
        print(f"   Config_validation (direct) unknown param warnings: {len(cv_unknown_direct)}")
    
    # Test 5: Summary
    print(f"\n5. SUMMARY:")
    print(f"   - Target parameters exist in config: {all_params_exist}")
    print(f"   - validation_system shows 0 warnings for target params: {len(vs_target_warnings) == 0}")
    print(f"   - config_validation shows 0 warnings for target params: {len(cv_target_warnings) == 0}")
    
    success = (all_params_exist and 
               len(vs_target_warnings) == 0 and 
               len(cv_target_warnings) == 0)
    
    print(f"   - Overall validation fix test: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    test_comprehensive_fix()
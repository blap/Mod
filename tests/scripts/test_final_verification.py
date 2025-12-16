"""
Final verification test to ensure the fix is complete and validation still works properly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.config.validation_system import ConfigValidator as ValidationSystemValidator
from src.qwen3_vl.config.config_validation import ConfigValidator as ConfigValidationValidator
from dataclasses import asdict
from src.qwen3_vl.config.factory import ConfigFactory


def test_final_verification():
    """Final test to verify the fix is complete and working."""
    print("=== FINAL VERIFICATION TEST ===")
    
    # Create config
    config = Qwen3VLConfig()
    
    # Define target parameters
    target_params = ['vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor']
    
    print(f"Testing the three target parameters: {target_params}")
    
    # Test 1: Normal config should have no warnings for target params
    print("\n1. Testing normal config with both validation systems:")
    
    vs_validator = ValidationSystemValidator()
    cv_validator = ConfigValidationValidator()
    
    vs_result = vs_validator.validate_config(config, strict=False)
    cv_result = cv_validator.validate_config(config, strict=False)
    
    print(f"   validation_system - Valid: {vs_result.valid}, Warnings: {len(vs_result.warnings)}")
    print(f"   config_validation - Valid: {cv_result['valid']}, Warnings: {len(cv_result['warnings'])}")
    
    # Check for target param warnings
    vs_target_warnings = [w for w in vs_result.warnings if any(p in w for p in target_params)]
    cv_target_warnings = [w for w in cv_result['warnings'] if any(p in w for p in target_params)]
    
    print(f"   Target param warnings (validation_system): {len(vs_target_warnings)}")
    print(f"   Target param warnings (config_validation): {len(cv_target_warnings)}")
    
    # Test 2: Config with unknown parameters should still generate warnings
    print("\n2. Testing config with unknown parameters:")
    
    # Create a config dict with an unknown parameter
    config_dict = asdict(config)
    config_dict['definitely_unknown_parameter_xyz'] = 'test_value'
    
    # Test the _check_unused_params methods directly
    vs_unused_warnings = vs_validator._check_unused_params(config_dict)
    cv_unused_warnings = cv_validator._check_unused_params(config_dict)
    
    print(f"   validation_system unused param warnings: {len(vs_unused_warnings)}")
    print(f"   config_validation unused param warnings: {len(cv_unused_warnings)}")
    
    # Check if the unknown parameter was detected
    vs_unknown_detected = any('definitely_unknown_parameter_xyz' in w for w in vs_unused_warnings)
    cv_unknown_detected = any('definitely_unknown_parameter_xyz' in w for w in cv_unused_warnings)
    
    print(f"   Unknown param detected by validation_system: {vs_unknown_detected}")
    print(f"   Unknown param detected by config_validation: {cv_unknown_detected}")
    
    # Test 3: Make sure target params are NOT in the unknown parameter warnings
    vs_target_in_unknown = any(p in ' '.join(vs_unused_warnings) for p in target_params)
    cv_target_in_unknown = any(p in ' '.join(cv_unused_warnings) for p in target_params)
    
    print(f"   Target params in validation_system unknown warnings: {vs_target_in_unknown}")
    print(f"   Target params in config_validation unknown warnings: {cv_target_in_unknown}")
    
    # Summary
    print(f"\n3. SUMMARY:")
    print(f"   - Target params generate 0 warnings in normal validation: {len(vs_target_warnings) == 0 and len(cv_target_warnings) == 0}")
    print(f"   - Unknown params still generate warnings: {vs_unknown_detected or cv_unknown_detected}")
    print(f"   - Target params are NOT flagged as unknown: {not vs_target_in_unknown and not cv_target_in_unknown}")
    
    success = (len(vs_target_warnings) == 0 and 
               len(cv_target_warnings) == 0 and 
               (vs_unknown_detected or cv_unknown_detected) and
               not vs_target_in_unknown and 
               not cv_target_in_unknown)
    
    print(f"\n   OVERALL RESULT: {'PASSED - Fix is working correctly!' if success else 'FAILED - Fix needs more work'}")
    
    return success


if __name__ == "__main__":
    test_final_verification()
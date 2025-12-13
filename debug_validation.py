"""
Direct test to check if the validation system properly recognizes the three parameters.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.config.validation_system import ConfigValidator as ValidationSystemValidator
from src.qwen3_vl.config.config_validation import ConfigValidator as ConfigValidationValidator
from dataclasses import asdict


def debug_validation():
    """Debug the validation process step by step."""
    print("Debugging validation process...")
    
    # Create a config
    config = Qwen3VLConfig()
    
    # Get the dict representation
    config_dict = asdict(config)
    print(f"Config dict keys (first 10): {list(config_dict.keys())[:10]}")
    print(f"Full config dict keys count: {len(config_dict.keys())}")
    
    # Check if our three parameters are in the config
    target_params = ['vision_model_type', 'pretraining_tp', 'rotary_embedding_scaling_factor']
    for param in target_params:
        if param in config_dict:
            print(f"  OK {param}: {config_dict[param]}")
        else:
            print(f"  NO {param}: NOT FOUND")
    
    # Test validation_system
    print("\nTesting validation_system:")
    vs_validator = ValidationSystemValidator()
    
    # Check the known params in validation system
    # We need to call _check_unused_params to see what it detects
    unused_warnings = vs_validator._check_unused_params(config_dict)
    print(f"Unused param warnings from validation_system: {unused_warnings}")
    
    # Look for our target params in the warnings
    target_warnings_vs = [w for w in unused_warnings if any(param in w for param in target_params)]
    print(f"Target param warnings (validation_system): {target_warnings_vs}")
    
    # Test config_validation
    print("\nTesting config_validation:")
    cv_validator = ConfigValidationValidator()
    
    # Check the known params in config validation
    unused_warnings_cv = cv_validator._check_unused_params(config_dict)
    print(f"Unused param warnings from config_validation: {unused_warnings_cv}")
    
    # Look for our target params in the warnings
    target_warnings_cv = [w for w in unused_warnings_cv if any(param in w for param in target_params)]
    print(f"Target param warnings (config_validation): {target_warnings_cv}")
    
    # Let's manually test if adding an unknown parameter triggers a warning
    print("\nTesting with an unknown parameter:")
    config_dict_with_unknown = config_dict.copy()
    config_dict_with_unknown['unknown_test_param'] = 'test_value'
    
    # Check validation_system with unknown param
    unused_warnings_unknown = vs_validator._check_unused_params(config_dict_with_unknown)
    print(f"Unused param warnings with unknown param (validation_system): {unused_warnings_unknown}")
    
    # Check config_validation with unknown param
    unused_warnings_cv_unknown = cv_validator._check_unused_params(config_dict_with_unknown)
    print(f"Unused param warnings with unknown param (config_validation): {unused_warnings_cv_unknown}")


if __name__ == "__main__":
    debug_validation()
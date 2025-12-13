"""
Test to see the specific error in config_validation.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.config.config_validation import ConfigValidator as ConfigValidationValidator


def test_specific_error():
    """Test to see what specific error is occurring."""
    print("Testing config_validation.py to see specific error...")
    
    config = Qwen3VLConfig()
    
    config_validation_validator = ConfigValidationValidator()
    result = config_validation_validator.validate_config(config, strict=False)
    
    print(f"Valid: {result['valid']}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")
    
    # Let's check if there's an issue with the __post_init__ validation
    print("\nChecking if config is valid by itself...")
    try:
        config = Qwen3VLConfig()
        print("Config creation successful")
        print(f"num_hidden_layers: {config.num_hidden_layers}")
        print(f"num_attention_heads: {config.num_attention_heads}")
    except Exception as e:
        print(f"Config creation failed: {e}")


if __name__ == "__main__":
    test_specific_error()
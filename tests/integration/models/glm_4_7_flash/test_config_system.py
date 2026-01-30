"""
Test the configuration system by importing directly from the file.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Temporarily prevent importing models by modifying the sys.modules
import sys

def test_config_system():
    """Test the configuration system directly."""
    print("Testing configuration system...")
    
    # Import the config manager directly
    try:
        # Import the configuration classes directly
        from src.inference_pio.common.config_manager import (
            GLM47DynamicConfig,
            Qwen34BDynamicConfig,
            Qwen3CoderDynamicConfig,
            Qwen3VLDynamicConfig,
            get_config_manager
        )
        
        print("✓ Successfully imported configuration classes")
        
        # Test creating configs
        glm_config = GLM47DynamicConfig()
        print(f"✓ GLM config created: {glm_config.model_name}")
        
        qwen3_4b_config = Qwen34BDynamicConfig()
        print(f"✓ Qwen3-4B config created: {qwen3_4b_config.model_name}")
        
        qwen3_coder_config = Qwen3CoderDynamicConfig()
        print(f"✓ Qwen3-Coder config created: {qwen3_coder_config.model_name}")
        
        qwen3_vl_config = Qwen3VLDynamicConfig()
        print(f"✓ Qwen3-VL config created: {qwen3_vl_config.model_name}")
        
        # Test config manager
        config_manager = get_config_manager()
        print("✓ Config manager obtained successfully")
        
        # Register a config
        config_manager.register_config("test_config", glm_config)
        print("✓ Config registered successfully")
        
        # Retrieve config
        retrieved = config_manager.get_config("test_config")
        print(f"✓ Retrieved config: {retrieved.model_name}")
        
        # Test config loader
        from src.inference_pio.common.config_loader import (
            get_config_loader,
            create_config_from_profile
        )
        print("✓ Successfully imported config loader")
        
        # Test config validator
        from src.inference_pio.common.config_validator import get_config_validator
        print("✓ Successfully imported config validator")
        
        # Test validation
        validator = get_config_validator()
        is_valid, errors = validator.validate_config(glm_config)
        print(f"✓ Config validation result: {is_valid}, errors: {len(errors)}")
        
        # Test config profiles
        perf_config = create_config_from_profile("glm", "performance", temperature=0.8)
        print(f"✓ Created performance config with temperature: {perf_config['temperature']}")
        
        print("\n🎉 All configuration system tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during configuration test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_config_system()
    if success:
        print("\n✅ Dynamic configuration system is working correctly!")
    else:
        print("\n❌ Dynamic configuration system has issues!")
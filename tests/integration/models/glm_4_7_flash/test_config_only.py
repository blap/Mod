"""
Test just the configuration system without loading models.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


def test_config_only():
    """Test configuration system without loading models."""
    print("Testing configuration system...")
    
    # Import just the config manager without importing models
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    from src.inference_pio.common.config_manager import (
        GLM47DynamicConfig,
        Qwen34BDynamicConfig,
        Qwen3CoderDynamicConfig,
        Qwen3VLDynamicConfig,
        get_config_manager
    )
    
    print("Successfully imported configuration classes")
    
    # Test creating configs
    glm_config = GLM47DynamicConfig()
    print(f"GLM config created: {glm_config.model_name}")
    
    qwen3_4b_config = Qwen34BDynamicConfig()
    print(f"Qwen3-4B config created: {qwen3_4b_config.model_name}")
    
    qwen3_coder_config = Qwen3CoderDynamicConfig()
    print(f"Qwen3-Coder config created: {qwen3_coder_config.model_name}")
    
    qwen3_vl_config = Qwen3VLDynamicConfig()
    print(f"Qwen3-VL config created: {qwen3_vl_config.model_name}")
    
    # Test config manager
    config_manager = get_config_manager()
    print("Config manager obtained successfully")
    
    # Register a config
    config_manager.register_config("test_config", glm_config)
    print("Config registered successfully")
    
    # Retrieve config
    retrieved = config_manager.get_config("test_config")
    print(f"Retrieved config: {retrieved.model_name}")
    
    print("Configuration system test completed successfully!")


if __name__ == "__main__":
    test_config_only()